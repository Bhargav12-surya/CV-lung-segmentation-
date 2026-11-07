"""Read-only audit of an existing MSF-HSAT run, cache, split and raw DICOM source.

Patient data are never modified. Reports go to a separately named output file.
Raw-header audits do not replace visual review or checks of cached voxel labels.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lung_ct_segmentation as pipeline


def resolved(path, root=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def grid_check(grid):
    spacing = np.asarray(grid["spacing_xyz_mm"], dtype=float)
    direction = np.asarray(grid["direction_xyz"], dtype=float).reshape(3, 3)
    origin = np.asarray(grid["origin_xyz_mm"], dtype=float)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Nonpositive/nonfinite physical spacing")
    if not np.isfinite(origin).all() or not np.isfinite(direction).all():
        raise ValueError("Nonfinite physical origin or direction")
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Nonorthonormal image direction")
    if len(grid["size_xyz"]) != 3 or min(grid["size_xyz"]) < 1:
        raise ValueError("Invalid physical image dimensions")


def binary_scan(array, chunk_depth=8):
    count = 0
    lower = np.asarray(array.shape, dtype=np.int64)
    upper = np.full(3, -1, dtype=np.int64)
    for z in range(0, array.shape[0], chunk_depth):
        part = np.asarray(array[z:z + chunk_depth])
        if np.any((part != 0) & (part != 1)):
            raise ValueError("Nonbinary or nonfinite label value")
        foreground = part != 0
        count += int(np.count_nonzero(foreground))
        if foreground.any():
            coords = np.nonzero(foreground)
            lo = np.asarray([v.min() for v in coords]) + [z, 0, 0]
            hi = np.asarray([v.max() for v in coords]) + [z, 0, 0]
            lower, upper = np.minimum(lower, lo), np.maximum(upper, hi)
    return count, (upper - lower + 1).tolist() if count else [0, 0, 0]


def audit_cache(base, record, full_hu=False):
    arrays = []
    try:
        directory, meta = pipeline.cache_path(base, record)
        grid_check(meta["native_grid"])
        grid_check(meta["working_grid"])
        result = {"status": "passed", "checks": [], "metadata": meta}
        for key, expected_shape in (("hu", meta["working_shape_zyx"]),
                                    ("mask", meta["working_shape_zyx"]),
                                    ("boundary", meta["working_shape_zyx"]),
                                    ("native_mask", meta["native_grid"]["size_xyz"][::-1])):
            if record["annotation_status"] == "unlabeled" and key != "hu":
                continue
            array = np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            arrays.append(array)
            if list(array.shape) != list(expected_shape):
                raise ValueError(f"{key} shape differs from recorded grid")
            if key == "hu":
                planes = range(array.shape[0]) if full_hu else np.unique(
                    np.linspace(0, array.shape[0] - 1, min(16, array.shape[0]), dtype=int))
                for z in planes:
                    if not np.isfinite(array[z]).all():
                        raise ValueError("Nonfinite CT value")
                result["hu_planes_checked"] = len(planes)
                result["hu_check_scope"] = "all" if full_hu else "up to 16 evenly spaced slices"
            else:
                count, extent = binary_scan(array)
                result[key + "_positive_voxels"] = count
                if key == "mask":
                    result["tumour_extent_zyx_voxels"] = extent
                    result["tumour_extent_zyx_mm"] = (np.asarray(extent) * meta["target_spacing_zyx_mm"]).tolist()
                    if record["annotation_status"] == "positive" and count == 0:
                        raise ValueError("Positive case has empty working mask")
                    fg = np.load(directory / "fg_indices.npy", mmap_mode="r", allow_pickle=False)
                    arrays.append(fg)
                    if len(fg) != min(count, 100000):
                        raise ValueError("Foreground index count differs from preparation sampling policy")
                    if len(fg) and (np.any(fg < 0) or np.any(fg >= array.size)
                                    or np.any(array.reshape(-1)[fg] != 1)):
                        raise ValueError("Foreground sample contains invalid coordinates")
                if key == "native_mask" and record["annotation_status"] == "positive" and count == 0:
                    raise ValueError("Positive case has empty native mask")
        if record["annotation_status"] != "unlabeled":
            native_ml = result["native_mask_positive_voxels"] * np.prod(meta["native_spacing_zyx_mm"]) / 1000
            if not np.isclose(native_ml, meta["native_reference_volume_ml"], rtol=1e-5, atol=1e-6):
                raise ValueError("Native label volume differs from recorded preprocessing volume")
        return result
    except (OSError, ValueError, KeyError, pipeline.UserError) as exc:
        return {"status": "missing" if isinstance(exc, FileNotFoundError) else "failed", "reason": str(exc)}
    finally:
        for array in arrays:
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()


def audit_raw(raw_root, record, roi, target_spacing):
    """Check actual CT/RTSTRUCT header geometry without reading PixelData."""
    pydicom = pipeline.require("pydicom")
    patient_dir = raw_root / record["patient_id"]
    series = {}
    structures = []
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Raw patient directory missing: {patient_dir}")
    for directory in sorted(p for p in patient_dir.rglob("*") if p.is_dir()):
        files = sorted(directory.glob("*.dcm"))
        if not files:
            continue
        header = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        modality = str(getattr(header, "Modality", ""))
        if modality == "CT":
            series[str(header.SeriesInstanceUID)] = files
        elif modality == "RTSTRUCT":
            for path in files:
                structures.append(pydicom.dcmread(str(path), stop_before_pixels=True))
    uid = record["source_identity"]
    files = series.get(uid)
    if not files:
        raise ValueError("No raw CT series matches the manifest source identity")
    matching = [rt for rt in structures if uid in pipeline.dicom_references(rt)[0]]
    if len(matching) != 1:
        raise ValueError(f"Expected one matching RTSTRUCT, found {len(matching)}")
    rt = matching[0]
    headers = [pydicom.dcmread(str(path), stop_before_pixels=True) for path in files]
    first = headers[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    rowdir, coldir = orientation[:3], orientation[3:]
    normal = np.cross(rowdir, coldir)
    basis = np.column_stack((rowdir, coldir, normal))
    if (not np.allclose(np.linalg.norm(normal), 1, atol=1e-4)
            or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-4)):
        raise ValueError("Invalid raw CT orientation")
    spacing_xy = np.asarray(first.PixelSpacing, dtype=float)[::-1]
    positions = []
    sop_positions = {}
    for head in headers:
        if (str(head.SeriesInstanceUID) != uid or str(head.PatientID) != record["patient_id"]
                or str(head.FrameOfReferenceUID) != str(first.FrameOfReferenceUID)
                or not np.allclose(head.ImageOrientationPatient, orientation, atol=1e-5)
                or not np.allclose(np.asarray(head.PixelSpacing)[::-1], spacing_xy, atol=1e-5)
                or (head.Rows, head.Columns) != (first.Rows, first.Columns)):
            raise ValueError("Mixed CT identity, orientation, spacing, or shape")
        position = np.asarray(head.ImagePositionPatient, dtype=float)
        positions.append(position)
        sop_positions[str(head.SOPInstanceUID)] = position
    positions = np.asarray(positions)
    positions = positions[np.argsort(positions @ normal)]
    steps = np.diff(positions @ normal)
    if len(steps) == 0 or np.any(steps <= 1e-4):
        raise ValueError("Missing/duplicate CT planes")
    spacing_z = float(np.median(steps))
    if not np.allclose(steps, spacing_z, atol=max(.01, spacing_z * .01), rtol=0):
        raise ValueError("Nonuniform raw CT slices")
    lateral = positions - positions[0] - np.outer((positions - positions[0]) @ normal, normal)
    if np.abs(lateral).max() > .05:
        raise ValueError("CT slice lateral displacement")
    _, refs, frames = pipeline.dicom_references(rt)
    if not refs.issubset(sop_positions) or (frames and frames != {str(first.FrameOfReferenceUID)}):
        raise ValueError("RTSTRUCT refers outside the matched CT series/frame")
    names = {str(item.ROIName): int(item.ROINumber) for item in rt.StructureSetROISequence}
    if roi not in names:
        raise ValueError("Target ROI absent in matched RTSTRUCT")
    selections = [item for item in rt.ROIContourSequence if int(item.ReferencedROINumber) == names[roi]]
    if len(selections) != 1:
        raise ValueError("Missing/duplicate target ROI contours")
    points = []
    for contour in selections[0].ContourSequence:
        array = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        if str(contour.ContourGeometricType) != "CLOSED_PLANAR" or not np.isfinite(array).all():
            raise ValueError("Unsupported/nonfinite target contour")
        z_distances = np.abs((positions - array[0]) @ normal)
        index = int(np.argmin(z_distances))
        if z_distances[index] > max(.1, spacing_z * .05) or np.ptp(array @ normal) > .05:
            raise ValueError("Target contour is not on a matched CT plane")
        delta = array - positions[index]
        x, y = delta @ rowdir / spacing_xy[0], delta @ coldir / spacing_xy[1]
        if x.min() < -.5 or x.max() > first.Columns - .5 or y.min() < -.5 or y.max() > first.Rows - .5:
            raise ValueError("Target contour extends outside CT image bounds")
        for referenced in getattr(contour, "ContourImageSequence", []):
            sop = str(referenced.ReferencedSOPInstanceUID)
            if sop not in sop_positions:
                raise ValueError("Target contour references an absent CT slice")
            if abs(float((array[0] - sop_positions[sop]) @ normal)) > max(.1, spacing_z * .05):
                raise ValueError("Contour physical plane differs from its referenced SOP plane")
        points.append(array)
    if not points:
        raise ValueError("Target contour is empty")
    coordinates = np.concatenate(points) @ np.column_stack((normal, coldir, rowdir))
    extent_mm = np.ptp(coordinates, axis=0) + np.asarray([spacing_z, spacing_xy[1], spacing_xy[0]])
    native_shape = [len(headers), int(first.Rows), int(first.Columns)]
    if native_shape != record["native_shape_zyx"]:
        raise ValueError("Raw native shape differs from manifest")
    if not np.allclose([spacing_z, spacing_xy[1], spacing_xy[0]], record["native_spacing_zyx_mm"], atol=.01):
        raise ValueError("Raw native spacing differs from manifest")
    sitk = pipeline.require("SimpleITK")
    orientation_probe = sitk.Image([1, 1, 1], sitk.sitkUInt8)
    orientation_probe.SetSpacing((float(spacing_xy[0]), float(spacing_xy[1]), spacing_z))
    orientation_probe.SetDirection(np.column_stack((rowdir, coldir, normal)).reshape(-1).tolist())
    oriented = sitk.DICOMOrient(orientation_probe, "LPS")
    return {"status": "passed", "ct_header_count": len(headers), "target_contour_count": len(points),
            "native_shape_zyx": native_shape, "native_spacing_zyx_mm": [spacing_z, *spacing_xy[::-1]],
            "tumour_extent_zyx_mm": extent_mm.tolist(),
            "extent_scope": "physical target contour bounds plus one native voxel; not a rasterized label check",
            "working_spacing_zyx_mm": target_spacing,
            "conditioning_native_spacing_zyx_mm": list(oriented.GetSpacing()[::-1])}


def percentile_summary(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return None
    return {"n": len(array), "p10": np.percentile(array, 10, axis=0).tolist(),
            "median": np.median(array, axis=0).tolist(), "p90": np.percentile(array, 90, axis=0).tolist(),
            "p95": np.percentile(array, 95, axis=0).tolist(), "max": array.max(axis=0).tolist()}


def inspect_run(run):
    config = pipeline.load_json(run / "resolved_config.json")
    history = pipeline.read_csv(run / "history.csv")
    result = {"path": str(run), "history_epochs": len(history),
              "history_last_epoch": int(history[-1]["epoch"]) if history else None,
              "configured_epochs": config["train_args"]["max_epochs"],
              "manifest": str(resolved(config["train_args"]["manifest"])),
              "splits": str(resolved(config["train_args"]["splits"])), "checkpoint_metadata": {}}
    for name in ("best.pt", "last.pt"):
        path = run / "checkpoints" / name
        if not path.is_file():
            result["checkpoint_metadata"][name] = {"status": "missing"}
            continue
        saved = pipeline.safe_load_checkpoint(path)
        matches = {key: pipeline.clean_json(saved[key]) == pipeline.clean_json(config[key]) for key in
                   ("manifest_fingerprint", "split_fingerprint", "acquisition_stats", "model_config")}
        if not all(matches.values()):
            raise ValueError(f"{name} does not match the resolved run configuration: {matches}")
        result["checkpoint_metadata"][name] = {
            "status": "matched", "completed_epochs": saved["completed_epochs"],
            "global_step": saved["global_step"], "best_metric": saved["best_metric"],
            "acquisition_stats": saved["acquisition_stats"], "config_matches": matches,
            "checkpoint_history_last_epoch": saved["history"][-1]["epoch"] if saved.get("history") else None,
            "model_state_keys": len(saved["model_state"]), "bytes": path.stat().st_size}
        if name == "last.pt" and saved["completed_epochs"] != result["history_last_epoch"]:
            raise ValueError("Latest checkpoint epoch differs from training history")
        del saved
    return result, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Defaults to newest run with resolved_config.json and history.csv")
    parser.add_argument("--manifest")
    parser.add_argument("--splits")
    parser.add_argument("--raw-root", default="data/nsclc_radiomics")
    parser.add_argument("--raw-limit", type=int, default=12, help="Bound raw checks; 0 checks all manifest cases")
    parser.add_argument("--full-hu", action="store_true", help="Check every cached HU slice instead of 16 sampled planes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = resolved(args.output)
    if output.exists():
        parser.error("Choose a new --output; existing audits are immutable")
    candidates = [resolved(args.run)] if args.run else sorted(
        (p.parent for p in (ROOT / "runs").glob("*/resolved_config.json")
         if (p.parent / "history.csv").is_file()
         and not pipeline.load_json(p).get("train_args", {}).get("overfit_cases", 0)),
        key=lambda p: (p / "history.csv").stat().st_mtime, reverse=True)
    if not candidates:
        parser.error("No matching non-diagnostic training run found; specify --run")
    started = time.perf_counter()
    run_report, config = inspect_run(candidates[0])
    manifest_path = resolved(args.manifest or run_report["manifest"])
    splits_path = resolved(args.splits or run_report["splits"])
    manifest, base = pipeline.load_manifest(manifest_path)
    split = pipeline.validate_split(manifest, pipeline.load_json(splits_path))
    if (config["manifest_fingerprint"] != manifest["fingerprint"]
            or config["split_fingerprint"] != split["fingerprint"]):
        raise ValueError("Resolved dataset/split do not match the selected run")
    partition = {cid: key for key, ids in split["partitions"].items() for cid in ids}
    records = manifest["records"]
    # Training first: physical context planning must be learned from train only.
    ordered = sorted(records, key=lambda r: (partition.get(r["case_id"]) != "train", r["case_id"]))
    raw_records = ordered if args.raw_limit == 0 else ordered[:args.raw_limit]
    raw_root = resolved(args.raw_root)
    report = {"created": pipeline.now(), "selected_run": run_report,
              "paths": {"manifest": str(manifest_path), "cache": str(base / "cases"),
                        "raw_root": str(raw_root), "splits": str(splits_path)},
              "partition_counts": {key: len(ids) for key, ids in split["partitions"].items()},
              "split_isolation": "patient/source/case disjoint; fingerprints matched",
              "raw_checks": [], "cache_checks": [], "limitations": []}
    metadata_for_stats = []
    for i, record in enumerate(ordered, 1):
        result = audit_cache(base, record, args.full_hu)
        metadata = result.pop("metadata", None)
        report["cache_checks"].append({"case_id": record["case_id"], "partition": partition.get(record["case_id"]), **result})
        if metadata is not None and partition.get(record["case_id"]) == "train":
            metadata_for_stats.append((record, metadata))
        if i % 50 == 0:
            print(f"Cache identities checked: {i}/{len(ordered)}", flush=True)
    for i, record in enumerate(raw_records, 1):
        try:
            result = audit_raw(raw_root, record, manifest["settings"]["target_roi"], manifest["settings"]["spacing_zyx_mm"])
        except (OSError, ValueError, KeyError, pipeline.UserError) as exc:
            result = {"status": "failed", "reason": str(exc)}
        report["raw_checks"].append({"case_id": record["case_id"], "partition": partition.get(record["case_id"]), **result})
        if i % 10 == 0:
            print(f"Raw geometry checked: {i}/{len(raw_records)}", flush=True)
    cache_counts = Counter(r["status"] for r in report["cache_checks"])
    raw_counts = Counter(r["status"] for r in report["raw_checks"])
    report["cache_counts"] = dict(cache_counts)
    report["raw_counts"] = dict(raw_counts)
    extents = [r["tumour_extent_zyx_mm"] for r in report["cache_checks"]
               if r["status"] == "passed" and r["partition"] == "train"]
    extent_scope = "working-grid masks"
    raw_extents = [r["tumour_extent_zyx_mm"] for r in report["raw_checks"]
                   if r["status"] == "passed" and r["partition"] == "train"]
    if len(raw_extents) > len(extents):
        extents = raw_extents
        extent_scope = "raw contour bounds; cache absent"
    report["training_extents_zyx_mm"] = {"scope": extent_scope, "summary": percentile_summary(extents)}
    report["patch_field_of_view"] = []
    for patch in ((32, 96, 96), (48, 128, 128)):
        fov = np.asarray(patch) * manifest["settings"]["spacing_zyx_mm"]
        report["patch_field_of_view"].append({"patch_zyx": patch, "extent_zyx_mm": fov.tolist(),
            "training_cases_with_extent_fitting_all_axes": int(np.all(np.asarray(extents) <= fov, axis=1).sum()) if extents else 0,
            "training_cases_measured": len(extents), "note": "Extent fit alone does not ensure sufficient anatomical margin or model accuracy"})
    # Cache metadata is required to reproduce exact preprocessing statistics.
    if len(metadata_for_stats) == len(split["partitions"]["train"]):
        training = pipeline.choose_records(manifest, split, "train")
        corrected = pipeline.acquisition_statistics(training, base)
        report["corrected_training_statistics"] = corrected
        report["saved_vs_corrected_std"] = {"saved": config["acquisition_stats"]["std"], "corrected": corrected["std"]}
    else:
        raw_training = [r for r in report["raw_checks"] if r["partition"] == "train" and r["status"] == "passed"]
        if len(raw_training) == len(split["partitions"]["train"]):
            patients = {}
            manifest_records = {r["case_id"]: r for r in records}
            for raw in raw_training:
                patients.setdefault(manifest_records[raw["case_id"]]["patient_id"], []).append(
                    pipeline.raw_descriptor(raw["conditioning_native_spacing_zyx_mm"],
                        raw["working_spacing_zyx_mm"], (2, 4, 4)))
            values = np.stack([np.mean(np.asarray(v, dtype=np.float64), axis=0) for v in patients.values()])
            std = values.std(axis=0)
            std[std < 1e-6] = 1.
            corrected = {"mean2": values.mean(axis=0).tolist(), "mean3": (values.mean(axis=0) +
                np.asarray([0.] * 6 + [math.log(2)] * 3)).tolist(), "std": std.tolist(),
                "training_patients": len(patients),
                "scope": "Recomputed from verified native raw CT headers and declared target spacing; all training cases only"}
            report["corrected_training_statistics"] = corrected
            report["saved_vs_corrected_std"] = {"saved": config["acquisition_stats"]["std"], "corrected": corrected["std"]}
        else:
            report["corrected_training_statistics"] = {"status": "unavailable", "reason": "Incomplete training cache metadata and raw headers; original checkpoint statistics remain untouched"}
    try:
        import psutil
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if "python" in (proc.info["name"] or "").lower():
                    processes.append({"pid": proc.pid, "command": proc.info["cmdline"], "created": proc.info["create_time"]})
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        report["python_processes"] = processes
        report["available_ram_gib"] = psutil.virtual_memory().available / 2**30
    except ImportError:
        report["python_processes"] = "psutil unavailable"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True, timeout=10)
        report["gpu_snapshot"] = gpu.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["gpu_snapshot"] = str(exc)
    report["limitations"].append("Header geometry and recorded grids do not alone establish clinical label correctness; visually inspect overlays.")
    if cache_counts.get("missing") or cache_counts.get("failed"):
        report["limitations"].append("Prepared cache is unavailable/incomplete. Cached label values, HU arrays and working/native reconstruction cannot all be verified until the same cache is restored.")
    report["seconds"] = time.perf_counter() - started
    report["status"] = "passed" if cache_counts.get("passed", 0) == len(records) and not raw_counts.get("failed") else "blocked_missing_or_invalid_cache"
    pipeline.atomic_json(output, report)
    print(json.dumps({"report": str(output), "status": report["status"], "cache_counts": cache_counts,
                      "raw_counts": raw_counts, "training_extent_cases": len(extents), "seconds": report["seconds"]}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
