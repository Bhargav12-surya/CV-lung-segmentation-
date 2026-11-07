"""Apply a frozen development operating point to saved validation probabilities.

This command never constructs a model, runs inference, trains, reads test cases,
or duplicates probability volumes. Original evaluations and operating points are
preserved. Output metrics use the same native-grid helpers as model evaluation.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import lung_ct_segmentation as pipeline


TIMING_SCOPE = (
    "CPU saved native probability loading, component filtering and metric computation; "
    "no model inference or GPU allocation was performed"
)


def matching_case_ids(ids, expected, label):
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise pipeline.UserError(f"{label} must contain exactly the complete validation case set, without duplicates.")


def probability_path(source, case_id):
    root = (source / "cases").resolve()
    path = (root / case_id / "probability.nii.gz").resolve()
    if not path.is_relative_to(root):
        raise pipeline.UserError("Saved probability path escapes the source evaluation directory.")
    return path


def assert_native_grid(image, metadata, case_id):
    actual = pipeline.grid_dict(image)
    expected = metadata["native_grid"]
    if (actual["size_xyz"] != expected["size_xyz"]
            or not np.allclose(actual["spacing_xyz_mm"], expected["spacing_xyz_mm"], atol=1e-5, rtol=0)
            or not np.allclose(actual["origin_xyz_mm"], expected["origin_xyz_mm"], atol=1e-3, rtol=0)
            or not np.allclose(actual["direction_xyz"], expected["direction_xyz"], atol=1e-5, rtol=0)):
        raise pipeline.UserError(f"Saved probability geometry differs from the native reference: {case_id}")


def validate_inputs(args):
    source = Path(args.evaluation).resolve()
    point_path = Path(args.operating_point).resolve()
    protocol = pipeline.load_json(source / "protocol.json")
    if protocol.get("partition") != "val":
        raise pipeline.UserError("Saved-probability rescoring requires a validation source, never test/external/predict.")
    manifest, base = pipeline.load_manifest(args.manifest)
    split = pipeline.validate_split(manifest, pipeline.load_json(args.splits))
    if (protocol.get("manifest_fingerprint") != manifest["fingerprint"]
            or protocol.get("split_fingerprint") != split["fingerprint"]
            or protocol.get("target") != manifest["settings"]["target_roi"]):
        raise pipeline.UserError("Source evaluation manifest, split or target differs from the supplied validation data.")
    records = pipeline.choose_records(manifest, split, "val")
    expected = [r["case_id"] for r in records]
    source_rows = pipeline.load_json(source / "patient_metrics.json")
    if not source_rows or any(row.get("status") != "success" for row in source_rows):
        raise pipeline.UserError("Source validation must be complete with no technical failures.")
    matching_case_ids([row["case_id"] for row in source_rows], expected, "Source metrics")
    if "case_ids" in protocol:
        matching_case_ids(protocol["case_ids"], expected, "Source protocol")
    by_id = {r["case_id"]: r for r in records}
    if any(row.get("patient_id") != by_id[row["case_id"]]["patient_id"] for row in source_rows):
        raise pipeline.UserError("Source metric case/patient identities differ from the validation manifest.")

    point = pipeline.load_json(point_path)
    fingerprint = pipeline.stable_hash({k: v for k, v in point.items() if k != "fingerprint"})
    if point.get("fingerprint") != fingerprint or point.get("source_partition") != "val":
        raise pipeline.UserError("Operating point fingerprint or validation-selection provenance is invalid.")
    for key in ("checkpoint_sha256", "manifest_fingerprint", "split_fingerprint", "target"):
        if not point.get(key) or point[key] != protocol.get(key):
            raise pipeline.UserError(f"Operating point {key} differs from the saved validation source.")
    matching_case_ids(point.get("case_ids"), expected, "Frozen operating point")
    threshold = float(point["threshold"])
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise pipeline.UserError("Frozen threshold must be finite and strictly between zero and one.")
    recipe = pipeline.postprocessing_recipe(point.get("postprocessing", "none"))
    if not 0 <= float(protocol.get("overlap", -1)) < 1:
        raise pipeline.UserError("Source reconstruction overlap is missing or invalid.")
    if "overlap" in point and point["overlap"] != protocol["overlap"]:
        raise pipeline.UserError("Operating point overlap differs from the saved probability reconstruction.")
    tolerance = float(protocol.get("surface_tolerance_mm", 0))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise pipeline.UserError("Source surface tolerance must be positive and finite.")
    expected_surface = "6-neighbor boundary voxel centers; pooled unweighted distances"
    if protocol.get("surface_convention") != expected_surface:
        raise pipeline.UserError("Source surface convention differs from the metric implementation.")
    for record in records:
        if not probability_path(source, record["case_id"]).is_file():
            raise pipeline.UserError(f"Missing saved validation probability for {record['case_id']}.")
    return source, point_path, protocol, manifest, base, records, point, threshold, recipe, tolerance


def rescore_saved_validation(args):
    (source, point_path, source_protocol, manifest, base, records, point,
     threshold, recipe, tolerance) = validate_inputs(args)
    out = Path(args.output).resolve()
    if out.exists():
        raise pipeline.UserError("Rescore output already exists; choose a fresh output directory.")
    repeats = getattr(args, "bootstrap_repeats", 1000)
    seed = getattr(args, "seed", 1234)
    if repeats < 1 or not 0 <= seed < 2**32:
        raise pipeline.UserError("Bootstrap repeats must be positive and seed must be in [0,2**32).")
    sitk = pipeline.require("SimpleITK")
    out.mkdir(parents=True)
    protocol = copy.deepcopy(source_protocol)
    protocol.update({
        "created": pipeline.now(), "partition": "val", "case_ids": [r["case_id"] for r in records],
        "threshold": threshold, "threshold_source": "frozen validation selection: " + point["fingerprint"],
        "postprocessing": recipe, "evaluation_kind": "saved_probability_rescore",
        "model_executed": False, "processing_device": "cpu", "timing_scope": TIMING_SCOPE,
        "source_evaluation": str(source), "source_protocol_sha256": pipeline.file_digest(source / "protocol.json"),
        "source_patient_metrics_sha256": pipeline.file_digest(source / "patient_metrics.json"),
        "source_operating_point": str(point_path),
        "source_inference_precision": source_protocol.get("precision"),
        "sampling": "No voxel samples regenerated; descriptive calibration remains in the original source evaluation.",
        "prediction_exports": "None; metrics computed from existing saved native probabilities without duplicating volumes.",
        "bootstrap_seed": seed, "bootstrap_repeats": repeats,
    })
    pipeline.atomic_json(out / "protocol.json", protocol)
    pipeline.atomic_json(out / "operating_point.json", point)
    rows = []
    started = time.perf_counter()
    for record in records:
        case_id = record["case_id"]
        row = {"case_id": case_id, "patient_id": record["patient_id"], "status": "success",
               "threshold": threshold, "postprocessing": recipe, "timing_scope": TIMING_SCOPE}
        case_started = time.perf_counter()
        reference = None
        image = probability = pred = None
        try:
            directory, metadata = pipeline.cache_path(base, record)
            image = sitk.ReadImage(str(probability_path(source, case_id)), sitk.sitkFloat32)
            assert_native_grid(image, metadata, case_id)
            probability = sitk.GetArrayFromImage(image)
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise pipeline.UserError(f"Saved probability must be finite and in [0,1]: {case_id}")
            reference = np.load(directory / "native_mask.npy", mmap_mode="r", allow_pickle=False)
            if reference.shape != probability.shape or np.any((reference != 0) & (reference != 1)):
                raise pipeline.UserError(f"Native reference shape or binary labels are invalid: {case_id}")
            spacing = metadata.get("native_spacing_zyx_mm", metadata["native_grid"]["spacing_xyz_mm"][::-1])
            pred = pipeline.postprocess_prediction(probability >= threshold, spacing, recipe)
            row.update(pipeline.evaluate_arrays(pred, reference, spacing, tolerance))
            row["native_spacing_z_mm"] = float(spacing[0])
        except (pipeline.UserError, OSError, RuntimeError, ValueError) as exc:
            row.update({"status": "failed", "reason": str(exc)})
        finally:
            if isinstance(reference, np.memmap):
                reference._mmap.close()
            # Release this native volume and its surface-metric temporaries
            # before opening the next probability file.
            del image, probability, pred, reference
        row["metric_rescore_seconds"] = time.perf_counter() - case_started
        rows.append(row)
        pipeline.atomic_json(out / "patient_metrics.json", rows)
        pipeline.write_csv(out / "patient_metrics.csv", rows)
        pipeline.atomic_json(out / "progress.json", {
            "updated": pipeline.now(), "completed_cases": len(rows), "expected_cases": len(records),
            "technical_failures": sum(r["status"] != "success" for r in rows),
            "elapsed_metric_rescore_seconds": time.perf_counter() - started, "timing_scope": TIMING_SCOPE,
        })
        if len(rows) % 5 == 0 or len(rows) == len(records) or row["status"] != "success":
            print(f"{len(rows)}/{len(records)} {case_id}: {row['status']} Dice={row.get('dice')}", flush=True)
    pipeline.atomic_json(out / "aggregates.json", pipeline.bootstrap_summary(rows, seed=seed, repeats=repeats))
    pipeline.write_csv(out / "failures.csv", [row for row in rows if row["status"] != "success"])
    if any(row["status"] != "success" for row in rows):
        raise pipeline.UserError(f"Rescoring had technical failures; partial records retained in {out}.")
    print(f"Completed {len(rows)} validation cases from saved probabilities: {out}")
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="Completed validation evaluation with saved native probabilities.")
    parser.add_argument("--operating-point", required=True, help="Frozen validation-selected threshold/component JSON for this source.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True, help="New output directory; existing results are never overwritten.")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        rescore_saved_validation(args)
    except (pipeline.UserError, OSError, ValueError, KeyError) as exc:
        print(f"Rescore failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
