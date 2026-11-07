"""Restore missing original caches from identity-verified existing local DICOM.

Original manifest, QC and split are never rewritten. --all verifies every missing
case's raw identity before restoration, stages one case at a time, then moves the
verified arrays within the same volume. Existing valid cache directories are kept.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ.setdefault(name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lung_ct_segmentation as pipeline


def destination(base, record):
    target = (base / record["cache_dir"]).resolve()
    if not target.is_relative_to((base / "cases").resolve()):
        raise pipeline.UserError("Manifest cache target escapes the original cases directory")
    return target


def source_row(raw_root, record, settings):
    patient_dir = raw_root / record["patient_id"]
    if not patient_dir.is_dir():
        raise pipeline.UserError(f"Missing raw patient directory: {patient_dir}")
    # A cache recovery already has the original SeriesInstanceUID and complete
    # source fingerprint. Inspect one header per series directory, then verify
    # EVERY file identity below. Preparation still inspects every CT header.
    # This avoids a redundant full-cohort header scan before actual recovery.
    pydicom = pipeline.require("pydicom")
    directories = {}
    for path in sorted(patient_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".dcm":
            directories.setdefault(path.parent, []).append(path)
    ct_files, structures = [], []
    uid, pid = record["source_identity"], record["patient_id"]
    for paths in directories.values():
        first = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
        modality = str(getattr(first, "Modality", ""))
        if modality == "CT" and str(first.SeriesInstanceUID) == uid:
            if str(first.PatientID) != pid:
                raise pipeline.UserError("Raw CT patient identity differs from original manifest")
            ct_files.extend(paths)
        elif modality == "RTSTRUCT":
            for path in paths:
                rt = pydicom.dcmread(str(path), stop_before_pixels=True)
                if uid in pipeline.dicom_references(rt)[0]:
                    structures.append(path)
    if not ct_files or len(structures) != 1:
        raise pipeline.UserError("Raw discovery did not recover exactly the original CT/RTSTRUCT pair")
    case_id = re.sub(r"[^A-Za-z0-9_.-]", "_", pid) + "-" + pipeline.stable_hash(uid)[:10]
    if case_id != record["case_id"]:
        raise pipeline.UserError("Raw CT/source identity does not reproduce the original case ID")
    row = {"case_id": case_id, "patient_id": pid, "route": "raw", "rtstruct": str(structures[0]),
           "ct_files": [str(p) for p in sorted(ct_files)], "series_uid": uid,
           "annotation_status": record["annotation_status"]}
    identities = [pipeline.file_identity(path, settings["source_hash"])
                  for path in row["ct_files"] + [row["rtstruct"]]]
    fingerprint = pipeline.stable_hash({"sources": identities, "settings": settings,
        "status": row["annotation_status"], "case_id": row["case_id"], "patient_id": row["patient_id"]})
    if fingerprint != record["fingerprint"]:
        raise pipeline.UserError(f"Raw source/settings fingerprint differs for {record['case_id']}; no cache identity may be relabelled")
    return row


def restore_one(record, raw_root, settings, output, base):
    case_output = output / "staging" / record["case_id"]
    prepare = argparse.Namespace(dicom_root=str(raw_root / record["patient_id"]), input_csv=None,
        images_dir=None, masks_dir=None, unlabeled=False, output=str(case_output),
        spacing=settings["spacing_zyx_mm"], boundary_width_mm=settings["boundary_width_mm"],
        roi=settings["target_roi"], source_hash=settings["source_hash"], release=settings["release"], qc_cases=0)
    pipeline.prepare_command(prepare)
    recovered, staging = pipeline.load_manifest(case_output / "manifest.json")
    if len(recovered["records"]) != 1 or recovered["records"][0] != record:
        raise pipeline.UserError("Regenerated case record differs from original. Staging retained; original files untouched.")
    directory, _ = pipeline.cache_path(staging, record)
    source = directory.resolve()
    if not source.is_relative_to((output / "staging").resolve()):
        raise pipeline.UserError("Recovery source escapes the explicitly named staging directory")
    target = destination(base, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise pipeline.UserError("Target appeared during recovery; no existing data overwritten")
    if source.drive.lower() != target.drive.lower():
        raise pipeline.UserError("Staging and destination must be on the same volume")
    # Both resolved absolute paths were checked inside prepared/staging and cases.
    source.rename(target)
    pipeline.cache_path(base, record)
    return str(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="prepared/nsclc/manifest.json")
    parser.add_argument("--splits", default="splits/main.json")
    parser.add_argument("--raw-root", default="data/nsclc_radiomics")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--case-id", help="Original training case ID; defaults to first original training case")
    select.add_argument("--all", action="store_true", help="Restore every original supervised case without changing partitions")
    parser.add_argument("--output", required=True, help="Separate recovery staging/progress directory on the cache volume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-free-ram-gib", type=float, default=1.5,
                        help="Wait before case preparation until this much system RAM is available")
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    manifest, base = pipeline.load_manifest(manifest_path)
    split = pipeline.validate_split(manifest, pipeline.load_json(args.splits))
    raw_root, output = Path(args.raw_root).resolve(), Path(args.output).resolve()
    if not output.is_relative_to(base.parent.resolve()) or output == base or output.is_relative_to(base):
        parser.error("--output must be a separate directory within the same prepared parent")
    progress_path = output / "recovery.json"
    if output.exists() and not args.resume:
        parser.error("Recovery output already exists; use --resume or a NEW directory")
    if args.resume:
        previous = pipeline.load_json(progress_path)
        if previous["manifest_fingerprint"] != manifest["fingerprint"] or previous["split_fingerprint"] != split["fingerprint"]:
            parser.error("Recovery data/split fingerprints differ from this progress record")
    selected = args.case_id or split["partitions"]["train"][0]
    if not args.all and selected not in split["partitions"]["train"]:
        parser.error("Single-case diagnostic recovery supports training cases only")
    by_id = {r["case_id"]: r for r in manifest["records"]}
    ids = sum((split["partitions"].get(p, []) for p in ("train", "val", "test", "external")), []) if args.all else [selected]
    records = [by_id[cid] for cid in ids]
    missing, existing = [], []
    for record in records:
        target = destination(base, record)
        if target.exists():
            pipeline.cache_path(base, record)
            existing.append(record["case_id"])
        else:
            missing.append(record)
    expected_bytes = sum(r["cache_bytes"] for r in missing)
    free_bytes = shutil.disk_usage(base).free
    required = int(expected_bytes * 1.02) + 2 * 2**30
    if free_bytes < required:
        raise pipeline.UserError(f"Insufficient disk: {free_bytes / 2**30:.2f} GiB free; conservative remaining requirement {required / 2**30:.2f} GiB")
    pipeline.require("SimpleITK").ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    manifest_digest, split_digest = pipeline.file_digest(manifest_path), pipeline.file_digest(args.splits)
    report = {"created": pipeline.now(), "status": "verifying_all_sources", "manifest_fingerprint": manifest["fingerprint"],
        "split_fingerprint": split["fingerprint"], "manifest_sha256_before": manifest_digest, "split_sha256_before": split_digest,
        "raw_root": str(raw_root), "destination": str(base), "selected_cases": len(records),
        "existing_valid_cases": existing, "missing_initial": len(missing), "source_verified_cases": [],
        "restored": [], "free_gib_before": free_bytes / 2**30, "expected_missing_gib": expected_bytes / 2**30,
        "settings": manifest["settings"], "thread_limit": 1}
    pipeline.atomic_json(progress_path, report)
    started = time.perf_counter()
    try:
        # Verify EVERY missing case before any restorative write. No statistics
        # are fitted here, and no patient may move across its original partition.
        for index, record in enumerate(missing, 1):
            source_row(raw_root, record, manifest["settings"])
            report["source_verified_cases"].append(record["case_id"])
            if index % 10 == 0 or index == len(missing):
                report["seconds"] = time.perf_counter() - started
                pipeline.atomic_json(progress_path, report)
                print(f"Original raw identity verified: {index}/{len(missing)}", flush=True)
        report["status"] = "restoring_missing_cases"
        pipeline.atomic_json(progress_path, report)
        for index, record in enumerate(missing, 1):
            try:
                import psutil
                while psutil.virtual_memory().available / 2**30 < args.min_free_ram_gib:
                    report["status"] = "waiting_for_available_memory"
                    report["seconds"] = time.perf_counter() - started
                    pipeline.atomic_json(progress_path, report)
                    print(f"Waiting for {args.min_free_ram_gib:.1f} GiB available system RAM before allocating case arrays", flush=True)
                    time.sleep(5)
                report["status"] = "restoring_missing_cases"
            except ImportError:
                pass
            case_started = time.perf_counter()
            restore_one(record, raw_root, manifest["settings"], output, base)
            elapsed = time.perf_counter() - case_started
            report["restored"].append({"case_id": record["case_id"], "seconds": elapsed, "cache_bytes": record["cache_bytes"]})
            report["seconds"] = time.perf_counter() - started
            pipeline.atomic_json(progress_path, report)
            mean = sum(r["seconds"] for r in report["restored"]) / index
            print(f"Restored original cache {index}/{len(missing)} in {elapsed:.1f}s; estimated remaining {mean * (len(missing) - index) / 60:.1f} min", flush=True)
            gc.collect()
        if pipeline.file_digest(manifest_path) != manifest_digest or pipeline.file_digest(args.splits) != split_digest:
            raise pipeline.UserError("Original manifest/split changed externally during recovery; investigate")
        report["status"], report["seconds"] = "complete_original_cache_restoration", time.perf_counter() - started
        report["manifest_split_preserved"] = True
        pipeline.atomic_json(progress_path, report)
        print(f"Recovery complete: {len(records)} original cases available; original manifest and split preserved", flush=True)
        return 0
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["reason"], report["seconds"] = str(exc), time.perf_counter() - started
        pipeline.atomic_json(progress_path, report)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except pipeline.UserError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
