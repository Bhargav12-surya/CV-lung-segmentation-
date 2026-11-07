#!/usr/bin/env python3
"""MSF-HSAT: a local, single-file 3D CT primary-tumor segmentation experiment.

Research task: primary GTV-1 membership, NOT a patient-level cancer diagnosis.
Run ``python lung_ct_segmentation.py --help`` and read the accompanying README.
No package installation, dataset download, cloud upload, or training occurs on import.

Axes are intentionally explicit: NumPy uses z,y,x; SimpleITK uses x,y,z;
networks use B,C,D,H,W. All physical distances in this program are millimeters.
The proposed architecture is a testable research hypothesis, not a proven advance.
"""
from __future__ import annotations

# 01. Imports and dependency boundaries ----------------------------------------
import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime as dt
import hashlib
import html
import importlib
import importlib.metadata
import importlib.util
import itertools
import json
import logging
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import signal
import sys
import tempfile
import time
import traceback
import uuid
from collections import OrderedDict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except ImportError:
    torch = nn = F = DataLoader = None
    Dataset = object

Module = nn.Module if nn is not None else object
VERSION = "1.0.0"
CACHE_SCHEMA = "msf-hsat-cache-v1"
BANDS = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
LOGGER = logging.getLogger("lung_ct")
PRESETS = {
    "compact": {"patch": (32, 96, 96), "widths": (16, 32, 64, 128, 192)},
    "balanced": {"patch": (48, 128, 128), "widths": (16, 32, 64, 128, 192)},
    "reference": {"patch": (64, 192, 192), "widths": (24, 48, 96, 192, 256)},
}


class UserError(RuntimeError):
    """An actionable input, geometry, dependency, or configuration error."""


def require(name: str):
    """Lazy imports keep --help/doctor usable before imaging packages are installed."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        package = {"SimpleITK": "SimpleITK", "rt_utils": "rt-utils",
                   "skimage.measure": "scikit-image"}.get(name, name.split(".")[0])
        raise UserError(f"Missing {name}. Install {package} in your active environment; "
                        "see README. This program never installs packages.") from exc


def require_torch():
    if torch is None:
        raise UserError("PyTorch is unavailable. Install the GPU build chosen at "
                        "https://pytorch.org/get-started/locally/ then run doctor.")
    if not hasattr(torch, "amp") or not hasattr(F, "scaled_dot_product_attention"):
        raise UserError("This implementation requires a modern PyTorch with torch.amp "
                        "and scaled_dot_product_attention (2.3 or newer recommended).")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean_json(obj):
    """Write strict JSON: undefined/infinite metrics are null + explicit status fields."""
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if np is not None and isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(clean_json(obj), sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(clean_json(obj), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(path, rows: Sequence[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean_json(v)) if isinstance(v, (dict, list, tuple))
                             else clean_json(v) for k, v in row.items()})
    os.replace(temp, path)


def append_csv(path, row: dict):
    """All history rows use the same schema, including null validation fields."""
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as stream:
            if next(csv.reader(stream)) != list(row):
                raise UserError(f"History schema mismatch: {path}")
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(clean_json(row))
        stream.flush()


def read_csv(path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path, mode="sha256"):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if mode == "sha256":
        result["sha256"] = file_digest(path)
    return result


def safe_case_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", str(value)) or value in (".", ".."):
        raise UserError(f"Unsafe case_id {value!r}; use letters, numbers, dots, dashes, underscores.")
    return str(value)


def positive_spacing(value):
    require("numpy")
    spacing = np.asarray(value, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise UserError(f"Expected three positive finite spacings in z,y,x millimeters; got {value}")
    return spacing


def validate_binary(mask, allow_empty=False):
    values = np.unique(mask)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise UserError(f"Reference mask must contain only 0 and 1, not {values[:12].tolist()}.")
    if not allow_empty and not np.any(mask):
        raise UserError("Empty reference in a positive case. Fix the target selection or explicitly "
                        "mark a verified negative; missing annotation is not a negative.")
    return np.asarray(mask, dtype=np.uint8)


def seed_everything(seed: int, deterministic=False):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def indexed_rng(seed, epoch, index):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(index)]))


def environment():
    result = {"time": now(), "program_version": VERSION, "python": sys.version,
              "platform": platform.platform(), "packages": {}, "cuda_available": False}
    for name in ("torch", "numpy", "scipy", "SimpleITK", "pydicom", "rt-utils",
                 "matplotlib", "tqdm", "scikit-image", "psutil"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if torch is not None:
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_build"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpus"] = []
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    result["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i),
                        "total_vram_gib": total_bytes / 2**30, "free_vram_gib": free_bytes / 2**30,
                        "bf16_supported": torch.cuda.is_bf16_supported()})
    try:
        import psutil
        result["available_ram_gib"] = psutil.virtual_memory().available / 2**30
    except ImportError:
        result["available_ram_gib"] = None
    return result


def doctor_command(args):
    result = environment()
    disk = shutil.disk_usage(Path(args.path).resolve())
    result["storage"] = {"path": str(Path(args.path).resolve()), "free_gib": disk.free / 2**30}
    result["guidance"] = [
        "Choose a compatible torch build: https://pytorch.org/get-started/locally/",
        "Core: python -m pip install numpy scipy SimpleITK matplotlib tqdm",
        "Raw RTSTRUCT: python -m pip install pydicom rt-utils",
        "Optional surface display: python -m pip install scikit-image",
        "Use --device cpu explicitly for debugging if CUDA is absent.",
        "Preset fit is unknown until a full forward/backward --preflight succeeds.",
    ]
    print(json.dumps(result, indent=2))
    if args.output:
        atomic_json(Path(args.output), result)


# 02. Physical grids, source matching, and disk-backed preparation --------------
def grid_dict(image):
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise UserError("Only scalar 3D CT/mask images are supported.")
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    positive_spacing(image.GetSpacing()[::-1])
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise UserError("Non-orthonormal/sheared image direction is unsupported; inspect geometry.")
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz_mm": list(image.GetOrigin()), "direction_xyz": list(image.GetDirection())}


def image_on_grid(array_zyx, grid):
    sitk = require("SimpleITK")
    image = sitk.GetImageFromArray(np.asarray(array_zyx))
    image.SetSpacing(tuple(grid["spacing_xyz_mm"]))
    image.SetOrigin(tuple(grid["origin_xyz_mm"]))
    image.SetDirection(tuple(grid["direction_xyz"]))
    if list(image.GetSize()) != grid["size_xyz"]:
        raise UserError("Array shape and recorded physical grid disagree.")
    return image


def empty_grid(grid, pixel_id=None):
    sitk = require("SimpleITK")
    image = sitk.Image([int(n) for n in grid["size_xyz"]], pixel_id or sitk.sitkFloat32)
    image.SetSpacing(tuple(grid["spacing_xyz_mm"]))
    image.SetOrigin(tuple(grid["origin_xyz_mm"]))
    image.SetDirection(tuple(grid["direction_xyz"]))
    return image


def grids_equal(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5, rtol=0)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3, rtol=0)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0))


def native_probability(probability, metadata):
    """Physical identity resampling reverses orientation/spacing without header copying."""
    sitk = require("SimpleITK")
    working = image_on_grid(np.asarray(probability, dtype=np.float32), metadata["working_grid"])
    native = empty_grid(metadata["native_grid"])
    return sitk.Resample(working, native, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def discover_pairs(args):
    rows = []
    if args.input_csv:
        source = Path(args.input_csv).resolve()
        for row in read_csv(source):
            for key in ("case_id", "patient_id", "image"):
                if not row.get(key):
                    raise UserError(f"{source}: every row needs {key}.")
            image = Path(row["image"])
            mask = Path(row["mask"]) if row.get("mask") else None
            row["image"] = str((source.parent / image).resolve() if not image.is_absolute() else image)
            row["mask"] = str((source.parent / mask).resolve() if mask and not mask.is_absolute() else mask or "")
            row["route"] = "paired"
            row["annotation_status"] = row.get("annotation_status") or ("positive" if mask else "unlabeled")
            rows.append(row)
    elif args.images_dir:
        images = sorted(p for p in Path(args.images_dir).iterdir() if p.name.endswith((".nii", ".nii.gz")))
        for image in images:
            stem = image.name.removesuffix(".gz").removesuffix(".nii")
            matches = [] if not args.masks_dir else [p for p in Path(args.masks_dir).iterdir()
                        if p.name in (stem + ".nii", stem + ".nii.gz")]
            if len(matches) != (0 if args.unlabeled else 1):
                raise UserError(f"{stem}: expected exactly one matching mask (or --unlabeled).")
            rows.append({"case_id": stem, "patient_id": stem, "image": str(image.resolve()),
                         "mask": str(matches[0].resolve()) if matches else "", "route": "paired",
                         "annotation_status": "unlabeled" if args.unlabeled else "positive"})
    else:
        raise UserError("prepare needs --input-csv, --images-dir, or --dicom-root.")
    if not rows:
        raise UserError("No input cases discovered; no synthetic substitute will be created.")
    return rows


def dicom_references(rt):
    """Collect actual referenced image/series UIDs instead of guessing directory order."""
    series, sops, frames = set(), set(), set()
    for frame in getattr(rt, "ReferencedFrameOfReferenceSequence", []):
        frames.add(str(frame.FrameOfReferenceUID))
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for item in getattr(study, "RTReferencedSeriesSequence", []):
                series.add(str(item.SeriesInstanceUID))
                sops.update(str(x.ReferencedSOPInstanceUID) for x in getattr(item, "ContourImageSequence", []))
    for roi in getattr(rt, "ROIContourSequence", []):
        for contour in getattr(roi, "ContourSequence", []):
            sops.update(str(x.ReferencedSOPInstanceUID) for x in getattr(contour, "ContourImageSequence", []))
    return series, sops, frames


def discover_dicom(args):
    pydicom = require("pydicom")
    groups, structures, ignored = {}, [], Counter()
    root = Path(args.dicom_root).resolve()
    LOGGER.info("Scanning DICOM headers under %s", root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        modality = str(getattr(ds, "Modality", ""))
        if modality == "CT":
            if getattr(ds, "NumberOfFrames", 1) != 1:
                ignored["enhanced_multiframe_ct_unsupported"] += 1
                continue
            uid = str(ds.SeriesInstanceUID)
            groups.setdefault(uid, []).append((path, ds))
        elif modality == "RTSTRUCT":
            structures.append((path, ds))
        elif modality == "SEG":
            ignored["DICOM_SEG_not_implemented_use_RTSTRUCT_or_paired_NIfTI"] += 1
    rows = []
    for rtpath, rt in structures:
        refs, sops, frames = dicom_references(rt)
        candidates = [uid for uid, slices in groups.items()
                      if (uid in refs or (not refs and sops and
                          sops.issubset({str(ds.SOPInstanceUID) for _, ds in slices}))) ]
        if len(candidates) != 1:
            rows.append({"case_id": "RT-" + stable_hash(str(rtpath))[:16],
                         "patient_id": str(getattr(rt, "PatientID", "UNKNOWN")), "route": "raw",
                         "discovery_error": f"RTSTRUCT matched {len(candidates)} CT series; require exactly one."})
            continue
        uid = candidates[0]
        slices = groups[uid]
        patient_ids = {str(ds.PatientID) for _, ds in slices}
        if len(patient_ids) != 1:
            raise UserError(f"Mixed patient IDs in CT SeriesInstanceUID {uid}")
        pid = next(iter(patient_ids))
        rows.append({"case_id": re.sub(r"[^A-Za-z0-9_.-]", "_", pid) + "-" + stable_hash(uid)[:10],
                     "patient_id": pid, "route": "raw", "rtstruct": str(rtpath),
                     "ct_files": [str(p) for p, _ in slices], "series_uid": uid,
                     "annotation_status": "positive"})
    LOGGER.info("Discovered %d RTSTRUCT-associated cases; ignored objects: %s", len(rows), dict(ignored))
    if not rows:
        raise UserError("No unambiguous CT+RTSTRUCT pairs found. SEG input is not implemented; "
                        "use a verified paired-volume conversion or RTSTRUCT source.")
    return rows


def validate_simple_polygon_xy(points):
    """Reject crossings/touching of nonadjacent edges without allocating an N*N matrix."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        raise UserError("Invalid planar polygon coordinates.")
    following = np.roll(points, -1, axis=0)
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    area = abs(float(np.sum(cross(points, following)))) / 2
    if area < 1e-6:
        raise UserError("Degenerate target contour.")
    for i in range(len(points)):
        indices = np.arange(i + 2, len(points))
        if i == 0:
            indices = indices[indices != len(points)-1]
        if not len(indices):
            continue
        a, b, c, d = points[i], following[i], points[indices], following[indices]
        # Bounding boxes avoid flagging intersections on disjoint collinear lines.
        boxes = (np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                 np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-8).all(axis=1)
        ab_c, ab_d = cross(b-a, c-a), cross(b-a, d-a)
        cd_a, cd_b = cross(d-c, a-c), cross(d-c, b-c)
        crossing = (ab_c * ab_d <= 1e-12) & (cd_a * cd_b <= 1e-12) & boxes
        if np.any(crossing):
            raise UserError("Self-intersecting/touching target contour requires verified external rasterization.")


def read_raw_case(row, roi_name):
    """Strict single-frame parallel-slice CT + CLOSED_PLANAR RTSTRUCT route.

    RT-utils rasterizes all contours on each plane together (OpenCV even/odd fill).
    We reject XOR, open, keyhole/self-intersecting and nonplanar contour encodings
    rather than pretending every possible RTSTRUCT encoding is implemented.
    A temporary folder isolates ONLY the referenced CT series for RT-utils.
    """
    pydicom, sitk = require("pydicom"), require("SimpleITK")
    builder = require("rt_utils").RTStructBuilder
    files = [Path(p) for p in row["ct_files"]]
    heads = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in files]
    first = heads[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    rowdir, coldir = orientation[:3], orientation[3:]
    normal = np.cross(rowdir, coldir)
    if not np.allclose(np.linalg.norm(normal), 1, atol=1e-4):
        raise UserError("Invalid CT orientation vectors.")
    for h in heads:
        if (str(h.SeriesInstanceUID) != row["series_uid"]
                or str(h.PatientID) != str(first.PatientID)
                or str(h.FrameOfReferenceUID) != str(first.FrameOfReferenceUID)
                or str(h.StudyInstanceUID) != str(first.StudyInstanceUID)
                or not np.allclose(h.ImageOrientationPatient, orientation, atol=1e-5)
                or not np.allclose(h.PixelSpacing, first.PixelSpacing, atol=1e-5)
                or (h.Rows, h.Columns) != (first.Rows, first.Columns)):
            raise UserError("CT series has mixed geometry.")
    positions = np.asarray([h.ImagePositionPatient for h in heads], dtype=float)
    order = np.argsort(positions @ normal)
    files, heads, positions = [files[i] for i in order], [heads[i] for i in order], positions[order]
    step = np.diff(positions @ normal)
    if len(step) == 0 or (step < 1e-4).any():
        raise UserError("Insufficient CT planes or duplicate slice positions.")
    spacing_z = float(np.median(step))
    if not np.allclose(step, spacing_z, atol=max(0.01, spacing_z * .01), rtol=0):
        raise UserError("Nonuniform/missing CT slice positions: this strict route excludes geometry gaps.")
    lateral = positions - positions[0] - np.outer((positions - positions[0]) @ normal, normal)
    if np.abs(lateral).max() > .05:
        raise UserError("Gantry tilt/lateral slice displacement requires a verified external conversion.")
    sop_set = {str(h.SOPInstanceUID) for h in heads}
    if len(sop_set) != len(heads):
        raise UserError("Duplicate SOPInstanceUIDs in CT series.")
    rt = pydicom.dcmread(row["rtstruct"], stop_before_pixels=True)
    _, referenced, frames = dicom_references(rt)
    if not referenced.issubset(sop_set):
        raise UserError("RTSTRUCT references CT slices missing from selected series.")
    if frames and frames != {str(first.FrameOfReferenceUID)}:
        raise UserError("RTSTRUCT and CT FrameOfReferenceUID mismatch.")
    rois = {str(r.ROIName): int(r.ROINumber) for r in rt.StructureSetROISequence}
    if len(rois) != len(rt.StructureSetROISequence):
        raise UserError("Ambiguous duplicate RTSTRUCT ROI names; select a curated source.")
    if roi_name not in rois:
        raise UserError(f"Requested ROI {roi_name!r} absent. Available exact names: {sorted(rois)}")
    selected = [r for r in rt.ROIContourSequence if int(r.ReferencedROINumber) == rois[roi_name]]
    if len(selected) != 1:
        raise UserError("Missing or duplicate primary ROI contour sequence.")
    for contour in getattr(selected[0], "ContourSequence", []):
        if str(contour.ContourGeometricType) != "CLOSED_PLANAR":
            raise UserError("Only CLOSED_PLANAR RTSTRUCT contours are supported in this route.")
        points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        if len(points) < 3 or np.ptp(points @ normal) > .05:
            raise UserError("Malformed/nonplanar target contour.")
        # Repeated non-closing vertices can encode keyholes; reject rather than fill incorrectly.
        core = points[:-1] if np.allclose(points[0], points[-1]) else points
        if len(np.unique(np.round(core, 5), axis=0)) != len(core):
            raise UserError("Keyhole/repeated contour vertices require external verified rasterization.")
        indices = np.argmin(abs((points[0] @ normal) - (positions @ normal)))
        if abs(points[0] @ normal - positions[indices] @ normal) > max(.1, spacing_z * .05):
            raise UserError("Contour does not lie on a selected CT plane.")
        contour_refs = {str(item.ReferencedSOPInstanceUID)
                        for item in getattr(contour, "ContourImageSequence", [])}
        if contour_refs and contour_refs != {str(heads[indices].SOPInstanceUID)}:
            raise UserError("Contour physical plane and referenced SOPInstanceUID disagree.")
        physical_delta = core - positions[indices]
        x = (physical_delta @ rowdir) / float(first.PixelSpacing[1])
        y = (physical_delta @ coldir) / float(first.PixelSpacing[0])
        validate_simple_polygon_xy(np.column_stack((x, y)))
        if x.min() < -.5 or x.max() > first.Columns - .5 or y.min() < -.5 or y.max() > first.Rows - .5:
            raise UserError("Target contour falls outside CT field of view.")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in files])
    image = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    # GDCM/SimpleITK already applies DICOM rescale slope/intercept. Do NOT apply them again.
    if not np.allclose(image.GetSpacing()[2], spacing_z, atol=.01):
        raise UserError("Image reader and inspected CT spacing disagree.")
    expected_direction = np.column_stack((rowdir, coldir, normal)).reshape(-1)
    if (image.GetSize() != (int(first.Columns), int(first.Rows), len(heads))
            or not np.allclose(image.GetOrigin(), positions[0], atol=.01)
            or not np.allclose(image.GetDirection(), expected_direction, atol=1e-4)
            or not np.allclose(image.GetSpacing()[:2], np.asarray(first.PixelSpacing)[::-1], atol=1e-5)):
        raise UserError("CT reader axes/origin/grid differ from inspected DICOM geometry.")
    with tempfile.TemporaryDirectory(prefix="ct-series-") as temp:
        for i, path in enumerate(files):
            target = Path(temp) / f"slice-{i:06d}.dcm"
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
        struct = builder.create_from(dicom_series_path=temp, rt_struct_path=row["rtstruct"])
        mask_rcs = struct.get_roi_mask_by_name(roi_name)
        actual_sops = [str(h.SOPInstanceUID) for h in struct.series_data]
        expected_sops = [str(h.SOPInstanceUID) for h in heads]
        if set(actual_sops) != sop_set or len(actual_sops) != len(expected_sops):
            raise UserError("RT-utils loaded unexpected CT images.")
        indices = [actual_sops.index(sop) for sop in expected_sops]
        if mask_rcs.shape != (int(first.Rows), int(first.Columns), len(heads)):
            raise UserError(f"Unexpected RT-utils mask axes {mask_rcs.shape}.")
        mask_zyx = np.transpose(mask_rcs[:, :, indices], (2, 0, 1)).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_zyx)
    mask.CopyInformation(image)
    acquisition = {"slice_thickness_mm": float_value(getattr(first, "SliceThickness", None)),
                   "reconstruction_kernel": str(getattr(first, "ConvolutionKernel", "")),
                   "series_uid": row["series_uid"], "target_roi": roi_name,
                   "frame_uid": str(first.FrameOfReferenceUID)}
    return image, mask, acquisition


def windows_from_hu(hu, channels=2):
    lung = (np.clip(hu, -1000, 400).astype(np.float32) + 1000) / 1400
    if channels == 1:
        return lung[None]
    soft = (np.clip(hu, -160, 240).astype(np.float32) + 160) / 400
    return np.stack((lung, soft), axis=0)


def boundary_band(mask, spacing_zyx_mm, width_mm):
    """Full-volume two-sided voxel-center band; never derive boundaries from patch cuts."""
    ndi = require("scipy.ndimage")
    positive_spacing(spacing_zyx_mm)
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.uint8)
    surface = binary ^ ndi.binary_erosion(binary, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    # Distance from foreground boundary VOXEL CENTERS: includes through-plane
    # surface voxels even when their spacing exceeds the requested band width.
    distance = ndi.distance_transform_edt(~surface, sampling=spacing_zyx_mm)
    return (distance <= width_mm).astype(np.uint8)


def prepare_command(args):
    require("numpy")
    sitk = require("SimpleITK")
    out = Path(args.output).resolve()
    if args.boundary_width_mm <= 0:
        raise UserError("--boundary-width-mm must be positive.")
    spacing = positive_spacing(args.spacing)
    if args.masks_dir and not args.images_dir:
        raise UserError("--masks-dir applies only to --images-dir.")
    if args.unlabeled and (not args.images_dir or args.masks_dir):
        raise UserError("--unlabeled applies only to an --images-dir without --masks-dir; CSV uses annotation_status.")
    rows = discover_dicom(args) if args.dicom_root else discover_pairs(args)
    ids = [safe_case_id(row["case_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise UserError("Duplicate case_id entries. Assign distinct case IDs and shared patient IDs.")
    out.mkdir(parents=True, exist_ok=True)
    settings = {"cache_schema": CACHE_SCHEMA, "spacing_zyx_mm": spacing.tolist(),
                "boundary_width_mm": args.boundary_width_mm, "target_roi": args.roi,
                "source_hash": args.source_hash, "release": args.release, "crop": "none_full_volume"}
    old = load_json(out / "manifest.json") if (out / "manifest.json").exists() else None
    if old and old["settings"] != settings:
        raise UserError("Existing preparation settings differ; use a new --output directory.")
    records, qc = [], []
    for row in rows:
        cid = row["case_id"]
        record = {"case_id": cid, "patient_id": row["patient_id"], "route": row["route"]}
        try:
            if row.get("discovery_error"):
                raise UserError(row["discovery_error"])
            if str(row["patient_id"]).upper() == "LUNG1-128":
                raise UserError("Known postoperative case LUNG1-128 has no primary GTV-1; excluded.")
            status = row.get("annotation_status", "positive")
            if status not in ("positive", "negative", "unlabeled"):
                raise UserError("annotation_status must be positive, negative, or unlabeled.")
            paths = (row.get("ct_files", []) + [row["rtstruct"]] if row["route"] == "raw"
                     else [row["image"]] + ([row["mask"]] if row.get("mask") else []))
            identities = [file_identity(path, args.source_hash) for path in paths]
            source_key = (row.get("series_uid") or identities[0].get("sha256") or
                          row.get("source_id") or stable_hash(identities[0]))
            fingerprint = stable_hash({"sources": identities, "settings": settings,
                                       "status": status, "case_id": cid, "patient_id": row["patient_id"]})
            relative = Path("cases") / f"{cid}-{fingerprint[:12]}"
            cache = out / relative
            if (cache / "complete.json").exists():
                metadata = load_json(cache / "metadata.json")
                marker = load_json(cache / "complete.json")
                if marker.get("fingerprint") != fingerprint:
                    raise UserError("Cache completion fingerprint mismatch.")
                for name, size in marker["files"].items():
                    if not (cache / name).exists() or (cache / name).stat().st_size != size:
                        raise UserError(f"Incomplete/corrupt cache: {name}. Preserve it and use new output.")
                cache_status = "hit"
            else:
                acquisition = {}
                if row["route"] == "raw":
                    image, mask_image, acquisition = read_raw_case(row, args.roi)
                else:
                    image = sitk.ReadImage(row["image"], sitk.sitkFloat32)
                    mask_image = sitk.ReadImage(row["mask"]) if row.get("mask") else None
                native_grid = grid_dict(image)
                if status != "unlabeled" and mask_image is None:
                    raise UserError("Labeled case has no mask path.")
                if status == "unlabeled" and mask_image is not None:
                    raise UserError("Unlabeled cases must not supply a reference mask.")
                if mask_image is not None:
                    grid_dict(mask_image)
                    if not grids_equal(image, mask_image):
                        raise UserError("CT/reference grids differ. Inspect registration; no header copying performed.")
                    native_mask = validate_binary(sitk.GetArrayFromImage(mask_image), status == "negative")
                    if status == "negative" and native_mask.any():
                        raise UserError("Explicit negative case contains foreground.")
                else:
                    native_mask = None
                native_spacing = list(image.GetSpacing()[::-1])
                oriented = sitk.DICOMOrient(image, "LPS")
                conditioning_native_spacing = list(oriented.GetSpacing()[::-1])
                target_xyz = spacing[::-1]
                size = np.ceil((np.asarray(oriented.GetSize()) - 1) *
                               np.asarray(oriented.GetSpacing()) / target_xyz).astype(int) + 1
                reference = sitk.Image(size.tolist(), sitk.sitkFloat32)
                reference.SetSpacing(tuple(target_xyz))
                reference.SetOrigin(oriented.GetOrigin())
                reference.SetDirection(oriented.GetDirection())
                working = sitk.Resample(oriented, reference, sitk.Transform(), sitk.sitkLinear,
                                        -1024., sitk.sitkFloat32)
                hu = sitk.GetArrayFromImage(working).astype(np.float32)
                if not np.isfinite(hu).all() or hu.min() < -40000 or hu.max() > 100000:
                    raise UserError("Nonfinite/implausible CT values; confirm input is HU, not normalized PNG data.")
                if mask_image is not None:
                    aligned = sitk.Resample(mask_image, reference, sitk.Transform(), sitk.sitkNearestNeighbor,
                                            0, sitk.sitkUInt8)
                    mask = validate_binary(sitk.GetArrayFromImage(aligned), status == "negative")
                    boundary = boundary_band(mask, spacing, args.boundary_width_mm)
                else:
                    mask = boundary = None
                metadata = {"schema": CACHE_SCHEMA, "fingerprint": fingerprint, "case_id": cid,
                            "patient_id": row["patient_id"], "annotation_status": status,
                            "native_grid": native_grid, "working_grid": grid_dict(working),
                            "native_spacing_zyx_mm": native_spacing,
                            "conditioning_native_spacing_zyx_mm": conditioning_native_spacing,
                            "target_spacing_zyx_mm": spacing.tolist(),
                            "working_shape_zyx": list(hu.shape), "crop_offset_zyx": [0, 0, 0],
                            "source_identity": source_key, "sources": identities,
                            "acquisition": acquisition, "settings": settings,
                            "hu_range": [float(hu.min()), float(hu.max())],
                            "native_reference_volume_ml": (float(native_mask.sum() * np.prod(native_spacing) / 1000)
                                                           if native_mask is not None else None),
                            "foreground_fraction": float(mask.mean()) if mask is not None else None}
                cache.parent.mkdir(parents=True, exist_ok=True)
                approximate_bytes = hu.nbytes + (mask.nbytes + boundary.nbytes + native_mask.nbytes
                                                if mask is not None else 0)
                free_bytes = shutil.disk_usage(cache.parent).free
                LOGGER.info("%s estimated cache %.2f GiB; free %.2f GiB", cid, approximate_bytes / 2**30, free_bytes / 2**30)
                if free_bytes < approximate_bytes + 256 * 2**20:
                    raise UserError("Insufficient destination storage for this case and safety margin; no old cache erased.")
                temp = cache.with_name(cache.name + ".partial-" + uuid.uuid4().hex)
                temp.mkdir()
                try:
                    np.save(temp / "hu.npy", hu, allow_pickle=False)
                    if mask is not None:
                        np.save(temp / "mask.npy", mask, allow_pickle=False)
                        np.save(temp / "boundary.npy", boundary, allow_pickle=False)
                        np.save(temp / "native_mask.npy", native_mask, allow_pickle=False)
                        fg = np.flatnonzero(mask)
                        if len(fg) > 100000:
                            fg = np.random.default_rng(0).choice(fg, 100000, replace=False)
                        np.save(temp / "fg_indices.npy", fg, allow_pickle=False)
                    atomic_json(temp / "metadata.json", metadata)
                    marker = {"fingerprint": fingerprint,
                              "files": {p.name: p.stat().st_size for p in temp.iterdir()}}
                    atomic_json(temp / "complete.json", marker)
                    if cache.exists():
                        raise UserError("Incomplete cache directory exists; retain it for inspection and use new output.")
                    os.replace(temp, cache)
                finally:
                    if temp.exists():
                        shutil.rmtree(temp)
                cache_status = "miss"
                if len(records) < args.qc_cases:
                    qc_panels(hu, mask, out / "qc" / f"{cid}-working.png", spacing, "Working-grid reference QC")
                    # Native and working panels retain their own physical aspect ratios.
                    qc_panels(sitk.GetArrayFromImage(image), native_mask,
                              out / "qc" / f"{cid}-native.png", native_spacing, "Native-grid reference QC")
            record.update({"cache_dir": relative.as_posix(), "fingerprint": fingerprint,
                           "source_identity": source_key, "annotation_status": status,
                           "native_spacing_zyx_mm": metadata["native_spacing_zyx_mm"],
                           "native_shape_zyx": metadata["native_grid"]["size_xyz"][::-1],
                           "working_shape_zyx": metadata["working_shape_zyx"],
                           "reference_volume_ml": metadata["native_reference_volume_ml"],
                           "foreground_fraction": metadata["foreground_fraction"],
                           "cache_bytes": sum(load_json(cache / "complete.json")["files"].values())})
            records.append(record)
            qc.append({"case_id": cid, "patient_id": row["patient_id"], "status": "included",
                       "cache": cache_status, "reason": "", "fingerprint": fingerprint})
            LOGGER.info("Prepared %s [%s] %s", cid, cache_status, status)
        except (UserError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            qc.append({"case_id": cid, "patient_id": row["patient_id"], "status": "excluded",
                       "cache": "failed", "reason": str(exc)})
            LOGGER.error("Excluded %s: %s", cid, exc)
    manifest = {"schema": CACHE_SCHEMA, "created": now(), "settings": settings, "records": records}
    manifest["fingerprint"] = stable_hash({"settings": settings, "records": records})
    if old and old["fingerprint"] != manifest["fingerprint"]:
        # Existing split/checkpoint identities must not silently start referring to a new cohort.
        atomic_json(out / ("candidate_manifest_" + uuid.uuid4().hex[:8] + ".json"), manifest)
        raise UserError("Prepared cohort differs from existing manifest. Candidate saved; use a new output "
                        "directory to change the cohort. The original manifest was retained.")
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "qc.json", qc)
    write_csv(out / "qc.csv", qc)
    write_csv(out / "manifest.csv", records)
    print(f"Prepared {len(records)}/{len(rows)} cases: {out / 'manifest.json'}")
    if not records:
        raise UserError("No eligible cases. Read qc.csv; training cannot proceed.")


def load_manifest(path):
    path = Path(path).resolve()
    manifest = load_json(path)
    expected = stable_hash({"settings": manifest["settings"], "records": manifest["records"]})
    if manifest.get("schema") != CACHE_SCHEMA or expected != manifest.get("fingerprint"):
        raise UserError("Manifest schema/fingerprint mismatch; do not hand-edit prepared manifests.")
    return manifest, path.parent


def cache_path(base, record):
    path = (Path(base) / record["cache_dir"]).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise UserError("Cache path escapes prepared-data directory.")
    metadata = load_json(path / "metadata.json")
    marker = load_json(path / "complete.json")
    if metadata["fingerprint"] != record["fingerprint"] or marker["fingerprint"] != record["fingerprint"]:
        raise UserError(f"Stale cache identity for {record['case_id']}")
    for name, size in marker["files"].items():
        if not (path / name).is_file() or (path / name).stat().st_size != size:
            raise UserError(f"Incomplete cache for {record['case_id']}: {name}")
    return path, metadata


# 03. Patient partitions, acquisition descriptors, and lazy patch sampling -----
def validate_split(manifest, split):
    if split["manifest_fingerprint"] != manifest["fingerprint"]:
        raise UserError("Split does not match this prepared manifest.")
    data = {r["case_id"]: r for r in manifest["records"]}
    partitions = split["partitions"]
    seen_cases, seen_patients, seen_sources = {}, {}, {}
    for name, cases in partitions.items():
        if name not in ("train", "val", "test", "external"):
            raise UserError(f"Unknown partition {name}")
        for cid in cases:
            if cid not in data:
                raise UserError(f"Split case missing from manifest: {cid}")
            r = data[cid]
            if r["annotation_status"] == "unlabeled":
                raise UserError("Unlabeled cases cannot appear in supervised partitions.")
            for value, seen, category in ((cid, seen_cases, "case"), (r["patient_id"], seen_patients, "patient"),
                                           (r["source_identity"], seen_sources, "source")):
                if value in seen and (seen[value] != name or category == "case"):
                    raise UserError(f"Duplicate/leaked {category} across partitions: {value}")
                seen[value] = name
    if split.get("fingerprint") != stable_hash({k: v for k, v in split.items() if k != "fingerprint"}):
        raise UserError("Split fingerprint mismatch.")
    return split


def split_command(args):
    manifest, _ = load_manifest(args.manifest)
    if Path(args.output).exists():
        validate_split(manifest, load_json(args.output))
        print(f"Existing immutable split validated: {args.output}")
        return
    if args.val_fraction <= 0 or not 0 < args.test_fraction < 1 or args.val_fraction + args.test_fraction >= 1:
        raise UserError("Need positive validation/test fractions with sum < 1.")
    records = [r for r in manifest["records"] if r["annotation_status"] != "unlabeled"]
    # Connected components group both patient IDs and duplicate source identities.
    parent = {r["case_id"]: r["case_id"] for r in records}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_patient, by_source = {}, {}
    for r in records:
        for lookup, key in ((by_patient, r["patient_id"]), (by_source, r["source_identity"])):
            if key in lookup:
                parent[find(r["case_id"])] = find(lookup[key])
            else:
                lookup[key] = r["case_id"]
    grouped = {}
    for r in records:
        grouped.setdefault(find(r["case_id"]), []).append(r["case_id"])
    groups = sorted(grouped.values(), key=lambda g: sorted(g)[0])
    np.random.default_rng(args.seed).shuffle(groups)
    if args.external:
        partitions = {"train": [], "val": [], "test": [], "external": sorted(sum(groups, []))}
    else:
        if len(groups) < 3:
            raise UserError("Need at least three independent patient/source groups for train/val/test.")
        nt = max(1, round(len(groups) * args.test_fraction))
        test_groups, development = groups[:nt], groups[nt:]
        if args.folds > 1:
            if not 0 <= args.fold < args.folds or len(development) < args.folds:
                raise UserError("Invalid fold index/count for available development groups.")
            validation = development[args.fold::args.folds]
            train = [g for i, g in enumerate(development) if i % args.folds != args.fold]
        else:
            nv = min(len(development) - 1, max(1, round(len(groups) * args.val_fraction)))
            validation, train = development[:nv], development[nv:]
        partitions = {"train": sorted(sum(train, [])), "val": sorted(sum(validation, [])),
                      "test": sorted(sum(test_groups, [])), "external": []}
    split = {"version": 1, "seed": args.seed, "fold": args.fold, "folds": args.folds,
             "manifest_fingerprint": manifest["fingerprint"], "partitions": partitions}
    split["fingerprint"] = stable_hash(split)
    validate_split(manifest, split)
    atomic_json(Path(args.output), split)
    print(json.dumps({k: len(v) for k, v in partitions.items()}, indent=2))


def choose_records(manifest, split, partition):
    validate_split(manifest, split)
    by_id = {r["case_id"]: r for r in manifest["records"]}
    result = [by_id[cid] for cid in split["partitions"][partition]]
    if not result:
        raise UserError(f"No cases in {partition} partition.")
    return result


def raw_descriptor(native_spacing, target_spacing, stage):
    native = positive_spacing(native_spacing)
    target = positive_spacing(target_spacing)
    stage_spacing = target * np.asarray(stage, dtype=float)
    return np.concatenate((np.log(native), np.log(target / native), np.log(stage_spacing))).astype(np.float32)


def acquisition_statistics(records, base):
    # One observation per training patient, not one per slice/patch/duplicate series.
    by_patient = {}
    for r in records:
        _, m = cache_path(base, r)
        by_patient.setdefault(r["patient_id"], []).append(raw_descriptor(
            m.get("conditioning_native_spacing_zyx_mm", m["native_spacing_zyx_mm"]), m["target_spacing_zyx_mm"], (2, 4, 4)))
    # Accumulate in float64: summing hundreds of identical float32 descriptors
    # can invent a ~1e-6 variance and defeat the constant-column guard.
    # Saved checkpoint statistics are still restored verbatim on resume/inference.
    values = np.stack([np.mean(np.asarray(v, dtype=np.float64), axis=0)
                       for v in by_patient.values()])
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    # E3 stage differs by known constant. Normalize each stage with its own train mean.
    return {"mean2": values.mean(axis=0).tolist(), "mean3": (values.mean(axis=0) +
            np.asarray([0.] * 6 + [math.log(2)] * 3)).tolist(), "std": std.tolist(),
            "training_patients": len(by_patient)}


def normalized_descriptors(metadata, stats):
    result = []
    for stage, key in (((2, 4, 4), "mean2"), ((4, 8, 8), "mean3")):
        raw = raw_descriptor(metadata.get("conditioning_native_spacing_zyx_mm", metadata["native_spacing_zyx_mm"]),
                             metadata["target_spacing_zyx_mm"], stage)
        result.append((raw - np.asarray(stats[key], dtype=np.float32)) / np.asarray(stats["std"], dtype=np.float32))
    return np.stack(result).astype(np.float32)  # stages E2,E3; shape 2,9


def extract_patch(array, start, shape, pad_value=0):
    """Returns a copy; the caller may augment it without modifying a read-only memmap."""
    shape = tuple(int(n) for n in shape)
    result = np.full(shape, pad_value, dtype=array.dtype)
    src, dst = [], []
    for n, a, p in zip(array.shape, start, shape):
        lo, hi = max(0, int(a)), min(n, int(a) + p)
        if hi <= lo:
            return result
        src.append(slice(lo, hi))
        dst.append(slice(lo - int(a), hi - int(a)))
    result[tuple(dst)] = array[tuple(src)]
    return result


def augment_patch(hu, mask, boundary, rng, enabled):
    if not enabled:
        return hu, mask, boundary
    ndi = require("scipy.ndimage")
    for axis in (1, 2):  # no default superior/inferior reversal
        if rng.random() < .5:
            hu, mask, boundary = [np.flip(a, axis=axis).copy() for a in (hu, mask, boundary)]
    if rng.random() < .65:
        angle, scale = rng.uniform(-10, 10) * math.pi / 180, rng.uniform(.9, 1.1)
        co, si = math.cos(angle) / scale, math.sin(angle) / scale
        matrix = np.array([[1., 0, 0], [0, co, -si], [0, si, co]])
        center = (np.asarray(hu.shape) - 1) / 2
        offset = center - matrix @ center
        hu = ndi.affine_transform(hu, matrix, offset, order=1, mode="constant", cval=-1024, prefilter=False)
        # The precomputed physical band is transformed with the target. Its width
        # therefore scales with the augmentation, an explicitly declared target convention.
        mask = ndi.affine_transform(mask, matrix, offset, order=0, mode="constant", cval=0, prefilter=False)
        boundary = ndi.affine_transform(boundary, matrix, offset, order=0, mode="constant", cval=0, prefilter=False)
    if rng.random() < .35:
        hu = (hu + rng.uniform(-30, 30)) * rng.uniform(.95, 1.05)
    if rng.random() < .2:
        hu = hu + rng.normal(0, rng.uniform(2, 15), size=hu.shape).astype(np.float32)
    if rng.random() < .15:
        hu = ndi.gaussian_filter(hu, sigma=(0, rng.uniform(.2, .6), rng.uniform(.2, .6)))
    # Intensity augmentation is applied in HU BEFORE generating both windows.
    return hu.astype(np.float32), mask.astype(np.uint8), boundary.astype(np.uint8)


class PatchDataset(Dataset):
    """Bounded per-worker memmap LRU, deterministic (seed,epoch,sample) randomness.

    No GPU tensors live here. Epoch is passed at construction; workers are recreated
    each epoch, so a stale persistent-worker copy cannot silently keep epoch zero.
    """
    def __init__(self, records, base, stats, patch, channels, samples, seed, epoch,
                 foreground_probability=.5, augment=True, max_open_cases=2, metadata_control="normal",
                 anatomical_background_probability=0., hard_negative_probability=0., hard_negatives=None):
        self.records, self.base, self.stats = records, str(base), stats
        self.patch, self.channels, self.samples = tuple(patch), channels, samples
        self.seed, self.epoch, self.foreground_probability = seed, epoch, foreground_probability
        self.augment, self.max_open_cases = augment, max_open_cases
        self.metadata_control = metadata_control
        self.anatomical_probability = anatomical_background_probability
        self.hard_probability = hard_negative_probability
        self.hard_negatives = hard_negatives or {}
        probabilities = (foreground_probability, anatomical_background_probability, hard_negative_probability)
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or sum(probabilities) > 1 + 1e-9:
            raise UserError("Sampling probabilities must be finite, nonnegative, and sum to at most one.")
        self._open = OrderedDict()
        self._q = []
        for r in records:
            _, m = cache_path(base, r)
            self._q.append(normalized_descriptors(m, stats))
        self.permutation = np.random.default_rng(seed + 712).permutation(len(records))

    def __len__(self):
        return self.samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        return state

    def close(self):
        for arrays in self._open.values():
            for value in arrays.values():
                if hasattr(value, "_mmap") and value._mmap is not None:
                    value._mmap.close()
        self._open.clear()

    def __getitem__(self, index):
        rng = indexed_rng(self.seed, self.epoch, index)
        selected = int(rng.integers(len(self.records)))
        if selected not in self._open:
            directory, metadata = cache_path(self.base, self.records[selected])
            arrays = {key: np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                      for key in ("hu", "mask", "boundary", "fg_indices")}
            if self.anatomical_probability:
                # Include nearby soft tissue as well as lung-like HU: sampling
                # only aerated lung neglects mediastinal false positives. This
                # image-derived region never uses a target for localization.
                coarse = arrays["hu"][::4, ::8, ::8]
                lung_like = (coarse >= -950) & (coarse <= -250)
                if lung_like.any():
                    coarse_spacing = np.asarray(metadata["target_spacing_zyx_mm"]) * (4, 8, 8)
                    near_lung = require("scipy.ndimage").distance_transform_edt(
                        ~lung_like, sampling=coarse_spacing) <= 24.
                    eligible = near_lung & (coarse >= -950) & (coarse <= 300)
                else:
                    eligible = np.zeros_like(lung_like)
                coords = np.argwhere(eligible)
                arrays["anatomical_centres"] = coords * np.asarray((4, 8, 8))
            self._open[selected] = arrays
            while len(self._open) > self.max_open_cases:
                _, expired = self._open.popitem(last=False)
                for value in expired.values():
                    if hasattr(value, "_mmap"):
                        value._mmap.close()
        self._open.move_to_end(selected)
        a = self._open[selected]
        kind, fallback = "uniform", False
        if not self.anatomical_probability and not self.hard_probability:
            # Preserve the exact old RNG sequence when new sampling is disabled.
            if len(a["fg_indices"]) and rng.random() < self.foreground_probability:
                kind = "foreground"
                center = np.asarray(np.unravel_index(int(rng.choice(a["fg_indices"])), a["mask"].shape))
            else:
                center = np.asarray([rng.integers(n) for n in a["hu"].shape])
        else:
            draw = rng.random()
            if draw < self.foreground_probability and len(a["fg_indices"]):
                kind = "foreground"
                center = np.asarray(np.unravel_index(int(rng.choice(a["fg_indices"])), a["mask"].shape))
            elif draw < self.foreground_probability + self.anatomical_probability:
                kind = "anatomical_background"
                center = self.background_center(a, a.get("anatomical_centres", ()), rng)
                fallback = center is None
            elif draw < self.foreground_probability + self.anatomical_probability + self.hard_probability:
                kind = "hard_negative"
                candidates = self.hard_negatives.get(self.records[selected]["case_id"], ())
                center = self.background_center(a, candidates, rng)
                fallback = center is None
            else:
                center = None
            if center is None:
                center = np.asarray([rng.integers(n) for n in a["hu"].shape])
                if fallback:
                    kind = "uniform_fallback"
        start = center - np.asarray(self.patch) // 2
        hu = extract_patch(a["hu"], start, self.patch, -1024.)
        mask = extract_patch(a["mask"], start, self.patch)
        band = extract_patch(a["boundary"], start, self.patch)
        hu, mask, band = augment_patch(hu, mask, band, rng, self.augment)
        q = self._q[selected].copy()
        if self.metadata_control == "constant":
            q.fill(0)
        elif self.metadata_control == "shuffled":
            q = self._q[int(self.permutation[selected])].copy()
        return {"image": torch.from_numpy(windows_from_hu(hu, self.channels).copy()),
                "mask": torch.from_numpy(mask[None].astype(np.float32)),
                "boundary": torch.from_numpy(band[None].astype(np.float32)),
                "q": torch.from_numpy(q), "sampling_kind": kind,
                "sampling_fallback": fallback, "contains_foreground": bool(mask.any()),
                "foreground_fraction": float(mask.mean())}

    def background_center(self, arrays, candidates, rng):
        """Try bounded training-background centres; never silently promise negatives."""
        if not len(candidates):
            return None
        for _ in range(24):
            center = np.asarray(candidates[int(rng.integers(len(candidates)))], dtype=int)
            start = center - np.asarray(self.patch) // 2
            bounds = tuple(slice(max(0, int(s)), min(n, int(s) + p))
                           for s, p, n in zip(start, self.patch, arrays["mask"].shape))
            if not arrays["mask"][bounds].any():
                return center
        return None


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_hard_negative_index(path, manifest, split, records, base):
    """Only accept bounded working-grid centres from this training partition."""
    if not path:
        return {}, None
    saved = load_json(path)
    fingerprint = stable_hash({k: v for k, v in saved.items() if k != "fingerprint"})
    if (saved.get("fingerprint") != fingerprint or saved.get("source_partition") != "train"
            or saved.get("manifest_fingerprint") != manifest["fingerprint"]
            or saved.get("split_fingerprint") != split["fingerprint"]
            or saved.get("coordinate_grid") != "working_zyx"):
        raise UserError("Hard negatives must be fingerprinted training-only predictions for this manifest/split.")
    train_ids = set(split["partitions"]["train"])
    if not set(saved.get("cases", {})).issubset(train_ids):
        raise UserError("Hard-negative index includes non-training patients.")
    result = {}
    for record in records:
        item = saved.get("cases", {}).get(record["case_id"])
        if item is None:
            continue
        _, metadata = cache_path(base, record)
        shape = metadata["working_shape_zyx"]
        if item.get("shape_zyx") != shape:
            raise UserError("Hard-negative coordinates use a different working grid.")
        centres = np.asarray(item["centres_zyx"])
        if centres.size == 0:
            continue
        if (centres.ndim != 2 or centres.shape[1] != 3 or not np.issubdtype(centres.dtype, np.integer)
                or (centres < 0).any() or (centres >= np.asarray(shape)).any()):
            raise UserError("Invalid hard-negative working-grid coordinates.")
        result[record["case_id"]] = centres.astype(np.int64)
    return result, fingerprint


# 04. Residual, multiscale, transformer, and wavelet building blocks -----------
@dataclass
class ModelConfig:
    name: str = "hybrid3d"
    channels: int = 2
    widths: tuple = (16, 32, 64, 128, 192)
    multiscale: bool = True
    transformer: bool = True
    adaptive_fusion: bool = True
    wavelet: bool = True
    decoder_conditioning: bool = True
    acquisition_conditioning: bool = True
    boundary: bool = True
    deep_supervision: bool = True
    spatial_gate: bool = False
    checkpointing: bool = False
    early_anisotropic: bool = True
    transformer_blocks: int = 2
    transformer_heads: int = 8
    window_heads: int = 4
    window: tuple = (4, 6, 6)
    max_global_tokens: int = 4096

    def validate(self):
        if self.name not in ("baseline3d", "hybrid3d") or self.channels not in (1, 2):
            raise UserError("Unknown model or invalid channel count.")
        if len(self.widths) != 5 or any(c < 4 for c in self.widths):
            raise UserError("Need five encoder widths, each >= 4.")
        if self.widths[-1] % self.transformer_heads:
            raise UserError("Bottleneck width must divide transformer head count.")
        if any(c % self.window_heads for c in self.widths[2:4]):
            raise UserError("E2/E3 widths must divide window-attention head count.")
        if self.wavelet and self.spatial_gate:
            raise UserError("Wavelet and spatial-only skip controls are mutually exclusive.")
        if self.name == "baseline3d" and any((self.multiscale, self.transformer, self.adaptive_fusion,
                                              self.wavelet, self.spatial_gate)):
            raise UserError("baseline3d must use its explicit residual U-Net configuration.")
        return self


def group_count(channels):
    # Never choose invalid GroupNorm divisibility after a width ablation.
    return next(n for n in (8, 4, 2, 1) if channels % n == 0)


class ConvNormAct(Module):
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1):
        super().__init__()
        kernel = (kernel,) * 3 if isinstance(kernel, int) else tuple(kernel)
        self.layers = nn.Sequential(
            nn.Conv3d(cin, cout, kernel, stride, tuple(k // 2 for k in kernel), groups=groups, bias=False),
            nn.GroupNorm(group_count(cout), cout), nn.SiLU(inplace=False))

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(Module):
    def __init__(self, cin, cout, stride=1, kernel=3):
        super().__init__()
        kernel = (kernel,) * 3 if isinstance(kernel, int) else tuple(kernel)
        self.first = ConvNormAct(cin, cout, kernel, stride)
        self.second = nn.Sequential(nn.Conv3d(cout, cout, kernel, padding=tuple(k // 2 for k in kernel), bias=False),
                                    nn.GroupNorm(group_count(cout), cout))
        self.shortcut = (nn.Identity() if cin == cout and stride in (1, (1, 1, 1)) else
                         nn.Conv3d(cin, cout, 1, stride=stride, bias=False))

    def forward(self, x):
        return F.silu(self.second(self.first(x)) + self.shortcut(x))


class MultiScaleResidual(Module):
    """C -> C/2; three parallel depthwise receptive fields -> C residual correction."""
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 2)
        self.reduce = ConvNormAct(channels, hidden, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv3d(hidden, hidden, 3, padding=d, dilation=d, groups=hidden, bias=False),
                          nn.GroupNorm(group_count(hidden), hidden), nn.SiLU()) for d in (1, 2, 3)])
        self.project = nn.Sequential(nn.Conv3d(3 * hidden, channels, 1, bias=False),
                                     nn.GroupNorm(group_count(channels), channels))

    def forward(self, x):
        reduced = self.reduce(x)
        return F.silu(x + self.project(torch.cat([branch(reduced) for branch in self.branches], dim=1)))


def position_encoding_3d(shape, channels, device, dtype):
    """Fixed separable sin/cos coordinates, generated for actual D,H,W, no fixed patch embedding."""
    coordinates = torch.meshgrid(*[torch.linspace(-1, 1, n, device=device) for n in shape], indexing="ij")
    frequency_count = max(1, math.ceil(channels / 6))
    frequencies = torch.exp(torch.arange(frequency_count, device=device, dtype=torch.float32) *
                            (-math.log(10000.) / max(1, frequency_count - 1)))
    values = []
    for coordinate in coordinates:
        angles = coordinate.reshape(-1, 1) * frequencies[None] * math.pi
        values += [torch.sin(angles), torch.cos(angles)]
    return torch.cat(values, dim=1)[None, :, :channels].to(dtype=dtype)


class AttentionBlock(Module):
    def __init__(self, channels, heads, expansion=2):
        super().__init__()
        if channels % heads:
            raise UserError("Attention channels must divide head count.")
        self.heads, self.head_dim = heads, channels // heads
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.qkv, self.project = nn.Linear(channels, 3 * channels), nn.Linear(channels, channels)
        self.ffn = nn.Sequential(nn.Linear(channels, expansion * channels), nn.GELU(),
                                 nn.Linear(expansion * channels, channels))

    def forward(self, tokens, valid=None):
        batch, length, channels = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # SDPA boolean masks: True means the key is allowed (opposite of some APIs).
        mask = valid[:, None, None, :] if valid is not None else None
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        x = tokens + self.project(attended)
        x = x + self.ffn(self.norm2(x))
        return x if valid is None else x * valid[:, :, None].to(x.dtype)


def partition_windows(x, window):
    """B,C,D,H,W -> (B*n_windows,N,C), reversible meta, valid-token mask."""
    batch, channels, d, h, w = x.shape
    wd, wh, ww = window
    pd, ph, pw = (-d) % wd, (-h) % wh, (-w) % ww
    padded = F.pad(x, (0, pw, 0, ph, 0, pd))
    dp, hp, wp = d + pd, h + ph, w + pw
    def pack(tensor):
        return (tensor.permute(0, 2, 3, 4, 1)
                .reshape(batch, dp // wd, wd, hp // wh, wh, wp // ww, ww, tensor.shape[1])
                .permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wd * wh * ww, tensor.shape[1]))
    valid = F.pad(torch.ones((batch, 1, d, h, w), device=x.device), (0, pw, 0, ph, 0, pd))
    return pack(padded), (batch, channels, d, h, w, dp, hp, wp, tuple(window)), pack(valid)[:, :, 0].bool()


def reverse_windows(tokens, metadata):
    batch, channels, d, h, w, dp, hp, wp, (wd, wh, ww) = metadata
    x = (tokens.reshape(batch, dp // wd, hp // wh, wp // ww, wd, wh, ww, channels)
         .permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(batch, dp, hp, wp, channels)
         .permute(0, 4, 1, 2, 3))
    return x[:, :, :d, :h, :w]


class WindowAttention3D(Module):
    def __init__(self, channels, heads=4, window=(4, 6, 6)):
        super().__init__()
        self.window = tuple(window)
        self.block = AttentionBlock(channels, heads)

    def forward(self, x):
        tokens, metadata, valid = partition_windows(x, self.window)
        tokens = tokens + position_encoding_3d(self.window, x.shape[1], x.device, x.dtype)
        return reverse_windows(self.block(tokens, valid), metadata)


class BottleneckTransformer(Module):
    def __init__(self, channels, config):
        super().__init__()
        self.limit = config.max_global_tokens
        self.capture, self.analysis_record = False, None
        self.blocks = nn.ModuleList([AttentionBlock(channels, config.transformer_heads)
                                     for _ in range(config.transformer_blocks)])

    def forward(self, x):
        if math.prod(x.shape[2:]) > self.limit:
            raise UserError(f"Bottleneck has {math.prod(x.shape[2:])} tokens; maximum is {self.limit}. "
                            "Use a smaller patch; dense input-voxel attention is forbidden.")
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + position_encoding_3d(x.shape[2:], x.shape[1], x.device, x.dtype)
        if self.capture:
            selected = torch.linspace(0, tokens.shape[1] - 1, min(128, tokens.shape[1]),
                                      device=x.device).round().long()
            probe = self.blocks[0].norm1(tokens[:, selected])
            block = self.blocks[0]
            qkv = block.qkv(probe).reshape(probe.shape[0], probe.shape[1], 3, block.heads, block.head_dim)
            q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attention = ((q.float() @ k.float().transpose(-1, -2)) / math.sqrt(block.head_dim)).softmax(-1)
            self.analysis_record = {"token_indices": selected.cpu().tolist(),
                "mean_head_attention": attention[0].mean(0).detach().cpu().tolist(),
                "scope": "restricted subset-key 128-token probe, not the full attention matrix"}
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape_as(x)


class Haar3D(Module):
    """Orthogonal one-level transform. Band letters describe z,y,x, in that order.

    L=[1,1]/sqrt(2); H=[1,-1]/sqrt(2), cross-correlation convention.
    Replicate-pad odd right/bottom/back edges; inverse crops to original dimensions.
    Energy equality refers to the even/padded tensor, not the unpadded odd input.
    All eight bands retain the full padded coefficient count.
    """
    def __init__(self):
        super().__init__()
        low = torch.tensor([1., 1.], dtype=torch.float64) / math.sqrt(2)
        high = torch.tensor([1., -1.], dtype=torch.float64) / math.sqrt(2)
        filters = [torch.einsum("i,j,k->ijk", z, y, x)
                   for z, y, x in itertools.product((low, high), repeat=3)]
        self.register_buffer("filters", torch.stack(filters)[:, None])

    def analysis(self, x):
        shape = tuple(x.shape[2:])
        padding = tuple(n % 2 for n in shape)
        if any(padding):
            x = F.pad(x, (0, padding[2], 0, padding[1], 0, padding[0]), mode="replicate")
        channels = x.shape[1]
        weight = self.filters.to(dtype=x.dtype).repeat(channels, 1, 1, 1, 1)
        out = F.conv3d(x, weight, stride=2, groups=channels)
        out = out.reshape(x.shape[0], channels, 8, *out.shape[2:]).transpose(1, 2)
        return out, shape  # B,8,C,D/2,H/2,W/2

    def synthesis(self, bands, original_shape):
        batch, _, channels, d, h, w = bands.shape
        coefficients = bands.transpose(1, 2).reshape(batch, 8 * channels, d, h, w)
        weight = self.filters.to(dtype=bands.dtype).repeat(channels, 1, 1, 1, 1)
        out = F.conv_transpose3d(coefficients, weight, stride=2, groups=channels)
        return out[:, :, :original_shape[0], :original_shape[1], :original_shape[2]]


class DetailDelta(Module):
    """Returns ONLY a correction, so R_b=H_b+delta(H_b) has exactly one identity."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                 nn.GroupNorm(group_count(channels), channels), nn.SiLU(),
                                 nn.Conv3d(channels, channels, 1))

    def forward(self, x):
        return self.net(x)


class ConditionedWaveletSkip(Module):
    def __init__(self, channels, config):
        super().__init__()
        self.haar = Haar3D()
        self.coarse = WindowAttention3D(channels, config.window_heads, config.window)
        self.details = nn.ModuleList([DetailDelta(channels) for _ in range(7)])
        self.decoder_conditioning = config.decoder_conditioning
        self.acquisition_conditioning = config.acquisition_conditioning
        hidden = max(8, channels // 4)
        self.context = nn.Conv3d(channels, hidden, 1) if self.decoder_conditioning else None
        self.descriptor = (nn.Sequential(nn.Linear(9, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                           if self.acquisition_conditioning else None)
        gate_inputs = channels + hidden * (int(self.decoder_conditioning) + int(self.acquisition_conditioning))
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv3d(gate_inputs, hidden, 1), nn.SiLU(), nn.Conv3d(hidden, channels, 1))
            for _ in range(7)])
        self.project = nn.Conv3d(channels, channels, 1, bias=False)
        self.kappa_logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))
        self.force_identity = False  # explicit mathematical self-test hook, off in normal execution
        self.capture = False
        self.analysis_record = None

    def forward(self, encoder, decoder, q):
        if encoder.shape != decoder.shape or q.shape != (encoder.shape[0], 9):
            raise UserError("Spectral skip encoder/context/descriptor shape mismatch.")
        bands, shape = self.haar.analysis(encoder)
        extra = []
        if self.context is not None:
            extra.append(F.adaptive_avg_pool3d(self.context(decoder), bands.shape[-3:]))
        if self.descriptor is not None:
            extra.append(self.descriptor(q).view(q.shape[0], -1, 1, 1, 1)
                         .expand(-1, -1, *bands.shape[-3:]))
        refined, gate_values = [self.coarse(bands[:, 0])], []
        for b in range(7):
            detail = bands[:, b + 1]
            gate = torch.sigmoid(self.gates[b](torch.cat([detail.abs()] + extra, dim=1)))
            refined.append(gate * (detail + self.details[b](detail)))
            if self.capture:
                gate_values.append(gate)
        changed = torch.stack(refined, dim=1)
        reconstruction = self.haar.synthesis(changed, shape)
        kappa = self.kappa_logit.sigmoid() * (0. if self.force_identity else 1.)
        output = encoder + kappa * self.project(reconstruction - encoder)
        if self.capture:
            # Bounded diagnostic snapshot from ONE selected inference patch, not every training step.
            d = bands.shape[-3] // 2
            def small_slice(t):
                return t[0, :min(4, t.shape[1]), d].detach().float().cpu().numpy()
            self.analysis_record = {
                "band_names": list(BANDS), "feature_shape": list(encoder.shape),
                "band_shape": list(bands.shape), "kappa": float(kappa.detach().cpu()),
                "before_energy": bands.float().square().mean(dim=(0, 2, 3, 4, 5)).detach().cpu().tolist(),
                "after_energy": changed.float().square().mean(dim=(0, 2, 3, 4, 5)).detach().cpu().tolist(),
                "band_slices": [small_slice(bands[:, b]) for b in range(8)],
                "gate_slices": [small_slice(g) for g in gate_values],
                "gate_mean": [float(g.float().mean().detach().cpu()) for g in gate_values],
                "gate_quantiles": [torch.quantile(g.detach().float().reshape(-1),
                                   torch.tensor([.05, .5, .95], device=g.device)).cpu().tolist()
                                   for g in gate_values],
                "interpretation": "Learned-feature diagnostics; not tissue spectra or causal explanations."}
        return output


class SpatialSkipControl(Module):
    """A comparable spatial gate with context/q, but no wavelet decomposition.

    Parameter counts are recorded; this is not asserted to be exactly capacity matched.
    """
    def __init__(self, channels, config):
        super().__init__()
        self.decoder_conditioning, self.acquisition_conditioning = config.decoder_conditioning, config.acquisition_conditioning
        self.q = nn.Linear(9, channels) if config.acquisition_conditioning else None
        self.gate = nn.Conv3d(channels * (1 + int(config.decoder_conditioning) + int(config.acquisition_conditioning)),
                              channels, 1)
        self.delta = DetailDelta(channels)

    def forward(self, encoder, decoder, descriptor):
        source = [encoder.abs()]
        if self.decoder_conditioning:
            source.append(decoder)
        if self.q is not None:
            source.append(self.q(descriptor)[:, :, None, None, None].expand_as(encoder))
        return encoder + .1 * torch.sigmoid(self.gate(torch.cat(source, 1))) * self.delta(encoder)


class AdaptiveFusion(Module):
    def __init__(self, source_channels, channels, adaptive=True):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv3d(c, channels, 1, bias=False) for c in source_channels])
        self.adaptive, self.channels, self.sources = adaptive, channels, len(source_channels)
        hidden = max(8, channels // 4)
        self.weights = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(),
                                     nn.Linear(hidden, self.sources * channels)) if adaptive else None
        self.concat = nn.Conv3d(self.sources * channels, channels, 1, bias=False) if not adaptive else None
        self.refine = ResidualBlock(channels, channels)
        self.capture, self.analysis_record = False, None

    def forward(self, features, shape):
        aligned = [F.interpolate(project(feature), size=shape, mode="trilinear", align_corners=False)
                   for project, feature in zip(self.projections, features)]
        if self.adaptive:
            summed = torch.stack(aligned, 1).sum(1)
            logits = self.weights(summed.mean(dim=(2, 3, 4))).view(summed.shape[0], self.sources, self.channels)
            weights = logits.float().softmax(dim=1)  # stable source softmax separately per CHANNEL
            if self.capture:
                if not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-4):
                    raise UserError("Fusion weights do not sum to one across sources.")
                self.analysis_record = weights[0].detach().float().cpu().numpy()
            fused = sum(feature * weights[:, i, :, None, None, None].to(feature.dtype)
                        for i, feature in enumerate(aligned))
        else:
            fused = self.concat(torch.cat(aligned, 1))
        return self.refine(fused)


# 05. Complete residual baseline and proposed 3D segmentation model ------------
class SegmentationModel(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.validate()
        c = config.widths
        stages = []
        for i in range(5):
            incoming = config.channels if i == 0 else c[i - 1]
            stride = (1, 1, 1) if i == 0 else ((1, 2, 2) if i == 1 else (2, 2, 2))
            kernel = (1, 3, 3) if config.early_anisotropic and i < 2 else (3, 3, 3)
            second = MultiScaleResidual(c[i]) if config.multiscale and i in (2, 3) else ResidualBlock(c[i], c[i], kernel=kernel)
            stages.append(nn.Sequential(ResidualBlock(incoming, c[i], stride, kernel), second))
        self.encoder = nn.ModuleList(stages)
        self.transformer = BottleneckTransformer(c[-1], config) if config.transformer else nn.Identity()
        self.up3, self.up2 = nn.Conv3d(c[4], c[3], 1), nn.Conv3d(c[3], c[2], 1)
        if config.wavelet:
            self.skip3, self.skip2 = ConditionedWaveletSkip(c[3], config), ConditionedWaveletSkip(c[2], config)
        elif config.spatial_gate:
            self.skip3, self.skip2 = SpatialSkipControl(c[3], config), SpatialSkipControl(c[2], config)
        else:
            self.skip3 = self.skip2 = None
        if config.name == "baseline3d":
            # A real two-source residual U-Net, without the proposed three-source fusion path.
            sources3, sources2, sources1 = [c[3], c[3]], [c[2], c[2]], [c[2], c[1]]
        else:
            sources3, sources2, sources1 = [c[3], c[3], c[2]], [c[2], c[2], c[1]], [c[2], c[1], c[0]]
        self.fuse3 = AdaptiveFusion(sources3, c[3], config.adaptive_fusion)
        self.fuse2 = AdaptiveFusion(sources2, c[2], config.adaptive_fusion)
        self.fuse1 = AdaptiveFusion(sources1, c[1], config.adaptive_fusion)
        self.final = nn.Sequential(ConvNormAct(c[1] + c[0], c[0], 1), ResidualBlock(c[0], c[0]))
        self.main_head = nn.Conv3d(c[0], 1, 1)
        self.boundary_head = nn.Conv3d(c[0], 1, 1) if config.boundary else None
        self.aux1 = nn.Conv3d(c[1], 1, 1) if config.deep_supervision else None
        self.aux2 = nn.Conv3d(c[2], 1, 1) if config.deep_supervision else None

    def run_block(self, block, x):
        if self.config.checkpointing and self.training and torch.is_grad_enabled():
            return activation_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, image, q, collect_aux=None):
        if image.ndim != 5 or image.shape[1] != self.config.channels or q.shape != (image.shape[0], 2, 9):
            raise UserError("Model expects image B,C,D,H,W and descriptor B,2,9.")
        features, value = [], image
        for stage in self.encoder:
            value = self.run_block(stage, value)
            features.append(value)
        e0, e1, e2, e3, e4 = features
        bottleneck = self.run_block(self.transformer, e4)
        u3 = F.interpolate(self.up3(bottleneck), size=e3.shape[2:], mode="trilinear", align_corners=False)
        s3 = self.skip3(e3, u3, q[:, 1]) if self.skip3 is not None else e3
        source3 = [u3, s3] + ([] if self.config.name == "baseline3d" else [e2])
        d3 = self.fuse3(source3, e3.shape[2:])
        u2 = F.interpolate(self.up2(d3), size=e2.shape[2:], mode="trilinear", align_corners=False)
        s2 = self.skip2(e2, u2, q[:, 0]) if self.skip2 is not None else e2
        source2 = [u2, s2] + ([] if self.config.name == "baseline3d" else [e1])
        d2 = self.fuse2(source2, e2.shape[2:])
        source1 = [d2, e1] + ([] if self.config.name == "baseline3d" else [e0])
        d1 = self.fuse1(source1, e1.shape[2:])
        d0 = self.final(torch.cat([F.interpolate(d1, size=e0.shape[2:], mode="trilinear", align_corners=False), e0], 1))
        output = {"logits": self.main_head(d0)}
        if collect_aux is None:
            collect_aux = self.training
        if collect_aux:
            if self.aux1 is not None:
                output["aux1"], output["aux2"] = self.aux1(d1), self.aux2(d2)
            if self.boundary_head is not None:
                output["boundary_logits"] = self.boundary_head(d0)
        return output

    def set_capture(self, enabled):
        for module in self.modules():
            if isinstance(module, (ConditionedWaveletSkip, AdaptiveFusion, BottleneckTransformer)):
                module.capture = enabled

    def captured_analysis(self):
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, (ConditionedWaveletSkip, AdaptiveFusion, BottleneckTransformer)) and module.analysis_record is not None:
                result[name] = clean_json(module.analysis_record)
        return result


def segmentation_loss(logits, target):
    """Per-example foreground soft Dice + mean BCE, then batch mean, FP32 reductions.

    Epsilon makes an exactly empty target/prediction well-defined. Sigmoid exists
    only on the Dice branch; BCE consumes raw logits for numerical stability.
    """
    logits, target = logits.float(), target.float()
    if logits.shape != target.shape:
        raise UserError(f"Loss grids disagree: {logits.shape} versus {target.shape}")
    probability = logits.sigmoid()
    axes = (1, 2, 3, 4)
    dice = 1 - (2 * (probability * target).sum(axes) + 1e-5) / (
        probability.sum(axes) + target.sum(axes) + 1e-5)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(axes)
    return (dice + bce).mean(), dice.mean(), bce.mean()


def total_loss(output, target, band, lambda_ds=1., lambda_boundary=.1):
    main, dice, bce = segmentation_loss(output["logits"], target)
    zero = main.new_zeros(())
    aux1 = aux2 = boundary = zero
    if "aux1" in output:
        aux1 = segmentation_loss(F.interpolate(output["aux1"], size=target.shape[2:], mode="trilinear", align_corners=False), target)[0]
        aux2 = segmentation_loss(F.interpolate(output["aux2"], size=target.shape[2:], mode="trilinear", align_corners=False), target)[0]
    if "boundary_logits" in output:
        boundary = F.binary_cross_entropy_with_logits(output["boundary_logits"].float(), band.float())
    loss = main + lambda_ds * (.5 * aux1 + .25 * aux2) + lambda_boundary * boundary
    parts = {"loss": loss, "dice_loss": dice, "bce_loss": bce,
             "aux1_loss": aux1, "aux2_loss": aux2, "boundary_loss": boundary}
    if not all(bool(torch.isfinite(value)) for value in parts.values()):
        raise UserError("Nonfinite loss. Check input HU, labels, precision, and learning rate; no NaNs were replaced.")
    return loss, {k: float(v.detach().cpu()) for k, v in parts.items()}


# 06. Physical complete-volume metrics and population summaries ----------------
def postprocessing_recipe(recipe="none", min_volume_ml=0.):
    """Normalize legacy string and versioned native-grid component recipes."""
    if isinstance(recipe, dict):
        mode = recipe.get("mode", "none")
        minimum = recipe.get("min_volume_ml", 0.)
        if recipe.get("connectivity", 6) != 6 or recipe.get("grid", "native") != "native":
            raise UserError("Only native-grid six-neighbour postprocessing recipes are supported.")
    else:
        mode, minimum = recipe, min_volume_ml
    if mode not in ("none", "largest", "min-volume"):
        raise UserError("Postprocessing must be none, largest, or min-volume.")
    minimum = float(minimum)
    if not math.isfinite(minimum) or minimum < 0 or (mode == "min-volume" and minimum <= 0):
        raise UserError("min-volume postprocessing requires a positive finite volume in mL.")
    return {"mode": mode, "min_volume_ml": minimum if mode == "min-volume" else 0.,
            "connectivity": 6, "grid": "native"}


def postprocess_prediction(mask, spacing, mode="none", min_volume_ml=0.):
    """Filter binary components on the native grid, without a reference mask.

    Largest-component ties select the first component in array traversal order.
    Physical minimum-volume filtering retains every component meeting the cutoff.
    """
    recipe = postprocessing_recipe(mode, min_volume_ml)
    spacing = positive_spacing(spacing)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise UserError("Postprocessing expects a 3D native-grid mask.")
    if recipe["mode"] == "none" or not mask.any():
        return mask.copy()
    ndi = require("scipy.ndimage")
    labels, count = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    if recipe["mode"] == "largest":
        return labels == int(np.argmax(sizes))
    keep = sizes * float(np.prod(spacing) / 1000) >= recipe["min_volume_ml"]
    keep[0] = False
    return keep[labels]


def evaluate_arrays(prediction, reference, spacing, tolerance_mm=2.):
    """Voxel-center surfaces with 6-neighbor erosion, unweighted pooled distances.

    This is NOT surfel-area-weighted Surface Dice. Never mix its values with a
    different convention without recomputing all models. Distances are in mm.
    """
    ndi = require("scipy.ndimage")
    spacing = positive_spacing(spacing)
    pred, ref = np.asarray(prediction, dtype=bool), np.asarray(reference, dtype=bool)
    if pred.ndim != 3 or ref.ndim != 3 or pred.shape != ref.shape:
        raise UserError("Metric arrays have different grids.")
    if tolerance_mm <= 0:
        raise UserError("Surface tolerance must be positive in mm.")
    p, r = int(pred.sum()), int(ref.sum())
    tp = int(np.count_nonzero(pred & ref))
    fp, fn = p - tp, r - tp
    voxel_ml = float(np.prod(spacing) / 1000)
    result = {"dice": 2 * tp / (p + r) if p + r else 1.,
              "iou": tp / (p + r - tp) if p + r - tp else 1.,
              "precision": tp / p if p else (1. if r == 0 else 0.),
              "recall": tp / r if r else (1. if p == 0 else 0.),
              "predicted_volume_ml": p * voxel_ml, "reference_volume_ml": r * voxel_ml,
              "relative_volume_error": (p - r) / r if r else None,
              "absolute_relative_volume_error": abs(p - r) / r if r else None,
              "false_positive_volume_ml": fp * voxel_ml, "false_negative_volume_ml": fn * voxel_ml,
              "complete_miss": int(r > 0 and p == 0), "reference_positive": int(r > 0),
              "empty_prediction": int(p == 0), "both_empty": int(p == 0 and r == 0),
              "zero_overlap": int(r > 0 and tp == 0),
              "disjoint_nonempty_prediction": int(r > 0 and p > 0 and tp == 0),
              "hd95_mm": None, "assd_mm": None, "surface_dice": None,
              "surface_tolerance_mm": tolerance_mm,
              "surface_convention": "6-neighbor boundary voxel centers; pooled unweighted distances"}
    if not p or not r:
        result["surface_status"] = "both_empty" if not p and not r else "one_empty_infinite_distance"
        result["surface_dice"] = 1. if not p and not r else 0.
        # Undefined distances remain null, explicitly counted in every summary.
        result["false_positive_components"] = int(ndi.label(pred)[1]) if not r else 0
        return result
    union_coordinates = np.nonzero(pred | ref)
    bounds = tuple(slice(max(0, int(c.min()) - 1), min(n, int(c.max()) + 2))
                   for c, n in zip(union_coordinates, pred.shape))
    pred, ref = pred[bounds], ref[bounds]
    structure = ndi.generate_binary_structure(3, 1)
    ps = pred ^ ndi.binary_erosion(pred, structure=structure, border_value=0)
    rs = ref ^ ndi.binary_erosion(ref, structure=structure, border_value=0)
    distance_to_ref = ndi.distance_transform_edt(~rs, sampling=spacing)
    p_to_r = distance_to_ref[ps]
    del distance_to_ref
    distance_to_pred = ndi.distance_transform_edt(~ps, sampling=spacing)
    r_to_p = distance_to_pred[rs]
    pooled = np.concatenate((p_to_r, r_to_p))
    result.update({"hd95_mm": float(np.quantile(pooled, .95)), "assd_mm": float(pooled.mean()),
                   "surface_dice": float(((p_to_r <= tolerance_mm).sum() + (r_to_p <= tolerance_mm).sum()) /
                                         (len(p_to_r) + len(r_to_p))), "surface_status": "finite"})
    labels, count = ndi.label(pred)
    touching = set(np.unique(labels[ref]).tolist()) - {0}
    result["false_positive_components"] = count - len(touching)
    return result


def bootstrap_summary(rows, seed=1234, repeats=1000):
    """One resampling unit per patient; multiple series average within their patient first."""
    valid_rows = [r for r in rows if r.get("status", "success") == "success"]
    # Infer the new accounting fields for historical metrics without rewriting
    # their files. A nonempty but disjoint prediction is NOT an empty prediction.
    def indicator(row, key):
        if key in row:
            return int(row[key])
        p, r, dice = row.get("predicted_volume_ml"), row.get("reference_volume_ml"), row.get("dice")
        if p is None or r is None:
            return 0
        return int({"empty_prediction": p == 0, "both_empty": p == 0 and r == 0,
                    "zero_overlap": r > 0 and dice == 0,
                    "disjoint_nonempty_prediction": p > 0 and r > 0 and dice == 0}[key])
    result = {"attempted_cases": len(rows), "successful_cases": len(valid_rows),
              "technical_failures": len(rows) - len(valid_rows),
              "complete_miss_cases": sum(int(r.get("complete_miss", 0)) for r in valid_rows),
              "nonfinite_surface_cases": sum(r.get("surface_status") in ("both_empty", "one_empty_infinite_distance") for r in valid_rows),
              "unknown_surface_status_cases": sum(r.get("surface_status") is None for r in valid_rows),
              "bootstrap_unit": "patient; case values averaged within patient",
              "failure_policy": "Technical failures are counted, retained in patient records, and excluded from available-case metric means; training checkpoint selection rejects any technical failure.",
              "metrics": {}}
    for key in ("empty_prediction", "both_empty", "zero_overlap", "disjoint_nonempty_prediction"):
        result[key + "_cases"] = sum(indicator(row, key) for row in valid_rows)
    for metric in ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm", "surface_dice",
                   "absolute_relative_volume_error", "inference_seconds", "peak_cuda_gib"):
        grouped = {}
        for row in valid_rows:
            value = row.get(metric)
            if value is not None and math.isfinite(float(value)):
                grouped.setdefault(row.get("patient_id") or row["case_id"], []).append(float(value))
        values = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
        if not len(values):
            result["metrics"][metric] = {"n_patients": 0, "mean": None, "ci95": None}
            continue
        rng = np.random.default_rng(seed)
        # Generate resamples in a loop to bound memory even for large cohorts.
        means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)])
        result["metrics"][metric] = {"n_patients": len(values), "mean": float(values.mean()),
             "median": float(np.median(values)), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
             "ci95": np.quantile(means, [.025, .975]).tolist(),
             "scope": "finite distances only; consult miss/nonfinite counts" if metric in ("hd95_mm", "assd_mm") else "available patients"}
    return result


# 07. Full-volume inference, original-grid exports, and recorded evaluations ---
def window_starts(length, patch, overlap):
    if length < 1 or patch < 1 or not 0 <= overlap < 1:
        raise UserError("Invalid window size/overlap; overlap must be in [0,1).")
    if length <= patch:
        return [0]
    stride = max(1, int(patch * (1 - overlap)))
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def gaussian_importance(patch):
    coordinates = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) / max(n / 8, 1)
                               for n in patch], indexing="ij")
    weight = np.exp(-.5 * sum(c * c for c in coordinates)).astype(np.float32)
    return np.maximum(weight, 1e-4)


def sliding_window_logits(volume, patch, overlap, predictor, accumulator_dir=None):
    """Complete CPU/disk accumulation of LOGITS. GPU holds at most one input patch.

    ``predictor`` maps one D,H,W HU patch to one D,H,W main-logit patch.
    This separation also lets mathematical coverage tests run without PyTorch.
    Temporary accumulators can be memory-mapped on disk; the returned logit map
    is an ordinary per-case array. No dataset-size-dependent resident cache exists.
    """
    shape, patch = tuple(volume.shape), tuple(int(v) for v in patch)
    importance = gaussian_importance(patch)
    if accumulator_dir is not None:
        Path(accumulator_dir).mkdir(parents=True, exist_ok=True)
    temp_context = (tempfile.TemporaryDirectory(prefix="window-", dir=accumulator_dir)
                    if accumulator_dir is not None else contextlib.nullcontext(None))
    with temp_context as temp:
        if temp is None:
            total, weight = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        else:
            total = np.lib.format.open_memmap(Path(temp) / "logits.npy", mode="w+", dtype=np.float32, shape=shape)
            weight = np.lib.format.open_memmap(Path(temp) / "weight.npy", mode="w+", dtype=np.float32, shape=shape)
            total[:] = weight[:] = 0
        try:
            for start in itertools.product(*[window_starts(n, p, overlap) for n, p in zip(shape, patch)]):
                hu = extract_patch(volume, start, patch, -1024.)
                logits = np.asarray(predictor(hu), dtype=np.float32)
                if logits.shape != patch or not np.isfinite(logits).all():
                    raise UserError(f"Invalid window logits {logits.shape}; expected {patch} and finite values.")
                extent = tuple(min(p, n - s) for p, n, s in zip(patch, shape, start))
                src = tuple(slice(0, n) for n in extent)
                dst = tuple(slice(s, s + n) for s, n in zip(start, extent))
                total[dst] += logits[src] * importance[src]
                weight[dst] += importance[src]
            if np.any(weight <= 0):
                raise UserError("Sliding-window coverage failure: uncovered voxels.")
            result = np.asarray(total / np.maximum(weight, 1e-8), dtype=np.float32).copy()
        finally:
            for array in (total, weight):
                if isinstance(array, np.memmap):
                    array.flush()
                    array._mmap.close()
    return result


def select_device(requested="auto", allow_cpu=False):
    require_torch()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if allow_cpu:
            return torch.device("cpu")
        raise UserError("No usable CUDA device. Run doctor, install a compatible GPU torch build, "
                        "or explicitly choose --device cpu for small debugging runs.")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UserError("Requested CUDA is unavailable. Run doctor.")
    if device.type not in ("cpu", "cuda"):
        raise UserError("Only CUDA and explicit CPU debugging are implemented/testable here; other backends are unsupported.")
    return device


def resolve_precision(device, requested):
    if requested == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device.type == "cuda" else "fp32")
    if device.type == "cpu" and requested != "fp32":
        raise UserError("CPU debugging uses fp32; choose --precision fp32.")
    if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise UserError("This CUDA device does not report BF16 support; choose fp16/fp32.")
    return requested


def autocast_context(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def inference_descriptor(metadata, stats, metadata_control):
    q = normalized_descriptors(metadata, stats)
    # Shuffling is a TRAINING-only fixed case-assignment intervention. Inference
    # uses true metadata; constant control deliberately stays zero at all stages.
    if metadata_control == "constant":
        q.fill(0)
    return q


def predict_working_volume(model, hu, metadata, stats, patch, overlap, device, precision,
                           metadata_control="normal", analysis=False, accumulator_dir=None):
    q = torch.from_numpy(inference_descriptor(metadata, stats, metadata_control)[None]).to(device)
    model.eval()
    captured = False
    analysis_record = {}
    def predictor(patch_hu):
        nonlocal captured, analysis_record
        # Capture the first spatial patch by default; it may contain background.
        # It is never selected using a hidden test reference mask.
        capture_this = analysis and not captured
        model.set_capture(capture_this)
        image = torch.from_numpy(windows_from_hu(patch_hu, model.config.channels)[None]).to(device)
        with torch.inference_mode(), autocast_context(device, precision):
            logits = model(image, q, collect_aux=False)["logits"][0, 0].float().cpu().numpy()
        if capture_this:
            analysis_record = model.captured_analysis()
            captured = True
        model.set_capture(False)
        return logits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    logits = sliding_window_logits(hu, patch, overlap, predictor, accumulator_dir)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    probability = require("scipy.special").expit(logits).astype(np.float32)
    return probability, {"inference_seconds": elapsed,
                         "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                         "analysis": analysis_record}


def safe_load_checkpoint(path, device="cpu"):
    require_torch()
    try:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError as exc:
        raise UserError("Installed torch lacks weights_only loading; update to a compatible release.") from exc
    if checkpoint.get("program_version") != VERSION:
        raise UserError("Checkpoint version differs from this program; use the matching source.")
    return checkpoint


def load_inference_model(path, device):
    saved = safe_load_checkpoint(path)
    config = ModelConfig(**saved["model_config"])
    model = SegmentationModel(config).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    return model, saved


def export_prediction(directory, probability_image, threshold, prediction=None):
    sitk = require("SimpleITK")
    directory.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(probability_image, str(directory / "probability.nii.gz"), True)
    if prediction is None:
        mask = sitk.Cast(probability_image >= threshold, sitk.sitkUInt8)
    else:
        mask = sitk.GetImageFromArray(np.asarray(prediction, dtype=np.uint8))
        if mask.GetSize() != probability_image.GetSize():
            raise UserError("Exported prediction and probability grids differ.")
        mask.CopyInformation(probability_image)
    sitk.WriteImage(mask, str(directory / "prediction.nii.gz"), True)
    return mask


def evaluate_records(model, records, base, stats, patch, overlap, threshold, tolerance,
                     device, precision, output=None, metadata_control="normal", figure_cases=6,
                     save_predictions=True, accumulator_dir=None, postprocessing="none"):
    sitk = require("SimpleITK")
    recipe = postprocessing_recipe(postprocessing)
    rows, all_analysis = [], []
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        cid = record["case_id"]
        row = {"case_id": cid, "patient_id": record["patient_id"], "status": "success"}
        try:
            directory, metadata = cache_path(base, record)
            hu = np.load(directory / "hu.npy", mmap_mode="r", allow_pickle=False)
            try:
                probability, timing = predict_working_volume(model, hu, metadata, stats, patch, overlap, device,
                    precision, metadata_control, analysis=output is not None and index < figure_cases,
                    accumulator_dir=accumulator_dir)
                native = native_probability(probability, metadata)
                native_array = sitk.GetArrayFromImage(native)
                pred = postprocess_prediction(native_array >= threshold, metadata["native_spacing_zyx_mm"], recipe)
                row.update({k: v for k, v in timing.items() if k != "analysis"})
                row.update({"native_spacing_z_mm": metadata["native_spacing_zyx_mm"][0],
                            "threshold": threshold, "postprocessing": recipe})
                if record["annotation_status"] == "unlabeled":
                    row["metric_status"] = "no_reference"
                else:
                    reference = np.load(directory / "native_mask.npy", mmap_mode="r", allow_pickle=False)
                    try:
                        row.update(evaluate_arrays(pred, reference, metadata["native_spacing_zyx_mm"], tolerance))
                        # Deterministic uniform native-grid voxel sample, no class balancing.
                        # The report treats every sampled voxel equally and states the sampling scheme.
                        if output is not None:
                            rng = np.random.default_rng(int(stable_hash(cid)[:8], 16))
                            chosen = rng.choice(native_array.size, min(20000, native_array.size), replace=False)
                            sample_path = output / "samples" / f"{cid}.npz"
                            sample_path.parent.mkdir(exist_ok=True)
                            np.savez_compressed(sample_path, probability=native_array.reshape(-1)[chosen],
                                                reference=np.asarray(reference).reshape(-1)[chosen],
                                                population_voxels=native_array.size, sampled_voxels=len(chosen))
                    finally:
                        reference._mmap.close()
                if output is not None:
                    case_out = output / "cases" / cid
                    case_out.mkdir(parents=True, exist_ok=True)
                    if save_predictions:
                        export_prediction(case_out, native, threshold, pred)
                    atomic_json(case_out / "metadata.json", {"case": record, "geometry": metadata,
                                "threshold": threshold, "postprocessing": recipe,
                                "logit_blending": "Gaussian sigma=patch/8; clipped min=1e-4",
                                "timing_includes": "working-grid sliding windows and input transfer; excludes native export/metrics"})
                    if index < figure_cases:
                        working_ref = (np.load(directory / "mask.npy", mmap_mode="r", allow_pickle=False)
                                       if record["annotation_status"] != "unlabeled" else None)
                        try:
                            native_mask = image_on_grid(pred.astype(np.uint8), metadata["native_grid"])
                            working_mask = sitk.GetArrayFromImage(sitk.Resample(
                                native_mask, empty_grid(metadata["working_grid"]), sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0., sitk.sitkUInt8)).astype(bool)
                            save_case_panels(hu, working_ref, probability, threshold, metadata["target_spacing_zyx_mm"],
                                             case_out, cid, prediction_mask=working_mask)
                        finally:
                            if isinstance(working_ref, np.memmap):
                                working_ref._mmap.close()
                    if timing["analysis"]:
                        analysis = {"case_id": cid, "native_spacing_zyx_mm": metadata["native_spacing_zyx_mm"],
                                    "selected_patch": "first sliding window, no reference-based selection",
                                    "modules": timing["analysis"]}
                        all_analysis.append(analysis)
                LOGGER.info("%s: Dice=%s, %.1fs", cid, row.get("dice", "unlabeled"), timing["inference_seconds"])
            finally:
                hu._mmap.close()
        except torch.cuda.OutOfMemoryError:
            raise UserError(f"CUDA OOM on case {cid}. Use a smaller NEW-run preset or CPU accumulation; "
                            "the architecture was not changed.")
        except (UserError, OSError, RuntimeError, ValueError) as exc:
            row.update({"status": "failed", "reason": str(exc)})
            LOGGER.error("Evaluation/prediction failed for %s: %s", cid, exc)
        rows.append(row)
        if output is not None:
            # Incremental write leaves a readable record if a later patient fails.
            atomic_json(output / "patient_metrics.json", rows)
            write_csv(output / "patient_metrics.csv", rows)
            atomic_json(output / "analysis.json", all_analysis)
    if output is not None:
        labeled = [r for r in rows if r.get("metric_status") != "no_reference"]
        atomic_json(output / "aggregates.json", bootstrap_summary(labeled))
        write_csv(output / "failures.csv", [r for r in rows if r["status"] != "success"])
    return rows


def inference_command(args):
    require_torch()
    device = select_device(args.device)
    manifest, base = load_manifest(args.manifest)
    model, saved = load_inference_model(args.checkpoint, device)
    precision = resolve_precision(device, args.precision)
    threshold = saved["threshold"]
    threshold_source = saved["threshold_source"]
    recipe = postprocessing_recipe("none")
    if args.operating_point:
        point = load_json(args.operating_point)
        fingerprint = stable_hash({k: v for k, v in point.items() if k != "fingerprint"})
        if (point.get("fingerprint") != fingerprint or point["source_partition"] != "val"
                or point["checkpoint_sha256"] != file_digest(args.checkpoint)
                or point["target"] != saved["target"] or point["split_fingerprint"] != saved["split_fingerprint"]
                or point["manifest_fingerprint"] != saved["manifest_fingerprint"]):
            raise UserError("Operating point is not a compatible frozen validation selection for this checkpoint.")
        threshold, threshold_source = point["threshold"], "frozen validation selection: " + point["fingerprint"]
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise UserError("Frozen operating threshold must be finite and in (0,1).")
        recipe = postprocessing_recipe(point.get("postprocessing", "none"))
        if "overlap" in point and args.overlap != point["overlap"]:
            raise UserError("Inference overlap differs from the frozen validation operating point.")
    partition = "unlabeled"
    if args.command == "evaluate":
        split = validate_split(manifest, load_json(args.splits))
        records = choose_records(manifest, split, args.partition)
        partition = args.partition
        if partition != "external":
            if saved["manifest_fingerprint"] != manifest["fingerprint"] or saved["split_fingerprint"] != split["fingerprint"]:
                raise UserError("Evaluation cohort/split differs from training. For separate data use partition external.")
        elif saved["manifest_fingerprint"] == manifest["fingerprint"]:
            raise UserError("External evaluation must use a separate manifest/cohort.")
        if partition == "external":
            training_ids = set(saved.get("cohort_patient_ids", []))
            source_ids = set(saved.get("cohort_source_identities", []))
            if any(r["patient_id"] in training_ids or r["source_identity"] in source_ids for r in records):
                raise UserError("External cohort overlaps known patient/source identities. Audit and remove duplicates.")
        if saved.get("overfit_cases"):
            LOGGER.warning("This checkpoint is an OVERFIT DIAGNOSTIC, not a generalization model.")
    else:
        records = [r for r in manifest["records"] if r["annotation_status"] == "unlabeled"]
        if not records:
            raise UserError("predict expects explicitly unlabeled prepared records. Use evaluate for labeled data.")
    out = Path(args.output).resolve()
    if out.exists():
        raise UserError("Inference output already exists; choose a new --output to preserve results.")
    out.mkdir(parents=True)
    protocol = {"program_version": VERSION, "created": now(), "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": file_digest(args.checkpoint), "model_name": model.config.name,
                "model_config": dataclasses.asdict(model.config), "manifest": str(Path(args.manifest).resolve()),
                "manifest_fingerprint": manifest["fingerprint"],
                "split_fingerprint": split["fingerprint"] if args.command == "evaluate" else None,
                "partition": partition, "case_ids": [r["case_id"] for r in records],
                "target": manifest["settings"]["target_roi"],
                "threshold": threshold, "threshold_source": threshold_source,
                "postprocessing": recipe, "surface_tolerance_mm": args.surface_tolerance_mm,
                "surface_convention": "6-neighbor boundary voxel centers; pooled unweighted distances",
                "patch": saved["patch"], "overlap": args.overlap, "precision": precision,
                "metadata_control": saved["metadata_control"], "parameters": sum(p.numel() for p in model.parameters()),
                "sampling": "uniform up to 20000 native voxels per case; inverse inclusion weights in pooled report"}
    if args.command == "evaluate" and protocol["target"] != saved["target"]:
        raise UserError("External target definition differs from training target.")
    atomic_json(out / "protocol.json", protocol)
    if args.operating_point:
        atomic_json(out / "operating_point.json", point)
    atomic_json(out / "environment.json", environment())
    rows = evaluate_records(model, records, base, saved["acquisition_stats"], saved["patch"], args.overlap,
                 threshold, args.surface_tolerance_mm, device, precision, out,
                 saved["metadata_control"], args.figure_cases, not args.no_save_predictions,
                 args.accumulator_dir, recipe)
    report_build(out, [out], args.figure_cases, args.dpi, args.vector, prepared_manifest=args.manifest)
    if any(r["status"] != "success" for r in rows):
        raise UserError(f"{sum(r['status'] != 'success' for r in rows)} cases failed; results retained in {out}.")
    print(f"Completed {len(rows)} cases: {out / 'report.html'}")


# 08. GPU preflight, atomic checkpoints, and resumable training ----------------
def model_config_from_args(args, preset):
    config = ModelConfig(name=args.model, channels=args.input_channels, widths=tuple(PRESETS[preset]["widths"]),
                         checkpointing=args.gradient_checkpointing, early_anisotropic=not args.isotropic_early)
    if args.model == "baseline3d":
        config.multiscale = config.transformer = config.adaptive_fusion = config.wavelet = False
        config.decoder_conditioning = config.acquisition_conditioning = False
    else:
        for key in ("multiscale", "transformer", "adaptive_fusion", "wavelet", "decoder_conditioning",
                    "acquisition_conditioning", "boundary", "deep_supervision"):
            if getattr(args, "no_" + key):
                setattr(config, key, False)
        config.spatial_gate = args.spatial_gate
        if config.spatial_gate:
            config.wavelet = False
    if args.no_boundary:
        config.boundary = False
    if args.no_deep_supervision:
        config.deep_supervision = False
    return config.validate()


def preflight_model(config, patch, batch_size, device, precision, lambda_ds, lambda_boundary,
                    real_batch=None):
    """Two full backward/AdamW steps, with real training patches when available."""
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = optimizer = image = target = output = loss = None
    try:
        model = SegmentationModel(config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        if real_batch is None:
            image = torch.rand((batch_size, config.channels, *patch), device=device)
            target = torch.zeros((batch_size, 1, *patch), device=device)
            target[:, :, patch[0]//4:patch[0]//2, patch[1]//4:patch[1]//2, patch[2]//4:patch[2]//2] = 1
            band = target
            q = torch.zeros((batch_size, 2, 9), device=device)
        else:
            image, target, band, q = [real_batch[k].to(device) for k in ("image", "mask", "boundary", "q")]
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                output = model(image, q, True)
                loss, _ = total_loss(output, target, band, lambda_ds, lambda_boundary)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
            if not torch.isfinite(norm):
                raise UserError("Nonfinite gradients during memory preflight.")
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {"status": "passed", "patch": list(patch), "batch_size": batch_size,
                "precision": precision, "seconds": time.perf_counter() - started,
                "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                "parameters": sum(p.numel() for p in model.parameters()),
                "optimizer_steps": 2, "target_foreground_fraction": float(target.float().mean().item()),
                "gradient_norm": float(norm.item()),
                "scope": ("two real-training-patch optimizer steps; allocation/finite-gradient probe, not accuracy"
                          if real_batch is not None else
                          "two synthetic positive-target optimizer steps; allocation probe, not real-data accuracy")}
    finally:
        del loss, output, image, target, optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def capture_rng():
    # No NumPy ndarray/custom globals in checkpoints: weights_only=True can load these types.
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), int(state[2]),
             int(state[3]), float(state[4])], "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise UserError("CUDA device-count differs from checkpoint RNG state; use the matching setup for resume.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_train_args(args):
    for key in ("max_epochs", "steps_per_epoch", "batch_size", "accumulation", "val_every"):
        if getattr(args, key) <= 0:
            raise UserError(f"--{key.replace('_','-')} must be positive.")
    if args.workers < 0 or not 0 <= args.foreground_probability <= 1:
        raise UserError("Invalid workers or foreground probability.")
    probabilities = [args.foreground_probability,
                     getattr(args, "anatomical_background_probability", 0.),
                     getattr(args, "hard_negative_probability", 0.)]
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or sum(probabilities) > 1 + 1e-9:
        raise UserError("Foreground, anatomical-background, and hard-negative probabilities must sum to at most 1.")
    if getattr(args, "hard_negative_probability", 0.) and not getattr(args, "hard_negative_index", None):
        raise UserError("--hard-negative-probability requires a train-only --hard-negative-index.")
    if getattr(args, "preflight_only", False) and args.resume:
        raise UserError("Preflight-only is a new allocation diagnostic; do not combine it with --resume.")
    if args.lr <= 0 or args.weight_decay < 0 or args.lambda_ds < 0 or args.lambda_boundary < 0:
        raise UserError("Invalid optimizer/loss hyperparameter.")
    if not 0 < args.threshold < 1:
        raise UserError("Prespecified threshold must be between zero and one.")
    if args.surface_tolerance_mm <= 0:
        raise UserError("Surface tolerance must be positive.")
    if args.warmup_epochs < 0 or args.early_stopping_checks < 0 or args.clip_grad <= 0:
        raise UserError("Warmup/early-stopping counts must be nonnegative; gradient clipping must be positive.")
    for key in ("lr", "weight_decay", "clip_grad", "foreground_probability", "lambda_ds", "lambda_boundary",
                "threshold", "surface_tolerance_mm"):
        if not math.isfinite(getattr(args, key)):
            raise UserError(f"Nonfinite --{key.replace('_','-')} is invalid.")


def train_command(args):
    require_torch()
    require("numpy")
    require("scipy")
    require("SimpleITK")
    tqdm = require("tqdm").tqdm
    manifest, base = load_manifest(args.manifest)
    split = validate_split(manifest, load_json(args.splits))
    run = Path(args.run).resolve()
    resume = safe_load_checkpoint(args.resume) if args.resume else None
    if resume is not None:
        if resume["manifest_fingerprint"] != manifest["fingerprint"] or resume["split_fingerprint"] != split["fingerprint"]:
            raise UserError("Resume data/split fingerprint mismatch. Start a separately labeled new run.")
        if run != Path(resume["run_path"]).resolve():
            raise UserError("Resume must use the original --run directory (move the whole run only with care).")
        # Missing fields in legacy checkpoints mean old sampling, never the
        # current CLI's new settings. Explicit recipe changes require a new run.
        for key, default in (("anatomical_background_probability", 0.), ("hard_negative_probability", 0.),
                             ("hard_negative_index", None), ("preflight_only", False)):
            if key not in resume["train_args"]:
                flag = "--" + key.replace("_", "-")
                if flag in args._provided and getattr(args, key) != default:
                    raise UserError(f"Legacy resume cannot change {flag}; start a new run.")
                setattr(args, key, default)
        # All scientific/optimization settings are restored. Explicit conflicting CLI
        # changes are rejected rather than silently ignored or applied halfway through.
        for key, value in resume["train_args"].items():
            if key in ("device", "manifest", "splits", "run", "resume", "_provided", "command"):
                continue
            flag = "--" + key.replace("_", "-")
            if flag in args._provided and clean_json(getattr(args, key, None)) != clean_json(value):
                raise UserError(f"Resume cannot change {flag}. Saved value: {value!r}; start a new run.")
            setattr(args, key, value)
    validate_train_args(args)
    device = select_device(args.device)
    precision = resolve_precision(device, resume["precision"] if resume else args.precision)
    seed_everything(args.seed, args.deterministic)
    train_records = choose_records(manifest, split, "train")
    validation_records = choose_records(manifest, split, "val")
    if args.overfit_cases:
        if args.overfit_cases not in (1, 2):
            raise UserError("--overfit-cases supports one or two actual training cases.")
        train_records = train_records[:args.overfit_cases]
        validation_records = train_records
        args.no_augmentation = True
    stats = resume["acquisition_stats"] if resume else acquisition_statistics(train_records, base)
    hard_negatives, hard_fingerprint = load_hard_negative_index(
        getattr(args, "hard_negative_index", None), manifest, split, train_records, base)
    if getattr(args, "hard_negative_probability", 0.) and not hard_negatives:
        raise UserError("Hard-negative sampling requested but this training subset has no mined centres.")
    if resume and resume.get("hard_negative_index_fingerprint") != hard_fingerprint:
        raise UserError("Hard-negative index changed since this run was saved.")
    implementation_sha256 = file_digest(Path(__file__))
    normalization_version = resume.get("normalization_version", "legacy-saved") if resume else "float64-training-patients-v1"
    if resume:
        config = ModelConfig(**resume["model_config"])
        patch, preset = tuple(resume["patch"]), resume["preset"]
        preflight = resume["preflight"]
    else:
        if run.exists():
            raise UserError("Run directory already exists; use --resume or a new --run.")
        candidates = [args.preset] if args.preset != "auto" else ["balanced", "compact"]
        # Auto is deliberately bounded and conservative: reference is opt-in.
        preflight = []
        config = None
        for preset in candidates:
            candidate = model_config_from_args(args, preset)
            patch = tuple(args.patch or PRESETS[preset]["patch"])
            if min(patch) < 8:
                raise UserError("Training patches must be at least 8 voxels in each dimension.")
            try:
                if args.preflight or args.preset == "auto" or getattr(args, "preflight_only", False):
                    LOGGER.info("Preflight: %s, patch=%s, batch=%s", preset, patch, args.batch_size)
                    probe_data = PatchDataset(train_records, base, stats, patch, candidate.channels,
                        args.batch_size, args.seed, 0, foreground_probability=1., augment=False)
                    try:
                        probe_items = [probe_data[i] for i in range(args.batch_size)]
                        real_batch = {key: torch.stack([r[key] for r in probe_items])
                                      for key in ("image", "mask", "boundary", "q")}
                    finally:
                        probe_data.close()
                    result = preflight_model(candidate, patch, args.batch_size, device, precision,
                                             args.lambda_ds, args.lambda_boundary, real_batch)
                else:
                    result = {"status": "not_run", "reason": "--preflight not selected; fit is unverified"}
                result["preset"] = preset
                preflight.append(result)
                config = candidate
                break
            except torch.cuda.OutOfMemoryError:
                preflight.append({"preset": preset, "status": "oom", "patch": patch})
                torch.cuda.empty_cache()
                if args.patch:
                    break  # changing only channel preset cannot guarantee a custom patch will fit
        if config is None:
            raise UserError(f"Preflight failed: {preflight}. Choose a smaller patch/batch in a new run.")
        run.mkdir(parents=True)
    # Probe RNG consumption must not change the actual initial weights or sample stream.
    seed_everything(args.seed, args.deterministic)
    model = SegmentationModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(args.steps_per_epoch / args.accumulation)
    total_updates = args.max_epochs * updates_per_epoch
    warmup_updates = min(args.warmup_epochs * updates_per_epoch, max(0, total_updates - 1))
    def schedule(step):
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return .01 + .99 * .5 * (1 + math.cos(math.pi * min(1., max(0., progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    start_epoch, global_step, best_metric, stale_checks = 0, 0, -math.inf, 0
    history = []
    if resume:
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        start_epoch, global_step = resume["completed_epochs"], resume["global_step"]
        best_metric, stale_checks, history = resume["best_metric"], resume["stale_checks"], resume["history"]
        restore_rng(resume["rng"])
        write_csv(run / "history.csv", history)
        atomic_json(run / "history.json", history)
    train_args = {k: clean_json(v) for k, v in vars(args).items() if not k.startswith("_") and k != "handler"}
    resolved = {"model_config": dataclasses.asdict(config), "patch": patch, "preset": preset,
                "precision": precision, "device": str(device), "train_args": train_args,
                "manifest_fingerprint": manifest["fingerprint"], "split_fingerprint": split["fingerprint"],
                "acquisition_stats": stats, "preflight": preflight,
                "parameters": sum(p.numel() for p in model.parameters()),
                "normalization_version": normalization_version, "implementation_sha256": implementation_sha256,
                "hard_negative_index_fingerprint": hard_fingerprint,
                "physical_patch_zyx_mm": (np.asarray(patch) * np.asarray(manifest["settings"]["spacing_zyx_mm"])).tolist(),
                "selection_criterion": "mean per-patient full-volume validation Dice" if not args.overfit_cases else
                                       "mean training-case diagnostic full-volume Dice", "created": now()}
    atomic_json(run / "resolved_config.json", resolved)
    atomic_json(run / "environment.json", environment())
    atomic_json(run / "splits.json", split)
    protocol = {"program_version": VERSION, "model_name": config.name, "model_config": dataclasses.asdict(config),
                "manifest": str(Path(args.manifest).resolve()), "manifest_fingerprint": manifest["fingerprint"],
                "split_fingerprint": split["fingerprint"], "partition": "development",
                "threshold": args.threshold, "threshold_source": "prespecified before training, not test-tuned",
                "surface_tolerance_mm": args.surface_tolerance_mm,
                "surface_convention": "6-neighbor boundary voxel centers; pooled unweighted distances",
                "postprocessing": "none", "target": manifest["settings"]["target_roi"],
                "parameters": resolved["parameters"], "precision": precision, "patch": patch,
                "overfit_cases": args.overfit_cases}
    atomic_json(run / "protocol.json", protocol)
    if not resume:
        shutil.copy2(Path(__file__), run / "implementation.py")
    if getattr(args, "preflight_only", False):
        atomic_json(run / "preflight_only.json", {"status": "passed", "preflight": preflight,
                     "scope": "no training run or accuracy result; discarded probe model"})
        print(f"Real-patch preflight passed: {run / 'resolved_config.json'}")
        return 0
    def make_checkpoint(completed_epochs):
        return {"program_version": VERSION, "run_path": str(run), "model_config": dataclasses.asdict(config),
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                "completed_epochs": completed_epochs, "global_step": global_step, "best_metric": best_metric,
                "stale_checks": stale_checks, "history": history, "rng": capture_rng(), "precision": precision,
                "patch": patch, "preset": preset, "preflight": preflight, "train_args": train_args,
                "acquisition_stats": stats, "manifest_fingerprint": manifest["fingerprint"],
                "split_fingerprint": split["fingerprint"], "target": protocol["target"],
                "threshold": args.threshold, "threshold_source": protocol["threshold_source"],
                "metadata_control": args.metadata_control, "overfit_cases": args.overfit_cases,
                "normalization_version": normalization_version, "implementation_sha256": implementation_sha256,
                "hard_negative_index_fingerprint": hard_fingerprint,
                "cohort_patient_ids": sorted({r["patient_id"] for r in manifest["records"]}),
                "cohort_source_identities": sorted({r["source_identity"] for r in manifest["records"]})}
    if not resume:
        # Epoch-boundary replay policy: epoch zero is recoverable even before one epoch finishes.
        atomic_checkpoint(run / "checkpoints" / "last.pt", make_checkpoint(0))
    try:
        for epoch in range(start_epoch, args.max_epochs):
            started = time.perf_counter()
            dataset = PatchDataset(train_records, base, stats, patch, config.channels,
                                   args.steps_per_epoch * args.batch_size, args.seed, epoch,
                                   args.foreground_probability, not args.no_augmentation, 2, args.metadata_control,
                                   getattr(args, "anatomical_background_probability", 0.),
                                   getattr(args, "hard_negative_probability", 0.), hard_negatives)
            loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                              "pin_memory": device.type == "cuda", "shuffle": False, "drop_last": False,
                              "worker_init_fn": seed_worker,
                              "generator": torch.Generator().manual_seed(args.seed + epoch)}
            if args.workers > 0:
                loader_options.update({"prefetch_factor": 2, "persistent_workers": False})
            loader = DataLoader(dataset, **loader_options)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums = Counter()
            sampling_counts, positive_patches, patch_foreground_sum, sampling_fallbacks = Counter(), 0, 0., 0
            gradient_norms, skipped_updates = [], 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.max_epochs}", unit="batch")
            for batch_index, batch in enumerate(progress):
                sampling_counts.update(batch["sampling_kind"])
                positive_patches += int(batch["contains_foreground"].sum().item())
                patch_foreground_sum += float(batch["foreground_fraction"].sum().item())
                sampling_fallbacks += int(batch["sampling_fallback"].sum().item())
                image = batch["image"].to(device, non_blocking=device.type == "cuda")
                target = batch["mask"].to(device, non_blocking=device.type == "cuda")
                band = batch["boundary"].to(device, non_blocking=device.type == "cuda")
                q = batch["q"].to(device, non_blocking=device.type == "cuda")
                group_start = (batch_index // args.accumulation) * args.accumulation
                group_size = min(args.accumulation, len(loader) - group_start)
                with autocast_context(device, precision):
                    output = model(image, q, True)
                    loss, parts = total_loss(output, target, band, args.lambda_ds, args.lambda_boundary)
                scaler.scale(loss / group_size).backward()
                for key, value in parts.items():
                    sums[key] += value
                if (batch_index + 1) % args.accumulation == 0 or batch_index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    if not torch.isfinite(norm):
                        raise UserError("Nonfinite gradient norm; epoch-boundary checkpoint retained.")
                    gradient_norms.append(float(norm.detach().cpu()))
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                        global_step += 1
                    else:
                        skipped_updates += 1
                    optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(loss=f"{parts['loss']:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2g}", step=global_step)
            dataset.close()
            del loader, dataset, progress
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
            row = {"epoch": epoch + 1, "global_step": global_step,
                   **{k: sums[k] / args.steps_per_epoch for k in
                      ("loss", "dice_loss", "bce_loss", "aux1_loss", "aux2_loss", "boundary_loss")},
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else None,
                   "peak_cuda_gib": peak, "epoch_seconds": epoch_seconds,
                   "training_patches_per_second": args.steps_per_epoch * args.batch_size / epoch_seconds,
                   "actual_foreground_patch_fraction": positive_patches / (args.steps_per_epoch * args.batch_size),
                   "mean_patch_foreground_fraction": patch_foreground_sum / (args.steps_per_epoch * args.batch_size),
                   "sampling_counts": dict(sampling_counts), "sampling_fallbacks": sampling_fallbacks,
                   "skipped_optimizer_updates": skipped_updates, "validation_dice": None,
                   "validation_hd95_mm_finite_only": None, "validation_surface_dice": None,
                   "validation_seconds": None, "validation_failures": None, "validation_complete_misses": None,
                   "best_validation_dice": best_metric if math.isfinite(best_metric) else None}
            improved = False
            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.max_epochs:
                val_started = time.perf_counter()
                LOGGER.info("Full-volume validation (%d cases)", len(validation_records))
                val_rows = evaluate_records(model, validation_records, base, stats, patch, args.overlap,
                    args.threshold, args.surface_tolerance_mm, device, precision, output=None,
                    metadata_control=args.metadata_control, accumulator_dir=args.accumulator_dir)
                summary = bootstrap_summary(val_rows, args.seed, repeats=200)
                validation_dir = run / "validation" / f"epoch_{epoch + 1:04d}"
                atomic_json(validation_dir / "patient_metrics.json", val_rows)
                write_csv(validation_dir / "patient_metrics.csv", val_rows)
                atomic_json(validation_dir / "aggregates.json", summary)
                row["validation_seconds"] = time.perf_counter() - val_started
                row["validation_failures"] = summary["technical_failures"]
                row["validation_complete_misses"] = summary["complete_miss_cases"]
                row["validation_zero_overlap_cases"] = summary.get("zero_overlap_cases")
                for destination, metric in (("validation_dice", "dice"),
                    ("validation_hd95_mm_finite_only", "hd95_mm"), ("validation_surface_dice", "surface_dice")):
                    row[destination] = summary["metrics"].get(metric, {}).get("mean")
                # Failed patients cannot be silently filtered to win checkpoint selection.
                if summary["technical_failures"]:
                    atomic_json(run / "validation_failure.json", {"epoch": epoch + 1, "rows": val_rows})
                    raise UserError("Validation had technical failures. Last complete epoch retained; fix inputs before resuming.")
                score = row["validation_dice"]
                if score is None:
                    raise UserError("Validation yielded no valid patient Dice.")
                improved = score > best_metric
                if improved:
                    best_metric, stale_checks = score, 0
                else:
                    stale_checks += 1
            row["best_validation_dice"] = best_metric if math.isfinite(best_metric) else None
            history.append(row)
            checkpoint = make_checkpoint(epoch + 1)
            if improved:
                atomic_checkpoint(run / "checkpoints" / "best.pt", checkpoint)
            atomic_checkpoint(run / "checkpoints" / "last.pt", checkpoint)
            write_csv(run / "history.csv", history)
            atomic_json(run / "history.json", history)
            atomic_json(run / "sampling" / f"epoch_{epoch + 1:04d}.json", {
                "counts": dict(sampling_counts), "fallbacks": sampling_fallbacks,
                "actual_foreground_patch_fraction": row["actual_foreground_patch_fraction"],
                "mean_patch_foreground_fraction": row["mean_patch_foreground_fraction"],
                "scope": "training patches after augmentation, not full-volume performance"})
            LOGGER.info("Epoch %d complete: loss %.4f, validation Dice %s", epoch + 1, row["loss"], row["validation_dice"])
            if args.early_stopping_checks > 0 and stale_checks >= args.early_stopping_checks:
                LOGGER.info("Early stop after %d non-improving validation checks.", stale_checks)
                break
    except KeyboardInterrupt:
        atomic_json(run / "interruption.json", {"time": now(), "policy": "replay unfinished epoch",
                                               "resume": str(run / "checkpoints" / "last.pt")})
        LOGGER.warning("Interrupted. Resume last.pt; the unfinished epoch will replay from its boundary.")
        return 130
    except torch.cuda.OutOfMemoryError as exc:
        atomic_json(run / "failure.json", {"time": now(), "reason": "CUDA OOM", "detail": str(exc),
                                         "action": "new run with smaller patch/batch; no automatic architecture change"})
        raise UserError("CUDA OOM. Last complete epoch is safe; use a smaller NEW-run configuration.") from exc
    except (UserError, RuntimeError, OSError, ValueError) as exc:
        atomic_json(run / "failure.json", {"time": now(), "reason": str(exc),
                                         "resume_policy": "last complete epoch; unfinished epoch replays"})
        raise
    report_inputs = [run]
    if args.overfit_cases:
        # Real training-case memorization audit, including original-grid and
        # axial/coronal/sagittal overlays. Never substitute validation patients.
        diagnostic_model, diagnostic_checkpoint = load_inference_model(run / "checkpoints" / "best.pt", device)
        diagnostic_out = run / "diagnostic_evaluation"
        if diagnostic_out.exists():
            diagnostic_out = run / ("diagnostic_evaluation_" + uuid.uuid4().hex[:8])
        diagnostic_out.mkdir(parents=True)
        atomic_json(diagnostic_out / "protocol.json", {
            **protocol, "partition": "train-diagnostic", "case_ids": [r["case_id"] for r in train_records],
            "checkpoint_sha256": file_digest(run / "checkpoints" / "best.pt"),
            "scope": "memorization of training patients, not generalization", "created": now()})
        diagnostic_rows = evaluate_records(diagnostic_model, train_records, base, diagnostic_checkpoint["acquisition_stats"],
            patch, args.overlap, args.threshold, args.surface_tolerance_mm, device, precision,
            output=diagnostic_out, metadata_control=args.metadata_control, figure_cases=len(train_records))
        if any(r["status"] != "success" for r in diagnostic_rows):
            atomic_json(run / "diagnostic_failure.json", {"rows": diagnostic_rows, "output": str(diagnostic_out)})
            raise UserError("Final training-case diagnostic export failed; inspect diagnostic_failure.json.")
        report_build(diagnostic_out, [diagnostic_out], len(train_records), args.dpi, args.vector,
                     prepared_manifest=args.manifest)
        LOGGER.info("Training-case diagnostic overlays/report: %s", diagnostic_out / "report.html")
        del diagnostic_model
    report_build(run, report_inputs, args.figure_cases, args.dpi, args.vector, prepared_manifest=args.manifest)
    print(f"Training finished. Best checkpoint: {run / 'checkpoints' / 'best.pt'}")
    return 0


# 09. White-background figures and genuine qualitative evidence ---------------
PALETTE = {"reference": "#0072B2", "prediction": "#D55E00", "false_positive": "#CC79A7",
           "false_negative": "#009E73", "blue": "#0072B2", "orange": "#E69F00"}


def plotting():
    mpl = require("matplotlib")
    mpl.use("Agg", force=True)
    mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white", "text.color": "#17212b",
        "axes.labelcolor": "#17212b", "xtick.color": "#17212b", "ytick.color": "#17212b",
        "axes.edgecolor": "#6b7280", "font.family": "DejaVu Sans", "font.size": 10,
        "axes.grid": False, "figure.dpi": 110, "savefig.transparent": False,
        "svg.fonttype": "none", "pdf.fonttype": 42})
    return require("matplotlib.pyplot")


def save_figure(fig, path, dpi=300, vector="none"):
    plt = plotting()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    fig.tight_layout(pad=1.5)
    files = []
    try:
        fig.savefig(path, dpi=dpi, facecolor="white", transparent=False, bbox_inches="tight")
        files.append(path)
        for suffix in (("svg", "pdf") if vector == "both" else (vector,) if vector != "none" else ()):
            dest = path.with_suffix("." + suffix)
            fig.savefig(dest, facecolor="white", transparent=False, bbox_inches="tight")
            files.append(dest)
    finally:
        plt.close(fig)
    return files


def display_plane(array, axis, index):
    return np.take(array, int(index), axis=axis)


def plane_aspect(spacing, axis):
    remaining = [v for i, v in enumerate(spacing) if i != axis]
    return float(remaining[0] / remaining[1])


def contour_if_present(ax, plane, color, label=None):
    if np.any(plane) and not np.all(plane):
        ax.contour(plane.astype(float), levels=[.5], colors=[color], linewidths=1.)
    if label:
        from matplotlib.lines import Line2D
        return Line2D([0], [0], color=color, lw=1.5, label=label)
    return None


def visual_center(reference, probability=None, shape=None):
    if reference is not None and np.any(reference):
        return [int(np.median(c)) for c in np.nonzero(reference)]
    if probability is not None and np.any(probability >= .5):
        return list(np.unravel_index(int(np.argmax(probability)), probability.shape))
    return [int(n // 2) for n in (shape or probability.shape)]


def qc_panels(hu, mask, path, spacing, title):
    plt = plotting()
    center = visual_center(mask, shape=hu.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for axis, ax in enumerate(axes):
        plane = display_plane(hu, axis, center[axis])
        ax.imshow(plane, cmap="gray", vmin=-1000, vmax=400, aspect=plane_aspect(spacing, axis))
        if mask is not None:
            contour_if_present(ax, display_plane(mask, axis, center[axis]), PALETTE["reference"])
        ax.set_title(f"{'Axial' if axis == 0 else 'Coronal' if axis == 1 else 'Sagittal'} array plane\nindex {center[axis]}")
        ax.set_axis_off()
    fig.suptitle(title + "\nBlue: reference; displayed plane names assume canonical working axes", fontsize=11)
    return save_figure(fig, path)


def save_case_panels(hu, reference, probability, threshold, spacing, out, case_id, dpi=300,
                     prediction_mask=None):
    plt = plotting()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(prediction_mask, dtype=bool) if prediction_mask is not None else probability >= threshold
    if pred.shape != hu.shape or probability.shape != hu.shape:
        raise UserError("Qualitative CT, probability, and prediction grids disagree.")
    center = visual_center(reference, probability)
    index = center[0]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    panels = [(hu[index], "CT lung window", "gray", -1000, 400),
              (reference[index] if reference is not None else np.zeros_like(pred[index]),
               "Reference" if reference is not None else "Reference unavailable", "Blues", 0, 1),
              (pred[index], f"Prediction (threshold {threshold:g})", "Oranges", 0, 1)]
    for ax, (data, title, cmap, low, high) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=low, vmax=high, aspect=plane_aspect(spacing, 0))
        ax.set_title(title)
        ax.set_axis_off()
    ax = axes[3]
    ax.imshow(hu[index], cmap="gray", vmin=-1000, vmax=400, aspect=plane_aspect(spacing, 0))
    handles = [contour_if_present(ax, pred[index], PALETTE["prediction"], "Prediction")]
    if reference is not None:
        handles.append(contour_if_present(ax, reference[index], PALETTE["reference"], "Reference"))
    ax.legend(handles=[h for h in handles if h is not None], loc="upper right", fontsize=8)
    ax.set_title("Contours on CT")
    ax.set_axis_off()
    fig.suptitle(f"{case_id}: axial z={index}; slice chosen for visualization only", fontsize=12)
    save_figure(fig, out / "segmentation.png", dpi)
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    for axis in range(3):
        image_plane = display_plane(hu, axis, center[axis])
        prob_plane = display_plane(probability, axis, center[axis])
        for row in range(2):
            ax = axes[row, axis]
            if row == 0:
                im = ax.imshow(prob_plane, cmap="viridis", vmin=0, vmax=1, aspect=plane_aspect(spacing, axis))
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Predicted membership")
            else:
                ax.imshow(image_plane, cmap="gray", vmin=-1000, vmax=400, aspect=plane_aspect(spacing, axis))
            contour_if_present(ax, display_plane(pred, axis, center[axis]), PALETTE["prediction"])
            if reference is not None:
                contour_if_present(ax, display_plane(reference, axis, center[axis]), PALETTE["reference"])
            ax.set_title(f"{'Axial Coronal Sagittal'.split()[axis]}, index={center[axis]}")
            ax.set_axis_off()
    fig.suptitle(f"{case_id}: probabilities and contours; blue reference, orange prediction")
    save_figure(fig, out / "tri_plane.png", dpi)
    if reference is not None:
        ref = np.asarray(reference, dtype=bool)
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        from matplotlib.patches import Patch
        for axis, ax in enumerate(axes):
            ref_plane = display_plane(ref, axis, center[axis])
            pred_plane = display_plane(pred, axis, center[axis])
            image_plane = display_plane(hu, axis, center[axis])
            ax.imshow(image_plane, cmap="gray", vmin=-1000, vmax=400, aspect=plane_aspect(spacing, axis))
            rgba = np.zeros((*ref_plane.shape, 4), np.float32)
            colors = require("matplotlib.colors")
            rgba[pred_plane & ~ref_plane] = colors.to_rgba(PALETTE["false_positive"], .8)
            rgba[~pred_plane & ref_plane] = colors.to_rgba(PALETTE["false_negative"], .8)
            ax.imshow(rgba, aspect=plane_aspect(spacing, axis))
            ax.set_title("Axial Coronal Sagittal".split()[axis])
            ax.set_axis_off()
        axes[-1].legend(handles=[Patch(color=PALETTE["false_positive"], label="False positive"),
                                Patch(color=PALETTE["false_negative"], label="False negative")], loc="upper right")
        fig.suptitle(f"{case_id}: errors relative to annotated primary target")
        save_figure(fig, out / "errors.png", dpi)
        occupied = np.flatnonzero(np.any(ref, axis=(1, 2)))
        if len(occupied):
            chosen = sorted(set([int(occupied[0]), int(occupied[min(1, len(occupied) - 1)]),
                                 int(occupied[max(0, len(occupied) - 2)]), int(occupied[-1])]))
            fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 5), squeeze=False)
            for ax, z in zip(axes[0], chosen):
                ax.imshow(hu[z], cmap="gray", vmin=-1000, vmax=400, aspect=plane_aspect(spacing, 0))
                contour_if_present(ax, ref[z], PALETTE["reference"])
                contour_if_present(ax, pred[z], PALETTE["prediction"])
                ax.set_title(f"Target extent: z={z}")
                ax.set_axis_off()
            fig.suptitle(f"{case_id}: target end planes on working grid (blue reference, orange prediction)")
            save_figure(fig, out / "extent.png", dpi)


def surface_figure(mask_image, reference, destination, dpi=300):
    """Marching cubes from REAL masks; vertices mapped from z,y,x to physical LPS."""
    measure = require("skimage.measure")
    sitk, plt = require("SimpleITK"), plotting()
    grid = grid_dict(mask_image)
    direction = np.asarray(grid["direction_xyz"]).reshape(3, 3)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    combined = []
    for array, color, name in ((sitk.GetArrayFromImage(mask_image), PALETTE["prediction"], "Prediction"),
                                (reference, PALETTE["reference"], "Reference")):
        if array is None or not np.any(array):
            continue
        # Coarse display only; metric masks remain untouched.
        step = max(1, int(math.ceil(max(array.shape) / 160)))
        sampled = np.asarray(array)[::step, ::step, ::step]
        padded = np.pad(sampled, 1)
        vertices, faces, _, _ = measure.marching_cubes(padded.astype(np.float32), level=.5)
        index_xyz = ((vertices - 1) * step)[:, ::-1]
        physical = (direction @ (index_xyz * np.asarray(grid["spacing_xyz_mm"])).T).T + grid["origin_xyz_mm"]
        ax.plot_trisurf(physical[:, 0], physical[:, 1], physical[:, 2], triangles=faces,
                        color=color, alpha=.4, linewidth=0, antialiased=True)
        combined.append(physical)
    if not combined:
        plt.close(fig)
        raise UserError("Both masks empty: there is no tumor surface to display.")
    points = np.concatenate(combined)
    ax.set_box_aspect(np.maximum(points.max(0) - points.min(0), 1))
    ax.set(xlabel="LPS x (mm)", ylabel="LPS y (mm)", zlabel="LPS z (mm)",
           title="Physical mask surfaces: blue reference, orange prediction")
    return save_figure(fig, destination, dpi)


FIGURE_TITLES = [
    "Cohort inclusion and exclusion", "Patient counts by partition", "Native through-plane spacing",
    "Native in-plane spacing", "Native and working dimensions", "Reference tumor volumes",
    "Foreground fractions", "QC reasons", "Native and working-grid QC", "Three-plane reference overlays",
    "Training objective", "Dice and BCE training losses", "Auxiliary and boundary training losses",
    "Full-volume validation Dice", "Full-volume surface metrics", "Learning-rate schedule",
    "Gradient norms", "GPU memory", "Training and validation timing", "Checkpoint selection progression",
    "Patient Dice and ranks", "IoU, precision and recall", "Surface metrics and undefined counts",
    "Dice versus target volume", "Boundary error versus target volume", "Performance versus native spacing",
    "Predicted versus reference volume", "Volume Bland-Altman", "Relative volume errors",
    "Complete misses and volume subgroups", "Paired model differences", "Compatible model metric/cost comparison",
    "CT/reference/prediction overlays", "Three-plane probabilities and contours", "False-positive/negative maps",
    "Best/median/worst ranked cases", "Target-extent examples", "Complete-miss examples",
    "Physical 3D mask surfaces", "Wavelet feature-band slices", "Wavelet band energies",
    "Directional gate distributions", "Spatial directional gate maps", "Fusion source/channel weights",
    "Gates versus native acquisition spacing", "Restricted bottleneck attention probe",
    "Patient-bootstrap confidence intervals", "Calibration and Brier/ECE", "Weighted voxel ROC/PR",
    "Entropy-ranked risk/coverage", "Actual multi-checkpoint disagreement", "Patient runtime and memory",
    "Completed ablation accuracy/cost", "Recorded synthetic acquisition stress tests"]


def float_value(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def recorded_metrics(root):
    path = root / "patient_metrics.json"
    scope = "recorded complete-volume evaluation"
    if not path.exists():
        candidates = sorted((root / "validation").glob("epoch_*/patient_metrics.json"))
        path = candidates[-1] if candidates else path
        scope = "latest complete-volume development validation; not necessarily the best checkpoint"
    return (load_json(path) if path.exists() else []), path, scope


def compatible_comparisons(roots):
    evaluations = []
    for root in roots:
        protocol_path = root / "protocol.json"
        if not protocol_path.exists():
            continue
        rows, source, scope = recorded_metrics(root)
        if not rows or "latest" in scope:
            continue  # final comparisons require explicit evaluation, not unmatched training epochs
        protocol = load_json(protocol_path)
        evaluations.append((root, protocol, rows))
    if len(evaluations) < 2:
        return [], "Need two completed explicit evaluation outputs, not two labels for one run."
    keys = ("manifest_fingerprint", "split_fingerprint", "partition", "target",
            "surface_tolerance_mm", "surface_convention", "overlap")
    baseline = evaluations[0]
    base_cases = {(r["case_id"], r["patient_id"]) for r in baseline[2]}
    policies = []
    for root, protocol, rows in evaluations:
        if any(protocol.get(k) != baseline[1].get(k) for k in keys):
            return [], f"Incompatible evaluation protocol in {root.name}; no pooled comparison."
        identities = [(r["case_id"], r["patient_id"]) for r in rows]
        if len(identities) != len(set(identities)):
            return [], "Comparison contains duplicate case identities."
        if set(identities) != base_cases or any(r["status"] != "success" for r in rows):
            return [], "Comparison patient/case sets differ or include technical failures."
        if "case_ids" in protocol and set(protocol["case_ids"]) != {r["case_id"] for r in rows}:
            return [], "Comparison lacks one or more expected evaluation cases."
        if any(r.get("dice") is None or not math.isfinite(float(r["dice"])) for r in rows):
            return [], "Comparison requires labeled case metrics."
        source = protocol.get("threshold_source", "")
        if source.startswith("frozen validation selection: "):
            point_path = root / "operating_point.json"
            if not point_path.exists():
                return [], "Validation-frozen comparison requires the saved operating_point.json."
            point = load_json(point_path)
            fingerprint = stable_hash({k: v for k, v in point.items() if k != "fingerprint"})
            if (point.get("fingerprint") != fingerprint or point.get("source_partition") != "val"
                    or source != "frozen validation selection: " + fingerprint
                    or any(point.get(k) != protocol.get(k) for k in
                           ("checkpoint_sha256", "target", "manifest_fingerprint", "split_fingerprint", "threshold"))
                    or postprocessing_recipe(point.get("postprocessing", "none")) !=
                       postprocessing_recipe(protocol.get("postprocessing", "none"))):
                return [], "Comparison operating point is not a compatible frozen validation selection."
            policies.append("validation-frozen")
        elif source == "prespecified before training, not test-tuned":
            policies.append("common-prespecified")
        else:
            return [], "Comparison requires a documented prespecified or validation-frozen operating policy."
    if len(set(policies)) != 1:
        return [], "Do not mix prespecified and validation-selected operating policies in one paired comparison."
    if policies[0] == "common-prespecified":
        for _, protocol, _ in evaluations[1:]:
            if (protocol.get("threshold") != baseline[1].get("threshold") or
                    postprocessing_recipe(protocol.get("postprocessing", "none")) !=
                    postprocessing_recipe(baseline[1].get("postprocessing", "none"))):
                return [], "Prespecified comparisons require a common threshold and postprocessing recipe."
    return evaluations, ""


def load_voxel_samples(root, cap=200000):
    paths = sorted((root / "samples").glob("*.npz"))
    probabilities, labels, weights, case_scores = [], [], [], {}
    per_case = max(1, cap // max(1, len(paths)))
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            p, y = data["probability"], data["reference"]
            population = int(data["population_voxels"])
            rng = np.random.default_rng(int(stable_hash(path.stem)[:8], 16))
            selected = rng.choice(len(p), min(per_case, len(p)), replace=False)
            p, y = p[selected], y[selected]
            if not len(p):
                continue
            probabilities.append(p)
            labels.append(y)
            weights.append(np.full(len(p), population / len(p), dtype=np.float64))
            clipped = np.clip(p, 1e-7, 1 - 1e-7)
            case_scores[path.stem] = float(np.mean(-clipped * np.log(clipped) - (1 - clipped) * np.log(1 - clipped)))
    if not probabilities:
        return None
    return np.concatenate(probabilities), np.concatenate(labels).astype(bool), np.concatenate(weights), case_scores


def calibration_values(probability, label, weight, bins=10):
    ids = np.minimum((probability * bins).astype(int), bins - 1)
    output = []
    for i in range(bins):
        selected = ids == i
        if not selected.any():
            continue
        mass = float(weight[selected].sum())
        output.append({"bin": i, "weight": mass, "predicted": float(np.average(probability[selected], weights=weight[selected])),
                       "observed": float(np.average(label[selected], weights=weight[selected]))})
    brier = float(np.average((probability - label) ** 2, weights=weight))
    ece = sum(r["weight"] * abs(r["predicted"] - r["observed"]) for r in output) / weight.sum()
    return output, brier, float(ece)


def weighted_roc_pr(probability, labels, weights):
    if not labels.any() or labels.all():
        raise UserError("Both reference classes are required for ROC/PR.")
    order = np.argsort(-probability, kind="stable")
    p, y, w = probability[order], labels[order], weights[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(w * y)[ends]
    fp = np.cumsum(w * ~y)[ends]
    recall = tp / np.sum(w * y)
    fpr = fp / np.sum(w * ~y)
    precision = tp / np.maximum(tp + fp, 1e-20)
    auc = float(np.trapezoid(np.r_[0, recall], np.r_[0, fpr])) if hasattr(np, "trapezoid") else float(np.trapz(np.r_[0, recall], np.r_[0, fpr]))
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    return np.r_[0, fpr], np.r_[0, recall], np.r_[1, precision], auc, ap


def report_build(output, roots, case_limit=6, dpi=300, vector="none", prepared_manifest=None,
                 split_path=None, include_surfaces=False):
    """Render only recorded evidence; each of the 54 families gets an explicit status."""
    plt = plotting()
    output, roots = Path(output).resolve(), [Path(r).resolve() for r in roots]
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    primary = roots[0]
    protocol = load_json(primary / "protocol.json") if (primary / "protocol.json").exists() else {}
    metrics, metric_source, metric_scope = recorded_metrics(primary)
    success = [r for r in metrics if r.get("status", "success") == "success" and "dice" in r]
    history = (load_json(primary / "history.json") if (primary / "history.json").exists()
               else read_csv(primary / "history.csv"))
    statuses = [{"figure_id": i + 1, "title": name, "status": "skipped",
                 "reason": "Required recorded artifacts are unavailable.", "source_artifacts": [], "files": []}
                for i, name in enumerate(FIGURE_TITLES)]
    def skip(number, reason, sources=()):
        statuses[number - 1].update(reason=reason, source_artifacts=[str(s) for s in sources])
    def make(number, draw, sources=(), caption=""):
        record = statuses[number - 1]
        record["source_artifacts"] = [str(s) for s in sources]
        try:
            fig = draw()
            files = save_figure(fig, figures / f"{number:02d}.png", dpi, vector)
            record.update(status="created", reason="", files=[os.path.relpath(p, output) for p in files], caption=caption)
        except (UserError, ValueError, OSError, RuntimeError, ImportError) as exc:
            record.update(status="failed", reason=str(exc))
            LOGGER.error("Figure %d failed: %s", number, exc)
    def existing(number, paths, caption=""):
        paths = [Path(p) for p in paths if Path(p).is_file()]
        if paths:
            statuses[number - 1].update(status="created", reason="", caption=caption,
                source_artifacts=[str(p) for p in paths], files=[os.path.relpath(p, output) for p in paths])
    def bars(names, values, title, ylabel="Cases"):
        fig, ax = plt.subplots(figsize=(max(7, min(13, len(names) * 1.2)), 5))
        ax.bar(np.arange(len(names)), values, color=PALETTE["blue"])
        ax.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
        return fig
    def hist(values, title, xlabel):
        values = np.asarray([v for v in values if v is not None], dtype=float)
        if not len(values):
            raise UserError("No finite observations for this plot.")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=min(25, max(1, int(math.sqrt(len(values))))), color=PALETTE["blue"], edgecolor="white")
        ax.set(title=f"{title} (n={len(values)})", xlabel=xlabel, ylabel="Cases")
        ax.grid(axis="y", alpha=.2)
        return fig
    def scatter(xkey, ykey, title, xlabel, ylabel, logx=False):
        valid = [(float_value(r.get(xkey)), float_value(r.get(ykey))) for r in success]
        valid = [(x, y) for x, y in valid if x is not None and y is not None and (not logx or x > 0)]
        if not valid:
            raise UserError("No compatible finite paired observations.")
        x, y = np.asarray(valid).T
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, s=30, alpha=.8, color=PALETTE["blue"])
        ax.set(title=f"{title} (n={len(x)})", xlabel=xlabel, ylabel=ylabel)
        if logx:
            ax.set_xscale("log")
        ax.grid(alpha=.2)
        return fig
    def history_plot(keys, title, ylabel):
        fig, ax = plt.subplots(figsize=(9, 5))
        lines = 0
        for key, label in keys:
            pairs = [(float_value(r.get("epoch")), float_value(r.get(key))) for r in history]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if pairs:
                x, y = np.asarray(pairs).T
                ax.plot(x, y, marker="o" if len(x) < 40 else None, label=label)
                lines += 1
        if not lines:
            plt.close(fig)
            raise UserError("No observations for the requested history fields.")
        ax.set(title=title, xlabel="Completed epoch", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend()
        return fig

    manifest, prepared_base, records = None, None, []
    manifest_file = Path(prepared_manifest or protocol.get("manifest", "__missing__"))
    if manifest_file.is_file():
        try:
            manifest, prepared_base = load_manifest(manifest_file)
            records = manifest["records"]
        except (UserError, ValueError, OSError) as exc:
            LOGGER.error("Report cannot load preparation data: %s", exc)
    qc = load_json(prepared_base / "qc.json") if prepared_base and (prepared_base / "qc.json").exists() else []
    if records:
        counts = Counter(r["status"] for r in qc) if qc else Counter(included=len(records))
        make(1, lambda: bars(list(counts), list(counts.values()), "Preparation cohort; consult exclusion reasons"), [manifest_file])
        make(3, lambda: hist([r["native_spacing_zyx_mm"][0] for r in records], "Native center-to-center slice spacing", "z spacing (mm)"), [manifest_file])
        def inplane():
            fig, ax = plt.subplots(figsize=(8, 5))
            for axis, name in ((1, "y"), (2, "x")):
                ax.hist([r["native_spacing_zyx_mm"][axis] for r in records], bins=15, alpha=.6, label=name)
            ax.set(title="Native in-plane spacings", xlabel="Spacing (mm)", ylabel="Cases")
            ax.legend()
            return fig
        make(4, inplane, [manifest_file])
        def dimensions():
            fig, axes = plt.subplots(1, 3, figsize=(13, 5))
            for axis, ax in enumerate(axes):
                for key, name in (("native_shape_zyx", "Native"), ("working_shape_zyx", "Working")):
                    ax.hist([r[key][axis] for r in records], bins=15, alpha=.6, label=name)
                ax.set(title="zyx"[axis] + " dimension", xlabel="Voxels", ylabel="Cases")
                ax.legend()
            return fig
        make(5, dimensions, [manifest_file])
        if any(r["reference_volume_ml"] is not None for r in records):
            make(6, lambda: hist([r["reference_volume_ml"] for r in records], "Annotated primary-target volume", "Native reference volume (mL)"), [manifest_file])
            make(7, lambda: hist([r["foreground_fraction"] for r in records], "Working-grid foreground fraction", "Fraction of volume voxels"), [manifest_file])
        if qc:
            reasons = Counter((r.get("reason") or "included") for r in qc)
            make(8, lambda: bars([s[:90] for s in reasons], list(reasons.values()), "Recorded QC outcomes"), [prepared_base / "qc.json"])
        qc_files = sorted((prepared_base / "qc").glob("*.png"))
        existing(9, qc_files[:case_limit * 2], "Native and working grids; each uses its own physical aspect ratio.")
        existing(10, sorted((prepared_base / "qc").glob("*-working.png"))[:case_limit], "Blue reference contours; inspection is required before training.")
    split_file = Path(split_path) if split_path else primary / "splits.json"
    if split_file.exists():
        partitions = load_json(split_file)["partitions"]
        by_id = {r["case_id"]: r for r in records}
        counts = {k: len({by_id[c]["patient_id"] for c in v}) if by_id and all(c in by_id for c in v) else len(v)
                  for k, v in partitions.items()}
        make(2, lambda: bars(list(counts), list(counts.values()), "Partition sizes", "Patients (cases if manifest unavailable)"), [split_file])
    if history:
        history_file = primary / "history.csv"
        history_specs = {
            11: ([("loss", "Patch training objective")], "Patch training objective", "Loss"),
            12: ([("dice_loss", "Soft foreground Dice loss"), ("bce_loss", "BCE")], "Patch loss components", "Loss"),
            14: ([("validation_dice", "Complete-volume validation Dice")], "Validation overlap; no patch/volume equivalence implied", "Dice"),
            15: ([("validation_hd95_mm_finite_only", "HD95, finite cases only")], "Validation HD95; consult miss counts", "Millimeters"),
            16: ([("learning_rate", "Learning rate")], "Optimizer learning rate at epoch end", "Learning rate"),
            17: ([("gradient_norm", "Mean pre-clipping norm")], "Gradient norms at optimizer boundaries", "L2 norm"),
            19: ([("epoch_seconds", "Training epoch"), ("validation_seconds", "Complete-volume validation")], "Recorded elapsed time", "Seconds"),
            20: ([("validation_dice", "Validation Dice"), ("best_validation_dice", "Best so far")], "Checkpoint criterion progression", "Complete-volume Dice")}
        for number, (keys, title, ylabel) in history_specs.items():
            if any(float_value(r.get(k)) is not None for r in history for k, _ in keys):
                make(number, lambda keys=keys, title=title, ylabel=ylabel: history_plot(keys, title, ylabel), [history_file])
        if protocol.get("model_config", {}).get("boundary") or protocol.get("model_config", {}).get("deep_supervision"):
            make(13, lambda: history_plot([("boundary_loss", "Boundary BCE"), ("aux1_loss", "D1 Dice+BCE"),
                                           ("aux2_loss", "D2 Dice+BCE")], "Unweighted auxiliary training losses", "Loss"), [history_file])
        else:
            skip(13, "Boundary and deep supervision were disabled; zero placeholders are not plotted.")
        if any(float_value(r.get("peak_cuda_gib")) is not None for r in history):
            make(18, lambda: history_plot([("peak_cuda_gib", "Training peak allocated")], "Measured CUDA allocation", "GiB"), [history_file])
        else:
            skip(18, "No measured CUDA memory; CPU runs do not have a CUDA allocation.")
    if success:
        def dice_rank():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            values = np.asarray([r["dice"] for r in success])
            axes[0].hist(values, bins=min(20, max(1, int(math.sqrt(len(values))))), color=PALETTE["blue"])
            axes[0].set(xlabel="Complete-volume Dice", ylabel="Cases", title="Overlap distribution")
            axes[1].plot(np.arange(1, len(values) + 1), np.sort(values), marker=".")
            axes[1].set(xlabel="Case rank", ylabel="Dice", title="Ranked case overlap")
            fig.suptitle(metric_scope)
            return fig
        make(21, dice_rank, [metric_source])
        def overlap_box():
            fig, ax = plt.subplots(figsize=(8, 5))
            keys = ("iou", "precision", "recall")
            ax.boxplot([[r[k] for r in success] for k in keys])
            ax.set_xticks([1, 2, 3], ["IoU", "Precision", "Recall"])
            ax.set(ylabel="Score", title="Complete-volume primary-target metrics")
            return fig
        make(22, overlap_box, [metric_source])
        def surface_plots():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            finite = [r["hd95_mm"] for r in success if r.get("hd95_mm") is not None]
            if finite:
                axes[0].hist(finite, bins=min(20, max(1, len(finite))), color=PALETTE["blue"])
            axes[0].set(xlabel="HD95 (mm)", ylabel="Cases", title=f"Finite n={len(finite)}; undefined/miss n={len(success)-len(finite)}")
            axes[1].hist([r["surface_dice"] for r in success if r.get("surface_dice") is not None], bins=10,
                         color=PALETTE["blue"])
            axes[1].set(xlabel="Voxel-center Surface Dice", ylabel="Cases",
                        title=f"Tolerance {success[0]['surface_tolerance_mm']:g} mm; misses retain zero")
            return fig
        make(23, surface_plots, [metric_source], "Unweighted boundary voxel centers; not surfel-area weighting. Undefined distances counted explicitly.")
        specs = {
            24: ("reference_volume_ml", "dice", "Overlap versus tumor size", "Reference volume (mL)", "Dice", True),
            25: ("reference_volume_ml", "hd95_mm", "Boundary error versus tumor size", "Reference volume (mL)", "Finite HD95 (mm)", True),
            26: ("native_spacing_z_mm", "dice", "Overlap versus native through-plane spacing", "Native z spacing (mm)", "Dice", False),
            27: ("reference_volume_ml", "predicted_volume_ml", "Predicted and reference volume", "Reference (mL)", "Prediction (mL)", False)}
        for number, values in specs.items():
            if any(float_value(r.get(values[0])) is not None and float_value(r.get(values[1])) is not None for r in success):
                make(number, lambda values=values: scatter(*values), [metric_source])
        def bland_altman():
            ref = np.asarray([r["reference_volume_ml"] for r in success])
            pred = np.asarray([r["predicted_volume_ml"] for r in success])
            mean, difference = (ref + pred) / 2, pred - ref
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(mean, difference, color=PALETTE["blue"])
            bias = float(difference.mean())
            ax.axhline(bias, color=PALETTE["orange"], label=f"Mean difference {bias:.3g} mL")
            if len(difference) > 1:
                sd = difference.std(ddof=1)
                for sign in (-1, 1):
                    ax.axhline(bias + sign * 1.96 * sd, ls="--", color="#777777")
            ax.set(xlabel="Mean reference/predicted volume (mL)", ylabel="Prediction - reference (mL)",
                   title="Volume Bland-Altman; descriptive limits assume suitable differences")
            ax.legend()
            return fig
        make(28, bland_altman, [metric_source])
        if any(r.get("relative_volume_error") is not None for r in success):
            make(29, lambda: hist([r.get("relative_volume_error") for r in success], "Signed relative volume error", "(Prediction - reference) / reference"), [metric_source])
        def subgroups():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            missed = sum(r["complete_miss"] for r in success)
            disjoint = sum(int(r.get("disjoint_nonempty_prediction",
                r.get("reference_volume_ml", 0) > 0 and r.get("predicted_volume_ml", 0) > 0 and r.get("dice") == 0))
                for r in success)
            axes[0].bar(["Empty target misses", "Nonempty zero-overlap", "Other evaluated cases", "Technical failures"],
                        [missed, disjoint, len(success) - missed - disjoint, len(metrics) - len(success)])
            axes[0].tick_params(axis="x", rotation=20)
            axes[0].set(ylabel="Cases", title="Failure and target-miss accounting")
            edges, labels = (0, 5, 20, math.inf), ("0–5", "5–20", "20+")
            groups = [[r["dice"] for r in success if lo <= r["reference_volume_ml"] < hi]
                      for lo, hi in zip(edges[:-1], edges[1:])]
            axes[1].bar(labels, [np.mean(g) if g else np.nan for g in groups])
            axes[1].set(xlabel="Prespecified reference volume band (mL)", ylabel="Mean Dice", title="Descriptive subgroups; not tuned cutoffs")
            for i, group in enumerate(groups):
                axes[1].text(i, .02, f"n={len(group)}", ha="center")
            return fig
        make(30, subgroups, [metric_source])
        summary = bootstrap_summary(metrics)
        atomic_json(output / "report_aggregates.json", summary)
        def intervals():
            keys = [k for k in ("dice", "iou", "precision", "recall", "surface_dice") if summary["metrics"][k]["n_patients"]]
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, key in enumerate(keys):
                item = summary["metrics"][key]
                mean, (lo, hi) = item["mean"], item["ci95"]
                ax.errorbar(mean, i, xerr=[[max(0, mean-lo)], [max(0, hi-mean)]], fmt="o", capsize=4)
            ax.set_yticks(range(len(keys)), keys)
            ax.set(xlabel="Mean and patient-bootstrap 95% interval", title="Patient-level overlap uncertainty; n reported in JSON")
            ax.grid(axis="x", alpha=.2)
            return fig
        make(47, intervals, [metric_source])
        if any(r.get("inference_seconds") is not None for r in success):
            def compute_plots():
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].hist([r["inference_seconds"] for r in success], bins=min(20, len(success)))
                axes[0].set(xlabel="Working-grid inference (s)", ylabel="Cases", title="Measured case runtime")
                memory = [r["peak_cuda_gib"] for r in success if r.get("peak_cuda_gib") is not None]
                if memory:
                    axes[1].hist(memory, bins=min(20, len(memory)))
                else:
                    axes[1].text(.5, .5, "CUDA memory unavailable", ha="center", transform=axes[1].transAxes)
                axes[1].set(xlabel="Peak CUDA allocated (GiB)", ylabel="Cases")
                return fig
            make(52, compute_plots, [metric_source])

    # Genuine paired comparisons, refusing mismatched protocols/cohorts.
    comparisons, comparison_reason = compatible_comparisons(roots)
    if comparisons:
        paired_effects = []
        def patient_dice(rows):
            grouped = {}
            for row in rows:
                grouped.setdefault(row["patient_id"], []).append(row["dice"])
            return {k: float(np.mean(v)) for k, v in grouped.items()}
        reference_scores = patient_dice(comparisons[0][2])
        for root, p, rows in comparisons[1:]:
            scores = patient_dice(rows)
            patients = sorted(reference_scores)
            differences = np.asarray([scores[x] - reference_scores[x] for x in patients])
            rng = np.random.default_rng(1234)
            ci = np.quantile([rng.choice(differences, len(differences), replace=True).mean() for _ in range(1000)], [.025, .975])
            paired_effects.append({"reference": comparisons[0][0].name, "comparison": root.name,
                                   "n_patients": len(patients), "mean_dice_difference": float(differences.mean()),
                                   "ci95": ci.tolist(), "differences": differences.tolist(),
                                   "operating_policy": "individually validation-frozen" if p.get("threshold_source", "").startswith("frozen validation selection: ") else "common prespecified",
                                   "reference_threshold": comparisons[0][1]["threshold"],
                                   "comparison_threshold": p["threshold"],
                                   "reference_postprocessing": comparisons[0][1].get("postprocessing", "none"),
                                   "comparison_postprocessing": p.get("postprocessing", "none")})
        atomic_json(output / "paired_comparisons.json", paired_effects)
        def effects_plot():
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, effect in enumerate(paired_effects):
                mean, (lo, hi) = effect["mean_dice_difference"], effect["ci95"]
                ax.errorbar(mean, i, xerr=[[max(0, mean-lo)], [max(0, hi-mean)]], fmt="o", capsize=5)
            ax.axvline(0, color="#777777", ls="--")
            ax.set_yticks(range(len(paired_effects)), [e["comparison"] for e in paired_effects])
            ax.set(xlabel="Paired patient Dice difference and 95% bootstrap interval",
                   title=f"Relative to {comparisons[0][0].name}")
            return fig
        make(31, effects_plot, [r / "patient_metrics.json" for r, _, _ in comparisons])
        def model_cost():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for root, p, rows in comparisons:
                mean = np.mean(list(patient_dice(rows).values()))
                runtime = np.mean([r["inference_seconds"] for r in rows])
                axes[0].scatter(p["parameters"] / 1e6, mean, label=root.name)
                axes[1].scatter(runtime, mean, label=root.name)
            axes[0].set(xlabel="Parameters (millions)", ylabel="Patient mean Dice")
            axes[1].set(xlabel="Mean working-grid case inference (s)", ylabel="Patient mean Dice")
            axes[1].legend(fontsize=8)
            fig.suptitle("Compatible completed runs; hardware/precision remain part of the comparison")
            return fig
        make(32, model_cost, [r / "protocol.json" for r, _, _ in comparisons])
        make(53, model_cost, [r / "protocol.json" for r, _, _ in comparisons], "Only actual completed experiments supplied to --inputs.")
    else:
        for number in (31, 32, 53):
            skip(number, comparison_reason)

    # Reuse original recorded qualitative panels; produce ranked selections only
    # when actual saved probability maps AND their prepared CT caches are present.
    case_folders = sorted((primary / "cases").glob("*"))
    for number, filename in ((33, "segmentation.png"), (34, "tri_plane.png"), (35, "errors.png"), (37, "extent.png")):
        existing(number, [p / filename for p in case_folders][:case_limit], "Reference-based slice selection is visualization only; predictions used full volumes.")
    ranked_paths = []
    if success and records and case_limit > 0:
        by_id = {r["case_id"]: r for r in records}
        ranked = sorted(success, key=lambda r: (r["dice"], r["case_id"]))
        selected = list(dict.fromkeys([ranked[0]["case_id"], ranked[len(ranked)//2]["case_id"], ranked[-1]["case_id"]]))
        for cid in selected[:case_limit]:
            folder = primary / "cases" / cid
            probability_path = folder / "probability.nii.gz"
            if cid not in by_id or not probability_path.exists():
                continue
            try:
                sitk = require("SimpleITK")
                directory, meta = cache_path(prepared_base, by_id[cid])
                hu = np.load(directory / "hu.npy", mmap_mode="r", allow_pickle=False)
                ref = np.load(directory / "mask.npy", mmap_mode="r", allow_pickle=False)
                try:
                    image = sitk.ReadImage(str(probability_path), sitk.sitkFloat32)
                    working = sitk.Resample(image, empty_grid(meta["working_grid"]), sitk.Transform(), sitk.sitkLinear, 0.)
                    native_pred = postprocess_prediction(sitk.GetArrayFromImage(image) >= protocol["threshold"],
                                                        meta["native_spacing_zyx_mm"], protocol.get("postprocessing", "none"))
                    pred_image = image_on_grid(native_pred.astype(np.uint8), meta["native_grid"])
                    working_pred = sitk.GetArrayFromImage(sitk.Resample(pred_image, empty_grid(meta["working_grid"]),
                        sitk.Transform(), sitk.sitkNearestNeighbor, 0., sitk.sitkUInt8)).astype(bool)
                    save_case_panels(hu, ref, sitk.GetArrayFromImage(working), protocol["threshold"],
                                     meta["target_spacing_zyx_mm"], folder, cid, dpi, prediction_mask=working_pred)
                    ranked_paths.append(folder / "segmentation.png")
                finally:
                    hu._mmap.close()
                    ref._mmap.close()
            except (UserError, OSError, RuntimeError, ValueError) as exc:
                LOGGER.warning("Ranked-case figure unavailable for %s: %s", cid, exc)
        existing(36, ranked_paths, "Deterministic worst / median / best by recorded complete-volume Dice; ties by case ID.")
        miss_paths = [primary / "cases" / r["case_id"] / "errors.png" for r in ranked
                      if r.get("zero_overlap", r.get("reference_volume_ml", 0) > 0 and r.get("dice") == 0)]
        existing(38, miss_paths[:case_limit], "Zero-overlap target failures, including nonempty disjoint predictions and empty target misses.")
        if not miss_paths:
            skip(38, "No zero-overlap failures with available panels; difficult-interface labels were not supplied.")
        if include_surfaces:
            for cid in selected[:1]:
                folder = primary / "cases" / cid
                if not (folder / "prediction.nii.gz").exists():
                    continue
                try:
                    sitk = require("SimpleITK")
                    directory, _ = cache_path(prepared_base, by_id[cid])
                    ref = np.load(directory / "native_mask.npy", mmap_mode="r", allow_pickle=False)
                    try:
                        surface_figure(sitk.ReadImage(str(folder / "prediction.nii.gz")), ref, folder / "surface.png", dpi)
                    finally:
                        ref._mmap.close()
                    existing(39, [folder / "surface.png"])
                except (UserError, OSError, RuntimeError, ValueError) as exc:
                    skip(39, str(exc))
        else:
            skip(39, "Optional real-mask surfaces require report --surfaces and scikit-image.")

    analysis_path = primary / "analysis.json"
    analyses = load_json(analysis_path) if analysis_path.exists() else []
    wavelets = [(a, name, m) for a in analyses for name, m in a["modules"].items() if name.startswith("skip") and isinstance(m, dict)]
    if wavelets:
        a, stage, band_data = wavelets[0]
        def band_slices():
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            for i, ax in enumerate(axes.flat):
                array = np.asarray(band_data["band_slices"][i])[0]
                vmax = max(float(np.abs(array).max()), 1e-8)
                im = ax.imshow(array, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.set_title(BANDS[i] + " / feature channel 0")
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=.046)
            fig.suptitle(f"{a['case_id']}, {stage}: learned feature subbands; grid {band_data['band_shape'][-3:]}")
            return fig
        make(40, band_slices, [analysis_path])
        def energies():
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(8)
            ax.bar(x-.18, band_data["before_energy"], .36, label="Before")
            ax.bar(x+.18, band_data["after_energy"], .36, label="After learned refinement")
            ax.set_xticks(x, BANDS)
            ax.set(ylabel="Mean squared feature coefficient", title=f"{stage}: actual selected-patch band energy")
            ax.legend()
            return fig
        make(41, energies, [analysis_path])
        def gate_summary():
            fig, ax = plt.subplots(figsize=(10, 5))
            quantiles = np.asarray(band_data["gate_quantiles"])
            mean = np.asarray(band_data["gate_mean"])
            # Quantile ranges are displayed as vertical bars independent of the mean;
            # mean need not lie inside the central 90% range for every distribution.
            ax.vlines(np.arange(7), quantiles[:, 0], quantiles[:, 2], color=PALETTE["blue"], lw=3, label="5–95%")
            ax.scatter(np.arange(7), mean, label="Mean", color=PALETTE["orange"])
            ax.set_xticks(np.arange(7), BANDS[1:])
            ax.set(ylim=(0, 1), ylabel="Internal sigmoid gate", title="Selected-patch gate distribution summaries")
            ax.legend()
            return fig
        make(42, gate_summary, [analysis_path])
        def gate_maps():
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            for i, ax in enumerate(axes.flat):
                if i >= 7:
                    ax.set_axis_off()
                    continue
                im = ax.imshow(np.asarray(band_data["gate_slices"][i])[0], cmap="viridis", vmin=0, vmax=1)
                ax.set_title(BANDS[i + 1] + " gate / feature channel 0")
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=.046)
            fig.suptitle(f"{stage}: feature-grid gates; not anatomical probability maps")
            return fig
        make(43, gate_maps, [analysis_path])
        distinct = {a["case_id"]: (a["native_spacing_zyx_mm"][0], float(np.mean(m["gate_mean"])))
                    for a, name, m in wavelets if name == stage}
        if len(distinct) >= 3 and len({v[0] for v in distinct.values()}) > 1:
            def acquisition_gates():
                fig, ax = plt.subplots(figsize=(8, 5))
                x, y = np.asarray(list(distinct.values())).T
                ax.scatter(x, y, color=PALETTE["blue"])
                ax.set(xlabel="Native z spacing (mm)", ylabel="Mean detail gate",
                       title="Selected-patch association; no causal interpretation")
                return fig
            make(45, acquisition_gates, [analysis_path])
        else:
            skip(45, "Need at least three actual captured cases and variable native spacing.")
    fusion = [(name, values) for a in analyses for name, values in a["modules"].items() if name.startswith("fuse")]
    if fusion:
        def fusion_weights():
            fig, axes = plt.subplots(len(fusion[:3]), 1, figsize=(12, 3 * len(fusion[:3])), squeeze=False)
            for ax, (name, values) in zip(axes[:, 0], fusion[:3]):
                im = ax.imshow(np.asarray(values), aspect="auto", cmap="viridis", vmin=0, vmax=1)
                ax.set(title=f"{name}: source weights sum to 1 per channel", xlabel="Channel", ylabel="Source index")
                fig.colorbar(im, ax=ax)
            return fig
        make(44, fusion_weights, [analysis_path])
    probes = [m for a in analyses for name, m in a["modules"].items() if name == "transformer"]
    if probes:
        def attention_probe():
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(np.asarray(probes[0]["mean_head_attention"]), cmap="viridis")
            ax.set(xlabel="Selected key index", ylabel="Selected query index", title=probes[0]["scope"])
            fig.colorbar(im, ax=ax, label="Restricted-probe softmax weight")
            return fig
        make(46, attention_probe, [analysis_path], "Explicit bounded diagnostic recomputes attention over a subset; not causal attribution.")

    samples = load_voxel_samples(primary)
    if samples is not None:
        p, y, weights, case_scores = samples
        bins, brier, ece = calibration_values(p, y, weights)
        probability_quantiles = {name: np.quantile(p[selected], [0., .25, .5, .75, .9, .99, 1.]).tolist()
            if selected.any() else None for name, selected in (("reference_positive", y), ("reference_negative", ~y))}
        threshold_diagnostics = []
        for threshold in sorted(set((.5, .9, .95, .98, .99, float(protocol.get("threshold", .5))))):
            selected = p >= threshold
            tp, fp, fn = float(weights[selected & y].sum()), float(weights[selected & ~y].sum()), float(weights[~selected & y].sum())
            threshold_diagnostics.append({"threshold": threshold, "weighted_precision": tp / (tp + fp) if tp + fp else 0.,
                "weighted_recall": tp / (tp + fn) if tp + fn else None,
                "sampled_predicted_positive_voxels": int(selected.sum())})
        atomic_json(output / "calibration.json", {"brier": brier, "ece_10_equal_width_bins": ece, "bins": bins,
            "sampled_voxels": len(p), "sampling": "uniform within case; deterministic bounded subsampling",
            "weighting": "native voxel count / retained sampled count per case",
            "sampled_reference_positive_voxels": int(y.sum()),
            "estimated_population_foreground_prevalence": float(np.average(y, weights=weights)),
            "probability_quantile_levels": [0., .25, .5, .75, .9, .99, 1.],
            "unweighted_sample_probability_quantiles_by_class": probability_quantiles,
            "threshold_diagnostics": threshold_diagnostics,
            "threshold_diagnostic_scope": "Descriptive sampled raw probabilities; no component filtering or threshold selection; not patient-level full-volume metrics.",
            "interpretation": "voxel-pooled descriptive analysis dominated by background; not patient diagnosis"})
        def calibration_plot():
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot([0, 1], [0, 1], ls="--", color="#777777")
            ax.plot([r["predicted"] for r in bins], [r["observed"] for r in bins], marker="o")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Weighted mean membership probability",
                   ylabel="Weighted reference fraction", title=f"Native-grid reliability: Brier={brier:.4g}, ECE={ece:.4g}")
            return fig
        make(48, calibration_plot, list((primary / "samples").glob("*.npz")), "Inverse inclusion weights; no class balancing or test-threshold selection.")
        if y.any() and not y.all():
            def roc_pr_plot():
                fpr, recall, precision, auc, ap = weighted_roc_pr(p, y, weights)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].plot(fpr, recall)
                axes[0].plot([0, 1], [0, 1], ls="--", color="#777777")
                axes[0].set(xlabel="Weighted voxel FPR", ylabel="Weighted voxel recall", title=f"Descriptive ROC AUC {auc:.4g}")
                axes[1].plot(recall, precision)
                axes[1].set(xlabel="Weighted voxel recall", ylabel="Weighted voxel precision", title=f"Descriptive average precision {ap:.4g}")
                return fig
            make(49, roc_pr_plot, list((primary / "samples").glob("*.npz")), "Sampled full-volume prevalence restored by inclusion weights; threshold remains frozen.")
        else:
            skip(49, "Both classes were not represented in the bounded real sample; no ROC/PR is fabricated.")
        eligible = [r for r in success if r["case_id"] in case_scores]
        if len(eligible) >= 3:
            def risk_coverage():
                ranked = sorted(eligible, key=lambda r: (case_scores[r["case_id"]], r["case_id"]))
                risk = np.cumsum([1 - r["dice"] for r in ranked]) / np.arange(1, len(ranked) + 1)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(np.arange(1, len(ranked) + 1) / len(ranked), risk)
                ax.set(xlabel="Fraction of cases retained (lowest mean sampled entropy first)",
                       ylabel="Mean 1 - Dice", title="Descriptive uncertainty ranking; background can dominate entropy")
                return fig
            make(50, risk_coverage, [metric_source, primary / "samples"])
    skip(51, "No ensemble-inference command is implemented. Multiple result folders are model comparisons, not ensemble disagreement.")
    skip(54, "No automated acquisition stress-test experiment is implemented; training noise is not a robustness study.")
    for number in (36, 38):
        if statuses[number - 1]["status"] == "skipped" and not statuses[number - 1]["source_artifacts"]:
            skip(number, statuses[number - 1]["reason"] + " Saved probabilities, prepared CT, and matching metrics are required.")
    atomic_json(output / "figure_manifest.json", statuses)
    write_csv(output / "figure_manifest.csv", statuses)
    summary_rows = [{"case_id": r.get("case_id"), "patient_id": r.get("patient_id"), "status": r.get("status"),
                     "Dice": r.get("dice"), "HD95_mm": r.get("hd95_mm"), "surface_status": r.get("surface_status"),
                     "complete_miss": r.get("complete_miss"), "empty_prediction": r.get("empty_prediction"),
                     "zero_overlap": r.get("zero_overlap", int(r.get("reference_volume_ml", 0) > 0 and r.get("dice") == 0)),
                     "reason": r.get("reason", "")} for r in metrics]
    def table(rows):
        if not rows:
            return "<p>No patient metrics recorded. Run evaluate after training.</p>"
        keys = list(rows[0])
        return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{html.escape(str(k))}</th>" for k in keys) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(str(clean_json(r.get(k)))) + "</td>" for k in keys) + "</tr>" for r in rows) + "</tbody></table></div>"
    cards = []
    for status in statuses:
        content = ""
        for file in status["files"]:
            if file.lower().endswith(".png"):
                escaped = html.escape(file, quote=True)
                content += f"<a href='{escaped}'><img loading='lazy' src='{escaped}' alt='{html.escape(status['title'])}'></a>"
        cards.append(f"<section id='figure-{status['figure_id']}'><h2>{status['figure_id']:02d}. {html.escape(status['title'])}</h2>"
                     f"<p class='status'>{status['status'].upper()}: {html.escape(status.get('reason') or status.get('caption',''))}</p>{content}</section>")
    counts = Counter(s["status"] for s in statuses)
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CT primary tumor segmentation — recorded results</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#17212b;background:white;margin:0 auto;padding:32px;max-width:1200px}
h1{font-size:30px}h2{font-size:21px}section{border-top:1px solid #dce3e8;padding:20px 0}img{max-width:100%;height:auto;background:white}
a{color:#0072b2}.status{font-size:14px;color:#475569}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px;border:1px solid #dce3e8;text-align:left}th{background:#fff}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white}
</style><h1>CT primary tumor segmentation</h1>""" + \
        f"<p>Recorded evidence only. {html.escape(metric_scope)}. Primary target: {html.escape(str(protocol.get('target','not recorded')))}.</p>" + \
        f"<p>Figure families: {dict(counts)}. Missing analyses are visible below; no example performance is substituted.</p>" + \
        "<p>Blue: reference. Orange: prediction. Magenta: false positive. Green: false negative. CT pixels preserve their grayscale window.</p>" + \
        "<p><a href='figure_manifest.csv'>Figure status CSV</a> · <a href='figure_manifest.json'>Figure provenance JSON</a></p>" + \
        "<h2>Patient records</h2>" + table(summary_rows) + "<h2>Protocol</h2><pre>" + \
        html.escape(json.dumps(clean_json(protocol), indent=2)) + "</pre>" + "".join(cards) + "</html>"
    (output / "report.html").write_text(document, encoding="utf-8")
    LOGGER.info("Report: %s; figures %s", output / "report.html", dict(counts))
    return statuses


def report_command(args):
    roots = [Path(p) for p in args.inputs]
    if not roots or any(not p.exists() for p in roots):
        raise UserError("Every report --inputs path must be an existing recorded output directory.")
    statuses = report_build(Path(args.output), roots, args.figure_cases, args.dpi, args.vector,
                            args.manifest, args.splits, args.surfaces)
    print(f"Offline report: {Path(args.output).resolve() / 'report.html'}")
    if any(s["status"] == "failed" for s in statuses):
        raise UserError("Some figures failed; inspect figure_manifest.json. Available evidence was retained.")


# 10. Validation-only operating points and explicitly requested ablations ------
def select_threshold_command(args):
    """Select ONE threshold using saved full-volume validation predictions only."""
    sitk = require("SimpleITK")
    root = Path(args.evaluation).resolve()
    protocol = load_json(root / "protocol.json")
    if protocol["partition"] != "val":
        raise UserError("Threshold selection requires an explicit val evaluation, never test/external/predict.")
    if Path(args.output).exists():
        raise UserError("Operating-point output already exists; preserve it and choose a new path.")
    manifest, base = load_manifest(args.manifest)
    if manifest["fingerprint"] != protocol["manifest_fingerprint"]:
        raise UserError("Validation manifest fingerprint mismatch.")
    rows = load_json(root / "patient_metrics.json")
    if not rows or any(r["status"] != "success" for r in rows):
        raise UserError("Threshold selection requires all validation cases to have completed successfully.")
    expected_ids = protocol.get("case_ids")
    if getattr(args, "splits", None):
        split = validate_split(manifest, load_json(args.splits))
        if split["fingerprint"] != protocol["split_fingerprint"]:
            raise UserError("Validation split fingerprint mismatch.")
        expected_ids = split["partitions"]["val"]
    if expected_ids is None:
        raise UserError("Legacy evaluation has no expected case list. Supply --splits to verify complete validation coverage.")
    actual_ids = [r["case_id"] for r in rows]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise UserError("Threshold selection requires exactly one result for every expected validation case.")
    grid = sorted(set(args.thresholds))
    if not grid or any(not 0 < x < 1 for x in grid):
        raise UserError("All candidate thresholds must be strictly between zero and one.")
    modes = sorted(set(getattr(args, "postprocess_modes", ("none",))),
                   key=lambda mode: ("none", "largest", "min-volume").index(mode))
    minimums = sorted(set(getattr(args, "min_volumes_ml", (.1, 1.))))
    if not modes:
        raise UserError("At least one postprocessing mode is required.")
    if "min-volume" in modes and (not minimums or any(not math.isfinite(v) or v <= 0 for v in minimums)):
        raise UserError("Candidate component volumes must be positive and finite in mL.")
    recipes = [postprocessing_recipe(mode, volume) for mode in modes
               for volume in (minimums if mode == "min-volume" else (0.,))]
    by_id = {r["case_id"]: r for r in manifest["records"]}
    candidates = [{"threshold": threshold, "postprocessing": recipe, "patients": {},
                   "empty_prediction_cases": 0, "zero_overlap_cases": 0}
                  for threshold in grid for recipe in recipes]
    for row in rows:
        cid = row["case_id"]
        if cid not in by_id or row["patient_id"] != by_id[cid]["patient_id"]:
            raise UserError("Validation result case/patient identity differs from the manifest.")
        directory, metadata = cache_path(base, by_id[cid])
        reference = np.load(directory / "native_mask.npy", mmap_mode="r", allow_pickle=False)
        try:
            path = root / "cases" / cid / "probability.nii.gz"
            if not path.exists():
                raise UserError(f"Missing saved validation probability: {path}; evaluate without --no-save-predictions.")
            probability_image = sitk.ReadImage(str(path))
            if not grids_equal(probability_image, empty_grid(metadata["native_grid"])):
                raise UserError(f"Validation probability geometry differs from the native reference: {cid}")
            probability = sitk.GetArrayFromImage(probability_image)
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise UserError(f"Validation probability must be finite and in [0,1]: {cid}")
            if probability.shape != reference.shape:
                raise UserError(f"Validation reference and probability shapes differ: {cid}")
            spacing = metadata.get("native_spacing_zyx_mm", metadata["native_grid"]["spacing_xyz_mm"][::-1])
            reference_bool = np.asarray(reference, dtype=bool)
            r = int(reference_bool.sum())
            # One native case and one candidate mask at a time. No cohort-sized
            # stack of probabilities or candidate masks is retained.
            for candidate in candidates:
                pred = postprocess_prediction(probability >= candidate["threshold"], spacing,
                                              candidate["postprocessing"])
                p, tp = int(pred.sum()), int(np.count_nonzero(pred & reference_bool))
                dice = 2 * tp / (p + r) if p + r else 1.
                candidate["patients"].setdefault(row["patient_id"], []).append(float(dice))
                candidate["empty_prediction_cases"] += int(p == 0)
                candidate["zero_overlap_cases"] += int(r > 0 and tp == 0)
        finally:
            reference._mmap.close()
    for candidate in candidates:
        candidate["mean_dice"] = float(np.mean([np.mean(v) for v in candidate.pop("patients").values()]))
    # Prefer the simpler recipe on exact score ties, then threshold closest to 0.5.
    chosen = min(candidates, key=lambda c: (-c["mean_dice"],
                 ("none", "largest", "min-volume").index(c["postprocessing"]["mode"]),
                 abs(c["threshold"] - .5), c["threshold"], c["postprocessing"]["min_volume_ml"]))
    means = {c["threshold"]: c["mean_dice"] for c in candidates
             if c["postprocessing"] == chosen["postprocessing"]}
    result = {"version": 2, "threshold": chosen["threshold"], "source_partition": "val",
              "criterion": "mean within-patient averaged native-volume Dice", "scores": means,
              "candidates": candidates, "selected_validation_dice": chosen["mean_dice"],
              "development_score_warning": "Threshold and component recipe were selected on these validation cases; this is not an independent performance estimate or a retraining gain.",
              "checkpoint_sha256": protocol["checkpoint_sha256"], "target": protocol["target"],
              "manifest_fingerprint": manifest["fingerprint"], "split_fingerprint": protocol["split_fingerprint"],
              "postprocessing": chosen["postprocessing"], "created": now(),
              "case_ids": sorted(row["case_id"] for row in rows)}
    if "overlap" in protocol:
        result["overlap"] = protocol["overlap"]
    result["fingerprint"] = stable_hash(result)
    atomic_json(Path(args.output), result)
    print(f"Frozen validation-selected threshold {chosen['threshold']:g}, "
          f"postprocessing {chosen['postprocessing']}: {args.output}")


ABLATIONS = ("baseline", "multiscale", "transformer", "fusion", "wavelet", "decoder", "acquisition",
             "full", "constant-metadata", "shuffled-metadata", "spatial-gate", "one-window")


def ablate_command(args):
    if args.resume:
        raise UserError("Resume individual ablation runs with train --resume, not ablate.")
    root = Path(args.run).resolve()
    if root.exists():
        raise UserError("Ablation parent directory already exists; use a new --run.")
    if len(set(args.configs)) != len(args.configs):
        raise UserError("Duplicate ablation names are not allowed.")
    root.mkdir(parents=True)
    atomic_json(root / "ablation_plan.json", {"requested": args.configs, "created": now(),
                 "warning": "Explicit selected jobs only; matching budget does not guarantee matching capacity or runtime."})
    outcomes = []
    for name in args.configs:
        job = copy.deepcopy(args)
        job.command, job.run, job.resume = "train", str(root / f"{name}-seed{args.seed}"), None
        job.model, job.input_channels, job.spatial_gate, job.metadata_control = "hybrid3d", 2, False, "normal"
        for key in ("multiscale", "transformer", "adaptive_fusion", "wavelet", "decoder_conditioning",
                    "acquisition_conditioning", "boundary", "deep_supervision"):
            setattr(job, "no_" + key, False)
        if name == "baseline":
            job.model, job.no_boundary = "baseline3d", True
        elif name in ("multiscale", "transformer", "fusion", "wavelet", "decoder", "acquisition"):
            level = ("multiscale", "transformer", "fusion", "wavelet", "decoder", "acquisition").index(name)
            job.no_transformer = level < 1
            job.no_adaptive_fusion = level < 2
            job.no_wavelet = level < 3
            job.no_decoder_conditioning = level < 4
            job.no_acquisition_conditioning = level < 5
            job.no_boundary = True
        elif name == "constant-metadata":
            job.metadata_control = "constant"
        elif name == "shuffled-metadata":
            job.metadata_control = "shuffled"
        elif name == "spatial-gate":
            job.spatial_gate, job.no_wavelet = True, True
        elif name == "one-window":
            job.input_channels = 1
        try:
            code = train_command(job)
            outcomes.append({"configuration": name, "run": job.run, "status": "success" if not code else "interrupted"})
            atomic_json(root / "ablation_outcomes.json", outcomes)
            if code:
                return code
        except (UserError, RuntimeError, OSError, ValueError) as exc:
            outcomes.append({"configuration": name, "run": job.run, "status": "failed", "reason": str(exc)})
            atomic_json(root / "ablation_outcomes.json", outcomes)
            raise
    print(f"Finished {len(outcomes)} explicitly selected training jobs. Evaluate each best checkpoint on the same partition.")
    return 0


# 11. Meaningful built-in mathematical and integration self-tests -------------
def self_test_command(args):
    require("numpy")
    results = []
    def check(name, callback, dependencies=()):
        missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
        if missing:
            results.append({"test": name, "status": "skipped", "reason": "Missing: " + ", ".join(missing)})
            return
        started = time.perf_counter()
        try:
            measured = callback()
            results.append({"test": name, "status": "passed", "seconds": time.perf_counter()-started,
                            "measurements": clean_json(measured)})
        except Exception as exc:  # test harness records failures and returns a failing exit code
            results.append({"test": name, "status": "failed", "reason": str(exc),
                            "traceback": traceback.format_exc()})
    with tempfile.TemporaryDirectory(prefix="lung-ct-selftest-") as temporary:
        temp = Path(temporary)
        def window_test():
            hu = np.asarray([-2000, -1000, -160, 0, 240, 400, 1000], dtype=np.float32)
            views = windows_from_hu(hu)
            assert views.shape == (2, 7)
            np.testing.assert_allclose(views[0, [0, 1, 5, 6]], [0, 0, 1, 1])
            np.testing.assert_allclose(views[1, [0, 2, 4, 6]], [0, 0, 1, 1])
            assert 0 <= views.min() <= views.max() <= 1
            assert windows_from_hu(hu, 1).shape == (1, 7)
        check("HU windows and one-window ablation", window_test)
        def binary_test():
            for bad in (np.asarray([0, 2]), np.asarray([0, np.nan]), np.zeros(3)):
                try:
                    validate_binary(bad)
                except UserError:
                    continue
                raise AssertionError("Invalid binary input accepted")
            validate_binary(np.zeros(3), allow_empty=True)
            validate_binary(np.asarray([0, 1]))
        check("Binary and empty-reference policy", binary_test)
        def polygon_test():
            rectangle = np.asarray([[0, 0], [4, 0], [4, 3], [0, 3]])
            validate_simple_polygon_xy(rectangle)
            validate_simple_polygon_xy(rectangle[::-1])
            for invalid in (np.asarray([[0, 0], [4, 4], [0, 4], [4, 0]]),
                            np.asarray([[0, 0], [1, 0], [2, 0]])):
                try:
                    validate_simple_polygon_xy(invalid)
                except UserError:
                    continue
                raise AssertionError("Self-crossing/degenerate contour accepted")
        check("Contour crossing/degeneracy rejection", polygon_test)
        def axis_test():
            spacing_xyz = np.array([.7, 1.2, 3.])
            origin = np.array([10., -5., 20.])
            direction = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            landmark_zyx = np.array([4., 6., 8.])
            physical = origin + direction @ (landmark_zyx[::-1] * spacing_xyz)
            recovered_zyx = ((direction.T @ (physical-origin)) / spacing_xyz)[::-1]
            np.testing.assert_allclose(recovered_zyx, landmark_zyx, atol=1e-12)
            for bad in ((0, 1, 2), (1, np.nan, 2), (1, 2)):
                try:
                    positive_spacing(bad)
                except UserError:
                    continue
                raise AssertionError("Bad spacing accepted")
            return {"landmark_lps_mm": physical.tolist()}
        check("Axis/physical landmark round trip and spacing validation", axis_test)
        def haar_reference_test():
            # Independent NumPy reference for the orthonormal sign/order convention.
            x = np.random.default_rng(4).normal(size=(8, 10, 12))
            coefficients = {}
            for letters in BANDS:
                value = x.copy()
                for axis, letter in enumerate(letters):
                    a = np.take(value, np.arange(0, value.shape[axis], 2), axis=axis)
                    b = np.take(value, np.arange(1, value.shape[axis], 2), axis=axis)
                    value = (a + (1 if letter == "L" else -1) * b) / math.sqrt(2)
                coefficients[letters] = value
            restored = np.zeros_like(x)
            for letters, value in coefficients.items():
                for axis in reversed(range(3)):
                    shape = list(value.shape)
                    shape[axis] *= 2
                    expanded = np.zeros(shape)
                    even, odd = [slice(None)]*3, [slice(None)]*3
                    even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
                    expanded[tuple(even)] = value / math.sqrt(2)
                    expanded[tuple(odd)] = (1 if letters[axis] == "L" else -1) * value / math.sqrt(2)
                    value = expanded
                restored += value
            error = float(np.abs(restored-x).max())
            energy_error = abs(float(sum(np.sum(c*c) for c in coefficients.values()) - np.sum(x*x)))
            assert error < 1e-12 and energy_error < 1e-9
            return {"max_reconstruction_error": error, "energy_absolute_error": energy_error,
                    "scope": "independent NumPy reference, not the PyTorch forward pass"}
        check("Independent 3D Haar mathematics", haar_reference_test)
        def patch_test():
            volume = np.arange(5*7*9).reshape(5, 7, 9).astype(np.float32)
            patch = extract_patch(volume, (-2, 3, 4), (7, 6, 8), -1024)
            assert patch.shape == (7, 6, 8) and (patch[:2] == -1024).all()
            np.testing.assert_array_equal(patch[2:7, :4, :5], volume[:, 3:, 4:])
            rng1, rng2 = indexed_rng(1, 2, 3), indexed_rng(1, 2, 3)
            np.testing.assert_array_equal(rng1.integers(100, size=20), rng2.integers(100, size=20))
        check("Patch padding and indexed randomness", patch_test)
        def sliding_test():
            volume = np.random.default_rng(8).normal(size=(9, 13, 17)).astype(np.float32)
            recovered = sliding_window_logits(volume, (6, 8, 10), .375, lambda p: p, str(temp))
            error = float(np.abs(recovered-volume).max())
            assert error < 2e-5
            small = np.full((1, 3, 2), 4., dtype=np.float32)
            result = sliding_window_logits(small, (4, 6, 8), .25, lambda p: np.full(p.shape, 7., np.float32))
            np.testing.assert_allclose(result, 7., atol=1e-5)
            return {"max_awkward_shape_error": error, "small_shape": list(result.shape)}
        check("Gaussian sliding-window coverage/shape and disk accumulation", sliding_test)
        def metric_test():
            a = np.zeros((7, 8, 9), bool)
            b = a.copy()
            a[2, 3, 4], b[3, 3, 4] = True, True
            perfect = evaluate_arrays(a, a, (3, 1, 1), 2)
            shifted = evaluate_arrays(a, b, (3, 1, 1), 2)
            miss = evaluate_arrays(np.zeros_like(a), a, (3, 1, 1), 2)
            empty = evaluate_arrays(np.zeros_like(a), np.zeros_like(a), (3, 1, 1), 2)
            assert perfect["dice"] == perfect["iou"] == perfect["surface_dice"] == 1
            assert shifted["dice"] == 0 and shifted["hd95_mm"] == 3 and shifted["surface_dice"] == 0
            assert miss["dice"] == 0 and miss["complete_miss"] == 1 and miss["hd95_mm"] is None
            assert empty["dice"] == 1 and empty["surface_status"] == "both_empty"
            return {"one_voxel_z_shift_mm": shifted["hd95_mm"], "miss_surface_status": miss["surface_status"]}
        check("Overlap, complete miss, and anisotropic physical surfaces", metric_test, ("scipy",))
        def boundary_test():
            mask = np.zeros((11, 13, 15), np.uint8)
            mask[3:8, 4:9, 5:10] = 1
            band = boundary_band(mask, (3, 1, 1), 1.5)
            assert not band[2, 6, 7] and band[5, 3, 7]
            patch = extract_patch(band, (4, 5, 6), (2, 2, 2))
            np.testing.assert_array_equal(patch, band[4:6, 5:7, 6:8])
        check("Full-volume physical boundary band and patch slicing", boundary_test, ("scipy",))
        def augmentation_test():
            mask = np.zeros((8, 16, 16), np.uint8)
            mask[2:6, 5:10, 6:12] = 1
            hu = mask.astype(np.float32)*400 - 900
            band = boundary_band(mask, (2.5, 1, 1), 2)
            first = augment_patch(hu, mask, band, indexed_rng(10, 1, 1), True)
            second = augment_patch(hu, mask, band, indexed_rng(10, 1, 1), True)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)
            assert np.isin(first[1], (0, 1)).all() and np.isin(first[2], (0, 1)).all()
            assert windows_from_hu(first[0]).shape == (2, 8, 16, 16)
        check("Joint augmentation determinism and valid labels", augmentation_test, ("scipy",))
        def split_test():
            records = []
            for i in range(18):
                records.append({"case_id": f"c{i}", "patient_id": f"p{i//2}", "source_identity": f"source{i}",
                                "annotation_status": "positive"})
            # A duplicated source under another patient identity must still group together.
            records[4]["source_identity"] = records[0]["source_identity"]
            manifest = {"schema": CACHE_SCHEMA, "settings": {}, "records": records}
            manifest["fingerprint"] = stable_hash({"settings": {}, "records": records})
            path, out = temp / "manifest.json", temp / "splits.json"
            atomic_json(path, manifest)
            options = argparse.Namespace(manifest=path, output=out, val_fraction=.2, test_fraction=.2,
                                         seed=42, folds=3, fold=0, external=False)
            split_command(options)
            split = validate_split(manifest, load_json(out))
            mapping = {cid: k for k, v in split["partitions"].items() for cid in v}
            assert mapping["c0"] == mapping["c4"]
            for i in range(0, 18, 2):
                assert mapping[f"c{i}"] == mapping[f"c{i+1}"]
            bad = copy.deepcopy(split)
            bad["partitions"]["test"].append(bad["partitions"]["train"][0])
            bad["fingerprint"] = stable_hash({k: v for k, v in bad.items() if k != "fingerprint"})
            try:
                validate_split(manifest, bad)
            except UserError:
                return {"duplicate_source_grouping": "verified", "fold": 0}
            raise AssertionError("Leakage accepted")
        check("Patient/source disjoint splits, development folds, leakage rejection", split_test)
        def stats_test():
            meta = {"native_spacing_zyx_mm": [3, .8, .8], "target_spacing_zyx_mm": [2.5, 1, 1]}
            q2 = raw_descriptor(meta["native_spacing_zyx_mm"], meta["target_spacing_zyx_mm"], (2, 4, 4))
            q3 = raw_descriptor(meta["native_spacing_zyx_mm"], meta["target_spacing_zyx_mm"], (4, 8, 8))
            stats = {"mean2": q2.tolist(), "mean3": q3.tolist(), "std": [1]*9}
            np.testing.assert_allclose(normalized_descriptors(meta, stats), 0., atol=1e-6)
        check("Acquisition axes and constant-column normalization", stats_test)
        def calibration_test():
            p = np.asarray([.1, .2, .8, .9])
            y = np.asarray([0, 0, 1, 1], bool)
            weights = np.asarray([1., 3., 2., 1.])
            bins, brier, ece = calibration_values(p, y, weights)
            fpr, tpr, precision, auc, ap = weighted_roc_pr(p, y, weights)
            assert abs(auc-1) < 1e-12 and abs(ap-1) < 1e-12
            assert brier > 0 and ece > 0
            return {"known_separable_roc_auc": auc, "known_separable_average_precision": ap}
        check("Weighted calibration and ROC/PR reference values", calibration_test)
        def bootstrap_test():
            rows = [{"case_id": f"c{i}", "patient_id": f"p{i//2}", "status": "success", "dice": .2*i,
                     "complete_miss": 0, "surface_status": "finite"} for i in range(4)]
            summary = bootstrap_summary(rows, repeats=100)
            assert summary["metrics"]["dice"]["n_patients"] == 2
            assert abs(summary["metrics"]["dice"]["mean"] - .3) < 1e-10
        check("Patient bootstrap units", bootstrap_test)
        def cache_test():
            root = temp / "cache-test"
            root.mkdir()
            array = np.arange(100, dtype=np.float32)
            np.save(root / "hu.npy", array)
            atomic_json(root / "metadata.json", {"fingerprint": "test"})
            atomic_json(root / "complete.json", {"fingerprint": "test", "files": {"hu.npy": (root / "hu.npy").stat().st_size}})
            record = {"cache_dir": "cache-test", "fingerprint": "test", "case_id": "test"}
            cache_path(temp, record)
            (root / "hu.npy").write_bytes(b"interrupted")
            try:
                cache_path(temp, record)
            except UserError:
                return
            raise AssertionError("Truncated cache accepted")
        check("Cache completion and truncated-file rejection", cache_test)
        def imaging_test():
            sitk = require("SimpleITK")
            array = np.zeros((7, 9, 11), np.float32)
            array[2, 4, 6] = 1
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((.8, 1.2, 3.))
            image.SetOrigin((11., -22., 33.))
            image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
            meta = {"native_grid": grid_dict(image), "working_grid": grid_dict(image)}
            restored = native_probability(array, meta)
            np.testing.assert_allclose(sitk.GetArrayFromImage(restored), array, atol=1e-6)
            np.testing.assert_allclose(restored.TransformIndexToPhysicalPoint((6, 4, 2)),
                                       image.TransformIndexToPhysicalPoint((6, 4, 2)), atol=1e-8)
            path = temp / "landmark.nii.gz"
            sitk.WriteImage(restored, str(path))
            assert grids_equal(image, sitk.ReadImage(str(path)))
        check("SimpleITK oblique/permuted-axis NIfTI and native-grid round trip", imaging_test, ("SimpleITK",))
        check("Raw CT/RTSTRUCT synthetic reference/rasterization/HU integration",
              lambda: dicom_self_test(temp), ("SimpleITK", "pydicom", "rt_utils"))
        def figure_test():
            plt = plotting()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot([0, 1, 2], [1, .5, .2])
            ax.set(title="Synthetic self-test — not a learning curve", xlabel="Test coordinate", ylabel="Test value")
            path = temp / "white.png"
            save_figure(fig, path, dpi=90)
            pixels = plt.imread(path)
            assert np.all(pixels[0, 0, :3] > .99)
            root = temp / "empty-results"
            root.mkdir()
            statuses = report_build(root, [root], 0, 90)
            assert len(statuses) == 54 and all(r["status"] == "skipped" for r in statuses)
            assert (root / "report.html").is_file()
            return {"white_corner_rgb": pixels[0, 0, :3].tolist(), "honest_empty_report_families": len(statuses)}
        check("White PNG and offline report with honest missing artifacts", figure_test, ("matplotlib",))
        check("PyTorch Haar even/odd/singleton reconstruction, energy, autograd",
              torch_haar_self_test, ("torch",))
        check("Padded window reconstruction and invalid-key masking",
              torch_window_self_test, ("torch",))
        check("3D baseline/hybrid shapes, gates, losses/backward, safe checkpoint agreement",
              lambda: torch_model_self_test(temp), ("torch",))
    counts = Counter(r["status"] for r in results)
    outcome = "FAILED" if counts["failed"] else "PARTIAL (missing dependencies)" if counts["skipped"] else "PASSED"
    result = {"time": now(), "program_version": VERSION, "status": outcome, "counts": dict(counts),
              "environment": environment(), "tests": results,
              "scope": "synthetic mathematical/integration checks only; no real patient training or clinical validation"}
    if args.output:
        atomic_json(Path(args.output), result)
    for item in results:
        print(f"{item['status'].upper():7} {item['test']}" + (": " + item['reason'] if item.get("reason") else ""))
    print(f"Self-test: {outcome}; {dict(counts)}")
    return 1 if counts["failed"] else (2 if counts["skipped"] and not args.available_only else 0)


def torch_haar_self_test():
    require_torch()
    transform = Haar3D().double()
    measurements = []
    for shape in ((6, 8, 10), (5, 7, 9), (1, 3, 1)):
        x = torch.randn(1, 2, *shape, dtype=torch.float64, requires_grad=True)
        bands, original = transform.analysis(x)
        restored = transform.synthesis(bands, original)
        error = float((restored-x).abs().max().detach())
        padded = F.pad(x, (0, shape[2] % 2, 0, shape[1] % 2, 0, shape[0] % 2), mode="replicate")
        energy_error = float((bands.square().sum()-padded.square().sum()).abs().detach())
        assert error < 1e-10 and energy_error < 1e-9
        bands.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        measurements.append({"shape": shape, "reconstruction_error": error, "energy_error_padded": energy_error})
    return measurements


def torch_window_self_test():
    require_torch()
    x = torch.randn(2, 8, 5, 7, 9)
    tokens, meta, valid = partition_windows(x, (4, 6, 6))
    torch.testing.assert_close(reverse_windows(tokens, meta), x, rtol=0, atol=0)
    assert valid.any(1).all() and (~valid).any()
    block = AttentionBlock(8, 2).eval()
    changed = tokens.clone()
    changed[~valid] = 9999.
    with torch.inference_mode():
        original_result, changed_result = block(tokens, valid), block(changed, valid)
    torch.testing.assert_close(original_result[valid], changed_result[valid], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(changed_result[~valid]) == 0
    return {"windows": int(len(tokens)), "padded_tokens": int((~valid).sum())}


def torch_model_self_test(temp):
    require_torch()
    seed_everything(7, deterministic=True)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(min(2, original_threads))
    results = []
    try:
        image = torch.rand(1, 2, 8, 16, 16)
        target = torch.zeros(1, 1, 8, 16, 16)
        target[:, :, 2:6, 5:10, 6:12] = 1
        q = torch.zeros(1, 2, 9)
        for name in ("baseline3d", "hybrid3d"):
            config = ModelConfig(name=name, widths=(4, 8, 16, 24, 32))
            if name == "baseline3d":
                config.multiscale = config.transformer = config.adaptive_fusion = config.wavelet = False
            model = SegmentationModel(config).train()
            model.set_capture(True)
            output = model(image, q, True)
            assert output["logits"].shape == target.shape
            loss, parts = total_loss(output, target, target)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            if name == "hybrid3d":
                for module in (model.skip2, model.skip3):
                    assert 0 < module.kappa_logit.sigmoid() < 1
                    assert all(0 <= g <= 1 for g in module.analysis_record["gate_mean"])
                module = model.skip3
                module.force_identity = True
                encoder = torch.randn(1, 24, 2, 2, 2)
                with torch.inference_mode():
                    identical = module(encoder, encoder, torch.zeros(1, 9))
                torch.testing.assert_close(identical, encoder, rtol=0, atol=0)
                module.force_identity = False
            model.eval()
            model.set_capture(False)
            with torch.inference_mode():
                main_only = model(image, q, False)["logits"]
                with_aux = model(image, q, True)["logits"]
            torch.testing.assert_close(main_only, with_aux, atol=0, rtol=0)
            checkpoint_path = Path(temp) / f"{name}.pt"
            atomic_checkpoint(checkpoint_path, {"program_version": VERSION, "model_state": model.state_dict(),
                              "model_config": dataclasses.asdict(config), "rng": capture_rng()})
            saved = safe_load_checkpoint(checkpoint_path)
            restored = SegmentationModel(ModelConfig(**saved["model_config"])).eval()
            restored.load_state_dict(saved["model_state"])
            with torch.inference_mode():
                again = restored(image, q, False)["logits"]
            torch.testing.assert_close(main_only, again, rtol=0, atol=0)
            results.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                            "output_shape": list(main_only.shape), "loss": parts, "checkpoint_max_error": float((again-main_only).abs().max())})
        # Ablation mechanisms are constructed, not merely renamed.
        spatial = ModelConfig(widths=(4, 8, 16, 24, 32), wavelet=False, spatial_gate=True,
                              transformer=False, multiscale=False, adaptive_fusion=False,
                              boundary=False, deep_supervision=False)
        control = SegmentationModel(spatial).eval()
        assert isinstance(control.skip2, SpatialSkipControl) and isinstance(control.transformer, nn.Identity)
        with torch.inference_mode():
            assert control(image, q, False)["logits"].shape == target.shape
        return results
    finally:
        torch.set_num_threads(original_threads)


def dicom_self_test(temp):
    """Small generated CT+RTSTRUCT fixture, including a planar hole and non-square slices.

    This is not patient data. It specifically catches the common rows/columns/slices
    transpose mistake and an accidental second HU rescale.
    """
    pydicom = require("pydicom")
    builder = require("rt_utils").RTStructBuilder
    sitk = require("SimpleITK")
    uid = require("pydicom.uid")
    dataset_module = require("pydicom.dataset")
    directory = Path(temp) / "dicom-fixture"
    directory.mkdir()
    series, study, frame = uid.generate_uid(), uid.generate_uid(), uid.generate_uid()
    files = []
    for z in range(5):
        metadata = dataset_module.FileMetaDataset()
        metadata.MediaStorageSOPClassUID = uid.CTImageStorage
        metadata.MediaStorageSOPInstanceUID = uid.generate_uid()
        metadata.TransferSyntaxUID = uid.ExplicitVRLittleEndian
        metadata.ImplementationClassUID = uid.generate_uid()
        path = directory / f"{z}.dcm"
        ds = dataset_module.FileDataset(str(path), {}, file_meta=metadata, preamble=b"\0"*128)
        ds.SOPClassUID, ds.SOPInstanceUID = metadata.MediaStorageSOPClassUID, metadata.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID, ds.StudyInstanceUID, ds.FrameOfReferenceUID = series, study, frame
        ds.PatientID, ds.PatientName, ds.Modality = "SYNTHETIC", "SelfTest^Only", "CT"
        ds.PatientBirthDate, ds.PatientSex = "20000101", "O"
        ds.StudyDate, ds.StudyTime, ds.StudyID = "20200101", "120000", "1"
        ds.SeriesNumber, ds.InstanceNumber = 1, z+1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [10, -20, 30+z*3]
        ds.PixelSpacing, ds.SliceThickness = [1.0, 1.0], 3
        ds.Rows, ds.Columns = 20, 20
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 2, -1000
        ds.PixelData = np.full((20, 20), 500+z, dtype="<i2").tobytes()
        pydicom.dcmwrite(str(path), ds, enforce_file_format=True)
        files.append(str(path))
    struct = builder.create_new(dicom_series_path=str(directory))
    mask_rcs = np.zeros((20, 20, 5), bool)
    mask_rcs[4:14, 4:14, 1:4] = True
    mask_rcs[7:11, 7:11, 1:4] = False
    struct.add_roi(mask=mask_rcs, name="GTV-1", approximate_contours=False)
    rtpath = Path(temp) / "fixture-rt.dcm"
    struct.save(str(rtpath))
    image, mask_image, _ = read_raw_case({"ct_files": files, "series_uid": series, "rtstruct": str(rtpath)}, "GTV-1")
    hu = sitk.GetArrayFromImage(image)
    np.testing.assert_allclose(hu[:, 0, 0], [0, 2, 4, 6, 8], atol=1e-6)
    actual = sitk.GetArrayFromImage(mask_image)
    expected = np.transpose(mask_rcs, (2, 0, 1))
    assert actual.shape == (5, 20, 20) and grids_equal(image, mask_image)
    # RT contour extraction/rasterization can change pixel-center edge inclusion;
    # a hole must remain, and overlap must be high. Record the actual measured value.
    dice = 2*np.count_nonzero(actual & expected)/(actual.sum()+expected.sum())
    assert dice > .90 and not actual[2, 8, 8]
    return {"synthetic_roundtrip_dice": float(dice), "hole_preserved": True, "hu_rescale": "once"}


# 12. Complete command-line interface and Windows-safe entry point ------------
def mine_negatives_command(args):
    """Mine only TRAIN working-grid false positives; never read validation/test labels."""
    require_torch()
    if not 0 < args.threshold < 1 or args.centres_per_case < 1 or args.max_cases < 0:
        raise UserError("Mining needs threshold in (0,1), positive centre budget, nonnegative max-cases.")
    output = Path(args.output).resolve()
    if output.exists():
        raise UserError("Hard-negative index already exists; choose a new output.")
    manifest, base = load_manifest(args.manifest)
    split = validate_split(manifest, load_json(args.splits))
    device = select_device(args.device)
    model, saved = load_inference_model(args.checkpoint, device)
    if (saved["manifest_fingerprint"] != manifest["fingerprint"]
            or saved["split_fingerprint"] != split["fingerprint"] or saved.get("overfit_cases")):
        raise UserError("Mining requires a non-diagnostic checkpoint trained on this manifest/split.")
    records = choose_records(manifest, split, "train")
    if args.max_cases:
        records = records[:args.max_cases]
    precision = resolve_precision(device, args.precision)
    result = {"version": 1, "created": now(), "source_partition": "train", "coordinate_grid": "working_zyx",
              "manifest_fingerprint": manifest["fingerprint"], "split_fingerprint": split["fingerprint"],
              "prediction_checkpoint_sha256": file_digest(args.checkpoint), "threshold": args.threshold,
              "selection": "bounded per-8-slice uniform false-positive candidates then global subsampling",
              "cases": {}}
    for record in records:
        directory, metadata = cache_path(base, record)
        hu = np.load(directory / "hu.npy", mmap_mode="r", allow_pickle=False)
        reference = np.load(directory / "mask.npy", mmap_mode="r", allow_pickle=False)
        try:
            probability, _ = predict_working_volume(model, hu, metadata, saved["acquisition_stats"], saved["patch"],
                args.overlap, device, precision, saved["metadata_control"])
            rng = np.random.default_rng(int(stable_hash(record["case_id"])[:8], 16) + args.seed)
            centres, false_positive_voxels = [], 0
            for z in range(0, len(probability), 8):
                slab = probability[z:z+8]
                candidates = np.flatnonzero((slab >= args.threshold) & (reference[z:z+8] == 0))
                false_positive_voxels += len(candidates)
                if len(candidates):
                    selected = rng.choice(candidates, min(args.centres_per_case, len(candidates)), replace=False)
                    coords = np.column_stack(np.unravel_index(selected, slab.shape))
                    coords[:, 0] += z
                    centres.extend(coords.tolist())
            if len(centres) > args.centres_per_case:
                centres = [centres[i] for i in rng.choice(len(centres), args.centres_per_case, replace=False)]
            result["cases"][record["case_id"]] = {"shape_zyx": list(probability.shape), "centres_zyx": centres,
                                                    "false_positive_voxels": false_positive_voxels}
        finally:
            hu._mmap.close()
            reference._mmap.close()
        del probability, slab, candidates
        LOGGER.info("Mined train case %d/%d (%d centres)", len(result["cases"]), len(records), len(centres))
    result["fingerprint"] = stable_hash(result)
    atomic_json(output, result)
    print(f"Training-only hard-negative index: {output}")


def add_figure_arguments(parser):
    parser.add_argument("--figure-cases", type=int, default=6, help="Bound selected qualitative/feature cases; 0 disables captures.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--vector", choices=("none", "svg", "pdf", "both"), default="none")


def add_train_arguments(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", choices=("baseline3d", "hybrid3d"), default="hybrid3d")
    parser.add_argument("--preset", choices=("auto", *PRESETS), default="auto")
    parser.add_argument("--patch", type=int, nargs=3, metavar=("D", "H", "W"))
    parser.add_argument("--input-channels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or explicit cpu debugging.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--preflight", action="store_true", help="Probe forward/backward/AdamW allocation before training; always on for auto preset.")
    parser.add_argument("--preflight-only", action="store_true", help="Run a real-training-patch allocation probe and stop; uses a NEW --run.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--isotropic-early", action="store_true", help="Use 3x3x3 early kernels instead of 1x3x3; strides unchanged.")
    for name in ("multiscale", "transformer", "adaptive-fusion", "wavelet", "decoder-conditioning",
                 "acquisition-conditioning", "boundary", "deep-supervision"):
        parser.add_argument("--no-" + name, action="store_true")
    parser.add_argument("--spatial-gate", action="store_true", help="Replace wavelets with comparable spatial-only gates.")
    parser.add_argument("--metadata-control", choices=("normal", "constant", "shuffled"), default="normal")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Number of microbatches per epoch, not optimizer updates.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0, help="Safe Windows default; no persistent workers across epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Require deterministic algorithms; unsupported operations fail.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--clip-grad", type=float, default=12.)
    parser.add_argument("--foreground-probability", type=float, default=.5)
    parser.add_argument("--anatomical-background-probability", type=float, default=0.,
                        help="Training tumour-free patches centred on lung-like HU; remainder is uniform sampling.")
    parser.add_argument("--hard-negative-probability", type=float, default=0.)
    parser.add_argument("--hard-negative-index", help="Fingerprint-checked train-only working-grid centres from mine-negatives.")
    parser.add_argument("--lambda-ds", type=float, default=1.)
    parser.add_argument("--lambda-boundary", type=float, default=.1)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--early-stopping-checks", type=int, default=0, help="0 disables; count validation checks, not epochs.")
    parser.add_argument("--overlap", type=float, default=.5)
    parser.add_argument("--threshold", type=float, default=.5, help="Prespecified operating point; optional later validation-only selection.")
    parser.add_argument("--surface-tolerance-mm", type=float, default=2.)
    parser.add_argument("--accumulator-dir", help="Optional scratch disk for CPU sliding-window accumulation arrays.")
    parser.add_argument("--resume", help="Original run/checkpoints/last.pt; scientific settings are restored and validated.")
    parser.add_argument("--overfit-cases", type=int, default=0, choices=(0, 1, 2), help="Actual TRAINING-case diagnostic; not generalization.")
    add_figure_arguments(parser)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect dependencies, CUDA, RAM and storage without training.")
    doctor.add_argument("--path", default=".")
    doctor.add_argument("--output")
    doctor.set_defaults(handler=doctor_command)
    prepare = commands.add_parser("prepare", help="Inspect real paired NIfTI or CT+RTSTRUCT and build a disk-backed cache.")
    inputs = prepare.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-csv")
    inputs.add_argument("--images-dir")
    inputs.add_argument("--dicom-root")
    prepare.add_argument("--masks-dir")
    prepare.add_argument("--unlabeled", action="store_true", help="Directory route with no reference masks.")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--spacing", type=float, nargs=3, default=(2.5, 1., 1.), metavar=("Z", "Y", "X"))
    prepare.add_argument("--boundary-width-mm", type=float, default=2.)
    prepare.add_argument("--roi", default="GTV-1", help="Exact RTSTRUCT ROI name and declared target for paired data.")
    prepare.add_argument("--release", required=True, help="User-supplied source version/release identifier.")
    prepare.add_argument("--source-hash", choices=("sha256", "stat"), default="sha256")
    prepare.add_argument("--qc-cases", type=int, default=6)
    prepare.set_defaults(handler=prepare_command)
    split = commands.add_parser("split", help="Write or validate an immutable patient/source-grouped partition.")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--val-fraction", type=float, default=.2)
    split.add_argument("--test-fraction", type=float, default=.2)
    split.add_argument("--folds", type=int, default=1)
    split.add_argument("--fold", type=int, default=0)
    split.add_argument("--external", action="store_true", help="Place a separate labeled manifest entirely in external.")
    split.set_defaults(handler=split_command)
    selftest = commands.add_parser("self-test", help="Mathematical and synthetic integration checks; no real-data performance claims.")
    selftest.add_argument("--output")
    selftest.add_argument("--available-only", action="store_true", help="Allow missing-dependency skips with PARTIAL status; failures still fail.")
    selftest.set_defaults(handler=self_test_command)
    train = commands.add_parser("train", help="Train exactly one explicit model configuration.")
    add_train_arguments(train)
    train.set_defaults(handler=train_command)
    mining = commands.add_parser("mine-negatives", help="Mine bounded false-positive centres from TRAINING predictions only.")
    mining.add_argument("--manifest", required=True)
    mining.add_argument("--splits", required=True)
    mining.add_argument("--checkpoint", required=True)
    mining.add_argument("--output", required=True)
    mining.add_argument("--device", default="auto")
    mining.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    mining.add_argument("--threshold", type=float, default=.9)
    mining.add_argument("--overlap", type=float, default=.5)
    mining.add_argument("--centres-per-case", type=int, default=256)
    mining.add_argument("--max-cases", type=int, default=0, help="0 means all training cases.")
    mining.add_argument("--seed", type=int, default=42)
    mining.set_defaults(handler=mine_negatives_command)
    for command in ("evaluate", "predict"):
        inference = commands.add_parser(command, help="Complete-volume " + command + " with original-grid outputs.")
        inference.add_argument("--manifest", required=True)
        inference.add_argument("--checkpoint", required=True)
        inference.add_argument("--output", required=True)
        inference.add_argument("--device", default="auto")
        inference.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
        inference.add_argument("--operating-point", help="Frozen JSON from select-threshold on validation; never test-tuned.")
        inference.add_argument("--overlap", type=float, default=.5)
        inference.add_argument("--surface-tolerance-mm", type=float, default=2.)
        inference.add_argument("--accumulator-dir")
        inference.add_argument("--no-save-predictions", action="store_true")
        if command == "evaluate":
            inference.add_argument("--splits", required=True)
            inference.add_argument("--partition", choices=("val", "test", "external"), default="val")
        add_figure_arguments(inference)
        inference.set_defaults(handler=inference_command)
    report = commands.add_parser("report", help="Offline white-background report from existing outputs; never reruns inference.")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--manifest")
    report.add_argument("--splits")
    report.add_argument("--surfaces", action="store_true", help="Optional real-mask 3D surfaces, requires scikit-image and saved predictions.")
    add_figure_arguments(report)
    report.set_defaults(handler=report_command)
    threshold = commands.add_parser("select-threshold", help="Freeze a threshold using saved VAL volume predictions only.")
    threshold.add_argument("--evaluation", required=True)
    threshold.add_argument("--manifest", required=True)
    threshold.add_argument("--splits", help="Verify validation coverage; required for legacy evaluations without case_ids.")
    threshold.add_argument("--output", required=True)
    threshold.add_argument("--thresholds", nargs="+", type=float, default=(.2, .3, .4, .5, .6, .7, .8))
    threshold.add_argument("--postprocess-modes", nargs="+", choices=("none", "largest", "min-volume"),
                           default=("none",), help="Validation-only component recipes to compare jointly with thresholds.")
    threshold.add_argument("--min-volumes-ml", nargs="+", type=float, default=(.1, 1.),
                           help="Physical component cutoffs for min-volume mode; ignored for other modes.")
    threshold.set_defaults(handler=select_threshold_command)
    ablate = commands.add_parser("ablate", help="Run only an explicitly selected sequence of implemented configurations.")
    add_train_arguments(ablate)
    ablate.add_argument("--configs", nargs="+", choices=ABLATIONS, required=True)
    ablate.set_defaults(handler=ablate_command)
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    args._provided = [token.split("=", 1)[0] for token in raw if token.startswith("--")]
    for name in ("figure_cases", "qc_cases"):
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(name.replace("_", "-") + " must be nonnegative")
    if hasattr(args, "dpi") and args.dpi < 50:
        parser.error("--dpi must be >= 50")
    if hasattr(args, "overlap") and not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0,1)")
    if hasattr(args, "seed") and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0,2**32)")
    if hasattr(args, "surface_tolerance_mm") and (not math.isfinite(args.surface_tolerance_mm) or args.surface_tolerance_mm <= 0):
        parser.error("--surface-tolerance-mm must be positive and finite")
    try:
        return args.handler(args) or 0
    except UserError as exc:
        LOGGER.error("%s", exc)
        return 1
    except (FileNotFoundError, PermissionError) as exc:
        LOGGER.error("File access failed: %s", exc)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())







