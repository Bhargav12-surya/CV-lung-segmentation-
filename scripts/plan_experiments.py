"""Prepare a controlled experiment plan and an opt-in single-run launcher.

Generation never starts training, downloads data, or changes the patient split.
The core pipeline remains lung_ct_segmentation.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lung_ct_segmentation as pipeline


VARIANTS = {
    "baseline": ("baseline3d", ()),
    "full": ("hybrid3d", ()),
    "no_wavelet": ("hybrid3d", ("--no-wavelet",)),
    "no_adaptive_fusion": ("hybrid3d", ("--no-adaptive-fusion",)),
    "no_acquisition": ("hybrid3d", ("--no-acquisition-conditioning",)),
}
REMOVAL_FIELDS = {"no_wavelet": "wavelet", "no_adaptive_fusion": "adaptive_fusion",
                  "no_acquisition": "acquisition_conditioning"}


def powershell_quote(value):
    """Use literal single-quoted tokens; never evaluate a generated command string."""
    return "'" + str(value).replace("'", "''") + "'"


def build_plan(reference, manifest, splits, run_root, seeds=(42, 43, 44),
               variants=tuple(VARIANTS), preset=None, overrides=None,
               count_parameters=True, python_executable=None, sampling="mixed"):
    """Return parser-validated configurations without opening dataset arrays."""
    seeds, variants, overrides = list(seeds), list(variants), overrides or {}
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unique integers in [0, 2**32).")
    if not variants or len(set(variants)) != len(variants) or any(name not in VARIANTS for name in variants):
        raise ValueError("Choose each supported configuration at most once.")
    original = reference.get("train_args", {})
    preset = preset or reference.get("preset", "compact")
    if preset not in pipeline.PRESETS:
        raise ValueError("Use a named fixed preset, not auto, for a controlled comparison.")
    if sampling not in ("legacy", "mixed"):
        raise ValueError("Sampling must be legacy or mixed.")
    shared = {
        "manifest": str(manifest), "splits": str(splits), "preset": preset,
        "input_channels": 2, "precision": "auto", "device": "auto", "workers": 0,
        "max_epochs": original.get("max_epochs", 100),
        "steps_per_epoch": original.get("steps_per_epoch", 1000),
        "batch_size": original.get("batch_size", 1),
        "accumulation": original.get("accumulation", 2),
        "lr": original.get("lr", 5e-5), "weight_decay": original.get("weight_decay", 1e-4),
        "warmup_epochs": original.get("warmup_epochs", 5),
        "clip_grad": original.get("clip_grad", 12.),
        "foreground_probability": original.get("foreground_probability", .25),
        "lambda_ds": original.get("lambda_ds", 1.),
        "lambda_boundary": original.get("lambda_boundary", .1),
        "val_every": original.get("val_every", 10),
        # A fixed planned update budget avoids model-dependent early termination.
        "early_stopping_checks": 0, "overlap": original.get("overlap", .5),
        "threshold": .5, "surface_tolerance_mm": original.get("surface_tolerance_mm", 2.),
    }
    if sampling == "mixed":
        # An explicit development hypothesis, not a validation-selected optimum.
        shared.update({"foreground_probability": .5,
                       "anatomical_background_probability": .4,
                       "hard_negative_probability": 0.})
    else:
        shared.update({"anatomical_background_probability": 0., "hard_negative_probability": 0.})
    shared.update(overrides)
    parser = pipeline.build_parser()
    parameter_counts, configs, experiments = {}, {}, []
    for variant in variants:
        model_name, removal_flags = VARIANTS[variant]
        for seed in seeds:
            run = Path(run_root) / f"{variant}_seed{seed}"
            argv = ["train", "--run", str(run), "--model", model_name, "--seed", str(seed), "--preflight"]
            for key, value in shared.items():
                flag = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        argv.append(flag)
                elif value is not None:
                    argv.append(flag)
                    if isinstance(value, (list, tuple)):
                        argv.extend(str(v) for v in value)
                    else:
                        argv.append(str(value))
            argv.extend(removal_flags)
            args = parser.parse_args(argv)
            pipeline.validate_train_args(args)
            model = pipeline.model_config_from_args(args, preset)
            configs[variant] = pipeline.clean_json(dataclasses.asdict(model))
            patch = tuple(args.patch or pipeline.PRESETS[preset]["patch"])
            # All shipped presets obey the exact encoder stride lattice. Odd-grid
            # mathematical support does not make an arbitrary patch a measured recipe.
            if any(n <= 0 or n % stride for n, stride in zip(patch, (8, 16, 16))):
                raise ValueError("Controlled patches must be positive multiples of (8,16,16).")
            if count_parameters and variant not in parameter_counts:
                pipeline.require_torch()
                probe = pipeline.SegmentationModel(model)
                parameter_counts[variant] = sum(p.numel() for p in probe.parameters())
                del probe
            steps = math.ceil(args.steps_per_epoch / args.accumulation) * args.max_epochs
            experiments.append({
                "name": f"{variant}_seed{seed}", "variant": variant, "seed": seed,
                "run": str(run), "argv": argv, "model_config": configs[variant],
                "patch": list(patch), "parameters": parameter_counts.get(variant),
                "planned_microbatches": args.max_epochs * args.steps_per_epoch,
                "planned_optimizer_updates": steps,
                "selection_criterion": "mean per-patient full-volume validation Dice at threshold 0.5",
                "normalization": "fresh training-only acquisition statistics; never resume historical weights",
                "validation_output": str(Path("evaluations") / Path(run_root).name / f"{variant}_seed{seed}_val"),
                "status": "planned_not_run",
            })
    # Validate every leave-one-out against the full model even if only a subset
    # of experiments was requested.
    full_args = parser.parse_args(["train", "--manifest", str(manifest), "--splits", str(splits),
                                   "--run", "unused", "--model", "hybrid3d", "--preset", preset])
    for key in ("gradient_checkpointing", "isotropic_early", "input_channels"):
        if key in shared:
            setattr(full_args, key, shared[key])
    full_config = pipeline.clean_json(dataclasses.asdict(pipeline.model_config_from_args(full_args, preset)))
    for name, field in REMOVAL_FIELDS.items():
        if name in configs:
            changed = {key for key in full_config if configs[name][key] != full_config[key]}
            if changed != {field}:
                raise ValueError(f"{name} changed unexpected model settings: {sorted(changed)}")
    return {
        "schema": "msf-hsat-controlled-plan-v1", "created": pipeline.now(),
        "program_version": pipeline.VERSION,
        "python_executable": python_executable or sys.executable,
        "pipeline": str(ROOT / "lung_ct_segmentation.py"), "workspace": str(ROOT),
        "manifest": str(manifest), "splits": str(splits), "run_root": str(run_root),
        "seeds": seeds, "shared_train_args": shared, "sampling_policy": sampling,
        "note": "Plans only. Compare on development data; no suite is launched by generation.",
        "experiments": experiments,
    }


def write_plan(plan, output):
    """Write one immutable plan directory. A launcher runs only an explicit name."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"Plan output already exists; choose a new --output: {output}")
    collisions = [e["run"] for e in plan["experiments"] if Path(e["run"]).exists()]
    if collisions:
        raise ValueError(f"Planned run directories already exist: {collisions}")
    output.mkdir(parents=True)
    pipeline.atomic_json(output / "plan.json", plan)
    for experiment in plan["experiments"]:
        pipeline.atomic_json(output / "configs" / f"{experiment['name']}.json", experiment)
    lines = ["# Generated plan: running without -Experiment lists names and does not train.",
             "param([string]$Experiment)", "$ErrorActionPreference = 'Stop'",
             "$Commands = @{"]
    for experiment in plan["experiments"]:
        tokens = [plan["pipeline"], *experiment["argv"]]
        lines.append("  " + powershell_quote(experiment["name"]) + " = @(" +
                     ", ".join(powershell_quote(token) for token in tokens) + ")")
    lines.extend(["}", "if (-not $Experiment) { $Commands.Keys | Sort-Object; return }",
                  "if (-not $Commands.ContainsKey($Experiment)) { throw 'Choose a listed experiment name.' }",
                  "Set-Location -LiteralPath " + powershell_quote(plan["workspace"]),
                  "$Python = " + powershell_quote(plan["python_executable"]),
                  "$RunArgs = $Commands[$Experiment]", "& $Python @RunArgs", "exit $LASTEXITCODE", ""])
    (output / "run_one.ps1").write_text("\n".join(lines), encoding="utf-8")
    commands = ["# Copy and run one command deliberately. No automatic suite execution."]
    for experiment in plan["experiments"]:
        tokens = [plan["python_executable"], plan["pipeline"], *experiment["argv"]]
        commands.extend(["", "# " + experiment["name"],
                         "& " + " ".join(powershell_quote(token) for token in tokens)])
    (output / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", default="runs/hybrid_compact_v2_seed42")
    parser.add_argument("--manifest")
    parser.add_argument("--splits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", default="runs/controlled_v3")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--configs", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--preset", choices=tuple(pipeline.PRESETS))
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--foreground-probability", type=float)
    parser.add_argument("--sampling", choices=("legacy", "mixed"), default="mixed",
                        help="mixed proposes 50%% foreground, 40%% anatomical, 10%% uniform; legacy keeps the reference foreground fraction.")
    parser.add_argument("--anatomical-background-probability", type=float)
    parser.add_argument("--hard-negative-probability", type=float)
    parser.add_argument("--hard-negative-index")
    parser.add_argument("--patch", nargs=3, type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-parameter-counts", action="store_true")
    args = parser.parse_args(argv)
    reference_path = Path(args.reference_run) / "resolved_config.json"
    reference = pipeline.load_json(reference_path)
    train_args = reference["train_args"]
    manifest_path = args.manifest or train_args["manifest"]
    split_path = args.splits or train_args["splits"]
    manifest, _ = pipeline.load_manifest(manifest_path)
    split = pipeline.validate_split(manifest, pipeline.load_json(split_path))
    overrides = {key: getattr(args, key) for key in ("max_epochs", "steps_per_epoch", "batch_size",
                                                   "accumulation", "foreground_probability",
                                                   "anatomical_background_probability", "hard_negative_probability",
                                                   "hard_negative_index", "patch")
                 if getattr(args, key) is not None}
    if args.gradient_checkpointing:
        overrides["gradient_checkpointing"] = True
    plan = build_plan(reference, manifest_path, split_path, args.run_root, args.seeds, args.configs,
                      args.preset, overrides, not args.skip_parameter_counts, sampling=args.sampling)
    plan.update({"reference_config": str(reference_path),
                 "reference_config_sha256": pipeline.file_digest(reference_path),
                 "source_sha256": pipeline.file_digest(ROOT / "lung_ct_segmentation.py"),
                 "manifest_fingerprint": manifest["fingerprint"], "split_fingerprint": split["fingerprint"],
                 "patient_partition_counts": {k: len(v) for k, v in split["partitions"].items()}})
    write_plan(plan, args.output)
    print(f"Prepared {len(plan['experiments'])} configurations; no training started: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
