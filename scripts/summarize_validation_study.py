"""Summarize the matched validation study. No training, inference, or test data."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lung_ct_segmentation as pipeline


def paired_dice(old_rows, new_rows, repeats=10000, seed=42):
    def grouped(rows):
        keys, patients = set(), {}
        for row in rows:
            key = (row["case_id"], row["patient_id"])
            if key in keys or row["status"] != "success":
                raise ValueError("Paired comparisons require unique successful cases.")
            keys.add(key)
            patients.setdefault(row["patient_id"], []).append(float(row["dice"]))
        return keys, {patient: float(np.mean(values)) for patient, values in patients.items()}
    old_keys, old = grouped(old_rows)
    new_keys, new = grouped(new_rows)
    if not old or old_keys != new_keys or set(old) != set(new):
        raise ValueError("Paired validation case/patient sets differ or are empty.")
    difference = np.asarray([new[p] - old[p] for p in sorted(old)])
    rng = np.random.default_rng(seed)
    means = [rng.choice(difference, len(difference), replace=True).mean() for _ in range(repeats)]
    return {"patients": len(difference), "mean_delta": float(difference.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "improved": int((difference > 0).sum()), "worsened": int((difference < 0).sum()),
            "tied": int((difference == 0).sum()), "bootstrap_repeats": repeats, "seed": seed,
            "scope": "Conditional paired patient bootstrap after validation selection; does not include selection uncertainty or establish held-out improvement."}


def load(path):
    return pipeline.load_json(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="evaluations/validation_study_20261107")
    args = parser.parse_args(argv)
    study = Path(args.study)
    raw_roots = [Path("evaluations/hybrid_val"), Path("evaluations/corrected_single_val")]
    selected_roots = [study / "previous_threshold_only", study / "corrected_threshold_only"]
    for roots in (raw_roots, selected_roots):
        compatible, reason = pipeline.compatible_comparisons(roots)
        if not compatible or any(protocol["partition"] != "val" for _, protocol, _ in compatible):
            raise ValueError("Not a compatible validation comparison: " + reason)
    groups = []
    for name, roots in (("common_0_5", raw_roots), ("selected_threshold_none", selected_roots)):
        rows = [load(root / "patient_metrics.json") for root in roots]
        aggregates = [load(root / "aggregates.json") for root in roots]
        groups.append({"policy": name, "sources": [str(p) for p in roots],
                       "previous": aggregates[0], "corrected": aggregates[1],
                       "paired_dice": paired_dice(*rows)})
    stage1 = {name: load(path) for name, path in {
        "previous": "evaluations/hybrid_threshold_matched_20261107.json",
        "corrected": "evaluations/corrected_single_threshold_20261107.json"}.items()}
    stage2 = {name: load(path) for name, path in {
        "previous": "evaluations/hybrid_operating_point_matched_20261107.json",
        "corrected": "evaluations/corrected_single_operating_point_20261107.json"}.items()}
    result = {"created": pipeline.now(), "partition": "val", "training_performed": False,
              "comparisons": groups, "threshold_study": stage1, "component_study": stage2,
              "failure_review": load("outputs/validation_study/failure_review.json"),
              "decision": {
                  "largest_component": "Not recommended as a default: corrected mean Dice rises but zero-overlap failures increase to 23/83.",
                  "threshold_0_98_none": "Mean-Dice-selected development operating point; 7 zero-overlap failures versus 5 at 0.5. Not a clinically validated recommendation.",
                  "miss_preserving_alternative": "Corrected threshold 0.7/none gives Dice 0.408334 with 5 zero-overlap cases in the investigated grid; a descriptive trade-off, not a separately validated selection.",
                  "next": "Inspect missed-small-target images and training-only small-target exposure before another training change; do not tune on the test set."},
              "limitations": ["Same validation patients selected checkpoints and operating points; tuning optimism applies.",
                              "Staged component study is not an exhaustive joint threshold/recipe search.",
                              "Higher mean Dice can conceal more completely missed tumours."]}
    output = study / "comparison.json"
    report = study / "comparison.md"
    if output.exists() or report.exists():
        raise ValueError("Comparison output already exists; preserve it and choose a fresh study.")
    pipeline.atomic_json(output, result)
    lines = ["# Matched validation operating-point study", "",
             "83 validation patients only. No retraining, new inference, or test-set selection.", "",
             "| Configuration | Previous mean Dice | Corrected mean Dice |",
             "| --- | ---: | ---: |"]
    for group in groups:
        lines.append(f"| {group['policy']} | {group['previous']['metrics']['dice']['mean']:.6f} | {group['corrected']['metrics']['dice']['mean']:.6f} |")
    lines += ["", "## Full-volume metrics at the independently selected threshold (0.98), no component removal", "",
              "| Metric | Previous | Corrected |", "| --- | ---: | ---: |"]
    selected = groups[1]
    for metric in ("dice", "precision", "recall", "hd95_mm", "surface_dice"):
        lines.append(f"| {metric} | {selected['previous']['metrics'][metric]['mean']:.6f} | {selected['corrected']['metrics'][metric]['mean']:.6f} |")
    paired = selected["paired_dice"]
    lines += ["", f"Paired mean Dice difference: {paired['mean_delta']:+.6f}; conditional patient-bootstrap 95% CI {paired['ci95']}.",
              f"Patients improved/worsened/tied: {paired['improved']}/{paired['worsened']}/{paired['tied']}.", "",
              "## Corrected-model operating-point trade-offs", "",
              "| Configuration | Mean Dice | Zero-overlap cases |", "| --- | ---: | ---: |"]
    default = groups[0]["corrected"]
    lines.append(f"| threshold 0.5, none | {default['metrics']['dice']['mean']:.6f} | {default['zero_overlap_cases']} |")
    for candidate in stage2["corrected"]["candidates"]:
        recipe = candidate["postprocessing"]
        label = recipe["mode"] + (f" {recipe['min_volume_ml']}mL" if recipe["mode"] == "min-volume" else "")
        lines.append(f"| threshold {candidate['threshold']}, {label} | {candidate['mean_dice']:.6f} | {candidate['zero_overlap_cases']} |")
    lines += ["", "Do not adopt largest-component filtering merely for its higher average Dice: it misses 23/83 targets.",
              "The original five missed targets all have maximum target probability below 0.2; higher thresholds or component removal cannot recover them.",
              "The diagnostic montage is in outputs/validation_study/zero_overlap_montage.png.", "",
              "These are development-selected scores, not held-out improvements or clinical accuracy. The paired interval is conditional on selected checkpoints and operating points.",
              "No new training or test evaluation was performed. See comparison.json and plan.json for complete provenance and the staged search policy.", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparison": str(output), "report": str(report), "paired_threshold_only": paired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
