#!/usr/bin/env python3
"""
Aggregate per-prompt judge verdicts into ASR tables.

Expects per-prompt judge verdicts in this layout:

    results/blackbox_asr/<model>/<judge>/<exp>_<dataset>_detailed.jsonl

where <exp> is `baseline` or `sae` and <dataset> is `salad`, `injections` or `safe_eval`.
Writes `asr_by_file.csv` and `asr_summary.csv` next to the model folders.

Usage:
    python scripts/blackbox/aggregate_asr.py [results/blackbox_asr]
"""

import json
import sys
from pathlib import Path

import pandas as pd

EXP_NAMES = {"baseline": "baseline", "sae": "SAE"}
DATASETS = ("safe_eval", "injections", "salad")


def parse_name(path: Path):
    """`sae_safe_eval_detailed.jsonl` -> ("SAE", "safe_eval")"""
    stem = path.name.removesuffix("_detailed.jsonl")
    exp, dataset = stem.split("_", 1)
    if exp not in EXP_NAMES or dataset not in DATASETS:
        raise ValueError(f"Unexpected file name: {path}")
    return EXP_NAMES[exp], dataset


def count_verdicts(path: Path):
    total = harmful = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "is_harmful" not in obj:
                continue
            total += 1
            harmful += bool(obj["is_harmful"])
    return total, harmful


def main(root: Path):
    rows = []
    for path in sorted(root.glob("*/*/*_detailed.jsonl")):
        model, judge = path.parts[-3], path.parts[-2]
        exp_version, dataset = parse_name(path)
        total, harmful = count_verdicts(path)
        if total == 0:
            continue
        rows.append({
            "model": model,
            "exp_version": exp_version,
            "dataset": dataset,
            "judge": judge,
            "file": str(path.relative_to(root)),
            "n_prompts": total,
            "n_harmful": harmful,
            "ASR": harmful / total,
        })

    keys = ["model", "exp_version", "dataset", "judge"]
    by_file = pd.DataFrame(rows).sort_values(keys + ["file"])
    summary = by_file.groupby(keys, as_index=False).agg(
        n_prompts=("n_prompts", "sum"), n_harmful=("n_harmful", "sum")
    )
    summary["ASR"] = summary["n_harmful"] / summary["n_prompts"]
    summary = summary.sort_values(keys)

    by_file.to_csv(root / "asr_by_file.csv", index=False)
    summary.to_csv(root / "asr_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {root / 'asr_by_file.csv'} and {root / 'asr_summary.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/blackbox_asr"))
