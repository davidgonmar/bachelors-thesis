#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

Row = Tuple[str, float, float, float]


def _pretty_method(raw: str) -> str:
    mapping = {
        "magnitude_weights": "Weight Magnitude",
        "magnitude_activations": "Activation Magnitude",
        "taylor": "Taylor",
    }
    return mapping.get(raw, raw.replace("_", " ").title())


def _extract_row(json_path: Path) -> Row:
    with json_path.open() as fp:
        data = json.load(fp)

    cfg = data["config"]
    method = _pretty_method(cfg["method"])
    tgt = float(cfg["target_sparsity"]) * 100.0
    pruned_acc = float(data["summary"]["mean_accuracy"])
    std_acc = float(data["summary"]["std_accuracy"])

    return (
        method,
        tgt,
        pruned_acc,
        std_acc,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a sparsity-vs-accuracy plot from pruning JSON stats"
    )
    ap.add_argument("results_dir", help="Directory with *_stats.json files")
    ap.add_argument(
        "--outfile",
        help="Write plot PDF to this path (default: sparsity_vs_accuracy.pdf)",
    )
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    files = sorted(res_dir.glob("*_stats.json"))
    if not files:
        raise SystemExit(f"No *_stats.json files found in {res_dir}")

    rows: List[Row] = [_extract_row(f) for f in files]
    rows.sort(key=lambda r: (round(r[1]), r[0]))

    data = {}
    for method, tgt, acc, std in rows:
        data.setdefault(method, {"tgt": [], "acc": [], "std": []})
        data[method]["tgt"].append(tgt)
        data[method]["acc"].append(acc)
        data[method]["std"].append(std)

    plt.figure(figsize=(8, 4.8))
    for method, vals in data.items():
        x, y, s = vals["tgt"], vals["acc"], vals["std"]
        plt.fill_between(
            x,
            [m - sd for m, sd in zip(y, s)],
            [m + sd for m, sd in zip(y, s)],
            alpha=0.2,
        )
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("Target Sparsity (%)")
    plt.ylabel("Pruned Accuracy (%)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = Path(args.outfile) if args.outfile else Path("sparsity_vs_accuracy.pdf")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
