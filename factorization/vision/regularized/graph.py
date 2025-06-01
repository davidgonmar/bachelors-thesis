#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt


def load_curve(json_path: Path, x_key: str):
    with json_path.open() as f:
        data = json.load(f)
    data = sorted(data, key=lambda d: d[x_key])
    xs = [item[x_key] for item in data]
    ys = [item["accuracy"] * 100 for item in data]
    return xs, ys


def weight_from_filename(path: Path) -> str:
    match = re.search(r"_([0-9]*\.?[0-9]+)(?=_[^_]+\.json$)", path.name)
    return match.group(1) if match else path.stem


def plot_curves(json_paths, x_key, x_label, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for jp in json_paths:
        xs, ys = load_curve(jp, x_key)
        weight = weight_from_filename(jp)
        ax.plot(xs, ys, label=f"Weight {weight}")
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=14)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot FLOPs Ratio vs Accuracy and Parameter Ratio vs Accuracy from JSONs"
    )
    parser.add_argument(
        "--flops-jsons",
        nargs="+",
        required=True,
        metavar="FILE",
        help="List of JSON result files containing 'flops_ratio' + accuracy",
    )
    parser.add_argument(
        "--params-jsons",
        nargs="+",
        required=True,
        metavar="FILE",
        help="List of JSON result files containing 'params_ratio' + accuracy",
    )
    parser.add_argument(
        "--output-flops",
        default="flops_vs_acc.pdf",
        help="Output filename for FLOPs vs Accuracy plot",
    )
    parser.add_argument(
        "--output-params",
        default="params_vs_acc.pdf",
        help="Output filename for Params vs Accuracy plot",
    )
    args = parser.parse_args()

    flops_paths = [Path(p) for p in args.flops_jsons]
    params_paths = [Path(p) for p in args.params_jsons]

    # Plot FLOPs ratio curves
    plot_curves(
        flops_paths,
        x_key="flops_ratio",
        x_label="FLOPs Ratio",
        out_path=args.output_flops,
    )

    # Plot Parameter count curves
    plot_curves(
        params_paths,
        x_key="params_ratio",
        x_label="Parameter Count Ratio",
        out_path=args.output_params,
    )

    print(f"Saved plots → {args.output_flops}, {args.output_params}")


if __name__ == "__main__":
    main()
