#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_last_accuracy(stats_path: Path):
    with stats_path.open() as f:
        data = json.load(f)
    iterations = data.get("iterations", [])
    if not iterations:
        raise ValueError(f"No iterations found in {stats_path}")
    last = iterations[-1]
    ftes = last.get("finetune_epochs", [])
    if ftes:
        return ftes[-1]["val_accuracy"]
    return last["after_prune_before_ft"]["accuracy"]


def main():
    parser = argparse.ArgumentParser(
        description="Gather last accuracies from multiple pruning results and output a LaTeX table."
    )
    parser.add_argument(
        "--results-dirs",
        nargs=2,
        type=Path,
        required=True,
        metavar="DIR",
        help="Two directories containing *_stats.json",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .tex file path"
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["taylor", "magnitude_weights", "magnitude_activations"],
        help="List of pruning methods to include",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.output

    headers = [d.name for d in args.results_dirs]
    methods = args.methods

    table = {}
    for method in methods:
        row = []
        for res_dir in args.results_dirs:
            stats_file = res_dir / f"{method}_stats.json"
            if not stats_file.exists():
                raise FileNotFoundError(f"Stats file not found: {stats_file}")
            acc = load_last_accuracy(stats_file)
            row.append(acc)
        table[method] = row

    with output_path.open("w") as tex:
        tex.write("\\begin{tabular}{l" + "r" * len(headers) + "}\n")
        tex.write("\\toprule\n")
        tex.write("Method & " + " & ".join(headers) + " \\\\ \n")
        tex.write("\\midrule\n")
        pretty = {
            "taylor": "Taylor",
            "magnitude_weights": "Weights Magnitude",
            "magnitude_activations": "Activations Magnitude",
        }
        for method in methods:
            name = pretty.get(method, method)
            vals_str = [f"{v:.4f}" for v in table[method]]
            tex.write(f"{name} & " + " & ".join(vals_str) + " \\\\ \n")
        tex.write("\\bottomrule\n")
        tex.write("\\end{tabular}\n")

    print(f"LaTeX table saved to: {output_path}")


if __name__ == "__main__":
    main()
