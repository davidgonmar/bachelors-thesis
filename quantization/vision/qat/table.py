#!/usr/bin/env python3
import os
import json
import re
import argparse
import statistics

N_SEEDS = 5


def parse_filename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    m = re.match(r"^([^_]+)(?:_results)?_w(\d+)a(\d+)(?:_s\d+)?$", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="results")
    parser.add_argument("-o", "--output", default="table.tex")
    args = parser.parse_args()

    # Collect all JSON result files
    files = sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".json")
    )

    data = {}
    for path in files:
        info = parse_filename(path)
        if info is None:
            continue
        method, w, a = info
        with open(path) as f:
            runs = json.load(f)
        best_acc = max(runs, key=lambda x: x["accuracy"])["accuracy"]
        data.setdefault((method, w, a), []).append(best_acc)

    # Begin LaTeX table (three centered columns)
    lines = [
        r"\begin{tabular}{ccc}",
        r"\hline",
        r"Type & Bit widths & Accuracy (\% mean $\pm$ std) \\",
        r"\hline",
    ]

    # Sort by method, then numeric W, then numeric A
    for (method, w, a), accs in sorted(
        data.items(), key=lambda x: (x[0][0], int(x[0][1]), int(x[0][2]))
    ):
        if len(accs) != N_SEEDS:
            raise ValueError(
                f"Expected {N_SEEDS} seeds, got {len(accs)} for {method} W{w}A{a}"
            )
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs)

        # One pair of $...$ around \pm, and exactly \\ for the row break
        row = f"{method} & W{w}A{a} & {mean:.2f} $\\pm$ {std:.2f} \\\\"
        lines.append(row)

    # End LaTeX table
    lines += [
        r"\hline",
        r"\end{tabular}",
    ]

    with open(args.output, "w") as out:
        out.write("\n".join(lines))


if __name__ == "__main__":
    main()
