import os
import json
import re
import argparse
import statistics

N_SEEDS = 5  # number of seeds per alpha


def parse_filename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    # Expect names like train_s{seed}_a{alpha}.json
    m = re.match(r"^train_s(\d+)_a([0-9]+\.?[0-9]*)$", name)
    if not m:
        return None
    seed, alpha = m.group(1), m.group(2)
    return seed, alpha


def main():
    parser = argparse.ArgumentParser(description="Render LaTeX table for QAT results")
    parser.add_argument(
        "-d", "--dir", default="results", help="Directory with JSON result files"
    )
    parser.add_argument("-o", "--output", default="table.tex", help="Output LaTeX file")
    args = parser.parse_args()

    files = sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".json")
    )

    data = {}
    for path in files:
        info = parse_filename(path)
        if info is None:
            continue
        seed, alpha = info
        with open(path) as f:
            runs = json.load(f)
        # take best accuracy over epochs
        best_acc = max(runs, key=lambda x: x["accuracy"])["accuracy"]
        data.setdefault(alpha, []).append(best_acc)

    lines = [
        r"\begin{tabular}{cc}",
        r"\hline",
        r"Alpha & Accuracy (\% mean $\pm$ std) \\",
        r"\hline",
    ]

    for alpha, accs in sorted(data.items(), key=lambda x: float(x[0])):
        if len(accs) != N_SEEDS:
            raise ValueError(
                f"Expected {N_SEEDS} seeds for alpha={alpha}, but got {len(accs)}"
            )
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs)
        row = f"{alpha} & {mean:.2f} $\\pm$ {std:.2f} \\\\"
        lines.append(row)

    lines += [
        r"\hline",
        r"\end{tabular}",
    ]

    with open(args.output, "w") as out:
        out.write("\n".join(lines))


if __name__ == "__main__":
    main()
