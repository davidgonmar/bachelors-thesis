#!/usr/bin/env python3
import os
import re
import json
import argparse
import statistics
from pathlib import Path

N_SEEDS = 5


def parse_filename(path):
    """
    Expect filenames like:
       qat_p8-4-2_s0.json
       lsq_p8-4_s3.json
    Returns (method, [bit1, bit2, ...]) or None.
    """
    name = Path(path).stem
    m = re.match(r"^([^_]+)_p([0-9]+(?:-[0-9]+)*)_s\d+$", name)
    if not m:
        return None
    method = m.group(1)
    bits_seq = m.group(2).split("-")
    return method, bits_seq


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from progressive QAT JSON results"
    )
    parser.add_argument(
        "-d", "--dir", default="results", help="Directory containing the JSON files"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="progressive_table.tex",
        help="Where to write the .tex table",
    )
    args = parser.parse_args()

    # collect all .json files
    files = sorted(
        [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".json")]
    )

    data = {}
    for path in files:
        info = parse_filename(path)
        if info is None:
            continue
        method, bits_seq = info
        with open(path) as f:
            runs = json.load(f)

        # only look at entries recorded at the final (target) bit width
        target_b = int(bits_seq[-1])
        accs_at_target = [r["accuracy"] for r in runs if r.get("bits") == target_b]
        if not accs_at_target:
            raise ValueError(f"No entries at bits={target_b} in {path}")
        best_acc = max(accs_at_target)

        key = (method, tuple(int(b) for b in bits_seq))
        data.setdefault(key, []).append(best_acc)

    # start LaTeX table
    lines = [
        r"\begin{tabular}{ccc}",
        r"\hline",
        r"Method & Bit schedule & Accuracy (\% mean $\pm$ std) \\",
        r"\hline",
    ]

    # sort by method, then by the numeric sequence of bits
    def sort_key(item):
        (method, bits_seq_tuple), _ = item
        return (method, bits_seq_tuple)

    for (method, bits_seq), accs in sorted(data.items(), key=sort_key):
        if len(accs) != N_SEEDS:
            raise ValueError(
                f"Expected {N_SEEDS} seeds for {method} p{'-'.join(map(str,bits_seq))}, got {len(accs)}"
            )
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs)

        # render schedule with LaTeX arrow in math mode
        bits_str = "$" + " \\rightarrow ".join(str(b) for b in bits_seq) + "$"
        row = f"{method} & {bits_str} & {mean:.2f} $\\pm$ {std:.2f} \\\\"
        lines.append(row)

    lines += [
        r"\hline",
        r"\end{tabular}",
    ]

    with open(args.output, "w") as out:
        out.write("\n".join(lines))

    print(f"Wrote table to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
