#!/usr/bin/env python3
import os
import re
import json
import argparse
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 16,  # base font size
        "axes.labelsize": 16,  # x & y labels
        "xtick.labelsize": 14,  # x tick labels
        "ytick.labelsize": 14,  # y tick labels
        "legend.fontsize": 14,  # legend text
        "lines.linewidth": 2,  # thicker plot lines
    }
)


def find_json_files(directory, pattern):
    prog = re.compile(pattern)
    return sorted(
        os.path.join(directory, fn) for fn in os.listdir(directory) if prog.match(fn)
    )


def load_full_curve(path):
    with open(path) as f:
        runs = json.load(f)
    if not runs:
        raise RuntimeError(f"{path!r} is empty")
    runs_sorted = sorted(runs, key=lambda r: r["epoch"])
    epochs = [r["epoch"] for r in runs_sorted]
    accs = [r["accuracy"] for r in runs_sorted]
    return epochs, accs, runs_sorted


def average_curves(curves):
    epochs = curves[0][0]
    mean_accs = []
    for i in range(len(epochs)):
        vals = [accs[i] for (_, accs, __) in curves]
        mean_accs.append(statistics.mean(vals))
    return epochs, mean_accs


def detect_bit_switches(runs_sorted):
    switches = []
    prev = runs_sorted[0].get("bits", None)
    for rec in runs_sorted[1:]:
        curr = rec.get("bits", None)
        if curr is not None and curr != prev:
            switches.append((rec["epoch"], f"{prev}→{curr}"))
            prev = curr
    return switches


def main():
    p = argparse.ArgumentParser(
        description="Compare static LSQ (2-bit) vs full progressive LSQ (8→4→2)"
    )
    p.add_argument(
        "--prog_dir",
        required=True,
        help="dir of progressive JSONs (lsq_p8-4-2_s*.json)",
    )
    p.add_argument(
        "--static_dir", required=True, help="dir of static    JSONs (lsq_w2a2_s*.json)"
    )
    p.add_argument(
        "--out_pdf", default="lsq_compare_full.pdf", help="where to save the PDF plot"
    )
    args = p.parse_args()

    prog_files = find_json_files(args.prog_dir, r"lsq_p8-4-2_s\d+\.json")
    static_files = find_json_files(args.static_dir, r"lsq_w2a2_s\d+\.json")
    if not prog_files or not static_files:
        raise RuntimeError("Missing JSON files in one of the directories")

    # load curves
    prog_curves = []
    for fp in prog_files:
        ep, ac, _ = load_full_curve(fp)
        prog_curves.append((ep, ac, None))
    static_curves = []
    for fs in static_files:
        ep, ac, _ = load_full_curve(fs)
        static_curves.append((ep, ac, None))

    # detect switches from first prog file
    _, _, runs0 = load_full_curve(prog_files[0])
    bit_switches = detect_bit_switches(runs0)

    # average across seeds
    prog_epochs, prog_acc = average_curves(prog_curves)
    static_epochs, static_acc = average_curves(static_curves)

    # shift static so its last epoch aligns with prog's last epoch
    shift = prog_epochs[-1] - static_epochs[-1]
    static_epochs = [e + shift for e in static_epochs]

    # wide figure with bigger fonts
    plt.figure(figsize=(12, 4))

    # plot both curves
    plt.plot(prog_epochs, prog_acc, label="progressive LSQ (8→4→2)")
    plt.plot(static_epochs, static_acc, label="static    LSQ (2-bit)")

    # vertical black lines at each bit-switch
    y_max = max(prog_acc) * 1.01
    for epoch, label in bit_switches:
        plt.axvline(epoch, color="black", linestyle="--")
        plt.text(
            epoch + 0.5,
            y_max,
            label,
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=18,
            color="black",
        )

    # black grid lines
    plt.grid(True, color="black", linestyle=":", linewidth=0.5)
    plt.xlim(1, 150)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(args.out_pdf, bbox_inches="tight")
    print(f"✅ Saved plot to {Path(args.out_pdf).absolute()}")


if __name__ == "__main__":
    main()
