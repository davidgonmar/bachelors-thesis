#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot training run results and save to PDF"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to JSON result file")
    parser.add_argument(
        "-o", "--output", default="training_plot.pdf", help="Output PDF file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")

    with open(args.input) as f:
        runs = json.load(f)

    epochs = []
    accuracies = []

    for idx, entry in enumerate(runs):
        acc = entry.get("accuracy")
        if acc is not None:
            accuracies.append(acc)
            epochs.append(entry.get("epoch", idx + 1))

    if not epochs or not accuracies:
        raise ValueError("No valid accuracy data found to plot.")

    plt.figure(figsize=(14, 7))
    plt.plot(
        epochs,
        accuracies,
        marker="o",
        linestyle="-",
        linewidth=3,
        color="tab:blue",
        label="Accuracy",
    )

    plt.title("Training Accuracy over Epochs", fontsize=20)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.8)
    plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Saved training plot to {args.output}")


if __name__ == "__main__":
    main()
