#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def _load_curve(stats_path: Path):
    with stats_path.open() as f:
        data = json.load(f)

    baseline_acc = data["baseline"]["accuracy"]
    prune_pcts = []
    prune_accs = []
    ft_accs = []

    for it in data["iterations"]:
        s = it["after_prune_before_ft"]["target_sparsity_ratio"]
        prune_pcts.append(s)
        prune_accs.append(it["after_prune_before_ft"]["accuracy"])

        ftes = it.get("finetune_epochs", [])
        if ftes:
            ft_accs.append(ftes[-1]["val_accuracy"])
        else:
            ft_accs.append(prune_accs[-1])

    return baseline_acc, prune_pcts, prune_accs, ft_accs


def main():
    p = argparse.ArgumentParser(description="Plot prune→fine‐tune with sparsity labels")
    p.add_argument("results_dir", type=Path, help="Dir containing *_stats.json")
    p.add_argument("output_pdf", type=Path, help="Path to save the PDF")
    args = p.parse_args()

    methods = ["taylor", "norm_weights", "norm_activations"]
    markers = {"taylor": "o", "norm_weights": "s", "norm_activations": "^"}
    pretty = {
        "taylor": "Taylor sensitivity",
        "norm_weights": "Weights Norm",
        "norm_activations": "Activations Norm",
    }

    first = args.results_dir / f"{methods[0]}_stats.json"
    if not first.exists():
        print(f"[ERROR] {first} not found")
        return

    baseline, prune_pcts, prune_accs, ft_accs = _load_curve(first)

    x = list(range(1 + 2 * len(prune_pcts)))
    labels = ["baseline"]
    for pct in prune_pcts:
        labels += [f"p {pct:.3f}", f"ft {pct:.3f}"]

    plt.figure(figsize=(10, 5))
    for m in methods:
        f = args.results_dir / f"{m}_stats.json"
        if not f.exists():
            print(f"[Warning] {f} missing, skipping {m}")
            continue

        b, pcts, pr_accs, ft_accs = _load_curve(f)
        y = [b]
        for pr, ft in zip(pr_accs, ft_accs):
            y += [pr, ft]

        plt.plot(x, y, label=pretty[m], marker=markers[m])

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Cycle steps (prune → fine‐tune)")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved → {args.output_pdf}")


if __name__ == "__main__":
    main()
