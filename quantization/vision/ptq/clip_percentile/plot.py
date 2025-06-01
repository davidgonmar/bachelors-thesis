import json
import argparse
import matplotlib.pyplot as plt


def render_plot(results, output_path=None):
    # give a much wider canvas: 12″ wide by 4″ tall (3:1)
    fig, ax = plt.subplots(figsize=(8, 4))

    data = {}
    for row in results:
        typ = row.get("type", "")
        if "_P" not in typ:
            continue
        acc = row.get("acc_mean", 0.0)
        bits_config, p_str = typ.rsplit("_P", 1)
        try:
            p = float(p_str)
        except ValueError:
            continue
        data.setdefault(bits_config, []).append((p, acc))

    for bits, vals in data.items():
        vals.sort(key=lambda x: x[0])
        ps = [v[0] for v in vals]
        accs = [v[1] for v in vals]
        ax.plot(ps, accs, label=bits)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    fig.tight_layout()

    if output_path:
        # ensure the PDF page is exactly your fig size, no extra padding
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--output_path")
    args = parser.parse_args()

    with open(args.json_path) as f:
        results = json.load(f)

    render_plot(results, args.output_path)


if __name__ == "__main__":
    main()
