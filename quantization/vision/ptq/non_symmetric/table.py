import json
import argparse


def render_table(results):
    lines = [
        "\\begin{tabular}{l c}",
        "\\toprule",
        "Type & Accuracy (\\%) \\\\",
        "\\midrule",
    ]
    for row in results:
        typ = row.get("type", "")
        mean = row.get("acc_mean", 0.0)
        std = row.get("acc_std", 0.0)
        lines.append(f"{typ} & {mean:.2f}\\pm{std:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--output_path")
    args = parser.parse_args()

    with open(args.json_path) as f:
        results = json.load(f)

    table = render_table(results)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(table)
    else:
        print(table)


if __name__ == "__main__":
    main()
