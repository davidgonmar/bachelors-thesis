import json
import os
import argparse

REG_WEIGHTS = [0.001, 0.002, 0.003, 0.004, 0.005]


def load_accuracy(log_path):
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
        return data[-1]["accuracy"]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tabular from logs.")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory with log JSON files."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save LaTeX table."
    )
    args = parser.parse_args()

    rows = []
    for weight in REG_WEIGHTS:
        log_file = f"training_log_{weight}.json"
        log_path = os.path.join(args.results_dir, log_file)
        acc = load_accuracy(log_path)
        rows.append((weight, acc if acc is not None else "N/A"))

    with open(args.output, "w") as f:
        f.write("\\begin{tabular}{cc}\n")
        f.write("\\toprule\n")
        f.write("Regularization Weight & Accuracy \\\\\n")
        f.write("\\midrule\n")
        for weight, acc in rows:
            if isinstance(acc, float):
                f.write(f"{weight} & {acc:.4f} \\\\\n")
            else:
                f.write(f"{weight} & N/A \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


if __name__ == "__main__":
    main()
