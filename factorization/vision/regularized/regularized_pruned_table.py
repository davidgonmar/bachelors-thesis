#!/usr/bin/env python3
import argparse
import pathlib
import json
import torch
import re
from compress.experiments import load_vision_model, get_cifar10_modifier
from compress.flops import count_model_flops

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR / "results" / "finetuned"

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=str(DEFAULT_DIR))
parser.add_argument("--out", default=None)
args = parser.parse_args()

results_dir = pathlib.Path(args.dir)
if not results_dir.is_absolute():
    results_dir = (SCRIPT_DIR / results_dir).resolve()

out_path = pathlib.Path(args.out) if args.out else results_dir / "table.tex"
out_path.parent.mkdir(parents=True, exist_ok=True)


# Load base model for normalization
def load_base():
    return load_vision_model(
        "resnet20",
        pretrained_path=None,
        strict=False,
        modifier_before_load=get_cifar10_modifier("resnet20"),
        modifier_after_load=None,
        model_args={"num_classes": 10},
        accept_model_directly=True,
    ).cuda()


base = load_base()
base_params = sum(p.numel() for p in base.parameters())
base_flops = count_model_flops(base, input_size=(1, 3, 32, 32), formatted=False)[
    "total"
]

# Containers for two categories
flop_rows = []
param_rows = []

# Gather metrics
for fp in results_dir.glob("*.json"):
    with fp.open() as f:
        data = json.load(f)
    stem = fp.stem
    acc = max(epoch["accuracy"] for epoch in data["train"])
    params_ratio = data["nparams"] / base_params
    flops_ratio = data["flops"]["total"] / base_flops

    # Classify by filename and extract target ratio
    if "flops" in stem:
        m = re.search(r"regularized_([\d\.]+)_flops_([\d\.]+)", stem)
        name = f"Weight {m.group(1)} – FLOPs target {m.group(2)}" if m else stem
        flop_rows.append((name, acc, params_ratio, flops_ratio))
    elif "params" in stem:
        m = re.search(r"regularized_([\d\.]+)_params_([\d\.]+)", stem)
        name = f"Weight {m.group(1)} – Params target {m.group(2)}" if m else stem
        param_rows.append((name, acc, params_ratio, flops_ratio))
    else:
        continue

# Sort by target ratio descending: flops by measured flops_ratio, params by measured params_ratio
flop_rows.sort(key=lambda x: x[3], reverse=True)
param_rows.sort(key=lambda x: x[2], reverse=True)


# Helper to build LaTeX table
def build_table(rows, col4_label):
    lines = [
        r"\begin{tabular}{lccc}",
        rf"Model & Accuracy & Params ratio & {col4_label}\\",  # two backslashes in the output
        r"\hline",
    ]
    for name, acc, pr, fr in rows:
        lines.append(rf"{name} & {acc:.4f} & {pr:.3f} & {fr:.3f}\\")  # ditto
    lines.append(r"\end{tabular}\n")
    return lines


# Generate tables (both show params and flops ratios)
flops_table = build_table(flop_rows, "FLOPs ratio")
params_table = build_table(param_rows, "FLOPs ratio")

# Write both tables into the output file
content = []
content.append("% Table: Models optimized by FLOPs")
content.extend(flops_table)
content.append("% Table: Models optimized by Params")
content.extend(params_table)

out_path.write_text("\n".join(content))
