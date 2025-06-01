#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compress.experiments import load_vision_model, get_cifar10_modifier
from compress.utils import get_all_convs_and_linears


def compute_layerwise_energy_curves(
    model: torch.nn.Module, layers: list[str]
) -> list[tuple[str, np.ndarray]]:
    named_modules = dict(model.named_modules())
    results: list[tuple[str, np.ndarray]] = []
    for key in layers:
        layer = named_modules[key]
        weight = layer.weight.data
        if weight.ndim == 4:
            out_c, in_c, k_h, k_w = weight.shape
            weight_2d = weight.reshape(out_c, in_c * k_h * k_w)
        else:
            weight_2d = weight
        with torch.no_grad():
            s = torch.linalg.svdvals(weight_2d)
        energy = s.pow(2).cumsum(0) / s.pow(2).sum()
        energy = np.insert(energy.cpu().numpy(), 0, 0.0)
        results.append((key, energy))
    return results


def plot_energy_collage(
    layer_energies: list[tuple[str, np.ndarray]], output_path: Path
):
    # Use larger font size for titles and labels
    plt.rcParams.update({"font.size": 16})

    n = len(layer_energies)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for idx in range(rows * cols):
        if idx < n:
            name, energy = layer_energies[idx]
            ax = axes[idx]
            ax.plot(range(len(energy)), energy, marker="o", markersize=2, linewidth=1)
            ax.set_title(name, fontsize=24)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, len(energy) - 1)
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.grid(True, linestyle="--", linewidth=0.3)
        else:
            axes[idx].remove()
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300)
    print(f"Saved collage to {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=Path, required=True)
    parser.add_argument(
        "--output_plot", type=Path, default=Path("sv_energy_selected_layers.pdf")
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = load_vision_model(
        "resnet20",
        pretrained_path=str(args.pretrained_path),
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet20"),
        modifier_after_load=None,
        model_args={"num_classes": 10},
    ).to(device)
    model.eval()
    layers_to_plot = [
        "conv1",
        "layer1.0.conv1",
        "layer1.2.conv2",
        "layer2.1.conv2",
        "layer3.0.conv1",
        "linear",
    ]
    layer_energies = compute_layerwise_energy_curves(model, layers_to_plot)
    plot_energy_collage(layer_energies, args.output_plot)


if __name__ == "__main__":
    main()
