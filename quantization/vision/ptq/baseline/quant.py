import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.quantization import to_quantized_offline, get_activations_vision
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean as mean,
    cifar10_std as std,
)
from compress.quantization.recipes import get_recipe_quant, get_quant_keys
from compress.quantization import ConvWeightsPerOutChannel, LinearWeightsPerOutChannel
from compress.layer_fusion import get_fuse_bn_keys
import argparse
from itertools import product
import json
import numpy as np
from compress import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--pretrained_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--output_path", type=str, default="quantization_results.json")
parser.add_argument("--percentile", type=float, default=0.995)
parser.add_argument("--train_subset_size", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--runs", type=int, default=5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def single_run(seed):
    seed_everything(seed)
    model = load_vision_model(
        args.model_name,
        pretrained_path=args.pretrained_path,
        strict=True,
        modifier_before_load=get_cifar10_modifier(args.model_name),
        modifier_after_load=None,
        model_args={"num_classes": 10},
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    indices = torch.randperm(len(train_dataset))[: args.train_subset_size]
    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, indices),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    recipe_fn = get_recipe_quant(args.model_name)
    fuse_bn_keys = get_fuse_bn_keys(args.model_name)
    quant_keys = get_quant_keys(args.model_name)

    with torch.no_grad():
        activations = get_activations_vision(model, train_loader, keys=quant_keys)
        orig = evaluate_vision_model(model, test_loader)

    data = {"original": {"loss": [orig["loss"]], "accuracy": [orig["accuracy"]]}}
    bit_widths = [2, 4, 8]
    for w_bits, a_bits in product(bit_widths, bit_widths):
        specs = recipe_fn(
            bits_activation=a_bits,
            bits_weight=w_bits,
            clip_percentile=args.percentile,
            leave_edge_layers_8_bits=True,
            symmetric=True,
            linear_weight_grouper=LinearWeightsPerOutChannel(),
            conv_weight_grouper=ConvWeightsPerOutChannel(),
        )
        quanted = to_quantized_offline(
            model,
            specs,
            activations=activations,
            inplace=False,
            fuse_bn_keys=fuse_bn_keys,
        )
        quanted.to(device)
        qr = evaluate_vision_model(quanted, test_loader)
        key = f"W{w_bits}A{a_bits}"
        data.setdefault(key, {"loss": [], "accuracy": []})
        data[key]["loss"].append(qr["loss"])
        data[key]["accuracy"].append(qr["accuracy"])
    return data


all_data = {}
for i in range(args.runs):
    seed = args.seed + i
    run_data = single_run(seed)
    for k, v in run_data.items():
        if k not in all_data:
            all_data[k] = {"loss": [], "accuracy": []}
        all_data[k]["loss"].extend(v["loss"])
        all_data[k]["accuracy"].extend(v["accuracy"])

results = []
for k, v in all_data.items():
    loss_arr = np.array(v["loss"])
    acc_arr = np.array(v["accuracy"]) * 100
    results.append(
        {
            "type": k,
            "loss_mean": float(loss_arr.mean()),
            "loss_std": float(loss_arr.std()),
            "acc_mean": float(acc_arr.mean()),
            "acc_std": float(acc_arr.std()),
        }
    )

with open(args.output_path, "w") as f:
    json.dump(results, f, indent=4)
