import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.factorization.factorize import (
    to_low_rank_manual,
    all_same_svals_energy_ratio,
    all_same_rank_ratio,
)
from compress.flops import count_model_flops
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean,
    cifar10_std,
)
import argparse
import json
from compress import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_path", type=str, default="resnet20.pth")
parser.add_argument("--model_name", type=str, default="resnet20")
parser.add_argument("--metric", type=str, default="energy")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_file", type=str, default=None)

args = parser.parse_args()
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.model_name),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

mean = cifar10_mean
std = cifar10_std

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

train_set = datasets.CIFAR10(
    root="data", train=True, transform=transform, download=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
eval_results = evaluate_vision_model(model, test_loader)
n_params = sum(p.numel() for p in model.parameters())
flops = count_model_flops(model, (1, 3, 32, 32))
flops_raw = count_model_flops(model, (1, 3, 32, 32), formatted=False)
print(
    f"Test Loss: {eval_results['loss']}, Test Accuracy: {eval_results['accuracy']}, Number of parameters: {n_params}, Flops: {flops}"
)

results = []

energies = [
    0.3,
    0.35,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9992,
    0.9995,
    0.9997,
    0.9999,
    0.99993,
    0.99995,
    0.99997,
    0.99999,
    1.0,
]

ratios = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


for x in (
    energies if args.metric == "energy" else ratios if args.metric == "rank" else None
):
    cfg = (
        all_same_svals_energy_ratio(
            model,
            energy=x,
        )
        if args.metric == "energy"
        else (
            all_same_rank_ratio(
                model,
                ratio=x,
            )
            if args.metric == "rank"
            else None
        )
    )
    model_lr = to_low_rank_manual(
        model,
        cfg_dict=cfg,
        inplace=False,
    )
    n_params_lr = sum(p.numel() for p in model_lr.parameters())
    model_lr.to(device)
    eval_results = evaluate_vision_model(model_lr, test_loader)

    fl = count_model_flops(model_lr, (1, 3, 32, 32), formatted=True)
    fl2 = count_model_flops(model_lr, (1, 3, 32, 32), formatted=False)
    s = "Energy ratio" if args.metric == "energy" else "Rank ratio"
    print(
        f"{s}: {x:.8f}, Test Loss: {eval_results['loss']:.4f}, Test Accuracy: {eval_results['accuracy']:.4f}, Ratio of parameters: {n_params_lr / n_params:.4f}, Flops: {fl}"
    )
    results.append(
        {
            "loss": eval_results["loss"],
            "accuracy": eval_results["accuracy"],
            "params_ratio": n_params_lr / n_params,
            "flops_ratio": fl2["total"] / flops_raw["total"],
            "metric_value": x,
            "metric_name": args.metric,
        }
    )

filename = (
    args.output_file
    if args.output_file is not None
    else f"factorization_results_{args.model_name}_{args.metric}.json"
)
with open(filename, "w") as f:
    json.dump(results, f, indent=4)
