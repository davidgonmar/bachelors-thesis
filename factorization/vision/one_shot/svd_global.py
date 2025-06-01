import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.factorization.factorize import to_low_rank_global
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
from compress.layer_fusion import resnet20_fuse_pairs
from compress.utils import get_all_convs_and_linears


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_path", type=str, default="cifar10_resnet20_hoyer_finetuned.pth"
)
parser.add_argument("--model_name", type=str, default="resnet20")
parser.add_argument(
    "--output_file", type=str, default="global_factorization_results_resnet20.json"
)
parser.add_argument("--metric", type=str, default="flops")
parser.add_argument("--seed", type=int, default=0)
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

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

train_set = datasets.CIFAR10(
    root="data", train=True, transform=transform, download=True
)
subset_train_set = torch.utils.data.Subset(
    train_set, torch.randint(0, len(train_set), (10000,))
)
train_loader = DataLoader(subset_train_set, batch_size=100, shuffle=True)

eval_results = evaluate_vision_model(model, test_loader)
n_params_orig = sum(p.numel() for p in model.parameters())
flops_orig = count_model_flops(model, (1, 3, 32, 32), formatted=False)
print(
    f"[original] Loss: {eval_results['loss']:.4f}, Acc: {eval_results['accuracy']:.4f}, Params: {n_params_orig}, FLOPs: {flops_orig}"
)

results = []

keys = get_all_convs_and_linears(model)

ratios = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
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
    1.0,
]

for ratio in ratios:
    model_lr = to_low_rank_global(
        model,
        sample_input=torch.randn(1, 3, 32, 32).float().cuda(),
        ratio_to_keep=ratio,
        bn_keys=resnet20_fuse_pairs,
        inplace=False,
        keys=keys,
        metric=args.metric,
    )
    n_params_lr = sum(p.numel() for p in model_lr.parameters())
    flops_formatted = count_model_flops(model_lr, (1, 3, 32, 32))
    flops_raw = count_model_flops(model_lr, (1, 3, 32, 32), formatted=False)
    eval_lr = evaluate_vision_model(model_lr.to(device), test_loader)
    print(
        f"[ratio={ratio:.2f}] Loss: {eval_lr['loss']:.4f}, Acc: {eval_lr['accuracy']:.4f}, Param‐keep: {n_params_lr/n_params_orig:.4f}, FLOPs: {flops_formatted}"
    )
    results.append(
        {
            "metric_value": ratio,
            "loss": eval_lr["loss"],
            "accuracy": eval_lr["accuracy"],
            "params_ratio": n_params_lr / n_params_orig,
            "flops_ratio": flops_raw["total"] / flops_orig["total"],
            "metric_name": args.metric,
        }
    )

with open(args.output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"All results written to {args.output_file}")
