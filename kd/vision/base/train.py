import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from compress import seed_everything
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.layer_fusion import resnet20_fuse_pairs
from compress.quantization import prepare_for_qat
from compress.quantization.recipes import get_recipe_quant
import argparse
import torch.nn.functional as F
from compress.quantization import separate_params


def _attach_feature_hooks(
    model: nn.Module, layer_names: List[str], store: Dict[str, torch.Tensor]
):
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(
                module.register_forward_hook(
                    lambda _, __, output, key=name: store.__setitem__(key, output)
                )
            )
    return hooks


class Decoder(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        hidden = max(c_in, c_out)
        self.trunk = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head_mean = nn.Conv2d(hidden, c_out, kernel_size=3, padding=1, bias=False)
        self.head_var = nn.Conv2d(hidden, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        features = self.trunk(x)
        mean = self.head_mean(features)
        var = F.softplus(self.head_var(features))
        return mean, var


class LinearDecoder(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        hidden = max(c_in, c_out)
        self.trunk = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.head_mean = nn.Linear(hidden, c_out)
        self.head_var = nn.Linear(hidden, c_out)

    def forward(self, x):
        features = self.trunk(x)
        mean = self.head_mean(features)
        var = F.softplus(self.head_var(features))
        return mean, var


def elementwise_gaussian_nll(
    mean_var, target: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    mean, var = mean_var
    mean, var, target = (
        mean.reshape(mean.shape[0], -1),
        var.reshape(var.shape[0], -1),
        target.reshape(target.shape[0], -1),
    )
    var = var + eps
    logvar = var.log()
    diff = target - mean
    nll = 0.5 * (logvar + (diff.pow(2) / var)).mean(dim=-1)
    return nll.mean()


parser = argparse.ArgumentParser()
parser.add_argument("--nbits", type=int, required=True)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--val_batch_size", type=int, default=256)
parser.add_argument("--pretrained_path", type=str, required=True)
parser.add_argument("--student_batches", type=int, default=1)
parser.add_argument("--decoder_batches", type=int, default=1)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
assert 0.0 <= args.alpha <= 1.0, "alpha must be in [0,1]"
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_common_tf = [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                *_common_tf,
            ]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(_common_tf),
    ),
    batch_size=args.val_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

teacher = load_vision_model(
    "resnet20",
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    model_args={"num_classes": 10},
).to(device)

teacher.eval()

[p.requires_grad_(False) for p in teacher.parameters()]

student_fp = load_vision_model(
    "resnet20",
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    model_args={"num_classes": 10},
)
quant_specs = get_recipe_quant("resnet20")(
    bits_activation=args.nbits,
    bits_weight=args.nbits,
    leave_edge_layers_8_bits=True,
    clip_percentile=0.995,
    symmetric=True,
)
student = prepare_for_qat(
    student_fp,
    specs=quant_specs,
    use_lsq=True,
    data_batch=next(iter(train_loader))[0][:1024].to(device),
    method_args={"online": False},
    fuse_bn_keys=resnet20_fuse_pairs,
).to(device)

layer_names = ["conv1", "layer1", "layer2", "layer3", "linear"]
teacher_feats: Dict[str, torch.Tensor] = {}
student_feats: Dict[str, torch.Tensor] = {}
_teacher_hooks = _attach_feature_hooks(teacher, layer_names, teacher_feats)
_student_hooks = _attach_feature_hooks(student, layer_names, student_feats)

with torch.no_grad():
    sample = next(iter(train_loader))[0][:32].to(device)
    teacher(sample)
    student(sample)

decoders = {}
for name in layer_names:
    s_feat = student_feats[name]
    t_feat = teacher_feats[name]
    if name == "linear":
        # linear decoder, input/output already flat
        s_feat = s_feat.reshape(s_feat.size(0), -1)
        t_feat = t_feat.reshape(t_feat.size(0), -1)
        decoders[name] = LinearDecoder(c_in=s_feat.shape[1], c_out=t_feat.shape[1]).to(
            device
        )
    else:
        decoders[name] = Decoder(c_in=s_feat.shape[1], c_out=t_feat.shape[1]).to(device)

teacher_feats.clear()
student_feats.clear()

criterion_ce = nn.CrossEntropyLoss()
params = separate_params(student)
opt_student = torch.optim.SGD(
    [
        {"params": params["quant_params"], "weight_decay": 0},
        {"params": params["others"], "weight_decay": 5e-4},
    ],
    lr=0.001,
    momentum=0.9,
)

opt_decoder = torch.optim.SGD(
    (p for d in decoders.values() for p in d.parameters()),
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4,
)
sch_student = torch.optim.lr_scheduler.StepLR(opt_student, step_size=80, gamma=0.1)
sch_decoder = torch.optim.lr_scheduler.StepLR(opt_decoder, step_size=80, gamma=0.1)

results: List[dict] = []
cycle_len = args.student_batches + args.decoder_batches

for epoch in range(1, args.epochs + 1):
    student.train()
    [d.train() for d in decoders.values()]
    run_loss = run_ce = run_kd = 0.0
    n_samples = 0

    for batch_idx, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    ):
        phase = batch_idx % cycle_len
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher(images)
        logits_s = student(images)

        feats_s = {
            name: (
                student_feats[name]
                if name != "linear"
                else student_feats[name].reshape(student_feats[name].size(0), -1)
            )
            for name in layer_names
        }
        feats_t = {
            name: (
                teacher_feats[name].detach()
                if name != "linear"
                else teacher_feats[name]
                .detach()
                .reshape(teacher_feats[name].size(0), -1)
            )
            for name in layer_names
        }

        if phase < args.student_batches:
            # no grad for decoders
            for d in decoders.values():
                for p in d.parameters():
                    p.requires_grad_(False)
            ce_loss = criterion_ce(logits_s, labels)
            kd_loss = torch.stack(
                [
                    elementwise_gaussian_nll(
                        decoders[name](feats_s[name]),
                        feats_t[name],
                    )
                    for name in layer_names
                ]
            ).mean()
            total_loss = args.alpha * ce_loss + (1.0 - args.alpha) * kd_loss
            opt_student.zero_grad(set_to_none=True)
            total_loss.backward()
            opt_student.step()
            run_loss += total_loss.item() * images.size(0)
            run_ce += ce_loss.item() * images.size(0)
            run_kd += kd_loss.item() * images.size(0)
            n_samples += images.size(0)
        else:
            for d in decoders.values():
                for p in d.parameters():
                    p.requires_grad_(True)
            kd_detached = torch.stack(
                [
                    elementwise_gaussian_nll(
                        decoders[name](feats_s[name].detach()),
                        feats_t[name],
                    )
                    for name in layer_names
                ]
            ).mean()
            opt_decoder.zero_grad(set_to_none=True)
            kd_detached.backward()
            opt_decoder.step()

        teacher_feats.clear()
        student_feats.clear()

    sch_student.step()
    sch_decoder.step()

    student.eval()
    [d.eval() for d in decoders.values()]
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = student(images).argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100.0 * correct / total

    results.append(
        dict(
            epoch=epoch,
            ce_loss=run_ce / n_samples,
            kd_loss=run_kd / n_samples,
            total_loss=run_loss / n_samples,
            accuracy=acc,
        )
    )
    print(
        f"Epoch {epoch:03d} | acc {acc:5.2f}% | CE {run_ce/n_samples:.4f} | KD {run_kd/n_samples:.4f} | total {run_loss/n_samples:.4f}"
    )


for h in _teacher_hooks + _student_hooks:
    h.remove()

Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Training complete. Log saved to {args.output_path}")
