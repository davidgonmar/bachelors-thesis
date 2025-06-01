import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.quantization import prepare_for_qat, requantize_qat
from compress.quantization.recipes import get_recipe_quant
from compress.layer_fusion import get_fuse_bn_keys
from compress import seed_everything


parser = argparse.ArgumentParser(description="Progressive QAT for CIFAR-10")
parser.add_argument("--method", default="qat", choices=["qat", "lsq"])
parser.add_argument(
    "--bits_list",
    nargs="+",
    type=int,
    help="Sequence of bit-widths to walk through.",
    required=True,
)
parser.add_argument(
    "--epoch_milestones",
    nargs="+",
    type=int,
    help="Epochs at which to switch to the *next* entry in bits_list.",
    required=True,
)
parser.add_argument("--model_name", default="resnet20")
parser.add_argument("--pretrained_path", default="resnet20.pth")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument(
    "--output_path",
    type=str,
    help="Where to save JSON results. Defaults to qat_prog_<bits>.json",
)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_tf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_tf
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

val_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=val_tf
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False,
)

model = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.model_name),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

current_bits = args.bits_list[0]
specs = get_recipe_quant(args.model_name)(
    bits_activation=current_bits,
    bits_weight=current_bits,
    leave_edge_layers_8_bits=True,
    clip_percentile=0.995,
    symmetric=True,
)
model = prepare_for_qat(
    model,
    specs=specs,
    use_lsq=(args.method == "lsq"),
    data_batch=torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            train_dataset, torch.randperm(len(train_dataset))[:1024]
        ),
        batch_size=128,
        shuffle=False,
    ),
    fuse_bn_keys=get_fuse_bn_keys(args.model_name),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)

# basically, we want to train for 100 epochs pre-conditioning on the optimal weights at the previous bit-width
epochs = max(args.epoch_milestones) + 100

scheduler = MultiStepLR(
    optimizer,
    [max(args.epoch_milestones) + 40, max(args.epoch_milestones) + 80],
    gamma=0.1,
)

bits_schedule = list(zip(args.epoch_milestones, args.bits_list[1:]))
bits_schedule.sort()
schedule_ptr = 0
results = []
for epoch in range(epochs):
    if schedule_ptr < len(bits_schedule) and epoch == bits_schedule[schedule_ptr][0]:
        current_bits = bits_schedule[schedule_ptr][1]
        specs = get_recipe_quant(args.model_name)(
            bits_activation=current_bits,
            bits_weight=current_bits,
            leave_edge_layers_8_bits=True,
            clip_percentile=0.995,
            symmetric=True,
        )
        # if lsq, pass data_batch
        if args.method == "lsq":
            data_batch = torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    train_dataset, torch.randperm(len(train_dataset))[:1024]
                ),
                batch_size=128,
                shuffle=False,
            )
            model = requantize_qat(
                model,
                specs=specs,
                data_batch=data_batch,
            )
        else:
            requantize_qat(model, specs=specs)
        print(f"\n[Epoch {epoch}] ': switched to {current_bits}-bit\n")
        schedule_ptr += 1

    model.train()
    running_loss = 0.0
    for imgs, lbls in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{epochs} • Train", leave=False
    ):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{epochs} • Val", leave=False
        ):
            preds = model(imgs.to(device)).argmax(1)
            total += lbls.size(0)
            correct += (preds.cpu() == lbls).sum().item()
    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader.dataset)
    print(
        f"Epoch {epoch+1:3d} | Loss {avg_loss:.4f} | "
        f"Acc {acc:5.2f}% | bits {current_bits}"
    )

    results.append(
        dict(epoch=epoch + 1, loss=avg_loss, accuracy=acc, bits=current_bits)
    )

out_path = Path(
    args.output_path or f"qat_prog_{'-'.join(map(str, args.bits_list))}.json"
)
with out_path.open("w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {out_path.resolve()}")
