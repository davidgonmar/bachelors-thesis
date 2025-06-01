import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from compress.quantization import prepare_for_qat
from compress.layer_fusion import get_fuse_bn_keys
import argparse
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.quantization.recipes import get_recipe_quant
import json
from compress import seed_everything


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 QAT Training")
parser.add_argument("--method", default="qat", type=str, choices=["qat", "lsq"])
parser.add_argument("--nbits_activations", default=2, type=int)
parser.add_argument("--nbits_weights", default=2, type=int)
parser.add_argument("--model_name", default="resnet20", type=str)
parser.add_argument("--pretrained_path", default="resnet20.pth", type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument(
    "--output_path", default=None, type=str, help="Where to save the JSON results"
)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)
train_dataset = datasets.CIFAR10(
    "./data", train=True, download=True, transform=data_transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
val_dataset = datasets.CIFAR10(
    "./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    ),
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)

model = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.model_name),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

specs = get_recipe_quant(args.model_name)(
    bits_activation=args.nbits_activations,
    bits_weight=args.nbits_weights,
    leave_edge_layers_8_bits=True,
    clip_percentile=0.995,
    symmetric=True,
)

# gather 1024 samples for calibration
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

epochs = 100
scheduler = MultiStepLR(
    optimizer,
    milestones=[40, 80],
    gamma=0.1,
)

results = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch+1} Training", leave=False
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Epoch {epoch+1} Val", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")
    results.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})

# determine output file
out_path = (
    args.output_path
    or f"qat_results_w{args.nbits_weights}a{args.nbits_activations}.json"
)
with open(out_path, "w") as f:
    json.dump(results, f, indent=4)
