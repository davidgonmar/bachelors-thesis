import argparse
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.quantization import prepare_for_qat
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.quantization.recipes import get_recipe_quant
from compress.layer_fusion import resnet20_fuse_pairs
from compress import seed_everything


def attach_feature_hooks(model, layer_names, store):
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(
                module.register_forward_hook(
                    lambda _, __, out, key=name: store.__setitem__(key, out)
                )
            )
    return hooks


class Matcher(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        hidden = max(c_in, c_out)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_out, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


parser = argparse.ArgumentParser(description="CIFAR-10 QAT with feature-KD only")
parser.add_argument(
    "--nbits", type=int, required=True, help="bit-width for activations & weights"
)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--val_batch_size", type=int, required=True)
parser.add_argument("--pretrained_path", type=str, required=True)
parser.add_argument("--student_batches", type=int, required=True)
parser.add_argument("--matchers_batches", type=int, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

seed_everything(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_tf = [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                *common_tf,
            ]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "./data", train=False, download=True, transform=transforms.Compose(common_tf)
    ),
    batch_size=args.val_batch_size,
    shuffle=False,
)

teacher = load_vision_model(
    "resnet20",
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    model_args={"num_classes": 10},
).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

stu_raw = load_vision_model(
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
    stu_raw,
    specs=quant_specs,
    use_lsq=True,
    data_batch=next(iter(train_loader))[0][:1024].to(device),
    method_args={"online": False},
    fuse_bn_keys=resnet20_fuse_pairs,
).to(device)

layer_names = ["layer1", "layer2", "layer3"]
teacher_feats, student_feats = {}, {}
teacher_hooks = attach_feature_hooks(teacher, layer_names, teacher_feats)
student_hooks = attach_feature_hooks(student, layer_names, student_feats)

dummy = next(iter(train_loader))[0][:100].to(device)
with torch.no_grad():
    teacher(dummy)
    student(dummy)

estimators = {}
for name in layer_names:
    c_in = student_feats[name].shape[1]
    c_out = teacher_feats[name].shape[1]
    estimators[name] = Matcher(c_in, c_out).to(device)
teacher_feats.clear()
student_feats.clear()

criterion_mse = nn.MSELoss()
optimizer_student = torch.optim.SGD(
    student.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4
)
optimizer_matchers = torch.optim.SGD(
    [p for m in estimators.values() for p in m.parameters()],
    lr=1e-2,
    momentum=0.9,
    weight_decay=5e-4,
)
scheduler_student = torch.optim.lr_scheduler.StepLR(
    optimizer_student, step_size=40, gamma=0.1
)

results = []

for epoch in range(1, args.epochs + 1):
    student.train()
    for m in estimators.values():
        m.train()

    running_loss = 0.0
    cycle_len = args.student_batches + args.matchers_batches

    for batch_idx, (images, _) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    ):
        phase = batch_idx % cycle_len
        images = images.to(device, non_blocking=True)

        teacher_feats.clear()
        student_feats.clear()
        with torch.no_grad():
            teacher(images)
        student(images)

        kd_losses = [
            criterion_mse(estimators[n](student_feats[n]), teacher_feats[n].detach())
            for n in layer_names
        ]
        kd_loss = torch.stack(kd_losses).mean()

        if phase < args.student_batches:
            optimizer_student.zero_grad(set_to_none=True)
            kd_loss.backward()
            optimizer_student.step()
        else:
            optimizer_matchers.zero_grad(set_to_none=True)
            kd_losses_det = [
                criterion_mse(
                    estimators[n](student_feats[n].detach()), teacher_feats[n].detach()
                )
                for n in layer_names
            ]
            torch.stack(kd_losses_det).mean().backward()
            optimizer_matchers.step()

        running_loss += kd_loss.item() * images.size(0)

    scheduler_student.step()

    student.eval()
    for m in estimators.values():
        m.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            preds = student(images).argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100.0 * correct / total

    avg_loss = running_loss / len(train_loader.dataset)
    results.append({"epoch": epoch, "kd_loss": avg_loss, "accuracy": acc})
    print(f"Epoch {epoch}: KD Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

for h in teacher_hooks + student_hooks:
    h.remove()

with open(args.output_path, "w") as f:
    json.dump(results, f, indent=4)
