import argparse
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.flops import count_model_flops
from compress import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--step_size", type=int, default=150)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="resnet20")
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, required=True)
    return parser.parse_args()


def build_loaders(bs, workers, pin):
    train_t = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    val_t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_t)
    val_ds = datasets.CIFAR10("data", train=False, download=True, transform=val_t)
    train_ld = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin
    )
    val_ld = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin
    )
    return train_ld, val_ld


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def main():
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = build_loaders(
        args.batch_size, args.workers, args.pin_memory
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model(
        args.model_name,
        pretrained_path=args.pretrained_path if args.pretrained_path else None,
        strict=True,
        modifier_before_load=get_cifar10_modifier(args.model_name),
        modifier_after_load=None,
        model_args={"num_classes": 10},
        accept_model_directly=True,
    ).to(device)

    # constant model metrics
    flops_raw = count_model_flops(model, input_size=(1, 3, 32, 32), formatted=False)
    n_params = sum(p.numel() for p in model.parameters())

    print("flops:", flops_raw)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # initialise root log object
    log = {"nparams": n_params, "flops": flops_raw, "train": []}

    # baseline evaluation (epoch 0)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    log["train"].append(
        {
            "epoch": 0,
            "train_loss": None,
            "val_loss": val_loss,
            "accuracy": val_acc,
            "learning_rate": args.lr,
        }
    )

    best_acc = val_acc
    # training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
        train_loss = loss_sum / len(train_loader.dataset)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}, train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if args.save_path and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

        log["train"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": val_acc,
                "learning_rate": lr_now,
            }
        )

    # write JSON
    with open(args.log_path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
