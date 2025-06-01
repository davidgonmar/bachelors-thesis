import argparse
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.factorization.regularizers import (
    SingularValuesRegularizer,
    extract_weights_and_reshapers,
)
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune ResNet on CIFAR-10 with Hoyer regularization and configurable hyperparameters"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--step_size", type=int, default=80)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default="resnet20")
    parser.add_argument("--pretrained_path", type=str, default="resnet20.pth")
    parser.add_argument(
        "--save_path", type=str, default="cifar10_resnet20_hoyer_finetuned.pth"
    )
    parser.add_argument("--reg_weight", type=float, default=0.005)
    parser.add_argument(
        "--log_path", type=str, required=True, help="Path to save training log as JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(0)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=val_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_vision_model(
        args.model_name,
        pretrained_path=args.pretrained_path,
        strict=True,
        modifier_before_load=get_cifar10_modifier(args.model_name),
        modifier_after_load=None,
        model_args={"num_classes": 10},
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    params_and_reshapers = extract_weights_and_reshapers(
        model, cls_list=(torch.nn.Conv2d, torch.nn.Linear), keywords={"weight"}
    )
    regularizer = SingularValuesRegularizer(
        metric="squared_hoyer_sparsity",
        params_and_reshapers=params_and_reshapers,
        weights=1.0,
        normalize=False,
    )

    log_data = []

    for epoch in range(args.epochs):
        model.train()
        train_loss, reg_loss = 0.0, 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            reg = regularizer()
            total_loss = loss + args.reg_weight * reg
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            reg_loss += reg.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        reg_loss /= len(train_loader.dataset)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Reg Loss: {reg_loss:.4f}"
        )
        print(f"Learning Rate after epoch {epoch+1}: {current_lr:.5f}")

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                val_loss += criterion(y_hat, y).item() * x.size(0)
                preds = y_hat.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_loss /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {acc:.4f}")

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "reg_loss": reg_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "learning_rate": current_lr,
        }
        log_data.append(epoch_log)

        torch.save(model, args.save_path)

    with open(args.log_path, "w") as f:
        json.dump(log_data, f, indent=2)


if __name__ == "__main__":
    main()
