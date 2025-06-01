import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
    evaluate_vision_model,
)
from compress.sparsity.prune import (
    WeightNormInterGroupPruner,
    ActivationNormInterGroupPruner,
    TaylorExpansionInterGroupPruner,
    get_sparsity_information_str,
    get_sparsity_information,
    fuse_bn_conv_sparse_train,
)
from compress.sparsity.recipes import (
    per_output_channel_resnet20_policy_dict,
)
from compress.sparsity.schedulers import get_scheduler
from compress.sparsity.policy import Metric
from compress.layer_fusion import (
    fuse_conv_bn,
    resnet20_fuse_pairs,
)
from compress.sparsity.runner import (
    VisionClassificationModelRunner,
)
from tqdm import tqdm
import json
from compress import seed_everything


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="  Training batches", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Iterative Magnitude Pruning with JSON logging"
    )
    parser.add_argument("--model", type=str, default="resnet20")
    parser.add_argument("--pretrained_path", type=str, default="resnet20.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument("--test_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--target_sparsity", type=float, default=0.05)
    parser.add_argument("--n_iters", type=int, default=12)
    parser.add_argument("--epochs_per_iter", type=int, default=5)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument(
        "--method",
        type=str,
        choices=["norm_weights", "norm_activations", "taylor"],
        default="norm_weights",
        help="Pruning method to use: 'norm_weights', 'norm_activations', or 'taylor'",
    )
    parser.add_argument("--calibration_samples", type=int, default=512)
    parser.add_argument("--calibration_bs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stats_file", type=str, required=True, help="Path to output JSON stats file"
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    trainset = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_test
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
    )
    testloader = DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_workers,
    )

    model = load_vision_model(
        args.model,
        pretrained_path=args.pretrained_path,
        strict=True,
        modifier_before_load=get_cifar10_modifier(args.model),
    )
    model = fuse_conv_bn(
        model, resnet20_fuse_pairs, fuse_impl=fuse_bn_conv_sparse_train
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(args.scheduler)

    stats_log = {"config": vars(args), "iterations": []}

    print("Evaluating baseline...")
    baseline = evaluate_vision_model(model, testloader)
    print(
        f"Baseline -> Accuracy: {baseline['accuracy']:.4f}, Loss: {baseline['loss']:.4f}"
    )
    stats_log["baseline"] = {
        "accuracy": baseline["accuracy"],
        "loss": baseline["loss"],
        "sparsity": get_sparsity_information_str(get_sparsity_information(model)),
    }

    for it in tqdm(range(1, args.n_iters + 1), desc="Pruning Iterations"):
        iter_record = {"iteration": it}
        print(f"\nIteration {it}: evaluating before prune...")
        before_prune = evaluate_vision_model(model, testloader)
        print(
            f"Iteration {it} before prune -> Accuracy: {before_prune['accuracy']:.4f}, Loss: {before_prune['loss']:.4f}"
        )
        iter_record["before_prune"] = {
            "accuracy": before_prune["accuracy"],
            "loss": before_prune["loss"],
        }

        ratio = scheduler(it, args.n_iters, args.target_sparsity)
        policies = per_output_channel_resnet20_policy_dict(
            inter_metric=Metric(name="sparsity_ratio", value=ratio),
        )
        del policies["linear"]
        del policies["conv1"]
        if args.method == "norm_weights":
            pruner = WeightNormInterGroupPruner(
                model,
                policies,
            )
        elif args.method == "norm_activations":
            pruner = ActivationNormInterGroupPruner(
                model,
                policies,
                VisionClassificationModelRunner(
                    model,
                    DataLoader(
                        torch.utils.data.Subset(
                            trainset,
                            torch.randint(
                                0, len(trainset), (args.calibration_samples,)
                            ),
                        ),
                        batch_size=args.batch_size,
                        shuffle=True,
                    ),
                ),
            )
        elif args.method == "taylor":
            pruner = TaylorExpansionInterGroupPruner(
                model,
                policies,
                VisionClassificationModelRunner(
                    model,
                    DataLoader(
                        torch.utils.data.Subset(
                            trainset,
                            torch.randint(
                                0, len(trainset), (args.calibration_samples,)
                            ),
                        ),
                        batch_size=1,  # for correct fisher diag estimation
                        shuffle=True,
                    ),
                ),
                approx="fisher_diag",
            )
        else:
            raise ValueError("Unknown pruning method: {}".format(args.method))

        print(f"Pruning iteration {it} at ratio {ratio:.4f}...")
        model = pruner.prune()
        sparsity_str = get_sparsity_information_str(get_sparsity_information(model))

        print("Evaluating immediately after prune (before finetuning)...")
        post_prune = evaluate_vision_model(model, testloader)
        print(
            f"Iteration {it} after prune -> Accuracy: {post_prune['accuracy']:.4f}, Loss: {post_prune['loss']:.4f}, Sparsity: {sparsity_str}"
        )
        iter_record["after_prune_before_ft"] = {
            "accuracy": post_prune["accuracy"],
            "loss": post_prune["loss"],
            "sparsity": get_sparsity_information(model),
            "target_sparsity_ratio": ratio,
        }

        epochs_stats = []
        epochs_this_iter = (
            args.epochs_per_iter * 2 if it == args.n_iters else args.epochs_per_iter
        )
        for epoch in tqdm(
            range(1, epochs_this_iter + 1),
            desc=f"  Finetune Iter {it}",
            leave=False,
        ):
            loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
            stats = evaluate_vision_model(model, testloader)
            print(
                f"Iteration {it} Epoch {epoch} -> Train Loss: {loss:.4f}, Val Acc: {stats['accuracy']:.4f}, Val Loss: {stats['loss']:.4f}"
            )
            epochs_stats.append(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_accuracy": stats["accuracy"],
                    "val_loss": stats["loss"],
                }
            )
        iter_record["finetune_epochs"] = epochs_stats

        stats_log["iterations"].append(iter_record)

    stats_log["final_sparsity"] = get_sparsity_information(model)

    print("\nFinal evaluation after all iterations...")

    with open(args.stats_file, "w") as f:
        json.dump(stats_log, f, indent=4)
    print(f"Stats written to {args.stats_file}")


if __name__ == "__main__":
    main()
