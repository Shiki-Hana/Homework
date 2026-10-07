"""Lab 01: three Linear layers for MNIST. Run: python train_mnist.py

The official test set is evaluated only after selecting the checkpoint by
validation loss. Paths default to this script's directory, independent of cwd.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST


class ThreeLayerMLP(nn.Module):
    """784 -> 256 -> 128 -> 10; exactly three trainable Linear layers."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        # CrossEntropyLoss expects raw logits; do not apply Softmax here.
        return self.layers(x)


def dataset(root, train, download):
    # Use torchvision's HTTPS mirror directly: the original HTTP host can stall.
    # torchvision still verifies the official MD5 checksums of all four files.
    MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
    raw = MNIST(str(root), train=train, download=download)
    # Same operation as ToTensor() followed by Normalize((0.1307,), (0.3081,)).
    # Precompute once to avoid repeatedly transforming the same small images.
    x = raw.data.unsqueeze(1).float().div_(255).sub_(0.1307).div_(0.3081)
    return TensorDataset(x, raw.targets.long())


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        count += y.numel()
        confusion += torch.bincount((10*y+pred).cpu(), minlength=100).reshape(10, 10)
    return {"loss": loss_sum/count, "correct": correct, "count": count,
            "accuracy": correct/count, "confusion_matrix": confusion.tolist()}


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base/"data")
    parser.add_argument("--output-dir", type=Path, default=base/"results")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model without retraining")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    model = ThreeLayerMLP().to(device)
    checkpoint = out/"best_model.pt"
    if args.evaluate:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        test_loader = DataLoader(dataset(args.data_dir, False, not args.no_download),
                                 batch_size=512, shuffle=False, num_workers=0)
        result = evaluate(model, test_loader, device)
        (out/"evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k != "confusion_matrix"}, indent=2))
        return
    # Avoid accidentally overwriting a finished experimental record.
    if checkpoint.exists():
        parser.error("Output already contains a model. Use --evaluate or a new --output-dir.")
    log_file = (out/"training_log.txt").open("w", encoding="utf-8")
    def log(message):
        print(message, flush=True)
        log_file.write(message+"\n")
        log_file.flush()
    config = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "optimizer": "Adam", "weight_decay": 0.0,
        "loss": "CrossEntropyLoss", "architecture": [784,256,128,10],
        "hidden_activation": "ReLU", "normalization_mean": 0.1307,
        "normalization_std": 0.3081, "train_size": 55000, "validation_size": 5000,
        "test_size": 10000, "selection": "minimum validation loss",
        "scheduler": "StepLR(step_size=4, gamma=0.5)", "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": np.__version__,
        "platform": platform.platform(), "num_workers": 0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    }
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    full = dataset(args.data_dir, True, not args.no_download)
    train, val = random_split(full, [55000,5000], generator=torch.Generator().manual_seed(args.seed))
    (out/"split_indices.json").write_text(json.dumps({"train":train.indices,"validation":val.indices}), encoding="utf-8")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_epoch, history = float("inf"), None, []
    log(f"Device: {config['device_name']}; parameters: {config['parameters']}")
    start = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        lr = optimizer.param_groups[0]["lr"]
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*y.numel()
            correct += (logits.argmax(1)==y).sum().item()
            total += y.numel()
        validation = evaluate(model,val_loader,device)
        row = {"epoch":epoch,"learning_rate":lr,"train_loss":loss_sum/total,
               "train_accuracy":correct/total,"validation_loss":validation["loss"],
               "validation_accuracy":validation["accuracy"]}
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save({"state_dict":model.state_dict(),"epoch":epoch,
                        "validation_loss":best_loss,"architecture":[784,256,128,10]}, checkpoint)
        with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        log(f"Epoch {epoch:02d}/{args.epochs}: train loss={row['train_loss']:.6f}, "
            f"train accuracy={row['train_accuracy']:.4%}, val loss={validation['loss']:.6f}, "
            f"val accuracy={validation['accuracy']:.4%}")
        scheduler.step()
    training_seconds = time.perf_counter()-start
    saved = torch.load(checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(saved["state_dict"])
    # Test data has not been used for optimization, checkpoint selection or tuning.
    test_loader = DataLoader(dataset(args.data_dir,False,not args.no_download),
                             batch_size=512,shuffle=False,num_workers=0)
    result = evaluate(model,test_loader,device)
    result.update({"best_epoch":best_epoch,"best_validation_loss":best_loss,
                   "training_seconds":training_seconds,
                   "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    (out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (out/"confusion_matrix.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["true/predicted"]+list(range(10)))
        writer.writerows([[i]+r for i,r in enumerate(result["confusion_matrix"])])
    log(f"Best epoch: {best_epoch}; test: {result['correct']}/{result['count']} "
        f"= {result['accuracy']:.4%}; test loss={result['loss']:.6f}")
    log(f"Training and validation time: {training_seconds:.2f} seconds")
    log_file.close()


if __name__ == "__main__":
    main()
