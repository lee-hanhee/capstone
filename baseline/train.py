"""Train and evaluate a small, reproducible multi-label transfer-learning baseline."""

import argparse
import csv
import json
import platform
from pathlib import Path
import random
import time

from huggingface_hub import model_info
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
import timm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from baseline.data import CLASSES
from baseline.model import MODEL_ID, SignClassifier
from baseline.thresholds import select_thresholds


def tensor(image, mean, std):
    return TF.normalize(TF.to_tensor(image.convert("RGB")), mean, std)


class Frames(Dataset):
    def __init__(self, root, rows, mean, std):
        self.root, self.rows, self.mean, self.std = root, rows, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            x = tensor(image, self.mean, self.std)
        return x, torch.tensor(row["labels"], dtype=torch.float32)


def metrics(labels, probabilities, thresholds=0.5):
    predicted = (probabilities >= np.asarray(thresholds)).astype(int)
    result = {"frames": len(labels),
              "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
              "label_accuracy": float((labels == predicted).mean()),
              "exact_match_accuracy": float((labels == predicted).all(axis=1).mean()),
              "per_class": {}}
    for j, label in enumerate(CLASSES):
        y, p = labels[:, j], predicted[:, j]
        both = len(np.unique(y)) == 2
        result["per_class"][label] = {
            "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
            "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "average_precision": float(average_precision_score(y, probabilities[:, j])) if both else None,
            "roc_auc": float(roc_auc_score(y, probabilities[:, j])) if both else None,
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, p, labels=[0, 1]).ravel().tolist(),
        }
    y_none = (labels == 0).all(axis=1)
    p_none = (predicted == 0).all(axis=1)
    result["irrelevant"] = {
        "accuracy": float(accuracy_score(y_none, p_none)),
        "precision": float(precision_score(y_none, p_none, zero_division=0)),
        "recall": float(recall_score(y_none, p_none, zero_division=0)),
        "f1": float(f1_score(y_none, p_none, zero_division=0)),
    }
    return result


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    truth, probs, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        truth.append(y.cpu().numpy())
        probs.append(logits.sigmoid().cpu().numpy())
    truth, probs = np.concatenate(truth), np.concatenate(probs)
    return loss_sum / len(truth), metrics(truth, probs), truth, probs


@torch.inference_mode()
def latency(model, device, size=224):
    model.to(device).eval()
    x = torch.zeros(1, 3, size, size, device=device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    for _ in range(20):
        model(x).sigmoid()
    sync()
    elapsed = []
    for _ in range(100):
        sync()
        start = time.perf_counter()
        model(x).sigmoid()
        sync()
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(elapsed)), "median_ms": float(np.median(elapsed)),
            "p95_ms": float(np.percentile(elapsed, 95)), "warmup": 20, "repeats": 100,
            "batch_size": 1, "includes": "model forward pass and sigmoid only",
            "excludes": "decode, preprocessing, camera, network, speech, cold start",
            "device": str(device), "is_ios_measurement": False}


def save_predictions(path, rows, labels, probabilities):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "source_video", "frame_index"] +
                        [f"true_{c}" for c in CLASSES] + [f"prob_{c}" for c in CLASSES])
        for row, target, probability in zip(rows, labels, probabilities):
            writer.writerow([row["image"], row["source_video"], row["frame_index"]] +
                            target.astype(int).tolist() + probability.tolist())


def finish_evaluation(model, loaders, subsets, device, criterion, output, checkpoint,
                      tune_thresholds=True):
    # Select thresholds before running test inference; test labels cannot affect them.
    _, val, val_y, val_probs = evaluate(model, loaders["val"], device, criterion)
    save_predictions(output / "validation_predictions.csv", subsets["val"], val_y, val_probs)
    thresholds = select_thresholds(val_y, val_probs) if tune_thresholds else [0.5] * 3
    config = dict(checkpoint["config"], thresholds=thresholds)
    checkpoint["config"] = config
    torch.save(checkpoint, output / "best.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "thresholds.json").write_text(json.dumps({
        "classes": CLASSES, "thresholds": thresholds,
        "selection": "per-class validation F1; nearest 0.5 then larger threshold on ties"
                     if tune_thresholds else "fixed 0.5",
    }, indent=2) + "\n")
    _, test, y, probs = evaluate(model, loaders["test"], device, criterion)
    save_predictions(output / "test_predictions.csv", subsets["test"], y, probs)
    return {"validation": val, "test": test,
            "selected_thresholds": dict(zip(CLASSES, thresholds)),
            "validation_selected_thresholds": metrics(val_y, val_probs, thresholds),
            "test_selected_thresholds": metrics(y, probs, thresholds)}, y, probs


def train(data_dir, output, epochs=10, max_seconds=1500, pretrained=True, device_name=None,
          settings=None):
    start = time.monotonic()
    settings = settings or {}
    seed = settings.get("seed", 42)
    epochs = settings.get("epochs", epochs)
    frozen_epochs = settings.get("frozen_epochs", 3)
    batch_size = settings.get("batch_size", 32)
    head_lr = settings.get("head_learning_rate", 1e-3)
    finetune_lr = settings.get("finetune_learning_rate", 1e-4)
    weight_decay = settings.get("weight_decay", 1e-4)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["classes"] != CLASSES:
        raise ValueError("Unexpected class order")
    size = manifest["image_size"]
    if settings.get("image_size", size) != size:
        raise ValueError("Config image size differs from prepared data")
    if settings.get("model_id", MODEL_ID) != MODEL_ID:
        raise ValueError("This simple trainer supports MobileNetV3-Small only")
    for key, expected in {"optimizer": "AdamW", "loss": "weighted_bce",
                          "augmentation": "none",
                          "checkpoint_selection": "validation_macro_f1_at_0.5"}.items():
        if settings.get(key, expected) != expected:
            raise ValueError(f"Unsupported {key}: {settings[key]}")
    if settings.get("threshold_selection", "fixed_0.5") not in {"fixed_0.5", "per_class_validation_f1"}:
        raise ValueError("Unsupported threshold selection")
    subsets = {s: [r for r in manifest["rows"] if r["split"] == s]
               for s in ["train", "val", "test"]}
    video_sets = {s: {r["source_video"] for r in rows} for s, rows in subsets.items()}
    if any(video_sets[a] & video_sets[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise ValueError("Source video leakage")
    for s, rows in subsets.items():
        y = np.array([r["labels"] for r in rows])
        if not len(rows) or np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(rows)):
            raise ValueError(f"Missing positive/negative class examples: {s}")

    revision = (settings.get("hf_revision") or model_info(MODEL_ID).sha) if pretrained else None
    model = SignClassifier(pretrained=pretrained, revision=revision).to(device)
    cfg = timm.data.resolve_model_data_config(model.backbone)
    mean, std = list(cfg["mean"]), list(cfg["std"])
    loaders = {s: DataLoader(Frames(data_dir, rows, mean, std), batch_size=batch_size,
                             shuffle=(s == "train"), num_workers=0)
               for s, rows in subsets.items()}
    train_y = np.array([r["labels"] for r in subsets["train"]])
    pos_weight = (len(train_y) - train_y.sum(0)) / train_y.sum(0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    config = {"model_id": MODEL_ID, "hf_revision": revision, "pretrained": pretrained,
              "classes": CLASSES, "seed": seed, "epochs_requested": epochs,
              "frozen_epochs": min(frozen_epochs, epochs), "batch_size": batch_size,
              "head_learning_rate": head_lr, "finetune_learning_rate": finetune_lr,
              "optimizer": "AdamW", "weight_decay": weight_decay,
              "thresholds": [0.5] * 3, "mean": mean, "std": std,
              "image_size": size, "pos_weight": pos_weight.tolist(),
              "selection": "highest validation macro F1; earliest epoch on ties",
              "max_seconds": max_seconds, "device": str(device),
              "torch": str(torch.__version__), "timm": timm.__version__,
              "python": platform.python_version(), "platform": platform.platform(),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "counts": manifest["counts"], "annotation_sha256": manifest["annotation_sha256"]}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=head_lr, weight_decay=weight_decay)
    best, history = -1.0, []
    for epoch in range(epochs):
        if epoch and time.monotonic() - start >= max_seconds:
            break
        if epoch == frozen_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)
        model.train()
        if epoch < frozen_epochs:
            model.backbone.eval()  # Freeze running batch-normalization statistics too.
        train_loss = 0.0
        for x, y in loaders["train"]:
            if time.monotonic() - start >= max_seconds:
                raise TimeoutError("Training budget timer expired; no partial epoch is reported")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        val_loss, val, _, _ = evaluate(model, loaders["val"], device, criterion)
        row = {"epoch": epoch + 1, "phase": "heads" if epoch < frozen_epochs else "finetune",
               "train_loss": train_loss / len(train_y), "val_loss": val_loss,
               "val_macro_f1": val["macro_f1"], "val_label_accuracy": val["label_accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "best_epoch": epoch + 1, "validation_metrics": val}, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    evaluation, y, probs = finish_evaluation(
        model, loaders, subsets, device, criterion, output, checkpoint,
        tune_thresholds=settings.get("threshold_selection") == "per_class_validation_f1")
    majority = (train_y.mean(0) >= 0.5).astype(float)
    baselines = {"always_absent": metrics(y, np.zeros_like(probs)),
                 "train_majority": metrics(y, np.tile(majority, (len(y), 1)))}
    device_latency = latency(model, device, size)
    torch.set_num_threads(1)
    cpu_latency = latency(model, torch.device("cpu"), size)
    cpu_latency["threads"] = 1
    result = {"status": "trained" if pretrained else "synthetic_smoke_test_only",
              "best_epoch": checkpoint["best_epoch"], "epochs_completed": len(history),
              **evaluation,
              "baselines": baselines, "latency_training_device": device_latency,
              "latency_server_cpu": cpu_latency,
              "parameters": sum(p.numel() for p in model.parameters()),
              "checkpoint_bytes": (output / "best.pt").stat().st_size,
              "wall_seconds": time.monotonic() - start, "ios_latency": None}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/baseline"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text()) if args.config else None
    data_dir = Path(settings["data_dir"]) if settings else args.data_dir
    output = Path(settings["output_dir"]) if settings else args.output
    if (output / "best.pt").exists():
        raise FileExistsError(f"Checkpoint already exists in {output}")
    train(data_dir, output, args.epochs, settings=settings)
    if settings:
        (output / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
