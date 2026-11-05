"""Evaluate the saved 224px checkpoint and select validation-only thresholds."""

import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.model import SignClassifier
from baseline.train import Frames, finish_evaluation


def evaluate_saved(settings, output):
    start = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(settings["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    data_dir = Path(settings["data_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if config["annotation_sha256"] != manifest["annotation_sha256"] or config["counts"] != manifest["counts"]:
        raise ValueError("Saved checkpoint and data annotations/splits differ")
    if config["image_size"] != manifest["image_size"] or settings["image_size"] != manifest["image_size"]:
        raise ValueError("Image sizes differ")
    model = SignClassifier(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    subsets = {s: [r for r in manifest["rows"] if r["split"] == s] for s in ["val", "test"]}
    loaders = {s: DataLoader(Frames(data_dir, rows, config["mean"], config["std"]),
                             batch_size=settings["batch_size"], shuffle=False)
               for s, rows in subsets.items()}
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"], device=device))
    output.mkdir(parents=True, exist_ok=True)
    evaluation, _, _ = finish_evaluation(model, loaders, subsets, device, criterion,
                                          output, checkpoint, tune_thresholds=True)
    original = json.loads((checkpoint_path.parent / "metrics.json").read_text())
    result = {**original, **evaluation, "status": "evaluated_saved_checkpoint",
              "source_checkpoint": str(checkpoint_path), "retrained": False,
              "evaluation_wall_seconds": time.monotonic() - start,
              "checkpoint_bytes": (output / "best.pt").stat().st_size}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_thresholds": result["selected_thresholds"],
                      "test_macro_f1_fixed": result["test"]["macro_f1"],
                      "test_macro_f1_selected": result["test_selected_thresholds"]["macro_f1"]}), flush=True)
