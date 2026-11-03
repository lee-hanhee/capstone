"""Predict sign presence for one original image; no Hugging Face access required."""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from baseline.data import letterbox
from baseline.model import SignClassifier
from baseline.train import tensor


def predict(checkpoint_path, image_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = SignClassifier(pretrained=False).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with Image.open(image_path) as image:
        x = tensor(letterbox(image), config["mean"], config["std"]).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(x).sigmoid()[0].tolist()
    present = [c for c, p, t in zip(config["classes"], probabilities, config["thresholds"]) if p >= t]
    return {"probabilities": dict(zip(config["classes"], probabilities)),
            "present": present, "irrelevant": not present}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/baseline/best.pt"))
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image), indent=2))
