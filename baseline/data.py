"""Build one unmarked, multi-label observation per source frame."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageOps

CLASSES = ["gate", "restaurant", "washroom"]
SIZE = 224
PADDING = (124, 116, 104)


def letterbox(image):
    return ImageOps.pad(image.convert("RGB"), (SIZE, SIZE),
                        method=Image.Resampling.BILINEAR, color=PADDING)


def prepare(data_dir, output):
    frames = defaultdict(lambda: defaultdict(list))
    hashes = {}
    for label in CLASSES:
        path = data_dir / label / "ground_truth_bulk_v1.json"
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, row in json.loads(path.read_text()).items():
            if row["class"] != label or row["file_name"] != name:
                raise ValueError(f"Inconsistent class/filename: {name}")
            if not (data_dir / label / name).is_file():
                raise ValueError(f"Missing image: {label}/{name}")
            frames[(row["source_video"], row["frame_index"])][label].append(row)

    rows, excluded = [], []
    (output / "images").mkdir(parents=True, exist_ok=True)
    for (video, frame), labels in sorted(frames.items()):
        if set(labels) != set(CLASSES):
            raise ValueError(f"Missing class for {video}:{frame}")
        for label, instances in labels.items():
            if len({r["object_present"] for r in instances}) != 1:
                raise ValueError(f"Conflicting presence for {video}:{frame}:{label}")
        targets = [int(labels[c][0]["object_present"]) for c in CLASSES]
        instances = [{"class": c, "bbox": r["bbox"],
                      "instance_index": r["instance_index"],
                      "truncated": r["truncated"], "annotation_file": r["file_name"]}
                     for c in CLASSES for r in labels[c] if r["object_present"]]
        entry = {"source_video": video, "frame_index": frame,
                 "labels": targets, "instances": instances}
        candidates = [(c, r) for c in CLASSES for r in labels[c]
                      if not r["object_present"] and r["bbox"] is None
                      and r["bbox_line_width"] == 0]
        if not candidates:
            excluded.append(entry)
            continue
        label, source = candidates[0]
        path = data_dir / label / source["file_name"]
        name = f"{Path(video).stem}_f{frame:06d}.png"
        with Image.open(path) as image:
            if image.size != (source["width"], source["height"]):
                raise ValueError(f"Wrong dimensions: {path}")
            letterbox(image).save(output / "images" / name)
        rows.append({**entry, "image": f"images/{name}",
                     "unmarked_source": f"{label}/{source['file_name']}",
                     "original_size": [source["width"], source["height"]]})

    # Split by video before training. No image or model performance informs this split.
    videos = sorted({r["source_video"] for r in rows})
    n_val = max(1, round(len(videos) * 0.15))
    n_test = n_val
    rng = random.Random(42)
    # The first unconstrained shuffle has no restaurant positives in test.
    # Select the first seeded assignment with class coverage, using labels only.
    for attempt in range(10000):
        rng.shuffle(videos)
        groups = {"test": videos[:n_test], "val": videos[n_test:n_test + n_val],
                  "train": videos[n_test + n_val:]}
        valid = True
        for members in groups.values():
            subset = [r for r in rows if r["source_video"] in members]
            for j in range(3):
                p = sum(r["labels"][j] for r in subset)
                positive_videos = {r["source_video"] for r in subset if r["labels"][j]}
                valid &= 0 < p < len(subset) and len(positive_videos) >= 2
        if valid:
            break
    else:
        raise ValueError("Cannot find a video split with class coverage")
    counts = {}
    for split, members in groups.items():
        subset = [r for r in rows if r["source_video"] in members]
        positives = [sum(r["labels"][j] for r in subset) for j in range(3)]
        if not subset or any(p == 0 or p == len(subset) for p in positives):
            raise ValueError(f"Split lacks positive/negative examples: {split}: {positives}")
        for row in subset:
            row["split"] = split
        counts[split] = {"frames": len(subset), "positive_frames": dict(zip(CLASSES, positives)),
                         "videos": sorted(members)}
    manifest = {"classes": CLASSES, "image_size": SIZE, "seed": 42,
                "preprocessing": "RGB, bilinear letterbox to 224x224; no crop or augmentation",
                "padding_rgb": list(PADDING), "annotation_sha256": hashes,
                "split_assignment_attempts": attempt + 1,
                "split_policy": "first seeded 13/3/3 video assignment with positive/negative examples and >=2 positive videos per class per partition",
                "counts": counts, "included_frames": len(rows),
                "excluded_frames": len(excluded), "rows": rows, "excluded": excluded}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"rows", "excluded"}}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("annotated"))
    parser.add_argument("--output", type=Path, default=Path("data/baseline"))
    args = parser.parse_args()
    prepare(args.data_dir, args.output)
