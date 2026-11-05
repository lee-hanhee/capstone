"""Prepare verified clean inputs using retained labels and a seeded 8/2/2 video split."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
import yaml

from baseline.data import CLASSES, letterbox


def prepare(config_path):
    settings = yaml.safe_load(config_path.read_text())
    source = Path(settings["clean_source"])
    export = json.loads((source / "manifest.json").read_text())
    previous_path = Path(settings["label_manifest"])
    previous = json.loads(previous_path.read_text())
    if previous["classes"] != CLASSES or settings["split_videos"] != [8, 2, 2]:
        raise ValueError("Expected class order and confirmed 8/2/2 video split")
    labels = {(r["source_video"], r["frame_index"]): r
              for r in previous["rows"] + previous["excluded"]}
    frames = defaultdict(list)
    for relative, record in export["records"].items():
        frames[(record["source_video"], record["frame_index"])].append((relative, record))
    rows = []
    for key, copies in sorted(frames.items()):
        old = labels[key]
        if {relative.split("/")[0] for relative, _ in copies} != set(CLASSES):
            raise ValueError(f"Missing class coverage: {key}")
        if len({r["sha256"] for _, r in copies}) != 1:
            raise ValueError(f"Copies have inconsistent source checksums: {key}")
        for relative, record in copies:
            c = relative.split("/")[0]
            if (not relative.endswith("_negative.png")) != bool(old["labels"][CLASSES.index(c)]):
                raise ValueError(f"Export filename and retained ground truth disagree: {relative}")
            if not (source / relative).is_file():
                raise FileNotFoundError(source / relative)
            if not relative.endswith("_negative.png") and not any(
                i["class"] == c and i["annotation_file"] == Path(relative).name for i in old["instances"]
            ):
                raise ValueError(f"Unmatched positive instance: {relative}")
        relative, record = sorted(copies)[0]
        rows.append({"source_video": key[0], "frame_index": key[1],
                     "labels": old["labels"], "instances": old["instances"],
                     "unmarked_source": relative, "source_sha256": record["sha256"],
                     "original_size": [record["width"], record["height"]],
                     "image": f"images/{Path(key[0]).stem}_f{key[1]:06d}.png"})

    videos = sorted({r["source_video"] for r in rows})
    if len(videos) != 12:
        raise ValueError(f"Expected confirmed 12-video export, found {len(videos)}")
    rng = random.Random(settings["seed"])
    for attempt in range(10000):
        rng.shuffle(videos)
        groups = {"test": videos[:2], "val": videos[2:4], "train": videos[4:]}
        valid = True
        for members in groups.values():
            subset = [r for r in rows if r["source_video"] in members]
            for j in range(3):
                positive = sum(r["labels"][j] for r in subset)
                positive_videos = {r["source_video"] for r in subset if r["labels"][j]}
                valid &= 0 < positive < len(subset) and len(positive_videos) >= 2
        if valid:
            break
    else:
        raise ValueError("No seeded split provides class coverage")
    counts = {}
    for split, members in groups.items():
        subset = [r for r in rows if r["source_video"] in members]
        for row in subset:
            row["split"] = split
        counts[split] = {"frames": len(subset), "videos": sorted(members),
                         "positive_frames": {c: sum(r["labels"][j] for r in subset)
                                             for j, c in enumerate(CLASSES)}}
    output = Path(settings["data_dir"])
    (output / "images").mkdir(parents=True, exist_ok=True)
    for n, row in enumerate(rows):
        path = source / row["unmarked_source"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError(f"Clean input checksum mismatch: {path}")
        with Image.open(path) as im:
            if list(im.size) != row["original_size"]:
                raise ValueError(f"Clean input dimensions mismatch: {path}")
            letterbox(im, settings["image_size"]).save(output / row["image"])
        if (n+1) % 300 == 0:
            print(f"Verified and prepared {n+1}/{len(rows)} clean source frames", flush=True)
    restored_keys = {(r["source_video"], r["frame_index"]) for r in previous["excluded"]}
    restored = sum(key in restored_keys for key in frames)
    manifest = {"classes": CLASSES, "image_size": settings["image_size"],
                "seed": settings["seed"], "counts": counts, "rows": rows,
                "included_frames": len(rows), "excluded_frames": 0, "excluded": [],
                "restored_previously_excluded_frames": restored,
                "clean_source": str(source), "clean_export_manifest_sha256":
                    hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
                "annotation_sha256": previous["annotation_sha256"],
                "label_manifest_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                "export_annotation_mapping_sha256": export["annotation_mapping_sha256"],
                "label_provenance": "Retained pre-export ground truth; all export presence labels and positive instance filenames cross-checked",
                "split_assignment_attempts": attempt+1,
                "split_policy": "First seeded 8/2/2 assignment with class coverage and >=2 positive videos per class per partition",
                "preprocessing": "Original clean RGB PNG, bilinear aspect-ratio-preserving padding; no augmentation",
                "notes": "12-video cohort; historical 19-video results are not directly comparable. These are existing captures, not independent new sessions."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"rows", "excluded"}}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/v3_224_clean/config.yaml"))
    prepare(parser.parse_args().config)
