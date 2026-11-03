"""Audit bulk labels without changing images or choosing a training policy.

Run from the repository root: python scripts/audit_dataset.py
Uses only the Python standard library.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


CLASSES = ("gate", "restaurant", "washroom")


def audit(root):
    frames = defaultdict(lambda: defaultdict(list))
    records = {}
    hashes = {}
    for label in CLASSES:
        path = root / label / "ground_truth_bulk_v1.json"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = json.loads(path.read_text())
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f"Expected nonempty annotation dictionary: {path}")
        for name, row in rows.items():
            if row["class"] != label or row["file_name"] != name:
                raise ValueError(f"Class or filename mismatch: {path}: {name}")
            if type(row["object_present"]) is not bool:
                raise ValueError(f"Expected Boolean object_present: {name}")
            if not (root / label / name).is_file():
                raise ValueError(f"Missing image: {label}/{name}")
            bbox = row["bbox"]
            if row["object_present"]:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Invalid positive bbox: {name}")
                x0, y0, x1, y1 = bbox
                if not (0 <= x0 < x1 <= row["width"] and
                        0 <= y0 < y1 <= row["height"]):
                    raise ValueError(f"Out-of-bounds bbox: {name}")
            elif bbox is not None or row["bbox_line_width"] != 0:
                raise ValueError(f"Negative image has annotation: {name}")
            frames[(row["source_video"], row["frame_index"])][label].append(row)
        records[label] = {
            "images": len(rows),
            "positive_instances": sum(r["object_present"] for r in rows.values()),
            "negative_images": sum(not r["object_present"] for r in rows.values()),
        }

    by_video = defaultdict(Counter)
    combinations = Counter()
    no_clean_copy = []
    clean_copies = []
    for (video, index), labels in sorted(frames.items()):
        if set(labels) != set(CLASSES):
            raise ValueError(f"Missing class annotations: {video}, {index}")
        dimensions = {(r["width"], r["height"], r["additional_rotation_clockwise"])
                      for rows in labels.values() for r in rows}
        if len(dimensions) != 1:
            raise ValueError(f"Inconsistent image geometry: {video}, {index}")
        for label, rows in labels.items():
            if len({r["object_present"] for r in rows}) != 1:
                raise ValueError(f"Conflicting class labels: {video}, {index}, {label}")
            if not rows[0]["object_present"] and len(rows) != 1:
                raise ValueError(f"Duplicate negative records: {video}, {index}, {label}")
        present = [c for c in CLASSES if labels[c][0]["object_present"]]
        combinations["+".join(present) or "none"] += 1
        by_video[video]["frames"] += 1
        for label in present:
            by_video[video][label] += 1
        clean = [f"{c}/{r['file_name']}" for c, rows in labels.items()
                 for r in rows if not r["object_present"]]
        entry = {"source_video": video, "frame_index": index,
                 "labels": {c: bool(labels[c][0]["object_present"]) for c in CLASSES}}
        if clean:
            clean_copies.append({**entry, "unmarked_image_candidates": clean})
        else:
            no_clean_copy.append(entry)

    return {
        "annotation_sha256": hashes,
        "classes": list(CLASSES),
        "image_records": records,
        "unique_source_frames": len(frames),
        "source_videos": len(by_video),
        "frame_label_combinations": dict(sorted(combinations.items())),
        "per_video_frame_counts": dict(sorted(by_video.items())),
        "frames_with_unmarked_copy": len(clean_copies),
        "frames_without_unmarked_copy": len(no_clean_copy),
        "unmarked_copy_candidates": clean_copies,
        "needs_clean_source": no_clean_copy,
        "notes": [
            "Positive PNGs contain drawn red boxes according to the dataset manifest.",
            "Unmarked candidates are negative for their folder class, not necessarily all classes.",
            "Candidates are identified from metadata; original video pixel identity is not verified.",
            "Separate instance images of a source frame are not independent observations.",
            "No training policy, split, exclusion, or pixel repair is applied by this audit.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("annotated"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/dataset_audit.json"))
    args = parser.parse_args()
    result = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"unmarked_copy_candidates", "needs_clean_source"}}, indent=2))


if __name__ == "__main__":
    main()
