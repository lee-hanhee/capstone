"""Generate figure-only Lato SVGs for the selected clean-input 224px model."""
import json
import csv
import textwrap

from PIL import Image, ImageOps
from matplotlib.patches import Rectangle
from slide_style import ROOT, SVG, PNG, NAVY, TEAL, ORANGE, SLATE, WHITE
from slide_style import plt, text, rect, block, arrow, canvas, table, save


def main():
    root = ROOT / "artifacts/v3_224_clean"
    r = json.loads((root / "metrics.json").read_text())
    history = json.loads((root / "history.json").read_text())
    m = r["test_selected_thresholds"]
    classes = ["gate", "restaurant", "washroom"]
    # These directories contain generated presentation exports, not experiment data.
    for path in SVG.glob("[0-9][0-9]_*.svg"):
        path.unlink()
    for path in PNG.glob("[0-9][0-9]_*.png"):
        path.unlink()

    fig, ax = canvas()
    block(ax, 90, 370, 180, 160, "Camera\nframe", False, 22)
    arrow(ax, [(270, 450), (320, 450)])
    block(ax, 320, 370, 255, 160, "224 × 224 RGB\nPad + normalize", False, 22)
    arrow(ax, [(575, 450), (625, 450)])
    block(ax, 625, 345, 340, 210, "MobileNetV3-Small\nShared features", size=24)
    text(ax, 795, 290, "1.52M parameters", 23, TEAL, "bold", "center")
    for y, c in [(630, "Gate"), (390, "Restaurant"), (150, "Washroom")]:
        arrow(ax, [(965, 450), (1010, 450), (1010, y+60), (1070, y+60)])
        block(ax, 1070, y, 350, 120, f"{c}\nLinear + sigmoid", size=23)
    text(ax, 800, 80, "All three absent → no relevant sign", 23, TEAL, "bold", "center")
    save(fig, "01_model_architecture")

    fig, ax = canvas()
    for x, value, label in [
        (100, "1.52M", "Parameters"),
        (600, "5.93 MiB", "Checkpoint"),
        (1100, "3", "Independent outputs"),
    ]:
        rect(ax, x, 300, 400, 390)
        text(ax, x+200, 555, value, 52, TEAL, "bold", "center")
        text(ax, x+200, 430, label, 27, weight="bold", ha="center")
    text(ax, 800, 185, "One shared MobileNetV3-Small backbone", 31, NAVY, "bold", "center")
    save(fig, "02_mobile_baseline")

    fig, ax = canvas()
    for x, label, n, frames in [(100, "Train", "8 videos", "800 frames"),
                                (600, "Validation", "2 videos", "200 frames"),
                                (1100, "Test", "2 videos", "200 frames")]:
        rect(ax, x, 240, 400, 420)
        text(ax, x+200, 570, label, 32, TEAL, "bold", "center")
        text(ax, x+200, 450, n, 35, weight="bold", ha="center")
        text(ax, x+200, 340, frames, 30, SLATE, ha="center")
    text(ax, 800, 120, "Same source video → same partition", 28, weight="bold", ha="center")
    save(fig, "03_video_split")

    fig, ax = canvas()
    block(ax, 130, 430, 570, 260, "Epochs 1–3\nTrain the three heads\nBackbone frozen", size=30)
    arrow(ax, [(700, 560), (900, 560)])
    block(ax, 900, 430, 570, 260, "Epochs 4–10\nTrain all layers\nBackbone unfrozen", size=30)
    text(ax, 415, 340, "Head learning rate: 0.001", 24, SLATE, ha="center")
    text(ax, 1185, 340, "Fine-tuning learning rate: 0.0001", 24, SLATE, ha="center")
    table(ax, 190, 235, [390, 430, 400], [
        ["Optimizer", "Loss", "Input variation"],
        ["AdamW · batch 32", "Weighted binary BCE", "No augmentation"],
    ], row_h=85, sizes=[22, 22, 22])
    save(fig, "04_training_method")

    fig, ax = canvas()
    left, bottom, width, height = 220, 190, 1220, 530
    for score in [0, 0.25, 0.5, 0.75, 1]:
        y = bottom+height*score
        ax.plot([left, left+width], [y, y], color=TEAL, alpha=0.25, linewidth=1)
        text(ax, left-30, y, f"{score:.2f}", 19, SLATE, ha="right")
    xs = [left+width*i/9 for i in range(10)]
    ys = [bottom+height*h["val_macro_f1"] for h in history]
    ax.plot(xs, ys, color=NAVY, linewidth=4, marker="o", markersize=9)
    split = (xs[2]+xs[3])/2
    ax.plot([split, split], [bottom, bottom+height], color=TEAL, linestyle="--", linewidth=2)
    best = r["best_epoch"]-1
    ax.plot(xs[best], ys[best], marker="*", markersize=25, color=ORANGE, zorder=6)
    text(ax, xs[best]+35, ys[best]+65, f"Selected: epoch {r['best_epoch']}\nF1 {history[best]['val_macro_f1']:.3f}", 23, ORANGE, "bold")
    for i, x in enumerate(xs):
        text(ax, x, bottom-35, str(i+1), 20, ha="center")
    text(ax, 800, 90, "Training epoch", 24, ha="center")
    text(ax, 75, 455, "Validation macro F1", 23, ha="center").set_rotation(90)
    text(ax, (xs[0]+xs[2])/2, 120, "Heads only", 21, TEAL, ha="center")
    text(ax, (xs[3]+xs[-1])/2, 120, "All layers", 21, TEAL, ha="center")
    save(fig, "05_checkpoint_selection")

    fig, ax = canvas()
    ax.add_patch(Rectangle((160, 785), 30, 25, facecolor=NAVY))
    text(ax, 205, 797, "Precision: positive predictions correct", 22)
    ax.add_patch(Rectangle((870, 785), 30, 25, facecolor=TEAL))
    text(ax, 915, 797, "Recall: actual signs found", 22)
    for c, y in zip(classes, [635, 440, 245]):
        text(ax, 110, y, c.title(), 27, weight="bold")
        for metric, offset, color in [("precision", 32, NAVY), ("recall", -32, TEAL)]:
            value = m["per_class"][c][metric]
            ax.add_patch(Rectangle((420, y+offset-20), 900*value, 40, facecolor=color))
            text(ax, 435+900*value, y+offset, f"{100*value:.1f}%", 22, color, "bold")
    for percent in [0, 25, 50, 75, 100]:
        text(ax, 420+percent*9, 130, f"{percent}%", 19, SLATE, ha="center")
    save(fig, "06_precision_recall")

    fig, ax = canvas()
    rows = [["Sign class", "Actual positive", "Missed signs", "False alarms"]]
    for c in classes:
        cm = m["per_class"][c]["confusion_matrix_tn_fp_fn_tp"]
        rows.append([c.title(), str(m["per_class"][c]["positives"]), str(cm[2]), str(cm[1])])
    table(ax, 160, 730, [340, 320, 320, 310], rows, row_h=145, sizes=[28, 27, 27, 27])
    save(fig, "07_error_counts")

    fig, ax = canvas()
    manifest = json.loads((ROOT / "data/v3_224_clean/manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    lookup = {row["image"]: row for row in manifest["rows"]}
    with (root / "test_predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))

    def labels(row):
        actual = [c for c in classes if int(row[f"true_{c}"])]
        predicted = [c for c, t in zip(classes, config["thresholds"]) if float(row[f"prob_{c}"]) >= t]
        return actual, predicted

    def target_area(row, missed_only=False):
        source = lookup[row["image"]]
        w, h = source["original_size"]
        actual, predicted = labels(row)
        targets = set(actual)-set(predicted) if missed_only else set(actual)
        return max(((i["bbox"][2]-i["bbox"][0])*(i["bbox"][3]-i["bbox"][1])/(w*h)
                    for i in source["instances"] if i["class"] in targets), default=0)

    correct = [row for row in predictions if labels(row)[0] and labels(row)[0] == labels(row)[1]]
    missed = [row for row in predictions if set(labels(row)[0])-set(labels(row)[1])]
    cases = [(max(correct, key=target_area), "Correct", TEAL),
             (max(missed, key=lambda row: target_area(row, True)), "Missed signs", ORANGE)]
    for x, (row, status, color) in zip([100, 850], cases):
        source = lookup[row["image"]]
        with Image.open(ROOT / "unannotated" / source["unmarked_source"]) as image:
            w, h = image.size
            scale = min(620/w, 560/h)
            iw, ih = w*scale, h*scale
            ax.imshow(image, extent=(x+(650-iw)/2, x+(650+iw)/2, 255+(560-ih)/2, 255+(560+ih)/2), zorder=2)
        actual, predicted = labels(row)
        text(ax, x+325, 850, status, 30, color, "bold", "center")
        text(ax, x+325, 185, "Actual: " + ", ".join(actual), 23, ha="center")
        label = "Predicted: " + (", ".join(predicted) or "none")
        text(ax, x+325, 105, "\n".join(textwrap.wrap(label, 35)), 23, color, "bold", "center")
    save(fig, "08_test_examples")

    fig, ax = canvas()
    for x, label, value, unit in [
        (335, "T4 GPU", f"{r['latency_training_device']['mean_ms']:.2f}", "ms"),
        (900, "Server CPU", f"{r['latency_server_cpu']['mean_ms']:.2f}", "ms · one thread"),
    ]:
        rect(ax, x, 335, 365, 370)
        text(ax, x+182, 625, label, 28, weight="bold", ha="center")
        text(ax, x+182, 505, value, 52, TEAL if value!="?" else ORANGE, "bold", "center")
        text(ax, x+182, 405, unit, 23, SLATE, ha="center")
    text(ax, 800, 220, f"{r['parameters']/1e6:.2f}M parameters  ·  {r['checkpoint_bytes']/1024**2:.2f} MiB checkpoint", 29, weight="bold", ha="center")
    save(fig, "09_mobile_evidence")

    fig, ax = canvas()
    block(ax, 120, 350, 400, 230, "Recognize signs", size=31)
    text(ax, 320, 265, "Implemented", 27, TEAL, "bold", "center")
    arrow(ax, [(520, 465), (600, 465)])
    block(ax, 600, 350, 400, 230, "Locate + read", dark=False, size=31)
    text(ax, 800, 265, "Next", 27, SLATE, ha="center")
    arrow(ax, [(1000, 465), (1080, 465)])
    block(ax, 1080, 350, 400, 230, "Interpret + speak", dark=False, size=31)
    text(ax, 1280, 265, "Next", 27, SLATE, ha="center")
    save(fig, "10_path_to_spoken_guidance")

    paths = sorted(PNG.glob("[0-9][0-9]_*.png"))
    sheet = Image.new("RGB", (1600, 5*460), WHITE)
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            thumb = ImageOps.contain(im.convert("RGB"), (780, 438))
            sheet.paste(thumb, ((i%2)*800+10, (i//2)*460+10))
    sheet.save(ROOT / "reports/progress_contact_sheet.png")
    print("Built ten current figure-only Lato SVGs, matching PNGs and contact sheet")


if __name__ == "__main__":
    main()
