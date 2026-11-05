"""Shared Lato/palette styling for presentation SVGs and matching PNGs."""
import os
from pathlib import Path

os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
SVG, PNG = ROOT / "reports/svg", ROOT / "reports/png"
NAVY, ORANGE, TEAL, SLATE, WHITE = "#1B2C4A", "#F07A2A", "#2F7F81", "#3B4A66", "#FFFFFF"
for filename in ["Lato-Regular.ttf", "Lato-Bold.ttf"]:
    path = ROOT / "artifacts/fonts" / filename
    if not path.exists():
        raise FileNotFoundError(f"Lato font required: {path}")
    font_manager.fontManager.addfont(path)
plt.rcParams.update({"font.family": "Lato", "svg.fonttype": "path", "image.composite_image": False})


def text(ax, x, y, label, size=20, color=NAVY, weight="normal", ha="left", va="center"):
    font = font_manager.FontProperties(fname=ROOT / "artifacts/fonts" /
                                      ("Lato-Bold.ttf" if weight == "bold" else "Lato-Regular.ttf"))
    return ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, linespacing=1.25, zorder=5, fontproperties=font)


def rect(ax, x, y, w, h, fill=WHITE, edge=TEAL, radius=20):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=fill, edgecolor=edge, linewidth=1.8, zorder=1))


def block(ax, x, y, w, h, label, dark=True, size=20):
    rect(ax, x, y, w, h, NAVY if dark else WHITE, NAVY if dark else TEAL)
    if dark:
        ax.plot([x+3, x+3], [y+10, y+h-10], color=ORANGE, linewidth=5,
                solid_capstyle="round", zorder=3)
    text(ax, x+w/2, y+h/2, label, size, WHITE if dark else NAVY, "bold", "center")


def arrow(ax, points):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=NAVY, linewidth=2, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=20, linewidth=2, color=NAVY, zorder=2))


def canvas():
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1600), ylim=(0, 900))
    ax.axis("off")
    return fig, ax


def table(ax, x, y, widths, rows, row_h=95, sizes=None):
    for i, row in enumerate(rows):
        top = y-i*row_h
        if i == 0:
            rect(ax, x, top-row_h, sum(widths), row_h, NAVY, NAVY, 10)
        elif i > 1:
            ax.plot([x, x+sum(widths)], [top, top], color=TEAL, linewidth=0.8)
        left = x
        for j, (value, width) in enumerate(zip(row, widths)):
            text(ax, left+18, top-row_h/2, str(value), (sizes or [20]*len(row))[j],
                 WHITE if i == 0 else NAVY, "bold" if i == 0 else "normal")
            left += width


def save(fig, name):
    SVG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    # Outlined SVGs retain exact Lato appearance without installing the font elsewhere.
    path = SVG / f"{name}.svg"
    fig.savefig(path, facecolor=WHITE)
    path.write_text(path.read_text().replace('width="1152pt" height="648pt"',
                                            'width="1600px" height="900px"', 1))
    fig.savefig(PNG / f"{name}.png", dpi=120, facecolor=WHITE)
    plt.close(fig)
