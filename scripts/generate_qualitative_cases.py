"""Generate Image-2-style qualitative mask comparison figures.

The input cases are real exports from the PR-MaGIC inference script.  Each row
contains the support image, query image, ground truth, baseline, original
selector output, and guarded output.  Prediction masks are shown as translucent
overlays on the query image, so the figure is readable without relying on a
black/white mask alone.

Expected input layout:

    results/visual_export/visuals/<obj_name>/
        support.png query.png ground_truth.png candidate_0.png
        original.png guarded.png meta.csv

Outputs:

    figures/generated/qualitative_rescue_cases.png/.pdf
    figures/generated/qualitative_gate_limitations.png/.pdf
    figures/generated/qualitative_fss_cases_all.png/.pdf
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "results" / "visual_export" / "visuals"
OUT = ROOT / "figures" / "generated"

RESCUE_CASES = ["5_class-760", "19_class-761", "27_class-762"]
LIMITATION_CASES = ["2_class-760", "17_class-761", "26_class-762"]


def read_meta(case_dir: Path) -> dict[str, str]:
    with (case_dir / "meta.csv").open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if mask.shape[::-1] != size:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 127
    return mask


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 100) -> Image.Image:
    base = image.convert("RGBA")
    color_layer = Image.new("RGBA", base.size, (*color, 0))
    color_array = np.asarray(color_layer).copy()
    color_array[mask, :3] = color
    color_array[mask, 3] = alpha
    color_layer = Image.fromarray(color_array, mode="RGBA")
    composite = Image.alpha_composite(base, color_layer).convert("RGB")

    # Add a thin contour so small masks remain visible after paper scaling.
    padded = np.pad(mask, 1, mode="constant")
    contour = (
        padded[1:-1, 1:-1]
        & (
            ~padded[:-2, 1:-1]
            | ~padded[2:, 1:-1]
            | ~padded[1:-1, :-2]
            | ~padded[1:-1, 2:]
        )
    )
    arr = np.asarray(composite).copy()
    arr[contour] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def label_image(image: Image.Image, title: str, subtitle: str | None = None) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 42), "white")
    canvas.paste(image, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((image.width // 2, 8), title, fill="black", anchor="ma")
    if subtitle:
        draw.text((image.width // 2, 25), subtitle, fill="#555555", anchor="ma")
    return canvas


def case_panels(case_name: str) -> tuple[list[Image.Image], dict[str, str]]:
    case_dir = CASE_ROOT / case_name
    meta = read_meta(case_dir)
    query = load_rgb(case_dir / "query.png")
    support = load_rgb(case_dir / "support.png")
    size = query.size

    gt = load_mask(case_dir / "ground_truth.png", size)
    baseline = load_mask(case_dir / "candidate_0.png", size)
    original = load_mask(case_dir / "original.png", size)
    guarded = load_mask(case_dir / "guarded.png", size)

    original_miou = float(meta["original_miou"])
    guarded_miou = float(meta["guarded_miou"])
    baseline_miou = float(meta["baseline_miou"])
    fallback = int(meta["guarded_fallback"]) == 1

    action = "fallback to baseline" if fallback else "accept refinement"
    status_color = "#a33b3b" if fallback else "#2d7042"

    panels = [
        support,
        query,
        overlay(query, gt, (49, 130, 78), alpha=92),
        overlay(query, baseline, (80, 80, 80), alpha=95),
        overlay(query, original, (205, 73, 62), alpha=100),
        overlay(query, guarded, (49, 105, 176), alpha=100),
    ]
    labels = {
        "case": case_name,
        "original_iter": meta["original_iter"],
        "guarded_iter": meta["guarded_iter"],
        "baseline_miou": f"{baseline_miou:.3f}",
        "original_miou": f"{original_miou:.3f}",
        "guarded_miou": f"{guarded_miou:.3f}",
        "action": action,
        "action_color": status_color,
    }
    return panels, labels


def draw_cases(case_names: list[str], stem: str, title: str, mode: str) -> None:
    missing = [name for name in case_names if not (CASE_ROOT / name / "meta.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing qualitative case directories: {missing}")

    column_titles = [
        "Reference image",
        "Target image",
        "Ground truth",
        "Baseline $M_0$",
        "Original selector",
        "Guarded selector",
    ]
    fig, axes = plt.subplots(
        len(case_names), 6,
        figsize=(14.4, 3.0 * len(case_names)),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.04, top=0.89, hspace=0.34, wspace=0.035)

    for col, title_text in enumerate(column_titles):
        axes[0, col].set_title(title_text, fontsize=11, fontweight="bold", pad=8)

    for row_index, case_name in enumerate(case_names):
        panels, labels = case_panels(case_name)
        for col, panel in enumerate(panels):
            ax = axes[row_index, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#777777")

        status = (
            f"{labels['case']}  |  {labels['action']}\n"
            f"baseline {labels['baseline_miou']}  ·  original {labels['original_miou']}  ·  guarded {labels['guarded_miou']}"
        )
        axes[row_index, 0].text(
            -0.04,
            -0.13,
            status,
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=labels["action_color"],
        )

    fig.suptitle(title, fontsize=14, y=0.975)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT / (stem + '.png')}")
    print(f"Saved {OUT / (stem + '.pdf')}")


def main() -> None:
    draw_cases(
        RESCUE_CASES,
        "qualitative_rescue_cases",
        "Guarded PR-MaGIC rescues degraded original selections",
        "rescue",
    )
    draw_cases(
        LIMITATION_CASES,
        "qualitative_gate_limitations",
        "Conservative gate limitations",
        "limitation",
    )
    draw_cases(
        RESCUE_CASES + LIMITATION_CASES,
        "qualitative_fss_cases_all",
        "Qualitative comparison of guarded candidate selection",
        "all",
    )


if __name__ == "__main__":
    main()
