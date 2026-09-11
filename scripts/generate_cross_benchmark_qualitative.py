"""Generate cross-benchmark qualitative mask comparisons.

Each row is one real exported episode and each column has the same meaning:
support image, query image, ground truth, baseline, original selector, and
guarded selector. Predictions are transparent overlays on the query image.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"

CASES = {
    "FSS-1000": ROOT / "results/visual_export/visuals/5_class-760",
    "COCO-20i": ROOT / "results/visual_export/coco_f0_s45_30/visuals/17_class-32",
    "Pascal-5i": ROOT / "results/visual_export/pascal_f0_s45_30_fixed/visuals/14_class-0",
}

LIMITATION_CASES = {
    "FSS-1000": ROOT / "results/visual_export/visuals/2_class-760",
    "COCO-20i": ROOT / "results/visual_export/coco_f0_s45_30/visuals/14_class-28",
    "Pascal-5i": ROOT / "results/visual_export/pascal_f0_s45_30_fixed/visuals/26_class-3",
}


def read_meta(case_dir: Path) -> dict[str, str]:
    with (case_dir / "meta.csv").open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if mask.shape[::-1] != size:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                size, Image.Resampling.NEAREST
            )
        ) > 127
    return mask


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    base[mask] = (0.55 * base[mask] + 0.45 * color_arr).astype(np.uint8)
    padded = np.pad(mask, 1, mode="constant")
    contour = mask & (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
    )
    base[contour] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(base)


def panels(case_dir: Path):
    meta = read_meta(case_dir)
    query = load_rgb(case_dir / "query.png")
    support = load_rgb(case_dir / "support.png")
    size = query.size
    gt = load_mask(case_dir / "ground_truth.png", size)
    baseline = load_mask(case_dir / "candidate_0.png", size)
    original = load_mask(case_dir / "original.png", size)
    guarded = load_mask(case_dir / "guarded.png", size)

    images = [
        support,
        query,
        overlay(query, gt, (40, 150, 70), 100),
        overlay(query, baseline, (90, 90, 90), 100),
        overlay(query, original, (205, 65, 55), 100),
        overlay(query, guarded, (45, 100, 190), 100),
    ]
    return images, meta


def draw(case_map: dict[str, Path], stem: str, title: str) -> None:
    labels = [
        "Support image",
        "Query image",
        "Ground truth",
        r"Baseline $M_0$",
        "Original selector",
        "Guarded selector",
    ]
    fig, axes = plt.subplots(
        len(case_map), 6,
        figsize=(14.6, 3.35 * len(case_map)),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.055, top=0.89, hspace=0.38, wspace=0.025)
    for col, label in enumerate(labels):
        axes[0, col].set_title(label, fontsize=11, fontweight="bold", pad=8)

    for row, (benchmark, case_dir) in enumerate(case_map.items()):
        images, meta = panels(case_dir)
        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.7)

        fallback = int(meta["guarded_fallback"]) == 1
        color = "#a33b3b" if fallback else "#2d7042"
        text = (
            f"{benchmark}  |  {meta['obj_name']}  |  "
            f"{'fallback to baseline' if fallback else 'accept refinement'}\n"
            f"baseline {float(meta['baseline_miou']):.3f}  ·  "
            f"original {float(meta['original_miou']):.3f}  ·  "
            f"guarded {float(meta['guarded_miou']):.3f}"
        )
        axes[row, 0].text(
            -0.04, -0.14, text,
            transform=axes[row, 0].transAxes,
            ha="left", va="top", fontsize=8.5, color=color,
        )

    fig.suptitle(title, fontsize=14, y=0.975)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT / (stem + '.png')}")
    print(f"Saved {OUT / (stem + '.pdf')}")


def main() -> None:
    draw(
        CASES,
        "qualitative_cross_benchmark_rescue",
        "Cross-benchmark examples of guarded rescue",
    )
    draw(
        LIMITATION_CASES,
        "qualitative_cross_benchmark_limitations",
        "Cross-benchmark examples of conservative gate limitations",
    )
    draw(
        {**CASES, **{f"{k} limitation": v for k, v in LIMITATION_CASES.items()}},
        "qualitative_cross_benchmark_all",
        "Cross-benchmark qualitative comparison",
    )


if __name__ == "__main__":
    main()
