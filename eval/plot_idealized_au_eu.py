#!/usr/bin/env python3
"""Create a paper-style illustration of the four AU/EU uncertainty regimes.

Each panel represents one configuration (molecule):
- x-axis: energy predicted by ensemble members
- y-axis: variance predicted by ensemble members

The four panels show idealized examples of:
- low aleatoric, low epistemic
- high aleatoric, low epistemic
- low aleatoric, high epistemic
- high aleatoric, high epistemic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "figure.titlesize": 18,
        "legend.fontsize": 10,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


CASES = [
    {
        "title": "Low AU, Low EU",
        "subtitle": "confident and consistent",
        "ref_energy": -13507.0,
        "energy": np.array([-13507.18, -13507.12, -13507.08, -13507.04, -13507.01, -13506.98, -13506.94, -13506.90]),
        "variance": np.array([0.010, 0.013, 0.011, 0.014, 0.012, 0.015, 0.013, 0.011]),
        "facecolor": "white",
        "accent": "#1b7f5a",
    },
    {
        "title": "High AU, Low EU",
        "subtitle": "consistent but noisy",
        "ref_energy": -13507.0,
        "energy": np.array([-13508.10, -13508.03, -13507.98, -13507.93, -13507.88, -13507.83, -13507.77, -13507.72]),
        "variance": np.array([0.155, 0.182, 0.168, 0.194, 0.173, 0.205, 0.187, 0.176]),
        "facecolor": "white",
        "accent": "#be6a15",
    },
    {
        "title": "Low AU, High EU",
        "subtitle": "confident but disagreeing",
        "ref_energy": -13507.0,
        "energy": np.array([-13508.35, -13508.00, -13507.65, -13507.32, -13506.92, -13506.58, -13506.28, -13505.95]),
        "variance": np.array([0.012, 0.014, 0.011, 0.013, 0.012, 0.015, 0.013, 0.011]),
        "facecolor": "white",
        "accent": "#2563a6",
    },
    {
        "title": "High AU, High EU",
        "subtitle": "noisy and disagreeing",
        "ref_energy": -13507.0,
        "energy": np.array([-13508.85, -13508.42, -13508.01, -13507.63, -13506.98, -13506.41, -13505.92, -13505.46]),
        "variance": np.array([0.146, 0.193, 0.167, 0.214, 0.179, 0.226, 0.201, 0.188]),
        "facecolor": "white",
        "accent": "#b23a35",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create a 2x2 idealized AU/EU figure for papers or slides.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/idealized_au_eu_cases.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=str,
        default="plots/idealized_au_eu_cases.pdf",
        help="Optional PDF output path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution for raster output.",
    )
    return parser.parse_args()



def add_case_panel(ax: plt.Axes, case: dict[str, object]) -> None:
    energy = np.asarray(case["energy"], dtype=float)
    variance = np.asarray(case["variance"], dtype=float)
    accent = str(case["accent"])

    ref_energy = float(case["ref_energy"])

    ax.set_facecolor(str(case["facecolor"]))
    ax.axvline(ref_energy, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.scatter(
        energy,
        variance,
        s=92,
        color="#2f2f2f",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    mean_energy = float(np.mean(energy))
    mean_variance = float(np.mean(variance))
    ax.scatter(
        [mean_energy],
        [mean_variance],
        s=78,
        marker="D",
        # color=accent,
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
    )

    ax.text(
        0.03,
        0.96,
        str(case["title"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        weight="bold",
    )
    ax.text(
        0.03,
        0.88,
        '',
        # str(case["subtitle"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        # color=accent,
    )

    ax.set_xlim(-13509.2, -13505.8)
    ax.set_xticks([-13509.0, -13508.0, -13507.0, -13506.0])
    ax.set_ylim(0.0, 0.25)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_box_aspect(1)
    ax.grid(alpha=0.18, linewidth=0.8)



def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.6), constrained_layout=True)
    axes = axes.ravel()

    for ax, case in zip(axes, CASES):
        add_case_panel(ax, case)
        ax.set_xlabel("Energy")

    axes[0].set_ylabel("Variance (AU)")

    # fig.suptitle("Idealized aleatoric and epistemic uncertainty regimes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    if args.output_pdf is not None:
        pdf_path = Path(args.output_pdf)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    if args.output_pdf is not None:
        print(f"Saved PDF: {args.output_pdf}")


if __name__ == "__main__":
    main()
