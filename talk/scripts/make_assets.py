#!/usr/bin/env python3
"""Crop pieces of the paper's Fig. 1 (Workflow.pdf) into PNGs for the slides.

Needs PyMuPDF (not in the mace env): run with any Python that has `pymupdf`.
Source: the Safari-safe export in the arXiv repository (commit f342f6b). Crop boxes are
in PDF points (page 1122 x 855). The ensemble equation is not cropped: the slides typeset
it with 1/M (population variance, as in the code), not the figure's 1/(M-1).
"""
from pathlib import Path

import pymupdf

TALK = Path(__file__).resolve().parents[1]
SOURCE = TALK.parent / "UQ-MF-Arxiv-" / "figures" / "Workflow.pdf"
CROPS = {
    "mve_box": (275, 0, 1122, 288),
    "training": (0, 300, 262, 640),
}

page = pymupdf.open(SOURCE)[0]
for name, box in CROPS.items():
    page.get_pixmap(clip=pymupdf.Rect(*box), dpi=300, alpha=False).save(TALK / "assets" / f"{name}.png")
    print("wrote", name)

# QR code to the public code repository (closing slide); needs `qrcode`.
import qrcode
import qrcode.image.svg

qrcode.make("https://github.com/matteotolloso/mace", image_factory=qrcode.image.svg.SvgPathImage,
            border=1).save(TALK / "assets" / "qr_code.svg")
print("wrote qr_code")
