"""Shared readable styling and SVG-only export for evaluation figures."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.text import Text


mpl.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "svg.fonttype": "none",
})


def save_svg(fig, path, **kwargs):
    path = Path(path).with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    for text in fig.findobj(Text):
        text.set_fontsize(max(16, text.get_fontsize()))
    kwargs.pop("format", None)
    fig.savefig(path, format="svg", **kwargs)

