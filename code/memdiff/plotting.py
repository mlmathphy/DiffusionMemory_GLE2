"""Shared figure styling for all examples.

Applies the niceplots style (mdolab/niceplots) when the package is
available and always layers the paper's sizing on top, so figures come
out identical on any machine that has niceplots and still render with
matplotlib defaults on one that does not.
"""

import matplotlib.pyplot as plt

PAPER_RC = {
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "lines.linewidth": 1.8,
}

# Model curves are drawn SOLID and semi-transparent over the opaque
# reference, rather than dashed: dash patterns fragment exactly where
# curves agree (the interesting region) and read as noise when several
# overlap, while transparency shows agreement as the reference reading
# through and disagreement as two clean separated lines.
MODEL_ALPHA = 0.75
BAND_ALPHA = 0.15


# Typography: match the manuscript. elsarticle typesets in Computer
# Modern, so the figures use CMU Serif with Computer Modern mathtext and
# figure text is indistinguishable from body text. This also fixes the
# mismatch niceplots leaves behind: its styles set a text font but not
# mathtext.fontset, so matplotlib renders every $...$ label in its
# DejaVu default while the surrounding text uses another family.
FONT_RC = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman",
                   "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
}


def setup_style():
    """niceplots layout (if installed) + the paper's typography/sizing."""
    try:
        import niceplots
        plt.style.use(niceplots.get_style())
    except Exception:
        pass
    plt.rcParams.update(PAPER_RC)
    plt.rcParams.update(FONT_RC)      # typography last: it must win


def paper_typography(fig, width_fraction=1.0):
    """Set readable point sizes after the PDF is scaled into the manuscript.

    The manuscript has a 6.5-inch text block. Apply before layout/save so
    explicit small legend/annotation sizes cannot override the print sizes.
    This changes text only; the plotted data and statistical calculations
    are untouched.
    """
    from matplotlib.text import Text

    scale = fig.get_figwidth() / (6.5 * width_fraction)
    for text in fig.findobj(Text):
        text.set_fontsize(9 * scale)
    for ax in fig.axes:
        ax.title.set_fontsize(10.5 * scale)
        ax.xaxis.label.set_fontsize(10.5 * scale)
        ax.yaxis.label.set_fontsize(10.5 * scale)
        ax.tick_params(axis="both", which="major", labelsize=9 * scale)
        ax.tick_params(axis="both", which="minor", labelsize=9 * scale)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(11 * scale)
