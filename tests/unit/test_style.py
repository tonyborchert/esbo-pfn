import matplotlib.pyplot as plt
import pytest

from esbopfn.plots.style import (
    DISPLAY_NAME,
    METHOD_COLORS,
    display_name,
    figsize,
    method_color,
    save,
    thesis_style,
)

THESIS_NAMES = [
    "ESRS", "BO", "ESBO-GP", "ESBO-T-TabPFN", "ESBO-Oracle", "DecPFN", "DecPFN-H", "XDecPFN-PI",
    "FT-PFN-EI", "LC-PFN", "LC-PFN-last", "DecPFN-no-St", "Segment-GP", "Trajectory-TabPFN",
    "Segment-TabPFN", "XDecPFN", "FT-PFN",
]


def test_names_and_colors():
    assert set(THESIS_NAMES) <= set(DISPLAY_NAME.values())
    for name in THESIS_NAMES:
        assert method_color(name).startswith("#")
    assert method_color("XDecPFN-PI") == METHOD_COLORS["XDecPFN"]
    assert method_color("decpfn/cartpole/binned/1M") == "#0072B2"
    assert display_name("lcpfn-last") == "LC-PFN-last"
    assert method_color("ESBO-Oracle", theme="black") == "#FFFFFF"
    assert figsize(2) == (3.02, 2.0)


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_render(tmp_path, theme, fmt):
    with thesis_style(theme, usetex=False) as rc:
        fig, ax = plt.subplots(figsize=figsize(3))
        ax.plot([0, 1], [0, 1], color=method_color("DecPFN", theme), label="DecPFN")
        ax.grid(True)
        ax.set_xlabel(r"$J_T$")
        ax.legend()
        out = save(fig, tmp_path / theme / "fig", fmt)
        plt.close(fig)
    assert rc["font.size"] == 9
    assert out.suffix == f".{fmt}" and out.stat().st_size > 0


def test_render_usetex_if_available(tmp_path):
    from esbopfn.plots.style import usetex_available

    if not usetex_available():
        pytest.skip("no TeX with mathdesign")
    with thesis_style("white"):
        fig, ax = plt.subplots(figsize=figsize(2))
        ax.plot([0, 1], [1, 0])
        ax.set_xlabel(r"$\theta$")
        out = save(fig, tmp_path / "tex", "pdf")
        plt.close(fig)
    assert out.stat().st_size > 0
