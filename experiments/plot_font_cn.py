from __future__ import annotations

from matplotlib import font_manager, rcParams


def configure_chinese_font() -> str:
    """Select an installed Chinese-capable font for matplotlib.

    Returns the chosen font name. Falls back to DejaVu Sans if no CJK font is found.
    """
    candidates = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in installed), None)

    if chosen is None:
        chosen = "DejaVu Sans"

    rcParams["font.family"] = chosen
    rcParams["font.sans-serif"] = [chosen]
    rcParams["axes.unicode_minus"] = False
    return chosen
