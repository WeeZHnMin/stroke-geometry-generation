from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def strokes_to_xy(strokes: Sequence[dict]) -> list[tuple[float, float, str]]:
    x = 0.0
    y = 0.0
    points = []
    for step in strokes:
        x += float(step["dx"])
        y += float(step["dy"])
        points.append((x, y, str(step["pen_state"])))
    return points


def save_strokes_png(strokes: Sequence[dict], path: str | Path, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = strokes_to_xy(strokes)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    prev = None
    for x, y, pen_state in points:
        if pen_state == "move" or prev is None:
            prev = (x, y)
            continue
        px, py = prev
        ax.plot([px, x], [py, y], color="black", linewidth=2)
        if pen_state in {"end_shape", "end_all"}:
            prev = None
        else:
            prev = (x, y)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.grid(True, alpha=0.2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
