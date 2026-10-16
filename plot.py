import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
DOCS = Path(__file__).parent / "docs"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RUNS = [("baseline-smollm2", "baseline (padded copies)"),
        ("paged-smollm2", "paged KV cache, SDPA attention"),
        ("paged-matmul-smollm2", "paged KV cache, reshape attention")]


def load(label):
    """The summary rows of one saved benchmark run."""
    return json.loads((RESULTS / f"{label}.json").read_text())["summary"]


def style(ax, title, ylabel, xlabel=None):
    """Quiet axes: light grid, no box, text in ink colours."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)


def new_figure(width=8, height=4.6, columns=1):
    """A figure with the chart surface colour."""
    fig, axes = plt.subplots(1, columns, figsize=(width, height), facecolor=SURFACE)
    return fig, axes


def lines(ax, key, unit_format, label_at=(40,)):
    """One line per run against the number of users, with its value written at the chosen user counts."""
    for colour, (label, name) in zip(SERIES, RUNS):
        rows = load(label)
        users, values = [r["users"] for r in rows], [r[key] for r in rows]
        ax.plot(users, values, color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        for u, value in zip(users, values):
            if u in label_at:
                ax.annotate(unit_format(value), (u, value), xytext=(8, 0), textcoords="offset points",
                            va="center", color=INK, fontsize=9)
    ax.set_xticks([1, 10, 30, 40])
    ax.set_xlim(0, 46)
    ax.set_ylim(bottom=0)


def save(fig, name):
    """Write the figure as a PNG into docs/."""
    DOCS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(DOCS / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved docs/{name}")


def throughput():
    """Total tokens per second against concurrent users."""
    fig, ax = new_figure()
    lines(ax, "total_tok_s", lambda v: f"{v:.0f}", label_at=(30, 40))
    style(ax, "Total throughput (higher is better)", "tokens / second", "concurrent users (30 seats)")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    save(fig, "throughput.png")


def latency():
    """Time to first token (p50 and p99) and the gap between tokens, against concurrent users."""
    fig, axes = new_figure(width=13, height=4.4, columns=3)
    for ax, key, title in zip(axes, ("ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms"),
                              ("TTFT p50", "TTFT p99", "ITL p50")):
        lines(ax, key, lambda v: f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms")
        style(ax, f"{title} (lower is better)", "milliseconds", "concurrent users")
    axes[0].legend(frameon=False, labelcolor=INK, loc="upper left", fontsize=8)
    save(fig, "latency.png")


def round_breakdown():
    """Where one decode round's time goes at about 30 users, before and after paging."""
    data = json.loads((RESULTS / "round-breakdown.json").read_text())["rounds"]
    parts = list(data[0]["parts"])
    order = [1, 0, 2, 3]
    fig, ax = new_figure(height=3.8)
    for row, run in enumerate(data):
        left = 0.0
        for part_index in order:
            part = parts[part_index]
            width = run["parts"][part]
            if width:
                ax.barh(row, width, left=left, height=0.5, color=SERIES[part_index], edgecolor=SURFACE, linewidth=2,
                        label=part if row == 0 else None)
            left += width
        ax.annotate(f"{left:.0f} ms", (left, row), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([f"{r['label']}  ({r['users']:.0f} users)" for r in data], color=INK)
    ax.invert_yaxis()
    style(ax, "One decode round at ~30 users (ms, lower is better)", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, labelcolor=INK, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.4, -0.12))
    save(fig, "round.png")


def prefix():
    """TTFT with a shared 500-token system prompt, prefix caching off against on."""
    off, on = load("prefix-off-smollm2"), load("prefix-on-smollm2")
    fig, axes = new_figure(width=11, height=4.2, columns=2)
    for ax, key, title in zip(axes, ("ttft_p50_ms", "ttft_p99_ms"), ("TTFT p50", "TTFT p99")):
        positions = range(len(off))
        for shift, rows, colour, name in ((-0.18, off, SERIES[0], "prefix caching off"),
                                          (0.18, on, SERIES[1], "prefix caching on")):
            values = [r[key] for r in rows]
            bars = ax.bar([p + shift for p in positions], values, width=0.34, color=colour, edgecolor=SURFACE,
                          linewidth=2, label=name)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value / 1000:.1f} s" if value >= 1000 else f"{value:.0f} ms",
                            (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4), textcoords="offset points",
                            ha="center", color=INK, fontsize=8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([f"{r['users']} user" + ("s" if r["users"] > 1 else "") for r in off])
        style(ax, f"{title}, shared 500-token prompt (lower is better)", "milliseconds")
    axes[0].legend(frameon=False, labelcolor=INK, loc="upper left")
    save(fig, "prefix.png")


def main():
    """Draw every chart from the saved results."""
    throughput()
    latency()
    round_breakdown()
    prefix()


if __name__ == "__main__":
    main()
