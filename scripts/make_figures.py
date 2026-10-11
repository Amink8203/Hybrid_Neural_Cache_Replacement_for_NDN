"""Generate the README's technical diagrams and chart from submitted data."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from ndn_cache.topology import build_topology

OUT = ROOT / "docs" / "assets"
INK = "#142C43"
MUTED = "#566B80"
TEAL = "#087F8C"
ORANGE = "#E89A45"
BG = "#F4F7FB"


def save(figure, name: str) -> None:
    figure.savefig(OUT / f"{name}.png", dpi=170, facecolor=figure.get_facecolor())
    vector = OUT / f"{name}.svg"
    figure.savefig(vector, facecolor=figure.get_facecolor(), metadata={"Date": None})
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines())
                      + "\n", encoding="utf-8")
    plt.close(figure)


def architecture() -> None:
    figure, axis = plt.subplots(figsize=(14, 7), facecolor=BG)
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(.04, .94, "A hybrid policy for each router", color=INK, fontsize=24, weight="bold")
    axis.text(.04, .885, "Recent popularity provides the prior. Two LSTMs learn a bounded adjustment.",
              color=MUTED, fontsize=13)

    def box(x, y, title, body, color=INK):
        axis.add_patch(FancyBboxPatch((x, y), .25, .17, boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor="white", edgecolor="#D6E0E9", linewidth=1.4))
        axis.text(x+.018, y+.125, title, color=color, fontsize=15, weight="bold", va="center")
        axis.text(x+.018, y+.075, body, color=MUTED, fontsize=11.5, va="top", linespacing=1.7)

    def arrow(start, end, color=TEAL):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                      linewidth=2.1, color=color))

    box(.05, .62, "1  Name encoder", "Hashed components + embedding\nName LSTM · hidden size 48")
    box(.375, .62, "2  Request history", "Local Interest sequence\nContext LSTM · hidden size 64")
    box(.70, .62, "3  Router-local state", "Independent context at each router\nShared model within a topology")
    arrow((.31, .705), (.36, .705))
    arrow((.635, .705), (.685, .705))
    axis.text(.05, .57, "Updated for every local Interest", color=MUTED, fontsize=11)
    axis.text(.05, .45, "WHEN A FULL CACHE NEEDS REPLACEMENT", color=TEAL, fontsize=12, weight="bold")
    box(.05, .21, "4  LRU shortlist", "16 cached packets\n8 oldest residents become candidates")
    box(.375, .21, "5  Retention score", "Prior + bounded neural residual\nCandidate name + context + 8 features", TEAL)
    box(.70, .21, "6  Replace a packet", "Evaluation: evict lowest score\nRL training: sample a victim")
    arrow((.31, .295), (.36, .295))
    arrow((.635, .295), (.685, .295))
    arrow((.50, .60), (.50, .395))
    axis.text(.515, .51, "Context", color=TEAL, fontsize=11, va="center")
    axis.text(.05, .105, "TRAINING", color=TEAL, fontsize=11, weight="bold")
    axis.text(.16, .105, "Future-reuse ranking pretraining  →  REINFORCE  →  Validation selects the checkpoint",
              color=INK, fontsize=12)
    axis.text(.05, .055, "The retention score ranks candidates; it is not a calibrated reuse probability.",
              color=MUTED, fontsize=11)
    save(figure, "architecture")


def topologies() -> None:
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.6), facecolor=BG)
    figure.subplots_adjust(left=.035, right=.98, top=.71, bottom=.19, wspace=.20)
    figure.text(.035, .90, "Four graphs. Eight routers. Independent local caches.",
                color=INK, fontsize=21, weight="bold")
    figure.text(.035, .81, "Producer at node-0  ·  Consumers at node-1 through node-7  ·  16 packets per router",
                color=MUTED, fontsize=12)
    for axis, kind, title in zip(axes, ["line", "ring", "star", "mesh"], ["Line", "Ring", "Star", "Full mesh"]):
        topology = build_topology(kind, num_nodes=8, star_center="auto")
        if kind == "line":
            points = {f"node-{i}": (-.85+i*.24, 0) for i in range(8)}
        elif kind == "star":
            leaves = [i for i in range(8) if i != 1]
            points = {"node-1": (0, 0)}
            for j, i in enumerate(leaves):
                theta = math.pi/2 + 2*math.pi*j/7
                points[f"node-{i}"] = (.8*math.cos(theta), .8*math.sin(theta))
        else:
            points = {f"node-{i}": (.8*math.cos(math.pi/2+2*math.pi*i/8),
                                     .8*math.sin(math.pi/2+2*math.pi*i/8)) for i in range(8)}
        for left, right in topology.edges:
            a, b = points[left], points[right]
            axis.plot([a[0], b[0]], [a[1], b[1]], color="#A6B7C8",
                       linewidth=1.2 if kind == "mesh" else 2, zorder=1)
        for i in range(8):
            x, y = points[f"node-{i}"]
            axis.scatter(x, y, s=235 if kind == "line" else 310,
                         color=ORANGE if i == 0 else TEAL, edgecolor=BG, linewidth=1.4, zorder=3)
            axis.text(x, y, str(i), color=INK if i == 0 else "white", fontsize=10,
                       weight="bold", va="center", ha="center", zorder=4)
        axis.set(xlim=(-1.08, 1.08), ylim=(-1.04, 1.04), aspect="equal")
        axis.set_title(f"{title}\n{len(topology.edges)} links", color=INK, fontsize=14, pad=6)
        axis.axis("off")
    figure.text(.035, .075, "● Producer", color=ORANGE, fontsize=12, weight="bold")
    figure.text(.17, .075, "● Consumer / caching router", color=TEAL, fontsize=12, weight="bold")
    figure.text(.52, .075, "The star hub is node-1; the producer remains a leaf.", color=MUTED, fontsize=11)
    save(figure, "topologies")


def results() -> None:
    with (ROOT / "artifacts" / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = {(row["topology"], row["policy"]): float(row["request_hit_ratio"])*100
                for row in csv.DictReader(stream)}
    figure, axis = plt.subplots(figsize=(13, 6.8), facecolor=BG)
    figure.subplots_adjust(left=.08, right=.975, top=.72, bottom=.18)
    axis.set_facecolor(BG)
    figure.text(.08, .92, "Higher cache hit ratio across all four topologies", color=INK,
                fontsize=22, weight="bold")
    figure.text(.08, .855, "74.81% hybrid mean  /  71.83% LRU mean  /  +2.98 percentage points",
                color=TEAL, fontsize=15, weight="bold")
    figure.text(.08, .795, "Held-out synthetic trace · seed 29 · 2,500 test requests · equal packet sizes",
                color=MUTED, fontsize=12)
    policies = ["neural", "lru", "lfu", "fifo", "random"]
    labels = ["Hybrid", "LRU", "LFU", "FIFO", "Random"]
    colors = [TEAL, "#547AAB", "#9BA9BA", "#E6AF69", "#9383AB"]
    x = np.arange(4)
    width = .14
    for i, (policy, label, color) in enumerate(zip(policies, labels, colors)):
        heights = [rows[(kind, policy)] for kind in ["line", "ring", "star", "mesh"]]
        bars = axis.bar(x + (i-2)*width, heights, width=width-.012, label=label, color=color, zorder=3)
        if policy == "neural":
            axis.bar_label(bars, labels=[f"{value:.2f}%" for value in heights],
                           padding=5, fontsize=11, color=TEAL, weight="bold")
    axis.set(ylim=(0, 90), xticks=x, xticklabels=["Line", "Ring", "Star", "Full mesh"],
             ylabel="Consumer cache hit ratio (%)", yticks=np.arange(0, 91, 15))
    axis.tick_params(axis="both", labelsize=12, colors=MUTED, length=0, pad=8)
    axis.yaxis.label.set_color(MUTED)
    axis.grid(axis="y", color="#D8E2EC", linewidth=.8, zorder=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.115), ncol=5,
                frameon=False, fontsize=12)
    figure.text(.08, .035, "Source: artifacts/summary.csv. One confirmatory seed; improvements apply to the selected hybrid policy.",
                color=MUTED, fontsize=10)
    save(figure, "results")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "ndn-neural-cache"})
    architecture()
    topologies()
    results()
    print(f"Wrote three PNG figures and SVG counterparts to {OUT}")


if __name__ == "__main__":
    main()
