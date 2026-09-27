"""Export slide-ready figures from the saved, circuit-grouped experiments.

Run: uv run --locked --extra report python research/make_presentation_figures.py

All plotted scores come from the checked-in validation JSON/OOF files. An
ablation's delta is always relative to its own paired baseline; the early
12-model sweep used a different grouped assignment and is a separate figure.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research"
OUT = SRC / "presentation_figures"
OUT.mkdir(exist_ok=True)

NAVY = "#17324D"
TEAL = "#116B7A"
GOLD = "#D89232"
RUST = "#A6543D"
INK = "#172A3A"
MUTED = "#607183"
GRID = "#D9E2E8"
PALE = "#F5F8FA"
TEAL_PALE = "#DBEEF0"
GRAY_PALE = "#E9EEF1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
})


def read(name: str) -> dict:
    return json.loads((SRC / f"{name}.json").read_text())


def export(fig, stem: str):
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    svg = OUT / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight")
    # Matplotlib leaves trailing spaces in multiline path data. They are not
    # meaningful in SVG, and removing them keeps the checked-in exports clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def title(fig, heading: str, subtitle: str, note: str):
    fig.text(.055, .965, heading, fontsize=24, fontweight="bold", color=NAVY,
             ha="left", va="top")
    fig.text(.055, .91, subtitle, fontsize=13, color=MUTED, ha="left", va="top")
    fig.text(.055, .032, note, fontsize=10, color=MUTED, ha="left", va="bottom")


def paired_bars(ax, labels, matched, structural, *, selected=(), xlim=None,
                xlabel="Score change (percentage points)"):
    y = np.arange(len(labels), dtype=float)
    for i in selected:
        ax.axhspan(i-.48, i+.48, color=TEAL_PALE, alpha=.45, zorder=0)
    ax.barh(y-.16, matched, height=.28, color=TEAL, zorder=3)
    ax.barh(y+.16, structural, height=.28, color=GOLD, zorder=3)
    ax.axvline(0, color=NAVY, lw=1.15, zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(y=.1)


def plot_initial_model_sweep():
    report = read("validation")
    feature_views = ["basic", "raw", "all"]
    models = ["ridge", "hist_boost", "random_forest", "extra_trees"]
    scores = {(row["view"], row["model"]): row["score"]
              for row in report["results"]}
    assert len(scores) == 12
    matrix = np.array([[scores[v, m] for m in models] for v in feature_views])
    cmap = LinearSegmentedColormap.from_list("slide", ["#E8EFF3", "#7CB8BA", TEAL])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=.08, right=.94, top=.78, bottom=.18)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=.69, vmax=.90)
    ax.set_xticks(range(4), ["Ridge", "Histogram\nboosting", "Random\nforest", "ExtraTrees"])
    ax.set_yticks(range(3), ["Basic counts\n+ depth", "Raw gate +\nconnectivity", "+ simplification\n+ timing"])
    ax.tick_params(length=0, labelsize=16, pad=14)
    for i in range(3):
        for j in range(4):
            best = j == 3
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=22, fontweight="bold" if best else "normal",
                    color="white" if matrix[i,j] > .85 else INK)
            if best:
                ax.add_patch(plt.Rectangle((j-.49, i-.49), .98, .98,
                                           fill=False, ec=NAVY, lw=2.5))
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(image, ax=ax, fraction=.03, pad=.025)
    cb.set_label("Official duration score · higher is better", rotation=90, labelpad=12)
    title(fig, "Start with QASM counts; choose a strong default model",
          "Initial 3 feature views × 4 out-of-box regressors · 5 circuit-grouped folds",
          "Source: research/validation.json · 1,497 labeled runs. This early fold assignment differs from later ablations.")
    export(fig, "01_initial_model_sweep")


def plot_foundations():
    physics = {r["view"]: r for r in read("chi_randomness_ablation")["results"]}
    geometry = {r["view"]: r for r in read("geometry_family_ablation")["results"]}
    a_names = ["χ upper bound", "Diversity only", "χ + diversity", "Effective χ only",
               "χ + all effective-χ", "χ × randomness terms"]
    a_keys = ["chi_only", "random_only", "chi_plus_random", "effective_chi_only",
              "chi_plus_random_effective", "chi_random_interactions"]
    b_names = ["Cut timeline", "Graph geometry", "Soft pattern scores",
               "Dense / liveness", "Timeline + graph", "All groups",
               "All + effective χ"]
    b_keys = ["timeline", "geometry", "family", "misc", "timeline_geometry",
              "all_new", "all_new_effective_peak"]

    def deltas(table, keys, base):
        return ([100*(table[k]["circuit"]["score"]-table[base]["circuit"]["score"])
                 for k in keys],
                [100*(table[k]["structural"]["score"]-table[base]["structural"]["score"])
                 for k in keys])

    a_match, a_stress = deltas(physics, a_keys, "baseline")
    b_match, b_stress = deltas(geometry, b_keys, "current_single")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.15, right=.96, top=.79, bottom=.17, wspace=.42)
    paired_bars(axes[0], a_names, a_match, a_stress, selected=(2,), xlim=(-1.0, 2.1))
    paired_bars(axes[1], b_names, b_match, b_stress, selected=(2, 5), xlim=(-1.0, 2.1))
    axes[0].set_title("A · Entanglement potential + diversity", loc="left", color=NAVY)
    axes[1].set_title("B · Geometry + circuit patterns", loc="left", color=NAVY)
    fig.legend(handles=[Patch(color=TEAL, label="Similar-distribution folds"),
                        Patch(color=GOLD, label="Structural-cluster stress")],
               loc="upper right", bbox_to_anchor=(.95, .9), ncol=2, frameon=False,
               fontsize=11)
    title(fig, "Static QASM structure helps, but no single proxy wins everywhere",
          "Paired score change versus each panel's own baseline; highlighted rows were carried forward",
          "Sources: chi_randomness_ablation.json and geometry_family_ablation.json · Same fixed circuit-grouped protocols within each panel.")
    export(fig, "02_static_feature_ablations")


def plot_chi_walk():
    d = read("chi_walk_probe")["runtime"]
    keys = ["cost_p2", "cost_p2_5", "cost_p3", "cost_overhead",
            "cost_high_overhead", "shape", "shape_cost", "cost_uncapped"]
    labels = ["Cost exponent 2", "Cost exponent 2.5", "Cost exponent 3",
              "Cost + overhead", "High overhead", "Walk shape",
              "Walk shape + cost", "Cost + uncapped"]
    match = [100*d[k]["matched"]["delta"] for k in keys]
    stress = [100*d[k]["structural"]["delta"] for k in keys]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=.24, right=.94, top=.8, bottom=.17)
    paired_bars(ax, labels, match, stress, selected=(6,), xlim=(0, 3.2))
    for i, (m, s) in enumerate(zip(match, stress)):
        ax.text(m+.05, i-.16, f"+{m:.2f}", va="center", fontsize=10, color=TEAL)
        ax.text(s+.05, i+.16, f"+{s:.2f}", va="center", fontsize=10, color=GOLD)
    fig.legend(handles=[Patch(color=TEAL, label="Similar-distribution folds"),
                        Patch(color=GOLD, label="Structural-cluster stress")],
               loc="upper right", bbox_to_anchor=(.94, .9), ncol=2, frameon=False)
    title(fig, "The χ walk adds trajectory information beyond static geometry",
          "Matched baseline 0.8992; stress baseline 0.7070 · score gain in percentage points",
          "Source: chi_walk_probe.json · Same-fold ExtraTrees ablations · χ is a potential-rank heuristic, not measured entanglement.")
    export(fig, "03_chi_walk_ablation")


def plot_angle_gating():
    d = read("rotation_filter_probe")["views"]
    keys = ["old_walk", "old_walk_plus_angles", "filtered_walk", "filtered_plus_angles"]
    labels = ["Original walk", "+ angle counts", "Near-0/π gated walk",
              "Gated walk + counts"]
    matched = [100*d[k]["matched"]["delta_vs_old"] for k in keys]
    stress = [100*d[k]["structural"]["delta_vs_old"] for k in keys]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=.23, right=.94, top=.8, bottom=.18)
    paired_bars(ax, labels, matched, stress, selected=(3,), xlim=(-.95, .9))
    ax.set_xticks(np.arange(-.8, .81, .2))
    for i, (m, s) in enumerate(zip(matched, stress)):
        if i:
            ax.text(m+(.025 if m>=0 else -.025), i-.16, f"{m:+.2f}",
                    ha="left" if m>=0 else "right", va="center", color=TEAL, fontsize=12)
            ax.text(s+(.025 if s>=0 else -.025), i+.16, f"{s:+.2f}",
                    ha="left" if s>=0 else "right", va="center", color=GOLD, fontsize=12)
    fig.legend(handles=[Patch(color=TEAL, label="Similar-distribution folds"),
                        Patch(color=GOLD, label="Structural-cluster stress")],
               loc="upper right", bbox_to_anchor=(.94, .9), ncol=2, frameon=False)
    title(fig, "Near-basis angle gating improves the expected holdout split",
          "Rotations within 0.3 rad of integer π preserve the walk's basis-state flag",
          "Source: rotation_filter_probe.json · Paired versus original walk (0.9105 matched; 0.7352 stress). Stress transfer worsened.")
    export(fig, "04_angle_gating")


def plot_algorithm_signals():
    geo = read("geometry_family_ablation")
    by = {r["view"]: r for r in geo["results"]}
    soft = (by["family"]["circuit"]["score"]-by["current_single"]["circuit"]["score"],
            by["family"]["structural"]["score"]-by["current_single"]["structural"]["score"])
    external = read("algorithm_geometry_probe")
    motifs = read("sequence_motif_probe")
    shor = read("shor_family_probe")
    rows = [
        ("Soft QASM fingerprints", soft[0], soft[1], True),
        ("External geometry classes", external["runtime"]["circuit"]["delta"],
         external["runtime"]["structural"]["delta"], False),
        ("Ordered QAOA / Shor motifs", motifs["runtime"]["combined"]["circuit"]["delta"],
         motifs["runtime"]["combined"]["structural"]["delta"], False),
        ("QPE / arithmetic classes", shor["runtime"]["circuit"]["delta"],
         shor["runtime"]["structural"]["delta"], False),
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=.28, right=.93, top=.78, bottom=.21)
    paired_bars(ax, [r[0] for r in rows], [100*r[1] for r in rows],
                [100*r[2] for r in rows], selected=(0,), xlim=(-.6, .8))
    ax.set_xticks(np.arange(-.6, .81, .2))
    fig.legend(handles=[Patch(color=TEAL, label="Similar-distribution folds"),
                        Patch(color=GOLD, label="Structural-cluster stress")],
               loc="upper right", bbox_to_anchor=(.93, .88), ncol=2, frameon=False)
    fig.text(.28, .145,
             "External geometry classifier: 96.9% on MQT reference families, yet 443/532 challenge circuits lie beyond its reference range.",
             fontsize=11, color=RUST, va="center")
    title(fig, "Soft algorithm patterns help; hard family guesses do not transfer",
          "Incremental paired runtime-score change from each experiment's own baseline · percentage points",
          "Sources: geometry_family_ablation, algorithm_geometry_probe, sequence_motif_probe, shor_family_probe JSON. No challenge algorithm labels.")
    export(fig, "05_algorithm_signals")


def production_oof_scores():
    rows = list(csv.DictReader((SRC / "production_model_oof.csv").open()))
    assert len(rows) == 1497
    scores = {}
    for split, field in (("matched", "matched_pred_s"),
                         ("structural", "structural_pred_s")):
        values = []
        for row in rows:
            actual = float(row["actual_s"])
            pred = float(row[field])
            if row["status"] == "timeout":
                pred = min(pred, 14400.0)
            values.append(max(0.0, 1-abs(math.log10(max(1e-9,pred)/actual))/2))
        scores[split] = float(np.mean(values))
    return scores


def plot_pruning():
    v7 = read("full_union_model_validation")
    v8 = read("categorical_model_validation")
    prod = json.loads((ROOT / "quantathon-harness/production_features.json").read_text())
    v9 = production_oof_scores()
    counts = [v7["selected_schema"]["selected_unique_columns"],
              v8["selected_schema"]["selected_unique_columns"], len(prod["columns"])]
    assert counts == [325, 241, 120]
    matched = [v8["splits"]["matched"]["current_v7"]["score"],
               v8["splits"]["matched"]["categorical_v8"]["score"], v9["matched"]]
    stress = [v8["splits"]["structural"]["current_v7"]["score"],
              v8["splits"]["structural"]["categorical_v8"]["score"], v9["structural"]]
    assert abs(matched[-1]-.927739)<1e-5 and abs(stress[-1]-.766378)<1e-5
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.08, right=.95, top=.77, bottom=.21, wspace=.25)
    labels = ["v7", "v8", "v9 production"]
    x = np.arange(3)
    bars = axes[0].bar(x, counts, width=.58, color=[GRAY_PALE, "#B1D8DC", TEAL])
    for b,n in zip(bars,counts):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+7, str(n),
                     ha="center", va="bottom", fontsize=20, fontweight="bold", color=NAVY)
    axes[0].set_ylim(0, 380)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Distinct fitted input columns")
    axes[0].set_title("A · Cut 63% of fitted inputs", loc="left", color=NAVY)
    axes[0].grid(axis="y", color=GRID)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(x, np.array(matched)*100, marker="o", markersize=11,
                 lw=2.5, color=TEAL, label="Similar-distribution")
    axes[1].plot(x, np.array(stress)*100, marker="s", markersize=10,
                 lw=2.5, color=GOLD, label="Structural stress")
    for i in range(3):
        axes[1].text(i, matched[i]*100+1.3, f"{matched[i]*100:.2f}%",
                     ha="center", color=TEAL, fontweight="bold", fontsize=13)
        axes[1].text(i, stress[i]*100-2.2, f"{stress[i]*100:.2f}%",
                     ha="center", color=GOLD, fontweight="bold", fontsize=13)
    axes[1].set_ylim(70, 97)
    axes[1].set_yticks([70, 75, 80, 85, 90, 95], ["70%", "75%", "80%", "85%", "90%", "95%"])
    axes[1].set_xticks(x, labels)
    axes[1].set_title("B · Grouped score holds", loc="left", color=NAVY)
    axes[1].grid(axis="y", color=GRID)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="center left", frameon=False, fontsize=11)
    title(fig, "Final pruning makes the production model smaller",
          "Same 1,497 labeled rows; 532 circuits; setting remains categorical in v8/v9",
          "Sources: full_union_model_validation.json, categorical_model_validation.json, production_model_oof.csv, production_features.json."
          " Feature selection reused released labels, so small gains may be optimistic.")
    export(fig, "06_final_pruning")


def card(ax, x, y, width, height, heading, lines, *, selected=(), index=0):
    box = FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.03,rounding_size=.14",
                         lw=1.2, ec=GRID, fc="white")
    ax.add_patch(box)
    ax.add_patch(plt.Rectangle((x,y+height-.67), width, .67,
                               facecolor=NAVY if index==0 else TEAL,
                               edgecolor="none"))
    ax.text(x+.22,y+height-.34,heading,va="center",ha="left",
            color="white",fontsize=14,fontweight="bold")
    spacing=(height-.9)/len(lines)
    for j,line in enumerate(lines):
        yy=y+height-.83-j*spacing
        if j in selected:
            ax.add_patch(plt.Rectangle((x+.09, yy-spacing+.10), width-.18,
                                       spacing-.04, fc=TEAL_PALE, ec="none"))
        ax.text(x+.20,yy-.10,line,ha="left",va="top",fontsize=10.2,
                color=NAVY if j in selected else MUTED,
                fontweight="bold" if j in selected else "normal")


def plot_feature_tree():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0,18);ax.set_ylim(0,10);ax.axis("off")
    specs=[
        ("QASM basics", ["Basic counts + depth", "Raw gate + connectivity",
                         "Simplification + timing", "Ridge / RF / HistGB",
                         "ExtraTrees selected"], (4,)),
        ("χ + diversity", ["χ upper only", "Diversity only", "χ + diversity",
                            "Effective χ only", "+ peak effective χ",
                            "+ all effective χ", "χ × randomness",
                            "75/25 effective-χ blend"], (2,7)),
        ("Geometry", ["Current χ baseline", "Cut timeline", "Interaction graph",
                       "Soft patterns", "Dense / liveness", "Timeline + graph",
                       "All groups", "All + effective χ"], (3,6)),
        ("χ walk", ["Cost p=2", "Cost p=2.5", "Cost p=3",
                     "Cost + overhead", "High overhead", "Shape only",
                     "Shape + cost", "Cost + uncapped"], (6,)),
        ("Angle gate", ["Original walk", "Original + angle counts",
                        "Near-0/π filtered", "Filtered + angle counts"], (3,)),
    ]
    width=3.25; gap=.31; start=.31; y=1.05; height=7.8
    xs=[start+i*(width+gap) for i in range(5)]
    for i,(heading,lines,selected) in enumerate(specs):
        card(ax,xs[i],y,width,height,heading,lines,selected=selected,index=i)
        if i<4:
            ax.add_patch(FancyArrowPatch((xs[i]+width+.02,y+height/2),
                                         (xs[i+1]-.05,y+height/2),
                                         arrowstyle="-|>", mutation_scale=16,
                                         lw=2,color=GOLD))
    fig.text(.055,.965,"Feature experiment tree",fontsize=25,fontweight="bold",
             color=NAVY,va="top")
    fig.text(.055,.92,"Selected packages in teal · other rows are paired ablations",
             fontsize=14,color=MUTED,va="top")
    fig.text(.055,.05,"Initial 12-model sweep is chart 01. An unhighlighted component may enter a later bundle; later ablations held ExtraTrees fixed.",
             fontsize=11,color=MUTED,va="bottom")
    export(fig,"07_feature_experiment_tree")


def architecture_box(ax,x,y,w,h,head,body,color=TEAL):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.03,rounding_size=.14",
                     fc="white",ec=color,lw=1.6)
    ax.add_patch(p)
    ax.text(x+.16,y+h-.30,head,fontsize=12.2,fontweight="bold",color=color,va="top")
    ax.text(x+.16,y+h-.82,body,fontsize=10,color=INK,va="top",linespacing=1.5)


def plot_model_tree():
    fig,ax=plt.subplots(figsize=(18,10))
    ax.set_xlim(0,18);ax.set_ylim(0,10);ax.axis("off")
    fig.text(.055,.965,"Model tree: from a stock regressor to the production system",
             fontsize=24,fontweight="bold",color=NAVY,va="top")
    fig.text(.055,.92,"The broad estimator sweep came first; later gains primarily came from representations and routing",
             fontsize=13,color=MUTED,va="top")
    main=[
        ("ExtraTrees", "Best of 4 initial\nregressor families\nlog₁₀(runtime)"),
        ("Feature-rich global", "QASM geometry + χ\nwalk + angle gate\nrefit on grouped folds"),
        ("Dual parser", "Primary + secondary\nDAG/angle scanner\n480 candidates"),
        ("Threshold experts", "One ExtraTrees per\nsetting; 50/50 log\nblend with global"),
        ("Timeout router", "One classifier per\nsetting; cap at\np(timeout) ≥ .35"),
        ("v9 production", "v7 325 → v8 241\n→ v9 120 inputs\ngrouped OOF 0.9277"),
    ]
    w=2.65;gap=.28;x0=.34;y=5.15;h=2.6
    for i,(head,body) in enumerate(main):
        x=x0+i*(w+gap)
        architecture_box(ax,x,y,w,h,head,body,NAVY if i==0 else TEAL)
        if i<5:
            ax.add_patch(FancyArrowPatch((x+w+.02,y+h/2),(x+w+gap-.05,y+h/2),
                                         arrowstyle="-|>",mutation_scale=17,lw=2,color=GOLD))
    alternatives=[
        ("Ridge / RF / HistGB", "Lower scores in the\ninitial 12-model sweep"),
        ("Compact union", "0.9197 matched;\nstronger structural;\nnot selected"),
        ("Family neural", "FiLM residual 0.9009;\nbelow tree baseline"),
        ("JEPA embedding", "Did not pass paired\nacceptance gate"),
        ("Hard family labels", "MQT geometry classes\nreduced runtime score"),
        ("Extra rules tested", "Monotonic threshold\nand neighbor fixes\nrejected"),
    ]
    for i,(head,body) in enumerate(alternatives):
        x=x0+i*(w+gap)
        architecture_box(ax,x,1.55,w,2.05,head,body,MUTED)
    ax.text(.35,4.34,"other branches tested / not shipped",fontsize=12,color=MUTED,fontweight="bold")
    fig.text(.055,.052,"Sources: validation.json, merged_model_validation.json, full_union_model_validation.json,"
             " paper_runtime_replication.json, jepa/results.json, production_model_oof.csv.",
             fontsize=10,color=MUTED,va="bottom")
    export(fig,"08_model_architecture_tree")


def main():
    plot_initial_model_sweep()
    plot_foundations()
    plot_chi_walk()
    plot_angle_gating()
    plot_algorithm_signals()
    plot_pruning()
    plot_feature_tree()
    plot_model_tree()
    print(f"Exported 8 PNG + 8 SVG figures to {OUT}")


if __name__ == "__main__":
    main()
