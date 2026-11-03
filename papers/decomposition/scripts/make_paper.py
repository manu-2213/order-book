"""The paper's tables and figures, generated from the committed result summaries so that nothing is copied by hand.

    python -m papers.decomposition.scripts.make_paper --refresh    # copy summaries from outputs/ and recompute the per-seed tails
    python -m papers.decomposition.scripts.make_paper              # tables and figures from results/full/ alone

Writes papers/decomposition/paper/generated/. Build with: cd papers/decomposition/paper && latexmk -pdf paper.tex
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from papers.decomposition import OUT
from papers.decomposition.scripts.make_tables import AXIS_LABELS, build, ci, code, executor, horizons_table, interval, phi_cells
from papers.decomposition.scripts.runaway_causes import RUNAWAY_TICKS, VARIANTS

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results" / "full"
GENERATED = HERE / "paper" / "generated"

GRIDS = {"deep": "e8_deep_cover", "low": "e8_low_cover", "horizons": "e8_horizons",
         "deep_cap": "e8_capital_deep", "low_cap": "e8_capital_low", "horizons_cap": "e8_capital_horizons"}
OTHERS = ("protocol", "detection_curve", "detection_curve_deep", "replications", "runaway_causes")
PENDING = ("rl_eval", "e8_rl")   # the second-generation policies; their tables appear once the cluster run is back
EXECUTORS = ("twap_market", "twap_passive", "twap_random", "pov", "almgren_chriss_1e-05", "almgren_chriss_1e-06", "almgren_chriss_1e-07",
             "obizhaeva_wang", "immediate", "limit_market", "mimic", "footprint")
TAIL_CELLS = (("twap_market", 0.4, 2000), ("twap_market", 0.4, 8000), ("pov", 0.4, 2000), ("immediate", 0.4, 2000))


# ----- from outputs/ to results/full/ --------------------------------------------------------- #

def tails(grid: str) -> dict:
    """Per cell: the mean, median and largest shortfall with all channels on, and how many seeds ran away."""
    cells: dict = {}
    with open(OUT / grid / "episodes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["channels"] not in ("----", "IPDA"):
                continue
            cells.setdefault((r["strategy"], r["size_fraction"], r["horizon"]), {}).setdefault(r["channels"], []).append(r)
    out = {}
    for (strategy, size, horizon), arms in cells.items():
        full, ghost = arms["IPDA"], arms["----"]
        cost = np.array([r["shortfall_bps"] for r in full], float)
        out[f"{strategy}|{size}|{horizon}"] = {
            "mean": float(np.nanmean(cost)), "median": float(np.nanmedian(cost)), "max": float(np.nanmax(cost)),
            "runaways": sum(r["impact_peak"] > RUNAWAY_TICKS for r in full), "quantity": float(np.mean([r["quantity"] for r in full])),
            "per_seed": [float(c) for c in cost] if (strategy, size, horizon) in TAIL_CELLS else None,
            "impact": {"arm1": [float(np.mean([r["q_over_v"] for r in ghost])), float(np.mean([r["impact_mean"] for r in ghost]))],
                       "arm5": [float(np.mean([r["q_over_v"] for r in full])), float(np.mean([r["impact_mean"] for r in full]))]},
        }
    return out


def refresh() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in (*GRIDS.values(), *OTHERS, *PENDING):
        if name in PENDING and not (OUT / name / "summary.json").exists():
            continue
        (RESULTS / name).mkdir(exist_ok=True)
        for file in ("summary.json", "manifest.json"):
            if (OUT / name / file).exists():
                shutil.copy(OUT / name / file, RESULTS / name / file)
    (RESULTS / "tails.json").write_text(json.dumps({key: tails(grid) for key, grid in GRIDS.items()}))
    heldout = OUT.parent / "stealth_execution" / "rl_heldout" / "summary.json"
    (RESULTS / "rl_heldout").mkdir(exist_ok=True)
    shutil.copy(heldout, RESULTS / "rl_heldout" / "summary.json")
    print(f"refreshed {RESULTS}")


# ----- tables --------------------------------------------------------------------------------- #

def load(name: str) -> dict:
    return json.loads((RESULTS / name / "summary.json").read_text())


def share(r: dict) -> str:
    return f"{r['reaction_share'][0]:.2f} [{r['reaction_share'][1]:.2f}, {r['reaction_share'][2]:.2f}]"


def row_key(run: dict, strategy: str, size, horizon) -> str:
    prefix = next(iter(run["summary"]["rows"])).split("|")[0]
    return f"{prefix}|{strategy}|{size}|{horizon}"


def table(caption: str, label: str, columns: str, lines: list[str], note: str = "") -> str:
    body = "\n".join(lines)
    note = rf"\par\smallskip\begin{{minipage}}{{\linewidth}}\footnotesize {note}\end{{minipage}}" if note else ""
    return (rf"\begin{{table}}[!htbp]\centering\small\caption{{{caption}}}\label{{{label}}}" "\n"
            rf"\begin{{adjustbox}}{{max width=\linewidth}}\begin{{tabular}}{{{columns}}}\toprule" "\n" + body + "\n"
            rf"\bottomrule\end{{tabular}}\end{{adjustbox}}{note}\end{{table}}")


def twap_table(deep: dict, cap: dict) -> str:
    head = r"Nominal & Realised & Mechanical & Reaction & Share & $\phi_I$ & $\phi_P$ & $\phi_D$ & $\phi_A$\\\midrule"
    lines = [head]
    for title, run in (("Panel A: main market, seeds 1000--1019", deep), ("Panel B: ten times the capital, seeds 1020--1039", cap)):
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{title}}}}}\\")
        for size in ("0.02", "0.05", "0.1", "0.2", "0.4"):
            r = run["summary"]["rows"][row_key(run, "twap_market", size, 2000)]
            lines.append(f"{float(size):.2f} & {r['realised_q_over_v']:.3f} & {r['mechanical'][0]:.2f} & {ci(r['reaction'])} & {share(r)} & {phi_cells(r)}\\\\")
    return table(r"The decomposition of \code{twap\_market}, deep-cover market, $T = 2000$. Costs in basis points, with 95\% intervals over seeds. The share is the reaction over cost net of natural drift.",
                 "tab:twap", "@{}rrrllrrrr@{}", lines)


def executors_table(deep: dict, cap: dict, tail: dict, tail_cap: dict) -> str:
    lines = [r" & \multicolumn{5}{c}{Nominal $Q/V = 0.2$} & \multicolumn{5}{c}{Nominal $Q/V = 0.4$}\\\cmidrule(lr){2-6}\cmidrule(l){7-11}",
             r"Executor & Mech. & Median IS & Share & Share, $10\times$ & Runaways & Mech. & Median IS & Share & Share, $10\times$ & Runaways\\\midrule"]
    for name in EXECUTORS:
        cells = []
        for size in ("0.2", "0.4"):
            r, q = deep["summary"]["rows"][row_key(deep, name, size, 2000)], cap["summary"]["rows"][row_key(cap, name, size, 2000)]
            a, b = tail[f"{name}|{size}|2000"], tail_cap[f"{name}|{size}|2000"]
            cells.append(f"{r['mechanical'][0]:.2f} & {a['median']:.1f} & {r['reaction_share'][0]:.2f} & {q['reaction_share'][0]:.2f} & {a['runaways']} / {b['runaways']}")
        lines.append(f"{code(name)} & " + " & ".join(cells) + r"\\")
    return table(r"The twelve executors, deep-cover market, $T = 2000$. Median IS is with every channel on, in basis points. Share is the reaction over cost net of drift, on means. Runaways counts the 20 seeds whose peak impact exceeds 500 ticks, in the main market and then with ten times the capital.",
                 "tab:executors", "@{}lrrrrcrrrrc@{}", lines, )


def horizon_table(main: dict, cap: dict, tail: dict, tail_cap: dict) -> str:
    lines = [r" & & & \multicolumn{3}{c}{Main market} & \multicolumn{4}{c}{Ten times the capital}\\\cmidrule(lr){4-6}\cmidrule(l){7-10}",
             r"$T$ & Nominal & Mean lots & Share & Median IS & Runaways & Reaction & Share & $\phi_P$ & Runaways\\\midrule"]
    for horizon in (500, 2000, 8000):
        for size in ("0.05", "0.2", "0.4"):
            r, q = main["summary"]["rows"][row_key(main, "twap_market", size, horizon)], cap["summary"]["rows"][row_key(cap, "twap_market", size, horizon)]
            a, b = tail[f"twap_market|{size}|{horizon}"], tail_cap[f"twap_market|{size}|{horizon}"]
            lines.append(f"{horizon} & {float(size):.2f} & {a['quantity']:,.0f} & {r['reaction_share'][0]:.2f} & {a['median']:.1f} & {a['runaways']} & "
                         f"{ci(q['reaction'])} & {share(q)} & {q['shapley']['price']['mean']:+.2f} & {b['runaways']}\\\\")
        if horizon != 8000:
            lines.append(r"\addlinespace")
    return table(r"\code{twap\_market} by horizon, deep-cover market. At fixed nominal $Q/V$ a longer horizon is also a larger order in lots.", "tab:horizon", "@{}rrrrrrllrr@{}", lines)


def concavity_table(deep: dict, cap: dict, main_h: dict, cap_h: dict) -> str:
    lines = [r"Market, sizes & $T$ & Executor & $\delta_{\text{mean}}$, Arm 1 & $\delta_{\text{mean}}$, Arm 5 & Difference & $\delta_{\text{peak}}$ difference\\\midrule"]
    for title, run, horizons in (("Main, 5", deep, (2000,)), (r"$10\times$, 5", cap, (2000,)), ("Main, 3", main_h, (500, 2000, 8000)), (r"$10\times$, 3", cap_h, (500, 2000, 8000))):
        prefix = next(iter(run["concavity"])).split("|")[0]
        for horizon in horizons:
            for name in ("twap_market", "twap_passive", "twap_random"):
                c = run["concavity"][f"{prefix}|{name}|{horizon}"]
                lines.append(f"{title} & {horizon} & {code(name)} & {c['arms']['arm1']['delta_impact_mean']:.2f} & {c['arms']['arm5']['delta_impact_mean']:.2f} & "
                             f"{interval(c['tests']['delta_impact_mean_arm1_minus_arm5'])} & {interval(c['tests']['delta_impact_peak_arm1_minus_arm5'])}\\\\")
        lines.append(r"\addlinespace")
    lines.pop()
    return table(r"Impact exponents in the deep-cover market. $\delta$ is the exponent of impact on realised $Q/V$; a positive difference, Arm~1 minus Arm~5, means reaction lowers the exponent. Intervals resample seeds and are not adjusted for multiplicity.",
                 "tab:concavity", "@{}lrlrrll@{}", lines)


def runaways_table(tails_all: dict) -> str:
    lines = [r" & \multicolumn{3}{c}{Deep cover} & \multicolumn{3}{c}{Low cover}\\\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Executor & 0.1 & 0.2 & 0.4 & 0.1 & 0.2 & 0.4\\\midrule"]
    for name in EXECUTORS:
        cells = []
        for main, cap in (("deep", "deep_cap"), ("low", "low_cap")):
            for size in ("0.1", "0.2", "0.4"):
                cells.append(f"{tails_all[main][f'{name}|{size}|2000']['runaways']} / {tails_all[cap][f'{name}|{size}|2000']['runaways']}")
        if any(c != "0 / 0" for c in cells):
            lines.append(f"{code(name)} & " + " & ".join(cells) + r"\\")
    return table(r"Seeds out of 20 whose peak impact exceeds 500 ticks, $T = 2000$, by nominal $Q/V$: main market / ten times the capital. Executors with none are omitted.",
                 "tab:runaways", "@{}lcccccc@{}", lines)


def causes_table(causes: dict) -> str:
    labels = {"baseline": "As run", "no_inv_limit": "No inventory limit", "no_vol_pull": "No volatility pull", "no_switch": "Switching frozen",
              "spec_capacity": "Unlimited speculators", "noise_budget": r"Noise pool $\times 100$", "all_three": "Last three together"}
    cells = [("twap_market|0.4|8000", r"\code{twap\_market} 0.4, $T{=}8000$"), ("pov|0.4|8000", r"\code{pov} 0.4, $T{=}8000$"),
             ("immediate|0.4|2000", r"\code{immediate} 0.4"), ("limit_market|0.4|2000", r"\code{limit\_market} 0.4"), ("pov|0.4|2000", r"\code{pov} 0.4")]
    lines = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{title}}}" for _, title in cells) + r"\\" + "".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(len(cells))),
             "Intervention" + " & Mean & Runaways" * len(cells) + r"\\\midrule"]
    for variant in VARIANTS:
        lines.append(labels[variant] + "".join(f" & {causes[key][variant]['mean']:.0f} & {causes[key][variant]['runaways']}" for key, _ in cells) + r"\\")
    return table(r"What removes the runaways. Mean IS in basis points and runaways out of 20, every channel on, deep cover, seeds 1000--1019, each intervention applied after the arrival in both worlds. Descriptive: no intervals, and not run through the ghost arm.",
                 "tab:causes", "@{}l" + "rc" * len(cells) + "@{}", lines)


def routes_table(low: dict) -> str:
    lines = [r" & \multicolumn{2}{c}{Nominal $Q/V = 0.2$} & \multicolumn{2}{c}{Nominal $Q/V = 0.4$}\\\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Executor & $\Delta$ mechanical & $\Delta$ reaction & $\Delta$ mechanical & $\Delta$ reaction\\\midrule"]
    names = [k.split("|")[1] for k in low["executors"] if k.split("|")[2] == "0.2"]
    for name in names:
        if name in ("immediate", "limit_market", "pov"):  # runaways dominate these means at the larger size
            continue
        cells = [low["executors"][f"low_cover|{name}|{size}|2000"] for size in ("0.2", "0.4")]
        lines.append(f"{executor(name)} & " + " & ".join(f"{ci(e['d_mechanical'])} & {ci(e['d_reaction'])}" for e in cells) + r"\\")
    return table(r"Where executors pay, against \code{twap\_market} on the same seeds, low-cover main market, $T = 2000$, in basis points with 95\% intervals. \code{immediate}, \code{limit\_market} and \code{pov} are left out because runaways dominate their means; they are in Appendix~\ref{app:grids}.",
                 "tab:routes", "@{}lllll@{}", lines)


def ppo_names(low: dict) -> list[str]:
    return sorted({k.split("|")[1] for k in low["summary"]["rows"] if "|rl@" in k}, key=lambda name: "sz0.40" in name)


def ppo_table(low: dict) -> str:
    lines = [r"Nominal & Forced & Passive & Mech. & Reaction & Share & $\phi_I$ & $\phi_P$ & $\phi_D$ & $\phi_A$ & $\Delta$ IS & $\Delta$ mech. & $\Delta$ reaction\\\midrule"]
    sizes = ("0.02", "0.05", "0.1", "0.2", "0.4")
    for name, title in ((n, f"Panel: {executor(n)}") for n in ppo_names(low)):
        lines.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{title}}}}}\\")
        for size in sizes:
            r, e = low["summary"]["rows"][f"low_cover|{name}|{size}|2000"], low["executors"][f"low_cover|{name}|{size}|2000"]
            lines.append(f"{float(size):.2f} & {r['forced_share']:.3f} & {1 - r['aggressive_share']:.2f} & {r['mechanical'][0]:.2f} & {ci(r['reaction'])} & {r['reaction_share'][0]:.2f} & "
                         f"{phi_cells(r)} & {ci(e['d_is'])} & {ci(e['d_mechanical'])} & {ci(e['d_reaction'])}\\\\")
    lines.append(r"\multicolumn{13}{@{}l}{\emph{Panel: \code{twap\_market}, the reference}}\\")
    for size in sizes:
        r = low["summary"]["rows"][f"low_cover|twap_market|{size}|2000"]
        lines.append(f"{float(size):.2f} & {r['forced_share']:.3f} & {1 - r['aggressive_share']:.2f} & {r['mechanical'][0]:.2f} & {ci(r['reaction'])} & {r['reaction_share'][0]:.2f} & "
                     f"{phi_cells(r)} & & & \\\\")
    return table(r"The decomposition of the PPO policies, low-cover main market, $T = 2000$, seeds 1000--1019, in basis points with 95\% intervals. Forced is the share of the order pushed through at the horizon, Passive the share of volume not sent aggressively. $\Delta$ is against \code{twap\_market} on the same seeds.",
                 "tab:ppo", "@{}rrrrlrrrrrlll@{}", lines)


def rl_eval_table(rows: list[dict]) -> str:
    """The second-generation policies against TWAP: validation seeds, then unseen seeds."""
    lines = [r"Policy & Market & Nominal & \multicolumn{2}{c}{IS, policy $-$ TWAP} & \multicolumn{2}{c}{Net of drift, policy $-$ TWAP} & \multicolumn{2}{c}{Exposure}\\",
             r" & & & Validation & Unseen & Validation & Unseen & Policy & TWAP\\\midrule"]
    by = {(r["name"], r["regime"], r["size"], r["seeds"]): r for r in rows}
    for name, regime, size in sorted({(r["name"], r["regime"], r["size"]) for r in rows}):
        v, u = by[(name, regime, size, "validation")], by[(name, regime, size, "unseen")]
        gap = lambda r, key: ci({"mean": r[key], "low": r[f"{key}_low"], "high": r[f"{key}_high"]})
        lines.append(f"{code(name)} & {code(regime)} & {size:.2f} & {gap(v, 'gap_is')} & {gap(u, 'gap_is')} & {gap(v, 'gap_net')} & {gap(u, 'gap_net')} & {u['exposure']:.3f} & {u['twap_exposure']:.3f}\\\\")
    return table(r"The second-generation policies against \code{twap\_market} on the same episodes, greedy, $T = 2000$, in basis points with 95\% intervals. Validation is seeds 800--879, on which the checkpoint was chosen; unseen is seeds 1000--1079, the estimate to quote. Exposure is the mean signal of the makers' detector in the executor's direction.",
                 "tab:rl_eval", "@{}lllllllrr@{}", lines)


def e8_rl_table(run: dict) -> str:
    """The decomposition of the second-generation policies next to TWAP, low cover with ten times the capital."""
    lines = [r"Executor & Nominal & Passive & Mech. & Reaction & Share & $\phi_I$ & $\phi_P$ & $\phi_D$ & $\phi_A$ & $\Delta$ IS & $\Delta$ mech. & $\Delta$ reaction\\\midrule"]
    for key, r in run["summary"]["rows"].items():
        _, strategy, size, _ = key.split("|")
        label = code(strategy.split("/")[-2]) if strategy.startswith("rl@") else code(strategy)
        e = run["executors"].get(key)
        deltas = f"{ci(e['d_is'])} & {ci(e['d_mechanical'])} & {ci(e['d_reaction'])}" if e else " & & "
        lines.append(f"{label} & {float(size):.2f} & {1 - r['aggressive_share']:.2f} & {r['mechanical'][0]:.2f} & {ci(r['reaction'])} & {r['reaction_share'][0]:.2f} & "
                     f"{phi_cells(r)} & {deltas}\\\\")
    return table(r"The decomposition of the second-generation policies, low-cover market with ten times the capital, $T = 2000$, seeds 1020--1039, in basis points with 95\% intervals. $\Delta$ is against \code{twap\_market} on the same seeds.",
                 "tab:e8_rl", "@{}lrrrlrrrrrlll@{}", lines)


def curve_pair_table(low: dict, deep: dict) -> str:
    lines = [r" & \multicolumn{3}{c}{Low cover} & \multicolumn{3}{c}{Deep cover}\\\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Axis, value & Flow signal & A3 signal & Leakage & Flow signal & A3 signal & Leakage\\\midrule"]
    base = lambda c: c["medium"]["axes"]["participation"][0]["detectors"]
    a, b = base(low), base(deep)
    lines.append(f"Arm 0 & {a['flow_medium']['signal_arm0']:+.3f} & {a['learned']['signal_arm0']:+.3f} & & {b['flow_medium']['signal_arm0']:+.3f} & {b['learned']['signal_arm0']:+.3f} & \\\\\\midrule")
    for axis, label in AXIS_LABELS.items():
        for x, y in zip(low["medium"]["axes"][axis], deep["medium"]["axes"][axis]):
            cell = lambda r: f"{r['detectors']['flow_medium']['signal'][0]:+.3f} & {r['detectors']['learned']['signal'][0]:+.3f} & {ci(r['leakage'])}"
            lines.append(f"{label} {x['value']:.2f} & {cell(x)} & {cell(y)}\\\\")
        lines.append(r"\addlinespace")
    lines.pop()
    return table(r"The detection curve at the medium threshold. Signal is the mean detector signal in the executor's direction, Leakage the cost of being visible to the flow detector, in basis points. Each axis moves one dial from a plain TWAP at nominal $Q/V = 0.2$.",
                 "tab:curve", "@{}lrrlrrl@{}", lines)


def appendix_tables(runs: dict, proto, curve, rep, held) -> dict:
    """The full grids from make_tables, with the captions and labels of the capital runs told apart."""
    tables = build(runs["deep"], runs["low"], proto, curve, rep, held)
    capital = build(runs["deep_cap"], runs["low_cap"], proto, curve, rep, held)
    mark = lambda text, old, new: text.replace(old, new)
    out = {"protocol": tables["protocol"], "replications": tables["replications"], "ppo_heldout": tables["ppo_heldout"], "split": tables["split"],
           "app_deep": tables["deep"], "app_low": tables["low"], "app_horizons": horizons_table(runs["horizons"]),
           "app_deep_cap": mark(mark(capital["deep"], "deep-cover market", "deep-cover market, ten times the capital, seeds 1020--1039"), "tab:e8_deep", "tab:e8_deep_cap"),
           "app_low_cap": mark(mark(capital["low"], "low-cover market", "low-cover market, ten times the capital, seeds 1020--1039"), "tab:e8_low", "tab:e8_low_cap"),
           "app_horizons_cap": mark(mark(horizons_table(runs["horizons_cap"]), "deep-cover market", "deep-cover market, ten times the capital, seeds 1020--1039"), "tab:e8_horizons", "tab:e8_horizons_cap")}
    return out


# ----- figures -------------------------------------------------------------------------------- #

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical slots in fixed order, each with its own marker or position
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dc"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "axes.titlesize": 8.5, "axes.titlelocation": "left"})
    return plt


def figure_decomposition(deep: dict, cap: dict, path: Path) -> None:
    plt = style()
    sizes = ("0.02", "0.05", "0.1", "0.2", "0.4")
    parts = (("Mechanical", None), ("Price", "price"), ("Inventory", "inventory"), ("Detection", "detection"), ("Predation", "predation"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    for ax, run, title in zip(axes, (deep, cap), ("Main market", "Ten times the capital")):
        rows = [run["summary"]["rows"][row_key(run, "twap_market", s, 2000)] for s in sizes]
        x = np.arange(len(sizes))
        up, down = np.zeros(len(sizes)), np.zeros(len(sizes))
        for i, (label, key) in enumerate(parts):
            values = np.array([r["mechanical"][0] if key is None else r["shapley"][key]["mean"] for r in rows])
            base = np.where(values >= 0, up, down)
            ax.bar(x, values, 0.62, bottom=base, color=SERIES[i], label=label, edgecolor="white", linewidth=0.8)
            up, down = up + np.maximum(values, 0), down + np.minimum(values, 0)
        ax.set_xticks(x, [f"{r['realised_q_over_v'] * 100:.1f}%" for r in rows])
        ax.set_xlabel("Realised participation")
        ax.set_title(title)
        ax.axhline(0, color=MUTED, linewidth=0.6)
    axes[0].set_ylabel("Cost net of drift (bps)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_horizon(cap: dict, path: Path) -> None:
    plt = style()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    horizons = (500, 2000, 8000)
    for i, (size, marker) in enumerate((("0.05", "o"), ("0.2", "s"), ("0.4", "^"))):
        rows = [cap["summary"]["rows"][row_key(cap, "twap_market", size, h)]["reaction_share"] for h in horizons]
        values = np.array(rows)
        ax.errorbar(range(3), values[:, 0], yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]], color=SERIES[i], marker=marker, markersize=4, linewidth=1.4,
                    capsize=2, label=f"Nominal {float(size):.2f}")
    ax.set_xticks(range(3), [f"{h:,}" for h in horizons])
    ax.set_xlabel("Horizon $T$ (steps)")
    ax.set_ylabel("Reaction share of net cost")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_tails(tails_all: dict, path: Path) -> None:
    plt = style()
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    cells = (("twap_market|0.4|2000", "deep", "deep_cap", "TWAP 0.4\n$T=2000$"), ("twap_market|0.4|8000", "horizons", "horizons_cap", "TWAP 0.4\n$T=8000$"),
             ("pov|0.4|2000", "deep", "deep_cap", "POV 0.4\n$T=2000$"), ("immediate|0.4|2000", "deep", "deep_cap", "Immediate 0.4\n$T=2000$"))
    rng = np.random.default_rng(0)
    for i, (key, main, cap, _) in enumerate(cells):
        for offset, grid, color, marker in ((-0.17, main, SERIES[0], "o"), (0.17, cap, SERIES[1], "s")):
            t = tails_all[grid][key]
            ax.scatter(i + offset + rng.uniform(-0.07, 0.07, len(t["per_seed"])), t["per_seed"], s=11, color=color, marker=marker, alpha=0.75, linewidths=0)
            ax.plot([i + offset - 0.13, i + offset + 0.13], [t["median"]] * 2, color=INK, linewidth=1.3)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([-10, 0, 10, 100, 1000], ["-10", "0", "10", "100", "1,000"])
    ax.set_xticks(range(len(cells)), [c[3] for c in cells])
    ax.set_ylabel("IS, all channels on (bps)")
    ax.scatter([], [], s=11, color=SERIES[0], marker="o", label="Main market")
    ax.scatter([], [], s=11, color=SERIES[1], marker="s", label="Ten times the capital")
    ax.plot([], [], color=INK, linewidth=1.3, label="Median")
    ax.set_ylim(top=30_000)  # room for the legend above the largest seed
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_exponents(deep_cap: dict, cap_h: dict, tails_all: dict, path: Path) -> None:
    plt = style()
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, grid, run, horizon, sizes, title in ((axes[0], "deep_cap", deep_cap, 2000, ("0.02", "0.05", "0.1", "0.2", "0.4"), "$T=2000$, five sizes"),
                                                 (axes[1], "horizons_cap", cap_h, 8000, ("0.05", "0.2", "0.4"), "$T=8000$, three sizes")):
        prefix = next(iter(run["concavity"])).split("|")[0]
        arms = run["concavity"][f"{prefix}|twap_market|{horizon}"]["arms"]
        for i, (arm, marker, label) in enumerate((("arm1", "o", "Ghost, Arm 1"), ("arm5", "s", "Full market, Arm 5"))):
            points = np.array([tails_all[grid][f"twap_market|{s}|{horizon}"]["impact"][arm] for s in sizes])
            ax.plot(points[:, 0], points[:, 1], color=SERIES[i], marker=marker, markersize=4, linewidth=1.4, label=rf"{label}, $\delta={arms[arm]['delta_impact_mean']:.2f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = (0.02, 0.05, 0.1, 0.2, 0.4)
        ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Realised $Q/V$")
        ax.set_title(title)
        ax.grid(True, which="major", axis="both")
        ax.legend(loc="center right")  # the empty band between the two lines
    axes[0].set_ylabel("Mean impact while executing (ticks)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_ppo(low: dict, path: Path) -> None:
    plt = style()
    sizes = ("0.02", "0.05", "0.1", "0.2", "0.4")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4))
    for ax, field, title in zip(axes, ("d_mechanical", "d_reaction"), ("Change in mechanical cost (bps)", "Change in reaction (bps)")):
        for i, (name, marker, label) in enumerate(zip(ppo_names(low), ("o", "s"), ("Trained at 0.20", "Trained at 0.40"))):
            cells = [low["executors"][f"low_cover|{name}|{size}|2000"][field] for size in sizes]
            mean = np.array([c["mean"] for c in cells])
            ax.errorbar(np.arange(len(sizes)) + (i - 0.5) * 0.18, mean, yerr=[mean - [c["low"] for c in cells], [c["high"] for c in cells] - mean], color=SERIES[i], marker=marker,
                        markersize=4, linewidth=0, elinewidth=1.2, capsize=2, label=label)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(range(len(sizes)), [f"{float(s):.2f}" for s in sizes])
        ax.set_xlabel("Nominal $Q/V$")
        ax.set_title(title)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="copy summaries from outputs/ and recompute the tails; needs the episode logs")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    runs = {key: load(grid) for key, grid in GRIDS.items()}
    tails_all = json.loads((RESULTS / "tails.json").read_text())
    proto, curve, deep_curve, rep = load("protocol")["summary"], load("detection_curve")["summary"], load("detection_curve_deep")["summary"], load("replications")
    held = load("rl_heldout")
    GENERATED.mkdir(parents=True, exist_ok=True)
    tables = appendix_tables(runs, proto, curve, rep, held)
    tables.update(twap=twap_table(runs["deep"], runs["deep_cap"]), executors=executors_table(runs["deep"], runs["deep_cap"], tails_all["deep"], tails_all["deep_cap"]),
                  horizon=horizon_table(runs["horizons"], runs["horizons_cap"], tails_all["horizons"], tails_all["horizons_cap"]),
                  concavity=concavity_table(runs["deep"], runs["deep_cap"], runs["horizons"], runs["horizons_cap"]), runaways=runaways_table(tails_all),
                  causes=causes_table(load("runaway_causes")["summary"]), routes=routes_table(runs["low"]), ppo=ppo_table(runs["low"]), curve=curve_pair_table(curve, deep_curve))
    pending = {"rl_eval": lambda: rl_eval_table(load("rl_eval")), "e8_rl": lambda: e8_rl_table(load("e8_rl"))}
    for name, make in pending.items():
        tables[name] = make() if (RESULTS / name / "summary.json").exists() else f"% {name}: no results yet; make_paper --refresh picks them up from outputs/decomposition/{name}/"
    for name, text in tables.items():
        (GENERATED / f"{name}.tex").write_text(re.sub(r"[ \t]+\n", "\n", text) + "\n")
    figure_decomposition(runs["deep"], runs["deep_cap"], GENERATED / "decomposition.pdf")
    figure_horizon(runs["horizons_cap"], GENERATED / "horizon.pdf")
    figure_tails(tails_all, GENERATED / "tails.pdf")
    figure_exponents(runs["deep_cap"], runs["horizons_cap"], tails_all, GENERATED / "exponents.pdf")
    figure_ppo(runs["low"], GENERATED / "ppo.pdf")
    print(f"wrote {len(tables)} tables and 5 figures to {GENERATED}")


if __name__ == "__main__":
    main()
