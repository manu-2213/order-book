"""Every table of the paper notes, generated from the result summaries so that nothing is copied by hand.

    python -m papers.decomposition.scripts.make_tables                         # from the pilot in results/pilot
    python -m papers.decomposition.scripts.make_tables --root outputs/decomposition --deep e8_deep_cover

Writes papers/decomposition/paper/tables/*.tex. Rebuild the notes afterwards with latexmk; the prose around the
tables quotes the pilot and has to be revised by hand when the full run replaces it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def _begin(caption_label: str, columns: str) -> list[str]:
    return [r"\begin{table}[!htbp]\centering\footnotesize", caption_label, r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{columns}}}\toprule"]


END = r"\bottomrule\end{tabular}\end{adjustbox}\end{table}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(HERE / "results" / "pilot"), help="folder holding one folder per experiment")
    parser.add_argument("--deep", default="e8_pilot", help="the deep-cover E8 run under --root")
    parser.add_argument("--low", default="e8_low_cover", help="the low-cover E8 run under --root")
    parser.add_argument("--heldout", default=None, help="rl_heldout summary; default: rl_heldout under --root")
    args = parser.parse_args()
    root = Path(args.root)
    load = lambda name: json.loads((root / name / "summary.json").read_text())
    pilot, low = load(args.deep), load(args.low)
    proto, curve, rep = load("protocol")["summary"], load("detection_curve")["summary"], load("replications")
    held = json.loads(Path(args.heldout).read_text()) if args.heldout else load("rl_heldout")
    tables = build(pilot, low, proto, curve, rep, held)
    out = HERE / "paper" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (out / f"{name}.tex").write_text(text + "\n")
    print(f"wrote {len(tables)} tables to {out}")


def ci(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['low']:+.2f}, {d['high']:+.2f}]"


def code(name: str) -> str:
    return r"\code{" + name.replace("_", "\\_") + "}"


def executor(name: str) -> str:
    if "sz0.20" in name:
        return "PPO (trained at 0.20)"
    if "sz0.40" in name:
        return "PPO (trained at 0.40)"
    return code(name)


def build(pilot, low, proto, curve, rep, held) -> dict:
    tables = {}

    lines = _begin(r"\caption{The decomposition, deep-cover market, $T=2000$. Costs in bps. ``Real'' is $Q/V$ realised against Arm~0's volume. Reaction with its 95\% CI.}\label{tab:e8_deep}", "@{}lrrrrrlrrrrr@{}")
    lines.append(r"Executor & $Q/V$ & Real & IS & Drift & Mech. & Reaction & Share & $\phi_I$ & $\phi_P$ & $\phi_D$ & $\phi_A$\\\midrule")
    for key, r in pilot["summary"]["rows"].items():
        _, strategy, size, _ = key.split("|")
        p = r["shapley"]
        lines.append(f"{code(strategy)} & {float(size):.2f} & {r['realised_q_over_v']:.3f} & {r['is']['arm5'][0]:.2f} & {r['natural_drift'][0]:.2f} & "
                     f"{r['mechanical'][0]:.2f} & {ci(r['reaction'])} & {r['reaction_share'][0]:.2f} & {p['inventory']['mean']:+.2f} & "
                     f"{p['price']['mean']:+.2f} & {p['detection']['mean']:+.2f} & {p['predation']['mean']:+.2f}\\\\")
    tables["deep"] = "\n".join(lines + [END])

    lines = _begin(r"\caption{Inside the reaction for \code{twap\_market}, deep cover. Left: inventory's increment depends on when it joins. Right: IS with the group live minus IS with the group blind, in Arm~2. These controls do not add up: removing fundamentalists lets the price run away.}\label{tab:e8_split}", "@{}rrrlll@{}")
    lines.append(r"$Q/V$ & Inv.\ after price & Inv.\ before price & Chartists & Makers & Fundamentalists\\\midrule")
    for key, r in pilot["summary"]["rows"].items():
        if "|twap_market|" not in key:
            continue
        split = r["price_split"]
        lines.append(f"{float(key.split('|')[2]):.2f} & {r['ladder']['inventory']['mean']:+.2f} & {r['ladder_plan_order']['inventory']['mean']:+.2f} & "
                     f"{ci(split['chartists'])} & {ci(split['makers'])} & {ci(split['fundamentalists'])}\\\\")
    tables["split"] = "\n".join(lines + [END])

    a = pilot["concavity"][next(k for k in pilot["concavity"] if "|twap_market|" in k)]
    b = pilot["concavity"][next(k for k in pilot["concavity"] if "|twap_passive|" in k)]
    lines = _begin(r"\caption{Impact exponent $\delta$ on four measures and reversion per arm, deep cover. Below: the tests, with bootstrap intervals over seeds.}\label{tab:e8_concavity}", "@{}lrrrrr|rrrrr@{}")
    lines.append(r" & \multicolumn{5}{c|}{\code{twap\_market}} & \multicolumn{5}{c}{\code{twap\_passive}}\\ Arm & $\delta_{\text{mean}}$ & $\delta_{\text{peak}}$ & $\delta_{\text{end}}$ & $\delta_{\text{cost}}$ & Rev. & $\delta_{\text{mean}}$ & $\delta_{\text{peak}}$ & $\delta_{\text{end}}$ & $\delta_{\text{cost}}$ & Rev.\\\midrule")
    arm = lambda x: f"{x['delta_impact_mean']:.2f} & {x['delta_impact_peak']:.2f} & {x['delta_impact_end']:.2f} & {x['delta_cost']:.2f} & {x['reversion']:.2f}"
    for k in range(1, 6):
        lines.append(f"{k} & {arm(a['arms'][f'arm{k}'])} & {arm(b['arms'][f'arm{k}'])}\\\\")
    lines.append(r"\midrule")
    interval = lambda t: f"{t['estimate']:+.2f} [{t['low']:+.2f}, {t['high']:+.2f}]"
    for name, label in (("delta_impact_mean_arm1_minus_arm5", r"$\delta_{\text{mean}}$, Arm 1 $-$ Arm 5"), ("delta_impact_peak_arm1_minus_arm5", r"$\delta_{\text{peak}}$, Arm 1 $-$ Arm 5"),
                        ("delta_cost_arm1_minus_arm5", r"$\delta_{\text{cost}}$, Arm 1 $-$ Arm 5"), ("reversion_arm5_minus_arm3", "Reversion, Arm 5 $-$ Arm 3")):
        lines.append(rf"\multicolumn{{3}}{{@{{}}l}}{{{label}}} & \multicolumn{{3}}{{l|}}{{{interval(a['tests'][name])}}} & \multicolumn{{5}}{{l}}{{{interval(b['tests'][name])}}}\\")
    tables["concavity"] = "\n".join(lines + [END])

    lines = _begin(r"\caption{The executors in the low-cover market. ``Net'' is IS minus drift. Differences are against \code{twap\_market} on the same seeds. Very large values come from a few runaway seeds; see the caveats.}\label{tab:e8_low}", "@{}lrrrrll@{}")
    lines.append(r"Executor & $Q/V$ & Net & Mech. & Share & $\Delta$ mech. & $\Delta$ reaction\\\midrule")
    for key, r in low["summary"]["rows"].items():
        _, strategy, size, _ = key.split("|")
        e = low["executors"].get(key)
        net = r["is"]["arm5"][0] - r["natural_drift"][0]
        lines.append(f"{executor(strategy)} & {float(size):.2f} & {net:.2f} & {r['mechanical'][0]:.2f} & {r['reaction_share'][0]:.2f} & "
                     f"{ci(e['d_mechanical']) if e else '--'} & {ci(e['d_reaction']) if e else '--'}\\\\")
    tables["low"] = "\n".join(lines + [END])

    lines = _begin(r"\caption{The protocol. Left: median variance reduction over cells per contrast. Right: background order flow shared by Arm~0 and Arm~5, per fifth of the window.}\label{tab:protocol}", r"@{}lrr@{\hskip 2em}lrrrrr@{}")
    lines.append(r"Contrast & Stateful & Counter & Scheme & 1st & 2nd & 3rd & 4th & 5th\\\midrule")
    coupling = {mode: np.mean(list(by.values()), axis=0) for mode, by in proto["flow_coupling"].items()}
    for i, (name, modes) in enumerate(proto["pooled"].items()):
        right = " & & & & & "
        if i < 2:
            mode = ("stateful", "counter")[i]
            right = mode + " & " + " & ".join(f"{x:.3f}" for x in coupling[mode])
        lines.append(f"{name.replace('(', '(Arm ')} & {modes['stateful']['median_variance_reduction']:.1f} & {modes['counter']['median_variance_reduction']:.1f} & {right}\\\\")
    tables["protocol"] = "\n".join(lines + [END])

    axes = curve["medium"]["axes"]
    base = axes["participation"][0]["detectors"]
    lines = _begin(r"\caption{The detection curve at the medium threshold, low-cover market. Signal is the mean signal in the executor's direction. Leakage in bps.}\label{tab:curve}", "@{}lrrrrrl@{}")
    lines.append(rf"Axis, value & Flow signal & A3 signal & A3 duty & A3 latency & Flow latency & Leakage\\ Arm 0 & {base['flow_medium']['signal_arm0']:+.3f} & {base['learned']['signal_arm0']:+.3f} & {base['learned']['duty_arm0']:.3f} & & & \\\midrule")
    labels = {"participation": "Nominal $Q/V$", "timing_jitter": "Timing jitter", "size_jitter": "Size jitter", "passive_share": "Passive share"}
    for axis, rows in axes.items():
        for r in rows:
            f, l = r["detectors"]["flow_medium"], r["detectors"]["learned"]
            lines.append(f"{labels[axis]} {r['value']:.2f} & {f['signal'][0]:+.3f} & {l['signal'][0]:+.3f} & {l['duty'][0]:.3f} & {l['latency_median']:.0f} & {f['latency_median']:.0f} & {ci(r['leakage'])}\\\\")
        lines.append(r"\midrule")
    lines.pop()
    tables["curve"] = "\n".join(lines + [END])

    sc, bucci = list(rep["shi_cartlidge"]["rows"].values()), list(rep["bucci"].items())
    lines = _begin(r"\caption{Replication hooks, deep cover. Left: Shi--Cartlidge, reaction by POV rate. Right: Bucci et al., exponents on the mid's displacement at small and large sizes, with the interval on their difference.}\label{tab:replications}", r"@{}rrl@{\hskip 2em}lrrl@{}")
    lines.append(r"Rate & Real & Reaction (bps) & Detecting & $\delta_{\text{small}}$ & $\delta_{\text{large}}$ & Small $-$ large\\\midrule")
    for i in range(max(len(sc), len(bucci))):
        left = f"{sc[i]['rate']:.2f} & {sc[i]['realised_q_over_v']:.3f} & {ci(sc[i]['reaction'])}" if i < len(sc) else " & & "
        right = " & & & "
        if i < len(bucci):
            label, x = bucci[i]
            t = x["low_minus_high"]
            right = f"{label.split('=')[1]} & {x['delta_low']:.2f} & {x['delta_high']:.2f} & {t['estimate']:+.2f} [{t['low']:+.2f}, {t['high']:+.2f}]"
        lines.append(f"{left} & {right}\\\\")
    tables["replications"] = "\n".join(lines + [END])

    lines = _begin(r"\caption{PPO against TWAP in the sweep's own evaluation (\code{ExecutionEnv}, stateful). ``Selected'' are the seeds the checkpoint and hyperparameters were chosen on.}\label{tab:ppo_heldout}", "@{}rlrrrll@{}")
    lines.append(r"$Q/V$ & Seeds & $n$ & TWAP & PPO & PPO $-$ TWAP & Two-sided $p$\\\midrule")
    for r in held:
        g = r["ppo_minus_twap"]
        lines.append(f"{r['size']:.2f} & {r['seeds']} & {r['n']} & {r['twap']:+.2f} & {r['ppo']:+.2f} & {ci(g)} & {g['p_two_sided']:.3f}\\\\")
    tables["ppo_heldout"] = "\n".join(lines + [END])
    return tables


if __name__ == "__main__":
    main()
