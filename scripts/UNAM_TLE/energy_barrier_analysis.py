#!/usr/bin/env python3
"""
TLE lateralization signal-check — energy_barrier_analysis: Healthy vs TLE (combined laterality).

The last planned GSoC deliverable for the TLE lateralization signal-check (per original scoping:
"energy_barrier_analysis, TLE vs control, well-powered"). Simpler than the
lateralization test -- one clean two-group comparison, no new ROI/direction
decisions, using metrics already computed in unam_results.csv.

Hypothesis: TLE reflects an altered control-energy landscape relative to
healthy controls (direction not pre-specified here, deliberately -- this is
an exploratory omnibus comparison, unlike the lateralization test which
had a directional a priori hypothesis).

Usage:
    python energy_barrier_analysis.py \
        --results /path/to/unam_results.csv \
        --out-dir /path/to/output
"""
import argparse
import os
import pandas as pd
from scipy import stats

METRICS = ["E_rest_to_demand", "E_demand_to_rest", "E_asymmetry",
           "teleport_ratio", "avg_ctrl_mean", "mod_ctrl_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.results)
    df["TLE_status"] = df["Group_detail"].apply(
        lambda g: "Healthy" if g == "Healthy" else "TLE")

    healthy = df[df["TLE_status"] == "Healthy"]
    tle = df[df["TLE_status"] == "TLE"]
    print(f"Healthy: N={len(healthy)}   TLE (Left+Right combined): N={len(tle)}")

    rows = []
    print(f"\n{'='*60}")
    print("Healthy vs TLE — Mann-Whitney U, Cliff's delta")
    print(f"{'='*60}")
    for metric in METRICS:
        h = healthy[metric].dropna()
        t = tle[metric].dropna()
        U, p = stats.mannwhitneyu(h, t, alternative="two-sided")
        gt = sum(x > y for x in h for y in t)
        lt = sum(x < y for x in h for y in t)
        delta = (gt - lt) / (len(h) * len(t))
        sig = "*" if p < 0.05 else " "
        print(f"  {metric:20s}: Healthy median={h.median():.4f}  "
              f"TLE median={t.median():.4f}  p={p:.3f}{sig}  delta={delta:.3f}")
        rows.append({"metric": metric, "Healthy_n": len(h), "Healthy_median": h.median(),
                     "TLE_n": len(t), "TLE_median": t.median(),
                     "MannWhitney_U": U, "p_value": p, "cliffs_delta": delta})

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "unam_energy_barrier_healthy_vs_tle.csv"),
        index=False)
    print(f"\nSaved: unam_energy_barrier_healthy_vs_tle.csv")


if __name__ == "__main__":
    main()