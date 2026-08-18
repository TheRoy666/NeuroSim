#!/usr/bin/env python3
"""
Manjunatha in/out-hub asymmetry check. Tests their headline finding --
that in/out-hub roles differ fundamentally under a directed EC operator,
unlike a symmetric SC/FC-based analysis where in-strength and out-strength
are trivially identical -- on real per-subject EC matrices computed by
compute_and_save_ec_matrices.py.

For each subject's EC matrix:
1. Compute in-strength (column sums) and out-strength (row sums) per node.
2. Correlate them directly across all nodes -- low correlation means
   directionality genuinely matters for identifying "important" nodes.
3. Identify top-K in-hubs and top-K out-hubs, compute their overlap --
   low overlap means the SAME asymmetry, from a different angle: a node
   that's important for receiving influence is often NOT the same node
   that's important for exerting it.

Run directly on the *_EC.npy files saved by compute_and_save_ec_matrices.py:
    python3 manjunatha_hub_asymmetry_check.py \\
        --ec-dir /path/to/ec_matrices_manjunatha --top-k 20
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def classify_relationship(rho):
    """Three-category classification, not a binary threshold -- a rho
    near 0 (roles independent, unpredictable from each other) and a rho
    strongly negative (roles actively trade off, genuine specialization)
    are mechanistically different findings that a single 'rho < 0.3 ->
    True' flag silently conflated. Found and fixed after real data
    showed this mattered: HCP/UNAM cluster strongly negative (-0.94 to
    -0.97, -0.12 to -0.76), ADNI clusters weakly positive (+0.14 to
    +0.39) -- the old flag marked both directions 'True' as if they were
    the same result."""
    if rho <= -0.3:
        return "inverse_specialization"
    elif rho >= 0.3:
        return "positive_redundancy"
    else:
        return "independent"


def analyze_subject(ec_path, top_k):
    EC = np.load(ec_path)
    N = EC.shape[0]
    EC_abs = np.abs(EC)
    np.fill_diagonal(EC_abs, 0)

    out_strength = EC_abs.sum(axis=1)  # row sums -- influence exerted
    in_strength = EC_abs.sum(axis=0)   # column sums -- influence received

    rho, p = spearmanr(out_strength, in_strength)

    out_hubs = set(np.argsort(out_strength)[-top_k:])
    in_hubs = set(np.argsort(in_strength)[-top_k:])
    overlap = len(out_hubs & in_hubs) / top_k
    jaccard = len(out_hubs & in_hubs) / len(out_hubs | in_hubs)

    return {
        "N": N, "top_k": top_k,
        "spearman_rho_in_vs_out_strength": rho,
        "p_value": p,
        "top_k_overlap_fraction": overlap,
        "top_k_jaccard": jaccard,
        "relationship_category": classify_relationship(rho),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec-dir", required=True,
                     help="Directory containing *_EC.npy files")
    ap.add_argument("--top-k", type=int, default=20,
                     help="Number of top hubs to compare for overlap")
    ap.add_argument("--out", default="manjunatha_hub_asymmetry_results.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.ec_dir, "*_EC.npy")))
    print(f"Found {len(files)} subject EC files")

    rows = []
    for f in files:
        sub = os.path.basename(f).replace("_EC.npy", "")
        result = analyze_subject(f, args.top_k)
        result["subject_id"] = sub
        rows.append(result)
        print(f"  {sub}: N={result['N']}, rho(in,out)={result['spearman_rho_in_vs_out_strength']:.4f}, "
              f"p={result['p_value']:.2e}, top-{args.top_k} overlap={result['top_k_overlap_fraction']:.2f}, "
              f"jaccard={result['top_k_jaccard']:.2f}, "
              f"category: {result['relationship_category']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    counts = df["relationship_category"].value_counts()
    for cat in ["inverse_specialization", "independent", "positive_redundancy"]:
        n = counts.get(cat, 0)
        print(f"{cat}: {n}/{len(df)} subjects")
    print(f"\nMedian rho(in,out): {df['spearman_rho_in_vs_out_strength'].median():.4f}")
    print(f"Median top-{args.top_k} hub overlap: {df['top_k_overlap_fraction'].median():.2f}")
    print(f"\nSaved to {args.out}")
