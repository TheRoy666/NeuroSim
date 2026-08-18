#!/usr/bin/env python3
"""
Manual sanity check on Test 2's sign convention -- before interpreting the
negative correlation as either a real finding or a bug, verify by hand on
the 3 most extreme hippocampal-asymmetry subjects.

primary_MTL_asym = L - R average controllability (from the lateralization script)
hipp_vol_asym    = L_hipp_vol - R_hipp_vol

If sign conventions are correct, this just directly shows the raw numbers
for eyeballing -- no correlation math, just look at the actual values.
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="unam_lateralization_scores.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    df["hipp_vol_asym"] = df["L_hipp_vol"] - df["R_hipp_vol"]
    valid = df.dropna(subset=["primary_MTL_asym", "hipp_vol_asym"])

    print("=== 5 subjects with MOST POSITIVE hipp_vol_asym (L hipp >> R hipp) ===")
    top = valid.nlargest(5, "hipp_vol_asym")
    print(top[["subject_id", "TLEside", "L_hipp_vol", "R_hipp_vol",
               "hipp_vol_asym", "primary_MTL_asym"]].to_string(index=False))

    print("\n=== 5 subjects with MOST NEGATIVE hipp_vol_asym (R hipp >> L hipp) ===")
    bot = valid.nsmallest(5, "hipp_vol_asym")
    print(bot[["subject_id", "TLEside", "L_hipp_vol", "R_hipp_vol",
               "hipp_vol_asym", "primary_MTL_asym"]].to_string(index=False))

    print("\nManually check: for the top group (L hipp bigger), is")
    print("primary_MTL_asym mostly NEGATIVE (L controllability lower)?")
    print("That's the pattern producing the negative correlation --")
    print("confirm it's not a units/sign bug in how either column was built.")


if __name__ == "__main__":
    main()