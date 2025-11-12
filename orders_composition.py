#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Steps:
1) Aggregate at order (/another taxa) level by the provided 'Taxonomy' labels (no renaming, no collapsing).
2) Global filter: keep orders with total reads >= min_total_reads across mapped S/P samples.
3) Presence per sample using FIXED thresholds: >= rel_presence_threshold of sample total (min >= min_abs_presence).
4) Mean number of orders per Sample–Material group (INCLUDING control & washout), using fixed thresholds.
5) Control cleaning: drop orders present (by fixed thresholds) in ANY control/washout sample (S or P).
6) Unique orders per Sample–Material group (EXCLUDING control & washout groups), using the SAME fixed thresholds.

Outputs:
- orders_count_by_group__INCLUDES_controls.csv
- orders_absent_in_control_plus_washout.csv
- unique_orders_by_group__AFTER_control_clean.csv
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# --------- mapping: sample -> material ----------
SAMPLE_TO_MATERIAL = {
    # S - BAL
    "S4": "BAL", "S9": "BAL", "S14": "BAL", "S28": "BAL", "S35": "BAL", "S39": "BAL",
    "S49": "BAL", "S52": "BAL", "S56": "BAL", "S59": "BAL", "S63": "BAL", "S66": "BAL",
    # S - biopsy
    "S3": "biopsy", "S8": "biopsy", "S13": "biopsy", "S27": "biopsy", "S34": "biopsy", "S38": "biopsy",
    "S43": "biopsy", "S48": "biopsy", "S53": "biopsy", "S55": "biopsy", "S58": "biopsy", "S62": "biopsy",
    "S65": "biopsy",
    # S - control
    "S12": "control", "S21": "control", "S30": "control", "S41": "control", "S46": "control",
    # S - cultured
    "S10": "cultured", "S17": "cultured", "S18": "cultured", "S19": "cultured", "S20": "cultured", "S22": "cultured",
    "S23": "cultured", "S24": "cultured", "S25": "cultured", "S31": "cultured", "S32": "cultured",
    # S - non-cultured
    "S1": "non-cultured", "S2": "non-cultured", "S6": "non-cultured", "S7": "non-cultured", "S16": "non-cultured",
    "S26": "non-cultured", "S33": "non-cultured", "S37": "non-cultured", "S42": "non-cultured", "S47": "non-cultured",
    "S51": "non-cultured", "S54": "non-cultured", "S57": "non-cultured", "S61": "non-cultured", "S64": "non-cultured",
    # S - washout
    "S5": "washout", "S11": "washout", "S15": "washout", "S29": "washout", "S36": "washout", "S40": "washout",
    "S45": "washout", "S50": "washout", "S60": "washout",
    # P - materials
    "P67": "blood", "P68": "biopsy", "P69": "BAL", "P70": "washout", "P71": "blood", "P72": "biopsy", "P73": "BAL",
    "P74": "washout",
    "P75": "blood", "P76": "biopsy", "P77": "BAL", "P78": "washout", "P79": "blood", "P80": "biopsy", "P81": "BAL",
    "P82": "washout",
    "P83": "blood", "P84": "biopsy", "P85": "BAL", "P86": "washout", "P87": "control",
}

CONTROL_MATERIALS = {"control", "washout"}


# ------------------------ helpers ------------------------

def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_excel(path)


def is_control_sample(sample_id: str) -> bool:
    return SAMPLE_TO_MATERIAL.get(sample_id) in CONTROL_MATERIALS


def group_key(sample_id: str) -> str:
    """Sample–Material key, e.g., 'S-BAL' or 'P-blood'."""
    disease = "S" if sample_id.startswith("S") else "P"
    return f"{disease}-{SAMPLE_TO_MATERIAL[sample_id]}"


# ------------------------ main ------------------------

def main(
        inp: Path,
        outdir: Path,
        min_total_reads: int = 100,  # global filter across all mapped samples
        rel_presence_threshold: float = 0.0001,  # 0.01% of sample total
        min_abs_presence: int = 1,  # at least 1 read
        round_digits: int = 2,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    df_raw = read_table(inp)
    tax_col = "Taxonomy" if "Taxonomy" in df_raw.columns else df_raw.columns[0]

    # --- select mapped S/P samples that exist in file ---
    mapped_samples = [s for s in SAMPLE_TO_MATERIAL if s in df_raw.columns]
    if not mapped_samples:
        print("No required S/P columns found in the input.", file=sys.stderr)
        sys.exit(1)

    # --- aggregate at order level by provided labels ---
    data = df_raw[[tax_col] + mapped_samples].copy()
    data = data.groupby(tax_col, as_index=False).sum()

    # --- global abundance filter ---
    totals_all = data[mapped_samples].sum(axis=1)
    data = data.loc[totals_all >= min_total_reads].copy().reset_index(drop=True)

    # --- presence per sample with FIXED thresholds (computed BEFORE control cleaning) ---
    sample_totals = data[mapped_samples].sum(axis=0)
    thresholds = (sample_totals * rel_presence_threshold).clip(lower=min_abs_presence).astype(float)
    present = data[mapped_samples].ge(thresholds, axis=1)  # bool orders×samples

    # ===================== (1) MEANS by Sample–Material (INCLUDING control & washout) =====================
    orders_per_sample = present.sum(axis=0)  # number of orders in each sample
    df_counts = pd.DataFrame({
        "Sample": list(orders_per_sample.index),
        "Group": [group_key(s) for s in orders_per_sample.index],
        "OrdersCount": orders_per_sample.values.astype(int),
    })
    mean_counts = (df_counts
                   .groupby("Group", as_index=False)
                   .agg(N=("OrdersCount", "size"),
                        Mean_Orders=("OrdersCount", "mean"),
                        SD_Orders=("OrdersCount", "std")))
    mean_counts["Mean_Orders"] = mean_counts["Mean_Orders"].round(round_digits)
    mean_counts["SD_Orders"] = mean_counts["SD_Orders"].round(round_digits)
    mean_counts.sort_values("Group").to_csv(outdir / "orders_count_by_group__INCLUDES_controls.csv", index=False)

    # ===================== (2) CONTROL CLEAN using FIXED thresholds =====================
    control_samples = [s for s in mapped_samples if is_control_sample(s)]
    if control_samples:
        present_in_controls = present[control_samples].any(axis=1)  # by fixed thresholds
        keep_mask = ~present_in_controls
    else:
        keep_mask = pd.Series(True, index=data.index)

    data_excl = data.loc[keep_mask].reset_index(drop=True)
    data_excl.to_csv(outdir / "orders_absent_in_control_plus_washout.csv", index=False)

    # presence matrix for cleaned set using the SAME fixed thresholds (subset rows only)
    present_clean = present.loc[keep_mask].reset_index(drop=True)

    # ===================== (3) UNIQUE ORDERS by Sample–Material (EXCLUDING control/washout groups) =====================
    # Build group -> sample list for non-control materials
    groups = {}
    for s in mapped_samples:
        if is_control_sample(s):
            continue
        g = group_key(s)
        groups.setdefault(g, []).append(s)

    # presence per group: present if present in >=1 sample of that group (fixed thresholds)
    group_presence = {}
    for g, cols in groups.items():
        group_presence[g] = present_clean[cols].any(axis=1) if cols else pd.Series(False, index=data_excl.index)
    gp = pd.DataFrame(group_presence) if group_presence else pd.DataFrame(index=data_excl.index)

    # unique in exactly one group
    unique_rows = []
    for g in gp.columns:
        only_here = gp[g] & (~gp[[c for c in gp.columns if c != g]].any(axis=1))
        idx = data_excl.index[only_here]
        if len(idx) == 0:
            continue
        cols = groups[g]
        n_samples_present = present_clean.loc[idx, cols].sum(axis=1).astype(int)
        total_reads_group = data_excl.loc[idx, cols].sum(axis=1).astype(float)
        tmp = pd.DataFrame({
            "Group": g,
            "Order": data_excl.loc[idx, tax_col].values,
            "N_samples_with_order_in_group": n_samples_present.values,
            "Total_reads_in_group": total_reads_group.values
        }).sort_values("Total_reads_in_group", ascending=False)
        unique_rows.append(tmp)

    unique_by_group = (pd.concat(unique_rows, ignore_index=True)
                       if unique_rows else
                       pd.DataFrame(
                           columns=["Group", "Order", "N_samples_with_order_in_group", "Total_reads_in_group"]))
    unique_by_group.to_csv(outdir / "unique_orders_by_group__AFTER_control_clean.csv", index=False)

    # ===================== (4) TOP-3 ORDERS PER GROUP (relative composition) =====================
    top3_rows = []
    for g, cols in sorted(groups.items()):
        if len(cols) == 0:
            continue

        reads_per_order = data_excl[cols].sum(axis=1)
        total_reads_group = float(reads_per_order.sum())
        if total_reads_group <= 0:
            continue

        perc = 100.0 * reads_per_order / total_reads_group
        sub = pd.DataFrame({
            "Order": data_excl["Taxonomy"],
            "Percent": perc
        }).sort_values("Percent", ascending=False).reset_index(drop=True)

        top = sub.head(3)
        while len(top) < 3:
            top.loc[len(top)] = ["", 0.0]

        others_percent = round(
            max(0.0, 100.0 - top["Percent"].iloc[:3].sum()),
            round_digits
        )

        top3_rows.append({
            "Group": g,
            "Top1_Order": top.iloc[0]["Order"],
            "Top1_Percent": round(float(top.iloc[0]["Percent"]), round_digits),
            "Top2_Order": top.iloc[1]["Order"],
            "Top2_Percent": round(float(top.iloc[1]["Percent"]), round_digits),
            "Top3_Order": top.iloc[2]["Order"],
            "Top3_Percent": round(float(top.iloc[2]["Percent"]), round_digits),
            "Others_Percent": others_percent,
        })

    top3_df = pd.DataFrame(top3_rows).sort_values("Group")
    top3_df.to_csv(outdir / "top3_orders_by_group.csv", index=False)

    print("Done. Saved to:", outdir)
    print(" - orders_count_by_group__INCLUDES_controls.csv")
    print(" - orders_absent_in_control_plus_washout.csv")
    print(" - unique_orders_by_group__AFTER_control_clean.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fixed-threshold pipeline without collapsing 'incertae/others': mean orders by Sample–Material (incl. controls) -> control cleaning -> unique orders per Sample–Material."
    )
    ap.add_argument("-i", "--input", required=True, help="Path to input table (TSV/CSV or XLS/XLSX)")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory")
    ap.add_argument("--min-total-reads", type=int, default=100,
                    help="Global minimum total reads per order across mapped samples (default 100)")
    ap.add_argument("--rel-presence-threshold", type=float, default=0.0001,
                    help="Per-sample presence threshold as fraction of sample total (0.0001 = 0.01%)")
    ap.add_argument("--min-abs-presence", type=int, default=1,
                    help="Minimum absolute reads for presence in a sample (default 1)")
    ap.add_argument("--round-digits", type=int, default=2, help="Rounding for mean/SD (default 2)")
    args = ap.parse_args()

    main(
        inp=Path(args.input),
        outdir=Path(args.outdir),
        min_total_reads=args.min_total_reads,
        rel_presence_threshold=args.rel_presence_threshold,
        min_abs_presence=args.min_abs_presence,
        round_digits=args.round_digits,
    )
