"""
SHAP Visualization Atlas - Publication-Grade Style (24 figures)
==============================================================
Using plot_style.py for unified styling:
- Gradient background #F8F9E4 -> #E5F2FB
- Bold black frame
- Noto Serif CJK SC font (globally set via rcParams, no per-plot specification)
- Canvas 8x6 (dependence plots 7x5)
- 300 DPI PNG
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.figures_style import (
    create_gradient_rect, style_ax, save,
    CAT_COLORS, GRADIENT_TOP, GRADIENT_BOTTOM,
    DARK, GRAY,
    FIG_SIZE,
    FONT_SIZE_SHAP_TICK, FONT_SIZE_SHAP_LABEL, FONT_SIZE_SHAP_LEGEND,
    FONT_SIZE_SHAP_ANNOT, FONT_SIZE_SHAP_CBAR,
)

import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR
OUT_DIR = os.path.join(OUTPUT_DIR, "figures", "shap")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "Laser Power", "Scan Speed", "Powder Feed Rate", "Spot Diameter", "Defocus",
    "C", "Cr", "Fe", "Mn",
    "Line Energy Density", "Powder-Energy Ratio", "Carbon Equivalent", "Ni Equivalent",
]

# Chinese -> English feature name mapping (CSV stores Chinese names)
FEATURE_CN_TO_EN = {
    "激光功率": "Laser Power",
    "扫描速度": "Scan Speed",
    "送粉速率": "Powder Feed Rate",
    "光斑直径": "Spot Diameter",
    "离焦量": "Defocus",
    "C": "C",
    "Cr": "Cr",
    "Fe": "Fe",
    "Mn": "Mn",
    "线能量密度": "Line Energy Density",
    "粉末能量比": "Powder-Energy Ratio",
    "碳当量": "Carbon Equivalent",
    "镍当量": "Ni Equivalent",
}

# Abbreviation <-> Full Name mapping (for axis label display)
FEATURE_ABBR = {
    "LP":  "Laser Power",
    "SS":  "Scan Speed",
    "PFR": "Powder Feed Rate",
    "SD":  "Spot Diameter",
    "DF":  "Defocus",
    "C":   "C",
    "Cr":  "Cr",
    "Mn":  "Mn",
    "Fe":  "Fe",
    "LED": "Line Energy Density",
    "PER": "Powder-Energy Ratio",
    "CE":  "Carbon Equivalent",
    "NE":  "Ni Equivalent",
}
FEATURE_EN_TO_ABBR = {v: k for k, v in FEATURE_ABBR.items()}


def load_shap_data():
    base_dir = os.path.join(OUTPUT_DIR, "shap_baseline")
    fs_dir = os.path.join(OUTPUT_DIR, "shap_fewshot")
    data = {}
    data["X_base_hv"] = pd.read_csv(os.path.join(base_dir, "X_train_hv.csv"))
    data["X_base_corr"] = pd.read_csv(os.path.join(base_dir, "X_train_corr.csv"))
    data["shap_base_hv"] = np.load(os.path.join(base_dir, "shap_vals_hv.npy"))
    data["shap_base_corr"] = np.load(os.path.join(base_dir, "shap_vals_corr.npy"))
    data["shap_inter_hv"] = np.load(os.path.join(base_dir, "shap_inter_hv.npy"))
    data["shap_inter_corr"] = np.load(os.path.join(base_dir, "shap_inter_corr.npy"))
    data["import_base"] = pd.read_csv(os.path.join(base_dir, "shap_import_baseline.csv"))
    data["X_fs"] = pd.read_csv(os.path.join(fs_dir, "X_fs.csv"))
    data["shap_fs_hv"] = np.load(os.path.join(fs_dir, "shap_vals_hv_fs.npy"))
    data["shap_fs_corr"] = np.load(os.path.join(fs_dir, "shap_vals_corr_fs.npy"))
    data["import_fs"] = pd.read_csv(os.path.join(fs_dir, "shap_import_fewshot.csv"))
    data["rockit_idx"] = pd.read_csv(os.path.join(fs_dir, "rockit_sample_index.csv"))
    return data


# ============================================================
# P0-1: Beeswarm Summary (4 figures)
# ============================================================
def plot_beeswarm(data):
    configs = [
        ("Baseline_Hardness", data["shap_base_hv"], data["X_base_hv"], "Baseline - Hardness"),
        ("Baseline_Corrosion", data["shap_base_corr"], data["X_base_corr"], "Baseline - Corrosion"),
        ("Fewshot_Hardness", data["shap_fs_hv"], data["X_fs"], "Few-shot - Hardness"),
        ("Fewshot_Corrosion", data["shap_fs_corr"], data["X_fs"], "Few-shot - Corrosion"),
    ]

    for i, (name, shap_vals, X, title) in enumerate(configs):
        print(f"  [Beeswarm {i+1}/4] {title}")

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        cmap = LinearSegmentedColormap.from_list(
            "shap_vivid", ["#728BDE", "#E246C9", "#E97A6F"], N=256)

        n_features = shap_vals.shape[1]
        feat_names = list(X.columns)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        order = np.argsort(mean_abs)

        for j, feat_idx in enumerate(order):
            vals = shap_vals[:, feat_idx]
            feat_vals = X.iloc[:, feat_idx].values
            fv_min, fv_max = feat_vals.min(), feat_vals.max()
            norm_fv = (feat_vals - fv_min) / (fv_max - fv_min) if fv_max > fv_min else np.zeros_like(feat_vals)

            np.random.seed(42)
            jitter = np.random.normal(0, 0.12, size=len(vals))
            sort_idx = np.argsort(norm_fv)

            ax.scatter(vals[sort_idx], j + jitter[sort_idx],
                       c=norm_fv[sort_idx], cmap=cmap, s=12, alpha=1.0,
                       edgecolors="none", zorder=3)

        display_labels = []
        for idx in order:
            cn = feat_names[idx]
            en = FEATURE_CN_TO_EN.get(cn, cn)
            display_labels.append(FEATURE_EN_TO_ABBR.get(en, en))
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(display_labels)
        ax.set_xlabel("SHAP value", fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
        ax.axvline(x=0, color=GRAY, linewidth=1.0, linestyle="--", alpha=0.6)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Feature value (low -> high)", fontsize=FONT_SIZE_SHAP_CBAR)
        cbar.ax.tick_params(labelsize=FONT_SIZE_SHAP_CBAR)

        create_gradient_rect(ax)
        # Many feature names on Y-axis, use SHAP small font size (also applies to X-axis tick labels)
        style_ax(ax, grid=False, right_top_ticks=False,
                 tick_labelsize=FONT_SIZE_SHAP_TICK)

        save(fig, f"beeswarm_{name}", OUT_DIR)
        plt.close(fig)

    print("  [P0] Beeswarm x 4 done")


# ============================================================
# P0-2: Waterfall Plot (8 figures)
# ============================================================
def plot_waterfall(data):
    rockit_rows = data["rockit_idx"]["row_index"].values
    rockit_ids = data["rockit_idx"]["sample_id"].values
    shap_hv = data["shap_fs_hv"]
    shap_corr = data["shap_fs_corr"]
    X_fs = data["X_fs"]

    for i, (row_idx, sample_id) in enumerate(zip(rockit_rows, rockit_ids)):
        for target, shap_vals, title_tag, unit in [
            ("Hardness", shap_hv, "Hardness", "HV"),
            ("Corrosion", shap_corr, "Corrosion", "log"),
        ]:
            print(f"  [Waterfall {i*2 + (1 if target=='Hardness' else 2)}/8] {sample_id} - {target}")

            sample_shap = shap_vals[row_idx]
            sample_feat = X_fs.iloc[row_idx]

            abs_order = np.argsort(np.abs(sample_shap))[::-1]
            top_n = min(10, len(abs_order))
            top_idx = abs_order[:top_n]

            names = [FEATURE_EN_TO_ABBR.get(FEATURE_NAMES[j], FEATURE_NAMES[j]) for j in top_idx]
            vals = sample_shap[top_idx]
            feat_vals = sample_feat.iloc[top_idx].values

            colors = ["#E97A6F" if v > 0 else "#728BDE" for v in vals]

            fig, ax = plt.subplots(figsize=FIG_SIZE)

            y_pos = np.arange(top_n)[::-1]
            bars = ax.barh(y_pos, vals, color=colors, edgecolor="black",
                           linewidth=1.0, height=0.65)

            x_max = max(abs(vals)) * 1.25
            for j, (bar, sv) in enumerate(zip(bars, vals)):
                w = bar.get_width()
                if w >= 0:
                    label_x = w + x_max * 0.02
                    ha = "left"
                else:
                    label_x = w - x_max * 0.02
                    ha = "right"
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                        f"{sv:+.1f}", ha=ha, va="center",
                        fontsize=FONT_SIZE_SHAP_ANNOT,
                        color=DARK, fontweight="bold")

            # Merge feature values into Y-axis labels to avoid left-side annotation overlap
            y_labels = [f"{n} ({fv:.2f})" for n, fv in zip(names, feat_vals)]
            ax.set_xlim(-x_max, x_max)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel(f"SHAP contribution ({unit})",
                         fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
            ax.axvline(x=0, color=GRAY, linewidth=1.0, linestyle="--", alpha=0.6)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#E97A6F", edgecolor="black", label="Positive"),
                Patch(facecolor="#728BDE", edgecolor="black", label="Negative"),
            ]
            ax.legend(handles=legend_elements, fontsize=FONT_SIZE_SHAP_LEGEND,
                      loc="lower right", frameon=False)

            create_gradient_rect(ax)
            # Many feature names on Y-axis, use SHAP small font size (also applies to X-axis tick labels)
            style_ax(ax, grid=False, right_top_ticks=False,
                     tick_labelsize=FONT_SIZE_SHAP_TICK)

            save(fig, f"waterfall_{sample_id}_{target}", OUT_DIR)
            plt.close(fig)

    print("  [P0] Waterfall x 8 done")


# ============================================================
# P1-1: Top5 Dependence Plot (10 figures)
# ============================================================
def plot_dependence(data):
    shap_hv = data["shap_fs_hv"]
    shap_corr = data["shap_fs_corr"]
    X_fs = data["X_fs"]
    import_fs = data["import_fs"]

    hv_top5_cn = import_fs.sort_values("abs_shap_mean_hv", ascending=False)["特征名"].head(5).tolist()
    corr_top5_cn = import_fs.sort_values("abs_shap_mean_corr", ascending=False)["特征名"].head(5).tolist()
    # Map Chinese names to English for display and indexing
    hv_top5 = [FEATURE_CN_TO_EN[cn] for cn in hv_top5_cn]
    corr_top5 = [FEATURE_CN_TO_EN[cn] for cn in corr_top5_cn]

    fs_columns = list(X_fs.columns)

    for i, feat_cn in enumerate(hv_top5_cn):
        feat_en = FEATURE_CN_TO_EN.get(feat_cn, feat_cn)
        print(f"  [Dependence {i+1}/10] Hardness - {feat_en}")
        feat_idx = fs_columns.index(feat_cn)

        fig, ax = plt.subplots(figsize=(7, 5))

        feat_vals = X_fs.iloc[:, feat_idx].values
        shap_vals_dep = shap_hv[:, feat_idx]
        fv_min, fv_max = feat_vals.min(), feat_vals.max()

        scatter = ax.scatter(feat_vals, shap_vals_dep, c=feat_vals, cmap="coolwarm",
                             s=25, alpha=0.65, edgecolors="black", linewidths=0.4,
                             vmin=fv_min, vmax=fv_max)

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_idx = np.argsort(feat_vals)
            xv = feat_vals[sorted_idx]
            yv = shap_vals_dep[sorted_idx]
            smoothed = lowess(yv, xv, frac=0.6)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#E8156E",
                    linewidth=2.5, label="LOWESS", zorder=5)
        except Exception:
            pass

        feat_abbr = FEATURE_EN_TO_ABBR.get(feat_en, feat_en)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(feat_abbr, fontsize=FONT_SIZE_SHAP_CBAR)
        cbar.ax.tick_params(labelsize=FONT_SIZE_SHAP_CBAR)

        ax.set_xlabel(feat_abbr, fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
        ax.set_ylabel("SHAP value (HV)", fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
        ax.axhline(y=0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend(fontsize=FONT_SIZE_SHAP_LEGEND, loc="best", frameon=False)

        create_gradient_rect(ax)
        style_ax(ax, grid=False, right_top_ticks=False)

        save(fig, f"dependence_Hardness_{feat_en}", OUT_DIR)
        plt.close(fig)

    for i, feat_cn in enumerate(corr_top5_cn):
        feat_en = FEATURE_CN_TO_EN.get(feat_cn, feat_cn)
        print(f"  [Dependence {i+6}/10] Corrosion - {feat_en}")
        feat_idx = fs_columns.index(feat_cn)

        fig, ax = plt.subplots(figsize=(7, 5))

        feat_vals = X_fs.iloc[:, feat_idx].values
        shap_vals_dep = shap_corr[:, feat_idx]
        fv_min, fv_max = feat_vals.min(), feat_vals.max()

        scatter = ax.scatter(feat_vals, shap_vals_dep, c=feat_vals, cmap="coolwarm",
                             s=25, alpha=0.65, edgecolors="black", linewidths=0.4,
                             vmin=fv_min, vmax=fv_max)

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_idx = np.argsort(feat_vals)
            xv = feat_vals[sorted_idx]
            yv = shap_vals_dep[sorted_idx]
            smoothed = lowess(yv, xv, frac=0.6)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#E8156E",
                    linewidth=2.5, label="LOWESS", zorder=5)
        except Exception:
            pass

        feat_abbr = FEATURE_EN_TO_ABBR.get(feat_en, feat_en)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(feat_abbr, fontsize=FONT_SIZE_SHAP_CBAR)
        cbar.ax.tick_params(labelsize=FONT_SIZE_SHAP_CBAR)

        ax.set_xlabel(feat_abbr, fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
        ax.set_ylabel("SHAP value (log)", fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
        ax.axhline(y=0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend(fontsize=FONT_SIZE_SHAP_LEGEND, loc="best", frameon=False)

        create_gradient_rect(ax)
        style_ax(ax, grid=False, right_top_ticks=False)

        save(fig, f"dependence_Corrosion_{feat_en}", OUT_DIR)
        plt.close(fig)

    print("  [P1] Dependence x 10 done")


# ============================================================
# P1-2: Baseline vs Few-shot Importance Comparison (2 figures)
# ============================================================
def plot_importance_comparison(data):
    import_base = data["import_base"]
    import_fs = data["import_fs"]

    for target_col, title_name, xlabel_text in [
        ("abs_shap_mean_hv", "Hardness", "SHAP |mean| (HV)"),
        ("abs_shap_mean_corr", "Corrosion", "SHAP |mean| (log)"),
    ]:
        print(f"  [Importance] {title_name}")

        fig, ax = plt.subplots(figsize=FIG_SIZE)

        merged = import_base[["特征名", target_col]].rename(
            columns={target_col: "Baseline"})
        merged["Few-shot"] = import_fs[target_col].values
        merged = merged.sort_values("Few-shot", ascending=True)
        # Map Chinese feature names to abbreviations for display
        merged["特征名"] = merged["特征名"].map(FEATURE_CN_TO_EN).map(FEATURE_EN_TO_ABBR)

        y = np.arange(len(merged))
        width = 0.38

        ax.barh(y + width/2, merged["Baseline"], width, color=CAT_COLORS[3],
                edgecolor="black", linewidth=1.0, label="Baseline", alpha=0.8)
        ax.barh(y - width/2, merged["Few-shot"], width, color=CAT_COLORS[5],
                edgecolor="black", linewidth=1.0, label="Few-shot", alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(merged["特征名"])
        ax.set_xlabel(xlabel_text, fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_SIZE_SHAP_LEGEND, loc="lower right", frameon=False)

        create_gradient_rect(ax)
        # Many feature names on Y-axis, use SHAP small font size (also applies to X-axis tick labels)
        style_ax(ax, grid=False, right_top_ticks=False,
                 tick_labelsize=FONT_SIZE_SHAP_TICK)

        save(fig, f"importance_compare_{title_name}", OUT_DIR)
        plt.close(fig)

    print("  [P1] Importance comparison x 2 done")


# ============================================================
# Main function
# ============================================================
def main():
    print("=" * 60)
    print("SHAP Visualization Atlas - Publication-Grade Style (24 figures)")
    print(f"Gradient: {GRADIENT_TOP} -> {GRADIENT_BOTTOM}")
    print(f"Resolution: 300 DPI | Format: PNG")
    print("=" * 60)
    print()

    print("[Loading data]")
    data = load_shap_data()
    print(f"  Baseline: {data['shap_base_hv'].shape}")
    print(f"  Few-shot: {data['shap_fs_hv'].shape}")
    print()

    print("[P0] Beeswarm Summary (4 figures)")
    plot_beeswarm(data)
    print()

    print("[P0] Waterfall Plot (8 figures)")
    plot_waterfall(data)
    print()

    print("[P1] Top5 Dependence Plot (10 figures)")
    plot_dependence(data)
    print()

    print("[P1] Importance Comparison (2 figures)")
    plot_importance_comparison(data)
    print()

    files = sorted([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])
    print("=" * 60)
    print(f"All done! {len(files)} figures total, saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
