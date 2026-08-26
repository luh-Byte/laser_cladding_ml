"""
数据可视化图谱(10张)
===================================
使用 plot_style.py 统一样式：
- 渐变背景 #F8F9E4 → #E5F2FB
- 加粗黑色图框
- Noto Serif CJK SC 字体 (全局 rcParams 设置)
- 300 DPI PNG
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# 导入统一样式
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.figures_style import (
    create_gradient_rect, style_ax, save,
    GRADIENT_TOP, GRADIENT_BOTTOM,
    PALETTE, CAT_COLORS, DARK, GRAY, GRID_COLOR,
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_HEATMAP,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FONT_SIZE_CBAR, FONT_SIZE_ANNOT, FONT_SIZE_TITLE,
    FONT_SIZE_TICK_HEAT, FONT_SIZE_ANNOT_HEAT,
    FONT_SIZE_LABEL_HEAT, FONT_SIZE_CBAR_HEAT,
    FONT_SIZE_SHAP_TICK, FONT_SIZE_SHAP_LABEL,
    FONT_SIZE_SHAP_LEGEND, FONT_SIZE_SHAP_ANNOT, FONT_SIZE_SHAP_CBAR,
)

import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR
OUT_DIR = os.path.join(OUTPUT_DIR, "figures")
SHAP_DIR = os.path.join(OUT_DIR, "shap")
os.makedirs(OUT_DIR, exist_ok=True)

from src.data_preprocessing import load_raw_data, clean_data
from src.feature_engineering import compute_derived_features


def load_all_data():
    df_raw = load_raw_data()
    df_clean, _ = clean_data(df_raw)
    df_feat = compute_derived_features(df_clean.copy())

    preds = {}
    for model in ["KNN", "SVR", "RF", "LightGBM"]:
        for target in ["硬度", "腐蚀电流"]:
            fname = os.path.join(OUTPUT_DIR, "results", "predictions",
                                 f"pred_{model}_{target}_rep0.csv")
            if os.path.exists(fname):
                preds[f"{model}_{target}"] = pd.read_csv(fname)

    summary = pd.read_excel(os.path.join(OUTPUT_DIR, "results", "model_summary.xlsx"))
    shap_base = pd.read_csv(os.path.join(OUTPUT_DIR, "shap_baseline", "shap_import_baseline.csv"))
    shap_fs = pd.read_csv(os.path.join(OUTPUT_DIR, "shap_fewshot", "shap_import_fewshot.csv"))
    pareto_global = pd.read_csv(os.path.join(OUTPUT_DIR, "results", "pareto_rockit485_fewshot", "pareto_global_front.csv"))
    pareto_local = pd.read_csv(os.path.join(OUTPUT_DIR, "results", "pareto_rockit485_local", "pareto_local_front.csv"))
    loo = pd.read_csv(os.path.join(OUTPUT_DIR, "loo_validation", "loo_results.csv"))

    return {
        "df_clean": df_clean, "df_feat": df_feat, "preds": preds,
        "summary": summary, "shap_base": shap_base, "shap_fs": shap_fs,
        "pareto_global": pareto_global, "pareto_local": pareto_local, "loo": loo,
    }


# ============================================================
# 图1：相关性矩阵热图
# ============================================================
# 特征名简写映射 (Abbreviation -> Full Name)
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
    "HV":  "Hardness",
    "IC":  "Corrosion Current",
}
FEATURE_CN_TO_ABBR = {
    "激光功率": "LP", "扫描速度": "SS", "送粉速率": "PFR",
    "光斑直径": "SD", "离焦量": "DF",
    "C": "C", "Cr": "Cr", "Mn": "Mn", "Fe": "Fe",
    "线能量密度": "LED", "粉末能量比": "PER",
    "碳当量": "CE", "镍当量": "NE",
}


def _gradient_bar_color(base_color):
    """Return a colormap for vertical bar gradient: white (bottom) -> base_color (top)."""
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(base_color)
    return LinearSegmentedColormap.from_list(
        "bar_grad", [(r, g, b, 0.15), (r, g, b, 1)], N=256)


def _apply_gradient_fill(ax, bars, base_color):
    """Apply vertical gradient fill to each bar: white at bottom, color on top."""
    cmap = _gradient_bar_color(base_color)
    for bar in bars:
        bar.set_edgecolor("none")
        x, y = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h == 0 or np.isnan(h):
            continue
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        ax.imshow(gradient, aspect="auto", cmap=cmap,
                  extent=[x, x + w, y, y + h], zorder=bar.get_zorder())
        bar.set_visible(False)


def _apply_3d_fill(ax, bars, base_color, zorder=3):
    """Apply horizontal 3D-like gradient to bars: dark on edges, light in center."""
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(base_color)
    # 左深 -> 中浅 -> 右深
    cmap = LinearSegmentedColormap.from_list(
        "bar3d",
        [(r, g, b, 0.95), (r, g, b, 0.25), (r, g, b, 0.95)],
        N=256)
    for bar in bars:
        bar.set_edgecolor("none")
        x, y = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h == 0 or np.isnan(h):
            continue
        # 水平渐变: 1行256列
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", cmap=cmap,
                  extent=[x, x + w, y, y + h], zorder=zorder)
        bar.set_visible(False)

def plot_correlation_heatmap(data):
    df = data["df_feat"]
    cols = ["激光功率", "扫描速度", "送粉速率", "光斑直径", "离焦量",
            "C", "Cr", "Mn", "Fe", "线能量密度", "粉末能量比",
            "碳当量", "镍当量", "硬度", "腐蚀电流"]
    cols_en = ["Laser Power", "Scan Speed", "Powder Feed Rate", "Spot Diameter",
               "Defocus", "C", "Cr", "Mn", "Fe", "Line Energy Density",
               "Powder-Energy Ratio", "Carbon Equivalent", "Ni Equivalent",
               "Hardness", "Corrosion Current"]
    # 简写用于轴标签显示
    cols_abbr = list(FEATURE_ABBR.keys())
    corr = df[cols].corr()

    cmap = LinearSegmentedColormap.from_list(
        "paper_diverging",
        ["#728BDE", "#F8F1F1", "#E85345"],
        N=256)

    fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols_abbr, rotation=45, ha="right", fontsize=FONT_SIZE_TICK_HEAT)
    ax.set_yticklabels(cols_abbr, fontsize=FONT_SIZE_TICK_HEAT)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color,
                    fontweight="bold")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Pearson r", fontweight="bold", fontsize=FONT_SIZE_CBAR_HEAT)
    cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR_HEAT)

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("black")

    save(fig, "01_correlation_heatmap", OUT_DIR)
    plt.close(fig)
    print("  [1/13] 相关性矩阵热图 ✓")


# ============================================================
# 图2：分组柱状图 — 模型性能对比
# ============================================================
def plot_model_comparison_bar(data):
    summary = data["summary"]
    hv_data = summary[summary["目标变量"] == "硬度"]
    corr_data = summary[summary["目标变量"] == "腐蚀电流"]
    models = ["KNN", "SVR", "RF", "LightGBM"]

    hv_r2 = []
    for m in models:
        row = hv_data[hv_data["模型"] == m].iloc[0]
        val_str = row["测试集R2(均值±std)"]
        mean_val = float(val_str.split("±")[0].strip())
        std_val = float(val_str.split("±")[1].strip())
        hv_r2.append((mean_val, std_val))

    corr_r2 = []
    for m in models:
        row = corr_data[corr_data["模型"] == m].iloc[0]
        val_str = row["log空间R2(均值±std)"]
        mean_val = float(val_str.split("±")[0].strip())
        std_val = float(val_str.split("±")[1].strip())
        corr_r2.append((mean_val, std_val))

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bars1 = ax.bar(x - width/2, [v[0] for v in hv_r2], width,
                   yerr=[v[1] for v in hv_r2], capsize=5,
                   color=CAT_COLORS[0], edgecolor="none",
                   label="Hardness (Test R²)", zorder=3)
    bars2 = ax.bar(x + width/2, [v[0] for v in corr_r2], width,
                   yerr=[v[1] for v in corr_r2], capsize=5,
                   color=CAT_COLORS[3], edgecolor="none",
                   label="Corrosion (log R²)", zorder=3)

    _apply_gradient_fill(ax, bars1, CAT_COLORS[0])
    _apply_gradient_fill(ax, bars2, CAT_COLORS[3])

    for bar, (mean_val, std_val) in zip(bars1, hv_r2):
        label_y = mean_val + std_val + 0.015
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f"{mean_val:.3f}", ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOT, fontweight="bold")
    for bar, (mean_val, std_val) in zip(bars2, corr_r2):
        label_y = mean_val + std_val + 0.015
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f"{mean_val:.3f}", ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("R²", fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left", frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)
    ax.set_ylim(min(0, min(v[0] for v in hv_r2) - 0.1),
                max(v[0] for v in hv_r2) + 0.15)

    save(fig, "02_model_comparison_bar", OUT_DIR)
    plt.close(fig)
    print("  [2/13] 分组柱状图 — 模型性能对比 ✓")


# ============================================================
# 图3：PCA聚类散点图
# ============================================================
def plot_pca_scatter(data):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = data["df_feat"]
    features = ["激光功率", "扫描速度", "送粉速率", "光斑直径", "离焦量",
                "C", "Cr", "Mn", "Fe", "线能量密度", "粉末能量比", "碳当量", "镍当量"]

    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    hardness = df["硬度"].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 顺序色标: 浅黄→珊瑚→品红(用色板颜色构建)
    seq_cmap = LinearSegmentedColormap.from_list(
        "seq_palette", ["#F8F9E4", "#E0BEB3", "#E97A6F", "#E8156E"], N=256)

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=hardness, cmap=seq_cmap, s=50,
                         edgecolors="black", linewidths=0.5, alpha=0.85, zorder=3)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Hardness (HV)", fontweight="bold", fontsize=FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)",
                  fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)",
                  fontsize=FONT_SIZE_LABEL, fontweight="bold")

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "03_pca_scatter", OUT_DIR)
    plt.close(fig)
    print("  [3/13] 聚类散点图 — PCA降维 ✓")


# ============================================================
# 图4：箱线+散点图 — 硬度按Cr分组
# ============================================================
def plot_violin_by_cr(data):
    df = data["df_clean"].copy()
    cr_q = df["Cr"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    labels = [f"Q1\n({cr_q[0]:.0f}-{cr_q[1]:.0f})",
              f"Q2\n({cr_q[1]:.0f}-{cr_q[2]:.0f})",
              f"Q3\n({cr_q[2]:.0f}-{cr_q[3]:.0f})",
              f"Q4\n({cr_q[3]:.0f}-{cr_q[4]:.0f})"]
    df["Cr组"] = pd.cut(df["Cr"], bins=cr_q, labels=labels, include_lowest=True)
    groups = [df[df["Cr组"] == l]["硬度"].dropna().values for l in labels]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    box_colors = [CAT_COLORS[0], CAT_COLORS[3], CAT_COLORS[5], CAT_COLORS[6]]
    positions = range(4)

    # 箱线图
    bp = ax.boxplot(
        groups, positions=positions, widths=0.45, patch_artist=True,
        showfliers=False,  # 离群点单独画
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, color="#555555"),
        capprops=dict(linewidth=1.2, color="#555555"),
        medianprops=dict(color="#E8156E", linewidth=2.5),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(box_colors[i])
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")

    # 散点图 (beeswarm-style jitter)
    for i, g in enumerate(groups):
        jitter = np.random.normal(i, 0.12, size=len(g))
        ax.scatter(jitter, g, color=box_colors[i], s=35, alpha=0.6,
                   edgecolors="black", linewidths=0.4, zorder=4)

    # 中位数标注
    for i, g in enumerate(groups):
        median = np.median(g)
        ax.text(i + 0.28, median, f"{median:.0f}", ha="left", va="center",
                fontsize=FONT_SIZE_ANNOT, color="#E8156E", fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_xlabel("Cr Content Group (wt%)", fontsize=FONT_SIZE_LABEL, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GRAY,
               markersize=10, label="IQR"),
        Line2D([0], [0], color="#E8156E", linewidth=2.5, label="Median"),
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE_LEGEND,
              loc="upper right", frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "04_violin_by_cr", OUT_DIR)
    plt.close(fig)
    print("  [4/13] 箱线+散点图 — 硬度按Cr分组 ✓")


# ============================================================
# 图5：堆叠柱状图 — 材料成分占比
# ============================================================
def plot_stacked_composition(data):
    df = data["df_clean"].copy()
    elements = ["C", "Cr", "Si", "Ni", "Fe", "Mn", "Mo"]
    elem_colors = CAT_COLORS[:7]

    df["Cr_group"] = pd.cut(df["Cr"], bins=[0, 20, 50, 80],
                             labels=["Cr<20%", "20-50%", "50-80%"])
    grouped = df.groupby("Cr_group")[elements].mean()
    n_groups = len(grouped)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(-0.5, n_groups - 0.3)

    bottom = np.zeros(n_groups)
    for i, elem in enumerate(elements):
        vals = grouped[elem].values
        bars = ax.bar(range(n_groups), vals, bottom=bottom,
                      color=elem_colors[i], edgecolor="black", linewidth=0.8,
                      label=elem, width=0.6, zorder=3)
        _apply_3d_fill(ax, bars, elem_colors[i], zorder=3)
        for j in range(n_groups):
            x = bars[j].get_x()
            w = bars[j].get_width()
            y = bars[j].get_y()
            h = bars[j].get_height()
            if h > 0 and not np.isnan(h):
                ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                           edgecolor="black", linewidth=0.8,
                                           zorder=5))
        for j, v in enumerate(vals):
            if v > 3:
                ax.text(j, bottom[j] + v/2, f"{v:.1f}",
                        ha="center", va="center", fontsize=FONT_SIZE_ANNOT,
                        color="black", fontweight="bold")
        bottom += vals

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(grouped.index, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Content (wt%)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    # 用代理图例恢复元素颜色 (3d fill隐藏了bar导致默认图例无色)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=elem_colors[i], edgecolor="black",
                            linewidth=0.8, label=elem) for i, elem in enumerate(elements)]
    ax.legend(handles=legend_handles, fontsize=FONT_SIZE_LEGEND,
              loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=len(elements), frameon=False)
    ax.set_ylim(0, 110)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "05_stacked_composition", OUT_DIR)
    plt.close(fig)
    print("  [5/13] 堆叠柱状图 — 材料成分占比 ✓")


# ============================================================
# 图6：帕累托前沿散点气泡图
# ============================================================
def plot_pareto_bubble(data):
    pg = data["pareto_global"]
    pl = data["pareto_local"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.scatter(pl["腐蚀电流"] * 1e7, pl["硬度"],
               s=(pl["激光功率"] / pl["激光功率"].max()) * 150 + 30,
               c=CAT_COLORS[5], alpha=0.45, edgecolors="black",
               linewidths=0.8, label="Local Pareto", zorder=3)
    ax.scatter(pg["腐蚀电流"] * 1e7, pg["硬度"],
               s=(pg["激光功率"] / pg["激光功率"].max()) * 150 + 30,
               c=CAT_COLORS[3], alpha=0.55, edgecolors="black",
               linewidths=0.8, label="Global Pareto", zorder=4)

    # 先确定Local Best点
    best_local = pl.iloc[pl["硬度"].argmax()]
    best_local_power = best_local['激光功率']

    # 功率标注: 仅标注与最近邻距离>0.1的点, 密集区域不标
    pg_x = (pg["腐蚀电流"].values * 1e7).astype(float)
    pg_y = pg["硬度"].values.astype(float)
    for idx, (_, row) in enumerate(pg.iterrows()):
        rx, ry = float(row["腐蚀电流"]) * 1e7, float(row["硬度"])
        min_dist = min(
            ((rx - pg_x[j])**2 + (ry - pg_y[j])**2)**0.5
            for j in range(len(pg_x)) if j != idx
        )
        if min_dist < 100:  # 密集区域跳过
            continue
        ax.annotate(f"{row['激光功率']:.0f}W",
                    xy=(rx, ry),
                    xytext=(14, 10), textcoords="offset points",
                    fontsize=FONT_SIZE_ANNOT, color=DARK,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5))

    ax.annotate(f"Local Best: {best_local['硬度']:.0f}HV, {best_local['激光功率']:.0f}W",
                xy=(best_local["腐蚀电流"] * 1e7, best_local["硬度"]),
                xytext=(-160, 10), textcoords="offset points",
                fontsize=FONT_SIZE_ANNOT, color=CAT_COLORS[5],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CAT_COLORS[5], lw=1.5))

    ax.set_xlabel(r"Corrosion Current ($\times 10^{-7}$ A/cm$^2$)",
                 fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_ylabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="lower right", frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "06_pareto_bubble", OUT_DIR)
    plt.close(fig)
    print("  [6/13] 散点气泡图 — 帕累托前沿 ✓")


# ============================================================
# 图7：相关散点图 — 硬度预测vs实测
# ============================================================
def plot_predicted_vs_actual_hardness(data):
    preds = data["preds"]
    target = "硬度"

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    all_vals = []
    for i, model in enumerate(["RF", "LightGBM"]):
        key = f"{model}_{target}"
        if key not in preds:
            continue
        p = preds[key]
        x_vals = p["真实值"].values
        y_vals = p["预测值"].values
        all_vals.extend(x_vals.tolist())
        all_vals.extend(y_vals.tolist())
        ax.scatter(x_vals, y_vals,
                   color=CAT_COLORS[i * 3], s=40, alpha=0.7,
                   edgecolors="black", linewidths=0.5, label=model, zorder=3)

    if all_vals:
        lo, hi = min(all_vals) - 20, max(all_vals) + 20
        ax.plot([lo, hi], [lo, hi], "--", color=GRAY, linewidth=2, label="y=x", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    ax.set_xlabel("Measured Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_ylabel("Predicted Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left", frameon=False)
    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "07_predicted_vs_actual_hardness", OUT_DIR)
    plt.close(fig)
    print("  [7/13] 相关散点图 — 硬度预测vs实测 ✓")


# ============================================================
# 图8：相关散点图 — 腐蚀预测vs实测
# ============================================================
def plot_predicted_vs_actual_corrosion(data):
    preds = data["preds"]
    target = "腐蚀电流"

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    all_vals = []
    for i, model in enumerate(["RF", "LightGBM"]):
        key = f"{model}_{target}"
        if key not in preds:
            continue
        p = preds[key]
        x_vals = p["真实值"].values * 1e6
        y_vals = p["预测值"].values * 1e6
        all_vals.extend(x_vals.tolist())
        all_vals.extend(y_vals.tolist())
        ax.scatter(x_vals, y_vals,
                   color=CAT_COLORS[i * 3], s=40, alpha=0.7,
                   edgecolors="black", linewidths=0.5, label=model, zorder=3)

    if all_vals:
        lo, hi = min(all_vals) - 0.5, max(all_vals) + 0.5
        ax.plot([lo, hi], [lo, hi], "--", color=GRAY, linewidth=2, label="y=x", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    ax.set_xlabel(r"Measured ($\times 10^{-6}$ A/cm$^2$)",
                 fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_ylabel(r"Predicted ($\times 10^{-6}$ A/cm$^2$)",
                 fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left", frameon=False)
    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "08_predicted_vs_actual_corrosion", OUT_DIR)
    plt.close(fig)
    print("  [8/13] 相关散点图 — 腐蚀预测vs实测 ✓")


# ============================================================
# 图9：趋势图 — 硬度vs功率含误差带
# ============================================================
def plot_hardness_power_trend(data):
    df = data["df_clean"]
    loo = data["loo"]

    # 自动检测功率轴上的大间隔, 排除孤立点, 避免空白区域
    sorted_powers = np.sort(df["激光功率"].unique())
    if len(sorted_powers) > 1:
        gaps = np.diff(sorted_powers)
        max_gap_idx = int(np.argmax(gaps))
        if gaps[max_gap_idx] > 500:
            upper_bound = sorted_powers[max_gap_idx]
        else:
            upper_bound = sorted_powers[-1]
    else:
        upper_bound = df["激光功率"].max()
    df_plot = df[df["激光功率"] <= upper_bound].copy()

    p_min = df_plot["激光功率"].min()
    p_max = df_plot["激光功率"].max()
    power_bins = np.linspace(p_min, p_max, 12)
    bin_centers = []
    bin_means = []
    bin_stds = []
    for i in range(len(power_bins) - 1):
        if i == len(power_bins) - 2:
            mask = (df_plot["激光功率"] >= power_bins[i]) & (df_plot["激光功率"] <= power_bins[i+1])
        else:
            mask = (df_plot["激光功率"] >= power_bins[i]) & (df_plot["激光功率"] < power_bins[i+1])
        subset = df_plot.loc[mask, "硬度"]
        if len(subset) == 0:
            continue
        bin_centers.append((power_bins[i] + power_bins[i+1]) / 2)
        bin_means.append(subset.mean())
        bin_stds.append(subset.std())
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.nan_to_num(bin_stds, nan=0)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                    color=CAT_COLORS[5], alpha=0.2, label="±1σ band", zorder=2)
    ax.plot(bin_centers, bin_means, color=CAT_COLORS[5], linewidth=2.5,
            marker="o", markersize=8, markeredgecolor="black",
            markeredgewidth=1.2, label="Mean", zorder=4)
    ax.scatter(df_plot["激光功率"], df_plot["硬度"], color=CAT_COLORS[2], s=20,
               alpha=0.25, edgecolors="none", label="Raw data", zorder=3)
    ax.scatter(loo["功率_W"], loo["硬度_实测_HV"],
               color=CAT_COLORS[3], s=100, marker="D", edgecolors="black",
               linewidths=1.5, zorder=5, label="Rockit485")

    ax.set_xlabel("Laser Power (W)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_ylabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_xlim(p_min - 100, p_max + 100)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left", frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "09_hardness_power_trend", OUT_DIR)
    plt.close(fig)
    print("  [9/13] 趋势图 — 硬度vs功率 ✓")


# ============================================================
# 图10：箱线图 — 硬度模型预测误差分布
# ============================================================
def plot_error_boxplot_hardness(data):
    preds = data["preds"]
    target = "硬度"

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    error_data = []
    labels = []
    for model in ["KNN", "SVR", "RF", "LightGBM"]:
        key = f"{model}_{target}"
        if key in preds:
            err = preds[key]["相对误差(%)"].values
            err_clipped = np.clip(err, -100, 200)
            error_data.append(err_clipped)
            labels.append(model)

    bp = ax.boxplot(error_data, positions=range(len(labels)),
                    widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(marker="o", markersize=4,
                                                      alpha=0.4))

    box_colors = [CAT_COLORS[0], CAT_COLORS[3], CAT_COLORS[5], CAT_COLORS[6]]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(box_colors[i])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#666666")
            line.set_linewidth(1.5)
    for line in bp["medians"]:
        line.set_color(DARK)
        line.set_linewidth(2.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Relative Error (%)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.axhline(y=0, color=CAT_COLORS[3], linewidth=1.5, linestyle="--", alpha=0.6)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "10_error_boxplot_hardness", OUT_DIR)
    plt.close(fig)
    print("  [10/13] 箱线图 — 硬度模型误差 ✓")


# ============================================================
# 图11：箱线图 — 腐蚀模型预测误差分布
# ============================================================
def plot_error_boxplot_corrosion(data):
    preds = data["preds"]
    target = "腐蚀电流"

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    error_data = []
    labels = []
    for model in ["KNN", "SVR", "RF", "LightGBM"]:
        key = f"{model}_{target}"
        if key in preds:
            err = preds[key]["相对误差(%)"].values
            err_clipped = np.clip(err, -100, 200)
            error_data.append(err_clipped)
            labels.append(model)

    bp = ax.boxplot(error_data, positions=range(len(labels)),
                    widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(marker="o", markersize=4,
                                                      alpha=0.4))

    box_colors = [CAT_COLORS[0], CAT_COLORS[3], CAT_COLORS[5], CAT_COLORS[6]]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(box_colors[i])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#666666")
            line.set_linewidth(1.5)
    for line in bp["medians"]:
        line.set_color(DARK)
        line.set_linewidth(2.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Relative Error (%)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.axhline(y=0, color=CAT_COLORS[3], linewidth=1.5, linestyle="--", alpha=0.6)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "11_error_boxplot_corrosion", OUT_DIR)
    plt.close(fig)
    print("  [11/13] 箱线图 — 腐蚀模型误差 ✓")


# ============================================================
# 图12：差异气泡图 — SHAP硬度重要性对比
# ============================================================
def plot_shap_bubble_hardness(data):
    shap_base = data["shap_base"]
    shap_fs = data["shap_fs"]
    target_col = "abs_shap_mean_hv"

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    merged = shap_base[["特征名", target_col]].rename(
        columns={target_col: "Baseline"})
    merged = merged.merge(
        shap_fs[["特征名", target_col]].rename(columns={target_col: "Few-shot"}),
        on="特征名", how="inner",
    )
    merged = merged.sort_values("Baseline", ascending=True)
    merged["特征名"] = merged["特征名"].map(FEATURE_CN_TO_ABBR)

    y = np.arange(len(merged))
    base_vals = merged["Baseline"].values
    fs_vals = merged["Few-shot"].values

    max_val = max(base_vals.max(), fs_vals.max()) if max(base_vals.max(), fs_vals.max()) > 0 else 1

    ax.scatter(base_vals, y + 0.18,
               s=base_vals / max_val * 300 + 20,
               color=CAT_COLORS[5], alpha=0.65, edgecolors="black",
               linewidths=0.8, label="Baseline", zorder=3)
    ax.scatter(fs_vals, y - 0.18,
               s=fs_vals / max_val * 300 + 20,
               color=CAT_COLORS[3], alpha=0.65, edgecolors="black",
               linewidths=0.8, label="Few-shot", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["特征名"], fontsize=FONT_SIZE_SHAP_TICK)
    ax.set_xlabel("SHAP |mean| (HV)", fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_SIZE_SHAP_LEGEND, loc="lower right", frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False,
             tick_labelsize=FONT_SIZE_SHAP_TICK)

    save(fig, "12_shap_bubble_hardness", OUT_DIR)
    plt.close(fig)
    print("  [12/13] 差异气泡图 — SHAP硬度重要性 ✓")


# ============================================================
# 图13：差异气泡图 — SHAP腐蚀重要性对比
# ============================================================
def plot_shap_bubble_corrosion(data):
    shap_base = data["shap_base"]
    shap_fs = data["shap_fs"]
    target_col = "abs_shap_mean_corr"

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    merged = shap_base[["特征名", target_col]].rename(
        columns={target_col: "Baseline"})
    merged = merged.merge(
        shap_fs[["特征名", target_col]].rename(columns={target_col: "Few-shot"}),
        on="特征名", how="inner",
    )
    merged = merged.sort_values("Baseline", ascending=True)
    merged["特征名"] = merged["特征名"].map(FEATURE_CN_TO_ABBR)

    y = np.arange(len(merged))
    base_vals = merged["Baseline"].values
    fs_vals = merged["Few-shot"].values

    max_val = max(base_vals.max(), fs_vals.max()) if max(base_vals.max(), fs_vals.max()) > 0 else 1

    ax.scatter(base_vals, y + 0.18,
               s=base_vals / max_val * 300 + 20,
               color=CAT_COLORS[5], alpha=0.65, edgecolors="black",
               linewidths=0.8, label="Baseline", zorder=3)
    ax.scatter(fs_vals, y - 0.18,
               s=fs_vals / max_val * 300 + 20,
               color=CAT_COLORS[3], alpha=0.65, edgecolors="black",
               linewidths=0.8, label="Few-shot", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["特征名"], fontsize=FONT_SIZE_SHAP_TICK)
    ax.set_xlabel("SHAP |mean| (log)", fontsize=FONT_SIZE_SHAP_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_SIZE_SHAP_LEGEND, loc="lower right", frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False,
             tick_labelsize=FONT_SIZE_SHAP_TICK)

    save(fig, "13_shap_bubble_corrosion", OUT_DIR)
    plt.close(fig)
    print("  [13/13] 差异气泡图 — SHAP腐蚀重要性 ✓")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("数据可视化图谱 — 论文级风格 (13张)")
    print(f"渐变色: {GRADIENT_TOP} → {GRADIENT_BOTTOM}")
    print(f"分辨率: 300 DPI | 格式: PNG")
    print("=" * 60)
    print()

    print("[加载数据]")
    data = load_all_data()
    print(f"  样本: {len(data['df_clean'])} 条")
    print()

    print("[绘制图表]")
    plot_correlation_heatmap(data)
    plot_model_comparison_bar(data)
    plot_pca_scatter(data)
    plot_violin_by_cr(data)
    plot_stacked_composition(data)
    plot_pareto_bubble(data)
    plot_predicted_vs_actual_hardness(data)
    plot_predicted_vs_actual_corrosion(data)
    plot_hardness_power_trend(data)
    plot_error_boxplot_hardness(data)
    plot_error_boxplot_corrosion(data)
    plot_shap_bubble_hardness(data)
    plot_shap_bubble_corrosion(data)

    print()
    print("=" * 60)
    print(f"全部完成！共 13 张图，保存于 {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
