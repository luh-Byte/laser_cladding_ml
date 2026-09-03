"""
趋势约束LOO对比图(3张)
======================
展示趋势约束 (Logistic Ranking Loss) 相比原始方法的改善:
  27. 硬度LOO对比 — 实测/原始/趋势约束 三条线
  28. 腐蚀LOO对比 — 实测/原始/趋势约束 三条线
  29. 指标对比柱状图 — R²与MAE的改善
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plot_style import (
    create_gradient_rect, style_ax, save,
    PALETTE, CAT_COLORS, DARK, GRAY, GRID_COLOR,
    FIG_SIZE, FIG_SIZE_WIDE,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FONT_SIZE_ANNOT, FONT_SIZE_TITLE,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = "outputs/figures"
TREND_DIR = "outputs/trend_constrained"

# 颜色
COLOR_ACTUAL = CAT_COLORS[1]     # 珊瑚 — 实测
COLOR_ORIG = CAT_COLORS[0]        # 粉 — 原始LOO
COLOR_TREND = CAT_COLORS[4]       # 青 — 趋势约束LOO


def _load_data():
    path = os.path.join(TREND_DIR, "loo_full_comparison.csv")
    return pd.read_csv(path)


# ============================================================
# 图27: 硬度LOO对比
# ============================================================
def plot_hardness_loo_comparison(df):
    powers = df["功率_W"].values
    actual = df["硬度_实测_HV"].values
    orig = df["硬度_原始LOO_HV"].values
    trend = df["硬度_趋势约束LOO_HV"].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 实测
    ax.plot(powers, actual, '-', color=COLOR_ACTUAL, linewidth=2.5,
            marker='o', markersize=9, markerfacecolor=COLOR_ACTUAL,
            markeredgecolor='black', markeredgewidth=1.0,
            label='Actual', zorder=5)
    # 原始LOO
    ax.plot(powers, orig, '--', color=COLOR_ORIG, linewidth=2.0,
            marker='s', markersize=7, markerfacecolor=COLOR_ORIG,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Original LOO', zorder=4)
    # 趋势约束LOO
    ax.plot(powers, trend, '-.', color=COLOR_TREND, linewidth=2.0,
            marker='D', markersize=7, markerfacecolor=COLOR_TREND,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Trend-Constrained LOO', zorder=4)

    # 标注峰值
    peak_actual = np.argmax(actual)
    peak_trend = np.argmax(trend)
    ax.annotate(f'Peak: {powers[peak_actual]}W',
                xy=(powers[peak_actual], actual[peak_actual]),
                xytext=(powers[peak_actual] + 80, actual[peak_actual] + 12),
                fontsize=FONT_SIZE_ANNOT, fontweight='bold', color=COLOR_ACTUAL,
                arrowprops=dict(arrowstyle='->', color=COLOR_ACTUAL, lw=1.5))

    # 趋势约束峰值标注
    if peak_trend == peak_actual:
        ax.annotate(f'✓ {powers[peak_trend]}W',
                    xy=(powers[peak_trend], trend[peak_trend]),
                    xytext=(powers[peak_trend] - 200, trend[peak_trend] - 15),
                    fontsize=FONT_SIZE_ANNOT, fontweight='bold', color=COLOR_TREND,
                    arrowprops=dict(arrowstyle='->', color=COLOR_TREND, lw=1.5))

    ax.set_xlabel("Laser Power (W)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Hardness LOO: Original vs Trend-Constrained",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, loc='upper left')
    ax.set_xticks(powers)

    y_min = min(actual.min(), orig.min(), trend.min())
    y_max = max(actual.max(), orig.max(), trend.max())
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "27_hardness_loo_comparison", OUT_DIR)
    plt.close(fig)
    print("  [27] 硬度LOO对比 — 原始 vs 趋势约束 ✓")


# ============================================================
# 图28: 腐蚀LOO对比
# ============================================================
def plot_corrosion_loo_comparison(df):
    powers = df["功率_W"].values
    actual = df["腐蚀_实测_Acm2"].values
    orig = df["腐蚀_原始LOO_Acm2"].values
    trend = df["腐蚀_趋势约束LOO_Acm2"].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 转为 ×10⁻⁷ A/cm² 便于阅读
    actual_p = actual * 1e7
    orig_p = orig * 1e7
    trend_p = trend * 1e7

    # 实测
    ax.plot(powers, actual_p, '-', color=COLOR_ACTUAL, linewidth=2.5,
            marker='o', markersize=9, markerfacecolor=COLOR_ACTUAL,
            markeredgecolor='black', markeredgewidth=1.0,
            label='Actual', zorder=5)
    # 原始LOO
    ax.plot(powers, orig_p, '--', color=COLOR_ORIG, linewidth=2.0,
            marker='s', markersize=7, markerfacecolor=COLOR_ORIG,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Original LOO', zorder=4)
    # 趋势约束LOO
    ax.plot(powers, trend_p, '-.', color=COLOR_TREND, linewidth=2.0,
            marker='D', markersize=7, markerfacecolor=COLOR_TREND,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Trend-Constrained LOO', zorder=4)

    # 标注谷值
    valley_actual = np.argmin(actual)
    valley_trend = np.argmin(trend)
    ax.annotate(f'Valley: {powers[valley_actual]}W',
                xy=(powers[valley_actual], actual_p[valley_actual]),
                xytext=(powers[valley_actual] - 250, actual_p[valley_actual] + 0.8),
                fontsize=FONT_SIZE_ANNOT, fontweight='bold', color=COLOR_ACTUAL,
                arrowprops=dict(arrowstyle='->', color=COLOR_ACTUAL, lw=1.5))

    if valley_trend == valley_actual:
        ax.annotate(f'✓ {powers[valley_trend]}W',
                    xy=(powers[valley_trend], trend_p[valley_trend]),
                    xytext=(powers[valley_trend] - 250, trend_p[valley_trend] + 1.0),
                    fontsize=FONT_SIZE_ANNOT, fontweight='bold', color=COLOR_TREND,
                    arrowprops=dict(arrowstyle='->', color=COLOR_TREND, lw=1.5))

    ax.set_xlabel("Laser Power (W)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Corrosion Current (×10⁻⁷ A/cm²)",
                  fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Corrosion LOO: Original vs Trend-Constrained",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, loc='upper right')
    ax.set_xticks(powers)

    y_min = min(actual_p.min(), orig_p.min(), trend_p.min())
    y_max = max(actual_p.max(), orig_p.max(), trend_p.max())
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "28_corrosion_loo_comparison", OUT_DIR)
    plt.close(fig)
    print("  [28] 腐蚀LOO对比 — 原始 vs 趋势约束 ✓")


# ============================================================
# 计算指标 (共用)
# ============================================================
def _compute_metrics(df):
    actual_hv = df["硬度_实测_HV"].values
    orig_hv = df["硬度_原始LOO_HV"].values
    trend_hv = df["硬度_趋势约束LOO_HV"].values
    actual_corr = df["腐蚀_实测_Acm2"].values
    orig_corr = df["腐蚀_原始LOO_Acm2"].values
    trend_corr = df["腐蚀_趋势约束LOO_Acm2"].values

    hv_r2_orig = 1 - np.sum((actual_hv - orig_hv)**2) / np.sum((actual_hv - np.mean(actual_hv))**2)
    hv_r2_trend = 1 - np.sum((actual_hv - trend_hv)**2) / np.sum((actual_hv - np.mean(actual_hv))**2)
    hv_mae_orig = np.mean(np.abs(orig_hv - actual_hv))
    hv_mae_trend = np.mean(np.abs(trend_hv - actual_hv))

    actual_corr_log = np.log10(actual_corr * 10000)
    orig_corr_log = np.log10(orig_corr * 10000)
    trend_corr_log = np.log10(trend_corr * 10000)
    corr_r2_orig = 1 - np.sum((actual_corr_log - orig_corr_log)**2) / \
                   np.sum((actual_corr_log - np.mean(actual_corr_log))**2)
    corr_r2_trend = 1 - np.sum((actual_corr_log - trend_corr_log)**2) / \
                    np.sum((actual_corr_log - np.mean(actual_corr_log))**2)

    orig_signs = np.sign(np.diff(orig_hv))
    trend_signs = np.sign(np.diff(trend_hv))
    actual_signs = np.sign(np.diff(actual_hv))
    hv_orig_match = int(np.sum(orig_signs == actual_signs))
    hv_trend_match = int(np.sum(trend_signs == actual_signs))

    orig_valley = np.argmin(orig_corr)
    trend_valley = np.argmin(trend_corr)
    actual_valley = np.argmin(actual_corr)
    corr_orig_ok = int(orig_valley == actual_valley)
    corr_trend_ok = int(trend_valley == actual_valley)

    return {
        "hv_r2_orig": hv_r2_orig, "hv_r2_trend": hv_r2_trend,
        "hv_mae_orig": hv_mae_orig, "hv_mae_trend": hv_mae_trend,
        "corr_r2_orig": corr_r2_orig, "corr_r2_trend": corr_r2_trend,
        "hv_orig_match": hv_orig_match, "hv_trend_match": hv_trend_match,
        "corr_orig_ok": corr_orig_ok, "corr_trend_ok": corr_trend_ok,
    }


# ============================================================
# 图29: R²对比柱状图
# ============================================================
def plot_r2_comparison(df):
    m = _compute_metrics(df)

    categories = ['Hardness\nR²', 'Corrosion\nlog R²']
    orig_vals = [m["hv_r2_orig"], m["corr_r2_orig"]]
    trend_vals = [m["hv_r2_trend"], m["corr_r2_trend"]]

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bars1 = ax.bar(x - width/2, orig_vals, width, color=COLOR_ORIG,
                    edgecolor='black', linewidth=0.8, label='Original', zorder=3)
    bars2 = ax.bar(x + width/2, trend_vals, width, color=COLOR_TREND,
                    edgecolor='black', linewidth=0.8, label='Trend-Constrained', zorder=3)

    for bar, val in zip(bars1, orig_vals):
        y_text = val + 0.05 if val > 0 else val - 0.15
        ax.text(bar.get_x() + bar.get_width()/2, y_text,
                 f"{val:.3f}", ha='center', va='bottom' if val > 0 else 'top',
                 fontsize=FONT_SIZE_ANNOT, fontweight='bold')
    for bar, val in zip(bars2, trend_vals):
        y_text = val + 0.05 if val > 0 else val - 0.15
        ax.text(bar.get_x() + bar.get_width()/2, y_text,
                 f"{val:.3f}", ha='center', va='bottom' if val > 0 else 'top',
                 fontsize=FONT_SIZE_ANNOT, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=FONT_SIZE_TICK, fontweight='bold')
    ax.set_ylabel("R²", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("LOO R² Comparison", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=8)
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, loc='lower right')
    ax.axhline(y=0, color=GRAY, linewidth=0.8, linestyle='-', alpha=0.5)

    y_min = min(min(orig_vals), min(trend_vals)) - 0.3
    y_max = max(max(orig_vals), max(trend_vals)) + 0.3
    ax.set_ylim(y_min, y_max)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "29_r2_comparison", OUT_DIR)
    plt.close(fig)
    print("  [29] R²对比柱状图 — 趋势约束改善 ✓")


# ============================================================
# 图30: 精度与趋势匹配
# ============================================================
def plot_accuracy_trend(df):
    m = _compute_metrics(df)

    cat2 = ['Hardness\nMAE (HV)', 'Hardness\nTrend Match', 'Corrosion\nValley Match']
    orig2 = [m["hv_mae_orig"], m["hv_orig_match"], m["corr_orig_ok"]]
    trend2 = [m["hv_mae_trend"], m["hv_trend_match"], m["corr_trend_ok"]]

    x = np.arange(3)
    width = 0.32

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bars_orig = ax.bar(x - width/2, orig2, width, color=COLOR_ORIG,
                       edgecolor='black', linewidth=0.8, label='Original', zorder=3)
    bars_trend = ax.bar(x + width/2, trend2, width, color=COLOR_TREND,
                        edgecolor='black', linewidth=0.8, label='Trend-Constrained', zorder=3)

    for bar, val in zip(bars_orig, orig2):
        if isinstance(val, float):
            label = f"{val:.1f}"
        else:
            label = "✓" if val == 1 else "✗"
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                label, ha='center', va='bottom',
                fontsize=FONT_SIZE_ANNOT, fontweight='bold')
    for bar, val in zip(bars_trend, trend2):
        if isinstance(val, float):
            label = f"{val:.1f}"
        else:
            label = "✓" if val == 1 else "✗"
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                label, ha='center', va='bottom',
                fontsize=FONT_SIZE_ANNOT, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cat2, fontsize=FONT_SIZE_TICK, fontweight='bold')
    ax.set_title("LOO Accuracy & Trend", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=8)
    ax.set_ylim(0, max(m["hv_mae_orig"], m["hv_mae_trend"]) * 1.5)

    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, loc='upper right')

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "30_accuracy_trend", OUT_DIR)
    plt.close(fig)
    print("  [30] 精度与趋势匹配 — 趋势约束改善 ✓")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("趋势约束LOO对比图 — 4张")
    print("=" * 60)

    df = _load_data()
    print(f"\n[数据] {len(df)} 个实验点\n")

    print("[绘制图表]")
    plot_hardness_loo_comparison(df)
    plot_corrosion_loo_comparison(df)
    plot_r2_comparison(df)
    plot_accuracy_trend(df)

    print(f"\n{'=' * 60}")
    print(f"完成！共 4 张对比图，保存于 {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
