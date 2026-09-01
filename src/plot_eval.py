"""
拟合精度可视化图谱(7张)
===================================
基于全量数据模型对4个Rockit485实验点的预测结果, 每张图独立输出:
  20. 哑铃图 — 硬度 实测vs预测
  21. 哑铃图 — 腐蚀 实测vs预测
  22. 面积差图 — 硬度趋势
  23. 面积差图 — 腐蚀趋势
  24. 棒棒糖图 — 硬度误差
  25. 棒棒糖图 — 腐蚀误差
  26. 热力矩阵 — 功率×目标相对误差
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plot_style import (
    create_gradient_rect, style_ax, save,
    PALETTE, CAT_COLORS, DARK, GRAY, GRID_COLOR,
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_HEATMAP,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FONT_SIZE_CBAR, FONT_SIZE_ANNOT, FONT_SIZE_TITLE,
)
from src.config import OUTPUT_DIR

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT_DIR = os.path.join(OUTPUT_DIR, "figures")
EVAL_DIR = os.path.join(OUTPUT_DIR, "full_data_eval")

COLOR_ACTUAL = CAT_COLORS[1]   # #E97A6F 珊瑚
COLOR_PRED = CAT_COLORS[4]      # #00BEB3 青
COLOR_FILL = CAT_COLORS[3]      # #E0BEB3 浅粉
COLOR_POS = CAT_COLORS[1]       # 偏高=珊瑚
COLOR_NEG = CAT_COLORS[4]       # 偏低=青


def _load_eval_data():
    return pd.read_csv(os.path.join(EVAL_DIR, "full_data_eval_results.csv"))


# ============================================================
# 图20: 哑铃图 — 硬度
# ============================================================
def plot_dumbbell_hardness(df):
    powers = df["功率_W"].values
    hv_actual = df["硬度_实测_HV"].values
    hv_pred = df["硬度_预测_HV"].values
    y_pos = np.arange(len(powers))[::-1]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for i in range(len(powers)):
        ax.plot([hv_actual[i], hv_pred[i]], [y_pos[i], y_pos[i]],
                color="#BDBDBD", linewidth=3, zorder=1, solid_capstyle="round")
        ax.scatter(hv_actual[i], y_pos[i], color=COLOR_ACTUAL, s=120,
                   zorder=3, edgecolors='black', linewidths=0.8)
        ax.scatter(hv_pred[i], y_pos[i], color=COLOR_PRED, s=120,
                   zorder=3, edgecolors='black', linewidths=0.8)
        err = hv_pred[i] - hv_actual[i]
        x_mid = (hv_actual[i] + hv_pred[i]) / 2
        ax.text(x_mid, y_pos[i] + 0.25, f"{err:+.1f} HV",
                ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT,
                fontweight='bold', color=DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p}W" for p in powers])
    ax.set_xlabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Hardness: Actual vs Predicted",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)

    hv_min = min(hv_actual.min(), hv_pred.min())
    hv_max = max(hv_actual.max(), hv_pred.max())
    margin = (hv_max - hv_min) * 0.15
    ax.set_xlim(hv_min - margin, hv_max + margin)
    ax.set_ylim(-0.6, len(powers) - 0.4)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ACTUAL,
                   markersize=10, markeredgecolor='black', label='Actual'),
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_PRED, color='w',
                   markersize=10, markeredgecolor='black', label='Predicted'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=FONT_SIZE_LEGEND, frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "20_dumbbell_hardness", OUT_DIR)
    plt.close(fig)
    print("  [20] 哑铃图 — 硬度 实测vs预测 ✓")


# ============================================================
# 图21: 哑铃图 — 腐蚀
# ============================================================
def plot_dumbbell_corrosion(df):
    powers = df["功率_W"].values
    corr_actual = df["腐蚀_实测_Acm2"].values
    corr_pred = df["腐蚀_预测_Acm2"].values
    y_pos = np.arange(len(powers))[::-1]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # log空间绘制
    a_log = np.log10(corr_actual * 10000)
    p_log = np.log10(corr_pred * 10000)

    for i in range(len(powers)):
        ax.plot([a_log[i], p_log[i]], [y_pos[i], y_pos[i]],
                color="#BDBDBD", linewidth=3, zorder=1, solid_capstyle="round")
        ax.scatter(a_log[i], y_pos[i], color=COLOR_ACTUAL, s=120,
                   zorder=3, edgecolors='black', linewidths=0.8)
        ax.scatter(p_log[i], y_pos[i], color=COLOR_PRED, s=120,
                   zorder=3, edgecolors='black', linewidths=0.8)
        ratio = corr_pred[i] / corr_actual[i]
        x_mid = (a_log[i] + p_log[i]) / 2
        ax.text(x_mid, y_pos[i] + 0.25, f"{ratio:.2f}x",
                ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT,
                fontweight='bold', color=DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p}W" for p in powers])
    ax.set_xlabel("Corrosion Current (log10[A/cm²×10⁴])",
                  fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Corrosion: Actual vs Predicted",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)

    cl_min = min(a_log.min(), p_log.min())
    cl_max = max(a_log.max(), p_log.max())
    margin = (cl_max - cl_min) * 0.15
    ax.set_xlim(cl_min - margin, cl_max + margin)
    ax.set_ylim(-0.6, len(powers) - 0.4)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ACTUAL,
                   markersize=10, markeredgecolor='black', label='Actual'),
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_PRED, color='w',
                   markersize=10, markeredgecolor='black', label='Predicted'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=FONT_SIZE_LEGEND, frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "21_dumbbell_corrosion", OUT_DIR)
    plt.close(fig)
    print("  [21] 哑铃图 — 腐蚀 实测vs预测 ✓")


# ============================================================
# 图22: 面积差图 — 硬度趋势
# ============================================================
def plot_area_diff_hardness(df):
    powers = df["功率_W"].values
    hv_actual = df["硬度_实测_HV"].values
    hv_pred = df["硬度_预测_HV"].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(powers, hv_actual, '-', color=COLOR_ACTUAL, linewidth=2.5,
            marker='o', markersize=8, markerfacecolor=COLOR_ACTUAL,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Actual', zorder=3)
    ax.plot(powers, hv_pred, '--', color=COLOR_PRED, linewidth=2.5,
            marker='s', markersize=7, markerfacecolor=COLOR_PRED,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Predicted', zorder=3)
    ax.fill_between(powers, hv_actual, hv_pred, alpha=0.25,
                    color=COLOR_FILL, zorder=2)

    ax.set_xlabel("Laser Power (W)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Hardness: Trend Agreement",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=8)
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, loc='lower right')
    ax.set_xticks(powers)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "22_area_diff_hardness", OUT_DIR)
    plt.close(fig)
    print("  [22] 面积差图 — 硬度趋势 ✓")


# ============================================================
# 图23: 面积差图 — 腐蚀趋势
# ============================================================
def plot_area_diff_corrosion(df):
    powers = df["功率_W"].values
    corr_actual = df["腐蚀_实测_Acm2"].values
    corr_pred = df["腐蚀_预测_Acm2"].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(powers, corr_actual * 1e7, '-', color=COLOR_ACTUAL, linewidth=2.5,
            marker='o', markersize=8, markerfacecolor=COLOR_ACTUAL,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Actual', zorder=3)
    ax.plot(powers, corr_pred * 1e7, '--', color=COLOR_PRED, linewidth=2.5,
            marker='s', markersize=7, markerfacecolor=COLOR_PRED,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Predicted', zorder=3)
    ax.fill_between(powers, corr_actual * 1e7, corr_pred * 1e7, alpha=0.25,
                    color=COLOR_FILL, zorder=2)

    ax.set_xlabel("Laser Power (W)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Corrosion Current (×10⁻⁷ A/cm²)",
                  fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Corrosion: Trend Agreement",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=8)
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=False, loc='upper right')
    ax.set_xticks(powers)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "23_area_diff_corrosion", OUT_DIR)
    plt.close(fig)
    print("  [23] 面积差图 — 腐蚀趋势 ✓")


# ============================================================
# 图24: 棒棒糖图 — 硬度误差
# ============================================================
def plot_lollipop_hardness(df):
    powers = df["功率_W"].values
    hv_err = ((df["硬度_预测_HV"] - df["硬度_实测_HV"]) / df["硬度_实测_HV"] * 100).values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for i in range(len(powers)):
        val = hv_err[i]
        color = COLOR_POS if val >= 0 else COLOR_NEG
        ax.plot([0, val], [i, i], color=color, linewidth=2.5, zorder=1)
        ax.scatter(val, i, color=color, s=150, zorder=3,
                   edgecolors='black', linewidths=0.8)
        offset = max(abs(val) * 0.15, 0.03)
        ax.text(val + (offset if val >= 0 else -offset), i,
                f"{val:+.2f}%", va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=FONT_SIZE_ANNOT, fontweight='bold', color=DARK)

    ax.axvline(x=0, color=GRAY, linewidth=1.2, linestyle='-', alpha=0.5, zorder=0)
    ax.set_yticks(range(len(powers)))
    ax.set_yticklabels([f"{p}W" for p in powers])
    ax.invert_yaxis()
    ax.set_xlabel("Relative Error (%)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Hardness Prediction Error",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)
    hv_abs_max = max(abs(hv_err).max(), 0.5) * 1.6
    ax.set_xlim(-hv_abs_max, hv_abs_max)

    legend_elements = [
        Patch(facecolor=COLOR_POS, edgecolor='black', label='Overestimate'),
        Patch(facecolor=COLOR_NEG, edgecolor='black', label='Underestimate'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=FONT_SIZE_LEGEND, frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "24_lollipop_hardness", OUT_DIR)
    plt.close(fig)
    print("  [24] 棒棒糖图 — 硬度误差 ✓")


# ============================================================
# 图25: 棒棒糖图 — 腐蚀误差
# ============================================================
def plot_lollipop_corrosion(df):
    powers = df["功率_W"].values
    corr_err = ((df["腐蚀_预测_Acm2"] - df["腐蚀_实测_Acm2"]) / df["腐蚀_实测_Acm2"] * 100).values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for i in range(len(powers)):
        val = corr_err[i]
        color = COLOR_POS if val >= 0 else COLOR_NEG
        ax.plot([0, val], [i, i], color=color, linewidth=2.5, zorder=1)
        ax.scatter(val, i, color=color, s=150, zorder=3,
                   edgecolors='black', linewidths=0.8)
        offset = max(abs(val) * 0.15, 1.0)
        ax.text(val + (offset if val >= 0 else -offset), i,
                f"{val:+.1f}%", va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=FONT_SIZE_ANNOT, fontweight='bold', color=DARK)

    ax.axvline(x=0, color=GRAY, linewidth=1.2, linestyle='-', alpha=0.5, zorder=0)
    ax.set_yticks(range(len(powers)))
    ax.set_yticklabels([f"{p}W" for p in powers])
    ax.invert_yaxis()
    ax.set_xlabel("Relative Error (%)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Corrosion Prediction Error",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)
    corr_abs_max = max(abs(corr_err).max(), 5.0) * 1.4
    ax.set_xlim(-corr_abs_max, corr_abs_max)

    legend_elements = [
        Patch(facecolor=COLOR_POS, edgecolor='black', label='Overestimate'),
        Patch(facecolor=COLOR_NEG, edgecolor='black', label='Underestimate'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=FONT_SIZE_LEGEND, frameon=False)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "25_lollipop_corrosion", OUT_DIR)
    plt.close(fig)
    print("  [25] 棒棒糖图 — 腐蚀误差 ✓")


# ============================================================
# 图26: 热力矩阵 — 功率×目标相对误差
# ============================================================
def plot_heatmap_matrix(df):
    powers = df["功率_W"].values
    hv_err = np.abs(df["硬度_预测_HV"] - df["硬度_实测_HV"]) / df["硬度_实测_HV"] * 100
    corr_err = np.abs(df["腐蚀_预测_Acm2"] - df["腐蚀_实测_Acm2"]) / df["腐蚀_实测_Acm2"] * 100
    data = np.column_stack([hv_err.values, corr_err.values])

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    cmap = LinearSegmentedColormap.from_list(
        "err_cmap", ["#F8F9E4", "#E0BEB3", "#E97A6F", "#E8156E"], N=256)

    vmax = max(data.max(), 5.0)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_color = 'white' if data[i, j] > vmax * 0.6 else 'black'
            ax.text(j, i, f"{data[i, j]:.2f}%", ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hardness", "Corrosion"],
                        fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_yticks(range(len(powers)))
    ax.set_yticklabels([f"{p}W" for p in powers],
                        fontsize=FONT_SIZE_TICK, fontweight='bold')
    ax.set_title("Relative Error Matrix (%)",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("|Relative Error| (%)",
                   fontsize=FONT_SIZE_CBAR, fontweight='bold')
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')

    save(fig, "26_heatmap_error_matrix", OUT_DIR)
    plt.close(fig)
    print("  [26] 热力矩阵 — 功率×目标相对误差 ✓")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("拟合精度可视化 — 7张独立评估图")
    print("=" * 60)

    df = _load_eval_data()
    print(f"\n[数据] {len(df)} 个实验点\n")

    print("[绘制图表]")
    plot_dumbbell_hardness(df)
    plot_dumbbell_corrosion(df)
    plot_area_diff_hardness(df)
    plot_area_diff_corrosion(df)
    plot_lollipop_hardness(df)
    plot_lollipop_corrosion(df)
    plot_heatmap_matrix(df)

    print(f"\n{'=' * 60}")
    print(f"全部完成！共 7 张评估图，保存于 {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
