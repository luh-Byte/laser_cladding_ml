"""
多区域 SHAP 极坐标可视化（论文级）
==================================
参考目标样式（扇区=特征）：
- 扇区 = 13 个特征，每个扇区内 4 根径向柱 = 4 组平均 |SHAP|（分类色 + 图例）
- 外圈散点 = 样本 SHAP 值（红蓝发散色 = 归一化特征值）
- 切线方向特征名标签 + 径向刻度 + 右侧颜色条/图例
- 输出: outputs/figures/31_shap_polar.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plot_style import TNR_FONT, FONT_SIZE_LEGEND, FONT_SIZE_TICK, FONT_SIZE_LABEL
from src.config import OUTPUT_DIR

# ============================================================
# 数据配置：4 组 SHAP
# ============================================================
GROUPS = [
    ("Baseline Hardness", "outputs/shap_baseline/shap_vals_hv.npy",
     "outputs/shap_baseline/X_train_hv.csv"),
    ("Baseline Corrosion", "outputs/shap_baseline/shap_vals_corr.npy",
     "outputs/shap_baseline/X_train_corr.csv"),
    ("Few-shot Hardness", "outputs/shap_fewshot/shap_vals_hv_fs.npy",
     "outputs/shap_fewshot/X_fs.csv"),
    ("Few-shot Corrosion", "outputs/shap_fewshot/shap_vals_corr_fs.npy",
     "outputs/shap_fewshot/X_fs.csv"),
]

# 分组分类色（4 组，色相区分明显）
GROUP_COLORS = ["#E64B35", "#3C8DBC", "#00A087", "#E18727"]

# 特征值发散色（红=高值，蓝=低值）
FEATURE_CMAP_HEX = ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"]

# 中文特征名 → 缩写标签（紧凑显示）
FEATURE_NAME_MAP = {
    "激光功率": "LP",
    "扫描速度": "SS",
    "送粉速率": "PFR",
    "光斑直径": "SD",
    "离焦量": "DF",
    "C": "C",
    "Cr": "Cr",
    "Fe": "Fe",
    "Mn": "Mn",
    "线能量密度": "LED",
    "粉末能量比": "PER",
    "碳当量": "CE",
    "镍当量": "NE",
}

OUT_DIR = os.path.join(OUTPUT_DIR, "figures")


def plot_shap_polar():
    """绘制论文级多区域 SHAP 极坐标图"""
    # ---- 加载数据 ----
    shap_arrays = []
    X_arrays = []
    for _, sp, dp in GROUPS:
        shap_arrays.append(np.load(sp))
        X_arrays.append(pd.read_csv(dp).values)

    raw_names = list(pd.read_csv(GROUPS[0][2]).columns)
    feature_names = [FEATURE_NAME_MAP.get(n, n) for n in raw_names]
    n_feat = len(feature_names)
    n_grp = len(GROUPS)

    # 平均 |SHAP|: (n_grp, n_feat)
    mean_abs = np.array([np.abs(sv).mean(axis=0) for sv in shap_arrays])

    # 各特征跨组 min/max（用于特征值归一化）
    feat_min = np.min([X.min(axis=0) for X in X_arrays], axis=0)
    feat_max = np.max([X.max(axis=0) for X in X_arrays], axis=0)

    # 按目标分别归一化（硬度/腐蚀 SHAP 量纲差异大，全局归一化会使腐蚀柱不可见）
    hv_max_mean = max(mean_abs[0].max(), mean_abs[2].max())
    corr_max_mean = max(mean_abs[1].max(), mean_abs[3].max())
    target_max_mean = [hv_max_mean, corr_max_mean, hv_max_mean, corr_max_mean]

    hv_max_shap = max(np.abs(shap_arrays[0]).max(), np.abs(shap_arrays[2]).max())
    corr_max_shap = max(np.abs(shap_arrays[1]).max(), np.abs(shap_arrays[3]).max())
    target_max_shap = [hv_max_shap, corr_max_shap, hv_max_shap, corr_max_shap]

    # ---- 布局参数 ----
    inner_radius = 0.30      # 柱起点
    bar_max = 0.78           # 柱最大外径
    scatter_center = 0.95    # 散点中心半径
    scatter_spread = 0.16    # 散点径向展幅
    label_radius = 1.17      # 特征名标签半径
    frame_radius = 1.30      # 画布外径

    sector_width = 2 * np.pi / n_feat
    sector_centers = np.linspace(0, 2 * np.pi, n_feat, endpoint=False) + sector_width / 2

    cmap = LinearSegmentedColormap.from_list("feature", FEATURE_CMAP_HEX)
    norm = Normalize(0, 1)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("#FBFBF6")  # 极浅米白背景

    # ---- 径向网格线 + 刻度 ----
    theta_grid = np.linspace(0, 2 * np.pi, 300)
    for frac, tick in [(0.25, "0.25"), (0.50, "0.50"), (0.75, "0.75"), (1.00, "1.00")]:
        r = inner_radius + frac * (bar_max - inner_radius)
        ax.plot(theta_grid, np.full_like(theta_grid, r), color="#DDDDDD",
                lw=0.8, zorder=0)
        ax.text(np.pi / 2, r + 0.01, tick, fontsize=7, color="#999999",
                ha="center", va="bottom", zorder=1)

    # ---- 扇区分隔线（极淡）----
    for f in range(n_feat):
        sep = f * sector_width
        ax.plot([sep, sep], [inner_radius, bar_max], color="#E8E8E8",
                lw=0.6, zorder=0)

    # ---- 径向柱（分组重要性）----
    for f in range(n_feat):
        center = sector_centers[f]
        for g in range(n_grp):
            sub_w = sector_width / n_grp
            theta = center - sector_width / 2 + (g + 0.5) * sub_w
            height = mean_abs[g, f] / target_max_mean[g] * (bar_max - inner_radius)
            ax.bar(theta, height, width=sub_w * 0.9, bottom=inner_radius,
                   color=GROUP_COLORS[g], alpha=0.95, zorder=3,
                   edgecolor="white", linewidth=0.6)

    # ---- 外圈散点（样本 SHAP，颜色=特征值）----
    rng = np.random.RandomState(42)
    for f in range(n_feat):
        center = sector_centers[f]
        for g in range(n_grp):
            sv = shap_arrays[g][:, f]
            fv = X_arrays[g][:, f]
            norm_fv = (fv - feat_min[f]) / (feat_max[f] - feat_min[f] + 1e-8)
            radii = scatter_center + (sv / target_max_shap[g]) * scatter_spread
            jitter = rng.uniform(-1, 1, size=len(radii)) * sector_width / 2 * 0.85
            theta = np.full_like(radii, center) + jitter
            ax.scatter(theta, radii, c=norm_fv, cmap=cmap, norm=norm,
                       s=15, alpha=0.85, zorder=10, edgecolor="none")

    # ---- 特征名标签（切线方向，居中对齐扇区中心角）----
    for f in range(n_feat):
        angle = sector_centers[f]
        angle_deg = np.degrees(angle)
        rotation = angle_deg - 90
        if angle_deg > 180:
            rotation -= 180
        ax.text(angle, label_radius, feature_names[f], ha="center", va="center",
                fontsize=FONT_SIZE_LABEL, fontweight="bold", color="#333333",
                rotation=rotation, rotation_mode="anchor", zorder=6)

    ax.set_ylim(0, frame_radius)
    ax.axis("off")

    # ---- 颜色条（特征值，右上）----
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.85, 0.64, 0.022, 0.28])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Feature Value (normalized)", fontsize=FONT_SIZE_LEGEND)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 2)

    # ---- 分组图例（右下角，无背景）----
    handles = [Patch(color=c, label=l) for c, l in zip(GROUP_COLORS, [g[0] for g in GROUPS])]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.02),
               fontsize=FONT_SIZE_LEGEND - 1, frameon=False, ncol=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "S32_shap_polar.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成:", out_path)


if __name__ == "__main__":
    plot_shap_polar()
