"""
新增可视化图谱(6张)
===================================
在现有13张图基础上, 不重复内容的新视角:
  14. 雷达图 — Cr分组归一化参数画像
  15. 平行坐标图 — 全样本多维特征分布
  16. 等高线热力图 — LP×SS 硬度响应面
  17. 散点矩阵图 — 关键变量两两关系
  18. 山脊图 — 功率区间硬度分布密度
  19. Andrews曲线 — 多维样本曲线投影
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
    GRADIENT_TOP, GRADIENT_BOTTOM,
    PALETTE, CAT_COLORS, DARK, GRAY, GRID_COLOR,
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_HEATMAP,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FONT_SIZE_CBAR, FONT_SIZE_ANNOT, FONT_SIZE_TITLE,
)
from src.data_preprocessing import load_raw_data, clean_data
from src.feature_engineering import compute_derived_features

import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR
OUT_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 简写映射
FEATURE_ABBR = {
    "LP": "Laser Power", "SS": "Scan Speed", "PFR": "Powder Feed Rate",
    "SD": "Spot Diameter", "DF": "Defocus",
    "C": "C", "Cr": "Cr", "Si": "Si", "Ni": "Ni",
    "Fe": "Fe", "Mn": "Mn", "Mo": "Mo",
    "LED": "Line Energy Density", "PER": "Powder-Energy Ratio",
    "CE": "Carbon Equivalent", "NE": "Ni Equivalent",
    "HV": "Hardness", "IC": "Corrosion Current",
}
FEATURE_CN_TO_ABBR = {
    "激光功率": "LP", "扫描速度": "SS", "送粉速率": "PFR",
    "光斑直径": "SD", "离焦量": "DF",
    "C": "C", "Cr": "Cr", "Si": "Si", "Ni": "Ni",
    "Fe": "Fe", "Mn": "Mn", "Mo": "Mo",
    "线能量密度": "LED", "粉末能量比": "PER",
    "碳当量": "CE", "镍当量": "NE",
    "硬度": "HV", "腐蚀电流": "IC",
}


def _load_data():
    """加载数据用于绘图"""
    df_raw = load_raw_data()
    df_clean, _ = clean_data(df_raw)
    df_feat = compute_derived_features(df_clean.copy())
    return df_clean, df_feat


# ============================================================
# 图14: 雷达图 — Cr分组归一化参数画像
# ============================================================
def plot_radar_chart(df_clean, df_feat):
    # 分组
    df = df_feat.copy()
    df["Cr_group"] = pd.cut(df["Cr"], bins=[0, 20, 50, 80],
                            labels=["Cr<20%", "20-50%", "50-80%"])

    # 雷达图维度: 工艺 + 成分 + 衍生 + 性能
    radar_features = ["激光功率", "扫描速度", "送粉速率", "线能量密度",
                      "碳当量", "镍当量", "硬度", "腐蚀电流"]
    radar_abbr = [FEATURE_CN_TO_ABBR[f] for f in radar_features]

    # 全局归一化 0-1
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df[radar_features]),
                           columns=radar_features, index=df.index)
    df_norm["Cr_group"] = df["Cr_group"].values

    # 各组均值
    group_means = df_norm.groupby("Cr_group")[radar_features].mean()

    # 绘制雷达图
    N = len(radar_features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.90)

    group_colors = [CAT_COLORS[0], CAT_COLORS[3], CAT_COLORS[5]]
    group_labels = ["Cr<20%", "20-50%", "50-80%"]

    for i, (grp, color) in enumerate(zip(group_labels, group_colors)):
        vals = group_means.loc[grp].values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, '-', color=color, linewidth=2.5, label=grp, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.15, zorder=2)
        # 端点标记
        ax.scatter(angles[:-1], vals[:-1], color=color, s=50, zorder=4,
                   edgecolors='black', linewidths=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_abbr, fontsize=FONT_SIZE_TICK, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=FONT_SIZE_TICK - 2)
    ax.set_rlabel_position(90)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1),
              fontsize=FONT_SIZE_LEGEND, frameon=False)

    ax.set_title("Normalized Parameter Profile by Cr Group",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)

    save(fig, "14_radar_cr_group", OUT_DIR)
    plt.close(fig)
    print("  [14] 雷达图 — Cr分组归一化参数画像 ✓")


# ============================================================
# 图15: 平行坐标图 — 全样本多维特征分布
# ============================================================
def plot_parallel_coordinates(df_clean, df_feat):
    from sklearn.preprocessing import MinMaxScaler

    df = df_feat.copy()
    features = ["激光功率", "扫描速度", "送粉速率", "Cr",
                "线能量密度", "碳当量", "镍当量", "硬度", "腐蚀电流"]
    abbr = [FEATURE_CN_TO_ABBR[f] for f in features]

    scaler = MinMaxScaler()
    norm_vals = scaler.fit_transform(df[features])
    hardness_raw = df["硬度"].values

    # 按硬度排序, 低硬度在下层
    sort_idx = np.argsort(hardness_raw)
    norm_sorted = norm_vals[sort_idx]
    hardness_sorted = hardness_raw[sort_idx]

    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

    # 顺序色标: 浅棕→珊瑚→品红(起点足够深, 避免与背景融合)
    seq_cmap = LinearSegmentedColormap.from_list(
        "seq_palette", ["#E0BEB3", "#E97A6F", "#E8156E"], N=256)
    norm_hv = (hardness_sorted - hardness_sorted.min()) / (hardness_sorted.max() - hardness_sorted.min() + 1e-9)

    n_samples = len(norm_sorted)
    for i in range(n_samples):
        color = seq_cmap(norm_hv[i])
        ax.plot(range(len(features)), norm_sorted[i],
                color=color, alpha=0.25, linewidth=0.7, zorder=2)

    # 坐标轴设置
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(abbr, fontsize=FONT_SIZE_TICK, fontweight='bold',
                       rotation=30, ha='right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Normalized Value (0-1)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_xlabel("")
    ax.set_xlim(-0.3, len(features) - 0.7)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=seq_cmap,
                               norm=plt.Normalize(vmin=hardness_sorted.min(),
                                                  vmax=hardness_sorted.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Hardness (HV)", fontweight='bold', fontsize=FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "15_parallel_coordinates", OUT_DIR)
    plt.close(fig)
    print("  [15] 平行坐标图 — 全样本多维特征分布 ✓")


# ============================================================
# 图16: 等高线热力图 — LP×SS 硬度响应面
# ============================================================
def plot_contour_heatmap(df_clean, df_feat):
    from scipy.interpolate import griddata

    df = df_feat.copy()
    x = df["激光功率"].values
    y = df["扫描速度"].values
    z = df["硬度"].values

    # 创建网格
    xi = np.linspace(x.min(), x.max(), 80)
    yi = np.linspace(y.min(), y.max(), 80)
    xi, yi = np.meshgrid(xi, yi)

    # 线性插值
    zi = griddata((x, y), z, (xi, yi), method='linear')

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 顺序色标: 浅蓝→浅棕→珊瑚→品红(热力图填充用, 起点浅蓝增加层次感)
    seq_cmap = LinearSegmentedColormap.from_list(
        "seq_palette", ["#E5F2FB", "#E0BEB3", "#E97A6F", "#E8156E"], N=256)

    cf = ax.contourf(xi, yi, zi, levels=20, cmap=seq_cmap, alpha=0.9)
    cs = ax.contour(xi, yi, zi, levels=8, colors='#212121',
                    linewidths=0.7, alpha=0.5)
    ax.clabel(cs, fontsize=FONT_SIZE_ANNOT - 1, fmt='%.0f',
              colors='#212121', inline=True, inline_spacing=3)

    # 散点
    ax.scatter(x, y, c=z, cmap=seq_cmap, s=30, edgecolors='black',
              linewidths=0.5, zorder=5, alpha=0.85)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Hardness (HV)", fontweight='bold', fontsize=FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)

    ax.set_xlabel("Laser Power (W)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Scan Speed (mm/s)", fontsize=FONT_SIZE_LABEL, fontweight='bold')

    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "16_contour_lp_ss", OUT_DIR)
    plt.close(fig)
    print("  [16] 等高线热力图 — LP×SS 硬度响应面 ✓")


# ============================================================
# 图17: 散点矩阵图 — 关键变量两两关系
# ============================================================
def plot_scatter_matrix(df_clean, df_feat):
    from pandas.plotting import scatter_matrix

    df = df_feat.copy()
    cols = ["激光功率", "Cr", "线能量密度", "碳当量", "硬度", "腐蚀电流"]
    abbr = [FEATURE_CN_TO_ABBR[c] for c in cols]

    df_plot = df[cols].copy()
    df_plot.columns = abbr
    hardness = df["硬度"].values

    fig, axes = plt.subplots(len(cols), len(cols), figsize=(10, 10))
    fig.subplots_adjust(hspace=0.12, wspace=0.12)

    # 顺序色标: 浅棕→珊瑚→品红(起点足够深)
    seq_cmap = LinearSegmentedColormap.from_list(
        "seq_palette", ["#E0BEB3", "#E97A6F", "#E8156E"], N=256)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax = axes[i, j]
            if i == j:
                # 对角线: KDE
                from scipy.stats import gaussian_kde
                data = df_plot.iloc[:, i].dropna().values
                if len(data) > 1 and data.std() > 0:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    ax.fill_between(x_range, kde(x_range), color=CAT_COLORS[0],
                                   alpha=0.4, zorder=2)
                    ax.plot(x_range, kde(x_range), color=CAT_COLORS[0], linewidth=1.5, zorder=3)
                ax.set_xticks([])
                ax.set_yticks([])
            elif i > j:
                # 下三角: 散点
                norm_h = (hardness - hardness.min()) / (hardness.max() - hardness.min() + 1e-9)
                colors = seq_cmap(norm_h)
                ax.scatter(df_plot.iloc[:, j], df_plot.iloc[:, i],
                           c=colors, s=15, alpha=0.6, edgecolors='none', zorder=2)
            else:
                # 上三角: 相关系数 (以0为中心对称: 负=蓝, 0=白, 正=红)
                corr = df_plot.iloc[:, i].corr(df_plot.iloc[:, j])
                corr_norm = (corr + 1) / 2  # [-1,1] → [0,1], 0.5对应r=0
                bg_cmap = LinearSegmentedColormap.from_list(
                    "corr_bg", ["#728BDE", "#FFFFFF", "#E97A6F"], N=256)
                ax.set_facecolor(bg_cmap(corr_norm))
                text_color = 'white' if abs(corr) > 0.5 else 'black'
                ax.text(0.5, 0.5, f"{corr:.2f}", transform=ax.transAxes,
                        ha='center', va='center', fontsize=14,
                        fontweight='bold', color=text_color)
                ax.set_xticks([])
                ax.set_yticks([])

            # 边框
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('black')

            # 标签
            if i == len(cols) - 1:
                ax.set_xlabel(abbr[j], fontsize=FONT_SIZE_TICK, fontweight='bold')
                ax.tick_params(axis='x', labelsize=FONT_SIZE_TICK - 2)
            if j == 0:
                ax.set_ylabel(abbr[i], fontsize=FONT_SIZE_TICK, fontweight='bold')
                ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK - 2)
            if i < len(cols) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    save(fig, "17_scatter_matrix", OUT_DIR)
    plt.close(fig)
    print("  [17] 散点矩阵图 — 关键变量两两关系 ✓")


# ============================================================
# 图18: 山脊图 — 功率区间硬度分布密度
# ============================================================
def plot_ridge_plot(df_clean, df_feat):
    from scipy.stats import gaussian_kde

    df = df_feat.copy()
    # 分6个功率区间
    p_min, p_max = df["激光功率"].min(), df["激光功率"].max()
    bins = np.linspace(p_min, p_max, 7)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}W" for i in range(len(bins)-1)]
    df["P_group"] = pd.cut(df["激光功率"], bins=bins, labels=labels, include_lowest=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

    ridge_colors = [CAT_COLORS[0], CAT_COLORS[1], CAT_COLORS[3],
                   CAT_COLORS[5], CAT_COLORS[6], CAT_COLORS[7]]

    # 从上到下排列
    y_offset = 0
    y_step = 1.0
    x_all = np.linspace(df["硬度"].min() - 50, df["硬度"].max() + 50, 300)

    for i, (grp, color) in enumerate(zip(labels, ridge_colors)):
        group_data = df[df["P_group"] == grp]["硬度"].dropna().values
        if len(group_data) < 3:
            continue
        if group_data.std() == 0:
            continue

        kde = gaussian_kde(group_data, bw_method=0.4)
        y_kde = kde(x_all)
        y_kde = y_kde / y_kde.max() * 0.85  # 归一化高度

        y_base = -i * y_step
        # 填充
        ax.fill_between(x_all, y_base, y_base + y_kde,
                        color=color, alpha=0.5, zorder=2)
        ax.plot(x_all, y_base + y_kde, color=color, linewidth=2, zorder=3)
        # 基线
        ax.axhline(y=y_base, color='gray', linewidth=0.5, alpha=0.3, zorder=1)
        # 组标签
        ax.text(df["硬度"].min() - 80, y_base + 0.15, grp,
                fontsize=FONT_SIZE_TICK, fontweight='bold', va='center', color=color)

    ax.set_xlabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Power Range", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(df["硬度"].min() - 200, df["硬度"].max() + 50)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "18_ridge_hardness_power", OUT_DIR)
    plt.close(fig)
    print("  [18] 山脊图 — 功率区间硬度分布密度 ✓")


# ============================================================
# 图19: Andrews曲线 — 多维样本曲线投影
# ============================================================
def plot_andrews_curve(df_clean, df_feat):
    from sklearn.preprocessing import MinMaxScaler

    df = df_feat.copy()
    features = ["激光功率", "扫描速度", "送粉速率", "Cr",
                "线能量密度", "碳当量", "镍当量", "硬度", "腐蚀电流"]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])

    # Andrews function: f(t) = x1/sqrt(2) + x2*sin(t) + x3*cos(t) + x4*sin(2t) + x5*cos(2t) + ...
    t = np.linspace(-np.pi, np.pi, 200)

    n_samples = X.shape[0]
    curves = np.zeros((n_samples, len(t)))

    for i in range(n_samples):
        x = X[i]
        curve = np.zeros_like(t)
        curve += x[0] / np.sqrt(2)
        k = 1
        j = 1
        while j < len(x) - 1:
            curve += x[j] * np.sin(k * t) + x[j+1] * np.cos(k * t)
            k += 1
            j += 2
        if j < len(x):
            curve += x[j] * np.sin(k * t)  # 奇数个特征, 最后一项
        curves[i] = curve

    hardness = df["硬度"].values
    # 顺序色标: 浅棕→珊瑚→品红(起点足够深)
    seq_cmap = LinearSegmentedColormap.from_list(
        "seq_palette", ["#E0BEB3", "#E97A6F", "#E8156E"], N=256)
    norm_h = (hardness - hardness.min()) / (hardness.max() - hardness.min() + 1e-9)

    # 按硬度排序
    sort_idx = np.argsort(hardness)
    curves_sorted = curves[sort_idx]
    norm_h_sorted = norm_h[sort_idx]

    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

    for i in range(n_samples):
        color = seq_cmap(norm_h_sorted[i])
        ax.plot(t, curves_sorted[i], color=color, alpha=0.3, linewidth=0.7, zorder=2)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
                       fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("t", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Andrews Curve Value", fontsize=FONT_SIZE_LABEL, fontweight='bold')

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=seq_cmap,
                               norm=plt.Normalize(vmin=hardness.min(), vmax=hardness.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Hardness (HV)", fontweight='bold', fontsize=FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "19_andrews_curve", OUT_DIR)
    plt.close(fig)
    print("  [19] Andrews曲线 — 多维样本曲线投影 ✓")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("新增可视化图谱 — 6张新视角图")
    print("=" * 60)

    df_clean, df_feat = _load_data()
    print(f"\n[数据] 样本数: {len(df_clean)}\n")

    print("[绘制图表]")
    plot_radar_chart(df_clean, df_feat)
    plot_parallel_coordinates(df_clean, df_feat)
    plot_contour_heatmap(df_clean, df_feat)
    plot_scatter_matrix(df_clean, df_feat)
    plot_ridge_plot(df_clean, df_feat)
    plot_andrews_curve(df_clean, df_feat)

    print(f"\n{'=' * 60}")
    print(f"全部完成！共 6 张新图，保存于 {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
