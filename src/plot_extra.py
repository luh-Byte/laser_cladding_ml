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
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plot_style import (
    create_gradient_rect, style_ax, save,
    GRADIENT_TOP, GRADIENT_BOTTOM,
    PALETTE, CAT_COLORS, DARK, GRAY, GRID_COLOR,
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_HEATMAP,
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    FONT_SIZE_CBAR, FONT_SIZE_ANNOT, FONT_SIZE_TITLE,
    MARGIN_10x10,
)
from src.data import load_raw_data, clean_data
from src.features import compute_derived_features

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
    fig.subplots_adjust(**MARGIN_10x10)

    group_colors = [CAT_COLORS[0], CAT_COLORS[3], CAT_COLORS[5]]
    group_labels = ["Cr<20%", "20-50%", "50-80%"]

    for i, (grp, color) in enumerate(zip(group_labels, group_colors)):
        vals = group_means.loc[grp].values.tolist()
        vals += vals[:1]
        ax.plot(np.array(angles), np.array(vals), '-', color=color, linewidth=2.5, label=grp, zorder=3)
        ax.fill(np.array(angles), np.array(vals), color=color, alpha=0.15, zorder=2)
        # 端点标记
        ax.scatter(np.array(angles[:-1]), np.array(vals[:-1]), color=color, s=50, zorder=4,
                   edgecolors='black', linewidths=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_abbr, fontsize=FONT_SIZE_TICK, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=FONT_SIZE_TICK - 2)
    ax.set_rlabel_position(90)  # type: ignore[attr-defined]

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1),
              fontsize=FONT_SIZE_LEGEND, frameon=False)

    ax.set_title("Normalized Parameter Profile by Cr Group",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)

    save(fig, "02c_radar_cr_group", OUT_DIR)
    plt.close(fig)
    print("  [14] 雷达图 — Cr分组归一化参数画像 ✓")


# ============================================================
# 图15: 标准化特征热力图 — 按硬度排序的全样本特征分布
# ============================================================
def plot_parallel_coordinates(df_clean, df_feat):
    from sklearn.preprocessing import MinMaxScaler

    df = df_feat.copy()
    features = ["激光功率", "扫描速度", "送粉速率", "Cr",
                "线能量密度", "碳当量", "镍当量", "硬度", "腐蚀电流"]
    abbr = [FEATURE_CN_TO_ABBR[f] for f in features]

    scaler = MinMaxScaler()
    norm_vals = scaler.fit_transform(df[features])
    sort_idx = np.argsort(df["硬度"].values)
    norm_sorted = norm_vals[sort_idx]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    heatmap_cmap = LinearSegmentedColormap.from_list(
        "feature_heatmap", ["#E5F2FB", "#E0BEB3", "#E97A6F", "#E8156E"], N=256)
    im = ax.imshow(norm_sorted, aspect="auto", cmap=heatmap_cmap,
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(abbr, fontsize=FONT_SIZE_TICK, fontweight='bold',
                       rotation=30, ha='right')
    ax.set_xlabel("Features", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Samples (sorted by hardness)",
                  fontsize=FONT_SIZE_LABEL, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Normalized feature value (0-1)",
                   fontweight='bold', fontsize=FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "01d_parallel_coordinates", OUT_DIR)
    plt.close(fig)
    print("  [15] 标准化特征热力图 — 按硬度排序 ✓")


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

    cbar = fig.colorbar(cf, ax=ax, shrink=1.0, pad=0.02)
    cbar.set_label("Hardness (HV)", fontweight='bold', fontsize=FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    ax.set_xlabel("Laser Power (W)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Scan Speed (mm/s)", fontsize=FONT_SIZE_LABEL, fontweight='bold')

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "02d_contour_lp_ss", OUT_DIR)
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
    # 腐蚀电流标注单位
    abbr[cols.index("腐蚀电流")] = "IC (µA/cm²)"

    df_plot = df[cols].copy()
    # 腐蚀电流转为 µA/cm², 消除科学计数法 (在列重命名前)
    df_plot["腐蚀电流"] = df_plot["腐蚀电流"] * 1e6
    df_plot.columns = abbr
    hardness = df["硬度"].values

    fig, axes = plt.subplots(len(cols), len(cols), figsize=(10, 10))
    fig.subplots_adjust(**MARGIN_10x10, hspace=0.12, wspace=0.12)

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
                spine.set_linewidth(FRAME_WIDTH)
                spine.set_color('black')

            # 刻度数字: 底部行显示 x (斜体), 左侧列显示 y, 其余隐藏
            if i < len(cols) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == len(cols) - 1:
                ax.tick_params(axis='x', labelsize=FONT_SIZE_TICK, rotation=30)
            else:
                ax.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
            ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)

    # 统一放置 x 轴标签 (底部一行, 不旋转)
    for j in range(len(cols)):
        ax_bottom = axes[len(cols) - 1, j]
        bbox = ax_bottom.get_position()
        fig.text(bbox.x0 + bbox.width / 2, bbox.y0 - 0.04,
                 abbr[j], ha='center', va='top',
                 fontsize=FONT_SIZE_TICK, fontweight='bold')

    # 统一放置 y 轴标签 (左侧一列)
    for i in range(len(cols)):
        ax_left = axes[i, 0]
        bbox = ax_left.get_position()
        fig.text(bbox.x0 - 0.04, bbox.y0 + bbox.height / 2,
                 abbr[i], ha='right', va='center',
                 fontsize=FONT_SIZE_TICK, fontweight='bold',
                 rotation=90)

    save(fig, "01e_scatter_matrix", OUT_DIR)
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

    fig, ax = plt.subplots(figsize=FIG_SIZE)

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

    save(fig, "02e_ridge_hardness_power", OUT_DIR)
    plt.close(fig)
    print("  [18] 山脊图 — 功率区间硬度分布密度 ✓")


# ============================================================
# 图19: 箱线图 — 不同功率区间的硬度分布
# ============================================================
def plot_andrews_curve(df_clean, df_feat):
    df = df_feat.copy()
    p_min, p_max = df["激光功率"].min(), df["激光功率"].max()
    bins = np.linspace(p_min, p_max, 7)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}W" for i in range(len(bins)-1)]
    df["P_group"] = pd.cut(df["激光功率"], bins=bins, labels=labels,
                            include_lowest=True)
    groups = [df.loc[df["P_group"] == label, "硬度"].dropna().values
              for label in labels]

    valid_groups = [(label, values) for label, values in zip(labels, groups)
                    if len(values) > 0]
    group_labels, group_values = zip(*valid_groups)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bp = ax.boxplot(group_values, patch_artist=True, widths=0.55,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.45))

    box_colors = [CAT_COLORS[i % len(CAT_COLORS)]
                  for i in range(len(group_values))]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for line in bp["medians"]:
        line.set_color(DARK)
        line.set_linewidth(2.0)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#666666")
            line.set_linewidth(1.2)

    ax.set_xticks(range(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels, rotation=25, ha="right",
                       fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("Laser Power Range", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel("Hardness (HV)", fontsize=FONT_SIZE_LABEL, fontweight='bold')
    ax.set_title("Hardness Distribution by Laser Power Range",
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)

    create_gradient_rect(ax)
    style_ax(ax, grid=False, right_top_ticks=False)

    save(fig, "02f_andrews_curve", OUT_DIR)
    plt.close(fig)
    print("  [19] 箱线图 — 不同功率区间的硬度分布 ✓")


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
