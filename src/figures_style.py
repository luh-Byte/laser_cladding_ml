"""
论文级统一绘图样式模块
======================
- 字体: Times New Roman (Liberation Serif, 度量兼容替代)
- 全局加粗 (bold)
- 渐变背景: #F8F9E4 → #E5F2FB
- 加粗黑色图框 (2.0pt)
- 内指向刻度线
- 统一画布: 8×6 inch (常规) / 10×8 inch (热图)
- 300 DPI PNG 输出
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 字体设置 — Times New Roman (Liberation Serif 度量兼容)
# ============================================================
import platform as _platform

if _platform.system() == "Windows":
    TNR_FONT = "Times New Roman"
else:
    TNR_FONT = "Liberation Serif"
    _LIBERATION_FONTS = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
    ]
    for _fp in _LIBERATION_FONTS:
        if os.path.exists(_fp):
            fm.fontManager.addfont(_fp)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [TNR_FONT, "Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

FONT_SERIF = [TNR_FONT]

# ============================================================
# 色板
# ============================================================
GRADIENT_TOP = "#F8F9E4"
GRADIENT_BOTTOM = "#E5F2FB"

PALETTE = [
    "#E5F2FB", "#F8F9E4", "#E97A6F", "#E8156E", "#E0BEB3",
    "#00BEB3", "#728BDE", "#E246C9", "#834BD4", "#CFCFCF",
]
CAT_COLORS = PALETTE[2:]

TRAIN_COLOR = "#728BDE"
TEST_COLOR = "#E97A6F"

FRAME_WIDTH = 2.0
DARK = "#212121"
GRAY = "#757575"
GRID_COLOR = "#E0E0E0"

# ============================================================
# 字号 — 统一适中, 避免重叠
# ============================================================
FONT_SIZE_LABEL = 15    #X/Y 轴标签
FONT_SIZE_TICK = 14     # 刻度标签
FONT_SIZE_LEGEND = 12   # 图例
FONT_SIZE_CBAR = 12     # 颜色条
FONT_SIZE_ANNOT = 10    # 注释
FONT_SIZE_TITLE = 13    # 标题

# 热图因行列多, 略小
FONT_SIZE_TICK_HEAT = 11     # 热图刻度
FONT_SIZE_ANNOT_HEAT = 8    #格子内数值
FONT_SIZE_LABEL_HEAT = 12   # 轴标签
FONT_SIZE_CBAR_HEAT = 10    # 颜色条

# SHAP图Y轴特征名多, 用小字号
FONT_SIZE_SHAP_TICK = 11

# 刻度标签
FONT_SIZE_SHAP_LABEL = 12       # 轴标签
FONT_SIZE_SHAP_LEGEND = 10      # 图例
FONT_SIZE_SHAP_ANNOT = 9        # 注释
FONT_SIZE_SHAP_CBAR = 10        # 颜色条

# ============================================================
# 画布尺寸 — 统一两种规格
# ============================================================
FIG_SIZE = (8, 6)           # 所有常规图统一 8×6
FIG_SIZE_WIDE = (14, 6)     # 双子图横向排列
FIG_SIZE_HEATMAP = (10, 8)  # 热图稍大

# ============================================================
# 全局 rcParams
# ============================================================
plt.rcParams.update({
    'font.size': FONT_SIZE_TICK,
    'font.weight': 'bold',

    'axes.labelsize': FONT_SIZE_LABEL,
    'axes.labelweight': 'bold',
    'axes.titlesize': FONT_SIZE_TITLE,
    'axes.titleweight': 'bold',
    'axes.linewidth': FRAME_WIDTH,

    'xtick.major.size': 5,
    'xtick.major.width': 1.5,
    'xtick.minor.size': 3,
    'xtick.minor.width': 1.0,
    'xtick.labelsize': FONT_SIZE_TICK,
    'xtick.direction': 'in',
    'xtick.top': True,

    'ytick.major.size': 5,
    'ytick.major.width': 1.5,
    'ytick.minor.size': 3,
    'ytick.minor.width': 1.0,
    'ytick.labelsize': FONT_SIZE_TICK,
    'ytick.direction': 'in',
    'ytick.right': True,

    'legend.fontsize': FONT_SIZE_LEGEND,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '#888888',
    'legend.borderpad': 0.6,
    'legend.handlelength': 1.8,

    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'savefig.facecolor': 'white',
})


# ============================================================
# 渐变背景
# ============================================================
def create_gradient_rect(ax, color_top=GRADIENT_TOP, color_bottom=GRADIENT_BOTTOM,
                         alpha=0.5, zorder=-2):
    """为坐标轴添加从上到下的渐变背景"""
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack([gradient] * 100)

    cmap = LinearSegmentedColormap.from_list(
        'paper_gradient', [color_top, color_bottom], N=256)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=alpha, zorder=zorder,
              extent=[0, 1, 0, 1], interpolation='bilinear', transform=ax.transAxes)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# ============================================================
# 统一坐标轴样式
# ============================================================
def style_ax(ax, grid=False, right_top_ticks=True, tick_labelsize=None):
    """
    统一设置坐标轴样式:
    - 黑色边框 (2.0pt)
    - 内指向刻度
    - 可选网格和四边刻度
    """
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_WIDTH)
        spine.set_color('black')

    ls = tick_labelsize if tick_labelsize else FONT_SIZE_TICK
    ax.tick_params(axis='both', which='major', labelsize=ls,
                   width=1.5, length=5, color='black')
    ax.tick_params(axis='both', which='minor', labelsize=ls,
                   width=1.0, length=3, color='black')

    # Ensure all tick labels are Times New Roman Bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(TNR_FONT)
        label.set_fontweight('bold')

    if right_top_ticks:
        ax.tick_params(axis='x', top=True)
        ax.tick_params(axis='y', right=True)
    else:
        ax.tick_params(axis='x', top=False)
        ax.tick_params(axis='y', right=False)

    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.4)


# ============================================================
# 保存图表
# ============================================================
def _make_all_text_bold(fig):
    """Force all text elements in figure to Times New Roman bold."""
    import matplotlib.text as mtext
    target_font = TNR_FONT  # "Liberation Serif" (Times New Roman compatible)

    # Find ALL Text objects in the figure (tick labels, annotations, labels, legend, etc.)
    for t in fig.findobj(mtext.Text):
        try:
            t.set_fontfamily(target_font)
            t.set_fontweight('bold')
        except Exception:
            pass


def save(fig, name, out_dir=None):
    """保存图表为PNG (300dpi), 强制全文字体加粗(Times New Roman Bold)"""
    if out_dir is None:
        from src.config import OUTPUT_DIR
        out_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)
    # Force a draw to populate all text objects
    fig.canvas.draw()
    _make_all_text_bold(fig)
    png_name = str(name).rsplit('.', 1)[0] + '.png'
    fig.savefig(f"{out_dir}/{png_name}", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    plt.close(fig)
    print(f"  {png_name}")
