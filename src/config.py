"""
激光熔覆涂层性能预测 - 全局配置
====================================
集中管理所有路径、随机种子、清洗阈值、特征定义、模型超参搜索空间。
所有模块从此文件读取配置，确保实验可复现、参数可迭代。
"""

import os

# ============================================================
# 1. 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_data.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

# ============================================================
# 2. 随机种子与实验设置
# ============================================================
BASE_RANDOM_SEED = 42
N_REPETITIONS = 10          # 10次重复划分实验
TEST_SIZE = 0.20             # 测试集比例 20%
CV_FOLDS = 5                 # 5折交叉验证（训练集专属）

# ============================================================
# 3. 数据清洗阈值（物理准则版）
# ============================================================
CLEANING_RULES = {
    "spot_diameter_max": 8.0,          # 光斑直径 > 8mm 剔除
    "scan_speed_max": 200.0,           # 扫描速度 > 200mm/s 剔除
    "elem_sum_min": 85.0,              # 7元素总和下限 85%（迭代优化：放宽以容纳含未列出元素的合法样本）
    "elem_sum_max": 115.0,             # 7元素总和上限 115%
    "hardness_min": 150.0,            # 硬度 < 150HV 剔除
    "hardness_max": 1800.0,           # 硬度 > 1800HV 剔除
    "corrosion_current_max": 1e-4,     # 腐蚀电流 > 1e-4 A/cm2 剔除
    "corrosion_current_min": 1e-9,     # 腐蚀电流 <= 1e-9 视为伪影剔除
    "negative_element_remove": True,  # 元素含量为负剔除
}

# ============================================================
# 4. 特征定义
# ============================================================

# 4.1 原始工艺特征（5项，已剔除"搭接率"——数据中无此列）
PROCESS_FEATURES = [
    "激光功率", "扫描速度", "送粉速率", "光斑直径", "离焦量"
]

# 4.2 成分特征（7项）
COMPOSITION_FEATURES = ["C", "Cr", "Si", "Ni", "Fe", "Mn", "Mo"]

# 4.3 物理衍生特征
DERIVED_FEATURE_FORMULAS = {
    "线能量密度":    "激光功率 / 扫描速度",                               # J/mm
    "面能量密度":    "激光功率 / (扫描速度 * 光斑直径)",                   # J/mm^2
    "粉末能量比":    "激光功率 / 送粉速率",                               # W*min/g
    "碳当量":        "C + Mn/6 + (Cr+Mo)/5 + Ni/15",                     # IIW标准
    "铬当量":        "Cr + 1.5*Si + 1.5*Mo",                             # Schaeffler
    "镍当量":        "Ni + 30*C + 0.5*Mn",                               # Schaeffler
}

# 4.4 共线性筛选阈值
CORRELATION_THRESHOLD = 0.90  # |r| > 0.9 剔除冗余特征

# 4.5 特征精简列表（基于RF双目标综合重要性排名，剔除排名最低的3个）
# 依据: 硬度RF重要性排名 + 腐蚀RF重要性排名，排名和最高的3个：Si(25)、Mo(23)、Ni(23)
FEATURES_TO_REMOVE = ["Si", "Mo", "Ni"]

# ============================================================
# 5. 模型超参搜索空间
# ============================================================

MODEL_ORDER = ["KNN", "SVR", "RF", "LightGBM"]  # 固定执行顺序

HYPERPARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["distance"],
        "metric": ["euclidean"],
    },
    "SVR": {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": [0.01, 0.1, 1, "scale"],
        "epsilon": [0.01, 0.05, 0.1],
    },
    "RF": {
        "n_estimators": [100, 150, 200],
        "max_depth": [6, 8, 10, None],
        "min_samples_leaf": [2, 3, 5],
    },
    "LightGBM": {
        "learning_rate": [0.05, 0.08, 0.1],
        "max_depth": [4, 5, 6, 7],
        "n_estimators": [200, 300, 500],
        "reg_lambda": [0.1, 1, 10],
    },
}

# ============================================================
# 6. 数据加载参数
# ============================================================
# Excel表头在第2行（index=1），跳过第3-5行（单位行+空行）
EXCEL_HEADER_ROW = 1
EXCEL_SKIP_ROWS = [2, 3, 4]  # 跳过单位行和空行

# 光斑直径脏数据修正
SPOT_DIAMETER_FIX = {"1.4.76": 1.476}
