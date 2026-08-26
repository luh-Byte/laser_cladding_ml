"""
数据预处理模块
==============
1. 加载Excel原始数据
2. 修正脏数据（光斑直径 '1.4.76' -> 1.476）
3. 按物理准则清洗异常样本
4. 双目标联合分位数分层抽样（硬度×腐蚀电流），10次重复
5. 数据隔离：测试集不参与标准化拟合、特征筛选、参数调优
"""

import pandas as pd
import numpy as np
from src.config import (
    DATA_PATH, EXCEL_HEADER_ROW, EXCEL_SKIP_ROWS, SPOT_DIAMETER_FIX,
    CLEANING_RULES, PROCESS_FEATURES, COMPOSITION_FEATURES,
    TEST_SIZE, N_REPETITIONS, BASE_RANDOM_SEED
)


def load_raw_data(filepath=None):
    """加载Excel原始数据，处理表头与空行"""
    path = filepath or DATA_PATH
    df = pd.read_excel(path, header=EXCEL_HEADER_ROW, skiprows=EXCEL_SKIP_ROWS)

    # 重命名Unnamed:12列为"其他元素含量"（后续会丢弃）
    if "Unnamed: 12" in df.columns:
        df = df.rename(columns={"Unnamed: 12": "其他元素含量"})

    # 光斑直径转数值（处理脏数据）
    df["光斑直径"] = df["光斑直径"].astype(str).str.strip()
    for bad_val, good_val in SPOT_DIAMETER_FIX.items():
        df["光斑直径"] = df["光斑直径"].replace(bad_val, str(good_val))
    df["光斑直径"] = pd.to_numeric(df["光斑直径"], errors="coerce")

    # 确保所有列都是数值型
    numeric_cols = PROCESS_FEATURES + COMPOSITION_FEATURES + [
        "其他元素含量", "硬度", "腐蚀电流", "腐蚀电流*10000", "熔宽", "熔高"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[数据加载] 原始数据: {df.shape[0]}行 x {df.shape[1]}列")
    return df


def clean_data(df):
    """按物理准则清洗异常样本，返回清洗后的数据和清洗日志"""
    original_count = len(df)
    cleaning_log = []
    df_clean = df.copy()

    # 规则1: 光斑直径 > 8mm
    mask = df_clean["光斑直径"] > CLEANING_RULES["spot_diameter_max"]
    n = mask.sum()
    cleaning_log.append(f"光斑直径>{CLEANING_RULES['spot_diameter_max']}mm: 剔除{n}行")
    df_clean = df_clean[~mask]

    # 规则2: 扫描速度 > 200mm/s
    mask = df_clean["扫描速度"] > CLEANING_RULES["scan_speed_max"]
    n = mask.sum()
    cleaning_log.append(f"扫描速度>{CLEANING_RULES['scan_speed_max']}mm/s: 剔除{n}行")
    df_clean = df_clean[~mask]

    # 规则3: 元素总含量 (7元素之和) 在 95~105% 区间外
    elem_sum = df_clean[COMPOSITION_FEATURES].sum(axis=1)
    mask = (elem_sum < CLEANING_RULES["elem_sum_min"]) | \
           (elem_sum > CLEANING_RULES["elem_sum_max"])
    n = mask.sum()
    cleaning_log.append(
        f"元素总含量不在{CLEANING_RULES['elem_sum_min']}~{CLEANING_RULES['elem_sum_max']}%: 剔除{n}行"
    )
    df_clean = df_clean[~mask]

    # 规则4: 元素含量为负
    if CLEANING_RULES["negative_element_remove"]:
        mask = (df_clean[COMPOSITION_FEATURES] < 0).any(axis=1)
        n = mask.sum()
        cleaning_log.append(f"元素含量为负: 剔除{n}行")
        df_clean = df_clean[~mask]

    # 规则5: 硬度异常
    mask = (df_clean["硬度"] < CLEANING_RULES["hardness_min"]) | \
           (df_clean["硬度"] > CLEANING_RULES["hardness_max"])
    n = mask.sum()
    cleaning_log.append(
        f"硬度<{CLEANING_RULES['hardness_min']}HV或>{CLEANING_RULES['hardness_max']}HV: 剔除{n}行"
    )
    df_clean = df_clean[~mask]

    # 规则6: 腐蚀电流异常（过大）
    mask = df_clean["腐蚀电流"] > CLEANING_RULES["corrosion_current_max"]
    n = mask.sum()
    cleaning_log.append(
        f"腐蚀电流>{CLEANING_RULES['corrosion_current_max']}A/cm2: 剔除{n}行"
    )
    df_clean = df_clean[~mask]

    # 规则7: 腐蚀电流极小值（测量伪影）
    mask = df_clean["腐蚀电流"] <= CLEANING_RULES["corrosion_current_min"]
    n = mask.sum()
    cleaning_log.append(
        f"腐蚀电流<={CLEANING_RULES['corrosion_current_min']}A/cm2(伪影): 剔除{n}行"
    )
    df_clean = df_clean[~mask]

    # 规则8: 熔高/熔宽比值异常（超出合理范围）
    if "熔宽" in df_clean.columns and "熔高" in df_clean.columns:
        ratio = df_clean["熔高"] / df_clean["熔宽"]
        mask = (ratio < 0.05) | (ratio > 1.0)
        n = mask.sum()
        cleaning_log.append(f"熔高/熔宽比值异常(<0.05或>1.0): 剔除{n}行")
        df_clean = df_clean[~mask]

    # 规则9: 离群点剔除 — 高功率区域孤立点
    #   a) 功率>=4800W且硬度异常高(>1500HV)
    #   b) 功率>=5500W的孤立点(与其他数据间断过大)
    mask = ((df_clean["激光功率"] >= 4800) & (df_clean["硬度"] > 1500)) | \
           (df_clean["激光功率"] >= 5500)
    n = mask.sum()
    cleaning_log.append(f"离群点(高功率区域孤立点): 剔除{n}行")
    df_clean = df_clean[~mask]

    # 删除含NaN的行
    before_nan = len(df_clean)
    df_clean = df_clean.dropna(
        subset=PROCESS_FEATURES + COMPOSITION_FEATURES + ["硬度", "腐蚀电流"]
    )
    nan_removed = before_nan - len(df_clean)
    if nan_removed > 0:
        cleaning_log.append(f"含NaN值: 剔除{nan_removed}行")

    # 重置索引
    df_clean = df_clean.reset_index(drop=True)

    removed = original_count - len(df_clean)
    print(f"\n[数据清洗] 原始{original_count}行 -> 清洗后{len(df_clean)}行 (剔除{removed}行, {removed/original_count*100:.1f}%)")
    for log in cleaning_log:
        print(f"  - {log}")

    return df_clean, cleaning_log


def generate_split_indices(y_hardness=None, y_corrosion_log=None,
                           n_reps=N_REPETITIONS, test_size=TEST_SIZE,
                           base_seed=BASE_RANDOM_SEED, n_bins=5):
    """
    生成重复划分的索引列表（双目标联合分位数分层抽样）。

    策略：
    1. 按硬度分位数分 n_bins 箱
    2. 按腐蚀电流分位数分 n_bins 箱
    3. 组合两箱标签形成联合分层标签
    4. 使用 StratifiedShuffleSplit 保证每层比例一致
    5. 若某组样本过少导致分层失败，自动降级为单目标（硬度）分层

    参数:
        y_hardness: 硬度目标值数组
        y_corrosion_log: 腐蚀电流log值数组
        n_reps: 重复次数
        test_size: 测试集比例
        base_seed: 基础随机种子
        n_bins: 分位数分箱数

    返回: [(train_idx, test_idx, seed, strat_method), ...]
    """
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    n_samples = len(y_hardness)
    all_indices = np.arange(n_samples)
    splits = []

    # ---- 构建联合分层标签 ----
    strat_labels = None
    strat_method = "random"

    if y_hardness is not None and y_corrosion_log is not None:
        # 硬度分位数分箱
        h_bins = pd.qcut(y_hardness, q=n_bins, labels=False, duplicates="drop")
        # 腐蚀电流分位数分箱
        c_bins = pd.qcut(y_corrosion_log, q=n_bins, labels=False, duplicates="drop")

        # 联合标签
        joint_labels = h_bins.astype(str) + "_" + c_bins.astype(str)
        unique_labels, counts = np.unique(joint_labels, return_counts=True)

        # 检查最小层样本数是否足够（每层至少2个样本才能分训练+测试）
        min_count = counts.min()
        n_layers = len(unique_labels)

        if min_count >= 2:
            strat_labels = joint_labels
            strat_method = f"joint_q{n_bins}"
            print(f"  [分层] 双目标联合分层: {n_layers}层, 最小层{min_count}个样本")
        else:
            # 降级为单目标（硬度）分层
            h_unique, h_counts = np.unique(h_bins, return_counts=True)
            if h_counts.min() >= 2:
                strat_labels = h_bins.astype(str)
                strat_method = f"hardness_q{n_bins}"
                print(f"  [分层] 联合分层失败（最小层{min_count}个样本），降级为硬度分层: "
                      f"{len(h_unique)}层, 最小层{h_counts.min()}个样本")
            else:
                strat_method = "random"
                print(f"  [分层] 硬度分层也失败（最小层{h_counts.min()}个样本），使用随机划分")

    # ---- 生成划分 ----
    if strat_labels is not None:
        sss = StratifiedShuffleSplit(
            n_splits=n_reps, test_size=test_size, random_state=base_seed
        )
        for i, (train_idx, test_idx) in enumerate(sss.split(all_indices, strat_labels)):
            seed = base_seed + i
            splits.append((train_idx, test_idx, seed, strat_method))
    else:
        # 纯随机划分（降级方案）
        for i in range(n_reps):
            seed = base_seed + i
            train_idx, test_idx = train_test_split(
                all_indices, test_size=test_size, random_state=seed, shuffle=True
            )
            splits.append((train_idx, test_idx, seed, "random"))

    return splits


def prepare_targets(df):
    """
    准备目标变量：
    - 硬度: 原始值直接使用
    - 腐蚀电流: 直接使用Excel中的"腐蚀电流*10000"列，再取log10
    """
    targets = {}
    targets["hardness"] = df["硬度"].values.copy()

    # 腐蚀电流: 直接使用Excel的"腐蚀电流*10000"列，再取log10
    icorr_10k = df["腐蚀电流*10000"].values.astype(float)
    # 确保正值后再log
    icorr_10k = np.maximum(icorr_10k, 1e-15)
    targets["corrosion_log"] = np.log10(icorr_10k)

    return targets
