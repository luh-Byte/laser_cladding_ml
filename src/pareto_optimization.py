"""
帕累托优化模块
===============
双目标：最大化硬度(HV) + 最小化腐蚀电流密度(A/cm2)

方法：逐维扫描式帕累托优化
- 对5个工艺参数逐个做"固定该参数→搜索其余4个"的扫描
- 每次扫描生成该参数维度下的帕累托前沿
- 汇总所有扫描，得到整体最优工艺参数空间

成分特征：取数据集中位数（典型成分），衍生特征自动计算
搜索方式：拉丁超立方采样(LHS) + 帕累托非支配筛选
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import (
    PROCESS_FEATURES, COMPOSITION_FEATURES,
    MODEL_DIR, RESULT_DIR
)
from src.feature_engineering import compute_derived_features, get_all_feature_names
from src.evaluation import load_model_bundle


# ============================================================
# 搜索空间定义（基于数据范围，合理外推±10%）
# ============================================================
SEARCH_SPACE = {
    "激光功率":   (200, 6500),    # W
    "扫描速度":   (1.0, 210),      # mm/s
    "送粉速率":   (0.3, 320),      # g/min
    "光斑直径":   (0.8, 8.0),      # mm
    "离焦量":     (-2.5, 2.5),     # mm
}

# 每个固定维度的离散点数
FIXED_DIM_POINTS = 15
# 每个固定点对应的自由参数采样数
SAMPLES_PER_FIXED_POINT = 5000


def get_typical_composition(df_clean):
    """获取典型成分（各元素中位数，归一化到100%）"""
    comp = df_clean[COMPOSITION_FEATURES].median().to_dict()
    total = sum(comp.values())
    for k in comp:
        comp[k] = comp[k] / total * 100.0
    return comp


def latin_hypercube_sampling(bounds, n_samples, seed=42):
    """
    拉丁超立方采样，在多维空间中均匀采样。
    bounds: dict {param_name: (low, high)}
    返回: DataFrame，每行为一个样本
    """
    rng = np.random.default_rng(seed)
    params = list(bounds.keys())
    n_dims = len(params)

    samples = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        perm = rng.permutation(n_samples)
        u = rng.random(n_samples)
        samples[:, i] = (perm + u) / n_samples

    for i, param in enumerate(params):
        low, high = bounds[param]
        samples[:, i] = low + samples[:, i] * (high - low)

    return pd.DataFrame(samples, columns=params)


def build_feature_matrix(process_params, composition, bundle):
    """
    根据工艺参数和成分构建完整特征矩阵，使用bundle中的scaler和selected_features。
    返回: X_scaled (n_samples, n_selected_features)
    """
    n = len(process_params)
    scaler = bundle['scaler']
    selected_features = bundle['selected_features']
    all_feature_names = bundle['all_feature_names']

    # 构建完整DataFrame（工艺 + 成分）
    data = {}
    for col in PROCESS_FEATURES:
        data[col] = process_params[col].values
    for col in COMPOSITION_FEATURES:
        data[col] = np.full(n, composition[col])

    df = pd.DataFrame(data)

    # 计算衍生特征
    df = compute_derived_features(df)

    # 按训练时的顺序排列全部特征
    df_full = df[all_feature_names].copy()

    # 用训练时的scaler做标准化
    X_scaled = scaler.transform(df_full.values)

    # 提取选中的特征列
    feat_idx = [all_feature_names.index(f) for f in selected_features]
    X_result = X_scaled[:, feat_idx]

    return X_result


def predict_hardness(X_scaled, bundle):
    """预测硬度（直接输出HV）"""
    return bundle['model'].predict(X_scaled)


def predict_corrosion(X_scaled, bundle):
    """预测腐蚀电流（Z-score → log空间 → 原始量纲 A/cm2）"""
    corr_zscore = bundle['model'].predict(X_scaled)
    transformer = bundle['corrosion_transformer']
    corr_log = corr_zscore * transformer['std'] + transformer['mean']
    corr_orig = np.power(10.0, corr_log) / 10000.0
    return corr_orig


def pareto_front(points, objectives):
    """
    计算2目标帕累托前沿（非支配解）。
    使用O(n log n)排序+扫描算法（仅支持2个目标）。
    objectives: list of (column_name, direction)，direction='max'或'min'
    """
    assert len(objectives) == 2, "仅支持2目标帕累托前沿计算"
    obj1_col, obj1_dir = objectives[0]
    obj2_col, obj2_dir = objectives[1]

    df = points.copy()

    # 将两个目标都转换为"越大越好"的方向
    if obj1_dir == 'min':
        df['_obj1'] = -df[obj1_col].values
    else:
        df['_obj1'] = df[obj1_col].values
    if obj2_dir == 'min':
        df['_obj2'] = -df[obj2_col].values
    else:
        df['_obj2'] = df[obj2_col].values

    # 按第一目标降序排序
    df = df.sort_values(by='_obj1', ascending=False).reset_index(drop=True)

    # 扫描：维护第二目标的最优值，遇到更优的就是帕累托点
    best_obj2 = -np.inf
    is_pareto = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if df['_obj2'].iloc[i] > best_obj2:
            is_pareto[i] = True
            best_obj2 = df['_obj2'].iloc[i]

    # 处理第一目标相同的点：只保留第二目标最好的
    # （排序后相邻的第一目标相同的点，只有第一个会被标记）

    pareto = df[is_pareto].copy()
    pareto = pareto.drop(columns=['_obj1', '_obj2'])
    pareto = pareto.reset_index(drop=True)

    return pareto


def scan_single_dimension(fixed_param, hardness_bundle, corrosion_bundle,
                          composition, n_fixed=FIXED_DIM_POINTS,
                          n_samples=SAMPLES_PER_FIXED_POINT, base_seed=42):
    """
    固定一个工艺参数维度，扫描其余4个参数。
    返回: (该维度帕累托前沿, 全部采样点)
    """
    free_params = [p for p in PROCESS_FEATURES if p != fixed_param]
    fixed_low, fixed_high = SEARCH_SPACE[fixed_param]
    fixed_values = np.linspace(fixed_low, fixed_high, n_fixed)

    all_results = []

    for idx, fixed_val in enumerate(fixed_values):
        free_bounds = {p: SEARCH_SPACE[p] for p in free_params}
        samples = latin_hypercube_sampling(free_bounds, n_samples, seed=base_seed + idx)
        samples[fixed_param] = fixed_val

        X = build_feature_matrix(samples[PROCESS_FEATURES], composition, hardness_bundle)

        hardness_pred = predict_hardness(X, hardness_bundle)
        corr_pred = predict_corrosion(X, corrosion_bundle)

        result_df = samples[PROCESS_FEATURES].copy()
        result_df["硬度"] = hardness_pred
        result_df["腐蚀电流"] = corr_pred
        result_df["固定参数"] = fixed_param
        result_df["固定参数值"] = fixed_val
        all_results.append(result_df)

    all_df = pd.concat(all_results, ignore_index=True)

    objectives = [("硬度", "max"), ("腐蚀电流", "min")]
    pareto = pareto_front(all_df, objectives)

    print(f"  [扫描] {fixed_param}: {len(all_df)}个采样 → 帕累托前沿{len(pareto)}个解")
    return pareto, all_df


def run_pareto_optimization(df_clean, model_dir=None, output_dir=None):
    """
    执行完整帕累托优化：
    1. 加载模型bundle
    2. 获取典型成分
    3. 逐个扫描5个工艺参数
    4. 汇总并计算整体帕累托前沿
    5. 提取代表性参数组合
    6. 输出结果
    """
    out_dir = output_dir or os.path.join(RESULT_DIR, "pareto")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("帕累托优化 - 双目标：最大化硬度 + 最小化腐蚀电流")
    print(f"扫描方式：逐维扫描（5个工艺参数 × {FIXED_DIM_POINTS}个固定点 "
          f"× {SAMPLES_PER_FIXED_POINT}个自由采样）")
    print("=" * 70)

    # 1. 加载模型
    hardness_bundle = load_model_bundle("LightGBM", "硬度", model_dir)
    corrosion_bundle = load_model_bundle("RF", "腐蚀电流", model_dir)
    print(f"\n[模型] 硬度: LightGBM ({type(hardness_bundle['model']).__name__})")
    print(f"[模型] 腐蚀: RF ({type(corrosion_bundle['model']).__name__})")
    print(f"[特征] {len(hardness_bundle['selected_features'])}项: "
          f"{hardness_bundle['selected_features']}")

    # 2. 典型成分
    composition = get_typical_composition(df_clean)
    comp_str = ", ".join(f"{k}={v:.1f}%" for k, v in composition.items())
    print(f"\n[典型成分] {comp_str}")

    # 3. 逐维扫描
    print(f"\n[步骤] 开始逐维扫描")
    all_pareto_fronts = {}
    all_samples_list = []

    for param_idx, param in enumerate(PROCESS_FEATURES):
        print(f"\n--- 扫描维度: {param} ---")
        pareto_df, samples_df = scan_single_dimension(
            param, hardness_bundle, corrosion_bundle, composition,
            base_seed=42 + param_idx * 10000
        )
        all_pareto_fronts[param] = pareto_df
        all_samples_list.append(samples_df)

        pareto_path = os.path.join(out_dir, f"pareto_{param}.csv")
        pareto_df.to_csv(pareto_path, index=False)

    # 4. 整体帕累托前沿
    print(f"\n[步骤] 计算整体帕累托前沿")
    all_samples_df = pd.concat(all_samples_list, ignore_index=True)
    all_samples_df = all_samples_df.drop_duplicates(
        subset=PROCESS_FEATURES
    ).reset_index(drop=True)

    objectives = [("硬度", "max"), ("腐蚀电流", "min")]
    global_pareto = pareto_front(all_samples_df, objectives)

    print(f"  总去重采样点数: {len(all_samples_df)}")
    print(f"  整体帕累托前沿解数: {len(global_pareto)}")

    # 5. 按硬度分层，提取代表性参数（每层取腐蚀电流最低的）
    print(f"\n[步骤] 提取代表性最优参数组合")
    h_min = global_pareto["硬度"].min()
    h_max = global_pareto["硬度"].max()
    n_intervals = 10
    interval = (h_max - h_min) / n_intervals

    representative = []
    for i in range(n_intervals):
        lo = h_min + i * interval
        hi = lo + interval
        mask = (global_pareto["硬度"] >= lo) & (global_pareto["硬度"] < hi)
        interval_df = global_pareto[mask]
        if len(interval_df) > 0:
            best = interval_df.loc[interval_df["腐蚀电流"].idxmin()]
            representative.append(best)

    rep_df = pd.DataFrame(representative).reset_index(drop=True)

    # 6. 保存结果
    global_path = os.path.join(out_dir, "pareto_global_front.csv")
    global_pareto.to_csv(global_path, index=False)
    print(f"\n[输出] 整体帕累托前沿: {global_path}")

    rep_path = os.path.join(out_dir, "pareto_representative_params.csv")
    rep_df.to_csv(rep_path, index=False)
    print(f"[输出] 代表性参数组合: {rep_path}")

    # 7. 打印摘要
    print(f"\n" + "=" * 70)
    print("帕累托优化结果摘要")
    print("=" * 70)
    print(f"总去重采样点: {len(all_samples_df)}")
    print(f"整体帕累托前沿: {len(global_pareto)}个解")
    print(f"硬度范围: {global_pareto['硬度'].min():.0f} ~ {global_pareto['硬度'].max():.0f} HV")
    print(f"腐蚀电流范围: {global_pareto['腐蚀电流'].min():.2e} ~ "
          f"{global_pareto['腐蚀电流'].max():.2e} A/cm2")
    print()
    print("代表性参数组合（按硬度分层，取腐蚀最低者）:")
    print("-" * 100)
    header = f"{'硬度(HV)':>9} {'腐蚀电流':>12} " + " ".join(f"{p:>9}" for p in PROCESS_FEATURES)
    print(header)
    print("-" * 100)
    for _, row in rep_df.iterrows():
        line = f"{row['硬度']:>9.0f} {row['腐蚀电流']:>12.2e} " + \
               " ".join(f"{row[p]:>9.2f}" for p in PROCESS_FEATURES)
        print(line)
    print("=" * 70)

    return global_pareto, all_pareto_fronts, rep_df


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_data, clean_data
    df = load_raw_data()
    df_clean, _ = clean_data(df)
    run_pareto_optimization(df_clean)
