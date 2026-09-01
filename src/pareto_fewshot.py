"""
Few-shot微调版 Rockit485 帕累托优化
=======================================
使用 Few-shot 微调后的模型（已纠正硬度趋势 + 大幅改善腐蚀预测）
生成 Rockit485 材料的专属帕累托最优工艺参数空间。
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import PROCESS_FEATURES, RESULT_DIR
from src.metrics import load_model_bundle
from src.pareto import (
    build_feature_matrix, predict_hardness,
    latin_hypercube_sampling, pareto_front,
    SEARCH_SPACE, FIXED_DIM_POINTS, SAMPLES_PER_FIXED_POINT
)
from src.fewshot import ROCKIT485_COMPOSITION, ROCKIT485_EXPERIMENTS


def predict_corrosion_calibrated(X_scaled, bundle):
    """使用校准因子预测腐蚀电流"""
    corr_zscore = bundle['model'].predict(X_scaled)
    transformer = bundle['corrosion_transformer']
    corr_log = corr_zscore * transformer['std'] + transformer['mean']
    corr_orig = np.power(10.0, corr_log) / 10000.0

    if 'calibration' in bundle and bundle['calibration'].get('applied', False):
        factor = bundle['calibration']['factor']
        corr_orig = corr_orig * factor

    return corr_orig


def run_fewshot_pareto():
    """Few-shot微调版帕累托优化"""
    print("=" * 80)
    print("Rockit485 Few-shot微调版 帕累托优化")
    print("=" * 80)
    print("模型: LightGBM_fewshot (硬度) + LightGBM_fewshot (腐蚀, 已校准)")
    print(f"成分: Rockit485 (C=0.15%, Cr=13%, Ni=4%, Mo=2.8%, Fe≈79%)")
    print(f"扫描方式：逐维扫描（5个工艺参数 × {FIXED_DIM_POINTS}个固定点 "
          f"× {SAMPLES_PER_FIXED_POINT}个自由采样）")

    # 加载 Few-shot 模型
    hardness_bundle = load_model_bundle("LightGBM_fewshot", "硬度")
    corrosion_bundle = load_model_bundle("LightGBM_fewshot", "腐蚀电流")

    cal_info = corrosion_bundle.get('calibration', {})
    print(f"\n[模型] 硬度: {hardness_bundle['model_type']} (fewshot weight={hardness_bundle.get('fewshot_weight', 'N/A')}x)")
    print(f"[模型] 腐蚀: {corrosion_bundle['model_type']} (fewshot weight={corrosion_bundle.get('fewshot_weight', 'N/A')}x)")
    print(f"[校准] 腐蚀校准因子: {cal_info.get('factor', 'N/A'):.4f}")

    # ---- 先验证4个实验点 ----
    print(f"\n{'='*60}")
    print("实验点验证（训练集内验证）")
    print("=" * 60)

    exp_df = pd.DataFrame(ROCKIT485_EXPERIMENTS)[PROCESS_FEATURES]
    X_exp = build_feature_matrix(exp_df, ROCKIT485_COMPOSITION, hardness_bundle)
    h_pred = predict_hardness(X_exp, hardness_bundle)
    c_pred = predict_corrosion_calibrated(X_exp, corrosion_bundle)

    print(f"\n硬度预测:")
    print(f"{'功率(W)':>8} {'实测(HV)':>10} {'预测(HV)':>10} {'误差%':>8}")
    print("-" * 45)
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        err = abs(h_pred[i] - exp['硬度']) / exp['硬度'] * 100
        print(f"{exp['激光功率']:>8d} {exp['硬度']:>10.1f} {h_pred[i]:>10.1f} {err:>7.1f}%")

    print(f"\n腐蚀预测（校准后）:")
    print(f"{'功率(W)':>8} {'实测(A/cm²)':>14} {'预测(A/cm²)':>14} {'倍数':>8}")
    print("-" * 50)
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        ratio = c_pred[i] / exp['腐蚀电流']
        print(f"{exp['激光功率']:>8d} {exp['腐蚀电流']:>14.2e} {c_pred[i]:>14.2e} {ratio:>7.2f}x")

    # 判断趋势
    peak_actual = np.argmax([e["硬度"] for e in ROCKIT485_EXPERIMENTS])
    peak_pred = np.argmax(h_pred)
    print(f"\n硬度峰值: 实测={ROCKIT485_EXPERIMENTS[peak_actual]['激光功率']}W, "
          f"预测={ROCKIT485_EXPERIMENTS[int(peak_pred)]['激光功率']}W")
    print(f"趋势正确: {'✓' if peak_actual == peak_pred else '✗'}")

    # ---- 逐维扫描帕累托 ----
    print(f"\n{'='*60}")
    print("逐维扫描帕累托优化")
    print("=" * 60)

    all_samples_list = []

    for param_idx, param in enumerate(PROCESS_FEATURES):
        print(f"\n--- 扫描维度: {param} ---")

        free_params = [p for p in PROCESS_FEATURES if p != param]
        fixed_low, fixed_high = SEARCH_SPACE[param]
        fixed_values = np.linspace(fixed_low, fixed_high, FIXED_DIM_POINTS)

        all_results = []
        for idx, fixed_val in enumerate(fixed_values):
            free_bounds = {p: SEARCH_SPACE[p] for p in free_params}
            samples = latin_hypercube_sampling(
                free_bounds, SAMPLES_PER_FIXED_POINT,
                seed=42 + param_idx * 10000 + idx
            )
            samples[param] = fixed_val

            X = build_feature_matrix(samples[PROCESS_FEATURES],
                                     ROCKIT485_COMPOSITION, hardness_bundle)

            h = predict_hardness(X, hardness_bundle)
            c = predict_corrosion_calibrated(X, corrosion_bundle)

            result_df = samples[PROCESS_FEATURES].copy()
            result_df["硬度"] = h
            result_df["腐蚀电流"] = c
            result_df["固定参数"] = param
            result_df["固定参数值"] = fixed_val
            all_results.append(result_df)

        all_df = pd.concat(all_results, ignore_index=True)
        all_samples_list.append(all_df)
        print(f"  [扫描] {param}: {len(all_df)}个采样")

    # ---- 整体帕累托前沿 ----
    print(f"\n[步骤] 计算整体帕累托前沿")
    all_samples_df = pd.concat(all_samples_list, ignore_index=True)
    all_samples_df = all_samples_df.drop_duplicates(
        subset=PROCESS_FEATURES
    ).reset_index(drop=True)

    objectives = [("硬度", "max"), ("腐蚀电流", "min")]
    global_pareto = pareto_front(all_samples_df, objectives)

    print(f"  总去重采样点数: {len(all_samples_df)}")
    print(f"  整体帕累托前沿解数: {len(global_pareto)}")

    # ---- 实验点在帕累托空间中的位置 ----
    print(f"\n[步骤] 实验点在帕累托空间中的位置")
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        h = h_pred[i]
        c = c_pred[i]
        dominated = False
        for _, p in global_pareto.iterrows():
            if p['硬度'] >= h and p['腐蚀电流'] <= c:
                if p['硬度'] > h or p['腐蚀电流'] < c:
                    dominated = True
                    break
        status = "被支配（非最优）" if dominated else "在前沿上（帕累托最优）"
        print(f"  {exp['激光功率']}W @15mm/s: 硬度={h:.0f}HV, "
              f"腐蚀={c:.2e} → {status}")

    # ---- 分层提取代表性参数 ----
    print(f"\n[步骤] 提取代表性最优参数组合（按硬度分层）")
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

    # ---- 保存结果 ----
    out_dir = os.path.join(RESULT_DIR, "pareto_rockit485_fewshot")
    os.makedirs(out_dir, exist_ok=True)

    global_path = os.path.join(out_dir, "pareto_global_front.csv")
    global_pareto.to_csv(global_path, index=False)
    print(f"\n[输出] 整体帕累托前沿: {global_path}")

    rep_path = os.path.join(out_dir, "pareto_representative_params.csv")
    rep_df.to_csv(rep_path, index=False)
    print(f"[输出] 代表性参数组合: {rep_path}")

    # ---- 结果摘要 ----
    print(f"\n{'='*80}")
    print("Rockit485 Few-shot版帕累托优化结果摘要")
    print("=" * 80)
    print(f"总去重采样点: {len(all_samples_df)}")
    print(f"整体帕累托前沿: {len(global_pareto)}个解")
    print(f"硬度范围: {global_pareto['硬度'].min():.0f} ~ {global_pareto['硬度'].max():.0f} HV")
    print(f"腐蚀电流范围: {global_pareto['腐蚀电流'].min():.2e} ~ "
          f"{global_pareto['腐蚀电流'].max():.2e} A/cm²")
    print()
    print("代表性参数组合（按硬度分层，取腐蚀最低者）:")
    print("-" * 105)
    header = f"{'硬度(HV)':>9} {'腐蚀电流':>12} " + " ".join(f"{p:>9}" for p in PROCESS_FEATURES)
    print(header)
    print("-" * 105)
    for _, row in rep_df.iterrows():
        line = f"{row['硬度']:>9.0f} {row['腐蚀电流']:>12.2e} " + \
               " ".join(f"{row[p]:>9.2f}" for p in PROCESS_FEATURES)
        print(line)
    print("=" * 80)

    return global_pareto, rep_df


if __name__ == "__main__":
    run_fewshot_pareto()
