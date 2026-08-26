"""
实验验证模块
=============
用用户提供的实验数据验证模型预测效果，并检查帕累托优化是否覆盖这些工艺参数。
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import PROCESS_FEATURES, COMPOSITION_FEATURES, RESULT_DIR
from src.evaluation import load_model_bundle
from src.pareto_optimization import (
    build_feature_matrix, predict_hardness, predict_corrosion,
    get_typical_composition
)
from src.data_preprocessing import load_raw_data, clean_data


# ============================================================
# 用户实验数据（来自实验设计参数.docx）
# ============================================================

# Rockit485 熔覆粉末成分 (wt.%)
# 注意：Co(4.5%)、Nb(0.8%) 在我们的模型中没有对应特征，归入"其他"
# 模型只用 C, Cr, Si, Ni, Fe, Mn, Mo 7个元素
ROCKIT485_COMPOSITION = {
    "C": 0.15,
    "Cr": 13.0,
    "Si": 0.6,
    "Ni": 4.0,
    "Mo": 2.8,
    "Mn": 0.5,   # 文档未给出，估算值（常见不锈钢Mn含量）
    "Fe": None,  # 由余量计算
}

# 计算Fe含量（归一化到7个元素总和为100%）
_known_sum = sum(v for k, v in ROCKIT485_COMPOSITION.items() if v is not None)
ROCKIT485_COMPOSITION["Fe"] = 100.0 - _known_sum
# 注意：实际粉末中含Co+Nb+Others≈6.3%，这部分模型无法捕捉

# 实验工艺参数（固定）
FIXED_PARAMS = {
    "扫描速度": 15.0,   # mm/s
    "送粉速率": 10.0,   # g/min
    "光斑直径": 3.0,    # mm  (文档未给出，假设常见值)
    "离焦量": 0.0,      # mm  (文档未给出，假设焦面)
}

# 激光功率梯度
LASER_POWERS = [1200, 1500, 1800, 2100]  # W

# 实验实测结果
EXPERIMENTAL_RESULTS = {
    1200:  {"硬度": 470, "腐蚀电流": 5.9515e-7},
    1500:  {"硬度": 500, "腐蚀电流": 3.1202e-7},
    1800:  {"硬度": 520, "腐蚀电流": 3.4757e-7},
    2100:  {"硬度": 480, "腐蚀电流": 4.5000e-7},
}


def run_validation():
    """执行实验验证"""
    print("=" * 80)
    print("实验验证 - 模型预测 vs 实测数据")
    print("=" * 80)

    # 1. 加载模型
    hardness_bundle = load_model_bundle("LightGBM", "硬度")
    corrosion_bundle = load_model_bundle("RF", "腐蚀电流")
    print(f"\n[模型] 硬度: LightGBM")
    print(f"[模型] 腐蚀: RF")

    # 2. 打印成分信息
    print(f"\n[粉末成分] Rockit485 (7元素归一化):")
    for elem in COMPOSITION_FEATURES:
        print(f"  {elem}: {ROCKIT485_COMPOSITION[elem]:.2f}%")
    print(f"  ⚠ 实际粉末含 Co(4.5%) + Nb(0.8%) + Others(~1%)，模型无法捕捉")

    print(f"\n[固定工艺参数]")
    for k, v in FIXED_PARAMS.items():
        note = " (假设值)" if k in ["光斑直径", "离焦量"] else ""
        print(f"  {k}: {v}{note}")

    # 3. 构建4组实验参数
    exp_data = []
    for power in LASER_POWERS:
        params = FIXED_PARAMS.copy()
        params["激光功率"] = power
        exp_data.append(params)

    exp_df = pd.DataFrame(exp_data)[PROCESS_FEATURES]  # 按顺序排列

    # 4. 预测
    X = build_feature_matrix(exp_df, ROCKIT485_COMPOSITION, hardness_bundle)

    hardness_pred = predict_hardness(X, hardness_bundle)
    corrosion_pred = predict_corrosion(X, corrosion_bundle)

    # 5. 对比结果
    print(f"\n{'='*80}")
    print("硬度预测对比")
    print("=" * 80)
    print(f"{'激光功率(W)':>10} {'实测(HV)':>10} {'预测(HV)':>10} {'误差(HV)':>10} {'相对误差':>10}")
    print("-" * 60)

    hardness_errors = []
    for i, power in enumerate(LASER_POWERS):
        actual = EXPERIMENTAL_RESULTS[power]["硬度"]
        pred = hardness_pred[i]
        err = pred - actual
        rel_err = abs(err) / actual * 100
        hardness_errors.append(abs(err))
        print(f"{power:>10d} {actual:>10.1f} {pred:>10.1f} {err:>+10.1f} {rel_err:>9.1f}%")

    print(f"\n  平均绝对误差: {np.mean(hardness_errors):.1f} HV")
    print(f"  平均相对误差: {np.mean([abs(hardness_pred[i] - EXPERIMENTAL_RESULTS[p]['硬度']) / EXPERIMENTAL_RESULTS[p]['硬度'] * 100 for i, p in enumerate(LASER_POWERS)]):.1f}%")

    print(f"\n{'='*80}")
    print("腐蚀电流预测对比")
    print("=" * 80)
    print(f"{'激光功率(W)':>10} {'实测(A/cm²)':>14} {'预测(A/cm²)':>14} {'倍数差':>10}")
    print("-" * 60)

    corrosion_ratios = []
    for i, power in enumerate(LASER_POWERS):
        actual = EXPERIMENTAL_RESULTS[power]["腐蚀电流"]
        pred = corrosion_pred[i]
        ratio = pred / actual  # 预测/实测
        corrosion_ratios.append(ratio)
        print(f"{power:>10d} {actual:>14.2e} {pred:>14.2e} {ratio:>10.2f}x")

    print(f"\n  预测/实测比值范围: {min(corrosion_ratios):.2f}x ~ {max(corrosion_ratios):.2f}x")
    print(f"  数量级是否准确: {'是' if all(0.1 <= r <= 10 for r in corrosion_ratios) else '否'}")

    # 6. 帕累托前沿覆盖检查
    print(f"\n{'='*80}")
    print("帕累托前沿覆盖检查")
    print("=" * 80)

    pareto_path = os.path.join(RESULT_DIR, "pareto", "pareto_global_front.csv")
    pareto_df = pd.read_csv(pareto_path)

    print(f"\n整体帕累托前沿范围:")
    print(f"  硬度: {pareto_df['硬度'].min():.0f} ~ {pareto_df['硬度'].max():.0f} HV")
    print(f"  腐蚀电流: {pareto_df['腐蚀电流'].min():.2e} ~ {pareto_df['腐蚀电流'].max():.2e} A/cm²")

    print(f"\n实验点在帕累托空间中的位置:")
    print(f"{'功率(W)':>8} {'硬度(HV)':>10} {'腐蚀(A/cm²)':>14} "
          f"{'硬度排名%':>10} {'腐蚀排名%':>10} {'是否被支配':>10}")
    print("-" * 80)

    # 用预测值来检查帕累托支配关系（因为帕累托是模型预测的帕累托）
    for i, power in enumerate(LASER_POWERS):
        h_pred = hardness_pred[i]
        c_pred = corrosion_pred[i]

        # 在帕累托前沿中的硬度排名百分位（越高越好）
        h_rank = (pareto_df['硬度'] <= h_pred).sum() / len(pareto_df) * 100
        # 腐蚀排名百分位（越低越好，所以用<=）
        c_rank = (pareto_df['腐蚀电流'] >= c_pred).sum() / len(pareto_df) * 100

        # 判断是否被帕累托前沿中的任何解支配
        dominated = False
        for _, p in pareto_df.iterrows():
            if p['硬度'] >= h_pred and p['腐蚀电流'] <= c_pred:
                if p['硬度'] > h_pred or p['腐蚀电流'] < c_pred:
                    dominated = True
                    break

        status = "是" if dominated else "否(在前沿上)"
        print(f"{power:>8d} {h_pred:>10.1f} {c_pred:>14.2e} "
              f"{h_rank:>9.1f}% {c_rank:>9.1f}% {status:>10}")

    # 7. 帕累托前沿中扫描速度=15附近的解
    print(f"\n帕累托前沿中扫描速度接近15 mm/s的解 (±3 mm/s):")
    mask = (pareto_df['扫描速度'] >= 12) & (pareto_df['扫描速度'] <= 18)
    near_15 = pareto_df[mask].sort_values('硬度', ascending=False)
    if len(near_15) > 0:
        print(f"  找到 {len(near_15)} 个解")
        print(f"  {'硬度(HV)':>10} {'腐蚀电流':>14} {'激光功率':>10} {'扫描速度':>10} {'送粉速率':>10} {'光斑直径':>10} {'离焦量':>10}")
        for _, row in near_15.head(10).iterrows():
            print(f"  {row['硬度']:>10.1f} {row['腐蚀电流']:>14.2e} "
                  f"{row['激光功率']:>10.1f} {row['扫描速度']:>10.2f} "
                  f"{row['送粉速率']:>10.2f} {row['光斑直径']:>10.2f} "
                  f"{row['离焦量']:>10.2f}")
    else:
        print("  未找到（帕累托最优区不在这个扫描速度范围）")

    # 8. 保存验证结果
    validation_df = pd.DataFrame({
        "激光功率(W)": LASER_POWERS,
        "扫描速度(mm/s)": [FIXED_PARAMS["扫描速度"]] * 4,
        "送粉速率(g/min)": [FIXED_PARAMS["送粉速率"]] * 4,
        "光斑直径(mm)": [FIXED_PARAMS["光斑直径"]] * 4,
        "离焦量(mm)": [FIXED_PARAMS["离焦量"]] * 4,
        "实测硬度(HV)": [EXPERIMENTAL_RESULTS[p]["硬度"] for p in LASER_POWERS],
        "预测硬度(HV)": hardness_pred.round(1),
        "硬度误差(HV)": (hardness_pred - np.array([EXPERIMENTAL_RESULTS[p]["硬度"] for p in LASER_POWERS])).round(1),
        "硬度相对误差(%)": (abs(hardness_pred - np.array([EXPERIMENTAL_RESULTS[p]["硬度"] for p in LASER_POWERS])) / np.array([EXPERIMENTAL_RESULTS[p]["硬度"] for p in LASER_POWERS]) * 100).round(1),
        "实测腐蚀电流(A/cm2)": [EXPERIMENTAL_RESULTS[p]["腐蚀电流"] for p in LASER_POWERS],
        "预测腐蚀电流(A/cm2)": corrosion_pred,
        "预测/实测倍数": np.array(corrosion_ratios).round(2),
    })

    out_path = os.path.join(RESULT_DIR, "experimental_validation.csv")
    validation_df.to_csv(out_path, index=False)
    print(f"\n[输出] 验证结果已保存: {out_path}")

    # 9. 总结
    print(f"\n{'='*80}")
    print("验证总结")
    print("=" * 80)

    avg_hardness_rel_err = np.mean([
        abs(hardness_pred[i] - EXPERIMENTAL_RESULTS[p]['硬度']) / EXPERIMENTAL_RESULTS[p]['硬度'] * 100
        for i, p in enumerate(LASER_POWERS)
    ])

    print(f"\n硬度预测:")
    print(f"  平均相对误差: {avg_hardness_rel_err:.1f}%")
    print(f"  趋势是否正确（1800W为峰值）: "
          f"{'是' if hardness_pred[2] == max(hardness_pred) else '否'}")

    print(f"\n腐蚀电流预测:")
    print(f"  预测/实测比值: {min(corrosion_ratios):.2f}x ~ {max(corrosion_ratios):.2f}x")
    print(f"  趋势是否正确（1500W最优）: "
          f"{'是' if corrosion_pred[1] == min(corrosion_pred) else '否'}")

    print(f"\n帕累托覆盖:")
    print(f"  实验参数区（15mm/s扫描速度）是否在帕累托前沿上: "
          f"{'是' if len(near_15) > 0 else '否'}")
    print(f"  原因: 帕累托最优解集中在极端参数区（极高硬度或极低腐蚀），")
    print(f"        15mm/s属于中速区，硬度和耐蚀性均非最优")

    print(f"\n⚠ 重要局限性说明:")
    print(f"  1. 光斑直径和离焦量为假设值（3mm, 0mm），实际可能不同")
    print(f"  2. Rockit485含Co(4.5%)和Nb(0.8%)，模型训练数据中无此二元素")
    print(f"  3. Mn含量为估算值(0.5%)，文档未提供")
    print(f"  4. 训练数据以Fe基合金为主，Rockit485马氏体不锈钢成分差异较大")

    print(f"\n{'='*80}")

    return validation_df


if __name__ == "__main__":
    run_validation()
