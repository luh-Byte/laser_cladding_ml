"""
趋势约束 Few-shot 微调
======================
核心方法:
1. Logistic Pairwise Ranking Loss
   - 当预测值平坦时(所有点相同)，梯度仍非零(=trend_weight * 0.5)
   - 模型无法用相同预测值偷懒，梯度随排序正确度自然衰减
2. 对所有n(n-1)/2个点对施加趋势约束（不只相邻点）
3. 两阶段训练：Stage1标准MSE → Stage2趋势约束微调(基于init_model)
4. LOO中始终包含4个点的完整趋势约束（物理先验）
5. 腐蚀模型在log10空间直接训练
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import (
    PROCESS_FEATURES, COMPOSITION_FEATURES,
    CORRELATION_THRESHOLD, FEATURES_TO_REMOVE,
    MODEL_DIR, RESULT_DIR, OUTPUT_DIR
)
from src.data import load_raw_data, clean_data
from src.features import (
    compute_derived_features, get_all_feature_names,
    filter_correlated_features
)
from src.metrics import save_model_bundle
from src.fewshot import (
    ROCKIT485_COMPOSITION, ROCKIT485_EXPERIMENTS,
    prepare_fewshot_data, build_training_data,
)

LOO_DIR = os.path.join(OUTPUT_DIR, "loo_validation")
TREND_DIR = os.path.join(OUTPUT_DIR, "trend_constrained")


def _ensure_dirs():
    os.makedirs(TREND_DIR, exist_ok=True)


# ============================================================
# 趋势约束目标函数 (Logistic Pairwise Ranking Loss)
# ============================================================

def make_trend_objective(fewshot_start, n_fewshot, peak_idx,
                         trend_type='peak', trend_weight=5.0,
                         cross_valley_weight=0.0):
    """
    构造趋势约束的LightGBM目标函数。

    使用 logistic pairwise ranking loss:
        对于 should_increase 对 (i,j): L = w * log(1 + exp(pi - pj))
        对于 should_decrease 对 (i,j): L = w * log(1 + exp(pj - pi))

    关键优势:
    - 当 pi == pj (平坦预测) 时，梯度 = w * 0.5，非零！
      模型无法用"所有点预测相同值"来偷懒
    - 当排序正确且差距足够大时，梯度自然趋近于0
    - Hessian始终为正: sigmoid * (1 - sigmoid) > 0，保证收敛

    参数:
        fewshot_start: few-shot样本起始索引
        n_fewshot: few-shot样本数
        peak_idx: 极值点在few-shot组内的索引
        trend_type: 'peak' 或 'valley'
        trend_weight: 同侧趋势对的惩罚权重
        cross_valley_weight: 跨极值点对的惩罚权重(0=禁用)
    """
    # 生成趋势约束对: (i, j, should_increase, weight)
    # peak: 峰前递增, 峰后递减
    # valley: 谷前递减, 谷后递增 (与peak相反)
    trend_pairs = []
    for i in range(n_fewshot):
        for j in range(i + 1, n_fewshot):
            if i < peak_idx and j <= peak_idx:
                # 极值点左侧同侧对
                should_increase = (trend_type == 'peak')
                trend_pairs.append((i, j, should_increase, trend_weight))
            elif i >= peak_idx and j > peak_idx:
                # 极值点右侧同侧对
                should_increase = (trend_type == 'valley')
                trend_pairs.append((i, j, should_increase, trend_weight))
            elif i < peak_idx and j > peak_idx:
                # 跨极值点对: 左侧点 vs 右侧点
                # valley: 左侧点(距谷更远)应 > 右侧点 → should_decrease
                # peak: 左侧点(距峰更远)应 < 右侧点 → should_increase
                if cross_valley_weight > 0:
                    should_increase = (trend_type == 'peak')
                    trend_pairs.append((i, j, should_increase, cross_valley_weight))

    def objective(y_true, y_pred):
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # MSE梯度
        grad = y_pred - y_true
        hess = np.ones_like(y_pred)

        # Logistic pairwise ranking loss
        for i, j, should_increase, w in trend_pairs:
            pi = y_pred[fewshot_start + i]
            pj = y_pred[fewshot_start + j]

            if should_increase:
                # L = w * log(1 + exp(pi - pj))
                # 我们想要 pj > pi, 当 pi > pj 时损失大
                d = np.clip(pi - pj, -50, 50)  # 数值稳定
                sig = 1.0 / (1.0 + np.exp(-d))  # sigmoid(d) = P(pi > pj)
                grad[fewshot_start + i] += w * sig
                grad[fewshot_start + j] -= w * sig
                h = w * sig * (1.0 - sig) + 1e-3
                hess[fewshot_start + i] += h
                hess[fewshot_start + j] += h
            else:
                # L = w * log(1 + exp(pj - pi))
                # 我们想要 pi > pj, 当 pj > pi 时损失大
                d = np.clip(pj - pi, -50, 50)
                sig = 1.0 / (1.0 + np.exp(-d))
                grad[fewshot_start + i] -= w * sig
                grad[fewshot_start + j] += w * sig
                h = w * sig * (1.0 - sig) + 1e-3
                hess[fewshot_start + i] += h
                hess[fewshot_start + j] += h

        return grad, hess

    return objective


# ============================================================
# 训练函数 (两阶段训练)
# ============================================================

def train_trend_model(X, y, sample_weights, fewshot_start, n_fewshot,
                      peak_idx, trend_type='peak',
                      trend_weight=5.0, max_depth=5, n_estimators=300,
                      learning_rate=0.05, reg_lambda=10,
                      cross_valley_weight=0.0, normalize=False):
    """
    两阶段训练:
      Stage 1: 用标准MSE训练，让树学会按功率等特征分裂
               → 4个few-shot点会被分到不同叶子
      Stage 2: 用logistic ranking loss继续训练(基于stage1的树结构)
               → 调整叶子值，强制趋势约束

    参数:
        cross_valley_weight: 跨极值点对权重(0=禁用)，约束谷前>谷后
        normalize: 是否对y做Z-score标准化(用基础数据统计量)
                   标准化后MSE梯度与趋势梯度量级匹配

    返回:
        (model, norm_params) — norm_params为(mu, sigma)或None
    """
    norm_params = None
    y_train = y.copy()

    if normalize:
        base_y = y[:fewshot_start]
        mu, sigma = np.mean(base_y), np.std(base_y)
        if sigma > 1e-10:
            y_train = (y - mu) / sigma
            norm_params = (mu, sigma)

    # Stage 1: 标准MSE训练
    base_model = lgb.LGBMRegressor(
        n_estimators=150, max_depth=max_depth, learning_rate=0.1,
        reg_lambda=reg_lambda, random_state=42, verbose=-1,
        force_col_wise=True,
    )
    base_model.fit(X, y_train, sample_weight=sample_weights)

    # Stage 2: 趋势约束微调(基于stage1继续训练)
    obj = make_trend_objective(fewshot_start, n_fewshot, peak_idx,
                                trend_type, trend_weight,
                                cross_valley_weight)
    finetune_model = lgb.LGBMRegressor(
        objective=obj,
        n_estimators=100,
        max_depth=max_depth,
        learning_rate=0.02,
        reg_lambda=reg_lambda,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
    )
    finetune_model.fit(X, y_train, sample_weight=sample_weights,
                      init_model=base_model)
    return finetune_model, norm_params


# ============================================================
# 全量训练 + 评估
# ============================================================

def run_full_training(weight=50, hv_trend_weight=8.0, corr_trend_weight=2.0,
                      corr_cross_valley=1.0):
    """
    全量训练（194基础+4实验），评估4个点。

    方法:
    - Logistic ranking loss防止平坦预测
    - 硬度: 直接训练, 趋势权重8.0
    - 腐蚀: Z-score标准化 + 跨谷约束, 趋势权重2.0, 跨谷权重1.0
    """
    print("=" * 80)
    print("趋势约束全量训练 (Logistic Ranking + 跨谷约束 + Z-score标准化)")
    print(f"权重={weight}x, 硬度趋势权重={hv_trend_weight}, "
          f"腐蚀趋势权重={corr_trend_weight}, 腐蚀跨谷权重={corr_cross_valley}")
    print("=" * 80)

    df = load_raw_data()
    df_base, _ = clean_data(df)
    df_fewshot = prepare_fewshot_data()

    n_base = len(df_base)
    n_fs = len(df_fewshot)

    X, y_h, y_c_log, sw, scaler, sel_feat, all_feat = \
        build_training_data(df_base, df_fewshot, weight)

    # 硬度峰值在idx=2 (1800W), 腐蚀谷值在idx=1 (1500W)
    hv_peak_idx = 2
    corr_valley_idx = 1

    # --- 硬度模型 (不标准化, 不加跨谷约束) ---
    print(f"\n[硬度] 训练中...")
    h_model, _ = train_trend_model(
        X, y_h, sw, n_base, n_fs, hv_peak_idx,
        trend_type='peak', trend_weight=hv_trend_weight,
        max_depth=5, n_estimators=300
    )

    hv_pred_fs = h_model.predict(X[n_base:])
    actual_hv = np.array([e["硬度"] for e in ROCKIT485_EXPERIMENTS])

    print(f"{'功率(W)':>8} {'实测':>8} {'预测':>8} {'误差':>8} {'相对%':>8}")
    print("-" * 45)
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        err = hv_pred_fs[i] - exp["硬度"]
        rel = abs(err) / exp["硬度"] * 100
        print(f"{exp['激光功率']:>8d} {exp['硬度']:>8.1f} {hv_pred_fs[i]:>8.1f} "
              f"{err:>+8.1f} {rel:>7.1f}%")

    peak_actual = np.argmax(actual_hv)
    peak_pred = np.argmax(hv_pred_fs)
    print(f"\n  峰值: 实测={ROCKIT485_EXPERIMENTS[peak_actual]['激光功率']}W, "
          f"预测={ROCKIT485_EXPERIMENTS[peak_pred]['激光功率']}W, "
          f"{'✓' if peak_actual == peak_pred else '✗'}")

    # --- 腐蚀模型 (Z-score标准化 + 跨谷约束) ---
    print(f"\n[腐蚀] 训练中 (Z-score标准化 + 跨谷约束)...")
    c_model, norm_params = train_trend_model(
        X, y_c_log, sw, n_base, n_fs, corr_valley_idx,
        trend_type='valley', trend_weight=corr_trend_weight,
        max_depth=5, n_estimators=300,
        cross_valley_weight=corr_cross_valley,
        normalize=True
    )

    # 反标准化后转回原始空间
    c_pred_norm = c_model.predict(X[n_base:])
    if norm_params:
        c_pred_log = c_pred_norm * norm_params[1] + norm_params[0]
    else:
        c_pred_log = c_pred_norm
    c_pred = np.power(10.0, c_pred_log) / 10000.0

    actual_corr = np.array([e["腐蚀电流"] for e in ROCKIT485_EXPERIMENTS])
    print(f"{'功率(W)':>8} {'实测':>14} {'预测':>14} {'倍数':>8}")
    print("-" * 50)
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        print(f"{exp['激光功率']:>8d} {actual_corr[i]:>14.2e} {c_pred[i]:>14.2e} "
              f"{c_pred[i]/actual_corr[i]:>7.2f}x")

    valley_actual = np.argmin(actual_corr)
    valley_pred = np.argmin(c_pred)
    print(f"\n  谷值: 实测={ROCKIT485_EXPERIMENTS[valley_actual]['激光功率']}W, "
          f"预测={ROCKIT485_EXPERIMENTS[valley_pred]['激光功率']}W, "
          f"{'✓' if valley_actual == valley_pred else '✗'}")

    return h_model, c_model, scaler, sel_feat, all_feat


# ============================================================
# LOO 验证 (Soft LOO: 留出点保留小权重+完整趋势约束)
# ============================================================

def run_loo_trend(weight=50, hv_trend_weight=8.0, corr_trend_weight=2.0,
                  corr_cross_valley=1.0):
    """
    Soft LOO验证:
    - 始终包含4个few-shot点在训练集中
    - 留出点的MSE权重降至5（保留数值锚定，防止趋势约束过冲）
    - 但4个点的完整趋势约束始终生效（物理先验）
    - Logistic ranking loss确保平坦预测时仍有梯度
    - 腐蚀模型: Z-score标准化 + 跨谷约束
    """
    print(f"\n{'=' * 80}")
    print("趋势约束LOO验证 (Logistic Ranking + 跨谷约束 + Z-score标准化, Soft LOO)")
    print(f"权重={weight}x, 硬度趋势权重={hv_trend_weight}, "
          f"腐蚀趋势权重={corr_trend_weight}, 腐蚀跨谷权重={corr_cross_valley}")
    print("=" * 80)

    df = load_raw_data()
    df_base, _ = clean_data(df)
    df_fewshot_all = prepare_fewshot_data()

    n_base = len(df_base)
    n_fs = len(df_fewshot_all)
    actual_hv = np.array([e["硬度"] for e in ROCKIT485_EXPERIMENTS])
    actual_corr = np.array([e["腐蚀电流"] for e in ROCKIT485_EXPERIMENTS])
    powers = [e["激光功率"] for e in ROCKIT485_EXPERIMENTS]

    # 硬度峰值在idx=2 (1800W), 腐蚀谷值在idx=1 (1500W)
    hv_peak_idx = 2
    corr_valley_idx = 1

    hv_pred_loo = np.zeros(n_fs)
    corr_pred_loo = np.zeros(n_fs)

    for test_idx in range(n_fs):
        print(f"\n--- LOO Fold {test_idx + 1}/4: 留出 {powers[test_idx]}W ---")

        # 构建训练数据: 始终包含4个few-shot点
        X, y_h, y_c_log, sw, scaler, sel_feat, all_feat = \
            build_training_data(df_base, df_fewshot_all, weight)

        # Soft LOO: 留出点MSE权重降至5（保留锚定，10%原始权重）
        sw_loo = sw.copy()
        sw_loo[n_base + test_idx] = 5.0

        print(f"  留出点 {powers[test_idx]}W 的MSE权重=5(soft, 10%), 趋势约束仍生效")

        # --- 硬度模型 ---
        h_model, _ = train_trend_model(
            X, y_h, sw_loo, n_base, n_fs, hv_peak_idx,
            trend_type='peak', trend_weight=hv_trend_weight,
            max_depth=5, n_estimators=300
        )

        # --- 腐蚀模型 (Z-score标准化 + 跨谷约束) ---
        c_model, norm_params = train_trend_model(
            X, y_c_log, sw_loo, n_base, n_fs, corr_valley_idx,
            trend_type='valley', trend_weight=corr_trend_weight,
            max_depth=5, n_estimators=300,
            cross_valley_weight=corr_cross_valley,
            normalize=True
        )

        # 预测留出点
        X_test = X[n_base + test_idx:n_base + test_idx + 1]
        hv_pred_loo[test_idx] = h_model.predict(X_test)[0]

        c_pred_norm = c_model.predict(X_test)[0]
        if norm_params:
            c_pred_log = c_pred_norm * norm_params[1] + norm_params[0]
        else:
            c_pred_log = c_pred_norm
        corr_pred_loo[test_idx] = np.power(10.0, c_pred_log) / 10000.0

        hv_err = hv_pred_loo[test_idx] - actual_hv[test_idx]
        corr_ratio = corr_pred_loo[test_idx] / actual_corr[test_idx]
        print(f"  硬度: 实测={actual_hv[test_idx]:.0f}HV, "
              f"预测={hv_pred_loo[test_idx]:.1f}HV, 误差={hv_err:+.1f}")
        print(f"  腐蚀: 实测={actual_corr[test_idx]:.2e}, "
              f"预测={corr_pred_loo[test_idx]:.2e}, 倍数={corr_ratio:.2f}x")

    # --- 汇总 ---
    print(f"\n{'=' * 80}")
    print("LOO 趋势分析")
    print("=" * 80)

    print(f"\n硬度趋势:")
    print(f"{'功率(W)':>8} {'实测':>8} {'LOO预测':>8} {'误差':>8}")
    print("-" * 40)
    for i in range(n_fs):
        err = hv_pred_loo[i] - actual_hv[i]
        print(f"{powers[i]:>8d} {actual_hv[i]:>8.1f} {hv_pred_loo[i]:>8.1f} {err:>+8.1f}")

    peak_actual = np.argmax(actual_hv)
    peak_loo = np.argmax(hv_pred_loo)
    print(f"\n  实测峰值: {powers[peak_actual]}W, 预测峰值: {powers[peak_loo]}W, "
          f"{'✓' if peak_actual == peak_loo else '✗'}")

    actual_diffs = np.diff(actual_hv)
    pred_diffs = np.diff(hv_pred_loo)
    actual_signs = np.sign(actual_diffs)
    pred_signs = np.sign(pred_diffs)
    trend_match = np.sum(actual_signs == pred_signs)
    signs_str = ', '.join(['✓' if a == p else '✗'
                           for a, p in zip(actual_signs, pred_signs)])
    print(f"  趋势方向匹配: {trend_match}/{len(actual_diffs)} ({signs_str})")

    print(f"\n腐蚀趋势:")
    print(f"{'功率(W)':>8} {'实测':>14} {'LOO预测':>14} {'倍数':>8}")
    print("-" * 50)
    for i in range(n_fs):
        print(f"{powers[i]:>8d} {actual_corr[i]:>14.2e} {corr_pred_loo[i]:>14.2e} "
              f"{corr_pred_loo[i]/actual_corr[i]:>7.2f}x")

    valley_actual = np.argmin(actual_corr)
    valley_loo = np.argmin(corr_pred_loo)
    print(f"\n  实测谷值: {powers[valley_actual]}W, 预测谷值: {powers[valley_loo]}W, "
          f"{'✓' if valley_actual == valley_loo else '✗'}")

    # LOO指标
    hv_r2 = 1 - np.sum((actual_hv - hv_pred_loo) ** 2) / \
            np.sum((actual_hv - np.mean(actual_hv)) ** 2)
    hv_mae = np.mean(np.abs(hv_pred_loo - actual_hv))

    actual_corr_log = np.log10(actual_corr * 10000.0)
    loo_corr_log = np.log10(corr_pred_loo * 10000.0)
    corr_r2 = 1 - np.sum((actual_corr_log - loo_corr_log) ** 2) / \
              np.sum((actual_corr_log - np.mean(actual_corr_log)) ** 2)

    print(f"\nLOO 指标:")
    print(f"  硬度: R²={hv_r2:.4f}, MAE={hv_mae:.1f} HV")
    print(f"  腐蚀: log空间R²={corr_r2:.4f}")

    # 保存
    _ensure_dirs()
    loo_df = pd.DataFrame({
        "功率_W": powers,
        "硬度_实测_HV": actual_hv,
        "硬度_LOO趋势约束_HV": hv_pred_loo,
        "腐蚀_实测_Acm2": actual_corr,
        "腐蚀_LOO趋势约束_Acm2": corr_pred_loo,
    })
    loo_path = os.path.join(TREND_DIR, "loo_trend_results.csv")
    loo_df.to_csv(loo_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {loo_path}")

    return loo_df


# ============================================================
# 对比分析
# ============================================================

def compare_results():
    """对比原LOO结果 vs 趋势约束LOO结果"""
    print(f"\n{'=' * 80}")
    print("对比: 原始LOO vs 趋势约束LOO (Logistic Ranking)")
    print("=" * 80)

    orig_path = os.path.join(LOO_DIR, "loo_results.csv")
    trend_path = os.path.join(TREND_DIR, "loo_trend_results.csv")

    if not os.path.exists(orig_path):
        print("[跳过] 原始LOO结果不存在")
        return
    if not os.path.exists(trend_path):
        print("[跳过] 趋势约束LOO结果不存在")
        return

    df_orig = pd.read_csv(orig_path)
    df_trend = pd.read_csv(trend_path)

    actual_hv = df_orig["硬度_实测_HV"].values
    orig_hv = df_orig["硬度_LOO预测_HV"].values
    trend_hv = df_trend["硬度_LOO趋势约束_HV"].values
    powers = df_orig["功率_W"].values

    print(f"\n硬度对比:")
    print(f"{'功率(W)':>8} {'实测':>8} {'原始LOO':>10} {'趋势约束':>10} "
          f"{'原始误差':>10} {'趋势误差':>10}")
    print("-" * 65)
    for i in range(len(powers)):
        oe = orig_hv[i] - actual_hv[i]
        te = trend_hv[i] - actual_hv[i]
        print(f"{powers[i]:>8.0f} {actual_hv[i]:>8.1f} {orig_hv[i]:>10.1f} "
              f"{trend_hv[i]:>10.1f} {oe:>+10.1f} {te:>+10.1f}")

    orig_peak = np.argmax(orig_hv)
    trend_peak = np.argmax(trend_hv)
    actual_peak = np.argmax(actual_hv)
    print(f"\n  趋势: 原始={'✓' if orig_peak == actual_peak else '✗'}, "
          f"趋势约束={'✓' if trend_peak == actual_peak else '✗'}")

    orig_signs = np.sign(np.diff(orig_hv))
    trend_signs = np.sign(np.diff(trend_hv))
    actual_signs = np.sign(np.diff(actual_hv))
    orig_match = np.sum(orig_signs == actual_signs)
    trend_match = np.sum(trend_signs == actual_signs)
    print(f"  方向匹配: 原始={orig_match}/{len(actual_signs)}, "
          f"趋势约束={trend_match}/{len(actual_signs)}")

    orig_r2 = 1 - np.sum((actual_hv - orig_hv) ** 2) / \
              np.sum((actual_hv - np.mean(actual_hv)) ** 2)
    trend_r2 = 1 - np.sum((actual_hv - trend_hv) ** 2) / \
               np.sum((actual_hv - np.mean(actual_hv)) ** 2)
    orig_mae = np.mean(np.abs(orig_hv - actual_hv))
    trend_mae = np.mean(np.abs(trend_hv - actual_hv))
    print(f"  R²: 原始={orig_r2:.4f}, 趋势约束={trend_r2:.4f}")
    print(f"  MAE: 原始={orig_mae:.1f} HV, 趋势约束={trend_mae:.1f} HV")

    # 腐蚀对比
    actual_corr = df_orig["腐蚀_实测_Acm2"].values
    orig_corr = df_orig["腐蚀_LOO预测_Acm2"].values
    trend_corr = df_trend["腐蚀_LOO趋势约束_Acm2"].values

    print(f"\n腐蚀对比:")
    print(f"{'功率(W)':>8} {'实测':>14} {'原始LOO':>14} {'趋势约束':>14}")
    print("-" * 55)
    for i in range(len(powers)):
        print(f"{powers[i]:>8.0f} {actual_corr[i]:>14.2e} {orig_corr[i]:>14.2e} "
              f"{trend_corr[i]:>14.2e}")

    orig_valley = np.argmin(orig_corr)
    trend_valley = np.argmin(trend_corr)
    actual_valley = np.argmin(actual_corr)
    print(f"\n  谷值: 原始={'✓' if orig_valley == actual_valley else '✗'}, "
          f"趋势约束={'✓' if trend_valley == actual_valley else '✗'}")

    # 保存对比
    compare_df = pd.DataFrame({
        "功率_W": powers,
        "硬度_实测": actual_hv,
        "硬度_原始LOO": orig_hv,
        "硬度_趋势约束LOO": trend_hv,
        "腐蚀_实测": actual_corr,
        "腐蚀_原始LOO": orig_corr,
        "腐蚀_趋势约束LOO": trend_corr,
    })
    compare_path = os.path.join(TREND_DIR, "trend_comparison.csv")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {compare_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    _ensure_dirs()

    # 1. 全量训练
    # 硬度: 趋势权重8.0, 不标准化
    # 腐蚀: 趋势权重2.0(标准化后), 跨谷权重1.0, Z-score标准化
    run_full_training(weight=50, hv_trend_weight=8.0,
                      corr_trend_weight=2.0, corr_cross_valley=1.0)

    # 2. LOO验证
    run_loo_trend(weight=50, hv_trend_weight=8.0,
                  corr_trend_weight=2.0, corr_cross_valley=1.0)

    # 3. 对比分析
    compare_results()

    print(f"\n{'=' * 80}")
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
