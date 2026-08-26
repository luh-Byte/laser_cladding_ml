"""
Rockit485 留一法（Leave-One-Out, LOO）验证
==============================================
对Few-shot微调模型进行最严谨的泛化性能评估。

方法：
  4个Rockit485实验点，循环4次：
    第i次：用除第i点外的3个点做Few-shot微调 → 预测第i个点
  汇总4次预测结果，计算LOO指标。

对比组：
  1. 基线模型（无微调，直接用196条训练的RF模型）
  2. Few-shot微调（本文方法，3个点微调 → 预测第4个点）

输出：
  outputs/loo_validation/
    loo_results.csv          — 4个点的预测详情（基线+Few-shot）
    loo_metrics_summary.csv  — LOO指标汇总
    loo_trend_analysis.csv   — 趋势分析
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    PROCESS_FEATURES, COMPOSITION_FEATURES,
    CORRELATION_THRESHOLD, FEATURES_TO_REMOVE,
    MODEL_DIR, RESULT_DIR, OUTPUT_DIR
)
from src.data_preprocessing import load_raw_data, clean_data
from src.feature_engineering import (
    compute_derived_features, get_all_feature_names,
    filter_correlated_features
)
from src.evaluation import load_model_bundle
from src.fewshot_finetune import (
    ROCKIT485_COMPOSITION, ROCKIT485_EXPERIMENTS,
    prepare_fewshot_data, build_training_data,
    train_fewshot_model
)


# ============================================================
# 输出目录
# ============================================================

LOO_DIR = os.path.join(OUTPUT_DIR, "loo_validation")


def _ensure_dir():
    os.makedirs(LOO_DIR, exist_ok=True)


# ============================================================
# LOO 核心函数
# ============================================================

def run_loo_fewshot(fewshot_weight=20):
    """
    执行Rockit485的留一法验证。

    返回:
        loo_df: DataFrame, 每个点的真实值和预测值（基线 + Few-shot）
        metrics: dict, LOO指标汇总
    """
    print("=" * 80)
    print("Rockit485 留一法（LOO）验证")
    print("=" * 80)
    print(f"Few-shot权重: {fewshot_weight}x")
    print(f"验证方式: 4折LOO（每次留1个点做测试，用其余3个微调）")

    # 加载基础数据
    df = load_raw_data()
    df_base, _ = clean_data(df)
    df_fewshot_all = prepare_fewshot_data()

    n_base = len(df_base)
    n_fewshot = len(df_fewshot_all)  # 4
    all_feat_names = get_all_feature_names()

    print(f"\n[数据] 基础训练样本: {n_base}")
    print(f"[数据] Few-shot总样本: {n_fewshot} (Rockit485)")

    # 存储每次LOO的预测结果
    results = []

    # ---- 1. 基线模型预测（无任何微调，用196条基础数据训练RF） ----
    print(f"\n{'='*60}")
    print("基线模型预测（无微调，RF模型，196条基础数据训练）")
    print("=" * 60)

    # 用基础数据训练基线RF模型（与fewshot同特征管道，保证公平对比）
    df_empty_fs = df_fewshot_all.iloc[0:0].copy()  # 空的fewshot表，仅用于管道复用
    X_base_all, y_h_base_all, y_c_base_all, sw_base_all, scaler_base, sel_feat_base, all_feat_base = \
        build_training_data(df_base, df_empty_fs, fewshot_weight=1)  # type: ignore[misc]

    # 训练基线RF硬度模型
    from sklearn.ensemble import RandomForestRegressor
    hv_base_model = RandomForestRegressor(
        n_estimators=150, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    hv_base_model.fit(X_base_all, y_h_base_all)

    # 训练基线RF腐蚀模型（Z-score）
    corr_base_mean = np.mean(y_c_base_all)
    corr_base_std = np.std(y_c_base_all)
    y_c_base_zscore = (y_c_base_all - corr_base_mean) / corr_base_std

    corr_base_model = RandomForestRegressor(
        n_estimators=150, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    corr_base_model.fit(X_base_all, y_c_base_zscore)

    # Rockit485的特征矩阵（用基线scaler和sel_feat）
    df_rockit_feat = compute_derived_features(df_fewshot_all.copy())
    X_rockit_raw = df_rockit_feat[all_feat_base].values
    X_rockit_scaled = scaler_base.transform(X_rockit_raw)
    feat_idx_base = [all_feat_base.index(f) for f in sel_feat_base]
    X_rockit = X_rockit_scaled[:, feat_idx_base]

    # 预测
    hv_pred_baseline = hv_base_model.predict(X_rockit)

    corr_pred_zscore = corr_base_model.predict(X_rockit)
    corr_pred_log = corr_pred_zscore * corr_base_std + corr_base_mean
    corr_pred_baseline = np.power(10.0, corr_pred_log) / 10000.0

    print(f"\n基线模型硬度预测:")
    print(f"{'功率(W)':>8} {'实测(HV)':>10} {'基线预测(HV)':>14} {'相对误差':>10}")
    print("-" * 55)
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        rel = abs(hv_pred_baseline[i] - exp["硬度"]) / exp["硬度"] * 100
        print(f"{exp['激光功率']:>8d} {exp['硬度']:>10.1f} {hv_pred_baseline[i]:>14.1f} {rel:>9.1f}%")

    print(f"\n基线模型腐蚀预测:")
    print(f"{'功率(W)':>8} {'实测(A/cm²)':>14} {'基线预测(A/cm²)':>18} {'倍数':>8}")
    print("-" * 55)
    actual_corr = np.array([e["腐蚀电流"] for e in ROCKIT485_EXPERIMENTS])
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        ratio = corr_pred_baseline[i] / actual_corr[i]
        print(f"{exp['激光功率']:>8d} {actual_corr[i]:>14.2e} {corr_pred_baseline[i]:>18.2e} {ratio:>7.2f}x")

    # ---- 2. LOO: Few-shot微调预测 ----
    print(f"\n{'='*60}")
    print("LOO: Few-shot微调（每次留1个点，用其余3个微调）")
    print("=" * 60)

    hv_pred_loo = np.zeros(n_fewshot)
    corr_pred_loo = np.zeros(n_fewshot)

    for test_idx in range(n_fewshot):
        print(f"\n--- LOO Fold {test_idx+1}/4: 测试点 = {ROCKIT485_EXPERIMENTS[test_idx]['激光功率']}W ---")

        # 训练用的3个点
        train_indices = [i for i in range(n_fewshot) if i != test_idx]
        df_fewshot_train = df_fewshot_all.iloc[train_indices].reset_index(drop=True)
        print(f"  训练Few-shot点: {[ROCKIT485_EXPERIMENTS[i]['激光功率'] for i in train_indices]} W")

        # 构建训练数据
        X_train, y_h_train, y_c_train, sw_train, scaler, sel_feat, all_feat = build_training_data(
            df_base, df_fewshot_train, fewshot_weight
        )  # type: ignore[misc]

        n_train_base = len(df_base)
        n_train_fs = len(df_fewshot_train)  # 3

        # 训练硬度模型
        h_model = train_fewshot_model(fewshot_weight, "lgb", "hardness",
                                       X_train, y_h_train, sw_train)

        # 训练腐蚀模型（Z-score）
        corr_mean = np.mean(y_c_train[:n_train_base])
        corr_std = np.std(y_c_train[:n_train_base])
        y_c_zscore = (y_c_train - corr_mean) / corr_std

        c_model = train_fewshot_model(fewshot_weight, "lgb", "corrosion",
                                       X_train, y_c_zscore, sw_train)

        # 构建测试点的特征
        df_test = df_fewshot_all.iloc[[test_idx]].reset_index(drop=True)
        df_test_feat = compute_derived_features(df_test.copy())
        X_test_raw = df_test_feat[all_feat].values
        X_test_scaled = scaler.transform(X_test_raw)

        feat_idx = [all_feat.index(f) for f in sel_feat]
        X_test = X_test_scaled[:, feat_idx]

        # 预测硬度
        hv_pred = h_model.predict(X_test)[0]  # type: ignore[index]
        hv_pred_loo[test_idx] = hv_pred

        # 预测腐蚀
        c_pred_z = c_model.predict(X_test)[0]  # type: ignore[index]
        c_pred_log = c_pred_z * corr_std + corr_mean
        c_pred = np.power(10.0, c_pred_log) / 10000.0
        corr_pred_loo[test_idx] = c_pred

        print(f"  硬度: 实测={ROCKIT485_EXPERIMENTS[test_idx]['硬度']:.0f}HV, "
              f"预测={hv_pred:.1f}HV, "
              f"误差={hv_pred - ROCKIT485_EXPERIMENTS[test_idx]['硬度']:+.1f}HV")
        print(f"  腐蚀: 实测={actual_corr[test_idx]:.2e}, "
              f"预测={c_pred:.2e}, "
              f"倍数={c_pred/actual_corr[test_idx]:.2f}x")

    # ---- 3. 汇总结果 ----
    print(f"\n{'='*60}")
    print("LOO 结果汇总")
    print("=" * 60)

    actual_hv = np.array([e["硬度"] for e in ROCKIT485_EXPERIMENTS])
    powers = [e["激光功率"] for e in ROCKIT485_EXPERIMENTS]

    # 构建结果DataFrame
    loo_df = pd.DataFrame({
        "功率_W": powers,
        "硬度_实测_HV": actual_hv,
        "硬度_基线预测_HV": hv_pred_baseline,
        "硬度_LOO预测_HV": hv_pred_loo,
        "腐蚀_实测_Acm2": actual_corr,
        "腐蚀_基线预测_Acm2": corr_pred_baseline,
        "腐蚀_LOO预测_Acm2": corr_pred_loo,
    })

    # ---- 4. 计算指标 ----
    metrics = _compute_metrics(actual_hv, hv_pred_baseline, hv_pred_loo,
                                actual_corr, corr_pred_baseline, corr_pred_loo,
                                powers)

    # ---- 5. 保存结果 ----
    _ensure_dir()
    loo_path = os.path.join(LOO_DIR, "loo_results.csv")
    loo_df.to_csv(loo_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] LOO详细结果: {loo_path}")

    metrics_path = os.path.join(LOO_DIR, "loo_metrics_summary.csv")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"[保存] LOO指标汇总: {metrics_path}")

    # ---- 6. 打印汇总表 ----
    _print_summary(loo_df, metrics, actual_hv, actual_corr, powers)

    return loo_df, metrics


def _compute_metrics(actual_hv, hv_base, hv_loo,
                     actual_corr, corr_base, corr_loo,
                     powers):
    """
    计算LOO各项指标。

    返回:
        DataFrame，行=指标，列=基线/LOO/提升
    """
    # 硬度指标
    hv_mae_base = np.mean(np.abs(hv_base - actual_hv))
    hv_mape_base = np.mean(np.abs(hv_base - actual_hv) / actual_hv) * 100
    hv_r2_base = 1 - np.sum((actual_hv - hv_base) ** 2) / np.sum((actual_hv - np.mean(actual_hv)) ** 2)

    hv_mae_loo = np.mean(np.abs(hv_loo - actual_hv))
    hv_mape_loo = np.mean(np.abs(hv_loo - actual_hv) / actual_hv) * 100
    hv_r2_loo = 1 - np.sum((actual_hv - hv_loo) ** 2) / np.sum((actual_hv - np.mean(actual_hv)) ** 2)

    # 腐蚀指标（log空间）
    actual_corr_log = np.log10(actual_corr * 10000.0)
    corr_base_log = np.log10(corr_base * 10000.0)
    corr_loo_log = np.log10(corr_loo * 10000.0)

    corr_r2_base = 1 - np.sum((actual_corr_log - corr_base_log) ** 2) / np.sum((actual_corr_log - np.mean(actual_corr_log)) ** 2)
    corr_r2_loo = 1 - np.sum((actual_corr_log - corr_loo_log) ** 2) / np.sum((actual_corr_log - np.mean(actual_corr_log)) ** 2)

    # GMR
    gmr_base = np.exp(np.mean(np.log(corr_base / actual_corr)))
    gmr_loo = np.exp(np.mean(np.log(corr_loo / actual_corr)))

    # 趋势判断
    peak_actual_idx = np.argmax(actual_hv)
    peak_base_idx = np.argmax(hv_base)
    peak_loo_idx = np.argmax(hv_loo)

    best_corr_actual_idx = np.argmin(actual_corr)
    best_corr_base_idx = np.argmin(corr_base)
    best_corr_loo_idx = np.argmin(corr_loo)

    # 构建指标表
    rows = [
        # 硬度
        {"目标": "硬度", "指标": "MAE (HV)", "基线": round(hv_mae_base, 2),
         "LOO_Fewshot": round(hv_mae_loo, 2),
         "相对变化": f"{(hv_mae_loo - hv_mae_base) / hv_mae_base * 100:+.1f}%"},
        {"目标": "硬度", "指标": "MAPE (%)", "基线": round(hv_mape_base, 2),
         "LOO_Fewshot": round(hv_mape_loo, 2),
         "相对变化": f"{(hv_mape_loo - hv_mape_base) / hv_mape_base * 100:+.1f}%"},
        {"目标": "硬度", "指标": "R²", "基线": round(hv_r2_base, 4),
         "LOO_Fewshot": round(hv_r2_loo, 4),
         "相对变化": f"{(hv_r2_loo - hv_r2_base):+.4f}"},
        {"目标": "硬度", "指标": "峰值功率点 (W)", "基线": powers[peak_base_idx],
         "LOO_Fewshot": powers[peak_loo_idx],
         "相对变化": f"实测={powers[peak_actual_idx]}W"},
        {"目标": "硬度", "指标": "趋势正确", "基线": "✓" if peak_base_idx == peak_actual_idx else "✗",
         "LOO_Fewshot": "✓" if peak_loo_idx == peak_actual_idx else "✗",
         "相对变化": "—"},
        # 腐蚀
        {"目标": "腐蚀电流", "指标": "log空间 R²", "基线": round(corr_r2_base, 4),
         "LOO_Fewshot": round(corr_r2_loo, 4),
         "相对变化": f"{(corr_r2_loo - corr_r2_base):+.4f}"},
        {"目标": "腐蚀电流", "指标": "几何均值倍数 (GMR)", "基线": round(gmr_base, 4),
         "LOO_Fewshot": round(gmr_loo, 4),
         "相对变化": f"{(gmr_loo - gmr_base) / gmr_base * 100:+.1f}%"},
        {"目标": "腐蚀电流", "指标": "最优功率点 (W)", "基线": powers[best_corr_base_idx],
         "LOO_Fewshot": powers[best_corr_loo_idx],
         "相对变化": f"实测={powers[best_corr_actual_idx]}W"},
        {"目标": "腐蚀电流", "指标": "趋势正确",
         "基线": "✓" if best_corr_base_idx == best_corr_actual_idx else "✗",
         "LOO_Fewshot": "✓" if best_corr_loo_idx == best_corr_actual_idx else "✗",
         "相对变化": "—"},
    ]

    return pd.DataFrame(rows)


def _print_summary(loo_df, metrics, actual_hv, actual_corr, powers):
    """打印美观的汇总表"""

    print(f"\n{'='*80}")
    print("硬度 LOO 对比（基线 vs Few-shot）")
    print("=" * 80)
    print(f"{'功率(W)':>8} {'实测(HV)':>10} {'基线(HV)':>10} {'基线误差%':>10} "
          f"{'LOO(HV)':>10} {'LOO误差%':>10} {'改善%':>10}")
    print("-" * 80)
    for i in range(len(powers)):
        base_err = abs(loo_df["硬度_基线预测_HV"].iloc[i] - actual_hv[i]) / actual_hv[i] * 100
        loo_err = abs(loo_df["硬度_LOO预测_HV"].iloc[i] - actual_hv[i]) / actual_hv[i] * 100
        improvement = base_err - loo_err
        print(f"{powers[i]:>8d} {actual_hv[i]:>10.1f} "
              f"{loo_df['硬度_基线预测_HV'].iloc[i]:>10.1f} {base_err:>9.1f}% "
              f"{loo_df['硬度_LOO预测_HV'].iloc[i]:>10.1f} {loo_err:>9.1f}% "
              f"{improvement:>+9.1f}%")

    print(f"\n{'='*80}")
    print("腐蚀电流 LOO 对比（基线 vs Few-shot）")
    print("=" * 80)
    print(f"{'功率(W)':>8} {'实测':>14} {'基线':>14} {'基线倍数':>10} "
          f"{'LOO':>14} {'LOO倍数':>10}")
    print("-" * 80)
    for i in range(len(powers)):
        base_ratio = loo_df["腐蚀_基线预测_Acm2"].iloc[i] / actual_corr[i]
        loo_ratio = loo_df["腐蚀_LOO预测_Acm2"].iloc[i] / actual_corr[i]
        print(f"{powers[i]:>8d} {actual_corr[i]:>14.2e} "
              f"{loo_df['腐蚀_基线预测_Acm2'].iloc[i]:>14.2e} {base_ratio:>9.2f}x "
              f"{loo_df['腐蚀_LOO预测_Acm2'].iloc[i]:>14.2e} {loo_ratio:>9.2f}x")

    print(f"\n{'='*80}")
    print("LOO 指标汇总")
    print("=" * 80)
    for _, row in metrics.iterrows():
        print(f"  [{row['目标']}] {row['指标']}: "
              f"基线={row['基线']}, LOO={row['LOO_Fewshot']}, 变化={row['相对变化']}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    run_loo_fewshot(fewshot_weight=20)
