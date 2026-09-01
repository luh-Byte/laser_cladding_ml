"""
Few-shot 微调
==============
用少量目标材料的实验数据（Rockit485的4个实验点），以高权重加入训练集，
让模型在目标成分区域"校准"趋势，纠正硬度预测的方向错误。

原理：
  训练数据中，中速区的硬度-功率关系被成分差异"污染"。
  用少量固定成分（Rockit485）的真实数据，以高权重强制模型学习
  该成分下的正确趋势（硬度随功率先升后降）。

权重策略：
  Few-shot样本权重 = base_weight（默认20倍）
  可以理解为：把4个实验点当成4×20=80个等效样本。
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    PROCESS_FEATURES, COMPOSITION_FEATURES,
    CORRELATION_THRESHOLD, FEATURES_TO_REMOVE,
    MODEL_DIR, RESULT_DIR
)
from src.data import load_raw_data, clean_data
from src.features import (
    compute_derived_features, get_all_feature_names,
    filter_correlated_features
)
from src.metrics import save_model_bundle, load_model_bundle


# ============================================================
# Rockit485 实验数据（Few-shot 样本）
# ============================================================

ROCKIT485_COMPOSITION = {
    "C": 0.15, "Cr": 13.0, "Si": 0.6, "Ni": 4.0,
    "Fe": 78.95, "Mn": 0.5, "Mo": 2.8,
}

ROCKIT485_EXPERIMENTS = [
    {"激光功率": 1200, "扫描速度": 15, "送粉速率": 10, "光斑直径": 3.0, "离焦量": 0.0,
     "硬度": 470, "腐蚀电流": 5.9515e-7},
    {"激光功率": 1500, "扫描速度": 15, "送粉速率": 10, "光斑直径": 3.0, "离焦量": 0.0,
     "硬度": 500, "腐蚀电流": 3.1202e-7},
    {"激光功率": 1800, "扫描速度": 15, "送粉速率": 10, "光斑直径": 3.0, "离焦量": 0.0,
     "硬度": 520, "腐蚀电流": 3.4757e-7},
    {"激光功率": 2100, "扫描速度": 15, "送粉速率": 10, "光斑直径": 3.0, "离焦量": 0.0,
     "硬度": 480, "腐蚀电流": 4.5000e-7},
]


def prepare_fewshot_data():
    """将Rockit485实验数据转为DataFrame，含腐蚀电流*10000列"""
    rows = []
    for exp in ROCKIT485_EXPERIMENTS:
        row = {**ROCKIT485_COMPOSITION, **exp}
        row["腐蚀电流*10000"] = row["腐蚀电流"] * 10000.0
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def build_training_data(
    df_base: pd.DataFrame,
    df_fewshot: pd.DataFrame,
    fewshot_weight: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str], list[str]]:
    """
    构建加权训练数据。

    返回:
        X_train_scaled, y_hardness, y_corrosion_log, sample_weights,
        scaler, selected_features, all_feat_names
    """
    # 合并基础数据 + few-shot数据
    df_combined = pd.concat([df_base, df_fewshot], ignore_index=True)

    # 计算衍生特征
    df_feat = compute_derived_features(df_combined.copy())
    all_feat_names = get_all_feature_names()

    # 样本权重
    n_base = len(df_base)
    n_fewshot = len(df_fewshot)
    sample_weights = np.ones(n_base + n_fewshot)
    sample_weights[n_base:] = fewshot_weight  # fewshot样本高权重

    # 标准化（用基础数据的统计量，避免fewshot影响分布）
    X_all_raw = df_feat[all_feat_names].values
    X_base_raw = X_all_raw[:n_base]

    scaler = StandardScaler()
    scaler.fit(X_base_raw)  # 只用基础数据fit
    X_scaled = scaler.transform(X_all_raw)

    # 共线性筛选（用基础数据判断）
    X_base_scaled = X_scaled[:n_base]
    selected_features, _ = filter_correlated_features(
        X_base_scaled, all_feat_names, CORRELATION_THRESHOLD
    )

    # 特征精简（用基础数据训练RF判断重要性）
    # 这里跳过重要性剔除的复杂流程，直接用预定义的精简列表
    selected_features = [f for f in selected_features if f not in FEATURES_TO_REMOVE]

    feat_idx = [all_feat_names.index(f) for f in selected_features]
    X_final = X_scaled[:, feat_idx]

    # 目标变量
    y_hardness = df_combined["硬度"].to_numpy(dtype=float)

    icorr_10k = df_combined["腐蚀电流*10000"].to_numpy(dtype=float)
    icorr_10k = np.maximum(icorr_10k, 1e-15)
    y_corr_log = np.log10(icorr_10k)

    return X_final, y_hardness, y_corr_log, sample_weights, scaler, selected_features, all_feat_names


def train_fewshot_model(fewshot_weight=20, model_type="lgb", target="hardness",
                        X=None, y=None, sample_weights=None):
    """
    训练Few-shot微调模型。

    参数:
        fewshot_weight: Few-shot样本的权重倍数
        model_type: 'lgb' 或 'rf'
        target: 'hardness' 或 'corrosion'
        X, y, sample_weights: 训练数据
    """
    if model_type == "lgb":
        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            reg_lambda=10, random_state=42, verbose=-1, force_col_wise=True
        )
    else:
        model = RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )

    model.fit(X, y, sample_weight=sample_weights)  # type: ignore[arg-type]
    return model


def run_fewshot_experiment(fewshot_weight=20):
    """
    运行完整的Few-shot微调实验，输出验证结果。
    """
    print("=" * 80)
    print(f"Few-shot 微调（权重={fewshot_weight}x）")
    print("=" * 80)

    # 加载数据
    df = load_raw_data()
    df_base, _ = clean_data(df)
    df_fewshot = prepare_fewshot_data()

    print(f"\n[数据] 基础训练样本: {len(df_base)}")
    print(f"[数据] Few-shot样本: {len(df_fewshot)} (Rockit485)")
    print(f"[数据] Few-shot权重: {fewshot_weight}x")
    print(f"[数据] 等效总样本: {len(df_base) + len(df_fewshot) * fewshot_weight}")

    # 构建训练数据
    X, y_h, y_c_log, sw, scaler, sel_feat, all_feat = build_training_data(
        df_base, df_fewshot, fewshot_weight
    )  # type: ignore[misc]

    print(f"[特征] 保留特征数: {len(sel_feat)}")
    print(f"[特征] 列表: {sel_feat}")

    # ---- 硬度模型 ----
    print(f"\n{'='*60}")
    print("硬度模型训练（LightGBM + 加权）")
    print("=" * 60)

    h_model = train_fewshot_model(fewshot_weight, "lgb", "hardness", X, y_h, sw)

    # 用实验点自身验证（留一法更严谨，但只有4个点，直接看趋势）
    n_base = len(df_base)
    X_fewshot = X[n_base:]
    h_pred_fewshot = h_model.predict(X_fewshot)

    print(f"\n硬度趋势验证（Rockit485 @15mm/s）:")
    print(f"{'功率(W)':>8} {'实测(HV)':>10} {'预测(HV)':>10} {'误差':>10} {'相对误差':>10}")
    print("-" * 60)
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        actual = exp["硬度"]
        pred = h_pred_fewshot[i]  # type: ignore[index]
        err = pred - actual
        rel = abs(err) / actual * 100
        print(f"{exp['激光功率']:>8d} {actual:>10.1f} {pred:>10.1f} {err:>+10.1f} {rel:>9.1f}%")

    # 判断趋势是否正确
    peak_actual = np.argmax([e["硬度"] for e in ROCKIT485_EXPERIMENTS])
    peak_pred = np.argmax(h_pred_fewshot)  # type: ignore[arg-type]
    peak_power_actual = ROCKIT485_EXPERIMENTS[peak_actual]["激光功率"]
    peak_power_pred = ROCKIT485_EXPERIMENTS[peak_pred]["激光功率"]
    trend_correct = peak_actual == peak_pred

    print(f"\n  实测峰值: {peak_power_actual}W ({peak_actual+1}/4)")
    print(f"  预测峰值: {peak_power_pred}W ({peak_pred+1}/4)")
    print(f"  趋势正确: {'✓' if trend_correct else '✗'}")

    # ---- 腐蚀模型（LightGBM + 加权 + log空间） ----
    print(f"\n{'='*60}")
    print("腐蚀模型训练（LightGBM + 加权 + log空间）")
    print("=" * 60)

    # 腐蚀用Z-score
    corr_mean = np.mean(y_c_log[:n_base])  # 用基础数据的均值
    corr_std = np.std(y_c_log[:n_base])
    y_c_zscore = (y_c_log - corr_mean) / corr_std

    actual_corr = np.array([e["腐蚀电流"] for e in ROCKIT485_EXPERIMENTS])

    c_model = train_fewshot_model(fewshot_weight, "lgb", "corrosion", X, y_c_zscore, sw)

    c_pred_zscore = c_model.predict(X_fewshot)  # type: ignore[union-attr]
    c_pred_log = c_pred_zscore * corr_std + corr_mean  # type: ignore[operator]
    c_pred = np.power(10.0, c_pred_log) / 10000.0

    ratios = c_pred / actual_corr
    geo_mean_ratio = np.exp(np.mean(np.log(ratios)))
    cal_factor = 1.0 / geo_mean_ratio

    print(f"\n腐蚀趋势验证（Rockit485 @15mm/s）:")
    print(f"{'功率(W)':>8} {'实测(A/cm²)':>14} {'预测(A/cm²)':>14} {'倍数':>8}")
    print("-" * 55)
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        print(f"{exp['激光功率']:>8d} {actual_corr[i]:>14.2e} {c_pred[i]:>14.2e} {ratios[i]:>7.2f}x")

    print(f"\n  几何均值倍数: {geo_mean_ratio:.2f}x")
    print(f"  校准因子: {cal_factor:.4f}")

    # 趋势判断
    pred_peak = np.argmin(c_pred)
    actual_peak = np.argmin(actual_corr)
    print(f"  最优功率点: 预测={ROCKIT485_EXPERIMENTS[pred_peak]['激光功率']}W, "
          f"实测={ROCKIT485_EXPERIMENTS[actual_peak]['激光功率']}W")
    print(f"  趋势正确: {'✓' if pred_peak == actual_peak else '✗'}")

    # 保存模型bundle
    print(f"\n{'='*60}")
    print("保存微调模型")
    print("=" * 60)

    # 硬度bundle
    h_bundle = {
        "model": h_model,
        "scaler": scaler,
        "selected_features": sel_feat,
        "all_feature_names": all_feat,
        "target": "硬度",
        "model_type": "LightGBM",
        "fewshot_weight": fewshot_weight,
        "fewshot_n": len(df_fewshot),
        "corrosion_transformer": None,
    }
    save_model_bundle(h_bundle, "LightGBM_fewshot", "硬度")

    # 腐蚀bundle
    c_bundle = {
        "model": c_model,
        "scaler": scaler,
        "selected_features": sel_feat,
        "all_feature_names": all_feat,
        "target": "腐蚀电流",
        "model_type": "LightGBM",
        "fewshot_weight": fewshot_weight,
        "fewshot_n": len(df_fewshot),
        "corrosion_transformer": {
            "mean": corr_mean,
            "std": corr_std,
            "scaling_factor": 10000.0,
        },
        "calibration": {
            "factor": cal_factor,
            "info": {
                "material": "Rockit485",
                "method": "fewshot_geometric_mean",
                "note": "Few-shot微调(LightGBM)后的校准因子",
            },
            "applied": True,
        }
    }
    save_model_bundle(c_bundle, "LightGBM_fewshot", "腐蚀电流")

    print(f"\n[完成] Few-shot微调模型已保存")
    print(f"  硬度: bundle_LightGBM_fewshot_硬度.pkl")
    print(f"  腐蚀: bundle_LightGBM_fewshot_腐蚀电流.pkl（校准因子{cal_factor:.4f}）")

    # ---- SHAP 中间数据计算（仅数据，不绘图） ----
    print(f"\n{'='*60}")
    print("SHAP 中间数据计算")
    print("=" * 60)

    from src.shap import compute_fewshot_shap
    X_fs_df = pd.DataFrame(X, columns=sel_feat)
    compute_fewshot_shap(
        h_model, c_model,
        X_fs_df, sel_feat,
        rockit_start_idx=n_base,
        rockit_n=len(df_fewshot),
    )

    return h_model, c_model, scaler, sel_feat, all_feat, cal_factor


if __name__ == "__main__":
    run_fewshot_experiment(fewshot_weight=20)
