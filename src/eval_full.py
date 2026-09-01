"""
全量数据模型评估
================
用全量数据（196条基础 + 4条Rockit485，权重20x）训练的Few-shot模型，
对4个实验点进行预测，计算 R²、MAE、RMSE 指标。

输出:
  outputs/full_data_eval/
    full_data_eval_results.csv  — 预测详情
    full_data_metrics.csv       — 指标汇总
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.data import load_raw_data, clean_data
from src.features import compute_derived_features, get_all_feature_names
from src.config import OUTPUT_DIR
from src.fewshot import (
    ROCKIT485_EXPERIMENTS, prepare_fewshot_data,
    build_training_data, train_fewshot_model,
)

EVAL_DIR = os.path.join(OUTPUT_DIR, "full_data_eval")
os.makedirs(EVAL_DIR, exist_ok=True)


def run_full_data_eval(fewshot_weight=20):
    """
    全量数据训练 → 预测4个实验点 → 计算R²/MAE/RMSE。
    """
    print("=" * 70)
    print("全量数据模型评估（Few-shot权重=20x）")
    print("=" * 70)

    # 加载数据
    df = load_raw_data()
    df_base, _ = clean_data(df)
    df_fewshot = prepare_fewshot_data()

    n_base = len(df_base)
    n_fewshot = len(df_fewshot)
    print(f"\n[数据] 基础样本: {n_base}")
    print(f"[数据] Few-shot样本: {n_fewshot} (Rockit485)")
    print(f"[数据] 权重: {fewshot_weight}x (等效{len(df_fewshot) * fewshot_weight}条)")

    # 构建训练数据（全量：196基础 + 4实验，权重20x）
    X, y_h, y_c_log, sw, scaler, sel_feat, all_feat = build_training_data(
        df_base, df_fewshot, fewshot_weight
    )

    print(f"[特征] 保留特征数: {len(sel_feat)}")
    print(f"[特征] 列表: {sel_feat}")

    # 提取4个实验点的特征（训练集中最后4行）
    X_rockit = X[n_base:]
    actual_hv = np.array([e["硬度"] for e in ROCKIT485_EXPERIMENTS])
    actual_corr = np.array([e["腐蚀电流"] for e in ROCKIT485_EXPERIMENTS])
    powers = [e["激光功率"] for e in ROCKIT485_EXPERIMENTS]

    # ---- 训练硬度模型 ----
    print(f"\n{'='*60}")
    print("训练硬度模型（LightGBM + 全量数据加权）")
    print("=" * 60)
    h_model = train_fewshot_model(fewshot_weight, "lgb", "hardness", X, y_h, sw)
    hv_pred = np.asarray(h_model.predict(X_rockit), dtype=float)

    # ---- 训练腐蚀模型（Z-score + log空间） ----
    print(f"\n{'='*60}")
    print("训练腐蚀模型（LightGBM + 全量数据加权 + Z-score）")
    print("=" * 60)
    corr_mean = np.mean(y_c_log[:n_base])
    corr_std = np.std(y_c_log[:n_base])
    y_c_zscore = (y_c_log - corr_mean) / corr_std
    c_model = train_fewshot_model(fewshot_weight, "lgb", "corrosion", X, y_c_zscore, sw)

    c_pred_z = np.asarray(c_model.predict(X_rockit), dtype=float)
    c_pred_log = c_pred_z * corr_std + corr_mean
    c_pred = np.power(10.0, c_pred_log) / 10000.0

    # ---- 计算指标 ----
    def calc_metrics(y_true, y_pred, label):
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        mape = np.mean(np.abs(y_pred - y_true) / y_true) * 100
        print(f"\n  [{label}] R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape}

    print(f"\n{'='*60}")
    print("评估指标")
    print("=" * 60)

    hv_metrics = calc_metrics(actual_hv, hv_pred, "硬度")
    # 腐蚀在log空间评估（与训练一致）
    actual_corr_log = np.log10(actual_corr * 10000.0)
    pred_corr_log = np.log10(c_pred * 10000.0)
    corr_metrics = calc_metrics(actual_corr_log, pred_corr_log, "腐蚀电流(log空间)")

    # ---- 预测详情表 ----
    print(f"\n{'='*60}")
    print("预测详情")
    print("=" * 60)

    print(f"\n硬度:")
    print(f"{'功率(W)':>8} {'实测(HV)':>10} {'预测(HV)':>10} {'误差':>10} {'相对误差':>10}")
    print("-" * 55)
    for i in range(len(powers)):
        err = hv_pred[i] - actual_hv[i]
        rel = abs(err) / actual_hv[i] * 100
        print(f"{powers[i]:>8d} {actual_hv[i]:>10.1f} {hv_pred[i]:>10.1f} {err:>+10.1f} {rel:>9.1f}%")

    print(f"\n腐蚀电流:")
    print(f"{'功率(W)':>8} {'实测(A/cm²)':>14} {'预测(A/cm²)':>14} {'倍数':>8}")
    print("-" * 50)
    for i in range(len(powers)):
        ratio = c_pred[i] / actual_corr[i]
        print(f"{powers[i]:>8d} {actual_corr[i]:>14.2e} {c_pred[i]:>14.2e} {ratio:>7.2f}x")

    # ---- 保存结果 ----
    results_df = pd.DataFrame({
        "功率_W": powers,
        "硬度_实测_HV": actual_hv,
        "硬度_预测_HV": hv_pred,
        "硬度_绝对误差": hv_pred - actual_hv,
        "硬度_相对误差_pct": np.abs(hv_pred - actual_hv) / actual_hv * 100,
        "腐蚀_实测_Acm2": actual_corr,
        "腐蚀_预测_Acm2": c_pred,
        "腐蚀_倍数": c_pred / actual_corr,
    })

    metrics_df = pd.DataFrame([
        {"目标": "硬度", "R²": hv_metrics["R²"], "MAE": hv_metrics["MAE"],
         "RMSE": hv_metrics["RMSE"], "MAPE(%)": hv_metrics["MAPE(%)"]},
        {"目标": "腐蚀电流(log)", "R²": corr_metrics["R²"], "MAE": corr_metrics["MAE"],
         "RMSE": corr_metrics["RMSE"], "MAPE(%)": corr_metrics["MAPE(%)"]},
    ])

    results_path = os.path.join(EVAL_DIR, "full_data_eval_results.csv")
    metrics_path = os.path.join(EVAL_DIR, "full_data_metrics.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"\n[保存] 预测详情: {results_path}")
    print(f"[保存] 指标汇总: {metrics_path}")

    return results_df, metrics_df


if __name__ == "__main__":
    run_full_data_eval(fewshot_weight=20)
