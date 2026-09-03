"""
生成趋势约束模型的评估数据
=========================
1. 用趋势约束模型生成全量训练预测 → 更新 full_data_eval_results.csv
2. 合并原始LOO + 趋势约束LOO → 更新 loo_results.csv (增加趋势约束列)
3. 保存趋势对比数据供新图使用
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR
from src.data import load_raw_data, clean_data
from src.features import compute_derived_features
from src.fewshot import (
    ROCKIT485_EXPERIMENTS,
    prepare_fewshot_data, build_training_data,
)
from src.trend_constrained_train import train_trend_model

EVAL_DIR = os.path.join(OUTPUT_DIR, "full_data_eval")
LOO_DIR = os.path.join(OUTPUT_DIR, "loo_validation")
TREND_DIR = os.path.join(OUTPUT_DIR, "trend_constrained")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def generate_full_training_eval():
    """
    用趋势约束模型对4个Rockit485点做全量训练预测,
    保存为 full_data_eval_results.csv
    """
    print("=" * 60)
    print("生成趋势约束模型评估数据")
    print("=" * 60)

    df = load_raw_data()
    df_base, _ = clean_data(df)
    df_fewshot = prepare_fewshot_data()

    n_base = len(df_base)
    n_fs = len(df_fewshot)

    # 构建训练数据
    X, y_h, y_c_log, sw, scaler, sel_feat, all_feat = \
        build_training_data(df_base, df_fewshot, fewshot_weight=50)

    # 硬度模型 (峰值在idx=2, 1800W, 不标准化)
    h_model, _ = train_trend_model(
        X, y_h, sw, n_base, n_fs, peak_idx=2,
        trend_type='peak', trend_weight=8.0,
        max_depth=5, n_estimators=300
    )

    # 腐蚀模型 (谷值在idx=1, 1500W, Z-score标准化 + 跨谷约束)
    c_model, norm_params = train_trend_model(
        X, y_c_log, sw, n_base, n_fs, peak_idx=1,
        trend_type='valley', trend_weight=2.0,
        max_depth=5, n_estimators=300,
        cross_valley_weight=1.0,
        normalize=True
    )

    # 预测4个实验点
    hv_pred = h_model.predict(X[n_base:])
    c_pred_norm = c_model.predict(X[n_base:])
    if norm_params:
        c_pred_log = c_pred_norm * norm_params[1] + norm_params[0]
    else:
        c_pred_log = c_pred_norm
    c_pred = np.power(10.0, c_pred_log) / 10000.0

    # 构建DataFrame
    rows = []
    for i, exp in enumerate(ROCKIT485_EXPERIMENTS):
        hv_actual = exp["硬度"]
        corr_actual = exp["腐蚀电流"]
        rows.append({
            "功率_W": exp["激光功率"],
            "硬度_实测_HV": hv_actual,
            "硬度_预测_HV": round(hv_pred[i], 2),
            "硬度_绝对误差": round(hv_pred[i] - hv_actual, 2),
            "硬度_相对误差_pct": round((hv_pred[i] - hv_actual) / hv_actual * 100, 4),
            "腐蚀_实测_Acm2": corr_actual,
            "腐蚀_预测_Acm2": c_pred[i],
            "腐蚀_倍数": round(c_pred[i] / corr_actual, 4),
        })

    df_eval = pd.DataFrame(rows)
    eval_path = os.path.join(EVAL_DIR, "full_data_eval_results.csv")
    df_eval.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {eval_path}")

    # 打印结果
    print(f"\n{'功率(W)':>8} {'硬度实测':>8} {'硬度预测':>8} {'误差%':>8} "
          f"{'腐蚀实测':>14} {'腐蚀预测':>14} {'倍数':>8}")
    print("-" * 75)
    for _, r in df_eval.iterrows():
        print(f"{r['功率_W']:>8.0f} {r['硬度_实测_HV']:>8.1f} "
              f"{r['硬度_预测_HV']:>8.1f} {r['硬度_相对误差_pct']:>+7.2f}% "
              f"{r['腐蚀_实测_Acm2']:>14.2e} {r['腐蚀_预测_Acm2']:>14.2e} "
              f"{r['腐蚀_倍数']:>7.2f}x")

    # 趋势判断
    actual_hv = np.array([e["硬度"] for e in ROCKIT485_EXPERIMENTS])
    actual_corr = np.array([e["腐蚀电流"] for e in ROCKIT485_EXPERIMENTS])
    peak_a, peak_p = np.argmax(actual_hv), np.argmax(hv_pred)
    valley_a, valley_p = np.argmin(actual_corr), np.argmin(c_pred)
    print(f"\n  硬度峰值: 实测={ROCKIT485_EXPERIMENTS[peak_a]['激光功率']}W, "
          f"预测={ROCKIT485_EXPERIMENTS[peak_p]['激光功率']}W, "
          f"{'✓' if peak_a == peak_p else '✗'}")
    print(f"  腐蚀谷值: 实测={ROCKIT485_EXPERIMENTS[valley_a]['激光功率']}W, "
          f"预测={ROCKIT485_EXPERIMENTS[valley_p]['激光功率']}W, "
          f"{'✓' if valley_a == valley_p else '✗'}")

    return df_eval


def generate_loo_comparison():
    """
    合并原始LOO + 趋势约束LOO, 保存对比数据
    用于绘图和新图
    """
    print(f"\n{'=' * 60}")
    print("生成LOO对比数据")
    print("=" * 60)

    # 原始LOO
    orig_path = os.path.join(LOO_DIR, "loo_results.csv")
    df_orig = pd.read_csv(orig_path)

    # 趋势约束LOO
    trend_path = os.path.join(TREND_DIR, "loo_trend_results.csv")
    df_trend = pd.read_csv(trend_path)

    # 合并
    df_compare = pd.DataFrame({
        "功率_W": df_orig["功率_W"],
        "硬度_实测_HV": df_orig["硬度_实测_HV"],
        "硬度_原始LOO_HV": df_orig["硬度_LOO预测_HV"],
        "硬度_趋势约束LOO_HV": df_trend["硬度_LOO趋势约束_HV"],
        "腐蚀_实测_Acm2": df_orig["腐蚀_实测_Acm2"],
        "腐蚀_原始LOO_Acm2": df_orig["腐蚀_LOO预测_Acm2"],
        "腐蚀_趋势约束LOO_Acm2": df_trend["腐蚀_LOO趋势约束_Acm2"],
    })

    # 计算误差
    df_compare["硬度_原始LOO误差"] = df_compare["硬度_原始LOO_HV"] - df_compare["硬度_实测_HV"]
    df_compare["硬度_趋势约束LOO误差"] = df_compare["硬度_趋势约束LOO_HV"] - df_compare["硬度_实测_HV"]
    df_compare["腐蚀_原始LOO倍数"] = df_compare["腐蚀_原始LOO_Acm2"] / df_compare["腐蚀_实测_Acm2"]
    df_compare["腐蚀_趋势约束LOO倍数"] = df_compare["腐蚀_趋势约束LOO_Acm2"] / df_compare["腐蚀_实测_Acm2"]

    # 保存
    compare_path = os.path.join(TREND_DIR, "loo_full_comparison.csv")
    df_compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"[保存] {compare_path}")

    # 打印
    print(f"\n{'功率(W)':>8} {'硬度实测':>8} {'原始LOO':>8} {'趋势约束':>8} "
          f"{'腐蚀实测':>14} {'原始LOO':>14} {'趋势约束':>14}")
    print("-" * 85)
    for _, r in df_compare.iterrows():
        print(f"{r['功率_W']:>8.0f} {r['硬度_实测_HV']:>8.1f} "
              f"{r['硬度_原始LOO_HV']:>8.1f} {r['硬度_趋势约束LOO_HV']:>8.1f} "
              f"{r['腐蚀_实测_Acm2']:>14.2e} {r['腐蚀_原始LOO_Acm2']:>14.2e} "
              f"{r['腐蚀_趋势约束LOO_Acm2']:>14.2e}")

    # 趋势对比
    actual_hv = df_compare["硬度_实测_HV"].values
    orig_hv = df_compare["硬度_原始LOO_HV"].values
    trend_hv = df_compare["硬度_趋势约束LOO_HV"].values
    actual_corr = df_compare["腐蚀_实测_Acm2"].values
    orig_corr = df_compare["腐蚀_原始LOO_Acm2"].values
    trend_corr = df_compare["腐蚀_趋势约束LOO_Acm2"].values

    print(f"\n  硬度峰值: 原始={df_compare['功率_W'].iloc[np.argmax(orig_hv)]:.0f}W "
          f"({'✓' if np.argmax(orig_hv)==np.argmax(actual_hv) else '✗'}), "
          f"趋势约束={df_compare['功率_W'].iloc[np.argmax(trend_hv)]:.0f}W "
          f"({'✓' if np.argmax(trend_hv)==np.argmax(actual_hv) else '✗'})")
    print(f"  腐蚀谷值: 原始={df_compare['功率_W'].iloc[np.argmin(orig_corr)]:.0f}W "
          f"({'✓' if np.argmin(orig_corr)==np.argmin(actual_corr) else '✗'}), "
          f"趋势约束={df_compare['功率_W'].iloc[np.argmin(trend_corr)]:.0f}W "
          f"({'✓' if np.argmin(trend_corr)==np.argmin(actual_corr) else '✗'})")

    # LOO指标
    hv_r2_orig = 1 - np.sum((actual_hv - orig_hv)**2) / np.sum((actual_hv - np.mean(actual_hv))**2)
    hv_r2_trend = 1 - np.sum((actual_hv - trend_hv)**2) / np.sum((actual_hv - np.mean(actual_hv))**2)
    hv_mae_orig = np.mean(np.abs(orig_hv - actual_hv))
    hv_mae_trend = np.mean(np.abs(trend_hv - actual_hv))

    print(f"\n  硬度 R²: 原始={hv_r2_orig:.4f}, 趋势约束={hv_r2_trend:.4f}")
    print(f"  硬度 MAE: 原始={hv_mae_orig:.1f} HV, 趋势约束={hv_mae_trend:.1f} HV")

    return df_compare


def generate_metrics(df_eval, df_compare):
    """
    生成指标汇总文件:
    1. 更新 full_data_metrics.csv (趋势约束模型, 4个Rockit485点)
    2. 生成 loo_metrics_comparison.csv (三列: 基线/原始LOO/趋势约束LOO)
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    print(f"\n{'=' * 60}")
    print("生成指标汇总")
    print("=" * 60)

    # --- 1. 全量训练指标 ---
    hv_actual = df_eval["硬度_实测_HV"].values
    hv_pred = df_eval["硬度_预测_HV"].values
    corr_actual = df_eval["腐蚀_实测_Acm2"].values
    corr_pred = df_eval["腐蚀_预测_Acm2"].values

    corr_actual_log = np.log10(corr_actual * 10000)
    corr_pred_log = np.log10(corr_pred * 10000)

    hv_r2 = r2_score(hv_actual, hv_pred)
    hv_mae = mean_absolute_error(hv_actual, hv_pred)
    hv_rmse = np.sqrt(mean_squared_error(hv_actual, hv_pred))
    hv_mape = np.mean(np.abs((hv_pred - hv_actual) / hv_actual)) * 100

    corr_r2 = r2_score(corr_actual_log, corr_pred_log)
    corr_mae = mean_absolute_error(corr_actual_log, corr_pred_log)
    corr_rmse = np.sqrt(mean_squared_error(corr_actual_log, corr_pred_log))
    corr_mape = np.mean(np.abs(corr_pred / corr_actual - 1)) * 100

    df_metrics = pd.DataFrame({
        "目标": ["硬度", "腐蚀电流(log)"],
        "R²": [hv_r2, corr_r2],
        "MAE": [hv_mae, corr_mae],
        "RMSE": [hv_rmse, corr_rmse],
        "MAPE(%)": [hv_mape, corr_mape],
    })
    metrics_path = os.path.join(EVAL_DIR, "full_data_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"[保存] {metrics_path}")
    print(f"  硬度: R²={hv_r2:.4f}, MAE={hv_mae:.2f} HV, RMSE={hv_rmse:.2f}, MAPE={hv_mape:.2f}%")
    print(f"  腐蚀: R²={corr_r2:.4f}, MAE={corr_mae:.4f}, RMSE={corr_rmse:.4f}, MAPE={corr_mape:.2f}%")

    # --- 2. LOO三列对比指标 ---
    actual_hv = df_compare["硬度_实测_HV"].values
    orig_hv = df_compare["硬度_原始LOO_HV"].values
    trend_hv = df_compare["硬度_趋势约束LOO_HV"].values
    actual_corr = df_compare["腐蚀_实测_Acm2"].values
    orig_corr = df_compare["腐蚀_原始LOO_Acm2"].values
    trend_corr = df_compare["腐蚀_趋势约束LOO_Acm2"].values

    actual_corr_log = np.log10(actual_corr * 10000)
    orig_corr_log = np.log10(orig_corr * 10000)
    trend_corr_log = np.log10(trend_corr * 10000)

    # 基线(原始数据中也有基线)
    df_orig_loo = pd.read_csv(os.path.join(LOO_DIR, "loo_results.csv"))
    baseline_hv = df_orig_loo["硬度_基线预测_HV"].values
    baseline_corr = df_orig_loo["腐蚀_基线预测_Acm2"].values
    baseline_corr_log = np.log10(baseline_corr * 10000)

    rows = []
    # 硬度行
    for metric_name, fn in [
        ("R²", lambda a, p: r2_score(a, p)),
        ("MAE (HV)", lambda a, p: mean_absolute_error(a, p)),
        ("RMSE (HV)", lambda a, p: np.sqrt(mean_squared_error(a, p))),
    ]:
        rows.append({
            "目标": "硬度",
            "指标": metric_name,
            "基线RF": fn(actual_hv, baseline_hv),
            "原始LOO": fn(actual_hv, orig_hv),
            "趋势约束LOO": fn(actual_hv, trend_hv),
        })

    # 趋势方向
    ds_a = np.sign(np.diff(actual_hv))
    rows.append({
        "目标": "硬度",
        "指标": "趋势方向匹配",
        "基线RF": f"{np.sum(np.sign(np.diff(baseline_hv)) == ds_a)}/3",
        "原始LOO": f"{np.sum(np.sign(np.diff(orig_hv)) == ds_a)}/3",
        "趋势约束LOO": f"{np.sum(np.sign(np.diff(trend_hv)) == ds_a)}/3",
    })
    # 峰值
    rows.append({
        "目标": "硬度",
        "指标": "峰值功率(W)",
        "基线RF": f"{df_compare['功率_W'].iloc[np.argmax(baseline_hv)]}",
        "原始LOO": f"{df_compare['功率_W'].iloc[np.argmax(orig_hv)]}",
        "趋势约束LOO": f"{df_compare['功率_W'].iloc[np.argmax(trend_hv)]}",
    })

    # 腐蚀行
    for metric_name, fn in [
        ("log空间R²", lambda a, p: r2_score(a, p)),
        ("log空间MAE", lambda a, p: mean_absolute_error(a, p)),
        ("几何均值倍数GMR", lambda a, p: np.exp(np.mean(np.log(np.abs(p) / np.abs(a))))),
    ]:
        rows.append({
            "目标": "腐蚀电流",
            "指标": metric_name,
            "基线RF": fn(actual_corr_log, baseline_corr_log),
            "原始LOO": fn(actual_corr_log, orig_corr_log),
            "趋势约束LOO": fn(actual_corr_log, trend_corr_log),
        })
    # 谷值
    rows.append({
        "目标": "腐蚀电流",
        "指标": "谷值功率(W)",
        "基线RF": f"{df_compare['功率_W'].iloc[np.argmin(baseline_corr)]}",
        "原始LOO": f"{df_compare['功率_W'].iloc[np.argmin(orig_corr)]}",
        "趋势约束LOO": f"{df_compare['功率_W'].iloc[np.argmin(trend_corr)]}",
    })

    df_loo_metrics = pd.DataFrame(rows)
    loo_metrics_path = os.path.join(TREND_DIR, "loo_metrics_comparison.csv")
    df_loo_metrics.to_csv(loo_metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {loo_metrics_path}")
    print(f"\n  {'目标':<8} {'指标':<16} {'基线RF':>14} {'原始LOO':>14} {'趋势约束LOO':>14}")
    print("  " + "-" * 72)
    for _, r in df_loo_metrics.iterrows():
        vals = []
        for v in [r["基线RF"], r["原始LOO"], r["趋势约束LOO"]]:
            try:
                vals.append(f"{float(v):>14.4f}")
            except (ValueError, TypeError):
                vals.append(f"{str(v):>14}")
        print(f"  {r['目标']:<8} {r['指标']:<16} {''.join(vals)}")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(LOO_DIR, exist_ok=True)
    os.makedirs(TREND_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. 生成全量训练评估数据
    df_eval = generate_full_training_eval()

    # 2. 生成LOO对比数据
    df_compare = generate_loo_comparison()

    # 3. 生成指标汇总
    generate_metrics(df_eval, df_compare)

    print(f"\n{'=' * 60}")
    print("评估数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
