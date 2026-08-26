"""
主训练管道
============
串联: 数据清洗 -> 特征工程 -> 标准化/共线性筛选 -> 四模型依次训练调参 -> 统一指标输出
固定执行顺序: KNN -> SVR -> 随机森林 -> LightGBM
双目标: 显微硬度(原始HV)、腐蚀电流密度(log10+Z-score, 反变换输出A/cm2)
10次重复划分实验 -> 均值±标准差
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加项目根路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    N_REPETITIONS, BASE_RANDOM_SEED, MODEL_ORDER,
    RESULT_DIR, MODEL_DIR
)
from src.data_preprocessing import (
    load_raw_data, clean_data, prepare_targets, generate_split_indices
)
from src.feature_engineering import (
    compute_derived_features, get_all_feature_names,
    FeaturePipeline, CorrosionTargetTransformer
)
from src.models import train_model, get_feature_importance
from src.evaluation import (
    compute_hardness_metrics, compute_corrosion_metrics,
    aggregate_repetition_results, export_summary_excel,
    export_predictions, export_feature_importance,
    save_model, IterationLogger
)


def run_hardness_experiment(X_train, X_test, y_train, y_test, selected_features, rep_idx):
    """单次硬度实验: 训练4模型, 返回各模型指标+模型对象"""
    results = {}
    predictions = {}
    models = {}

    for model_name in MODEL_ORDER:
        print(f"\n  --- {model_name} (硬度) ---")
        model, best_params, cv_score = train_model(model_name, X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = compute_hardness_metrics(y_test, y_test_pred, y_train, y_train_pred)
        metrics["best_params"] = str(best_params)
        metrics["cv_R2"] = cv_score
        results[model_name] = metrics
        predictions[model_name] = {
            "y_true": y_test, "y_pred": y_test_pred,
            "errors": np.abs(y_test - y_test_pred),
        }
        models[model_name] = model

    return results, predictions, models


def run_corrosion_experiment(X_train, X_test,
                             y_train_log, y_test_log,
                             corrosion_transformer,
                             selected_features, rep_idx):
    """单次腐蚀电流实验: 训练4模型, 返回各模型指标+模型对象"""
    # 转换为Z-score空间供模型训练
    y_train_zscore = corrosion_transformer.transform(y_train_log)
    y_test_zscore = corrosion_transformer.transform(y_test_log)

    results = {}
    predictions = {}
    models = {}

    for model_name in MODEL_ORDER:
        print(f"\n  --- {model_name} (腐蚀电流) ---")
        model, best_params, cv_score = train_model(model_name, X_train, y_train_zscore)

        # 预测（Z-score空间）
        y_train_pred_z = model.predict(X_train)
        y_test_pred_z = model.predict(X_test)

        # 转回log空间用于评估
        y_train_pred_log = corrosion_transformer.inverse_transform_to_log(y_train_pred_z)
        y_test_pred_log = corrosion_transformer.inverse_transform_to_log(y_test_pred_z)

        # log空间指标
        from sklearn.metrics import r2_score
        train_log_r2 = r2_score(y_train_log, y_train_pred_log)

        # 测试集指标
        metrics = compute_corrosion_metrics(y_test_log, y_test_pred_log, corrosion_transformer)
        metrics["best_params"] = str(best_params)
        metrics["cv_R2"] = cv_score
        metrics["overfitting"] = train_log_r2 - metrics["log_R2"]
        metrics["train_log_R2"] = train_log_r2
        results[model_name] = metrics

        # 转回原始量纲用于预测数据集输出
        y_test_pred_orig = corrosion_transformer.log_to_original(y_test_pred_log)
        y_test_true_orig = corrosion_transformer.log_to_original(y_test_log)
        predictions[model_name] = {
            "y_true": y_test_true_orig,
            "y_pred": y_test_pred_orig,
            "errors": np.abs(y_test_true_orig - y_test_pred_orig),
        }
        models[model_name] = model

    return results, predictions, models


def main():
    print("=" * 70)
    print("激光熔覆涂层性能预测 - 机器学习训练管道")
    print("固定顺序: KNN -> SVR -> 随机森林 -> LightGBM")
    print("双目标: 显微硬度(HV) + 腐蚀电流密度(A/cm2)")
    print(f"重复实验: {N_REPETITIONS}次, 测试集比例: 20%")
    print("=" * 70)

    # ====== 1. 数据加载与清洗 ======
    print("\n[步骤1] 数据加载与清洗")
    df = load_raw_data()
    df_clean, cleaning_log = clean_data(df)

    # ====== 2. 特征工程 ======
    print("\n[步骤2] 物理衍生特征计算")
    df_feat = compute_derived_features(df_clean)
    all_features = get_all_feature_names()
    X_full = df_feat[all_features].copy()
    print(f"  全部特征: {len(all_features)}项")
    print(f"  特征列表: {all_features}")

    # ====== 3. 准备目标变量 ======
    print("\n[步骤3] 目标变量准备")
    targets = prepare_targets(df_clean)
    print(f"  硬度范围: {targets['hardness'].min():.1f} - {targets['hardness'].max():.1f} HV")
    print(f"  腐蚀电流log10(Excel腐蚀电流*10000)范围: {targets['corrosion_log'].min():.4f} - {targets['corrosion_log'].max():.4f}")

    # ====== 4. 生成10次划分（双目标联合分层抽样） ======
    n_samples = len(df_clean)
    splits = generate_split_indices(
        y_hardness=targets["hardness"],
        y_corrosion_log=targets["corrosion_log"],
    )
    print(f"\n[步骤4] 生成 {N_REPETITIONS} 次重复划分 (样本总数: {n_samples})")
    print(f"  分层方式: {splits[0][3]}")

    # ====== 5. 迭代训练 ======
    print(f"\n[步骤5] 开始 {N_REPETITIONS} 次重复实验")
    logger = IterationLogger()

    # 存储所有重复的结果: {target: {model: [metrics_rep1, ...]}}
    all_hardness_results = {m: [] for m in MODEL_ORDER}
    all_corrosion_results = {m: [] for m in MODEL_ORDER}
    all_best_params_h = {m: [] for m in MODEL_ORDER}
    all_best_params_c = {m: [] for m in MODEL_ORDER}

    best_hardness_model = None
    best_corrosion_model = None
    best_hardness_r2 = -np.inf
    best_corrosion_r2 = -np.inf
    best_hardness_info = {}
    best_corrosion_info = {}

    # SHAP专用：跟踪最优RF模型及对应训练集
    best_rf_hv_model = None
    best_rf_corr_model = None
    best_rf_hv_r2 = -np.inf
    best_rf_corr_r2 = -np.inf
    best_rf_hv_X_train = None   # DataFrame, 带列名
    best_rf_corr_X_train = None
    best_rf_hv_feat_names = None
    best_rf_corr_feat_names = None

    feature_importance_all = {}  # {model: {target: {feature: importance}}}

    for rep_idx, (train_idx, test_idx, seed, strat_method) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"重复实验 {rep_idx + 1}/{N_REPETITIONS} (seed={seed}, "
              f"训练集{len(train_idx)}行, 测试集{len(test_idx)}行)")
        print(f"{'='*60}")

        # --- 特征管道: 仅训练集fit ---
        feat_pipe = FeaturePipeline()
        X_train, selected_features = feat_pipe.fit_transform(
            X_full.iloc[train_idx].reset_index(drop=True)
        )
        X_test, _ = feat_pipe.transform(
            X_full.iloc[test_idx].reset_index(drop=True)
        )

        # --- 硬度实验 ---
        y_train_h = targets["hardness"][train_idx]
        y_test_h = targets["hardness"][test_idx]

        h_results, h_preds, h_models = run_hardness_experiment(
            X_train, X_test, y_train_h, y_test_h,
            selected_features, rep_idx
        )

        for model_name in MODEL_ORDER:
            all_hardness_results[model_name].append(h_results[model_name])
            all_best_params_h[model_name].append(h_results[model_name]["best_params"])
            logger.log(model_name, "硬度", rep_idx,
                       h_results[model_name]["best_params"], h_results[model_name])

            # 保存第1次实验的预测数据集
            if rep_idx == 0:
                export_predictions(model_name, "硬度", rep_idx,
                                  h_preds[model_name]["y_true"],
                                  h_preds[model_name]["y_pred"],
                                  h_preds[model_name]["errors"])

            # 记录最佳模型
            if h_results[model_name]["test_R2"] > best_hardness_r2:
                best_hardness_r2 = h_results[model_name]["test_R2"]
                best_hardness_model = h_models[model_name]
                best_hardness_info = {
                    "model_name": model_name,
                    "rep_idx": rep_idx,
                    "params": h_results[model_name]["best_params"],
                    "r2": h_results[model_name]["test_R2"],
                }

            # SHAP：记录最优RF硬度模型及对应训练集
            if model_name == "RF":
                X_train_df = pd.DataFrame(X_train, columns=selected_features)
                if h_results[model_name]["test_R2"] > best_rf_hv_r2:
                    best_rf_hv_r2 = h_results[model_name]["test_R2"]
                    best_rf_hv_model = h_models[model_name]
                    best_rf_hv_X_train = X_train_df
                    best_rf_hv_feat_names = selected_features

            # 特征重要性（仅RF和LightGBM，取第1次实验，复用已训练模型）
            if rep_idx == 0 and model_name in ("RF", "LightGBM"):
                imp = get_feature_importance(h_models[model_name], model_name, selected_features)
                if imp:
                    feature_importance_all.setdefault(model_name, {})["硬度"] = imp
                    export_feature_importance(imp, model_name, "硬度")

        # --- 腐蚀电流实验 ---
        y_train_c_log = targets["corrosion_log"][train_idx]
        y_test_c_log = targets["corrosion_log"][test_idx]

        # 腐蚀变换器: 仅训练集fit
        corr_transformer = CorrosionTargetTransformer()
        corr_transformer.fit(y_train_c_log)

        c_results, c_preds, c_models = run_corrosion_experiment(
            X_train, X_test, y_train_c_log, y_test_c_log,
            corr_transformer, selected_features, rep_idx
        )

        for model_name in MODEL_ORDER:
            all_corrosion_results[model_name].append(c_results[model_name])
            all_best_params_c[model_name].append(c_results[model_name]["best_params"])
            logger.log(model_name, "腐蚀电流", rep_idx,
                       c_results[model_name]["best_params"], c_results[model_name])

            if rep_idx == 0:
                export_predictions(model_name, "腐蚀电流", rep_idx,
                                  c_preds[model_name]["y_true"],
                                  c_preds[model_name]["y_pred"],
                                  c_preds[model_name]["errors"])

            if c_results[model_name]["log_R2"] > best_corrosion_r2:
                best_corrosion_r2 = c_results[model_name]["log_R2"]
                best_corrosion_model = c_models[model_name]
                best_corrosion_info = {
                    "model_name": model_name,
                    "rep_idx": rep_idx,
                    "params": c_results[model_name]["best_params"],
                    "r2": c_results[model_name]["log_R2"],
                }

            # SHAP：记录最优RF腐蚀模型及对应训练集
            if model_name == "RF":
                X_train_df = pd.DataFrame(X_train, columns=selected_features)
                if c_results[model_name]["log_R2"] > best_rf_corr_r2:
                    best_rf_corr_r2 = c_results[model_name]["log_R2"]
                    best_rf_corr_model = c_models[model_name]
                    best_rf_corr_X_train = X_train_df
                    best_rf_corr_feat_names = selected_features

            # 特征重要性（复用已训练模型）
            if rep_idx == 0 and model_name in ("RF", "LightGBM"):
                imp = get_feature_importance(c_models[model_name], model_name, selected_features)
                if imp:
                    feature_importance_all.setdefault(model_name, {})["腐蚀电流"] = imp
                    export_feature_importance(imp, model_name, "腐蚀电流")

    # ====== 6. 汇总结果 ======
    print(f"\n{'='*70}")
    print("[步骤6] 结果汇总与输出")
    print(f"{'='*70}")

    # 汇总硬度结果
    hardness_summary = {}
    for model_name in MODEL_ORDER:
        agg = aggregate_repetition_results(all_hardness_results[model_name])
        hardness_summary[model_name] = {
            "最优参数": _most_frequent_param(all_best_params_h[model_name]),
            "训练集R2(均值±std)": _fmt_agg(agg.get("train_R2")),
            "测试集R2(均值±std)": _fmt_agg(agg.get("test_R2")),
            "测试集MAE(均值±std)": _fmt_agg(agg.get("test_MAE")),
            "测试集RMSE(均值±std)": _fmt_agg(agg.get("test_RMSE")),
            "过拟合度(均值±std)": _fmt_agg(agg.get("overfitting")),
        }

    # 汇总腐蚀电流结果
    corrosion_summary = {}
    for model_name in MODEL_ORDER:
        agg = aggregate_repetition_results(all_corrosion_results[model_name])
        corrosion_summary[model_name] = {
            "最优参数": _most_frequent_param(all_best_params_c[model_name]),
            "log空间R2(均值±std)": _fmt_agg(agg.get("log_R2")),
            "原始量纲R2(均值±std)": _fmt_agg(agg.get("orig_R2")),
            "原始量纲RMSE(均值±std)": _fmt_agg(agg.get("orig_RMSE")),
            "平均相对误差(均值±std)": _fmt_agg(agg.get("mean_rel_error")),
            "数量级准确率(均值±std)": _fmt_agg(agg.get("magnitude_accuracy")),
            "过拟合度(均值±std)": _fmt_agg(agg.get("overfitting")),
        }

    # 打印汇总表
    print("\n" + "-" * 70)
    print("【显微硬度预测结果汇总】")
    print("-" * 70)
    _print_summary_table(hardness_summary)

    print("\n" + "-" * 70)
    print("【腐蚀电流密度预测结果汇总】")
    print("-" * 70)
    _print_summary_table(corrosion_summary)

    # 导出Excel
    all_results = {"硬度": hardness_summary, "腐蚀电流": corrosion_summary}
    excel_path = export_summary_excel(all_results)

    # 保存最佳模型权重
    print(f"\n[最佳模型] 硬度: {best_hardness_info['model_name']} "
          f"(rep{best_hardness_info['rep_idx']+1}, R2={best_hardness_info['r2']:.4f})")
    print(f"[最佳模型] 腐蚀: {best_corrosion_info['model_name']} "
          f"(rep{best_corrosion_info['rep_idx']+1}, log_R2={best_corrosion_info['r2']:.4f})")
    if best_hardness_model is not None:
        save_model(best_hardness_model, best_hardness_info['model_name'], "硬度")
    if best_corrosion_model is not None:
        save_model(best_corrosion_model, best_corrosion_info['model_name'], "腐蚀电流")

    # 保存迭代日志
    logger.save()

    # ====== 7. SHAP 中间数据计算（仅数据，不绘图） ======
    print(f"\n{'='*70}")
    print("[步骤7] SHAP 中间数据计算")
    print(f"{'='*70}")

    from src.shap_utils import compute_baseline_shap
    if best_rf_hv_model is not None and best_rf_corr_model is not None:
        compute_baseline_shap(
            best_rf_hv_model, best_rf_corr_model,
            best_rf_hv_X_train, best_rf_corr_X_train,
            best_rf_hv_feat_names, best_rf_corr_feat_names,
        )
        print(f"\n[最佳RF] 硬度: rep{best_hardness_info.get('rep_idx', '?')+1 if isinstance(best_hardness_info.get('rep_idx'), int) else '?'} "
              f"(R2={best_rf_hv_r2:.4f})")
        print(f"[最佳RF] 腐蚀: rep{best_corrosion_info.get('rep_idx', '?')+1 if isinstance(best_corrosion_info.get('rep_idx'), int) else '?'} "
              f"(log_R2={best_rf_corr_r2:.4f})")
    else:
        print("  [跳过] RF模型未找到，跳过SHAP计算")

    print(f"\n{'='*70}")
    print("训练管道执行完成!")
    print(f"结果目录: {RESULT_DIR}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"{'='*70}")


def _fmt_agg(agg_tuple):
    """格式化 (mean, std) 为字符串"""
    if agg_tuple is None:
        return "N/A"
    mean, std = agg_tuple
    return f"{mean:.4f} ± {std:.4f}"


def _most_frequent_param(params_list):
    """取10次实验中出现频率最高的参数组合"""
    from collections import Counter
    counter = Counter(params_list)
    return counter.most_common(1)[0][0] if counter else "N/A"


def _print_summary_table(summary):
    """打印汇总表"""
    if not summary:
        return
    headers = list(list(summary.values())[0].keys())
    # 表头
    header_str = f"{'模型':<12}" + "".join(f"{h:<28}" for h in headers)
    print(header_str)
    print("-" * len(header_str))
    for model_name, metrics in summary.items():
        row = f"{model_name:<12}"
        for h in headers:
            val = str(metrics[h])[:26]
            row += f"{val:<28}"
        print(row)


if __name__ == "__main__":
    main()
