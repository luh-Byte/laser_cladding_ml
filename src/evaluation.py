"""
评估与输出模块
==============
1. 显微硬度预测指标: R2, MAE, RMSE
2. 腐蚀电流密度预测指标: log空间R2, 原始量纲R2, RMSE, 平均相对误差, 数量级准确率
3. 结果汇总为Excel
4. 输出最优模型权重、预测数据集、参数迭代日志
"""

import numpy as np
import pandas as pd
import os
import pickle
from src.config import RESULT_DIR, MODEL_DIR, MODEL_ORDER


def compute_hardness_metrics(y_true, y_pred, y_train_true, y_train_pred):
    """显微硬度预测指标（原始HV量纲）"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    return {
        "train_R2": r2_score(y_train_true, y_train_pred),
        "test_R2": r2_score(y_true, y_pred),
        "test_MAE": mean_absolute_error(y_true, y_pred),
        "test_RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "overfitting": r2_score(y_train_true, y_train_pred) - r2_score(y_true, y_pred),
    }


def compute_corrosion_metrics(y_true_log, y_pred_log, corrosion_transformer):
    """
    腐蚀电流密度预测指标。
    输入为log空间的真值和预测值。
    转换回原始量纲后计算指标。
    """
    from sklearn.metrics import r2_score, mean_squared_error

    # log空间R2
    log_r2 = r2_score(y_true_log, y_pred_log)

    # 转回原始量纲 (A/cm2)
    y_true_orig = corrosion_transformer.log_to_original(y_true_log)
    y_pred_orig = corrosion_transformer.log_to_original(y_pred_log)

    # 原始量纲R2
    orig_r2 = r2_score(y_true_orig, y_pred_orig)

    # 原始量纲RMSE
    orig_rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))

    # 平均相对误差
    rel_errors = np.abs(y_pred_orig - y_true_orig) / np.abs(y_true_orig)
    mean_rel_error = np.mean(rel_errors)

    # 数量级预测准确率（预测值与真值在同一数量级内，即误差<0.5个数量级）
    log_errors = np.abs(np.log10(y_pred_orig) - np.log10(y_true_orig))
    magnitude_acc = np.mean(log_errors < 0.5)

    return {
        "log_R2": log_r2,
        "orig_R2": orig_r2,
        "orig_RMSE": orig_rmse,
        "mean_rel_error": mean_rel_error,
        "magnitude_accuracy": magnitude_acc,
        "overfitting": None,  # 由调用方补充
    }


def aggregate_repetition_results(results_list):
    """
    汇总10次重复实验结果，计算均值±标准差。
    results_list: [{metric: value, ...}, ...] 每次实验的指标
    返回: {metric: (mean, std)}
    """
    if not results_list:
        return {}
    all_keys = results_list[0].keys()
    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in results_list if r.get(key) is not None]
        if not values:
            continue
        # 跳过非数值字段（如best_params字符串）
        if isinstance(values[0], (int, float, np.floating, np.integer)):
            aggregated[key] = (float(np.mean(values)), float(np.std(values)))
    return aggregated


def export_summary_excel(all_results, output_dir=None):
    """
    将所有模型、两个目标的结果汇总为Excel。
    all_results: dict, 结构为 {target: {model_name: {metrics}}}
    """
    out_dir = output_dir or RESULT_DIR
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, "model_summary.xlsx")

    rows = []
    for target_name, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            row = {"目标变量": target_name, "模型": model_name}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(filepath, index=False)
    print(f"\n[结果输出] 汇总表已保存: {filepath}")
    return filepath


def export_predictions(model_name, target_name, rep_idx,
                       y_true, y_pred, errors, output_dir=None):
    """输出单次实验的预测数据集（真实值、预测值、误差值）"""
    out_dir = output_dir or os.path.join(RESULT_DIR, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({
        "真实值": y_true,
        "预测值": y_pred,
        "绝对误差": errors,
        "相对误差(%)": errors / np.abs(y_true) * 100,
    })
    filename = f"pred_{model_name}_{target_name}_rep{rep_idx}.csv"
    filepath = os.path.join(out_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath


def export_feature_importance(importance_dict, model_name, target_name, output_dir=None):
    """输出特征重要性"""
    if importance_dict is None:
        return None
    out_dir = output_dir or os.path.join(RESULT_DIR, "feature_importance")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame([
        {"特征": k, "重要性": v} for k, v in
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    ])
    filename = f"importance_{model_name}_{target_name}.csv"
    filepath = os.path.join(out_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath


def save_model(model, model_name, target_name, output_dir=None):
    """保存最优模型权重（兼容旧格式）"""
    out_dir = output_dir or MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    filename = f"model_{model_name}_{target_name}.pkl"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


def save_model_bundle(bundle, model_name, target_name, output_dir=None):
    """
    保存完整模型包，包含模型、标准化器、特征列表等推理所需全部信息。
    bundle = {
        'model': 训练好的模型,
        'scaler': StandardScaler（fit在训练数据上）,
        'selected_features': 选中的特征名列表,
        'corrosion_transformer': 腐蚀电流的log变换参数（仅腐蚀模型需要）,
    }
    """
    out_dir = output_dir or MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    filename = f"bundle_{model_name}_{target_name}.pkl"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f)
    return filepath


def load_model_bundle(model_name, target_name, model_dir=None):
    """加载完整模型包"""
    model_dir = model_dir or MODEL_DIR
    filename = f"bundle_{model_name}_{target_name}.pkl"
    filepath = os.path.join(model_dir, filename)
    with open(filepath, "rb") as f:
        bundle = pickle.load(f)
    return bundle


class IterationLogger:
    """参数迭代日志记录器"""

    def __init__(self, output_dir=None):
        self.logs = []
        self.out_dir = output_dir or os.path.join(RESULT_DIR, "logs")
        os.makedirs(self.out_dir, exist_ok=True)

    def log(self, model_name, target_name, rep_idx, params, metrics):
        self.logs.append({
            "模型": model_name,
            "目标变量": target_name,
            "重复次数": rep_idx,
            "参数": str(params),
            "训练R2": metrics.get("train_R2"),
            "测试R2": metrics.get("test_R2") or metrics.get("log_R2"),
            "MAE": metrics.get("test_MAE"),
            "RMSE": metrics.get("test_RMSE") or metrics.get("orig_RMSE"),
            "过拟合度": metrics.get("overfitting"),
        })

    def save(self):
        if not self.logs:
            return None
        filepath = os.path.join(self.out_dir, "iteration_log.csv")
        df = pd.DataFrame(self.logs)
        df.to_csv(filepath, index=False)
        print(f"[结果输出] 迭代日志已保存: {filepath}")
        return filepath
