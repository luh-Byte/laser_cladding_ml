"""
SHAP 中间数据计算模块
=====================
仅计算并保存SHAP相关中间数据（npy/csv/pkl），不生成任何图片。
适用于树模型（RF、LightGBM）的 TreeExplainer。
"""

import os
import numpy as np
import pandas as pd
import pickle

from src.config import OUTPUT_DIR


# ============================================================
# 目录管理
# ============================================================

SHAP_BASELINE_DIR = os.path.join(OUTPUT_DIR, "shap_baseline")
SHAP_FEWSHOT_DIR = os.path.join(OUTPUT_DIR, "shap_fewshot")


def _ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


# ============================================================
# 基线模型 SHAP
# ============================================================

def compute_baseline_shap(hv_model, corr_model,
                          X_train_hv, X_train_corr,
                          feat_names_hv, feat_names_corr):
    """
    计算基线RF模型的SHAP值与交互值，保存到 shap_baseline 目录。

    参数:
        hv_model: 最优硬度RF模型对象
        corr_model: 最优腐蚀RF模型对象
        X_train_hv: 硬度训练集特征矩阵（DataFrame，列名=特征名）
        X_train_corr: 腐蚀训练集特征矩阵（DataFrame）
        feat_names_hv: 硬度模型特征名列表
        feat_names_corr: 腐蚀模型特征名列表

    输出文件（outputs/shap_baseline/）:
        best_rf_hv.pkl          - 最优硬度RF模型
        best_rf_corr.pkl        - 最优腐蚀RF模型
        X_train_hv.csv          - 硬度训练特征矩阵
        X_train_corr.csv        - 腐蚀训练特征矩阵
        shap_vals_hv.npy        - 硬度SHAP值 (n_samples, n_features)
        shap_vals_corr.npy      - 腐蚀SHAP值 (n_samples, n_features)
        shap_inter_hv.npy       - 硬度SHAP交互值 (n_samples, n_features, n_features)
        shap_inter_corr.npy     - 腐蚀SHAP交互值 (n_samples, n_features, n_features)
        shap_import_baseline.csv - SHAP重要性统计表
    """
    import shap

    _ensure_dir(SHAP_BASELINE_DIR)
    print(f"\n[SHAP] 基线模型SHAP计算 → {SHAP_BASELINE_DIR}")

    # ---- 保存模型pkl ----
    with open(os.path.join(SHAP_BASELINE_DIR, "best_rf_hv.pkl"), "wb") as f:
        pickle.dump(hv_model, f)
    with open(os.path.join(SHAP_BASELINE_DIR, "best_rf_corr.pkl"), "wb") as f:
        pickle.dump(corr_model, f)
    print(f"  [保存] 模型pkl: best_rf_hv.pkl, best_rf_corr.pkl")

    # ---- 保存训练特征矩阵 ----
    X_train_hv.to_csv(os.path.join(SHAP_BASELINE_DIR, "X_train_hv.csv"), index=False)
    X_train_corr.to_csv(os.path.join(SHAP_BASELINE_DIR, "X_train_corr.csv"), index=False)
    print(f"  [保存] 训练集: X_train_hv.csv ({len(X_train_hv)}行, {len(feat_names_hv)}列), "
          f"X_train_corr.csv ({len(X_train_corr)}行, {len(feat_names_corr)}列)")

    # ---- 硬度 SHAP ----
    print(f"  [计算] 硬度SHAP值 ...")
    explainer_hv = shap.TreeExplainer(hv_model)
    shap_vals_hv = explainer_hv.shap_values(X_train_hv.values)
    # 处理多输出情况：RF回归通常返回单个数组，但有些版本返回列表
    if isinstance(shap_vals_hv, list):
        shap_vals_hv = shap_vals_hv[1] if len(shap_vals_hv) == 2 else shap_vals_hv[0]
    shap_vals_hv = np.asarray(shap_vals_hv)
    np.save(os.path.join(SHAP_BASELINE_DIR, "shap_vals_hv.npy"), shap_vals_hv)

    # 硬度 SHAP 交互值
    print(f"  [计算] 硬度SHAP交互值 ...")
    shap_inter_hv = explainer_hv.shap_interaction_values(X_train_hv.values)
    if isinstance(shap_inter_hv, list):
        shap_inter_hv = shap_inter_hv[1] if len(shap_inter_hv) == 2 else shap_inter_hv[0]
    shap_inter_hv = np.asarray(shap_inter_hv)
    np.save(os.path.join(SHAP_BASELINE_DIR, "shap_inter_hv.npy"), shap_inter_hv)

    # ---- 腐蚀 SHAP ----
    print(f"  [计算] 腐蚀SHAP值（log空间）...")
    explainer_corr = shap.TreeExplainer(corr_model)
    shap_vals_corr = explainer_corr.shap_values(X_train_corr.values)
    if isinstance(shap_vals_corr, list):
        shap_vals_corr = shap_vals_corr[1] if len(shap_vals_corr) == 2 else shap_vals_corr[0]
    shap_vals_corr = np.asarray(shap_vals_corr)
    np.save(os.path.join(SHAP_BASELINE_DIR, "shap_vals_corr.npy"), shap_vals_corr)

    # 腐蚀 SHAP 交互值
    print(f"  [计算] 腐蚀SHAP交互值 ...")
    shap_inter_corr = explainer_corr.shap_interaction_values(X_train_corr.values)
    if isinstance(shap_inter_corr, list):
        shap_inter_corr = shap_inter_corr[1] if len(shap_inter_corr) == 2 else shap_inter_corr[0]
    shap_inter_corr = np.asarray(shap_inter_corr)
    np.save(os.path.join(SHAP_BASELINE_DIR, "shap_inter_corr.npy"), shap_inter_corr)

    # ---- SHAP 重要性统计表 ----
    print(f"  [计算] SHAP重要性统计 ...")

    # 硬度 SHAP 重要性
    abs_shap_hv = np.mean(np.abs(shap_vals_hv), axis=0)
    # 模型原生重要性
    native_imp_hv = hv_model.feature_importances_

    # 腐蚀 SHAP 重要性
    abs_shap_corr = np.mean(np.abs(shap_vals_corr), axis=0)
    native_imp_corr = corr_model.feature_importances_

    # 合并到一个表（取两个目标的特征并集）
    all_feats = list(dict.fromkeys(feat_names_hv + feat_names_corr))
    rows = []
    for feat in all_feats:
        row = {"特征名": feat}
        if feat in feat_names_hv:
            idx = feat_names_hv.index(feat)
            row["abs_shap_mean_hv"] = abs_shap_hv[idx]
            row["model_native_importance_hv"] = native_imp_hv[idx]
        else:
            row["abs_shap_mean_hv"] = np.nan
            row["model_native_importance_hv"] = np.nan
        if feat in feat_names_corr:
            idx = feat_names_corr.index(feat)
            row["abs_shap_mean_corr"] = abs_shap_corr[idx]
            row["model_native_importance_corr"] = native_imp_corr[idx]
        else:
            row["abs_shap_mean_corr"] = np.nan
            row["model_native_importance_corr"] = np.nan
        rows.append(row)

    import_df = pd.DataFrame(rows)
    import_df.to_csv(os.path.join(SHAP_BASELINE_DIR, "shap_import_baseline.csv"), index=False)
    print(f"  [保存] shap_import_baseline.csv ({len(import_df)}行)")

    print(f"  [完成] 基线SHAP数据已全部保存")
    return True


# ============================================================
# Few-shot 模型 SHAP
# ============================================================

def compute_fewshot_shap(hv_model, corr_model,
                         X_fs, feat_names,
                         rockit_start_idx, rockit_n=4):
    """
    计算Few-shot微调LightGBM模型的SHAP值，保存到 shap_fewshot 目录。

    参数:
        hv_model: Few-shot硬度LightGBM模型
        corr_model: Few-shot腐蚀LightGBM模型
        X_fs: 完整训练特征矩阵（DataFrame，包含基础数据+fewshot数据）
        feat_names: 特征名列表
        rockit_start_idx: Rockit485样本在X_fs中的起始行索引
        rockit_n: Rockit485样本数量（默认4个）

    输出文件（outputs/shap_fewshot/）:
        X_fs.csv                  - 完整训练特征矩阵
        shap_vals_hv_fs.npy       - 硬度SHAP值
        shap_vals_corr_fs.npy     - 腐蚀SHAP值
        shap_import_fewshot.csv   - SHAP重要性统计表
        rockit_sample_index.csv   - Rockit485样本行号
    """
    import shap

    _ensure_dir(SHAP_FEWSHOT_DIR)
    print(f"\n[SHAP] Few-shot模型SHAP计算 → {SHAP_FEWSHOT_DIR}")

    # ---- 保存训练特征矩阵 ----
    X_fs.to_csv(os.path.join(SHAP_FEWSHOT_DIR, "X_fs.csv"), index=False)
    print(f"  [保存] X_fs.csv ({len(X_fs)}行, {len(feat_names)}列)")

    # ---- Rockit样本索引表 ----
    rockit_indices = list(range(rockit_start_idx, rockit_start_idx + rockit_n))
    rockit_df = pd.DataFrame({
        "row_index": rockit_indices,
        "sample_id": [f"Rockit485_{i+1}" for i in range(rockit_n)],
    })
    rockit_df.to_csv(os.path.join(SHAP_FEWSHOT_DIR, "rockit_sample_index.csv"), index=False)
    print(f"  [保存] rockit_sample_index.csv (Rockit485行号: {rockit_indices})")

    # ---- 硬度 SHAP ----
    print(f"  [计算] 硬度SHAP值 ...")
    explainer_hv = shap.TreeExplainer(hv_model)
    shap_vals_hv = explainer_hv.shap_values(X_fs.values)
    if isinstance(shap_vals_hv, list):
        shap_vals_hv = shap_vals_hv[1] if len(shap_vals_hv) == 2 else shap_vals_hv[0]
    shap_vals_hv = np.asarray(shap_vals_hv)
    np.save(os.path.join(SHAP_FEWSHOT_DIR, "shap_vals_hv_fs.npy"), shap_vals_hv)

    # ---- 腐蚀 SHAP ----
    print(f"  [计算] 腐蚀SHAP值（log空间）...")
    explainer_corr = shap.TreeExplainer(corr_model)
    shap_vals_corr = explainer_corr.shap_values(X_fs.values)
    if isinstance(shap_vals_corr, list):
        shap_vals_corr = shap_vals_corr[1] if len(shap_vals_corr) == 2 else shap_vals_corr[0]
    shap_vals_corr = np.asarray(shap_vals_corr)
    np.save(os.path.join(SHAP_FEWSHOT_DIR, "shap_vals_corr_fs.npy"), shap_vals_corr)

    # ---- SHAP 重要性统计表 ----
    print(f"  [计算] SHAP重要性统计 ...")
    abs_shap_hv = np.mean(np.abs(shap_vals_hv), axis=0)
    abs_shap_corr = np.mean(np.abs(shap_vals_corr), axis=0)

    import_df = pd.DataFrame({
        "特征名": feat_names,
        "abs_shap_mean_hv": abs_shap_hv,
        "abs_shap_mean_corr": abs_shap_corr,
    })
    import_df.to_csv(os.path.join(SHAP_FEWSHOT_DIR, "shap_import_fewshot.csv"), index=False)
    print(f"  [保存] shap_import_fewshot.csv ({len(import_df)}行)")

    print(f"  [完成] Few-shot SHAP数据已全部保存")
    return True
