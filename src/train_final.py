"""
最终模型训练
============
用全部清洗后数据 + 最优超参数 训练最终模型，保存为完整模型包(bundle)。
最终模型用于帕累托优化和部署推理。
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESS_FEATURES, COMPOSITION_FEATURES,
    FEATURES_TO_REMOVE, CORRELATION_THRESHOLD
)
from src.data_preprocessing import load_raw_data, clean_data
from src.feature_engineering import (
    compute_derived_features, get_all_feature_names,
    filter_correlated_features
)
from src.evaluation import save_model_bundle


# 最优超参数（来自10次重复实验的最优模型参数）
BEST_PARAMS = {
    "硬度": {
        "model_name": "LightGBM",
        "params": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "reg_lambda": 10,
        }
    },
    "腐蚀电流": {
        "model_name": "RF",
        "params": {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_leaf": 5,
        }
    }
}


def train_final_model():
    """用全部数据训练最终模型，保存为bundle"""
    print("=" * 70)
    print("最终模型训练 - 使用全部清洗后数据")
    print("=" * 70)

    # 1. 数据加载与清洗
    df = load_raw_data()
    df_clean, cleaning_log = clean_data(df)
    print(f"\n[数据] 清洗后样本数: {len(df_clean)}")

    # 2. 构建特征
    df_feat = compute_derived_features(df_clean.copy())
    all_feat_names = get_all_feature_names()
    X_raw = df_feat[all_feat_names].values

    # 3. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 4. 共线性筛选
    selected_features, dropped = filter_correlated_features(
        X_scaled, all_feat_names, CORRELATION_THRESHOLD
    )
    print(f"[特征] 共线性剔除: {dropped}")

    # 5. 重要性精简
    if FEATURES_TO_REMOVE:
        selected_features = [f for f in selected_features
                             if f not in FEATURES_TO_REMOVE]
        print(f"[特征] 重要性剔除: {FEATURES_TO_REMOVE}")
    print(f"[特征] 最终保留 {len(selected_features)} 项")

    # 获取选中特征的列索引
    feat_idx = [all_feat_names.index(f) for f in selected_features]
    X_final = X_scaled[:, feat_idx]

    # 6. 训练硬度模型
    print(f"\n--- 训练硬度模型 ({BEST_PARAMS['硬度']['model_name']}) ---")
    hardness_model_name = BEST_PARAMS["硬度"]["model_name"]
    hardness_params = BEST_PARAMS["硬度"]["params"]
    y_hardness = df_clean["硬度"].values

    import lightgbm as lgb
    hardness_model = lgb.LGBMRegressor(
        random_state=42, verbose=-1, force_col_wise=True,
        **hardness_params
    )
    hardness_model.fit(X_final, y_hardness)
    print(f"  参数: {hardness_params}")

    hardness_bundle = {
        'model': hardness_model,
        'scaler': scaler,
        'selected_features': selected_features,
        'all_feature_names': all_feat_names,
        'target': '硬度',
    }
    save_model_bundle(hardness_bundle, hardness_model_name, "硬度")
    print(f"  已保存: bundle_{hardness_model_name}_硬度.pkl")

    # 7. 训练腐蚀模型（Z-score空间）
    print(f"\n--- 训练腐蚀模型 ({BEST_PARAMS['腐蚀电流']['model_name']}) ---")
    corrosion_model_name = BEST_PARAMS["腐蚀电流"]["model_name"]
    corrosion_params = BEST_PARAMS["腐蚀电流"]["params"]

    icorr_10k = df_clean["腐蚀电流*10000"].values.astype(float)
    icorr_10k = np.maximum(icorr_10k, 1e-15)
    y_corr_log = np.log10(icorr_10k)

    corr_mean = np.mean(y_corr_log)
    corr_std = np.std(y_corr_log)
    y_corr_zscore = (y_corr_log - corr_mean) / corr_std
    print(f"  log空间均值: {corr_mean:.4f}, std: {corr_std:.4f}")

    from sklearn.ensemble import RandomForestRegressor
    corrosion_model = RandomForestRegressor(
        random_state=42, n_jobs=-1,
        **corrosion_params
    )
    corrosion_model.fit(X_final, y_corr_zscore)
    print(f"  参数: {corrosion_params}")

    corrosion_bundle = {
        'model': corrosion_model,
        'scaler': scaler,
        'selected_features': selected_features,
        'all_feature_names': all_feat_names,
        'target': '腐蚀电流',
        'corrosion_transformer': {'mean': corr_mean, 'std': corr_std},
    }
    save_model_bundle(corrosion_bundle, corrosion_model_name, "腐蚀电流")
    print(f"  已保存: bundle_{corrosion_model_name}_腐蚀电流.pkl")

    print(f"\n{'='*70}")
    print("最终模型训练完成!")
    print(f"{'='*70}")

    return hardness_bundle, corrosion_bundle


if __name__ == "__main__":
    train_final_model()
