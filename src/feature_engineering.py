"""
特征工程模块
============
1. 计算6项物理衍生特征
2. Z-score标准化（仅用训练集拟合，防止数据泄露）
3. 共线性筛选（|r|>0.9剔除冗余特征）
4. 腐蚀电流目标的标准化与反变换
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import (
    PROCESS_FEATURES, COMPOSITION_FEATURES, DERIVED_FEATURE_FORMULAS,
    CORRELATION_THRESHOLD, FEATURES_TO_REMOVE
)


def compute_derived_features(df):
    """
    计算6项物理衍生特征，返回含衍生特征的DataFrame。
    公式基于冶金学标准：
    - 线能量密度 = 功率/速度 (J/mm)
    - 面能量密度 = 功率/(速度*光斑直径) (J/mm^2)
    - 粉末能量比 = 功率/送粉速率 (W*min/g)
    - 碳当量 = C + Mn/6 + (Cr+Mo)/5 + Ni/15 (IIW)
    - 铬当量 = Cr + 1.5*Si + 1.5*Mo (Schaeffler)
    - 镍当量 = Ni + 30*C + 0.5*Mn (Schaeffler)
    """
    df_feat = df.copy()

    P = df_feat["激光功率"].values
    v = df_feat["扫描速度"].values
    spot = df_feat["光斑直径"].values
    powder = df_feat["送粉速率"].values
    C = df_feat["C"].values
    Cr = df_feat["Cr"].values
    Si = df_feat["Si"].values
    Ni = df_feat["Ni"].values
    Fe = df_feat["Fe"].values
    Mn = df_feat["Mn"].values
    Mo = df_feat["Mo"].values

    # 防止除零
    v_safe = np.where(v == 0, 1e-6, v)
    spot_safe = np.where(spot == 0, 1e-6, spot)
    powder_safe = np.where(powder == 0, 1e-6, powder)

    df_feat["线能量密度"] = P / v_safe
    df_feat["面能量密度"] = P / (v_safe * spot_safe)
    df_feat["粉末能量比"] = P / powder_safe
    df_feat["碳当量"] = C + Mn / 6.0 + (Cr + Mo) / 5.0 + Ni / 15.0
    df_feat["铬当量"] = Cr + 1.5 * Si + 1.5 * Mo
    df_feat["镍当量"] = Ni + 30.0 * C + 0.5 * Mn

    return df_feat


def get_all_feature_names():
    """返回全部18项特征名（5工艺+7成分+6衍生）"""
    derived = list(DERIVED_FEATURE_FORMULAS.keys())
    return PROCESS_FEATURES + COMPOSITION_FEATURES + derived


def filter_correlated_features(X_train, feature_names, threshold=CORRELATION_THRESHOLD):
    """
    基于|r|>0.9共线性筛选，仅在训练集上计算。
    返回保留的特征列名列表。
    策略: 逐对检查，若两特征|r|>threshold，保留排在前面的，剔除后面的。
    """
    corr_matrix = pd.DataFrame(X_train, columns=feature_names).corr()
    to_drop = set()
    n = len(feature_names)
    for i in range(n):
        if feature_names[i] in to_drop:
            continue
        for j in range(i + 1, n):
            if feature_names[j] in to_drop:
                continue
            r = float(corr_matrix.iloc[i, j])
            if abs(r) > threshold:
                # 保留前者，剔除后者
                to_drop.add(feature_names[j])

    selected = [f for f in feature_names if f not in to_drop]
    dropped = list(to_drop)
    return selected, dropped


class FeaturePipeline:
    """
    特征处理管道：
    1. 标准化（仅fit训练集）
    2. 共线性筛选（仅fit训练集）
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_features = None
        self.all_features = get_all_feature_names()
        self.dropped_features = None

    def fit(self, X_train_df):
        """
        在训练集上拟合：
        - 先标准化
        - 共线性筛选
        - 基于重要性的特征精简
        X_train_df: DataFrame, 列为全部18项特征
        """
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train_df.values)

        # 共线性筛选
        self.selected_features, self.dropped_features = filter_correlated_features(
            X_train_scaled, list(X_train_df.columns)
        )

        print(f"  [特征筛选] 共线性剔除(|r|>{CORRELATION_THRESHOLD}): "
              f"{self.dropped_features if self.dropped_features else '无'}")

        # 基于重要性的特征精简
        if FEATURES_TO_REMOVE:
            remaining_remove = [f for f in FEATURES_TO_REMOVE if f in self.selected_features]
            if remaining_remove:
                self.selected_features = [f for f in self.selected_features
                                          if f not in remaining_remove]
                self.dropped_features.extend(remaining_remove)
                print(f"  [特征精简] 重要性剔除: {remaining_remove}")

        print(f"  [特征筛选] 保留特征: {len(self.selected_features)}项")

        return self

    def transform(self, X_df):
        """用训练集的scaler和selected_features转换数据"""
        X_scaled = self.scaler.transform(X_df.values)
        X_selected = pd.DataFrame(X_scaled, columns=X_df.columns)[self.selected_features]
        return X_selected.values, self.selected_features

    def fit_transform(self, X_train_df):
        self.fit(X_train_df)
        return self.transform(X_train_df)


class CorrosionTargetTransformer:
    """
    腐蚀电流目标变量变换器：
    - 输入: log10(Excel的"腐蚀电流*10000"列) 值
    - 训练时: Z-score标准化（仅fit训练集）
    - 预测时: 反Z-score -> 10^result / 10000 -> 原始 A/cm2
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.mean_ = None
        self.std_ = None

    def fit(self, y_train_log):
        """y_train_log: log10(I * 10000) 数组"""
        y_train_log = np.asarray(y_train_log).reshape(-1, 1)
        self.scaler.fit(y_train_log)
        self.mean_ = self.scaler.mean_[0]
        self.std_ = self.scaler.scale_[0]
        return self

    def transform(self, y_log):
        """log值 -> Z-score标准化"""
        y_log = np.asarray(y_log).reshape(-1, 1)
        return self.scaler.transform(y_log).flatten()

    def inverse_transform_to_log(self, y_zscore):
        """Z-score -> log10(I*10000)"""
        y_zscore = np.asarray(y_zscore).reshape(-1, 1)
        return self.scaler.inverse_transform(y_zscore).flatten()

    def inverse_transform_to_original(self, y_zscore):
        """Z-score -> 原始 A/cm2"""
        y_log = self.inverse_transform_to_log(y_zscore)
        # 10^log = I * 10000, 再 /10000 = I (A/cm2)
        y_original = np.power(10.0, y_log) / 10000.0
        return y_original

    def log_to_original(self, y_log):
        """log10(I*10000) -> 原始 A/cm2"""
        return np.power(10.0, y_log) / 10000.0
