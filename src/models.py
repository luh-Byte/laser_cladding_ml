"""
模型训练模块
=============
四模型依次训练（固定顺序 KNN -> SVR -> RF -> LightGBM），均使用5折网格搜索CV（训练集专属）。
每个模型输出: 最优参数、最优模型、训练/测试集预测值。
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from src.config import HYPERPARAM_GRIDS, CV_FOLDS, BASE_RANDOM_SEED


def train_knn(X_train, y_train):
    """KNN: K近邻回归，距离加权"""
    model = KNeighborsRegressor()
    param_grid = HYPERPARAM_GRIDS["KNN"]
    grid = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring="r2",
        n_jobs=2, return_train_score=True
    )  # type: ignore[arg-type]
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"  [KNN] 最优参数: {grid.best_params_}, CV R2: {grid.best_score_:.4f}")
    return best_model, grid.best_params_, grid.best_score_


def train_svr(X_train, y_train):
    """SVR: 支持向量回归，RBF核"""
    model = SVR()
    param_grid = HYPERPARAM_GRIDS["SVR"]
    grid = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring="r2",
        n_jobs=2, return_train_score=True
    )  # type: ignore[arg-type]
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"  [SVR] 最优参数: {grid.best_params_}, CV R2: {grid.best_score_:.4f}")
    return best_model, grid.best_params_, grid.best_score_


def train_rf(X_train, y_train):
    """随机森林: 核心可解释模型"""
    model = RandomForestRegressor(random_state=BASE_RANDOM_SEED)
    param_grid = HYPERPARAM_GRIDS["RF"]
    grid = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring="r2",
        n_jobs=2, return_train_score=True
    )  # type: ignore[arg-type]
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"  [RF] 最优参数: {grid.best_params_}, CV R2: {grid.best_score_:.4f}")
    return best_model, grid.best_params_, grid.best_score_


def train_lightgbm(X_train, y_train):
    """LightGBM: 精度标杆模型"""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        random_state=BASE_RANDOM_SEED,
        verbose=-1,
        force_col_wise=True,
    )
    param_grid = HYPERPARAM_GRIDS["LightGBM"]
    grid = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring="r2",
        n_jobs=2, return_train_score=True
    )  # type: ignore[arg-type]
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"  [LightGBM] 最优参数: {grid.best_params_}, CV R2: {grid.best_score_:.4f}")
    return best_model, grid.best_params_, grid.best_score_


# 模型训练函数映射
MODEL_TRAINERS = {
    "KNN": train_knn,
    "SVR": train_svr,
    "RF": train_rf,
    "LightGBM": train_lightgbm,
}


def train_model(model_name, X_train, y_train):
    """
    统一训练接口。
    返回: (best_model, best_params, cv_score)
    """
    trainer = MODEL_TRAINERS[model_name]
    return trainer(X_train, y_train)


def get_feature_importance(model, model_name, feature_names):
    """
    提取特征重要性（仅RF和LightGBM支持）。
    返回: dict {feature_name: importance}
    """
    if model_name == "RF":
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    elif model_name == "LightGBM":
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    else:
        return None


def get_permutation_importance(model, X_test, y_test, feature_names, scoring="r2"):
    """排列重要性（适用于所有模型）"""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        random_state=BASE_RANDOM_SEED, scoring=scoring
    )
    return dict(zip(feature_names, result.importances_mean))  # type: ignore[attr-defined]
