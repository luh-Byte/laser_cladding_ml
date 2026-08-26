"""
工艺设计器
===========
Rockit485 材料的激光熔覆工艺预测与优化设计工具。

提供三类功能：
1. 正向预测：给定工艺参数 → 预测硬度 + 腐蚀电流
2. 逆向设计：给定目标性能 → 推荐满足条件的工艺参数
3. 约束寻优：固定部分参数 → 在其余维度上搜索最优解

使用方式：
    python -m src.process_designer predict --power 1500 --speed 15 --feed 10 --spot 3.0 --defocus 0
    python -m src.process_designer batch --input my_params.xlsx
    python -m src.process_designer reverse --min_hardness 550 --max_corrosion 3.5e-7
    python -m src.process_designer optimize --fix power=1800 --maximize hardness
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import PROCESS_FEATURES, RESULT_DIR
from src.evaluation import load_model_bundle
from src.pareto_optimization import (
    build_feature_matrix, predict_hardness,
    latin_hypercube_sampling, pareto_front,
)
from src.fewshot_finetune import ROCKIT485_COMPOSITION, ROCKIT485_EXPERIMENTS
from src.fewshot_pareto import predict_corrosion_calibrated


# ============================================================
# 模型加载（全局单例）
# ============================================================
_h_bundle = None
_c_bundle = None


def _get_models():
    """懒加载模型，避免重复IO"""
    global _h_bundle, _c_bundle
    if _h_bundle is None:
        _h_bundle = load_model_bundle("LightGBM_fewshot", "硬度")
    if _c_bundle is None:
        _c_bundle = load_model_bundle("LightGBM_fewshot", "腐蚀电流")
    return _h_bundle, _c_bundle


# ============================================================
# 1. 正向预测
# ============================================================

def predict_process(power, speed, feed_rate, spot_diameter, defocus,
                    composition=None):
    """
    预测单组工艺参数对应的硬度和腐蚀电流。

    参数:
        power: float, 激光功率 (W)
        speed: float, 扫描速度 (mm/s)
        feed_rate: float, 送粉速率 (g/min)
        spot_diameter: float, 光斑直径 (mm)
        defocus: float, 离焦量 (mm)
        composition: dict, 材料成分（默认Rockit485）

    返回:
        dict: {'硬度': float, '腐蚀电流': float, '工艺参数': dict}
    """
    if composition is None:
        composition = ROCKIT485_COMPOSITION

    h_bundle, c_bundle = _get_models()

    params = pd.DataFrame([{
        "激光功率": power,
        "扫描速度": speed,
        "送粉速率": feed_rate,
        "光斑直径": spot_diameter,
        "离焦量": defocus,
    }])

    X = build_feature_matrix(params[PROCESS_FEATURES], composition, h_bundle)
    hardness = predict_hardness(X, h_bundle)[0]
    corrosion = predict_corrosion_calibrated(X, c_bundle)[0]

    return {
        "工艺参数": {
            "激光功率(W)": power,
            "扫描速度(mm/s)": speed,
            "送粉速率(g/min)": feed_rate,
            "光斑直径(mm)": spot_diameter,
            "离焦量(mm)": defocus,
        },
        "硬度(HV)": round(hardness, 1),
        "腐蚀电流(A/cm²)": corrosion,
    }


def predict_batch(param_list, composition=None):
    """
    批量预测多组工艺参数。

    参数:
        param_list: list of dict, 每个dict包含5个工艺参数
        composition: dict, 材料成分（默认Rockit485）

    返回:
        DataFrame: 工艺参数 + 预测结果
    """
    if composition is None:
        composition = ROCKIT485_COMPOSITION

    h_bundle, c_bundle = _get_models()
    df = pd.DataFrame(param_list)

    X = build_feature_matrix(df[PROCESS_FEATURES], composition, h_bundle)
    df["硬度(HV)"] = np.round(predict_hardness(X, h_bundle), 1)
    df["腐蚀电流(A/cm²)"] = predict_corrosion_calibrated(X, c_bundle)

    return df


# ============================================================
# 2. 逆向设计 — 目标驱动的工艺推荐
# ============================================================

def reverse_design(min_hardness=None, max_corrosion=None,
                   search_space=None, n_samples=100000, top_k=10,
                   composition=None):
    """
    逆向设计：给定性能目标，搜索满足条件的工艺参数。

    参数:
        min_hardness: float, 最低硬度要求 (HV)，None则不约束
        max_corrosion: float, 最高腐蚀电流要求 (A/cm²)，None则不约束
        search_space: dict, 搜索范围（默认基于数据范围的合理工程区间）
        n_samples: int, 采样数
        top_k: int, 返回多少个最优解
        composition: dict, 材料成分

    返回:
        DataFrame: 满足条件的工艺参数（按综合评分排序）
    """
    if composition is None:
        composition = ROCKIT485_COMPOSITION
    if search_space is None:
        search_space = {
            "激光功率":   (800, 3000),
            "扫描速度":   (5, 30),
            "送粉速率":   (3, 25),
            "光斑直径":   (1.5, 5.0),
            "离焦量":     (-2.5, 2.5),
        }

    h_bundle, c_bundle = _get_models()

    # 采样
    samples = latin_hypercube_sampling(search_space, n_samples, seed=42)
    X = build_feature_matrix(samples[PROCESS_FEATURES], composition, h_bundle)
    h_pred = predict_hardness(X, h_bundle)
    c_pred = predict_corrosion_calibrated(X, c_bundle)

    result = samples[PROCESS_FEATURES].copy()
    result["硬度(HV)"] = h_pred
    result["腐蚀电流(A/cm²)"] = c_pred

    # 筛选满足条件的
    mask = pd.Series([True] * len(result))
    if min_hardness is not None:
        mask &= result["硬度(HV)"] >= min_hardness
    if max_corrosion is not None:
        mask &= result["腐蚀电流(A/cm²)"] <= max_corrosion

    candidates = result[mask].copy()

    if len(candidates) == 0:
        print(f"[警告] 没有找到满足条件的工艺参数（硬度≥{min_hardness}, 腐蚀≤{max_corrosion}）")
        print(f"  当前搜索范围内：硬度最高 {result['硬度(HV)'].max():.0f}HV, "
              f"腐蚀最低 {result['腐蚀电流(A/cm²)'].min():.2e}")
        return pd.DataFrame()

    # 综合评分（硬度越高越好，腐蚀越低越好，归一化后等权）
    h_min, h_max = candidates["硬度(HV)"].min(), candidates["硬度(HV)"].max()
    c_min, c_max = candidates["腐蚀电流(A/cm²)"].min(), candidates["腐蚀电流(A/cm²)"].max()

    h_range = h_max - h_min if h_max > h_min else 1.0
    c_range = c_max - c_min if c_max > c_min else 1.0

    candidates["评分_硬度"] = (candidates["硬度(HV)"] - h_min) / h_range
    candidates["评分_耐腐蚀"] = (c_max - candidates["腐蚀电流(A/cm²)"]) / c_range
    candidates["综合评分"] = 0.5 * candidates["评分_硬度"] + 0.5 * candidates["评分_耐腐蚀"]

    candidates = candidates.sort_values("综合评分", ascending=False).head(top_k)
    candidates = candidates.reset_index(drop=True)
    candidates.index = candidates.index + 1  # 从1开始编号

    return candidates


# ============================================================
# 3. 约束寻优 — 固定部分参数，优化其余
# ============================================================

def constrained_optimize(fixed_params, objective="hardness",
                         search_space=None, n_samples=50000,
                         composition=None):
    """
    约束优化：固定部分工艺参数，在其余维度上搜索最优解。

    参数:
        fixed_params: dict, 固定的参数 {参数名: 值}
        objective: str, 优化目标
            - 'hardness': 最大化硬度
            - 'corrosion': 最小化腐蚀电流
            - 'both': 帕累托多目标优化
        search_space: dict, 搜索范围
        n_samples: int, 采样数
        composition: dict, 材料成分

    返回:
        DataFrame: 最优工艺参数及预测性能
    """
    if composition is None:
        composition = ROCKIT485_COMPOSITION
    if search_space is None:
        search_space = {
            "激光功率":   (800, 3000),
            "扫描速度":   (5, 30),
            "送粉速率":   (3, 25),
            "光斑直径":   (1.5, 5.0),
            "离焦量":     (-2.5, 2.5),
        }

    h_bundle, c_bundle = _get_models()

    # 确定自由参数
    free_params = [p for p in PROCESS_FEATURES if p not in fixed_params]
    free_bounds = {p: search_space[p] for p in free_params}

    print(f"[约束优化] 固定参数: {fixed_params}")
    print(f"[约束优化] 自由参数: {free_params}")
    print(f"[约束优化] 优化目标: {objective}")
    print(f"[约束优化] 采样数: {n_samples}")

    # 采样自由参数
    samples = latin_hypercube_sampling(free_bounds, n_samples, seed=42)

    # 填充固定参数
    for param, val in fixed_params.items():
        samples[param] = val

    # 预测
    X = build_feature_matrix(samples[PROCESS_FEATURES], composition, h_bundle)
    h_pred = predict_hardness(X, h_bundle)
    c_pred = predict_corrosion_calibrated(X, c_bundle)

    result = samples[PROCESS_FEATURES].copy()
    result["硬度(HV)"] = h_pred
    result["腐蚀电流(A/cm²)"] = c_pred

    if objective == "hardness":
        best = result.iloc[result["硬度(HV)"].argmax()]
        return pd.DataFrame([best]).reset_index(drop=True)
    elif objective == "corrosion":
        best = result.iloc[result["腐蚀电流(A/cm²)"].argmin()]
        return pd.DataFrame([best]).reset_index(drop=True)
    elif objective == "both":
        pareto = pareto_front(result, [("硬度(HV)", "max"), ("腐蚀电流(A/cm²)", "min")])
        pareto = pareto.sort_values("硬度(HV)", ascending=False).reset_index(drop=True)
        return pareto
    else:
        raise ValueError(f"未知优化目标: {objective}，可选: hardness/corrosion/both")


# ============================================================
# 命令行接口
# ============================================================

def _print_result(result_dict):
    """美观打印单组预测结果"""
    print()
    print("=" * 60)
    print("工艺参数预测结果")
    print("=" * 60)
    print("【工艺参数】")
    for k, v in result_dict["工艺参数"].items():
        print(f"  {k}: {v}")
    print()
    print("【预测性能】")
    print(f"  显微硬度:  {result_dict['硬度(HV)']:.1f} HV")
    print(f"  腐蚀电流:  {result_dict['腐蚀电流(A/cm²)']:.2e} A/cm²")
    print("=" * 60)


def cmd_predict(args):
    """正向预测命令"""
    result = predict_process(
        power=args.power,
        speed=args.speed,
        feed_rate=args.feed,
        spot_diameter=args.spot,
        defocus=args.defocus,
    )
    _print_result(result)


def cmd_batch(args):
    """批量预测命令"""
    if args.input.endswith(".xlsx") or args.input.endswith(".xls"):
        df = pd.read_excel(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        print(f"不支持的文件格式: {args.input}")
        return

    # 检查必要列
    required = set(PROCESS_FEATURES)
    cols = set(df.columns)
    missing = required - cols
    if missing:
        print(f"[错误] 输入文件缺少必要列: {missing}")
        print(f"  需要的列: {PROCESS_FEATURES}")
        return

    param_list = df[PROCESS_FEATURES].to_dict("records")
    result = predict_batch(param_list)

    # 保存
    out_path = args.output or "process_predictions.xlsx"
    result.to_excel(out_path, index=False)
    print(f"批量预测完成，共 {len(result)} 组")
    print(f"结果已保存至: {out_path}")
    print()
    print(result.head(min(10, len(result))).to_string(index=False))


def cmd_reverse(args):
    """逆向设计命令"""
    min_h = args.min_hardness
    max_c = args.max_corrosion

    if min_h is None and max_c is None:
        print("[错误] 请至少指定一个目标：--min-hardness 或 --max-corrosion")
        return

    result = reverse_design(
        min_hardness=min_h,
        max_corrosion=max_c,
        top_k=args.top_k,
    )

    if len(result) == 0:
        return

    print()
    print("=" * 80)
    print(f"逆向设计结果（Top {len(result)}，满足 硬度≥{min_h}HV, 腐蚀≤{max_c}）")
    print("=" * 80)
    display_cols = PROCESS_FEATURES + ["硬度(HV)", "腐蚀电流(A/cm²)", "综合评分"]
    print(result[display_cols].to_string())

    # 保存
    out_dir = os.path.join(RESULT_DIR, "reverse_design")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"reverse_h{min_h}_c{max_c}.xlsx")
    result.to_excel(out_path, index=False)
    print(f"\n结果已保存至: {out_path}")


def cmd_optimize(args):
    """约束优化命令"""
    fixed = {}
    if args.fix:
        for item in args.fix:
            key, val = item.split("=")
            fixed[key.strip()] = float(val.strip())

    result = constrained_optimize(
        fixed_params=fixed,
        objective=args.objective,
    )

    print()
    print("=" * 80)
    if args.objective == "both":
        print(f"约束帕累托优化结果（共 {len(result)} 个最优解）")
    else:
        print(f"约束优化结果（目标: {args.objective}）")
    print("=" * 80)

    display_cols = PROCESS_FEATURES + ["硬度(HV)", "腐蚀电流(A/cm²)"]
    print(result[display_cols].to_string(index=False))

    # 保存
    out_dir = os.path.join(RESULT_DIR, "constrained_opt")
    os.makedirs(out_dir, exist_ok=True)
    fixed_str = "_".join([f"{k}{v}" for k, v in fixed.items()]) or "none"
    out_path = os.path.join(out_dir, f"opt_{args.objective}_fix{fixed_str}.xlsx")
    result.to_excel(out_path, index=False)
    print(f"\n结果已保存至: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rockit485 激光熔覆工艺设计器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 1. 单组工艺预测
  python -m src.process_designer predict --power 1500 --speed 15 --feed 10 --spot 3.0 --defocus 0

  # 2. 批量预测（从Excel/CSV读取）
  python -m src.process_designer batch --input my_params.xlsx --output predictions.xlsx

  # 3. 逆向设计：找硬度≥550HV 且 腐蚀≤3.5e-7 A/cm² 的工艺
  python -m src.process_designer reverse --min-hardness 550 --max-corrosion 3.5e-7

  # 4. 约束优化：固定功率1800W，最大化硬度
  python -m src.process_designer optimize --fix 激光功率=1800 --objective hardness

  # 5. 约束优化：固定功率+扫描速度，帕累托寻优
  python -m src.process_designer optimize --fix 激光功率=1500 --fix 扫描速度=15 --objective both
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="功能")

    # --- predict ---
    p_pred = subparsers.add_parser("predict", help="单组工艺参数预测")
    p_pred.add_argument("--power", type=float, required=True, help="激光功率 (W)")
    p_pred.add_argument("--speed", type=float, required=True, help="扫描速度 (mm/s)")
    p_pred.add_argument("--feed", type=float, required=True, help="送粉速率 (g/min)")
    p_pred.add_argument("--spot", type=float, required=True, help="光斑直径 (mm)")
    p_pred.add_argument("--defocus", type=float, required=True, help="离焦量 (mm)")

    # --- batch ---
    p_batch = subparsers.add_parser("batch", help="批量预测（从Excel/CSV读取）")
    p_batch.add_argument("--input", type=str, required=True, help="输入文件路径 (.xlsx/.csv)")
    p_batch.add_argument("--output", type=str, default=None, help="输出文件路径")

    # --- reverse ---
    p_rev = subparsers.add_parser("reverse", help="逆向设计：给定目标找工艺")
    p_rev.add_argument("--min-hardness", type=float, default=None, help="最低硬度要求 (HV)")
    p_rev.add_argument("--max-corrosion", type=float, default=None, help="最高腐蚀电流 (A/cm²)")
    p_rev.add_argument("--top-k", type=int, default=10, help="返回Top K个解")

    # --- optimize ---
    p_opt = subparsers.add_parser("optimize", help="约束优化：固定参数+寻优其余")
    p_opt.add_argument("--fix", action="append", default=[],
                       help='固定参数，格式："参数名=值"，可多次使用')
    p_opt.add_argument("--objective", type=str, default="hardness",
                       choices=["hardness", "corrosion", "both"],
                       help="优化目标: hardness(最大硬度)/corrosion(最小腐蚀)/both(帕累托)")

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "reverse":
        cmd_reverse(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
