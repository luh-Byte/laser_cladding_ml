# 激光熔覆涂层性能预测 — 机器学习工作流（Excel列版本）

## 项目概述

基于激光熔覆工艺参数和粉末成分，预测两项关键性能指标：
- **显微硬度** (HV)
- **腐蚀电流密度** (A/cm²)

> 本版本直接使用Excel中的"腐蚀电流*10000"列作为腐蚀电流的中间表示，不再在代码中计算。

核心技术路线：标准ML训练管道 → 双目标联合分层抽样 → Few-shot微调（Rockit485专属）→ SHAP可解释性分析 → 帕累托多目标优化。

## 环境要求

| 依赖 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.10+ | 运行环境 |
| pandas | 2.0+ | 数据处理 |
| numpy | 1.24+ | 数值计算 |
| scikit-learn | 1.3+ | 模型/预处理 |
| lightgbm | 4.0+ | 梯度提升模型 |
| openpyxl | 3.1+ | 读取Excel |
| matplotlib | 3.7+ | 绘图 |
| statsmodels | 0.14+ | LOWESS平滑 |
| shap | 0.45+ | SHAP可解释性分析 |

## 安装步骤

```bash
# 1. 解压项目
unzip laser_cladding_ml.zip
cd laser_cladding_ml

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

## 目录结构

```
laser_cladding_ml/
├── data/
│   └── raw_data.xlsx              # 原始实验数据（215条）
├── src/
│   ├── config.py                  # 全局配置（路径/种子/清洗阈值/特征/超参）
│   ├── data_preprocessing.py      # 数据加载/清洗/双目标分层抽样
│   ├── feature_engineering.py     # 6项物理衍生特征/标准化/共线性筛选
│   ├── models.py                  # 4种模型定义+超参搜索
│   ├── train.py                  # 主训练管道（10次重复分层实验+SHAP）
│   ├── train_final.py            # 全量数据最终模型训练
│   ├── evaluation.py             # 指标计算/模型保存/日志
│   ├── pareto_optimization.py    # 通用逐维扫描帕累托优化
│   ├── validate_experiment.py    # 实验点验证
│   ├── shap_utils.py             # SHAP中间数据计算（仅数据，不绘图）
│   ├── fewshot_finetune.py       # Few-shot微调（Rockit485）+ SHAP
│   ├── fewshot_pareto.py         # Rockit485专属帕累托优化
│   ├── process_designer.py       # 工艺设计器（预测/逆向/约束优化）
│   ├── loo_validation.py         # Rockit485留一法(LOO)验证
│   ├── plot_style.py              # 统一绘图样式（渐变背景/字体/画布）
│   ├── plot_figures.py            # 数据可视化（10张图，300DPI，全英文）
│   ├── plot_shap.py              # SHAP可视化（24张核心图，全英文）
│   └── loo_validation.py         # Rockit485留一法(LOO)验证
├── outputs/
│   ├── models/                    # 训练好的模型权重
│   ├── results/                   # 结果文件（CSV/Excel）
│   ├── figures/                   # 10张数据可视化图（300DPI PNG）
│   ├── figures/shap/              # 24张SHAP可视化图（300DPI PNG）
│   ├── shap_baseline/             # 基线模型SHAP中间数据
│   └── shap_fewshot/              # Few-shot模型SHAP中间数据
├── requirements.txt
└── README.md                      # 本文件
```

## 复现步骤（按顺序执行）

### 步骤1：基线模型训练 + SHAP

训练4种模型（KNN → SVR → RF → LightGBM），10次重复双目标分层抽样。训练完成后自动计算基线RF模型的SHAP中间数据。

```bash
python -m src.train
```

**输出：**
- `outputs/results/model_summary.xlsx` — 4模型×2目标的R²/MAE/RMSE汇总
- `outputs/results/logs/iteration_log.csv` — 10次迭代详细日志
- `outputs/results/predictions/` — 第1次实验的预测值对比
- `outputs/results/feature_importance/` — RF/LightGBM特征重要性
- `outputs/models/model_RF_硬度.pkl` — 最佳基线硬度模型
- `outputs/models/model_RF_腐蚀电流.pkl` — 最佳基线腐蚀模型
- `outputs/shap_baseline/` — 基线RF模型SHAP中间数据（9个文件）

**预期结果（10次均值）：**

| 目标 | 最佳模型 | 测试集R² | 测试集MAE |
|------|---------|---------|----------|
| 硬度 | RF | 0.64 ± 0.08 | 98.7 HV |
| 腐蚀电流(log) | RF | 0.32 ± 0.14 | — |

### 步骤2：Few-shot微调训练 + SHAP

用Rockit485的4个实验点（20倍权重）微调LightGBM模型，纠正预测趋势。训练完成后自动计算Few-shot模型的SHAP中间数据。

```bash
python -m src.fewshot_finetune
```

**输出：**
- `outputs/models/bundle_LightGBM_fewshot_硬度.pkl` — 微调硬度模型
- `outputs/models/bundle_LightGBM_fewshot_腐蚀电流.pkl` — 微调腐蚀模型（含校准因子）
- `outputs/shap_fewshot/` — Few-shot模型SHAP中间数据（5个文件）

**预期结果：**

| 目标 | 趋势正确 | 误差 |
|------|---------|------|
| 硬度 | ✓ 峰值1800W | 平均0.8% |
| 腐蚀 | ✓ 最优1500W | 校准因子0.9768 |

硬度预测对比：

| 功率(W) | 实测(HV) | 预测(HV) | 相对误差 |
|---------|----------|----------|----------|
| 1200 | 470 | 474.2 | 0.9% |
| 1500 | 500 | 497.9 | 0.4% |
| 1800 | 520 | 511.5 | 1.6% |
| 2100 | 480 | 481.6 | 0.3% |

腐蚀预测对比（校准后）：

| 功率(W) | 实测(A/cm²) | 预测(A/cm²) | 倍数 |
|---------|-------------|-------------|------|
| 1200 | 5.95e-07 | 5.71e-07 | 0.96x |
| 1500 | 3.12e-07 | 3.33e-07 | 1.07x |
| 1800 | 3.48e-07 | 3.63e-07 | 1.04x |
| 2100 | 4.50e-07 | 4.21e-07 | 0.94x |

### 步骤3：帕累托多目标优化

使用微调后的模型，对Rockit485材料生成帕累托最优工艺参数空间。

```bash
python -m src.fewshot_pareto
```

**输出：**
- `outputs/results/pareto_rockit485_fewshot/pareto_global_front.csv` — 整体帕累托前沿
- `outputs/results/pareto_rockit485_fewshot/pareto_representative_params.csv` — 代表性参数组合

**扫描方式：** 逐维扫描（5个工艺参数 × 5个固定点 × 75000个自由采样 = 37.5万采样点）

### 步骤4：工艺设计器（预测 / 逆向设计 / 约束优化）

Rockit485 专属工艺设计工具，支持三种使用模式：

```bash
# 模式1：单组工艺预测
python -m src.process_designer predict --power 1600 --speed 12 --feed 8 --spot 2.8 --defocus -0.5

# 模式2：逆向设计（给定性能目标，推荐工艺）
python -m src.process_designer reverse --min-hardness 550 --max-corrosion 3.5e-7

# 模式3：约束优化（固定部分参数，优化其余）
python -m src.process_designer optimize --fix 激光功率=1800 --objective both

# 模式4：批量预测（从Excel读取多组参数）
python -m src.process_designer batch --input my_params.xlsx --output predictions.xlsx
```

**输出：**
- `outputs/results/reverse_design/` — 逆向设计结果（Excel）
- `outputs/results/constrained_opt/` — 约束优化结果（Excel）

### 步骤5：留一法验证（LOO）

对Rockit485的4个实验样本进行留一法交叉验证，对比基线模型与LOO模型的预测效果。

```bash
python -m src.loo_validation
```

**输出：**
- `outputs/loo_validation/loo_results.csv` — 每个样本的实测/基线预测/LOO预测
- `outputs/loo_validation/loo_metrics_summary.csv` — 汇总指标（MAE/MAPE/GMR）

**预期结果：**

| 目标 | 基线MAPE | LOO MAPE | 改善 |
|------|---------|---------|------|
| 硬度 | 17.6% | 6.3% | 64%↓ |
| 腐蚀 | 26.5x | 0.99x | 96%↓ |

### 步骤6（可选）：基线实验验证

用基线模型（非微调）验证实验点。

```bash
python -m src.validate_experiment
```

### 步骤7（可选）：全量数据最终模型

用全部196条数据训练最终模型（不分训练/测试集）。

```bash
python -m src.train_final
```

### 步骤8：数据可视化绘图

生成全部34张论文级图表（10张数据图 + 24张SHAP图），所有文字均为英文。

```bash
# 10张数据可视化图
python -m src.plot_figures

# 24张SHAP可视化图
python -m src.plot_shap
```

**输出：**
- `outputs/figures/` — 10张数据可视化图（300DPI PNG）
- `outputs/figures/shap/` — 24张SHAP可视化图（300DPI PNG）

## SHAP 中间数据说明

训练流水线只产出SHAP中间数据（npy/csv/pkl），不生成任何图片。所有绘图需使用独立的外部脚本。

### 基线SHAP → `outputs/shap_baseline/`

| 文件 | 格式 | 形状/大小 | 说明 |
|------|------|----------|------|
| `best_rf_hv.pkl` | pickle | — | 最优硬度RF模型（10次中表现最好的那一轮） |
| `best_rf_corr.pkl` | pickle | — | 最优腐蚀RF模型 |
| `X_train_hv.csv` | CSV | 156行×13列 | 硬度训练集特征矩阵（标准化后） |
| `X_train_corr.csv` | CSV | 156行×13列 | 腐蚀训练集特征矩阵 |
| `shap_vals_hv.npy` | npy | (156, 13) | 硬度SHAP值 |
| `shap_vals_corr.npy` | npy | (156, 13) | 腐蚀SHAP值（log空间） |
| `shap_inter_hv.npy` | npy | (156, 13, 13) | 硬度SHAP交互值 |
| `shap_inter_corr.npy` | npy | (156, 13, 13) | 腐蚀SHAP交互值 |
| `shap_import_baseline.csv` | CSV | 13行 | SHAP重要性 + 模型原生重要性 |

### Few-shot SHAP → `outputs/shap_fewshot/`

| 文件 | 格式 | 形状/大小 | 说明 |
|------|------|----------|------|
| `X_fs.csv` | CSV | 200行×13列 | 完整训练特征矩阵（196基础+4×20重复Rockit） |
| `shap_vals_hv_fs.npy` | npy | (200, 13) | 硬度SHAP值 |
| `shap_vals_corr_fs.npy` | npy | (200, 13) | 腐蚀SHAP值（log空间） |
| `shap_import_fewshot.csv` | CSV | 13行 | SHAP重要性统计 |
| `rockit_sample_index.csv` | CSV | 4行 | Rockit485样本行号（196~199） |

> 注意：腐蚀SHAP值对应log10(I×10000)空间，量级远小于硬度SHAP值（HV量纲），属正常现象。

## 绘图方案

SHAP数据支持以下图表，按优先级排序。绘图脚本需独立编写（不包含在训练流水线中）。

### 统一绘图样式 (`plot_style.py`)

所有图表使用统一样式模块：
- **字体**：Liberation Serif（Times New Roman 度量兼容替代），全英文加粗
- **背景**：渐变色 #F8F9E4 → #E5F2FB
- **图框**：加粗黑色边框（2.0pt）
- **画布**：统一三种尺寸（8×6 / 14×6 / 10×8 inch）
- **分辨率**：300 DPI PNG
- **文字语言**：全部英文（特征名、轴标签、图例等）
- **色板**：#E5F2FB #F8F9E4 #E97A6F #E8156E #E0BEB3 #00BEB3 #728BDE #E246C9 #834BD4 #CFCFCF

### 绘图脚本

```bash
# 10张数据可视化图
python -m src.plot_figures

# 24张SHAP可视化图
python -m src.plot_shap
```

**输出目录：**
- `outputs/figures/` — 10张数据可视化图
- `outputs/figures/shap/` — 24张SHAP可视化图

**10张数据图清单：**

| 序号 | 文件名 | 内容 |
|------|--------|------|
| 1 | 01_correlation_heatmap.png | 相关性矩阵热图 |
| 2 | 02_model_comparison_bar.png | 模型性能对比柱状图 |
| 3 | 03_pca_scatter.png | PCA降维聚类散点图 |
| 4 | 04_violin_by_cr.png | 硬度按Cr分组小提琴图 |
| 5 | 05_stacked_composition.png | 材料成分堆叠柱状图 |
| 6 | 06_pareto_bubble.png | 帕累托前沿散点气泡图 |
| 7 | 07_predicted_vs_actual.png | 预测vs实测散点图 |
| 8 | 08_hardness_power_trend.png | 硬度vs功率趋势图 |
| 9 | 09_error_boxplot.png | 模型误差箱线图 |
| 10 | 10_shap_bubble.png | SHAP重要性差异气泡图 |

**24张SHAP图清单：**

| 类别 | 数量 | 内容 |
|------|------|------|
| Beeswarm | 4 | 基线/Few-shot × 硬度/腐蚀 |
| Waterfall | 8 | 4个Rockit485样本 × 硬度/腐蚀 |
| Dependence | 10 | 硬度Top5 + 腐蚀Top5特征 |
| Importance | 2 | 基线vs Few-shot重要性对比 |

### P0 必画图（论文核心图，约12张）

| 序号 | 图表 | 数据来源 | 说明 |
|------|------|---------|------|
| 1 | SHAP Beeswarm（硬度，基线） | `shap_vals_hv.npy` + `X_train_hv.csv` | 全局特征效应总览 |
| 2 | SHAP Beeswarm（腐蚀，基线） | `shap_vals_corr.npy` + `X_train_corr.csv` | 全局特征效应总览 |
| 3 | SHAP Beeswarm（硬度，Few-shot） | `shap_vals_hv_fs.npy` + `X_fs.csv` | 微调后特征效应 |
| 4 | SHAP Beeswarm（腐蚀，Few-shot） | `shap_vals_corr_fs.npy` + `X_fs.csv` | 微调后特征效应 |
| 5-8 | Waterfall Plot（4个Rockit样本，硬度） | 对应行的SHAP值 | 单样本预测解释 |
| 9-12 | Waterfall Plot（4个Rockit样本，腐蚀） | 对应行的SHAP值 | 单样本预测解释 |

### P1 重要图（支撑分析，约12张）

| 序号 | 图表 | 数据来源 | 说明 |
|------|------|---------|------|
| 13-17 | Dependence Plot（硬度Top5特征） | SHAP值 + 特征值 | 单特征非线性效应 |
| 18-22 | Dependence Plot（腐蚀Top5特征） | SHAP值 + 特征值 | 单特征非线性效应 |
| 23 | 基线vs Few-shot 硬度重要性对比 | 两份shap_import | 微调前后变化 |
| 24 | 基线vs Few-shot 腐蚀重要性对比 | 两份shap_import | 微调前后变化 |

### P2 补充图（可选，约8张）

| 序号 | 图表 | 数据来源 | 说明 |
|------|------|---------|------|
| 25-27 | Interaction Dependence（硬度Top3交互对） | `shap_inter_hv.npy` | 特征交互效应 |
| 28-30 | Interaction Dependence（腐蚀Top3交互对） | `shap_inter_corr.npy` | 特征交互效应 |
| 31 | SHAP重要性 vs 模型原生重要性对比 | `shap_import_baseline.csv` | 两种重要性验证 |
| 32 | Rockit样本在beeswarm上高亮标注 | Few-shot SHAP + 索引表 | 特殊样本定位 |

合计约32张图，核心图约24张（P0+P1）。

### 材料科学合理性结论

- **硬度**：SHAP方向全部符合冶金学规律（送粉率正效应、扫描速度负效应、C/碳当量正效应等）
- **腐蚀**：Cr负效应、Mn正效应、碳当量正效应均正确；Fe/Ni排名偏高是"合金体系代理效应"（高Fe对应铁基不锈钢体系），需在论文中说明
- **Few-shot微调**：成分特征重要性提升、工艺参数效应更清晰，符合单材料体系预期

## 核心技术细节

### 1. 数据清洗（9条物理准则）

| 规则 | 阈值 | 剔除数 |
|------|------|--------|
| 光斑直径过大 | > 8mm | 8 |
| 扫描速度过大 | > 200mm/s | 4 |
| 元素总含量异常 | <85% 或 >115% | 4 |
| 元素含量为负 | <0 | 0 |
| 硬度异常 | <150 或 >1800 HV | 1 |
| 腐蚀电流过大 | >1e-4 A/cm² | 0 |
| 腐蚀电流伪影 | <=1e-9 A/cm² | 2 |
| 熔高/熔宽比值异常 | <0.05 或 >1.0 | 0 |
| 高功率区域孤立点 | >=4800W&硬度>1500HV 或 >=5500W | 2 |
| NaN值 | — | 0 |

原始215条 → 清洗后194条（剔除9.8%）

### 2. 特征工程（18→13项）

- 5项原始工艺特征：激光功率、扫描速度、送粉速率、光斑直径、离焦量
- 7项成分特征：C, Cr, Si, Ni, Fe, Mn, Mo
- 6项物理衍生特征：线能量密度、面能量密度、粉末能量比、碳当量、铬当量、镍当量
- 共线性筛选（|r|>0.9）：剔除铬当量、面能量密度
- 重要性精简：剔除Si、Mo、Ni
- **最终保留13项特征**

### 3. 双目标联合分层抽样

- 按硬度5分位数 + 腐蚀5分位数联合分层
- 联合分层失败时（最小层<2个样本）自动降级为硬度分层
- 使用StratifiedShuffleSplit保证10次划分的分布一致性

### 4. Few-shot微调

- Rockit485实验数据：4个功率点（1200/1500/1800/2100W）
- 权重策略：Few-shot样本权重=20倍（等效80个样本）
- 标准化仅用基础数据fit（防止Few-shot数据影响分布）
- 硬度模型：LightGBM（max_depth=4, lr=0.05, reg_lambda=10）
- 腐蚀模型：LightGBM（log10+Z-score空间训练）
- 腐蚀校准因子：几何均值反比法

### 5. 帕累托优化

- 目标：最大化硬度 + 最小化腐蚀电流
- 方法：逐维扫描（固定1个参数，采样其余4个）
- 采样：拉丁超立方采样（LHS）
- 帕累托前沿：非支配排序

### 6. SHAP计算

- 基线模型：RF + TreeExplainer，含交互值
- Few-shot模型：LightGBM + TreeExplainer，不含交互值
- 仅输出中间数据（npy/csv/pkl），不调用任何绘图函数
- 腐蚀SHAP在log10(I×10000)空间

## 常见问题

### Q: 运行报错 ModuleNotFoundError: No module named 'sklearn'
```bash
pip install scikit-learn
```

### Q: 运行报错 ModuleNotFoundError: No module named 'shap'
```bash
pip install shap
```

### Q: Excel读取报错
确保 `data/raw_data.xlsx` 存在且未被损坏。文件由openpyxl引擎读取。

### Q: 模型加载报错 KeyError
模型bundle与代码版本不匹配时，删除 `outputs/models/` 下所有 `.pkl` 文件，重新运行步骤1和步骤2。

### Q: SHAP计算很慢
基线SHAP含交互值计算，156样本×13特征约需1~2分钟，属正常速度。

### Q: 想修改实验参数
所有配置集中在 `src/config.py`，修改后重新运行即可。

## Rockit485 材料成分

| 元素 | 含量(wt.%) |
|------|-----------|
| C | 0.15 |
| Cr | 13.0 |
| Si | 0.6 |
| Ni | 4.0 |
| Mo | 2.8 |
| Mn | 0.5 |
| Fe | 余量(≈78.95) |

注：实际粉末含Co(4.5%)+Nb(0.8%)，模型训练数据中无此二元素，通过Few-shot微调+校准因子补偿。
