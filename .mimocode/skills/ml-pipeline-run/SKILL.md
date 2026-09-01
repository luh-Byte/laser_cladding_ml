---
name: ml-pipeline-run
description: "Run the complete laser cladding ML pipeline: train models, fine-tune, optimize Pareto, validate, and generate all 43 figures. Use when user asks to run the full pipeline, regenerate all outputs, or retrain models."
---

# ML Pipeline Full Run

Execute the complete laser cladding ML pipeline in the correct dependency order.

## Prerequisites

- Virtual environment activated (`.venv/`)
- Raw data at `data/raw_data.xlsx`
- Working directory: project root

## Pipeline Steps

Run each step sequentially. **Do not skip steps** — later steps depend on earlier outputs.

### Layer 1: Training (no inter-dependencies, but run sequentially for memory safety)

```powershell
.venv\Scripts\python.exe -m src.train
```
- Duration: ~15-20 min (10 repetitions × 4 models × 5-fold CV)
- Outputs: predictions, model_summary.xlsx, SHAP baseline data, model weights
- Note: Uses `n_jobs=2` (not -1) to avoid OOM on 4-core/16GB machines

```powershell
.venv\Scripts\python.exe -m src.train_final
```
- Duration: ~30s
- Outputs: bundle_LightGBM_硬度.pkl, bundle_RF_腐蚀电流.pkl

```powershell
.venv\Scripts\python.exe -m src.fewshot
```
- Duration: ~1min
- Outputs: fewshot bundles, SHAP fewshot data

### Layer 2: Optimization & Validation (depends on Layer 1)

```powershell
.venv\Scripts\python.exe -m src.pareto
```
- Duration: ~2min (375,000 samples)
- Outputs: pareto_global_front.csv, representative_params.csv

```powershell
.venv\Scripts\python.exe -m src.pareto_fewshot
```
- Duration: ~2min
- Outputs: fewshot pareto results

```powershell
.venv\Scripts\python.exe -m src.validate_loo
```
- Duration: ~2min
- Outputs: loo_results.csv, loo_metrics_summary.csv

```powershell
.venv\Scripts\python.exe -m src.validate_exp
```
- Duration: ~30s
- Outputs: experimental_validation.csv

### Layer 3: Figures (depends on all above, no inter-dependency between figure scripts)

```powershell
.venv\Scripts\python.exe -m src.plot_main
```
- Duration: ~30s
- Outputs: 19 figures (01-19_*.png)

```powershell
.venv\Scripts\python.exe -m src.plot_extra
```
- Duration: ~30s
- Outputs: 6 figures (14-19_*.png)

```powershell
.venv\Scripts\python.exe -m src.plot_shap
```
- Duration: ~30s
- Outputs: 24 SHAP figures

## Total Duration

~25-30 minutes end-to-end on i5-11300H / 16GB RAM.

## Output Structure

```
outputs/
├── figures/           # 19 main + 6 extra figures
│   └── shap/          # 24 SHAP figures
├── models/            # Model bundles (pkl)
├── results/
│   ├── model_summary.xlsx
│   ├── predictions/
│   ├── feature_importance/
│   ├── pareto/
│   ├── pareto_rockit485_fewshot/
│   └── logs/
├── shap_baseline/     # SHAP intermediate data
├── shap_fewshot/      # Fewshot SHAP data
└── loo_validation/    # LOO results
```

## Quick Single-Figure Regeneration

For visual tweaks without re-running the full pipeline:

```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); from src.plot_main import plot_stacked_composition, load_all_data; data = load_all_data(); plot_stacked_composition(data)"
```

Replace `plot_stacked_composition` with the function name from `plot_main.py`, `plot_extra.py`, or `plot_shap.py`.

## Troubleshooting

- **TerminatedWorkerError**: Reduce `n_jobs` in `src/models.py` (currently set to 2)
- **FileNotFoundError**: Run Layer 1 before Layer 2
- **scipy TypeError**: Ensure scipy < 1.14 (Python 3.11rc2 compatibility)
