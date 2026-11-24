# Formula 1 Stint & Lap Time Data Science Pipeline

An end-to-end exploratory data analysis, statistical inference, and lap-time modeling pipeline built on real Formula 1 timing and weather data via [`FastF1`](https://docs.fastf1.dev/).

## Project Overview

In motorsport analytics, raw timing logs contain operational noise and measurement artifacts—such as pit-in/pit-out laps, safety car periods, traffic, and changing track conditions—that can distort naive analyses. This project demonstrates a clean data science workflow:

1. **Data Collection & Noise Filtering (`src/data_loader.py`):** Merges session lap timing with time-aligned weather telemetry (`merge_asof`) and removes pit laps, missing compound/age records, and anomalous slow laps.
2. **Statistical Inference (`src/inference.py`):** Evaluates lap-time differences between tyre compounds using **bootstrap confidence intervals** and **non-parametric permutation testing** without relying on parametric normality assumptions.
3. **Baseline vs. Multivariate Modeling & Residual Analysis (`src/modeling.py`):** Compares a single-feature baseline (`TyreLife`) against a regularized `Ridge` regression pipeline incorporating tyre age, race progression (proxy for fuel burn), track temperature, and compound encoding, inspecting residual structure and error metrics.
4. **Diagnostic Visualization (`src/visualize.py`):** Generates compound distributions, stint degradation trends, and residual-vs-predicted plots.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_analysis.py
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── inference.py
    ├── modeling.py
    └── visualize.py
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_analysis.py --year 2024 --gp Bahrain --session R
```

Generated charts are saved under `./outputs/`, and downloaded session data is cached locally in `./f1_cache/`.

## Visual Outputs

- `outputs/lap_times_by_compound.png`: Clean lap-time spread across available tyre compounds.
- `outputs/tyre_degradation_curve.png`: Mean lap time as a function of tyre age (`TyreLife`) per compound.
- `outputs/residuals_vs_predicted.png`: Residual diagnostic plot to check heteroscedasticity or unmodeled effects (such as car performance differences across teams or fuel load dynamics).

## Key Limitations & Next Steps

- **Car/Team Heterogeneity:** Different constructors have distinct aerodynamic and power-unit baselines; adding team/driver fixed effects or fuel-corrected lap times is a natural next step.
- **Multi-Race Generalization:** Extending from a single Grand Prix to full-season cross-validation tests how well degradation patterns generalize across circuit layouts.