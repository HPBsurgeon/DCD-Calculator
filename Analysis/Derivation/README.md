# Derivation Analysis

This directory contains scripts for exploratory and interpretive analysis of the LightGBM model developed for DCD prediction.

## Scripts

- `train_split_model.py`: Train/test split with LightGBM, shows feature importance (gain/split).
- `variable_effect_plot.py`: Visualizes predicted probability changes as a single variable is varied.
- `shap_analysis.py`: Computes SHAP values and shows summary plot.
- `repeated_split_auc.py`: Runs multiple train/test splits and calculates average AUC.

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
