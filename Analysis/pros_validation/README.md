# Prospective Validation

This directory contains the script for validating the LightGBM model on a prospective dataset.

## Files

- `pros_validation.py`: Predicts outcomes, evaluates AUC, accuracy, FPR/FNR, and plots performance by prediction bins.
- `data/prospective_data.csv`: (Optional) Example input data.
- `../train_model.py`: Pre-trained model (must be run before using this script).

## Usage

1. Make sure the model in `train_model.py` is trained or imported.
2. Place your prospective dataset (CSV format) in the `data/` directory.
3. Run the script:

```bash
python pros_validation.py
