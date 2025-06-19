# train_model.py

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from utils.prepare_data import load_and_prepare_data

# Load and preprocess training and validation data
x, t, x_vali, t_vali = load_and_prepare_data("your_input_data.csv")  # Replace with actual CSV path

# Store models and AUC scores from each Optuna trial
trial_models = []

def LGMOptuna(trial):
    # Suggest random seed
    random_state = trial.suggest_int('random_state', 1, 10000)

    # Create LightGBM datasets with class-balanced weights
    dtrain = lgb.Dataset(x, label=t, weight=compute_sample_weight(class_weight='balanced', y=t).astype('float32'))
    dtest = lgb.Dataset(x_vali, label=t_vali)

    # Define hyperparameters to be optimized
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': random_state,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 10,
    }

    # Train model with current hyperparameters
    model = lgb.train(params, dtrain, valid_sets=[dtest], verbose_eval=False)

    # Predict and evaluate performance
    pred_probs = model.predict(x_vali)
    pred_classes = (pred_probs >= 0.5).astype(int)

    auc = roc_auc_score(t_vali, pred_probs)
    f1 = f1_score(t_vali, pred_classes)
    recall = recall_score(t_vali, pred_classes)
    precision = precision_score(t_vali, pred_classes)

    print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    # Save model and score
    trial_models.append((auc, model))
    return 1 / auc  # Inverse for minimization

# Run Optuna optimization
study = optuna.create_study()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(LGMOptuna, n_trials=10)

# Select the best model by highest AUC
best_auc, best_model = max(trial_models, key=lambda x: x[0])

# Train a general LightGBM model with fixed parameters
train_data = lgb.Dataset(x, label=t)
test_data = lgb.Dataset(x_vali, label=t_vali, reference=train_data)

general_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'early_stopping_rounds': 10
}

general_model = lgb.train(
    general_params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
    verbose_eval=False
)

# Final evaluation
y_pred = general_model.predict(x_vali, num_iteration=general_model.best_iteration)
print(f"Final AUC: {roc_auc_score(t_vali, y_pred):.4f}")
