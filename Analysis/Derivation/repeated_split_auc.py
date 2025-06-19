import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils.prepare_data import load_derivation_data

x, t = load_derivation_data("your_input_data.csv")
n_splits = 10
auc_scores = []

for _ in range(n_splits):
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)
    model = lgb.train({'objective': 'binary', 'metric': 'auc', 'verbosity': -1},
                      lgb.Dataset(x_train, label=t_train),
                      valid_sets=[lgb.Dataset(x_test, label=t_test)],
                      num_boost_round=100)
    y_pred = model.predict(x_test)
    auc_scores.append(roc_auc_score(t_test, y_pred))

print(f"Average AUC over {n_splits} splits: {np.mean(auc_scores):.4f}")

import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from utils.prepare_data import load_derivation_data

x, t = load_derivation_data("your_input_data.csv")
n_splits = 5
auc_list = []

for _ in range(n_splits):
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)

    def LGMOptuna(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': trial.suggest_int('random_state', 1, 10000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'early_stopping_rounds': 10,
            'verbose_eval': False
        }
        dtrain = lgb.Dataset(x_train, label=t_train, weight=compute_sample_weight(class_weight='balanced', y=t_train))
        dtest = lgb.Dataset(x_test, label=t_test)
        model = lgb.train(params, dtrain, valid_sets=[dtrain, dtest])
        pred = model.predict(x_test)
        return 1 / roc_auc_score(t_test, pred)

    study = optuna.create_study()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(LGMOptuna, n_trials=10)
    auc_list.append(1 / study.best_value)

print("AUCs from Optuna splits:", auc_list)
print(f"Average AUC: {np.mean(auc_list):.4f}")
