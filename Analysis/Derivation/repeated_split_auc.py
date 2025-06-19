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
