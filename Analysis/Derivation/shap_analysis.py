import lightgbm as lgb
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from utils.prepare_data import load_derivation_data

x, t = load_derivation_data("your_input_data.csv")
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, random_state=42)
model = lgb.train({'objective': 'binary', 'metric': 'auc', 'verbosity': -1},
                  lgb.Dataset(x_train, label=t_train),
                  valid_sets=[lgb.Dataset(x_test, label=t_test)])

explainer = shap.Explainer(model, x_train)
shap_values = explainer(x_train)
shap.summary_plot(shap_values.values, x_train, feature_names=x.columns, plot_type="dot")
