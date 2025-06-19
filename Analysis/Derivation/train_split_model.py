import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils.prepare_data import load_derivation_data

x, t = load_derivation_data("your_input_data.csv")
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, random_state=42)

params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt', 'early_stopping_rounds': 10}
dtrain = lgb.Dataset(x_train, label=t_train)
dtest = lgb.Dataset(x_test, label=t_test)
model = lgb.train(params, dtrain, valid_sets=[dtest])

importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Gain': model.feature_importance(importance_type='gain'),
    'Split': model.feature_importance(importance_type='split')
})

# Plot Gain
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Gain'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()

# Plot Split
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Split'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (Split)')
plt.tight_layout()
plt.show()
