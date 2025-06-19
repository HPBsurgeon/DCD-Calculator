import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.prepare_data import load_derivation_data

x, t = load_derivation_data("your_input_data.csv")
target = 'BMI'
d = 1
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, random_state=42)
model = lgb.train({'objective': 'binary', 'metric': 'auc', 'verbosity': -1},
                  lgb.Dataset(x_train, label=t_train),
                  valid_sets=[lgb.Dataset(x_test, label=t_test)])

lower, upper = int(x[target].min()), int(x[target].max())
x_vali = pd.DataFrame({col: [x[col].mode()[0] if x[col].dtype.name == 'category' else x[col].median()] for col in x.columns})

X = list(range(lower, upper+1, d))
y = []
for val in X:
    x_vali[target] = val
    y.append(model.predict(x_vali)[0])

plt.figure(figsize=(12, 8))
plt.plot(X, y, marker='o', lw=4)
plt.xlabel(target, fontsize=18)
plt.ylabel('Predicted Probability', fontsize=18)
plt.title(f'Effect of {target} on Prediction', fontsize=20)
plt.grid(True)
plt.show()
