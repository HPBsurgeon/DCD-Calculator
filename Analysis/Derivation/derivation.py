import datetime
import pandas as pd
import numpy as np
import tqdm

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import joblib
from joblib import load
import pickle
import optuna
import optuna.integration.lightgbm as lgb

import itertools
import shap


df1616 = pd.read_excel('derivation_2_1616.xlsx')

df1616_2 = df1616[['AGE_DON','GCS','pupil','gag','corneal','cough','motor','OBV',
            'ini_MAP','ini_HR','ini_urine','end_MAP','end_HR','end_urine',
            'Hct','Plt','end_Hct','end_Plt','Na','K','end_Na','end_K','ini_ph',
            'ini_PaCO2','HCO3','ini_SaO2','ini_RR','ini_PEEP','end_ph','end_PaCO2',
            'end_HCO3','end_SaO2','end_RR','end_PEEP','initial_PF_ratio','end_PF_ratio','arrest_his',
            'CSTATUS_60','CSTATUS_45','CSTATUS_30',
            'Mechanism_of_injury3','Transfusion2','sedation2','BMI']]

df1616_2 = df1616_2.dropna(subset=df1616_2.columns.difference([
            'GCS','pupil','gag','corneal','cough','motor','OBV','sedation2',]))

x = df1616_2.drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30'], axis=1)
t = np.array(df1616_2['CSTATUS_30'].tolist())
x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size = 0.2)


optuna.logging.set_verbosity(optuna.logging.WARNING)
target = 'K'
plt.figure(figsize=(9, 6))
median_list = [e for e in x.columns.tolist() if e != target]
x_vali = pd.DataFrame(index=[0], columns=x.columns)
for c in median_list:
    if x[c].dtype == 'object' or x[c].dtype.name == 'category':
        x_vali[c] = pd.Categorical(x[c].mode()[0])
    else:
        x_vali[c] = x[c].median()

if target in ['K','HgB','Hct','HCO3','ini_SaO2','end_K','end_HgB','end_Hct','end_HCO3','end_SaO2']:
    d = 0.1 
elif target in ['ini_ph','end_ph']:
    d = 0.01
else:
    d = 1
mean = x[target].mean()
sem = x[target].sem()
lower = x[target].min()
upper = x[target].max()
if target=='end_Plt':
    upper=400
if target=='end_MAP':
    upper=130
if target=='end_ph':
    lower=7.0
if target=='BMI':
    upper=40
if target=='end_PF_ratio':
    upper=600
if target=='initial_PF_ratio':
    upper=600

target_range_list = [i for i in range(int((upper-lower)/d)+2)]
df_proba = pd.DataFrame(np.array([0] * len(target_range_list)), index=target_range_list)
n = 10
for i in range(n):
    dtrain = lgb.Dataset(x_train, label=t_train)
    dtest = lgb.Dataset(x_test, label=t_test_light)
    params = {
      'objective':'binary',
      'metric':'auc',
      'verbosity':-1,
      'boosting_type':'gbdt',
      'early_stopping_rounds':10
      }
    model_light = lgb.train(params,dtrain,valid_sets = dtest,)
    prob_list = []
    for a in range(int((upper-lower)/d)+2):
        x_vali[target] = a*d + lower
        proba = model_light.predict(x_vali)[0]
        prob_list.append(proba)

    df_proba[i] = prob_list

proba_mean = df_proba.mean(axis='columns').tolist()

X = [i*d + lower for i in range(int((upper-lower)/d)+2)]
y = proba_mean

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(X, y, marker='o',lw=6)
ax.set_xlabel('K', fontsize=38) 
ax.set_ylabel("Probability", fontsize=38)
np.mean(np.array(y))
ax.set_ylim(0.16, 0.30) 
y_ticks = np.arange(0.16, 0.30, 0.2)
ax.tick_params(axis='x', labelsize=31)
ax.tick_params(axis='y', labelsize=31)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


shap_values_list = []

n = 10
for i in range(n):
    dtrain = lgb.Dataset(x_train, label=t_train)
    dtest = lgb.Dataset(x_test, label=t_test_light)
    params = {'objective': 'binary','metric': 'auc','verbosity': -1,'boosting_type': 'gbdt',
        'early_stopping_rounds': 10}

    model_light = lgb.train(params, dtrain, valid_sets=dtest)
    explainer = shap.Explainer(model_light, x_train)
    shap_values = explainer(x_train)
    shap_values_list.append(shap_values.values)

mean_shap_values = np.mean(np.array(shap_values_list), axis=0)

shap.summary_plot(mean_shap_values, x_train, feature_names=x.columns, plot_type="dot", max_display=20)



