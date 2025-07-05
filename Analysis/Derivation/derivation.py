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

import xgboost as xgb

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

shap_values_list = []
n = 10
for i in range(n):
    x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size=0.2,
                                                            #   random_state=i
                                                              )
    dtrain = lgb.Dataset(x_train, label=t_train)
    dtest = lgb.Dataset(x_test, label=t_test_light)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'early_stopping_rounds': 10
    }

    model_light = lgb.train(params, dtrain, valid_sets=dtest)
    explainer = shap.Explainer(model_light, x_train)
    shap_values = explainer(x_train)
    shap_values_list.append(np.abs(shap_values.values))

mean_shap_values = np.mean(np.array(shap_values_list), axis=0)
feature_importance = np.mean(mean_shap_values, axis=0)

importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance based on SHAP values')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

model = xgb.XGBClassifier(
    objective='binary:logistic',  
    eval_metric='logloss',        
    max_depth=6,                 
    learning_rate=0.05,           
    subsample=0.8,                
    colsample_bytree=0.9,         
    n_estimators=100,             
    random_state=42,              
    early_stopping_rounds=10,
    enable_categorical=True       
)

model.fit(
    x_train, t_train,
    eval_set=[(x_train, t_train), (x_test, t_test)],  )

y_pred = model.predict_proba(x_test)[:, 1]  
auc_score = roc_auc_score(t_test, y_pred)
print(f"ROC AUC: {auc_score}")


df1616['CSTATUS_30'] = np.select(
    [df1616['Survival_time'] <= 30,
     df1616['Survival_time'] > 30],
    [1, 0],
    default=np.nan)

df1616['CSTATUS_45'] = np.select(
    [df1616['Survival_time'] <= 45,
     df1616['Survival_time'] > 45],
    [1, 0],
    default=np.nan)

df1616['CSTATUS_60'] = np.select(
    [df1616['Survival_time'] < 60,
     (df1616['Survival_time'] == 60) & (df1616['Status_120'] == 1),
     df1616['Survival_time'] > 60],
    [1, 1, 0],
    default=np.nan)

df1616['BMI'] = df1616['WGT_KG_DON_CALC'] / (df1616['HGT_CM_DON_CALC'] / 100) ** 2

df1616['BMI_category'] = np.select(
    [(df1616['BMI'] >= 0) & (df1616['BMI'] < 30),
     (df1616['BMI'] >= 30)],
    [0, 1],
    default=np.nan)

df1616['arrest_his'] = np.select(
    [df1616['cardiac arrest'] == 'y',
     df1616['cardiac arrest'].isin(['n', 'u'])],
    [1, 0],
    default=np.nan)

df1616['O2_index2'] = df1616['end_FiO2'] * df1616['mean airway'] / df1616['end_PaO2']

df1616['O2_index_new'] = np.select(
    [df1616['O2_index2'] > 3.0,
     df1616['O2_index2'] <= 3.0],
    [1, 0],
    default=np.nan)

df1616['GCS_category'] = np.select(
    [df1616['GCS'] == 3,
     df1616['GCS'] >= 4],
    [0, 1],
    default=np.nan)

df1616['end_MAP_category'] = np.select(
    [df1616['end_MAP'] < 75,
     df1616['end_MAP'] >= 75],
    [1, 0],
    default=np.nan)

df1616['initial_PF_ratio_category'] = np.select(
    [df1616['initial_PF_ratio'] >= 400,
     (df1616['initial_PF_ratio'] >= 300) & (df1616['initial_PF_ratio'] < 400),
     (df1616['initial_PF_ratio'] >= 200) & (df1616['initial_PF_ratio'] < 300),
     (df1616['initial_PF_ratio'] >= 100) & (df1616['initial_PF_ratio'] < 200),
     df1616['initial_PF_ratio'] < 100],
    [0, 1, 2, 3, 4],
    default=np.nan)

df1616['end_PF_ratio_category'] = np.select(
    [df1616['end_PF_ratio'] >= 400,
     (df1616['end_PF_ratio'] >= 300) & (df1616['end_PF_ratio'] < 400),
     (df1616['end_PF_ratio'] >= 200) & (df1616['end_PF_ratio'] < 300),
     (df1616['end_PF_ratio'] >= 100) & (df1616['end_PF_ratio'] < 200),
     df1616['end_PF_ratio'] < 100],
    [0, 1, 2, 3, 4],
    default=np.nan)

df1616['end_Na_category'] = np.select(
    [df1616['end_Na'] < 135,
     (df1616['end_Na'] >= 135) & (df1616['end_Na'] < 146),
     (df1616['end_Na'] >= 146) & (df1616['end_Na'] < 156),
     df1616['end_Na'] >= 156],
    [1, 0, 2, 3],
    default=np.nan)

df1616['end_Plt_category'] = np.select(
    [df1616['end_Plt'] < 100,
     (df1616['end_Plt'] >= 100) & (df1616['end_Plt'] < 150),
     df1616['end_Plt'] >= 150],
    [2, 1, 0],
    default=np.nan)

df1616['end_HCO3_category'] = np.select(
    [df1616['end_HCO3'] < 18,
     (df1616['end_HCO3'] >= 18) & (df1616['end_HCO3'] < 22),
     df1616['end_HCO3'] >= 22],
    [2, 1, 0],
    default=np.nan)

df1616['end_ph_category'] = np.select(
    [df1616['end_ph'] < 7.35,
     (df1616['end_ph'] >= 7.35) & (df1616['end_ph'] <= 7.45),
     df1616['end_ph'] > 7.45],
    [1, 0, 2],
    default=np.nan)

dtype_map = {
    'GCS': 'float64',
    'pupil': 'category',
    'gag': 'category',
    'corneal': 'category',
    'cough': 'category',
    'motor': 'category',
    'OBV': 'category',
    'end_MAP_category': 'Int64',
    'end_Na_category': 'Int64',
    'end_Plt_category': 'Int64',
    'initial_PF_ratio_category': 'Int64',
    'end_PF_ratio_category': 'Int64',
    'end_ph_category': 'Int64',
    'arrest_his': 'category',
    'Mechanism_of_injury3': 'category',
    'BMI_category': 'Int64'
}

for col, dtype in dtype_map.items():
    if col in df1616.columns:
        df1616[col] = df1616[col].astype(dtype)

df1616_3 = df1616[['Validation','UNET_ID','CSTATUS_60','CSTATUS_45','CSTATUS_30',
                    'GCS','pupil','gag','corneal','cough','motor','OBV',
                    'end_MAP_category','end_Na_category','end_Plt_category',
                    'initial_PF_ratio_category','end_PF_ratio_category',
                    'end_ph_category','arrest_his','Mechanism_of_injury3',
                    'BMI_category']]

df1616_3 = df1616_3.dropna(subset=df1616_3.columns.difference([
            'GCS','pupil','gag','corneal','cough','motor','OBV']))
df1616_3 = df1616_3.dropna(subset=[
            'pupil','gag','corneal','cough','motor','OBV'], thresh=5)

x = df1616_3.drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30','UNET_ID','Validation'], axis=1)
t = np.array(df1616_3['CSTATUS_30'].tolist())
x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size = 0.2)

##10times
def LGMOptuna(trial):
    random_state = trial.suggest_int('random_state', 1, 10000)#
    dtrain = lgb.Dataset(x_train, label=t_train, weight=compute_sample_weight(class_weight='balanced', y=t_train).astype('float32'))
    dtest = lgb.Dataset(x_test, label=t_test_light, weight=np.ones(len(x_test)).astype('float32'))
    params = {
      'objective':'binary',
      'metric':'auc',
      'verbosity':-1,
      'boosting_type':'gbdt',
      'random_state':trial.suggest_int('random_state', 1, 10000), #元々1500,
      'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
      'subsample': trial.suggest_float('subsample', 0.6, 1.0),
      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
      'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
      'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
      'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
      'early_stopping_rounds':10,
      'verbose_eval': False,
      }
    model_light = lgb.train(params,
                             dtrain,
                             valid_sets = [dtrain, dtest],
                            #  early_stopping_rounds=10,
                            #  verbose_eval = False
                             )
    predicted_light = model_light.predict(x_test)
    auc_l = roc_auc_score(t_test_light, predicted_light)
    import pickle
    # directory = '###
    # with open(directory + '/model_light1.pickle', mode = 'wb') as f:
    #     pickle.dump(model_light, f)    
    return 1/auc_l

study = optuna.create_study()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(LGMOptuna, 10)
auc_light = 1/study.best_value
1/study.best_value



