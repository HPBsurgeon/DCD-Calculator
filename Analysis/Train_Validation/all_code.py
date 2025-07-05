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
df398 = pd.read_excel('retro_validation_2_398.xlsx')
df207 = pd.read_excel('pros_validation_2_207.xlsx')

unet = pd.concat([df1616, df398], axis=0, ignore_index=True)

unet['CSTATUS_30'] = np.select(
    [unet['Survival_time'] <= 30,
    unet['Survival_time'] > 30,],
    [1, 0],
    default=np.nan)

unet['CSTATUS_45'] = np.select(
    [unet['Survival_time'] <= 45,
    unet['Survival_time'] > 45,],
    [1, 0],
    default=np.nan)

unet['CSTATUS_60'] = np.select(
    [unet['Survival_time'] < 60,
    (unet['Survival_time'] == 60)&(unet['Status_120'] == 1),
    unet['Survival_time'] > 60,],
    [1, 1, 0],
    default=np.nan)


unet['BMI'] = unet['WGT_KG_DON_CALC'] / (unet['HGT_CM_DON_CALC'] / 100) ** 2

unet['BMI_category'] = np.select(
    [(unet['BMI'] >= 0) & (unet['BMI'] < 30),
    (unet['BMI'] >= 30)],
    [0, 1],
    default=np.nan)

unet['arrest_his'] = np.select(
    [unet['cardiac arrest'] == 'y',
    unet['cardiac arrest'].isin(['n', 'u'])],
    [1, 0],
    default=np.nan)

unet['O2_index2'] = unet['end_FiO2'] * unet['mean airway'] / unet['end_PaO2']
unet['O2_index_new'] = np.select(
    [unet['O2_index2'] > 3.0,
    unet['O2_index2'] <= 3.0],
    [1, 0],
    default=np.nan)


unet['GCS_category'] = np.select([unet['GCS'] == 3,
                                  unet['GCS'] >= 4],
                                 [0, 1], 
                                 default=np.nan)

unet['end_MAP_category'] = np.select([unet['end_MAP'] < 75,
                                  unet['end_MAP'] >= 75],
                                 [1, 0], 
                                 default=np.nan)

unet['initial_PF_ratio_category'] = np.select(
    [unet['initial_PF_ratio'] >= 400,
    (unet['initial_PF_ratio'] >= 300) & (unet['initial_PF_ratio'] < 400),
    (unet['initial_PF_ratio'] >= 200) & (unet['initial_PF_ratio'] < 300),
    (unet['initial_PF_ratio'] >= 100) & (unet['initial_PF_ratio'] < 200),
    unet['initial_PF_ratio'] < 100],
    [0, 1, 2, 3, 4],
    default=np.nan)

unet['end_PF_ratio_category'] = np.select(
    [unet['end_PF_ratio'] >= 400,
    (unet['end_PF_ratio'] >= 300) & (unet['end_PF_ratio'] < 400),
    (unet['end_PF_ratio'] >= 200) & (unet['end_PF_ratio'] < 300),
    (unet['end_PF_ratio'] >= 100) & (unet['end_PF_ratio'] < 200),
    unet['end_PF_ratio'] < 100],
    [0, 1, 2, 3, 4],
    default=np.nan)

unet['end_Na_category'] = np.select(
    [unet['end_Na'] < 135,
    (unet['end_Na'] >= 135) & (unet['end_Na'] < 146),
    (unet['end_Na'] >= 146) & (unet['end_Na'] < 156),
    unet['end_Na'] >= 156],
    [1, 0, 2, 3],
    default=np.nan)

unet['end_Plt_category'] = np.select(
    [unet['end_Plt'] < 100,
    (unet['end_Plt'] >= 100) & (unet['end_Plt'] < 150),
    unet['end_Plt'] >= 150],
    [2, 1, 0],
    default=np.nan)

unet['end_HCO3_category'] = np.select(
    [unet['end_HCO3'] < 18,
    (unet['end_HCO3'] >= 18) & (unet['end_HCO3'] < 22),
    unet['end_HCO3'] >= 22],
    [2, 1, 0],
    default=np.nan)

unet['end_ph_category'] = np.select(
    [unet['end_ph'] < 7.35,
    (unet['end_ph'] >= 7.35) & (unet['end_ph'] <= 7.45),
    unet['end_ph'] > 7.45],
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
    if col in unet.columns:
        unet[col] = unet[col].astype(dtype)

unet
ML_df = unet[[
            'Validation',
            'UNET_ID',
            'CSTATUS_60','CSTATUS_45','CSTATUS_30',
            'GCS','pupil','gag','corneal','cough','motor','OBV',
            'end_MAP_category','end_Na_category','end_Plt_category',
            'initial_PF_ratio_category','end_PF_ratio_category',
            'end_ph_category','arrest_his','Mechanism_of_injury3',
            'BMI_category']]

df1 = ML_df[ML_df['Validation']=='d']
df1 = df1.dropna(subset=df1.columns.difference([
            'GCS','pupil','gag','corneal','cough','motor','OBV']))
df1 = df1.dropna(subset=[
            'pupil','gag','corneal','cough','motor','OBV'], thresh=3)

x = df1.drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30','UNET_ID','Validation'], axis=1)
t = np.array(df1['CSTATUS_30'].tolist())


df2 = ML_df[ML_df['Validation']=='v']
df2 = df2.dropna(subset=df2.columns.difference([
            'GCS','pupil','gag','corneal','cough','motor','OBV']))
df2 = df2.dropna(subset=[
            'pupil','gag','corneal','cough','motor','OBV'], thresh=3)

x_vali = df2.drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30', 'UNET_ID','Validation'], axis=1)
t_vali = np.array(df2['CSTATUS_30'].tolist())

# Store the model and score for each trial
trial_models = []
def LGMOptuna(trial):
    random_state = trial.suggest_int('random_state', 1, 10000)
    
    dtrain = lgb.Dataset(x, label=t, weight=compute_sample_weight(class_weight='balanced', y=t).astype('float32'))
    dtest = lgb.Dataset(x_vali, label=t_vali, weight=np.ones(len(x_vali)).astype('float32'))
    
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

    model_light = lgb.train(params, dtrain, valid_sets=[dtrain, dtest])
    predicted_probs = model_light.predict(x_vali)
    predicted_classes = (predicted_probs >= 0.5).astype(int)
    
    auc_l = roc_auc_score(t_vali, predicted_probs)
    f1 = f1_score(t_vali, predicted_classes)
    recall = recall_score(t_vali, predicted_classes)
    precision = precision_score(t_vali, predicted_classes)
    print(f"AUC: {auc_l:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    trial_models.append((auc_l, model_light)) 

    return 1 / auc_l 

study = optuna.create_study()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(LGMOptuna, n_trials=10)

best_auc, best_model = max(trial_models, key=lambda x: x[0])
# model_path = '###'
# with open(model_path, 'wb') as f:
#     pickle.dump(best_model, f)
# best_auc

train_data = lgb.Dataset(x, label=t)
test_data = lgb.Dataset(x_vali, label=t_vali, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'early_stopping_rounds':10
}

best_model = lgb.train(params,
                  train_data,
                  valid_sets=[train_data, test_data],
                  num_boost_round=100,)

y_pred = best_model.predict(x_vali, num_iteration=best_model.best_iteration)
roc_auc_score(t_vali, y_pred)

x_vali2 = ML_df[ML_df['Validation']=='v'].drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30', 'UNET_ID','Validation'], axis=1)
df_retro = unet[unet['Validation']=='v']

df_retro['LGBM_score'] = best_model.predict(x_vali2)*100

df_retro = df_retro.dropna(subset=[
            'pupil','gag','corneal','cough','motor','OBV'], thresh=3)

mask = df_retro['DCD_Nscore'].notna()  
nscore_preds = df_retro.loc[mask, 'DCD_Nscore'].tolist()
t = df_retro.loc[mask, 'CSTATUS_30'].tolist()
print("AUC (no missing only):", roc_auc_score(t, nscore_preds))

nscore_preds = np.array(df_retro['DCD_Nscore'].fillna(0).tolist())#
t = df_retro['CSTATUS_30'].tolist()
print("AUC (filled na):", roc_auc_score(t, nscore_preds))

mask = df_retro['Corolado_Proba'].notna() 
corolado_preds = df_retro.loc[mask, 'Corolado_Proba'].tolist()
t = df_retro.loc[mask, 'CSTATUS_30'].tolist()
print("AUC (no missing only):", roc_auc_score(t, corolado_preds))

corolado_preds = np.array(df_retro['Corolado_Proba'].fillna(0).tolist())#
t = df_retro['CSTATUS_30'].tolist()
print("AUC (filled na):", roc_auc_score(t, corolado_preds))

scaled_preds = np.array(df_retro['LGBM_score'].tolist())
t = df_retro['CSTATUS_30'].tolist()
print("AUC:", roc_auc_score(t, scaled_preds))

fpr, tpr, thresholds = roc_curve(t, scaled_preds)

fpr, tpr, thresholds = roc_curve(df_retro['CSTATUS_30'].tolist(), df_retro['LGBM_score'].tolist())
print("AUC:", roc_auc_score(df_retro['CSTATUS_30'].tolist(), df_retro['LGBM_score'].tolist()))

accuracies = [(threshold, accuracy_score(t, scaled_preds >= threshold)) for threshold in thresholds]
best_cutoff, best_acc = max(accuracies, key=lambda x: x[1])
print(f"Best cutoff (ACC): {best_cutoff:.2f}")
print(f"Maximum accuracy: {best_acc:.4f}")

final_preds = (scaled_preds >= best_cutoff).astype(int) 
tn, fp, fn, tp = confusion_matrix(t, final_preds).ravel()
fpr_total = fp / len(t)
fnr_total = fn / len(t)
acc_total = 1 - (fpr_total+fnr_total)
print(f"FPR (over total): {fpr_total:.4f}")
print(f"FNR (over total): {fnr_total:.4f}")

def compute_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    lower = max(0, p - 1.96 * se)
    upper = min(1, p + 1.96 * se)
    return lower, upper

FPR_CI = compute_ci(fpr_total, len(t))
FNR_CI = compute_ci(fnr_total, len(t))
ACC_CI = compute_ci(acc_total, len(t))

cutoffs = np.linspace(0, 100, 100)
fpr_list, fnr_list, acc_list = [], [], []

for cutoff in cutoffs:
    binary_preds = (scaled_preds >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(t, binary_preds).ravel()
    n = len(t)
    fpr_val = fp / n
    fnr_val = fn / n
    acc_val = (tp + tn) / n
    fpr_list.append(fpr_val)
    fnr_list.append(fnr_val)
    acc_list.append(acc_val)

plt.figure(figsize=(8, 5))
plt.plot(cutoffs, fpr_list, label='Futile Procurement Rate', color='#1f77b4', linewidth=3.5)
plt.plot(cutoffs, fnr_list, label='Missed Opportunity Rate', color='#d62728', linewidth=3.5)
plt.plot(cutoffs, acc_list, label='Accuracy', color='#2ca02c', linewidth=3.5)

plt.axvline(best_cutoff, color='black', linestyle='--', linewidth=2.5)

plt.xlabel('Index Cutoff Threshold', fontsize=18)
plt.ylabel('Rate', fontsize=18)
plt.title('Cutoff-dependent Metrics', fontsize=17)
plt.legend(loc='upper center', bbox_to_anchor=(0.84, 0.33), fontsize=10)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# plt.savefig("Fig5c.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.show()

metrics_names = ['Futile\nProcurement\nRate', 'Missed\nOpportunity\nRate', 'Accuracy']
values = [fpr_total, fnr_total, acc_total]
ci_lower = [FPR_CI[0], FNR_CI[0], ACC_CI[0]]
ci_upper = [FPR_CI[1], FNR_CI[1], ACC_CI[1]]

colors = ['#1f77b4', '#d62728', '#2ca02c']

fig, ax = plt.subplots(figsize=(5, 6))

bars = ax.bar(metrics_names, values, color=colors, linewidth=2, alpha=0.9, width=0.6)

for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
    ax.plot([i - 0.1, i + 0.1], [lower, lower], color='black', linewidth=2)  
    ax.plot([i - 0.1, i + 0.1], [upper, upper], color='black', linewidth=2)  
    ax.vlines(i, lower, upper, color='black', linewidth=2)  

# for i, value in enumerate(values):
#     ax.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=12)

for i, value in enumerate(values):
    ax.text(i+0.3, value + 0.012, 
            f'{value:.3f}', 
            ha='center', va='bottom', fontsize=12, 
            rotation=0, 
            transform=ax.transData)

ax.tick_params(axis='both', which='major', labelsize=15, width=2)
ax.set_title('LGBM Model', fontsize=25)
ax.set_ylabel('Rate', fontsize=24)
ax.set_ylim(0, 1)  

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=13.5)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("Fig4a.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.show()

df = df2.copy()
df['ML_predi'] = general_model.predict(x_vali)*100
df['ML_predi2'] = (df['ML_predi'] >= 50).astype(int) #best_cutoff
df['Match_Discrepancy'] = np.where(df['ML_predi2'] == df['CSTATUS_30'], 0, 1)

bins = sorted(set([0, 20] + list(np.arange(30, df['ML_predi'].max() + 10, 10))))
df['ML_predi_bins'] = pd.cut(df['ML_predi'], bins=bins, right=False)

bin_counts = df.groupby('ML_predi_bins')['Match_Discrepancy'].sum()
total_counts = df['ML_predi_bins'].value_counts().sort_index()
percent_accuracy = (1 - bin_counts / total_counts) * 100

percent_labels = [f"{int(bin.left)}-{int(bin.right)}" for bin in total_counts.index]

# Plotting the chart
fig, ax1 = plt.subplots(figsize=(9, 7))

# Bar chart (left Y-axis)
colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(total_counts))) 
ax1.bar(percent_labels, total_counts, color=colors, width=0.6, label='Count')
ax1.set_xlabel('YDF Model Index', fontsize=20)
ax1.set_ylabel('Count', fontsize=20)
ax1.tick_params(axis='x', rotation=45, labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0, max(total_counts) + 10)  # Add extra space above bars for the legend

# Line plot (right Y-axis)
ax2 = ax1.twinx()
ax2.plot(percent_labels, percent_accuracy, color='#1a5d8f', marker='o', label='Accuracy (%)')
ax2.set_ylabel('Accuracy (%)', fontsize=17)
ax2.tick_params(axis='y', labelsize=15)
ax2.set_ylim(0, 100)

# Add accuracy percentage above each point
for i, acc in enumerate(percent_accuracy):
    ax2.text(i, acc + 3, f"{acc:.1f}%", ha='center', fontsize=12, color='#1a5d8f')

# Combine legends and place them in the empty space
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=13, ncol=2)

# Add title
plt.title('Retrospective Validation', fontsize=24)

plt.tight_layout()
plt.show()

columns = [
           'K_S', 
           'K_T', 
           'M_F', 
           'Y_B',
           'J_K',
           'D_S',
           'Y_S',
           'Jenny',
           'Kliment',
           'Melcher',
           'M_K'
           ]

unet['Human'] = unet[columns].apply(lambda row: row.value_counts().get(1, 0) / row.count(), axis=1)
unet['Human']
unet['Human2'] = (unet['Human'] > 0.5).astype(int)

import shap
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'

all_shap_values = []
all_X_values = []
cl_list = ['K_S', 'K_T', 'M_F', 'Y_B', 'J_K', 'D_S', 'Y_S', 'Jenny', 'Kliment', 'Melcher', 'M_K']

feature_name_dict = {
    'end_PF_ratio_category': 'End PF Ratio Category',
    'initial_PF_ratio_category': 'Initial PF Ratio Category',
    'BMI_category': 'BMI Category',
    'end_ph_category': 'End pH Category',
    'end_Na_category': 'End Na Category',
    'end_MAP_category': 'End MAP Category',
    'end_Plt_category': 'End Platelet Category',
    'Mechanism_of_injury3': 'Mechanism of Injury',
    'arrest_his': 'Arrest History',
    'GCS': 'Glasgow Coma Scale',
    'OBV': 'OBV',
    'pupil': 'Pupil Reaction',
    'gag': 'Gag Reflex',
    'corneal': 'Corneal Reflex',
    'cough': 'Cough Reflex',
    'motor': 'Motor Response'
}

for c in cl_list:
    df = unet[unet['Validation'] == 'v'].dropna(subset=['recorder'])[
        ['GCS', 'pupil', 'gag', 'corneal', 'cough', 'motor', 'OBV',
         'end_MAP_category', 'end_Plt_category', 'end_Na_category',
         'end_ph_category', 'initial_PF_ratio_category', 'end_PF_ratio_category',
         'arrest_his', 'CSTATUS_30', 'Mechanism_of_injury3', 'BMI_category', c]
    ].astype('float')

    df[c] = df[c].apply(lambda x: np.random.choice([0, 1]) if pd.isna(x) else x)

    X = df.drop([c, 'CSTATUS_30'], axis=1)
    y = df[c].tolist()

    model = LGBMClassifier()
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    mask = ((df[c] == 0) & (df['CSTATUS_30'] == 1)).values
    X_filtered = X[mask]
    shap_values_filtered = shap_values.values[mask]

    all_shap_values.append(shap_values_filtered)
    all_X_values.append(X_filtered)

combined_shap_values = np.vstack(all_shap_values)
combined_X_values = pd.concat(all_X_values, axis=0).reset_index(drop=True)
combined_X_values_renamed = combined_X_values.rename(columns=feature_name_dict)

false_negative_count = 45  
if len(combined_X_values) > false_negative_count:
    sampled_indices = np.random.choice(len(combined_X_values), size=false_negative_count, replace=False)
    combined_X_sampled = combined_X_values.iloc[sampled_indices]
    combined_shap_sampled = combined_shap_values[sampled_indices]
else:
    combined_X_sampled = combined_X_values
    combined_shap_sampled = combined_shap_values

# # --- SHAP Summary Plot ---
# shap.summary_plot(combined_shap_sampled, combined_X_sampled.rename(columns=feature_name_dict), plot_type='dot', max_display=16)

# fig = plt.gcf()
# fig.suptitle("Human Simulation Model SHAP Summary Plot for False Negative Cases", fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.95]) 
# plt.show()

# --- SHAP Summary Plot ---
plt.figure(figsize=(10, 8)) 
shap.summary_plot(combined_shap_sampled, combined_X_sampled.rename(columns=feature_name_dict), plot_type='dot', max_display=16, show=False)

plt.title("Human Simulation Model SHAP Summary Plot for False Negative Cases", fontsize=16, pad=20)

plt.tight_layout()
# plt.savefig("Fig4f.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.show()


# # Function to randomly fill NaN values in specified columns with 0 or 1
# def random_fill_na(df, columns):
#     df_copy = df.copy()
#     for col in columns:
#         df_copy[col] = df_copy[col].apply(lambda x: np.random.choice([0, 1]) if pd.isna(x) else x)
#     return df_copy

# # Define columns to randomly fill
# columns_to_fill = ['K_S', 'K_T', 'M_F', 'Y_B', 'J_K', 'D_S', 'Y_S', 'M_K', 'Jenny', 'Kliment', 'Melcher']

# # Simulations
# results = []
# for _ in range(100):
#     simulated_df = random_fill_na(df, columns_to_fill)
#     simulated_df['agree'] = simulated_df[columns_to_fill].sum(axis=1)
#     results.append(simulated_df['agree'].value_counts().sort_index())

# # Combine all simulations into a single DataFrame
# simulation_df = pd.DataFrame(results).fillna(0)
simulation_df = pd.read_excel('/Users/yanagawarintaro/Desktop/research/re_DCD_prediction/Figure3_preserve.xlsx')

# Calculate the mean and 95% CI for each "agree" value
mean_distribution = simulation_df.mean(axis=0)
lower_bound = simulation_df.quantile(0.025, axis=0)
upper_bound = simulation_df.quantile(0.975, axis=0)

# Aggregate custom distribution
custom_distribution = {
    '0vs11': mean_distribution.get(0, 0) + mean_distribution.get(11, 0),
    '1vs10': mean_distribution.get(1, 0) + mean_distribution.get(10, 0),
    '2vs9': mean_distribution.get(2, 0) + mean_distribution.get(9, 0),
    '3vs8': mean_distribution.get(3, 0) + mean_distribution.get(8, 0),
    '4vs7': mean_distribution.get(4, 0) + mean_distribution.get(7, 0),
    '5vs6': mean_distribution.get(5, 0) + mean_distribution.get(6, 0),
}

custom_ci = {
    pair: (
        lower_bound.get(int(pair.split('vs')[0]), 0) + lower_bound.get(int(pair.split('vs')[1]), 0),
        upper_bound.get(int(pair.split('vs')[0]), 0) + upper_bound.get(int(pair.split('vs')[1]), 0)
    )
    for pair in custom_distribution.keys()
}

# Convert to DataFrame for plotting
custom_distribution_df = pd.DataFrame(list(custom_distribution.items()), columns=['Pair', 'Frequency'])
custom_distribution_df['Lower_CI'] = [custom_ci[pair][0] for pair in custom_distribution.keys()]
custom_distribution_df['Upper_CI'] = [custom_ci[pair][1] for pair in custom_distribution.keys()]

# Custom colormap: Gradual darkening of specified colors
colors = [plt.cm.Reds(0.15), plt.cm.Reds(0.45),  plt.cm.Reds(0.75),  plt.cm.Reds(0.95)]
labels = ['Strong Agree', 'General Agree', 'Weak Agree', 'Poor Agree']

# Plotting with adjustments for thicker border, thinner bars, and bold ticks
fig, ax = plt.subplots(figsize=(8, 8))

# Create bars with specified shades
bars = ax.bar(custom_distribution_df['Pair'], custom_distribution_df['Frequency'], color=[plt.cm.Reds(0.15),plt.cm.Reds(0.15),plt.cm.Reds(0.45),plt.cm.Reds(0.75),plt.cm.Reds(0.95),plt.cm.Reds(0.95)], alpha=0.9, width=0.6)

# # Add 95% CI as lines
# for index, row in custom_distribution_df.iterrows():
#     ax.plot([index - 0.2, index + 0.2], [row['Lower_CI'], row['Lower_CI']], color='black', linewidth=2)
#     ax.plot([index - 0.2, index + 0.2], [row['Upper_CI'], row['Upper_CI']], color='black', linewidth=2)
#     ax.vlines(index, row['Lower_CI'], row['Upper_CI'], color='black', linewidth=2)

# Add value labels
for index, row in custom_distribution_df.iterrows():
    ax.text(index, row['Frequency'] + (row['Upper_CI'] - row['Frequency']) * 0.1,
            f"{row['Frequency']:.0f}", ha='right', fontsize=13)

# Customize plot
ax.set_title('Agree Distribution', fontsize=25)
ax.set_xlabel('Difference of Opinion', fontsize=24)
ax.set_ylabel('Frequency', fontsize=24)

# Add legend
legend_patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=labels[i]) for i in range(len(colors))]
ax.legend(handles=legend_patches, fontsize=14, title='Agreement Levels', title_fontsize=16)

# Customize ticks to be bold
plt.xticks(fontsize=18)
plt.yticks(fontsize=20)

# Remove right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Make left and bottom spines thicker
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(2)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Fig3a.pdf", format='pdf', dpi=400, bbox_inches='tight')

plt.show()

x_vali2 = ML_df[ML_df['Validation']=='v'].drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30', 'UNET_ID','Validation'], axis=1)
df_retro = unet[unet['Validation']=='v']

df_retro['LGBM_score'] = general_model.predict(x_vali2)*100

df_retro = df_retro.dropna(subset=[
            'pupil','gag','corneal','cough','motor','OBV'], thresh=2)

df_retro

def calculate_metrics_with_ci(df, pred_column, outcome_column):
    df_non_missing = df[df[pred_column].notna() & df[outcome_column].notna()]
    n = len(df_non_missing) 
    
    true_positive = ((df_non_missing[pred_column] == 1) & (df_non_missing[outcome_column] == 1)).sum()
    true_negative = ((df_non_missing[pred_column] == 0) & (df_non_missing[outcome_column] == 0)).sum()
    false_positive = ((df_non_missing[pred_column] == 1) & (df_non_missing[outcome_column] == 0)).sum()
    false_negative = ((df_non_missing[pred_column] == 0) & (df_non_missing[outcome_column] == 1)).sum()
    
    FPR = false_positive / n if n > 0 else 0
    FNR = false_negative / n if n > 0 else 0
    ACC = (true_positive + true_negative) / n if n > 0 else None

    def compute_ci(p, n):
        se = np.sqrt(p * (1 - p) / n)
        lower = max(0, p - 1.96 * se)
        upper = min(1, p + 1.96 * se)
        return lower, upper

    return (FPR, FNR, ACC), (compute_ci(FPR, n), compute_ci(FNR, n), compute_ci(ACC, n))

columns = ['K_S','K_T','M_F','Y_B','J_K','D_S','Y_S','M_K','Jenny','Kliment','Melcher',]
outcome_column = 'CSTATUS_30'
FPRs, FNRs, ACCs = [], [], []
FPR_CI, FNR_CI, ACC_CI = [], [], []

for col in columns:
    metrics, ci = calculate_metrics_with_ci(df_retro, col, outcome_column)
    FPRs.append(metrics[0])
    FNRs.append(metrics[1])
    ACCs.append(metrics[2])
    FPR_CI.append(ci[0])
    FNR_CI.append(ci[1])
    ACC_CI.append(ci[2])

avg_FPR, avg_FNR, avg_ACC = np.mean(FPRs), np.mean(FNRs), np.mean(ACCs)
avg_FPR_CI = (np.mean([ci[0] for ci in FPR_CI]), np.mean([ci[1] for ci in FPR_CI]))
avg_FNR_CI = (np.mean([ci[0] for ci in FNR_CI]), np.mean([ci[1] for ci in FNR_CI]))
avg_ACC_CI = (np.mean([ci[0] for ci in ACC_CI]), np.mean([ci[1] for ci in ACC_CI]))

metrics_names = ['Futile\nProcurement\nRate', 'Missed\nOpportunity\nRate', 'Accuracy']
values = [avg_FPR, avg_FNR, avg_ACC]
ci_lower = [avg_FPR_CI[0], avg_FNR_CI[0], avg_ACC_CI[0]]
ci_upper = [avg_FPR_CI[1], avg_FNR_CI[1], avg_ACC_CI[1]]
colors = ['#1f77b4', '#d62728', '#2ca02c']  

fig, ax = plt.subplots(figsize=(5, 6))

bars = ax.bar(metrics_names, values, color=colors, linewidth=2, alpha=0.9, width=0.6)

for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
    ax.vlines(i, lower, upper, color='black', linewidth=2) 
    ax.plot([i - 0.1, i + 0.1], [lower, lower], color='black', linewidth=2)  
    ax.plot([i - 0.1, i + 0.1], [upper, upper], color='black', linewidth=2)  

for i, value in enumerate(values):
    ax.text(i+0.3, value + 0.012,  
            f'{value:.3f}', 
            ha='center', va='bottom', fontsize=12, 
            rotation=0, 
            transform=ax.transData)

ax.tick_params(axis='both', which='major', labelsize=15, width=2)
ax.set_title('Humans Average', fontsize=22)#Worst 3 
ax.set_ylabel('Rate', fontsize=20)
ax.set_ylim(0, 1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=13.5)
plt.yticks(fontsize=18)

plt.tight_layout()
plt.savefig("Fig4b.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.show()


df = df398.copy()
probabilities = np.array(df['ML_predi'].tolist())
model_predictions = (probabilities >= 0.52).astype(int)
probabilities = np.array(df['Human'].tolist())
human_predictions = (probabilities >= 0.5).astype(int)
KS_predictions = df['K_S2'].tolist()
KT_predictions = df['K_T2'].tolist()
MF_predictions = df['M_F2'].tolist()
YB_predictions = df['Y_B2'].tolist()
JK_predictions = df['J_K2'].tolist()
DS_predictions = df['D_S2'].tolist()
YS_predictions = df['Y_S2'].tolist()
MK_predictions = df['M_K2'].tolist()
Jenny_predictions = df['Jenny2'].tolist()
Kliment_predictions = df['Kliment2'].tolist()
Melcher_predictions = df['Melcher2'].tolist()
DCDNScore_predictions = df['DCD_Nscore2'].tolist()
Corolado_predictions = df['Corolado'].tolist()
actual_outcomes = df['CSTATUS_30'].tolist()

# データフレームの作成
data = pd.DataFrame({
    'Probability': probabilities,
    'KS_Prediction': KS_predictions,
    'Model_Prediction': model_predictions,
    'Human_Prediction': human_predictions,
    'KT_Prediction': KT_predictions,
    'MF_Prediction': MF_predictions,
    'YB_Prediction': YB_predictions,
    'JK_Prediction': JK_predictions,
    'DS_Prediction': DS_predictions,
    'YS_Prediction': YS_predictions,
    'MK_Prediction': MK_predictions,
    'Jenny_Prediction': Jenny_predictions,
    'Kliment_Prediction': Kliment_predictions,
    'Melcher_Prediction': Melcher_predictions,
    'DCDNScore_Prediction': DCDNScore_predictions,
    'Corolado_Prediction': Corolado_predictions,
    'Actual_Outcome': actual_outcomes
})

# Custom bins: 0/n, 1/n, ..., n/n
n = 12
bins = [i/n for i in range(n+1)]
labels = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins)-1)]
data['Probability_Bin'] = pd.cut(data['Probability'], bins=bins, labels=labels, include_lowest=True)

# グループの定義
grouped_bins = {
    'Strong Agreement': [labels[0], labels[11],labels[1], labels[10]],
    'General Agreement': [labels[2], labels[9]],
    'Weak Agreement': [labels[3], labels[8]],
    'Poor Agreement': [labels[4],labels[7],labels[5],labels[6]],
}

# grouped_bins = {
#     'Strong Agreement': [labels[0], labels[11],labels[1], labels[10]],
#     'General Agreement': [labels[2], labels[9],labels[3], labels[8]],
#     'Poor Agreement': [labels[4],labels[7],labels[5],labels[6]],
# }
# grouped_bins = {
#     'Very Strong Agreement': [labels[0], labels[6]],
#     'General Agreement': [labels[1], labels[5]],
#     'Weak Agreement': [labels[2], labels[4]],
#     'Very Poor Agreement': [labels[3]],
# }

# 各グループの精度を計算
def calculate_group_accuracy(group):
    subset = data[data['Probability_Bin'].isin(group)]
    accuracy = {}
    accuracy['KS'] = (subset['KS_Prediction'] == subset['Actual_Outcome']).sum() / subset['KS_Prediction'].notna().sum()
    accuracy['Model'] = (subset['Model_Prediction'] == subset['Actual_Outcome']).sum() / subset['Model_Prediction'].notna().sum()
    accuracy['Human'] = (subset['Human_Prediction'] == subset['Actual_Outcome']).sum() / subset['Model_Prediction'].notna().sum()
    accuracy['KT'] = (subset['KT_Prediction'] == subset['Actual_Outcome']).sum() / subset['KT_Prediction'].notna().sum()
    accuracy['MF'] = (subset['MF_Prediction'] == subset['Actual_Outcome']).sum() / subset['MF_Prediction'].notna().sum()
    accuracy['YB'] = (subset['YB_Prediction'] == subset['Actual_Outcome']).sum() / subset['YB_Prediction'].notna().sum()
    accuracy['JK'] = (subset['JK_Prediction'] == subset['Actual_Outcome']).sum() / subset['JK_Prediction'].notna().sum()
    accuracy['DS'] = (subset['DS_Prediction'] == subset['Actual_Outcome']).sum() / subset['DS_Prediction'].notna().sum()
    accuracy['YS'] = (subset['YS_Prediction'] == subset['Actual_Outcome']).sum() / subset['YS_Prediction'].notna().sum()
    accuracy['MK'] = (subset['MK_Prediction'] == subset['Actual_Outcome']).sum() / subset['MK_Prediction'].notna().sum()
    accuracy['Jenny'] = (subset['Jenny_Prediction'] == subset['Actual_Outcome']).sum() / subset['Jenny_Prediction'].notna().sum()
    accuracy['Kliment'] = (subset['Kliment_Prediction'] == subset['Actual_Outcome']).sum() / subset['Kliment_Prediction'].notna().sum()
    accuracy['Melcher'] = (subset['Melcher_Prediction'] == subset['Actual_Outcome']).sum() / subset['Melcher_Prediction'].notna().sum()
    accuracy['DCDNScore'] = (subset['DCDNScore_Prediction'] == subset['Actual_Outcome']).sum() / subset['DCDNScore_Prediction'].notna().sum()
    accuracy['Corolado'] = (subset['Corolado_Prediction'] == subset['Actual_Outcome']).sum() / subset['Corolado_Prediction'].notna().sum()
    return pd.Series(accuracy)

# 各グループに関数を適用
accuracy_df = pd.DataFrame({k: calculate_group_accuracy(v) for k, v in grouped_bins.items()}).T
accuracy_df['Human Average'] = accuracy_df[['KS', 'KT', 'MF', 'YB', 'JK', 'DS','YS','MK','Jenny','Kliment','Melcher']].mean(axis=1)

# プロット
fig, ax = plt.subplots(figsize=(9.5, 12))
accuracy_df['KS'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['KT'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['MF'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['YB'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['JK'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='Individual Surgeon')
accuracy_df['DS'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['YS'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['MK'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['Jenny'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['Kliment'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['Melcher'].plot(kind='line', ax=ax, marker='o', color=(0, 0, 0, 0.25), markersize=10, linestyle='-', linewidth=4, label='')
accuracy_df['DCDNScore'].plot(kind='line', ax=ax, marker='o', color='#2ca02c', markersize=10, linestyle='-', linewidth=4, label='DCD-N score Prediction')
accuracy_df['Corolado'].plot(kind='line', ax=ax, marker='o', color='#1a5d8f', markersize=10, linestyle='-', linewidth=4, label='Colorado Prediction')
accuracy_df['Model'].plot(kind='line', ax=ax, marker='o', color='#d62728', markersize=10, linestyle='-', linewidth=4, label='LGBM Model')
# accuracy_df['Human'].plot(kind='line', ax=ax, marker='o', color='black', markersize=10, linestyle='-', linewidth=4, label='Human Majority Decision Accuracy')
accuracy_df['Human Average'].plot(kind='line', ax=ax, marker='o', color='black', markersize=10, linestyle='-', linewidth=4, label='Human Average Accuracy')

# Define custom offsets for label placement to avoid overlap
def adjust_text_position(index, value, line_type):
    # Adjust specific overlapping points with more granular control
    if index == 0:  # First point
        if line_type == 'Model':  # Red line
            return 0.02  # Move label up
        elif line_type == 'DCDNScore':  # Blue line
            return -0.024 # Move label down
    if index == 1:  # Second point
        if line_type == 'Model':
            return -0.02
        elif line_type == 'DCDNScore':
            return 0.015
        elif line_type == 'Corolado':
            return -0.02
    if index == 2:  # Second point
        if line_type == 'Human Average':
            return -0.02
        elif line_type == 'Corolado':
            return 0.012
        elif line_type == 'DCDNScore':
            return -0.02
    if index == 3:  # Second point
        if line_type == 'DCDNScore':
            return -0.02
    # Default adjustment for non-overlapping points
    return 0.01

# Plot and add labels with adjusted positions
for i, value in enumerate(accuracy_df['Human Average']):
    ax.text(i, value + adjust_text_position(i, value, 'Human Average'), f'{value:.2f}', 
            ha='center', fontsize=13, color='black')
for i, value in enumerate(accuracy_df['Corolado']):
    ax.text(i, value + adjust_text_position(i, value, 'Corolado'), f'{value:.2f}', 
            ha='center', fontsize=13, color='#1a5d8f')
for i, value in enumerate(accuracy_df['DCDNScore']):
    ax.text(i, value + adjust_text_position(i, value, 'DCDNScore'), f'{value:.2f}', 
            ha='center', fontsize=13, color='#2ca02c')
for i, value in enumerate(accuracy_df['Model']):
    ax.text(i, value + adjust_text_position(i, value, 'Model'), f'{value:.2f}', 
            ha='center', fontsize=13, color='#d62728')

# プロットの仕上げ
ax.set_title('Accuracy (30min)',  fontsize=27)
ax.set_xlabel('Human Confidence Level',  fontsize=24)
ax.set_ylabel('Accuracy',  fontsize=24)
ax.set_yticks(np.arange(0.3, 1.1, 0.1))
ax.legend(fontsize=18)
ax.tick_params(axis='both', labelsize=22)
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xticks(rotation=45, fontsize=19)
# plt.savefig("Fig3b.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.show()

x_pros = pros.drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30', 
                    'ID','Survival_time','Status_120',], axis=1)
t_pros = np.array(pros['CSTATUS_30'].tolist())

scaled_pros = best_model.predict(x_pros)* 100

print(roc_auc_score(t_pros, scaled_pros))

thresholds = roc_curve(t_pros, scaled_pros)[2]
best_cutoff = max(thresholds, key=lambda th: accuracy_score(t_pros, scaled_pros >= th))
print(f"Best cutoff: {best_cutoff:.2f}")

final_preds = (scaled_pros >= 52).astype(int)
acc = accuracy_score(t_pros, final_preds)
print(f"Accuracy: {acc:.4f}")

tn, fp, fn, tp = confusion_matrix(t_pros, final_preds).ravel()
total = len(t_pros)
print(f"FPR (total): {fp / total:.4f}")
print(f"FNR (total): {fn / total:.4f}")

df = pros.copy()
df['ML_predi'] = scaled_pros.tolist()
df['ML_predi2'] = (df['ML_predi'] >= best_cutoff).astype(int)
df['Match_Discrepancy'] = np.where(df['ML_predi2'] == df['CSTATUS_30'], 0, 1)
bins = sorted(set([0, 20] + list(np.arange(30, df['ML_predi'].max() + 10, 10))))
df['ML_predi_bins'] = pd.cut(df['ML_predi'], bins=bins, right=False)
bin_counts = df.groupby('ML_predi_bins')['Match_Discrepancy'].sum()
total_counts = df['ML_predi_bins'].value_counts().sort_index()
percent_accuracy = (1 - bin_counts / total_counts) * 100

# X-axis labels for percentage bins
percent_labels = [f"{int(bin.left)}-{int(bin.right)}" for bin in total_counts.index]

# Plotting the chart
fig, ax1 = plt.subplots(figsize=(9, 7))

# Bar chart (left Y-axis)
colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(total_counts))) 
ax1.bar(percent_labels, total_counts, color=colors, width=0.6, label='Count')
ax1.set_xlabel('LGBM Model Index', fontsize=20)
ax1.set_ylabel('Count', fontsize=20)
ax1.tick_params(axis='x', rotation=45, labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0, max(total_counts) + 10)  # Add extra space above bars for the legend

# Line plot (right Y-axis)
ax2 = ax1.twinx()
ax2.plot(percent_labels, percent_accuracy, color='#1a5d8f', marker='o', label='Accuracy (%)')
ax2.set_ylabel('Accuracy (%)', fontsize=17)
ax2.tick_params(axis='y', labelsize=15)
ax2.set_ylim(0, 100)

# Add accuracy percentage above each point
for i, acc in enumerate(percent_accuracy):
    ax2.text(i, acc + 3, f"{acc:.1f}%", ha='center', fontsize=12, color='#1a5d8f')

# Combine legends and place them in the empty space
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=13, ncol=2)

# Add title
plt.title('Prospective Validation', fontsize=24)

plt.tight_layout()
plt.savefig("Fig5b.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.show()










