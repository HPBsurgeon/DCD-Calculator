# pros_validation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, confusion_matrix
)

# === Load or define your trained model ===
# from train_model.py or use joblib.load('model.pkl')
from train_model import general_model  # or best_model

# === Load prospective dataset ===
# Replace this with your actual data source
df207 = pd.read_csv("path_to_prospective_data.csv")  # <- set actual path

# === Feature selection ===
pros = df207[[
    'ID', 'Survival_time', 'Status_120',
    'GCS', 'pupil', 'gag', 'corneal', 'cough', 'motor', 'OBV',
    'MAP', 'Na', 'Plt',
    'initial_PF_ratio', 'end_PF_ratio', 'end_ph',
    'arrest_his', 'Mechanism_of_injury', 'BMI'
]]

# === Create outcome labels ===
pros['CSTATUS_30'] = np.select(
    [pros['Survival_time'] <= 30,
     pros['Survival_time'] > 30],
    [1, 0],
    default=np.nan
)
pros['CSTATUS_45'] = np.select(
    [pros['Survival_time'] <= 45,
     pros['Survival_time'] > 45],
    [1, 0],
    default=np.nan
)
pros['CSTATUS_60'] = np.select(
    [pros['Survival_time'] < 60,
     (pros['Survival_time'] == 60) & (pros['Status_120'] == 1),
     pros['Survival_time'] > 60],
    [1, 1, 0],
    default=np.nan
)

# === Type casting (to match training format) ===
data_types = {
    "GCS": "float64",
    "pupil": "category", "gag": "category", "corneal": "category",
    "cough": "category", "motor": "category", "OBV": "category",
    "MAP": "Int64", "Na": "Int64", "Plt": "Int64",
    "initial_PF_ratio": "Int64", "end_PF_ratio": "Int64", "end_ph": "Int64",
    "arrest_his": "category", "Mechanism_of_injury": "category",
    "BMI": "Int64"
}
for col, dtype in data_types.items():
    if col in pros.columns:
        pros[col] = pros[col].astype(dtype)

# === Prepare input/output ===
x_pros = pros.drop(['CSTATUS_60', 'CSTATUS_45', 'CSTATUS_30', 
                    'ID', 'Survival_time', 'Status_120'], axis=1)
t_pros = np.array(pros['CSTATUS_30'].tolist())

# === Predict ===
scaled_pros = general_model.predict(x_pros) * 100
print("Prospective AUC:", roc_auc_score(t_pros, scaled_pros))

# === Best cutoff based on accuracy ===
thresholds = roc_curve(t_pros, scaled_pros)[2]
best_cutoff = max(thresholds, key=lambda th: accuracy_score(t_pros, scaled_pros >= th))
print(f"Best cutoff: {best_cutoff:.2f}")

# === Fixed threshold performance (e.g., 52) ===
final_preds = (scaled_pros >= 52).astype(int)
acc = accuracy_score(t_pros, final_preds)
print(f"Accuracy: {acc:.4f}")

tn, fp, fn, tp = confusion_matrix(t_pros, final_preds).ravel()
total = len(t_pros)
print(f"FPR (total): {fp / total:.4f}")
print(f"FNR (total): {fn / total:.4f}")

# === Binned prediction analysis ===
df = pros.copy()
df['ML_predi'] = scaled_pros
df['ML_predi2'] = (df['ML_predi'] >= best_cutoff).astype(int)
df['Match_Discrepancy'] = np.where(df['ML_predi2'] == df['CSTATUS_30'], 0, 1)

# Define bin ranges (0-20% and 30-100% in 10% increments)
bins = sorted(set([0, 20] + list(np.arange(30, df['ML_predi'].max() + 10, 10))))
df['ML_predi_bins'] = pd.cut(df['ML_predi'], bins=bins, right=False)

# Calculate accuracy per bin
bin_counts = df.groupby('ML_predi_bins')['Match_Discrepancy'].sum()
total_counts = df['ML_predi_bins'].value_counts().sort_index()
percent_accuracy = (1 - bin_counts / total_counts) * 100
percent_labels = [f"{int(bin.left)}-{int(bin.right)}" for bin in total_counts.index]

# === Plot ===
fig, ax1 = plt.subplots(figsize=(9, 7))

# Bar plot: number of cases per prediction bin
colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(total_counts))) 
ax1.bar(percent_labels, total_counts, color=colors, width=0.6, label='Count')
ax1.set_xlabel('LGBM Model Index', fontsize=20)
ax1.set_ylabel('Count', fontsize=20)
ax1.tick_params(axis='x', rotation=45, labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0, max(total_counts) + 10)

# Line plot: accuracy per bin
ax2 = ax1.twinx()
ax2.plot(percent_labels, percent_accuracy, color='#1a5d8f', marker='o', label='Accuracy (%)')
ax2.set_ylabel('Accuracy (%)', fontsize=17)
ax2.tick_params(axis='y', labelsize=15)
ax2.set_ylim(0, 100)

# Add accuracy text annotations
for i, acc in enumerate(percent_accuracy):
    ax2.text(i, acc + 3, f"{acc:.1f}%", ha='center', fontsize=12, color='#1a5d8f')

# Combined legend and title
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=13, ncol=2)
plt.title('Prospective Validation', fontsize=24)

plt.tight_layout()
plt.show()
