#Retrospective
##(x t):model creation dataset
##(x_vali t_vali):retrospective validation dataset

# Create dataset
train_data = lgb.Dataset(x, label=t)
test_data = lgb.Dataset(x_vali, label=t_vali, reference=train_data)

# Set parameters
params = {
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'objective': 'binary',  # Binary classification
    'metric': 'binary_logloss',  # Binary log loss as the evaluation metric
    'num_leaves': 31,  # Maximum number of leaves in a tree
    'learning_rate': 0.05,  # Learning rate
    'feature_fraction': 0.9,  # Feature subset fraction for boosting
    'bagging_fraction': 0.8,  # Data subset fraction for bagging
    'bagging_freq': 5,  # Frequency of bagging
    'verbose': 0,  # Verbosity level
    'early_stopping_rounds': 10  # Early stopping rounds
}

# Train model
model_total = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],  # Validation sets
    num_boost_round=100,  # Maximum number of boosting iterations
)

# Predict on test data
y_pred = model_total.predict(x_vali, num_iteration=model_total.best_iteration)

# Calculate AUC score
auc_score = roc_auc_score(t_vali, y_pred)
print(f"ROC AUC Score: {auc_score:.4f}")
joblib.dump(model_total, '####################################################.joblib')

def LGMOptuna(trial):
    dtrain = lgb.Dataset(x, label=t, weight=compute_sample_weight(class_weight='balanced', y=t).astype('float32'))
    dtest = lgb.Dataset(x_vali, label=t_vali, weight=np.ones(len(x_vali)).astype('float32'))
    
    params = {
        'objective': 'binary',  # Binary classification
        'metric': 'auc',  # AUC as evaluation metric
        'verbosity': -1,  # Suppress output
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        'random_state': trial.suggest_int('random_state', 1, 10000),  # Random state for reproducibility
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),  # Learning rate
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Subsampling fraction
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Column sampling fraction
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Minimum sum of instance weight
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),  # L2 regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  # L1 regularization
        'early_stopping_rounds': 10,  # Early stopping criteria
    }
    
    # Train model
    model_light = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dtest],  # Validation sets
    )
    
    # Predict
    predicted_probs = model_light.predict(x_vali)
    predicted_classes = (predicted_probs >= 0.5).astype(int)  # Convert probabilities to binary classes
    
    # Compute evaluation metrics
    auc_l = roc_auc_score(t_vali, predicted_probs)
    f1 = f1_score(t_vali, predicted_classes)
    recall = recall_score(t_vali, predicted_classes)
    precision = precision_score(t_vali, predicted_classes)
    
    # Display results
    print(f"AUC: {auc_l:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    # Save model
    directory = '##################################################'
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    model_path = os.path.join(directory, '###########################.joblib')
    with open(model_path, mode='wb') as f:
        pickle.dump(model_light, f)
    
    # Return inverse of AUC for optimization
    return 1 / auc_l

# Run optimization with Optuna
study = optuna.create_study()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(LGMOptuna, 10)

# Retrieve optimized AUC score
auc_light = 1 / study.best_value
print(f"Optimized AUC Score: {auc_light:.4f}")

#Prospective
##(x t):prospective validation dataset

# Load the model
model_path = 'Upload_models_created_in_retrospective_validation_phase'
try:
    model_total = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Obtain predictions
try:
    predi = model_total.predict(x)
except Exception as e:
    print(f"Error in prediction: {e}")
    exit()

# Convert predictions and test labels to NumPy arrays
x = np.array(predi)
y = np.array(t.tolist())

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y, x)
auc_ROC = auc(fpr, tpr)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fpr, tpr, color='r', lw=5, label=f'ROC curve (area = {auc_ROC:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
ax.fill_between(fpr, tpr, 0, color='r', alpha=0.2)  # Fill the area under the curve

# Set axis labels and title
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=26)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=26)
ax.set_title('Prospective ROC Curve', fontweight='bold', fontsize=26)
ax.set_facecolor('white')
fig.set_facecolor('white')

# Adjust tick label size
ax.tick_params(axis='both', which='major', labelsize=20)

# Configure legend
ax.legend(loc="lower right", prop={'weight':'bold', 'size': 20})
ax.grid(False)

# Make axis borders thicker
for spine in ax.spines.values():
    spine.set_linewidth(3)
    spine.set_color('black')

plt.show()

# Determine the optimal cutoff point
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal cutoff threshold:", optimal_threshold)

# Output AUC
print("AUC:", auc_ROC)