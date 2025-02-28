# Suppress Optuna logging output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Split the dataset into training and test sets
x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size=0.2, random_state=####)

# Train LightGBM model
print("Training LightGBM model...")
dtrain = lgb.Dataset(x_train, label=t_train)
dtest = lgb.Dataset(x_test, label=t_test_light)
params = {
    'objective': 'binary',  # Binary classification
    'metric': 'auc',  # Use AUC as evaluation metric
    'verbosity': -1,  # Suppress verbose output
    'boosting_type': 'gbdt',  # Use gradient boosting
    'early_stopping_rounds': 10  # Enable early stopping
}
model_light = lgb.train(params, dtrain, valid_sets=dtest)

# Extract feature importance
importance_gain = model_light.feature_importance(importance_type='gain')
importance_split = model_light.feature_importance(importance_type='split')

# Store feature importance in DataFrame
importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Gain': importance_gain,
    'Split': importance_split
}).sort_values(by='Gain', ascending=False)

# Plot Feature Importance (Gain)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Gain'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (Gain)')
plt.xlabel('Importance (Gain)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Plot Feature Importance (Split)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Split'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (Split)')
plt.xlabel('Importance (Split)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# Set the target variable for analysis
target = '#####'
plt.figure(figsize=(9, 6))

# Extract median or mode for other variables
drop_target_list = [e for e in x.columns.tolist() if e != target]
x_vali = pd.DataFrame(index=[0], columns=x.columns)
for col in drop_target_list:
    if x[col].dtype == 'object' or x[col].dtype.name == 'category':
        x_vali[col] = pd.Categorical(x[col].mode()[0])  # Use mode for categorical variables
    else:
        x_vali[col] = x[col].median()  # Use median for numerical variables

# Set step size for variable increments
d = 1  # Step size for BMI analysis

# Get min and max values for the target variable
lower, upper = x[target].min(), x[target].max()
if target == '#####':
    upper = #####  # Limit maximum value to #####

# Generate range list for target variable values
target_range_list = [i for i in range(int((upper - lower) / d) + 2)]
df_proba = pd.DataFrame(np.zeros(len(target_range_list)), index=target_range_list)

# Train LightGBM model and predict probabilities
n_iterations = 10
for i in range(n_iterations):
    dtrain = lgb.Dataset(x_train, label=t_train)
    dtest = lgb.Dataset(x_test, label=t_test_light)
    params = {
        'objective': 'binary',  # Binary classification
        'metric': 'auc',  # Use AUC as evaluation metric
        'verbosity': -1,  # Suppress verbose output
        'boosting_type': 'gbdt',  # Use gradient boosting
        'early_stopping_rounds': 10  # Enable early stopping
    }
    
    model_light = lgb.train(params, dtrain, valid_sets=dtest)
    prob_list = []
    
    for a in range(int((upper - lower) / d) + 2):
        x_vali[target] = a * d + lower  # Adjust ##### value step by step
        proba = model_light.predict(x_vali)[0]  # Predict probability
        prob_list.append(proba)
    
    df_proba[i] = prob_list

# Calculate mean probability
y_proba_mean = df_proba.mean(axis='columns').tolist()

# Plot the probability distribution of #####
X = [i * d + lower for i in range(int((upper - lower) / d) + 2)]
y = y_proba_mean

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(X, y, marker='o', lw=6)  # Line plot with markers
ax.set_xlabel('#####', fontsize=38)  # X-axis label
ax.set_ylabel('Probability', fontsize=38)  # Y-axis label
ax.set_ylim(0.16, 0.30)  # Set Y-axis range
ax.tick_params(axis='x', labelsize=31)
ax.tick_params(axis='y', labelsize=31)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


# Train LightGBM model with SHAP analysis
print("Training LightGBM model...")
n_iterations = 10
shap_values_list = []

for i in range(n_iterations):
    dtrain = lgb.Dataset(x_train, label=t_train)
    dtest = lgb.Dataset(x_test, label=t_test_light)
    params = {
        'objective': 'binary',  # Binary classification
        'metric': 'auc',  # Use AUC as evaluation metric
        'verbosity': -1,  # Suppress verbose output
        'boosting_type': 'gbdt',  # Use gradient boosting
        'early_stopping_rounds': 10  # Enable early stopping
    }
    model_light = lgb.train(params, dtrain, valid_sets=dtest)
    
    # Calculate SHAP values
    explainer = shap.Explainer(model_light, x_train)
    shap_values = explainer(x_train)
    shap_values_list.append(shap_values.values)

# Compute average SHAP values
mean_shap_values = np.mean(np.array(shap_values_list), axis=0)

# Extract feature importance
importance_gain = model_light.feature_importance(importance_type='gain')
importance_split = model_light.feature_importance(importance_type='split')

# Store feature importance in DataFrame
importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Gain': importance_gain,
    'Split': importance_split
}).sort_values(by='Gain', ascending=False)

# Plot Feature Importance (Gain)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Gain'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (Gain)')
plt.xlabel('Importance (Gain)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Plot Feature Importance (Split)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Split'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (Split)')
plt.xlabel('Importance (Split)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# SHAP Summary Plot
shap.summary_plot(mean_shap_values, x_train, feature_names=x.columns, plot_type="dot", max_display=20)

###With the following selected variables
# Perform multiple train-test splits and compute average AUC
auc_scores = []
n_splits = #####

for _ in range(n_splits):
    x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size=0.2)
    
    # Create dataset
    train_data = lgb.Dataset(x_train, label=t_train)
    test_data = lgb.Dataset(x_test, label=t_test_light, reference=train_data)
    
    # Set parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',  # Use 'multiclass' for multi-class classification or 'regression' for regression
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    
    # Train the model
    model_total = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        num_boost_round=100,
    )
    
    # Make predictions on test data
    y_pred = model_total.predict(x_test, num_iteration=model_total.best_iteration)
    
    # Compute AUC score
    auc_score = roc_auc_score(t_test_light, y_pred)
    auc_scores.append(auc_score)

# Compute average AUC
average_auc = np.mean(auc_scores)
print(f"Average AUC over {n_splits} splits: {average_auc:.4f}")


###search for maximum auc model
auc_list=[]
for _ in range(n_splits):
    x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size = 0.2)
    def LGMOptuna(trial):
        dtrain = lgb.Dataset(x_train, label=t_train, weight=compute_sample_weight(class_weight='balanced', y=t_train).astype('float32'))
        dtest = lgb.Dataset(x_test, label=t_test_light, weight=np.ones(len(x_test)).astype('float32'))
        params = {
        'objective':'binary',
        'metric':'auc',
        'verbosity':-1,
        'boosting_type':'gbdt',
        'random_state':trial.suggest_int('random_state', 1, 10000), 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 10,
        'verbose_eval': False
        }
        model_light = lgb.train(params,
                                dtrain,
                                valid_sets = [dtrain, dtest],
                                )
        predicted_light = model_light.predict(x_test)
        auc_l = roc_auc_score(t_test_light, predicted_light)
    
        return 1/auc_l

    study = optuna.create_study()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(LGMOptuna, 10)
    auc_list.append(1/study.best_value)

auc_list



