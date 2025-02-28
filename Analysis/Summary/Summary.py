#continuous variable summary
data_columns = ['#################################']
#unet: dataframe

# filtering
df1 = unet[unet['Validation'] != 'v'].dropna(subset=['recorder'])[data_columns]
df2 = unet[unet['Validation'] == 'v'].dropna(subset=['recorder'])[data_columns]

# cross table
summary_table = pd.DataFrame(index=data_columns, columns=['Development Data', 'Retrospective Validation Data', 'p-value'])

one_decimal_columns = ['#################################']

# Format the summary table
for col in data_columns:
    if col in one_decimal_columns:
        # Format to one decimal place for the specific columns
        summary_table.loc[col, 'Development Data'] = f"{float(df1[col].median()):.1f} ({float(df1[col].quantile(0.25)):.1f}-{float(df1[col].quantile(0.75)):.1f})"
        summary_table.loc[col, 'Retrospective Validation Data'] = f"{float(df2[col].median()):.1f} ({float(df2[col].quantile(0.25)):.1f}-{float(df2[col].quantile(0.75)):.1f})"
    else:
        # Format to no decimal places (integer) for other columns
        summary_table.loc[col, 'Development Data'] = f"{int(df1[col].median())} ({int(df1[col].quantile(0.25))}-{int(df1[col].quantile(0.75))})"
        summary_table.loc[col, 'Retrospective Validation Data'] = f"{int(df2[col].median())} ({int(df2[col].quantile(0.25))}-{int(df2[col].quantile(0.75))})"

    # Format p-value to three decimal places
    try:
        summary_table.loc[col, 'p-value'] = f"{mannwhitneyu(df1[col].dropna(), df2[col].dropna()).pvalue:.3f}"
    except ValueError:
        summary_table.loc[col, 'p-value'] = "NaN"  # Handle cases with insufficient data

summary_table

#categorical variable summary
# List of categorical variables
categorical_columns = ['############################']

# Filtering
df1 = unet[unet['Validation'] != 'v'].dropna(subset=['recorder'])[categorical_columns]
df2 = unet[unet['Validation'] == 'v'].dropna(subset=['recorder'])[categorical_columns]

# Create summary table
summary_table = pd.DataFrame(columns=['Variable', 'Category', 'Development Data (n, %)', 'Validation Data (n, %)', 'p-value'])

# Perform statistical tests for each categorical variable
for col in categorical_columns:
    # Get counts for each category and fill missing categories with 0
    counts_df1 = df1[col].value_counts().to_dict()
    counts_df2 = df2[col].value_counts().to_dict()
    all_categories = set(counts_df1.keys()).union(set(counts_df2.keys()))
    counts_df1 = {k: counts_df1.get(k, 0) for k in all_categories}
    counts_df2 = {k: counts_df2.get(k, 0) for k in all_categories}
    
    # Get total counts
    total_df1 = len(df1)
    total_df2 = len(df2)

    # Compute percentages and add to the summary table
    rows = []
    for category in all_categories:
        count1 = counts_df1[category]
        count2 = counts_df2[category]
        percent1 = (count1 / total_df1) * 100 if total_df1 > 0 else 0
        percent2 = (count2 / total_df2) * 100 if total_df2 > 0 else 0

        rows.append({
            'Variable': col if category == list(all_categories)[0] else "",  # Show variable name only for the first category
            'Category': category,
            'Development Data (n, %)': f"{count1} ({percent1:.1f}%)",
            'Validation Data (n, %)': f"{count2} ({percent2:.1f}%)",
            'p-value': ""  # Placeholder for p-value
        })

    # Add rows to summary table
    summary_table = pd.concat([summary_table, pd.DataFrame(rows)], ignore_index=True)

    # Create contingency table and compute p-value
    contingency_table = pd.DataFrame([counts_df1, counts_df2], index=['Development', 'Validation']).T
    if contingency_table.shape[1] == 2 and all(contingency_table.sum() > 0):
        try:
            chi2, p, _, _ = chi2_contingency(contingency_table)
        except ValueError:
            _, p = fisher_exact(contingency_table)
    else:
        p = np.nan

    # Assign p-value to the first row of the variable
    summary_table.loc[summary_table['Variable'] == col, 'p-value'] = f"{p:.3f}" if not pd.isna(p) else "NaN"

# Reset index for formatting
summary_table.reset_index(drop=True, inplace=True)
# summary_table.to_excel('formatted_categorical_comparison.xlsx')
summary_table

#donor survival curve after extubation
# Load dataset (replace 'your_data.csv' with actual filename)
df = pd.read_csv('your_data.csv')

# Remove rows with missing values in key columns
df_cleaned = df.dropna(subset=['Survival_time', 'Status_120'])

# Initialize and fit Kaplan-Meier estimator for full dataset
kmf_full = KaplanMeierFitter()
kmf_full.fit(durations=df_cleaned['Survival_time'], event_observed=df_cleaned['Status_120'])

# Filter data for Survival_time <= 200
df_filtered = df_cleaned[df_cleaned['Survival_time'] <= 200]

# Initialize and fit Kaplan-Meier estimator for filtered dataset
kmf_filtered = KaplanMeierFitter()
kmf_filtered.fit(durations=df_filtered['Survival_time'], event_observed=df_filtered['Status_120'])

# Plot survival curves
plt.figure(figsize=(9, 8))
kmf_full.plot_survival_function(ax=plt.gca(), linewidth=2.5, color='#d62728', label='All donors after extubation')
kmf_filtered.plot_survival_function(ax=plt.gca(), linewidth=2.5, color='blue', label='Only expired donors')

# Customize plot
plt.title('Survival Curve from Extubation', fontsize=25)
plt.xlabel('Time (minutes)', fontsize=23)
plt.ylabel('Survival Probability', fontsize=23)
plt.grid(False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlim(0, 120)
plt.ylim(0, 1)
plt.xticks(ticks=np.arange(15, 121, 15), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=20, loc='upper right', frameon=False)

# Add vertical and horizontal reference lines at time=45 minutes
time_point = 45
survival_prob_full = kmf_full.survival_function_at_times(time_point).values[0]
plt.axvline(x=time_point, ymin=0, ymax=survival_prob_full, color='black', linestyle='--', linewidth=2)
plt.axhline(y=survival_prob_full, xmin=0, xmax=time_point/120, color='black', linestyle='--', linewidth=2)

# Add Number at Risk and Event count
time_points = np.linspace(0, 120, 9)
at_risk = [sum(df_cleaned['Survival_time'] >= t) for t in time_points]
events = [sum((df_cleaned['Survival_time'] < t) & (df_cleaned['Status_120'] == 1)) for t in time_points]

for i, (t, n, e) in enumerate(zip(time_points, at_risk, events)):
    plt.text(t, -0.20, f"{int(n)}", fontsize=15, ha='center')  # Number at risk
    plt.text(t, -0.25, f"{int(e)}", fontsize=15, ha='center')  # Event count

# Add labels for Number at Risk and Event count
plt.text(-10, -0.20, "Number at risk", fontsize=15, ha='right')
plt.text(-10, -0.25, "Event", fontsize=15, ha='right')

# Show plot
plt.show()
