import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from lifelines import KaplanMeierFitter


df = pd.read_excel('Final_kaplan_data.xlsx')
df_cleaned = df.dropna(subset=['Survival_time', 'Status_120'])

kmf_full = KaplanMeierFitter()
kmf_full.fit(durations=df_cleaned['Survival_time'], event_observed=df_cleaned['Status_120'])
df_filtered = df_cleaned[df_cleaned['Status_120'] == 1]

kmf_filtered = KaplanMeierFitter()
kmf_filtered.fit(durations=df_filtered['Survival_time'], event_observed=df_filtered['Status_120'])

plt.figure(figsize=(9, 8))

kmf_full.plot_survival_function(ax=plt.gca(), linewidth=2.5, color='#d62728', label='All donors after extubation')

kmf_filtered.plot_survival_function(ax=plt.gca(), linewidth=2.5, color='blue', label='Only expired donors')

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

time_point = 45
survival_prob_full = kmf_full.survival_function_at_times(time_point).values[0]
plt.axvline(x=time_point, ymin=0, ymax=survival_prob_full, color='black', linestyle='--', linewidth=2)
plt.axhline(y=survival_prob_full, xmin=0, xmax=time_point/120, color='black', linestyle='--', linewidth=2)

time_points = np.linspace(0, 120, 9)  # 0, 15, ..., 120
at_risk = [sum(df_cleaned['Survival_time'] > t) for t in time_points]
events = [sum((df_cleaned['Survival_time'] <= t) & (df_cleaned['Status_120'] == 1)) for t in time_points]
censored = [sum((df_cleaned['Survival_time'] <= t) & (df_cleaned['Status_120'] == 0)) for t in time_points]

for t, n, e, c in zip(time_points, at_risk, events, censored):
    plt.text(t, -0.20, f"{int(n)}", fontsize=15, ha='center')  # Number at risk
    plt.text(t, -0.25, f"{int(e)}", fontsize=15, ha='center')  # Event
    plt.text(t, -0.30, f"{int(c)}", fontsize=15, ha='center')  # Censored

plt.text(-10, -0.20, "Number at risk", fontsize=15, ha='right')
plt.text(-10, -0.25, "Event", fontsize=15, ha='right')
plt.text(-10, -0.30, "Censored", fontsize=15, ha='right')

# plt.savefig("Fig1a.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.savefig("Fig1a.eps", format='eps', dpi=400, bbox_inches='tight')
plt.show()

cstatus_30_counts = df['Time_distri_2'].value_counts().sort_index()
# cstatus_30_counts = df['60_60_time_2'].value_counts().sort_index()
# cstatus_30_counts = df['80_80_time_2'].value_counts().sort_index()
cstatus_30_counts = df['50_time'].value_counts().sort_index()
total_count = cstatus_30_counts.sum()

plt.figure(figsize=(8, 8))

cmap = cm.get_cmap('Reds') 
gradient_values = np.linspace(0.4, 0.6, len(cstatus_30_counts) - 1)  
left_colors = [cmap(value) for value in gradient_values]  

right_color = (26/255, 93/255, 143/255, 0.7)  
bar_colors = left_colors + [right_color]
bars = plt.bar(cstatus_30_counts.index, cstatus_30_counts.values, width=0.5, color=bar_colors)

plt.xlabel('After 50mmHg', fontsize=23)##80mmHg/80% Extubation 60%/60mmHg
plt.ylabel('Frequency', fontsize=23)
plt.title('', fontweight='bold', fontsize=23)


# plt.xticks([0, 1, 2, 3], ['~30', '30~45', '45~60', 'Did not\nReach 80%/80mmHg'], rotation='horizontal', fontsize=18)
plt.xticks([0, 1, 2, 3, 4], ['~30', '31~45', '46~60','61~', 'Did not \nexpire'], rotation='horizontal', fontsize=18)
# plt.xticks([0, 1, 2, 3], ['~30', '30~45', '45~60','60~'], rotation='horizontal', fontsize=18)
plt.yticks(fontsize=20)##Did not\nReach 80/80

y_max = cstatus_30_counts.max() * 1.2 
plt.ylim(0, y_max)

for i, bar in enumerate(bars):
    count = cstatus_30_counts[i]
    percentage = (count / total_count) * 100
    color = 'black'  
    plt.text(
        bar.get_x() + bar.get_width() / 2,  
        bar.get_height() + y_max * 0.02,  
        f'{count}\n({percentage:.1f}%)',  
        ha='center', va='bottom', fontsize=15, color=color
    )


ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(2)

plt.tight_layout()
plt.savefig("Fig1e.pdf", format='pdf', dpi=400, bbox_inches='tight')
plt.show()

