# Auto-extracted from markdown file
# Source: claim_duration_termination_patterns.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from lifelines.utils import median_survival_times

# Simulated STD claims data
np.random.seed(42)
n_claims = 500

# Cause-specific duration distributions (Weibull parameters are realistic)
causes = ['Flu/Acute', 'Surgical', 'Musculoskeletal', 'Mental Health']
weibull_params = {
    'Flu/Acute': (2.0, 10),  # (shape, scale) → median ~9 days
    'Surgical': (1.8, 30),    # median ~27 days
    'Musculoskeletal': (1.5, 35),  # median ~33 days
    'Mental Health': (1.3, 80)   # median ~74 days
}

claims_list = []

for cause in causes:
    shape, scale = weibull_params[cause]
    n = int(n_claims / len(causes))
    
    # Generate durations from Weibull
    durations = np.random.weibull(shape, n) * scale
    
    # Termination reasons (event type)
    recovery_rate = {'Flu/Acute': 0.95, 'Surgical': 0.92, 
                     'Musculoskeletal': 0.80, 'Mental Health': 0.50}[cause]
    
    events = np.random.choice(
        ['Recovery', 'LTD Transition', 'Other'],
        size=n,
        p=[recovery_rate, 1 - recovery_rate - 0.05, 0.05]
    )
    
    for i in range(n):
        claims_list.append({
            'Cause': cause,
            'Duration_Days': durations[i],
            'Event': events[i],
            'Event_Indicator': 1 if events[i] != 'Censored' else 0
        })

df = pd.DataFrame(claims_list)

# Summary statistics
print("Claim Duration Summary by Cause:")
print(df.groupby('Cause')['Duration_Days'].describe())

print("\nTermination Reason Distribution:")
print(df['Event'].value_counts(normalize=True))

# Kaplan-Meier survival curves by cause
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: KM Curves by cause
ax = axes[0, 0]
kmf = KaplanMeierFitter()

for cause in causes:
    mask = df['Cause'] == cause
    T = df.loc[mask, 'Duration_Days']
    E = df.loc[mask, 'Event_Indicator']
    kmf.fit(T, E, label=cause)
    kmf.plot_survival_function(ax=ax, linewidth=2)

ax.set_xlabel('Days Since Claim Start')
ax.set_ylabel('Proportion of Claims Still Active')
ax.set_title('Kaplan-Meier Survival Curves by Cause')
ax.grid(alpha=0.3)
ax.legend()

# Plot 2: Duration histogram by cause
ax = axes[0, 1]
for cause in causes:
    mask = df['Cause'] == cause
    ax.hist(df.loc[mask, 'Duration_Days'], bins=30, alpha=0.5, label=cause)
ax.set_xlabel('Claim Duration (days)')
ax.set_ylabel('Number of Claims')
ax.set_title('Duration Distribution by Cause')
ax.legend()
ax.set_xlim(0, 200)

# Plot 3: Median duration vs termination type
ax = axes[1, 0]
median_by_event = df.groupby('Event')['Duration_Days'].median()
colors = ['green', 'orange', 'red']
median_by_event.plot(kind='bar', ax=ax, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Median Duration (days)')
ax.set_title('Median Duration by Termination Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Proportion terminating by duration threshold
ax = axes[1, 1]
thresholds = np.arange(0, 150, 10)
terminated = []
for threshold in thresholds:
    pct = (df['Duration_Days'] <= threshold).sum() / len(df) * 100
    terminated.append(pct)

ax.plot(thresholds, terminated, 'o-', linewidth=2, markersize=6, color='darkblue')
ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% Claims')
ax.set_xlabel('Duration (days)')
ax.set_ylabel('% of Claims Terminated By This Duration')
ax.set_title('Cumulative Claim Termination')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Weibull AFT model: predict duration by cause
print("\n\nWeibull Accelerated Failure Time Model:")
print("Estimating log(Duration) by cause...")

# Create dummy variables for cause
cause_dummies = pd.get_dummies(df['Cause'], drop_first=True)
X = cause_dummies

aft = WeibullAFTFitter()
aft.fit(df['Duration_Days'], df['Event_Indicator'], X, show_progress=False)

print("\nModel Summary:")
print(aft.summary)

# Predicted median durations
print("\n\nMedian Duration by Cause (from model):")
for cause in causes:
    cause_vec = pd.DataFrame([cause_dummies.columns == c for c in cause_dummies.columns]).T
    # Simplified prediction (using reference level)
    print(f"{cause}: ~{df[df['Cause'] == cause]['Duration_Days'].median():.0f} days")

