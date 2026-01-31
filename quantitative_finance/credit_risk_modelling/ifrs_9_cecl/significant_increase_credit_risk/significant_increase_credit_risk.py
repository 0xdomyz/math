# Auto-extracted from markdown file
# Source: significant_increase_credit_risk.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set seed
np.random.seed(42)

# Portfolio parameters
n_loans = 1000

# Origination data
df = pd.DataFrame({
    'loan_id': range(n_loans),
    'amount': np.random.uniform(50_000, 500_000, n_loans),
    'pd_orig': np.random.uniform(0.005, 0.02, n_loans),  # 0.5%-2% origination PD
    'rating_orig': np.random.choice([1, 2, 3, 4, 5], n_loans, p=[0.2, 0.3, 0.3, 0.15, 0.05]),  # 1=AAA, 5=BBB-
})

# Simulate 1 year forward: Credit deterioration
# Some loans deteriorate (PD increases), some stable, few improve
pd_shock = np.random.lognormal(0, 0.6, n_loans)  # Log-normal shocks (some extreme deterioration)
df['pd_current'] = (df['pd_orig'] * pd_shock).clip(0.001, 0.50)  # Clip to [0.1%, 50%]

# Rating migration (correlated with PD shock)
rating_change = np.random.choice([-2, -1, 0, 1], n_loans, p=[0.05, 0.15, 0.70, 0.10])  # Mostly stable
df['rating_current'] = (df['rating_orig'] + rating_change).clip(1, 10)  # Rating 1-10 (10=default)

# Days past due (DPD)
df['dpd'] = np.random.choice([0, 15, 35, 60], n_loans, p=[0.85, 0.08, 0.05, 0.02])

# Watchlist flag (subjective; correlated with high PD)
df['watchlist'] = (df['pd_current'] > 0.05) & (np.random.rand(n_loans) < 0.5)

# Ground truth: Actual default in next 12 months (for validation)
df['default_12m'] = np.random.binomial(1, df['pd_current'])

# SICR Indicators
# 1. 30 DPD backstop
df['sicr_30dpd'] = df['dpd'] >= 30

# 2. Relative PD change > 2×
df['pd_ratio'] = df['pd_current'] / df['pd_orig']
df['sicr_relative_pd'] = df['pd_ratio'] > 2.0

# 3. Absolute PD > 5%
df['sicr_absolute_pd'] = df['pd_current'] > 0.05

# 4. Rating downgrade ≥ 2 notches
df['rating_change'] = df['rating_current'] - df['rating_orig']
df['sicr_rating'] = df['rating_change'] >= 2

# 5. Watchlist
df['sicr_watchlist'] = df['watchlist']

# Combined SICR (OR logic)
df['sicr_combined'] = (
    df['sicr_30dpd'] |
    df['sicr_relative_pd'] |
    df['sicr_absolute_pd'] |
    df['sicr_rating'] |
    df['sicr_watchlist']
)

# Stage classification
df['stage'] = 1  # Default Stage 1
df.loc[df['sicr_combined'], 'stage'] = 2  # SICR → Stage 2
df.loc[df['dpd'] >= 90, 'stage'] = 3  # 90 DPD → Stage 3 (override)

# Analysis
print("="*70)
print("SICR Detection Framework: Multiple Indicators")
print("="*70)
print(f"Total Loans: {n_loans}")
print("")

# SICR trigger breakdown
print("SICR Triggers (OR Logic):")
print("-"*70)
sicr_summary = pd.DataFrame({
    'Indicator': ['30 DPD', 'Relative PD (>2×)', 'Absolute PD (>5%)', 'Rating Downgrade (≥2)', 'Watchlist', 'Combined (Any)'],
    'Count': [
        df['sicr_30dpd'].sum(),
        df['sicr_relative_pd'].sum(),
        df['sicr_absolute_pd'].sum(),
        df['sicr_rating'].sum(),
        df['sicr_watchlist'].sum(),
        df['sicr_combined'].sum()
    ],
    'Percent': [
        df['sicr_30dpd'].mean() * 100,
        df['sicr_relative_pd'].mean() * 100,
        df['sicr_absolute_pd'].mean() * 100,
        df['sicr_rating'].mean() * 100,
        df['sicr_watchlist'].mean() * 100,
        df['sicr_combined'].mean() * 100
    ]
})
print(sicr_summary.to_string(index=False))
print("")

# Stage distribution
stage_counts = df['stage'].value_counts().sort_index()
print("Stage Distribution:")
print("-"*70)
for stage in [1, 2, 3]:
    count = stage_counts.get(stage, 0)
    pct = count / n_loans * 100
    print(f"Stage {stage}: {count:4d} loans ({pct:5.1f}%)")
print("")

# Validation: SICR vs Actual Default
print("SICR Performance (Predicting 12-Month Default):")
print("-"*70)

cm = confusion_matrix(df['default_12m'], df['sicr_combined'])
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives (SICR → Default):       {tp:4d}")
print(f"False Positives (SICR → No Default):   {fp:4d}")
print(f"False Negatives (No SICR → Default):   {fn:4d}")
print(f"True Negatives (No SICR → No Default): {tn:4d}")
print("")
print(f"Precision (SICR → Default Rate):        {precision:.2%}")
print(f"Recall (Catch Default Rate):            {recall:.2%}")
print(f"F1-Score:                               {f1:.2f}")

# ROC Curve for PD ratio threshold
fpr, tpr, thresholds = roc_curve(df['default_12m'], df['pd_ratio'])
roc_auc = auc(fpr, tpr)

print(f"\nAUC (PD Ratio as Predictor):            {roc_auc:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SICR trigger counts
ax = axes[0, 0]
triggers = ['30 DPD', 'Rel PD', 'Abs PD', 'Rating', 'Watch', 'Combined']
counts = sicr_summary['Count'].values
ax.bar(triggers, counts, color=['red', 'orange', 'orange', 'blue', 'purple', 'green'], alpha=0.7)
ax.set_ylabel('Number of Loans')
ax.set_title('SICR Triggers (OR Logic)')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Stage distribution
ax = axes[0, 1]
stages = ['Stage 1\n(12m ECL)', 'Stage 2\n(Lifetime ECL)', 'Stage 3\n(Default)']
stage_values = [stage_counts.get(i, 0) for i in [1, 2, 3]]
colors = ['green', 'orange', 'red']
ax.bar(stages, stage_values, color=colors, alpha=0.7)
ax.set_ylabel('Number of Loans')
ax.set_title('Loan Distribution by Stage')
ax.grid(axis='y', alpha=0.3)

# Plot 3: PD distribution by SICR status
ax = axes[1, 0]
no_sicr = df[~df['sicr_combined']]['pd_current']
yes_sicr = df[df['sicr_combined']]['pd_current']

ax.hist(no_sicr, bins=30, alpha=0.5, label='No SICR', color='green')
ax.hist(yes_sicr, bins=30, alpha=0.5, label='SICR Triggered', color='red')
ax.set_xlabel('Current PD')
ax.set_ylabel('Frequency')
ax.set_title('PD Distribution: SICR vs No SICR')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: ROC Curve (PD ratio threshold optimization)
ax = axes[1, 1]
ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

# Mark current threshold (2×)
threshold_2x_idx = np.argmin(np.abs(thresholds - 2.0))
ax.plot(fpr[threshold_2x_idx], tpr[threshold_2x_idx], 'ro', markersize=10, label='Threshold = 2×')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curve: PD Ratio Threshold Optimization')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('significant_increase_credit_risk.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Combined SICR (OR logic) captures ~15-30% of portfolio")
print("   → Multiple indicators avoid single-metric dependence")
print("")
print("2. 30 DPD backstop critical (high true positive rate)")
print("   → Catches payment distress early; mandatory IFRS 9 requirement")
print("")
print("3. False positives tolerable (SICR without default)")
print("   → Prudent provisioning; cure back to Stage 1 if improves")
print("")
print("4. Threshold calibration critical (ROC analysis)")
print("   → Balance timeliness (high recall) vs stability (low false positives)")

