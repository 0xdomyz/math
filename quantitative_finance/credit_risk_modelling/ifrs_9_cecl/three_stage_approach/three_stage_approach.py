# Auto-extracted from markdown file
# Source: three_stage_approach.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# Portfolio parameters
n_loans = 1000
loan_amount = 100_000  # $100k per loan
lgd = 0.40  # 40% loss given default

# Origination: All loans start in Stage 1
df = pd.DataFrame({
    'loan_id': range(n_loans),
    'amount': loan_amount,
    'stage': 1,
    'pd_12m_orig': np.random.uniform(0.002, 0.01, n_loans),  # 0.2%-1% 12m PD at origination
    'pd_lifetime_orig': np.random.uniform(0.03, 0.08, n_loans),  # 3%-8% lifetime PD
    'rating_orig': np.random.choice(['AAA', 'AA', 'A', 'BBB'], n_loans, p=[0.1, 0.3, 0.4, 0.2])
})

# Simulate 1 year forward: Some loans migrate to Stage 2 or Stage 3
# Stage 2 triggers: PD increase > 2x origination, or rating downgrade 2+ notches
# Stage 3 triggers: Default (random draw based on PD)

# Current PD (after 1 year): Most stable, some deteriorate
pd_multiplier = np.random.lognormal(0, 0.5, n_loans)  # Log-normal shocks
df['pd_12m_current'] = df['pd_12m_orig'] * pd_multiplier
df['pd_lifetime_current'] = df['pd_lifetime_orig'] * pd_multiplier

# Clip PD to [0, 1]
df['pd_12m_current'] = df['pd_12m_current'].clip(0, 1)
df['pd_lifetime_current'] = df['pd_lifetime_current'].clip(0, 1)

# SICR detection: Relative PD increase > 2x
df['sicr_flag'] = (df['pd_12m_current'] / df['pd_12m_orig']) > 2.0

# Default simulation: Draw from Bernoulli(PD_12m)
df['default_flag'] = np.random.binomial(1, df['pd_12m_current'])

# Stage classification
def classify_stage(row):
    if row['default_flag'] == 1:
        return 3  # Default
    elif row['sicr_flag']:
        return 2  # SICR
    else:
        return 1  # No SICR

df['stage'] = df.apply(classify_stage, axis=1)

# ECL calculation
def calculate_ecl(row):
    if row['stage'] == 1:
        # 12-month ECL
        return row['amount'] * row['pd_12m_current'] * lgd
    elif row['stage'] == 2:
        # Lifetime ECL (performing)
        return row['amount'] * row['pd_lifetime_current'] * lgd
    else:
        # Stage 3: Default; ECL = LGD (PD = 100%)
        return row['amount'] * lgd

df['ecl'] = df.apply(calculate_ecl, axis=1)

# Summary statistics
print("="*70)
print("IFRS 9 Three-Stage Model: Loan Portfolio ECL")
print("="*70)
print(f"Total Loans: {n_loans}")
print(f"Total Exposure: ${df['amount'].sum():,.0f}")
print("")

stage_summary = df.groupby('stage').agg({
    'loan_id': 'count',
    'amount': 'sum',
    'ecl': 'sum'
}).rename(columns={'loan_id': 'count'})

stage_summary['coverage_ratio'] = (stage_summary['ecl'] / stage_summary['amount']) * 100

print("Stage Distribution:")
print("-"*70)
print(stage_summary)
print("")

total_ecl = df['ecl'].sum()
total_exposure = df['amount'].sum()
overall_coverage = (total_ecl / total_exposure) * 100

print(f"Total ECL Allowance: ${total_ecl:,.0f}")
print(f"Overall Coverage Ratio: {overall_coverage:.2f}%")
print("")

# Stage 2 breakdown
stage2_df = df[df['stage'] == 2]
if len(stage2_df) > 0:
    avg_pd_increase = (stage2_df['pd_12m_current'] / stage2_df['pd_12m_orig']).mean()
    print(f"Stage 2 Loans: Average PD increase {avg_pd_increase:.1f}× from origination")
    print("")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Stage distribution
ax = axes[0, 0]
stage_counts = df['stage'].value_counts().sort_index()
colors = ['green', 'orange', 'red']
ax.bar(stage_counts.index, stage_counts.values, color=colors, alpha=0.7)
ax.set_xlabel('Stage')
ax.set_ylabel('Number of Loans')
ax.set_title('Loan Distribution by Stage')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Stage 1\n(12m ECL)', 'Stage 2\n(Lifetime ECL)', 'Stage 3\n(Default)'])
ax.grid(axis='y', alpha=0.3)

# Plot 2: ECL by stage
ax = axes[0, 1]
stage_ecl = df.groupby('stage')['ecl'].sum()
ax.bar(stage_ecl.index, stage_ecl.values / 1e6, color=colors, alpha=0.7)
ax.set_xlabel('Stage')
ax.set_ylabel('ECL Allowance ($M)')
ax.set_title('Total ECL by Stage')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax.grid(axis='y', alpha=0.3)

# Plot 3: PD distribution by stage
ax = axes[1, 0]
for stage, color, label in zip([1, 2, 3], colors, ['Stage 1', 'Stage 2', 'Stage 3']):
    stage_data = df[df['stage'] == stage]['pd_12m_current']
    if len(stage_data) > 0:
        ax.hist(stage_data, bins=20, alpha=0.5, color=color, label=label)

ax.set_xlabel('Current 12-Month PD')
ax.set_ylabel('Frequency')
ax.set_title('PD Distribution by Stage')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Coverage ratio by stage
ax = axes[1, 1]
coverage = (df.groupby('stage')['ecl'].sum() / df.groupby('stage')['amount'].sum()) * 100
ax.bar(coverage.index, coverage.values, color=colors, alpha=0.7)
ax.set_xlabel('Stage')
ax.set_ylabel('Coverage Ratio (%)')
ax.set_title('ECL Coverage Ratio by Stage')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax.axhline(overall_coverage, color='black', linestyle='--', linewidth=2, label=f'Overall: {overall_coverage:.2f}%')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('three_stage_approach.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("Key Observations:")
print("="*70)
print("1. Stage 1 loans: Majority of portfolio; low ECL (0.1-1% coverage)")
print("2. Stage 2 loans: Elevated PD; significantly higher ECL (3-10% coverage)")
print("3. Stage 3 loans: Defaulted; highest ECL (40%+ coverage = LGD)")
print("")
print("4. SICR detection critical: Delayed Stage 2 migration → understated provisions")
print("5. Total ECL = sum across stages; weighted by exposure distribution")

