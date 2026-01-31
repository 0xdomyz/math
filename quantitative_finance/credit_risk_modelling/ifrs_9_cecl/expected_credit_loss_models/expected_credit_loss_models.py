# Auto-extracted from markdown file
# Source: expected_credit_loss_models.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# Portfolio parameters
n_loans = 500
loan_amounts = np.random.uniform(50_000, 500_000, n_loans)
maturities = np.random.randint(1, 10, n_loans)  # 1-10 years remaining
lgd = 0.40  # 40% LGD (constant for simplicity)
eir = 0.05  # 5% effective interest rate (discount rate)

# PD term structure (annual marginal PD)
# Base scenario: Increasing PD over time (credit deterioration)
def generate_pd_term_structure(maturity, base_pd=0.01, scenario='base'):
    """Generate annual marginal PD for loan over maturity."""
    if scenario == 'base':
        # Gradual increase: 1%, 1.5%, 2%, 2.5%, ...
        pds = [base_pd * (1 + 0.5 * t) for t in range(maturity)]
    elif scenario == 'adverse':
        # Recession: Higher PD (2× base)
        pds = [base_pd * 2 * (1 + 0.5 * t) for t in range(maturity)]
    elif scenario == 'upside':
        # Benign: Lower PD (0.5× base)
        pds = [base_pd * 0.5 * (1 + 0.5 * t) for t in range(maturity)]
    
    return np.array(pds)

# Calculate 12-month ECL
def calc_12m_ecl(ead, pd_12m, lgd):
    """Stage 1: 12-month ECL (no discounting for simplicity)."""
    return ead * pd_12m * lgd

# Calculate lifetime ECL
def calc_lifetime_ecl(ead, pds, lgd, eir):
    """Stage 2/3: Lifetime ECL with discounting."""
    ecl = 0
    survival_prob = 1.0
    
    for t, pd_t in enumerate(pds, start=1):
        marginal_loss = ead * pd_t * lgd * survival_prob
        discount_factor = 1 / (1 + eir) ** t
        ecl += marginal_loss * discount_factor
        survival_prob *= (1 - pd_t)
    
    return ecl

# Generate portfolio
portfolio = []

for i in range(n_loans):
    loan = {
        'loan_id': i,
        'amount': loan_amounts[i],
        'maturity': maturities[i],
        'base_pd_12m': np.random.uniform(0.005, 0.02, 1)[0]  # 0.5%-2% 12m PD
    }
    
    # PD term structures for each scenario
    loan['pds_base'] = generate_pd_term_structure(loan['maturity'], loan['base_pd_12m'], 'base')
    loan['pds_adverse'] = generate_pd_term_structure(loan['maturity'], loan['base_pd_12m'], 'adverse')
    loan['pds_upside'] = generate_pd_term_structure(loan['maturity'], loan['base_pd_12m'], 'upside')
    
    # 12-month ECL (Stage 1)
    loan['ecl_12m'] = calc_12m_ecl(loan['amount'], loan['base_pd_12m'], lgd)
    
    # Lifetime ECL by scenario (Stage 2)
    loan['ecl_lifetime_base'] = calc_lifetime_ecl(loan['amount'], loan['pds_base'], lgd, eir)
    loan['ecl_lifetime_adverse'] = calc_lifetime_ecl(loan['amount'], loan['pds_adverse'], lgd, eir)
    loan['ecl_lifetime_upside'] = calc_lifetime_ecl(loan['amount'], loan['pds_upside'], lgd, eir)
    
    # Scenario-weighted ECL (Stage 2): 50% base, 30% adverse, 20% upside
    loan['ecl_lifetime_weighted'] = (
        0.5 * loan['ecl_lifetime_base'] +
        0.3 * loan['ecl_lifetime_adverse'] +
        0.2 * loan['ecl_lifetime_upside']
    )
    
    portfolio.append(loan)

df = pd.DataFrame(portfolio)

# Summary statistics
print("="*70)
print("IFRS 9 ECL Model: 12-Month vs Lifetime ECL")
print("="*70)
print(f"Number of Loans: {n_loans}")
print(f"Total Exposure: ${df['amount'].sum():,.0f}")
print(f"Average Maturity: {df['maturity'].mean():.1f} years")
print("")

# Aggregate ECL
total_12m = df['ecl_12m'].sum()
total_lifetime_base = df['ecl_lifetime_base'].sum()
total_lifetime_weighted = df['ecl_lifetime_weighted'].sum()

print("Aggregate ECL:")
print("-"*70)
print(f"12-Month ECL (Stage 1):      ${total_12m:,.0f}")
print(f"Lifetime ECL (Base):         ${total_lifetime_base:,.0f}")
print(f"Lifetime ECL (Weighted):     ${total_lifetime_weighted:,.0f}")
print("")

# Coverage ratios
coverage_12m = (total_12m / df['amount'].sum()) * 100
coverage_lifetime = (total_lifetime_weighted / df['amount'].sum()) * 100

print(f"Coverage Ratio (12m ECL):    {coverage_12m:.2f}%")
print(f"Coverage Ratio (Lifetime):   {coverage_lifetime:.2f}%")
print(f"Lifetime ECL / 12m ECL:      {total_lifetime_weighted / total_12m:.1f}×")
print("")

# Scenario impact
scenario_impact = (df['ecl_lifetime_adverse'].sum() - df['ecl_lifetime_upside'].sum()) / df['ecl_lifetime_base'].sum() * 100
print(f"Scenario Impact: Adverse vs Upside = {scenario_impact:.0f}% swing")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: 12m vs Lifetime ECL by loan
ax = axes[0, 0]
ax.scatter(df['ecl_12m'], df['ecl_lifetime_weighted'], alpha=0.5, s=30, color='blue')
ax.plot([0, df['ecl_12m'].max()], [0, df['ecl_12m'].max()], 'r--', linewidth=1, label='1:1 line')
ax.set_xlabel('12-Month ECL ($)')
ax.set_ylabel('Lifetime ECL (Weighted) ($)')
ax.set_title('Comparison: 12-Month vs Lifetime ECL')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: ECL by maturity
ax = axes[0, 1]
maturity_grouped = df.groupby('maturity').agg({'ecl_12m': 'sum', 'ecl_lifetime_weighted': 'sum'})
x = maturity_grouped.index
ax.bar(x - 0.2, maturity_grouped['ecl_12m'] / 1e3, width=0.4, label='12m ECL', alpha=0.7, color='green')
ax.bar(x + 0.2, maturity_grouped['ecl_lifetime_weighted'] / 1e3, width=0.4, label='Lifetime ECL', alpha=0.7, color='orange')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Total ECL ($1000s)')
ax.set_title('ECL by Maturity')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Scenario comparison
ax = axes[1, 0]
scenarios = ['Base', 'Adverse', 'Upside', 'Weighted']
scenario_ecl = [
    df['ecl_lifetime_base'].sum() / 1e6,
    df['ecl_lifetime_adverse'].sum() / 1e6,
    df['ecl_lifetime_upside'].sum() / 1e6,
    df['ecl_lifetime_weighted'].sum() / 1e6
]
colors = ['blue', 'red', 'green', 'purple']
ax.bar(scenarios, scenario_ecl, color=colors, alpha=0.7)
ax.set_ylabel('Total Lifetime ECL ($M)')
ax.set_title('Scenario Impact on Lifetime ECL')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Coverage ratio distribution
ax = axes[1, 1]
df['coverage_12m'] = (df['ecl_12m'] / df['amount']) * 100
df['coverage_lifetime'] = (df['ecl_lifetime_weighted'] / df['amount']) * 100

ax.hist(df['coverage_12m'], bins=30, alpha=0.5, label='12m ECL', color='green')
ax.hist(df['coverage_lifetime'], bins=30, alpha=0.5, label='Lifetime ECL', color='orange')
ax.set_xlabel('Coverage Ratio (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Coverage Ratios')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('expected_credit_loss_models.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Lifetime ECL typically 3-8× higher than 12-month ECL")
print("   → Stage 1→2 migration causes significant provision increase")
print("")
print("2. Longer maturity loans have disproportionately higher lifetime ECL")
print("   → Credit risk compounds over time; PD term structure critical")
print("")
print("3. Scenario weighting smooths ECL volatility")
print("   → Avoids overreaction to single economic view")
print("")
print("4. Adverse scenario ECL ~50-100% higher than base")
print("   → Stress testing reveals tail risk; capital adequacy check")

