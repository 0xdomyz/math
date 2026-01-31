# Auto-extracted from markdown file
# Source: lifetime_vs_12month_ecl.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# Portfolio parameters
maturities = [1, 2, 3, 5, 7, 10, 15, 20, 30]  # Years
loan_amount = 1_000_000  # $1M per loan
annual_pd = 0.02  # 2% flat annual PD for simplicity
lgd = 0.40  # 40% LGD
eir = 0.05  # 5% effective interest rate

# Calculate 12-month ECL (constant across maturities)
ecl_12m = loan_amount * annual_pd * lgd
print("="*70)
print("12-Month vs Lifetime ECL: Maturity Sensitivity")
print("="*70)
print(f"Loan Amount: ${loan_amount:,}")
print(f"Annual PD: {annual_pd:.1%}")
print(f"LGD: {lgd:.1%}")
print(f"EIR (Discount Rate): {eir:.1%}")
print("")
print(f"12-Month ECL (Stage 1): ${ecl_12m:,.0f} (constant across maturities)")
print("")

# Calculate lifetime ECL for each maturity
results = []

for maturity in maturities:
    # Marginal PD each year (constant annual_pd for simplicity)
    pds = [annual_pd] * maturity
    
    # Calculate year-by-year ECL
    lifetime_ecl = 0
    survival_prob = 1.0
    
    for year in range(1, maturity + 1):
        pd_year = pds[year - 1]
        marginal_loss = loan_amount * pd_year * lgd * survival_prob
        discount_factor = 1 / (1 + eir) ** year
        ecl_year = marginal_loss * discount_factor
        lifetime_ecl += ecl_year
        survival_prob *= (1 - pd_year)
    
    # Cumulative PD over lifetime
    cumulative_pd = 1 - survival_prob
    
    # Store results
    results.append({
        'maturity': maturity,
        'ecl_12m': ecl_12m,
        'ecl_lifetime': lifetime_ecl,
        'ratio': lifetime_ecl / ecl_12m,
        'cumulative_pd': cumulative_pd,
        'coverage_ratio_12m': (ecl_12m / loan_amount) * 100,
        'coverage_ratio_lifetime': (lifetime_ecl / loan_amount) * 100
    })

df = pd.DataFrame(results)

# Display table
print("Lifetime ECL by Maturity:")
print("-"*70)
print(f"{'Maturity':<10} {'12m ECL':<15} {'Lifetime ECL':<15} {'Ratio':<10} {'Cumul PD':<12}")
print("-"*70)
for _, row in df.iterrows():
    print(f"{row['maturity']:<10} ${row['ecl_12m']:<14,.0f} ${row['ecl_lifetime']:<14,.0f} {row['ratio']:<10.1f}× {row['cumulative_pd']:<11.1%}")

print("")

# Key observations
print("Key Observations:")
print("-"*70)
short_term = df[df['maturity'] == 2].iloc[0]
medium_term = df[df['maturity'] == 5].iloc[0]
long_term = df[df['maturity'] == 30].iloc[0]

print(f"Short-term (2Y): Lifetime ECL {short_term['ratio']:.1f}× higher than 12m")
print(f"Medium-term (5Y): Lifetime ECL {medium_term['ratio']:.1f}× higher than 12m")
print(f"Long-term (30Y): Lifetime ECL {long_term['ratio']:.1f}× higher than 12m")
print("")
print(f"Stage 1→2 transfer impact (5Y loan): +${medium_term['ecl_lifetime'] - medium_term['ecl_12m']:,.0f} provision charge")

# Scenario: Portfolio of mixed maturities
print("\n" + "="*70)
print("Portfolio Example: Mixed Maturities")
print("="*70)

# Simulate portfolio with distribution across maturities
portfolio_distribution = {
    1: 50,   # 50 loans with 1-year maturity
    2: 100,
    3: 150,
    5: 200,
    7: 150,
    10: 100,
    15: 50,
    20: 30,
    30: 20
}

total_loans = sum(portfolio_distribution.values())
portfolio_ecl_12m = 0
portfolio_ecl_lifetime = 0

for maturity, count in portfolio_distribution.items():
    row = df[df['maturity'] == maturity].iloc[0]
    portfolio_ecl_12m += row['ecl_12m'] * count
    portfolio_ecl_lifetime += row['ecl_lifetime'] * count

print(f"Total Loans: {total_loans}")
print(f"Total Exposure: ${loan_amount * total_loans:,.0f}")
print("")
print(f"Portfolio 12-Month ECL (Stage 1): ${portfolio_ecl_12m:,.0f}")
print(f"Portfolio Lifetime ECL (Stage 2): ${portfolio_ecl_lifetime:,.0f}")
print(f"Average Ratio: {portfolio_ecl_lifetime / portfolio_ecl_12m:.1f}×")
print("")
print(f"If 20% of portfolio migrates Stage 1→2:")
print(f"  → Provision increase: ${0.20 * (portfolio_ecl_lifetime - portfolio_ecl_12m):,.0f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ECL by maturity
ax = axes[0, 0]
ax.plot(df['maturity'], df['ecl_12m'] / 1e3, 'g-', linewidth=2, marker='o', label='12-Month ECL')
ax.plot(df['maturity'], df['ecl_lifetime'] / 1e3, 'r-', linewidth=2, marker='s', label='Lifetime ECL')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('ECL ($1000s)')
ax.set_title('12-Month vs Lifetime ECL by Maturity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Ratio (Lifetime / 12m)
ax = axes[0, 1]
ax.plot(df['maturity'], df['ratio'], 'b-', linewidth=2, marker='o')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Lifetime ECL / 12-Month ECL (Ratio)')
ax.set_title('ECL Ratio: Maturity Sensitivity')
ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='1× (Equal)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Coverage ratios
ax = axes[1, 0]
width = 0.35
x = np.arange(len(df['maturity']))
bars1 = ax.bar(x - width/2, df['coverage_ratio_12m'], width, label='12m ECL', alpha=0.7, color='green')
bars2 = ax.bar(x + width/2, df['coverage_ratio_lifetime'], width, label='Lifetime ECL', alpha=0.7, color='red')

ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Coverage Ratio (%)')
ax.set_title('Coverage Ratio by Maturity')
ax.set_xticks(x)
ax.set_xticklabels(df['maturity'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Cumulative PD vs Maturity
ax = axes[1, 1]
ax.plot(df['maturity'], df['cumulative_pd'] * 100, 'purple', linewidth=2, marker='o')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Cumulative PD (%)')
ax.set_title('Cumulative Default Probability vs Maturity')
ax.grid(alpha=0.3)

# Add annotation for key points
for mat in [5, 10, 30]:
    row = df[df['maturity'] == mat].iloc[0]
    ax.annotate(f"{row['cumulative_pd']:.1%}", 
                xy=(mat, row['cumulative_pd'] * 100), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('lifetime_vs_12month_ecl.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Lifetime ECL increases with maturity (more time = more default risk)")
print("   → 5Y: 4× higher; 10Y: 7× higher; 30Y: 14× higher")
print("")
print("2. Discounting reduces long-dated ECL impact")
print("   → Without discounting, 30Y would be ~20× higher")
print("")
print("3. Stage 1→2 transfer causes provisioning cliff")
print("   → 5Y loan: +$25k; 10Y loan: +$50k; 30Y loan: +$100k")
print("")
print("4. Portfolio weighted average ratio depends on maturity mix")
print("   → Longer-dated portfolios (mortgages): Higher Stage 1→2 impact")

