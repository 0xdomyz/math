# Auto-extracted from markdown file
# Source: capital_requirements_standardized_approach.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Bank corporate loan portfolio
portfolio = {
    'AAA': {'exposure': 50, 'rating': 'AAA'},
    'AA': {'exposure': 100, 'rating': 'AA'},
    'A': {'exposure': 150, 'rating': 'A'},
    'BBB': {'exposure': 300, 'rating': 'BBB'},
    'BB': {'exposure': 200, 'rating': 'BB'},
    'B': {'exposure': 100, 'rating': 'B'},
    'CCC': {'exposure': 50, 'rating': 'CCC'},
}

# 1. STANDARDIZED APPROACH (External ratings)
print("="*100)
print("STANDARDIZED APPROACH (SA) - Risk Weighted Assets Calculation")
print("="*100)

# Risk weights by rating
sa_rw = {
    'AAA': 0.20, 'AA': 0.20, 'A': 0.50,
    'BBB': 1.00, 'BB': 1.00, 'B': 1.50, 'CCC': 1.50,
}

sa_results = {}
total_sa_rwa = 0

for loan_type, data in portfolio.items():
    exposure = data['exposure']
    rating = data['rating']
    rw = sa_rw[rating]
    rwa = exposure * rw
    total_sa_rwa += rwa
    sa_results[loan_type] = {'exposure': exposure, 'rw': rw, 'rwa': rwa}
    print(f"{loan_type} {rating:>5}: ${exposure:>4.0f}M exposure, RW={rw*100:>5.0f}%, RWA=${rwa:>6.1f}M")

sa_capital = 0.08 * total_sa_rwa
print(f"\nTotal SA RWA: ${total_sa_rwa:.1f}M")
print(f"SA Capital (8%): ${sa_capital:.1f}M")

# 2. FOUNDATION IRB APPROACH (Bank estimates PD, regulator provides LGD/EAD)
print(f"\n" + "="*100)
print("FOUNDATION IRB - Bank-Estimated PD, Regulatory LGD/EAD")
print("="*100)

# Historical PD by rating (bank's data)
irb_pd = {
    'AAA': 0.02, 'AA': 0.05, 'A': 0.10,
    'BBB': 0.50, 'BB': 2.00, 'B': 5.00, 'CCC': 15.00,
}

# Regulatory LGD & EAD (foundation fixed)
irb_lgd = 0.45  # Regulatory floor for unsecured corporate
irb_ead = 1.00  # Regulatory EAD (100% outstanding)

# IRB formula parameters
maturity_adj = 1.0  # Assume 3-year loans → maturity adjustment ≈ 1.0
correlation = 0.12  # Corporate correlation

def calculate_irb_rw(pd, lgd, ead, correlation=0.12, maturity=1.0):
    """Calculate IRB risk weight using Basel formula."""
    # N(x) = cumulative normal
    # N^{-1}(x) = inverse normal
    
    pd_norm = norm.ppf(pd / 100)  # Convert PD % to decimal, then inverse normal
    
    # Correlation-adjusted maturity factor
    b = (0.11852 - 0.05478 * np.log(pd / 100)) ** 2
    maturity_factor = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    
    # N(.) calculation
    numerator = np.log(correlation / (1 - correlation)) + np.sqrt(correlation / (1 - correlation)) * norm.ppf(pd / 100)
    tail_quantile = norm.ppf(0.999)  # 99.9% confidence
    
    n_val = norm.cdf((numerator + np.sqrt(1 - correlation) * tail_quantile) / 
                     np.sqrt(correlation))
    
    # RW formula
    rw = 1.06 * 12.5 * (pd / 100 * lgd / 100 + np.sqrt(correlation / (1 - correlation)) * 
                        np.sqrt(1 - correlation) * (norm.ppf(pd / 100) - np.sqrt(correlation) * tail_quantile)) * maturity_factor
    
    # Simplified approximation
    rw_approx = 1.06 * 12.5 * (pd / 100 * lgd / 100)
    
    return max(min(rw_approx, 3.25), 0.001)  # Cap at 325%, floor at 0.1%

irb_results = {}
total_irb_rwa = 0

print(f"Using: LGD = {irb_lgd*100:.0f}%, EAD = {irb_ead*100:.0f}%, Correlation = {correlation*100:.0f}%\n")

for loan_type, data in portfolio.items():
    exposure = data['exposure']
    pd = irb_pd[loan_type]
    rw = calculate_irb_rw(pd, irb_lgd, irb_ead, correlation)
    rwa = exposure * rw
    total_irb_rwa += rwa
    irb_results[loan_type] = {'exposure': exposure, 'pd': pd, 'rw': rw, 'rwa': rwa}
    print(f"{loan_type} {data['rating']:>5}: PD={pd:>6.2f}%, RW={rw*100:>5.1f}%, RWA=${rwa:>6.1f}M")

irb_capital = 0.08 * total_irb_rwa
print(f"\nTotal IRB RWA: ${total_irb_rwa:.1f}M")
print(f"IRB Capital (8%): ${irb_capital:.1f}M")

# 3. ADVANCED IRB (Bank estimates PD, LGD, EAD - lower LGD for collateral)
print(f"\n" + "="*100)
print("ADVANCED IRB - Bank-Estimated PD, LGD, EAD (with Collateral)")
print("="*100)

# Bank estimates with collateral/loan structure
advanced_lgd = {
    'AAA': 0.20, 'AA': 0.25, 'A': 0.30,
    'BBB': 0.40, 'BB': 0.50, 'B': 0.60, 'CCC': 0.70,
}

# EAD accounts for undrawn commitments (30% drawdown probability)
advanced_ead_multiple = {
    'AAA': 0.90, 'AA': 0.92, 'A': 0.95,
    'BBB': 1.00, 'BB': 1.00, 'B': 1.00, 'CCC': 1.00,
}

adv_results = {}
total_adv_rwa = 0

print(f"Using: Collateral-adjusted LGD, Bank EAD estimates\n")

for loan_type, data in portfolio.items():
    exposure = data['exposure']
    rating = data['rating']
    pd = irb_pd[loan_type]
    lgd = advanced_lgd[rating]
    ead = advanced_ead_multiple[rating]
    rw = calculate_irb_rw(pd, lgd, ead, correlation)
    rwa = exposure * rw * ead
    total_adv_rwa += rwa
    adv_results[loan_type] = {'exposure': exposure, 'pd': pd, 'lgd': lgd, 'ead': ead, 'rw': rw, 'rwa': rwa}
    print(f"{loan_type} {rating:>5}: PD={pd:>6.2f}%, LGD={lgd*100:>5.0f}%, EAD={ead*100:>5.0f}%, RWA=${rwa:>6.1f}M")

adv_capital = 0.08 * total_adv_rwa
print(f"\nTotal Advanced IRB RWA: ${total_adv_rwa:.1f}M")
print(f"Advanced IRB Capital (8%): ${adv_capital:.1f}M")

# 4. OUTPUT FLOOR (72.5% of SA RWA)
print(f"\n" + "="*100)
print("OUTPUT FLOOR (72.5% constraint)")
print("="*100)

floor_value = 0.725 * total_sa_rwa
print(f"SA RWA: ${total_sa_rwa:.1f}M")
print(f"72.5% Floor: ${floor_value:.1f}M")
print(f"Advanced IRB RWA: ${total_adv_rwa:.1f}M")

if total_adv_rwa < floor_value:
    floored_rwa = floor_value
    print(f"Advanced IRB FLOORED to: ${floored_rwa:.1f}M (binding floor)")
else:
    floored_rwa = total_adv_rwa
    print(f"Advanced IRB above floor (no constraint)")

floored_capital = 0.08 * floored_rwa

# COMPARISON TABLE
print(f"\n" + "="*100)
print("CAPITAL REQUIREMENT COMPARISON")
print("="*100)

comparison = pd.DataFrame({
    'Approach': ['Standardized (SA)', 'Foundation IRB (F-IRB)', 'Advanced IRB (A-IRB)', 'Advanced IRB (w/ Floor)'],
    'Total RWA': [f'${total_sa_rwa:.1f}M', f'${total_irb_rwa:.1f}M', f'${total_adv_rwa:.1f}M', f'${floored_rwa:.1f}M'],
    'Capital (8%)': [f'${sa_capital:.1f}M', f'${irb_capital:.1f}M', f'${adv_capital:.1f}M', f'${floored_capital:.1f}M'],
    'Capital vs SA': ['Baseline', f'{(irb_capital/sa_capital - 1)*100:+.1f}%', 
                      f'{(adv_capital/sa_capital - 1)*100:+.1f}%',
                      f'{(floored_capital/sa_capital - 1)*100:+.1f}%'],
})

print(comparison.to_string(index=False))

rwa_reduction = (1 - total_adv_rwa / total_sa_rwa) * 100
print(f"\nAdvanced IRB RWA reduction vs SA: {rwa_reduction:.1f}%")
print(f"Output floor prevents RWA reduction >27.5% (current: {rwa_reduction:.1f}%)")

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RWA by rating and approach
ax = axes[0, 0]
ratings = list(portfolio.keys())
sa_rwas = [sa_results[r]['rwa'] for r in ratings]
irb_rwas = [irb_results[r]['rwa'] for r in ratings]
adv_rwas = [adv_results[r]['rwa'] for r in ratings]

x = np.arange(len(ratings))
width = 0.25

ax.bar(x - width, sa_rwas, width, label='Standardized', alpha=0.8)
ax.bar(x, irb_rwas, width, label='Foundation IRB', alpha=0.8)
ax.bar(x + width, adv_rwas, width, label='Advanced IRB', alpha=0.8)

ax.set_ylabel('Risk-Weighted Assets ($M)')
ax.set_title('RWA by Rating and Approach')
ax.set_xticks(x)
ax.set_xticklabels(ratings)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Total capital requirements
ax = axes[0, 1]
approaches = ['SA', 'F-IRB', 'A-IRB', 'A-IRB\n(Floored)']
capitals = [sa_capital, irb_capital, adv_capital, floored_capital]
colors_cap = ['blue', 'orange', 'green', 'red']

bars = ax.bar(approaches, capitals, color=colors_cap, alpha=0.7)
ax.set_ylabel('Capital Required ($M)')
ax.set_title('Total Capital Requirement Comparison')
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar, cap in zip(bars, capitals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${cap:.1f}M',
            ha='center', va='bottom', fontweight='bold')

# Plot 3: RWA breakdown by rating
ax = axes[1, 0]
ax.barh(ratings, [adv_results[r]['rwa'] for r in ratings], color='teal', alpha=0.7)
ax.set_xlabel('RWA ($M)')
ax.set_title('Advanced IRB RWA Breakdown by Rating')
ax.grid(alpha=0.3, axis='x')

# Plot 4: Risk weight by rating
ax = axes[1, 1]
sa_rws = [sa_results[r]['rw']*100 for r in ratings]
irb_rws = [irb_results[r]['rw']*100 for r in ratings]
adv_rws = [adv_results[r]['rw']*100 for r in ratings]

x = np.arange(len(ratings))
ax.plot(x, sa_rws, 'o-', label='Standardized', linewidth=2, markersize=6)
ax.plot(x, irb_rws, 's-', label='Foundation IRB', linewidth=2, markersize=6)
ax.plot(x, adv_rws, '^-', label='Advanced IRB', linewidth=2, markersize=6)

ax.set_ylabel('Risk Weight (%)')
ax.set_title('Risk Weight Curves by Rating and Approach')
ax.set_xticks(x)
ax.set_xticklabels(ratings)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
plt.show()

# Gaming risk illustration
print(f"\n" + "="*100)
print("MODEL GAMING ILLUSTRATION")
print("="*100)
print(f"\nWhat if Advanced IRB underestimates LGD by 50%?")
gamed_lgd = {k: v * 0.5 for k, v in advanced_lgd.items()}
gamed_rwa = 0
for loan_type, data in portfolio.items():
    exposure = data['exposure']
    rating = data['rating']
    pd = irb_pd[loan_type]
    lgd = gamed_lgd[rating]
    ead = advanced_ead_multiple[rating]
    rw = calculate_irb_rw(pd, lgd, ead, correlation)
    rwa = exposure * rw * ead
    gamed_rwa += rwa
gamed_capital = 0.08 * gamed_rwa
gamed_reduction = (1 - gamed_rwa / total_sa_rwa) * 100

print(f"Gamed A-IRB RWA: ${gamed_rwa:.1f}M (reduction: {gamed_reduction:.1f}%)")
print(f"With 72.5% floor, gamed RWA capped at: ${floor_value:.1f}M")
print(f"Floor prevents capital arbitrage: ${gamed_capital:.1f}M → ${floored_capital:.1f}M (${floored_capital - gamed_capital:.1f}M difference)")

