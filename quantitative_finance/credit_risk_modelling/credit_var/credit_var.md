# Credit Value-at-Risk (Credit VaR)

## 1. Concept Skeleton
**Definition:** Maximum portfolio loss from credit events at specified confidence level over holding period; e.g., 99% Credit VaR = worst-case 1-year loss with 1% probability  
**Purpose:** Quantify tail credit risk, set capital buffers, compare portfolios on risk-adjusted basis  
**Prerequisites:** Value-at-risk framework, credit metrics (PD, LGD, EAD), portfolio statistics, loss distributions

## 2. Comparative Framing
| Risk Measure | Statistic | Time Horizon | Use Case | Advantage |
|--------------|-----------|-------------|----------|-----------|
| **Credit VaR** | Percentile loss | 1-year typical | Capital requirement | Intuitive threshold |
| **Expected Loss** | Mean loss | Annual | Provisioning/pricing | Conservative baseline |
| **Expected Shortfall** | Mean of tail | Beyond VaR | Regulatory (Basel III) | Captures tail severity |
| **Stress Testing** | Scenario loss | Variable | Extreme event | Model-independent |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio 99% Credit VaR = $50M over 1 year. In worst 1% of outcomes, lose ≥ $50M; expected loss only $5M

**Failure Case:**  
VaR ignores correlation breakdown during crisis. 2008: Portfolio VaR 99% = $100M assumed, actual loss $500M when correlations → 1

**Edge Case:**  
VaR of $0 when no defaults likely in 99% of paths. True for AAA portfolios; uninformative but technically correct

## 4. Layer Breakdown
```
Credit VaR Framework:
├─ Definition:
│   ├─ VaR(α) = Loss amount L such that P(Loss > L) = 1 - α
│   ├─ Example: VaR(99%) = 99th percentile of loss distribution
│   ├─ Holding period: Typically 1 year for credit risk
│   └─ Confidence level: 99% (regulatory), 95% (internal)
├─ Calculation Methods:
│   ├─ Parametric (delta-normal):
│   │   ├─ Assume normal loss distribution
│   │   ├─ VaR = μ + σ × Z_α
│   │   └─ Fast but may underestimate tails
│   ├─ Historical simulation:
│   │   ├─ Use empirical loss distribution
│   │   ├─ Sort historical losses, pick percentile
│   │   └─ No distribution assumption but limited history
│   ├─ Monte Carlo:
│   │   ├─ Simulate asset values, default scenarios
│   │   ├─ Calculate loss in each path
│   │   └─ Flexible but computationally intensive
│   └─ Intensity-based (reduced-form):
│       ├─ Model default as Poisson jump
│       ├─ Calibrate to historical/market data
│       └─ Hierarchical computation
├─ Loss Distribution Components:
│   ├─ Expected loss (EL): First moment, mean
│   ├─ Unexpected loss (UL): Volatility around mean
│   ├─ VaR: Combines EL + multiple of UL
│   ├─ Tail risk: Losses beyond VaR (Expected Shortfall)
│   └─ Skewness: Asymmetry (credit losses skewed left)
├─ Portfolio VaR:
│   ├─ Single-name VaR: Individual exposure
│   ├─ Diversification benefit: VaR_portfolio < Σ VaR_i
│   ├─ Correlation impact: Higher correlation → Higher portfolio VaR
│   └─ Concentration: Large exposures dominate VaR
├─ VaR Dynamics:
│   ├─ Time-varying: Increases in crisis (correlation spike)
│   ├─ Term structure: Multi-year VaR > 1-year
│   ├─ Regime-dependent: High vs low volatility states
│   └─ Liquidity impact: Illiquid portfolios have higher VaR
└─ Limitations:
    ├─ Tail risk: VaR ignores magnitude of losses > VaR
    ├─ Non-subadditivity: Diversifying may increase VaR
    ├─ Model risk: Sensitive to distributional assumptions
    └─ Fat tails: Actual losses exceed model VaR in crisis
```

## 5. Mini-Project
Calculate and analyze Credit VaR:
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Credit Value-at-Risk Analysis ===")

# Portfolio of loans
n_loans = 100
portfolio = pd.DataFrame({
    'Loan_ID': np.arange(n_loans),
    'Amount': np.random.lognormal(12, 1.5, n_loans),
    'Rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'], n_loans, 
                               p=[0.05, 0.10, 0.20, 0.35, 0.20, 0.10]),
})

# PD and LGD by rating
rating_params = {
    'AAA': {'PD': 0.0001, 'LGD': 0.30},
    'AA': {'PD': 0.0005, 'LGD': 0.32},
    'A': {'PD': 0.002, 'LGD': 0.35},
    'BBB': {'PD': 0.008, 'LGD': 0.40},
    'BB': {'PD': 0.030, 'LGD': 0.45},
    'B': {'PD': 0.100, 'LGD': 0.50}
}

portfolio['PD'] = portfolio['Rating'].map(lambda r: rating_params[r]['PD'])
portfolio['LGD'] = portfolio['Rating'].map(lambda r: rating_params[r]['LGD'])
portfolio['EL'] = portfolio['PD'] * portfolio['LGD'] * portfolio['Amount']

print(f"Portfolio size: {len(portfolio)} loans")
print(f"Total exposure: ${portfolio['Amount'].sum()/1e6:.1f}M")
print(f"Average PD: {portfolio['PD'].mean():.2%}")
print(f"Total Expected Loss: ${portfolio['EL'].sum()/1e3:.0f}K")

# Method 1: Parametric (Delta-Normal) VaR
print("\n=== Method 1: Parametric (Delta-Normal) VaR ===")

# Assume losses normally distributed around expected loss
total_el = portfolio['EL'].sum()
total_variance = (portfolio['PD'] * (1 - portfolio['PD']) * (portfolio['LGD'] * portfolio['Amount'])**2).sum()
total_std = np.sqrt(total_variance)

# Account for correlation
correlation_matrix = np.ones((len(portfolio), len(portfolio))) * 0.30  # Common correlation
np.fill_diagonal(correlation_matrix, 1.0)

# More accurate: Meucci approach
# Var[L] = Σᵢ Var[Lᵢ] + 2 × Σᵢ<ⱼ Cov[Lᵢ, Lⱼ]
variances = (portfolio['PD'] * (1 - portfolio['PD'])) * (portfolio['LGD'] * portfolio['Amount'])**2
pairwise_cov = 0
for i in range(len(portfolio)):
    for j in range(i+1, len(portfolio)):
        std_i = np.sqrt(variances.iloc[i])
        std_j = np.sqrt(variances.iloc[j])
        pairwise_cov += 2 * 0.30 * std_i * std_j

total_var_correlated = variances.sum() + pairwise_cov
total_std_correlated = np.sqrt(total_var_correlated)

# VaR at different confidence levels
confidence_levels = [0.95, 0.99, 0.999]
var_parametric = {}

for conf in confidence_levels:
    z_score = stats.norm.ppf(conf)
    var_amount = total_el + z_score * total_std_correlated
    var_parametric[conf] = var_amount

print("Parametric VaR (assuming normal distribution):")
print("Confidence | VaR Amount | UL = VaR - EL")
print("-" * 45)
for conf in confidence_levels:
    ul = var_parametric[conf] - total_el
    print(f"{conf*100:6.1f}%    | ${var_parametric[conf]/1e6:9.2f}M | ${ul/1e6:8.2f}M")

# Method 2: Monte Carlo Simulation
print("\n=== Method 2: Monte Carlo Credit VaR ===")

n_simulations = 10000
default_matrix = np.random.rand(n_simulations, len(portfolio))
losses_mc = np.zeros(n_simulations)

for sim in range(n_simulations):
    for i, loan in enumerate(portfolio.itertuples()):
        if default_matrix[sim, i] < loan.PD:
            # Default occurs
            losses_mc[sim] += loan.LGD * loan.Amount
        else:
            # Add small loss for mark-to-market (optional)
            pass

var_mc = {}
for conf in confidence_levels:
    var_mc[conf] = np.percentile(losses_mc, conf * 100)

print(f"Simulations: {n_simulations}")
print("Monte Carlo VaR:")
print("Confidence | VaR Amount | UL = VaR - EL")
print("-" * 45)
for conf in confidence_levels:
    ul = var_mc[conf] - total_el
    print(f"{conf*100:6.1f}%    | ${var_mc[conf]/1e6:9.2f}M | ${ul/1e6:8.2f}M")

# Method 3: Historical Simulation (using simulated history)
print("\n=== Method 3: Historical Simulation ===")

# Generate "historical" loss distribution
n_historical = 1000
losses_hist = np.zeros(n_historical)

for period in range(n_historical):
    period_default = np.random.rand(len(portfolio)) < portfolio['PD'].values
    losses_hist[period] = (period_default * portfolio['LGD'].values * portfolio['Amount'].values).sum()

var_hist = {}
for conf in confidence_levels:
    var_hist[conf] = np.percentile(losses_hist, conf * 100)

print("Historical Simulation VaR:")
print("Confidence | VaR Amount | UL = VaR - EL")
print("-" * 45)
for conf in confidence_levels:
    ul = var_hist[conf] - total_el
    print(f"{conf*100:6.1f}%    | ${var_hist[conf]/1e6:9.2f}M | ${ul/1e6:8.2f}M")

# Comparison and sensitivity
print("\n=== VaR Methodology Comparison ===")
print("Confidence | Parametric | Monte Carlo | Historical")
print("-" * 55)
for conf in confidence_levels:
    print(f"{conf*100:6.1f}%    | ${var_parametric[conf]/1e6:9.2f}M  | ${var_mc[conf]/1e6:10.2f}M | ${var_hist[conf]/1e6:10.2f}M")

# Expected Shortfall (ES) = Average of tail losses
print("\n=== Expected Shortfall (CVaR) ===")
print("Average loss in worst 1% scenarios:")
for conf in confidence_levels:
    threshold = np.percentile(losses_mc, conf * 100)
    tail_losses = losses_mc[losses_mc >= threshold]
    es = tail_losses.mean()
    print(f"{conf*100:6.1f}%: VaR = ${var_mc[conf]/1e6:.2f}M, ES = ${es/1e6:.2f}M (difference = ${(es-var_mc[conf])/1e6:.2f}M)")

# Concentration analysis: Which loans drive VaR?
print("\n=== Loan Concentration in VaR ===")
portfolio['Contribution_to_EL'] = portfolio['EL'] / portfolio['EL'].sum()
portfolio['Contribution_to_VaR'] = portfolio['Amount'] * portfolio['PD'] * portfolio['LGD'] / (portfolio['Amount'] * portfolio['PD'] * portfolio['LGD']).sum()

top_contributors = portfolio.nlargest(5, 'Amount')[['Loan_ID', 'Amount', 'Rating', 'Contribution_to_VaR']]
print("\nTop 5 loans by VaR contribution:")
print(top_contributors.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Loss distribution (Monte Carlo)
ax1 = axes[0, 0]
ax1.hist(losses_mc, bins=50, edgecolor='black', alpha=0.7)
for conf in [0.95, 0.99]:
    var_line = var_mc[conf]
    ax1.axvline(var_line, linestyle='--', linewidth=2, label=f'{conf*100:.0f}% VaR = ${var_line/1e6:.2f}M')
ax1.axvline(total_el, color='g', linestyle='--', linewidth=2, label=f'Expected Loss = ${total_el/1e6:.2f}M')
ax1.set_xlabel('Loss Amount ($M)')
ax1.set_ylabel('Frequency')
ax1.set_title('Monte Carlo Loss Distribution')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: VaR comparison across methods
ax2 = axes[0, 1]
x_pos = np.arange(len(confidence_levels))
width = 0.25
var_param_vals = [var_parametric[c]/1e6 for c in confidence_levels]
var_mc_vals = [var_mc[c]/1e6 for c in confidence_levels]
var_hist_vals = [var_hist[c]/1e6 for c in confidence_levels]

ax2.bar(x_pos - width, var_param_vals, width, label='Parametric', alpha=0.7, edgecolor='black')
ax2.bar(x_pos, var_mc_vals, width, label='Monte Carlo', alpha=0.7, edgecolor='black')
ax2.bar(x_pos + width, var_hist_vals, width, label='Historical', alpha=0.7, edgecolor='black')
ax2.set_ylabel('VaR ($M)')
ax2.set_title('VaR by Methodology')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{c*100:.0f}%' for c in confidence_levels])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Loss tail
ax3 = axes[0, 2]
sorted_losses = np.sort(losses_mc)
tail_start = int(len(sorted_losses) * 0.95)
ax3.plot(sorted_losses[tail_start:], linewidth=2)
ax3.axhline(var_mc[0.99], color='r', linestyle='--', linewidth=2, label='99% VaR')
ax3.axhline(var_mc[0.95], color='orange', linestyle='--', linewidth=2, label='95% VaR')
ax3.set_xlabel('Percentile (from 95th)')
ax3.set_ylabel('Loss Amount ($M)')
ax3.set_title('Tail Losses (Top 5%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: VaR vs EL decomposition
ax4 = axes[1, 0]
components = ['Expected\nLoss', '95% VaR', '99% VaR', '99.9% VaR']
values = [total_el/1e6, var_mc[0.95]/1e6, var_mc[0.99]/1e6, var_mc[0.999]/1e6]
ul_values = [0, values[1]-values[0], values[2]-values[0], values[3]-values[0]]

ax4.bar(components, ul_values, color=['green', 'yellow', 'orange', 'red'], alpha=0.7, edgecolor='black')
ax4.bar(components, [total_el/1e6, total_el/1e6, total_el/1e6, total_el/1e6], alpha=0.7, 
       color='steelblue', edgecolor='black')
ax4.set_ylabel('Amount ($M)')
ax4.set_title('VaR Decomposition: EL + UL')
for i, (comp, val, ul) in enumerate(zip(components, values, ul_values)):
    ax4.text(i, val + 0.1, f'${val:.2f}M', ha='center', va='bottom')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Rating distribution
ax5 = axes[1, 1]
rating_counts = portfolio['Rating'].value_counts().sort_index()
ax5.bar(rating_counts.index, rating_counts.values, edgecolor='black', alpha=0.7)
ax5.set_ylabel('Number of Loans')
ax5.set_title('Portfolio Composition by Rating')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Sensitivity: Correlation impact
ax6 = axes[1, 2]
correlations = np.linspace(0, 1, 20)
var_by_corr = []

for rho in correlations:
    total_var_temp = variances.sum()
    pairwise_cov_temp = 0
    for i in range(len(portfolio)):
        for j in range(i+1, len(portfolio)):
            std_i = np.sqrt(variances.iloc[i])
            std_j = np.sqrt(variances.iloc[j])
            pairwise_cov_temp += 2 * rho * std_i * std_j
    total_std_temp = np.sqrt(total_var_temp + pairwise_cov_temp)
    var_99 = total_el + stats.norm.ppf(0.99) * total_std_temp
    var_by_corr.append(var_99)

ax6.plot(correlations, np.array(var_by_corr)/1e6, linewidth=2)
ax6.fill_between(correlations, np.array(var_by_corr)/1e6, alpha=0.2)
ax6.set_xlabel('Default Correlation')
ax6.set_ylabel('99% Credit VaR ($M)')
ax6.set_title('VaR Sensitivity to Correlation\n(Higher correlation → Higher VaR)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Credit VaR Summary ===")
print(f"Portfolio 99% VaR (MC): ${var_mc[0.99]/1e6:.2f}M")
print(f"Capital buffer for tail risk: ${(var_mc[0.99] - total_el)/1e6:.2f}M")
print(f"Ratio VaR/EL: {var_mc[0.99]/total_el:.2f}x")
```

## 6. Challenge Round
When is Credit VaR problematic?
- **Model risk**: Normal distribution assumption fails; actual losses have fat tails
- **Correlation changes**: VaR assumes stable correlations; crises show jumps to 1.0
- **Tail events**: VaR ignores losses beyond the threshold (Expected Shortfall better)
- **Liquidity**: Illiquid portfolios harder to value; mark-to-market prices stale
- **Non-additivity**: Diversification can increase VaR (tail concentration risk)

## 7. Key References
- [Basel III Credit VaR Framework](https://www.bis.org/basel_framework/chapter/CRE/40.htm) - Regulatory credit VaR standards
- [Cornish-Fisher VaR](https://en.wikipedia.org/wiki/Value_at_risk) - Alternative to normal assumption using higher moments
- [Credit Risk Plus Model](https://www.investopedia.com/terms/c/credit-risk-plus.asp) - Credit metrics intensity-based approach

---
**Status:** Core portfolio risk metric for capital planning | **Complements:** Expected Loss, Correlation, Concentration Risk
