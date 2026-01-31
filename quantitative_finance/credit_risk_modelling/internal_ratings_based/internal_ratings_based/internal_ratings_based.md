# Internal Ratings-Based Approach (IRB)

## 1. Concept Skeleton
**Definition:** Basel II/III regulatory framework allowing banks to use internal models for estimating PD, LGD, EAD to calculate credit risk capital requirements  
**Purpose:** Align regulatory capital with actual bank risk profiles, incentivize better risk management, reduce regulatory arbitrage  
**Prerequisites:** Credit risk fundamentals (PD/LGD/EAD), Basel regulations, logistic regression, validation techniques

## 2. Comparative Framing
| Approach | Standardized | Foundation IRB (F-IRB) | Advanced IRB (A-IRB) |
|----------|-------------|---------------------|-------------------|
| **PD Estimation** | Fixed supervisory buckets | Bank estimates PD | Bank estimates PD |
| **LGD Estimation** | Fixed supervisory (45%) | Fixed supervisory | Bank estimates LGD |
| **EAD Estimation** | Fixed supervisory | Fixed supervisory | Bank estimates EAD |
| **Capital Requirement** | Simple risk weights | RWA formula with bank PD | RWA formula with PD/LGD/EAD |
| **Regulatory Approval** | Not required | Moderate scrutiny | Extensive validation |

## 3. Examples + Counterexamples

**Simple Example:**  
Corporate loan €10M: Bank estimates PD=2%, supervisory LGD=45%, EAD=€10M → RWA=€3.8M, capital (8%)=€304k

**Failure Case:**  
Bank underestimates PD during boom (1% vs true 3%) → insufficient capital → Basel floor (72.5% of standardized RWA) triggers, negates IRB benefit

**Edge Case:**  
Low-default portfolios (sovereigns, banks): Few defaults observed → PD estimation unreliable → margin of conservatism (MoC) required, longer data windows (5-7 years)

## 4. Layer Breakdown
```
IRB Framework Structure:
├─ Eligibility & Qualification:
│   ├─ Minimum Requirements: Data history ≥5 years, internal use test, stress testing
│   ├─ Supervisory Approval: Central bank validation, ongoing monitoring
│   ├─ Asset Classes: Corporate, sovereign, bank, retail, equity
│   └─ F-IRB vs A-IRB: Sequential adoption, A-IRB requires additional validation
├─ Risk Parameter Estimation:
│   ├─ Probability of Default (PD):
│   │   ├─ Definition: 1-year default probability for non-defaulted obligor
│   │   ├─ Rating Models: Logistic regression, scorecards, expert judgment
│   │   ├─ Calibration: Long-run average default rate (through-the-cycle)
│   │   ├─ PD Floor: 0.03% minimum (3 basis points) to avoid zero PD
│   │   └─ Validation: Backtesting via binomial tests, traffic lights
│   ├─ Loss Given Default (LGD):
│   │   ├─ Definition: (EAD - Recoveries) / EAD, economic LGD (downturn conditions)
│   │   ├─ A-IRB Estimation: Regression on collateral value, seniority, workout time
│   │   ├─ Downturn LGD: Higher LGD during economic stress (asset value ↓, recovery time ↑)
│   │   ├─ LGD Floor: Supervisory floors vary by asset class (0%-10%)
│   │   └─ F-IRB Values: 45% unsecured, 0%-35% secured depending on collateral
│   ├─ Exposure at Default (EAD):
│   │   ├─ Definition: Outstanding balance + undrawn commitments × CCF
│   │   ├─ Credit Conversion Factor (CCF): % of undrawn converting to exposure
│   │   ├─ A-IRB Estimation: Historical drawdown analysis, behavioral models
│   │   ├─ Revolving Facilities: Higher CCF (75%-100%) due to correlation with default
│   │   └─ F-IRB Values: 75% for commitments, 100% for drawn amounts
│   └─ Maturity (M):
│       ├─ Effective Maturity: Weighted average time to cash flows
│       ├─ Maturity Adjustment: b(PD) = [0.11852-0.05478·ln(PD)]² (corporate exposure)
│       └─ Floor/Cap: 1 year ≤ M ≤ 5 years (retail exempt from maturity adjustment)
├─ Risk-Weighted Assets (RWA) Calculation:
│   ├─ Corporate/Sovereign/Bank Formula:
│   │   RWA = K × 12.5 × EAD × 1.06 (1.06 = scaling factor)
│   │   Capital Requirement K = [LGD × N(G(PD) + √ρ · G(0.999)) - PD × LGD] × (1+(M-2.5)b(PD))/(1-1.5b(PD))
│   │   where N = cumulative standard normal, G = inverse standard normal
│   │   ρ (correlation) = 0.12(1-e^(-50PD))/(1-e^(-50)) + 0.24[1-(1-e^(-50PD))/(1-e^(-50))]
│   ├─ Retail Formula (no maturity adjustment):
│   │   K = [LGD × N(G(PD) + √ρ · G(0.999)) - PD × LGD]
│   │   ρ varies: 0.03-0.16 depending on retail sub-class (mortgage, revolving, other)
│   ├─ Basel III Output Floor: RWA_IRB ≥ 72.5% × RWA_Standardized (from 2023)
│   └─ Expected Loss (EL): EL = PD × LGD × EAD (deducted from capital or provisions)
├─ Model Validation & Governance:
│   ├─ Backtesting: Compare predicted PD vs realized default rates (binomial test)
│   ├─ Benchmarking: Compare bank estimates to supervisory benchmarks, peers
│   ├─ Use Test: Internal models must drive business decisions (pricing, limits)
│   ├─ Independent Validation: Separate from model development, report to senior management
│   └─ Regulatory Review: Supervisory on-site inspections, model approval process
└─ Capital Impact vs Standardized:
    ├─ IRB Advantage: Lower RWA for low-risk exposures (AAA: 10% vs SA: 20%)
    ├─ IRB Penalty: Higher RWA for high-risk (B-: 350% vs SA: 150%)
    ├─ Output Floor Impact: Constrains IRB benefit, particularly for A-IRB banks
    └─ Cyclicality: Through-the-cycle PD smooths capital volatility vs point-in-time
```

**Interaction:** Rating models estimate PD → Combine with LGD/EAD → RWA formula → Capital requirement → Backtesting validates

## 5. Mini-Project
Implement IRB capital calculation with PD estimation and RWA formula:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# =====================================
# STEP 1: PD MODEL ESTIMATION
# =====================================
print("="*70)
print("IRB APPROACH: PD ESTIMATION AND CAPITAL CALCULATION")
print("="*70)

np.random.seed(42)
n = 5000

# Generate synthetic credit data (5 years history)
financial_ratio = np.random.normal(0, 1, n)  # Debt/EBITDA proxy
leverage = np.random.normal(0, 1, n)  # Leverage ratio
profitability = np.random.normal(0, 1, n)  # ROE proxy

# True default probability (logit model)
z = -3.5 + 0.8*financial_ratio + 0.6*leverage - 0.5*profitability
prob_default = 1 / (1 + np.exp(-z))
defaults = (np.random.uniform(0, 1, n) < prob_default).astype(int)

# Create dataset
data = pd.DataFrame({
    'FinancialRatio': financial_ratio,
    'Leverage': leverage,
    'Profitability': profitability,
    'Default': defaults
})

print(f"\nPortfolio Statistics:")
print(f"   Observations: {n}")
print(f"   Defaults: {defaults.sum()} ({defaults.mean():.2%})")

# Train PD model (logistic regression)
X = data[['FinancialRatio', 'Leverage', 'Profitability']]
y = data['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pd_model = LogisticRegression(random_state=42, max_iter=1000)
pd_model.fit(X_train, y_train)

# Predict PD
data['PD'] = pd_model.predict_proba(X)[:, 1]

# Apply PD floor (0.03% minimum per Basel)
data['PD_IRB'] = np.maximum(data['PD'], 0.0003)

print(f"\nPD Model Performance:")
print(f"   Training AUC: {roc_auc_score(y_train, pd_model.predict_proba(X_train)[:, 1]):.4f}")
print(f"   Test AUC: {roc_auc_score(y_test, pd_model.predict_proba(X_test)[:, 1]):.4f}")
print(f"   Mean PD: {data['PD_IRB'].mean():.4%}")
print(f"   Min PD (after floor): {data['PD_IRB'].min():.4%}")
print(f"   Max PD: {data['PD_IRB'].max():.4%}")

# =====================================
# STEP 2: IRB RWA CALCULATION
# =====================================
print("\n" + "="*70)
print("IRB RWA CALCULATION")
print("="*70)

def calculate_correlation(pd, asset_class='corporate'):
    """
    Calculate asset value correlation per Basel IRB formula.
    Corporate/Sovereign/Bank: ρ = 0.12(1-e^(-50PD))/(1-e^(-50)) + 0.24[1-(1-e^(-50PD))/(1-e^(-50))]
    """
    if asset_class == 'corporate':
        rho = 0.12 * (1 - np.exp(-50*pd)) / (1 - np.exp(-50)) + \
              0.24 * (1 - (1 - np.exp(-50*pd)) / (1 - np.exp(-50)))
    elif asset_class == 'retail_mortgage':
        rho = 0.15
    elif asset_class == 'retail_revolving':
        rho = 0.04
    else:  # retail_other
        rho = 0.03 * (1 - np.exp(-35*pd)) / (1 - np.exp(-35)) + \
              0.16 * (1 - (1 - np.exp(-35*pd)) / (1 - np.exp(-35)))
    return rho

def calculate_maturity_adjustment(pd, maturity=2.5):
    """
    Calculate maturity adjustment b(PD) per Basel formula.
    b(PD) = [0.11852 - 0.05478 · ln(PD)]²
    """
    b_pd = (0.11852 - 0.05478 * np.log(pd))**2
    ma = (1 + (maturity - 2.5) * b_pd) / (1 - 1.5 * b_pd)
    return ma

def calculate_irb_capital(pd, lgd, ead, maturity=2.5, asset_class='corporate'):
    """
    Calculate IRB capital requirement K per Basel formula.
    
    K = [LGD × N(G(PD) + √ρ · G(0.999)) - PD × LGD] × MA
    
    where:
    - N: cumulative standard normal
    - G: inverse standard normal (quantile function)
    - ρ: correlation
    - MA: maturity adjustment (1.0 for retail)
    """
    # Asset correlation
    rho = calculate_correlation(pd, asset_class)
    
    # Capital requirement formula
    g_pd = stats.norm.ppf(pd)  # G(PD)
    g_999 = stats.norm.ppf(0.999)  # G(0.999) = 3.090
    
    n_term = stats.norm.cdf(g_pd + np.sqrt(rho) * g_999)  # N(G(PD) + √ρ · G(0.999))
    
    # Capital (before maturity adjustment)
    k_base = lgd * n_term - pd * lgd
    
    # Apply maturity adjustment (not for retail)
    if asset_class == 'corporate':
        ma = calculate_maturity_adjustment(pd, maturity)
        k = k_base * ma
    else:
        k = k_base
    
    # RWA and capital requirement
    rwa = k * 12.5 * ead * 1.06  # 1.06 = scaling factor
    capital_req = rwa * 0.08  # 8% of RWA
    
    return {
        'K': k,
        'RWA': rwa,
        'Capital': capital_req,
        'Correlation': rho
    }

# Apply to portfolio
# Assume: LGD=45% (F-IRB unsecured), EAD=€100k per obligor, M=2.5 years
data['LGD'] = 0.45
data['EAD'] = 100000  # €100k
data['Maturity'] = 2.5

# Calculate IRB metrics for each obligor
irb_results = []
for idx, row in data.iterrows():
    result = calculate_irb_capital(
        pd=row['PD_IRB'],
        lgd=row['LGD'],
        ead=row['EAD'],
        maturity=row['Maturity'],
        asset_class='corporate'
    )
    irb_results.append(result)

irb_df = pd.DataFrame(irb_results)
data = pd.concat([data, irb_df], axis=1)

print(f"\nPortfolio IRB Capital:")
print(f"   Total EAD: €{data['EAD'].sum()/1e6:.1f}M")
print(f"   Total RWA: €{data['RWA'].sum()/1e6:.1f}M")
print(f"   Total Capital Required: €{data['Capital'].sum()/1e6:.2f}M")
print(f"   Average Risk Weight: {(data['RWA'].sum()/data['EAD'].sum()):.2%}")
print(f"   Average Capital K: {data['K'].mean():.4%}")

# =====================================
# STEP 3: STANDARDIZED APPROACH COMPARISON
# =====================================
print("\n" + "="*70)
print("COMPARISON: IRB vs STANDARDIZED APPROACH")
print("="*70)

# Standardized approach: Fixed risk weights based on external ratings
# Assume corporate exposures rated BBB (100% risk weight)
data['RWA_Standardized'] = data['EAD'] * 1.00  # 100% risk weight
data['Capital_Standardized'] = data['RWA_Standardized'] * 0.08

print(f"\nStandardized Approach:")
print(f"   Total RWA: €{data['RWA_Standardized'].sum()/1e6:.1f}M")
print(f"   Total Capital: €{data['Capital_Standardized'].sum()/1e6:.2f}M")
print(f"   Risk Weight: 100% (BBB rating)")

# Basel III Output Floor: IRB RWA must be ≥ 72.5% of Standardized RWA
output_floor = 0.725
data['RWA_Floor'] = data['RWA_Standardized'] * output_floor
data['RWA_Final'] = np.maximum(data['RWA'], data['RWA_Floor'])
data['Capital_Final'] = data['RWA_Final'] * 0.08

print(f"\nWith Basel III Output Floor (72.5%):")
print(f"   Floor RWA: €{data['RWA_Floor'].sum()/1e6:.1f}M")
print(f"   Final RWA (max of IRB and Floor): €{data['RWA_Final'].sum()/1e6:.1f}M")
print(f"   Final Capital: €{data['Capital_Final'].sum()/1e6:.2f}M")
print(f"   Floor Binding: {(data['RWA_Final'] > data['RWA']).mean():.1%} of exposures")

capital_savings = data['Capital_Standardized'].sum() - data['Capital_Final'].sum()
print(f"\n   Capital Savings vs Standardized: €{capital_savings/1e6:.2f}M ({capital_savings/data['Capital_Standardized'].sum():.1%})")

# =====================================
# STEP 4: PD BACKTESTING
# =====================================
print("\n" + "="*70)
print("PD MODEL BACKTESTING")
print("="*70)

# Group into PD buckets and calculate realized default rates
data['PD_Bucket'] = pd.cut(data['PD_IRB'], bins=[0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0],
                            labels=['<0.5%', '0.5-1%', '1-2%', '2-5%', '5-10%', '>10%'])

backtest_results = data.groupby('PD_Bucket').agg({
    'Default': ['count', 'sum', 'mean'],
    'PD_IRB': 'mean'
}).round(4)

backtest_results.columns = ['_'.join(col).strip() for col in backtest_results.columns.values]
backtest_results = backtest_results.rename(columns={
    'Default_count': 'N_Obligors',
    'Default_sum': 'Defaults',
    'Default_mean': 'Realized_DR',
    'PD_IRB_mean': 'Predicted_PD'
})

print("\nBacktesting by PD Bucket:")
print(backtest_results)

# Binomial test for calibration
# H₀: Realized default rate = Predicted PD
for bucket in backtest_results.index:
    n_obs = backtest_results.loc[bucket, 'N_Obligors']
    n_defaults = backtest_results.loc[bucket, 'Defaults']
    pred_pd = backtest_results.loc[bucket, 'Predicted_PD']
    
    # Binomial test: p-value for observed defaults given predicted PD
    p_value = stats.binom_test(n_defaults, n_obs, pred_pd, alternative='two-sided')
    backtest_results.loc[bucket, 'P_Value'] = p_value
    backtest_results.loc[bucket, 'Calibration'] = 'Pass' if p_value > 0.05 else 'Fail'

print("\nCalibration Test (Binomial):")
print(backtest_results[['Predicted_PD', 'Realized_DR', 'P_Value', 'Calibration']])

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: PD Distribution
axes[0, 0].hist(data['PD_IRB']*100, bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(data['PD_IRB'].mean()*100, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean PD = {data["PD_IRB"].mean():.2%}')
axes[0, 0].axvline(0.03, color='blue', linestyle='--', linewidth=2, label='PD Floor = 0.03%')
axes[0, 0].set_xlabel('PD (%)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Probability of Default (PD)')
axes[0, 0].legend()

# Plot 2: IRB Risk Weight vs PD
pd_range = np.linspace(0.001, 0.20, 100)
rw_range = []
for pd in pd_range:
    result = calculate_irb_capital(pd, lgd=0.45, ead=1, maturity=2.5)
    rw_range.append(result['RWA'])

axes[0, 1].plot(pd_range*100, np.array(rw_range)*100, linewidth=2, label='IRB Risk Weight')
axes[0, 1].axhline(100, color='red', linestyle='--', linewidth=2, label='Standardized (BBB = 100%)')
axes[0, 1].axhline(72.5, color='orange', linestyle='--', linewidth=2, label='Output Floor (72.5%)')
axes[0, 1].set_xlabel('PD (%)')
axes[0, 1].set_ylabel('Risk Weight (%)')
axes[0, 1].set_title('IRB Risk Weight vs PD (LGD=45%, M=2.5y)')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 20)
axes[0, 1].set_ylim(0, 300)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Capital Comparison
methods = ['Standardized', 'IRB', 'IRB + Floor']
capital_values = [
    data['Capital_Standardized'].sum()/1e6,
    data['Capital'].sum()/1e6,
    data['Capital_Final'].sum()/1e6
]

axes[1, 0].bar(methods, capital_values, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Capital Required (€M)')
axes[1, 0].set_title('Capital Requirement Comparison')
for i, v in enumerate(capital_values):
    axes[1, 0].text(i, v + 0.5, f'€{v:.1f}M', ha='center', fontweight='bold')

# Plot 4: Backtesting Calibration
backtest_plot = backtest_results[['Predicted_PD', 'Realized_DR']].reset_index(drop=True)
x_pos = np.arange(len(backtest_plot))
width = 0.35

axes[1, 1].bar(x_pos - width/2, backtest_plot['Predicted_PD']*100, width, 
               label='Predicted PD', alpha=0.8)
axes[1, 1].bar(x_pos + width/2, backtest_plot['Realized_DR']*100, width, 
               label='Realized Default Rate', alpha=0.8)
axes[1, 1].set_xlabel('PD Bucket')
axes[1, 1].set_ylabel('Default Rate (%)')
axes[1, 1].set_title('PD Calibration: Predicted vs Realized')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(backtest_results.index, rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("IRB approach implementation complete:")
print(f"• PD model AUC: {roc_auc_score(y_test, pd_model.predict_proba(X_test)[:, 1]):.3f} (discriminatory power)")
print(f"• IRB capital: €{data['Capital'].sum()/1e6:.1f}M vs Standardized: €{data['Capital_Standardized'].sum()/1e6:.1f}M")
print(f"• Output floor impact: {(data['RWA_Final'] > data['RWA']).mean():.0%} exposures constrained")
print(f"• Calibration: {(backtest_results['Calibration']=='Pass').sum()}/{len(backtest_results)} buckets pass binomial test")
print(f"• Capital savings: {capital_savings/data['Capital_Standardized'].sum():.1%} vs standardized approach")
```

## 6. Challenge Round
When does IRB provide less capital benefit than expected?
- **Output floor binding:** Low PD portfolio (high-quality corporates) where IRB RWA < 72.5% of standardized → floor constrains benefit
- **High PD obligors:** IRB risk weights exceed standardized (e.g., PD=10% → RWA≈350% vs standardized 150%)
- **Downturn LGD requirement:** A-IRB banks must use stressed LGD, increasing capital vs F-IRB fixed 45%
- **Model conservatism:** Margin of conservatism (MoC) in low-default portfolios inflates PD estimates
- **Procyclicality concern:** Through-the-cycle PD may be higher than point-in-time during booms, reducing IRB advantage

Regulatory challenges: Use test compliance (models drive decisions), ongoing validation costs, supervisory scrutiny, potential model rejection.

## 7. Key References
- [BIS Basel II Framework - IRB Approach](https://www.bis.org/publ/bcbs107.htm) - Original IRB methodology
- [BIS Basel III Reforms (2017) - Output Floor](https://www.bis.org/bcbs/publ/d424.htm) - 72.5% floor introduction
- [EBA Guidelines on PD/LGD Estimation (2017)](https://www.eba.europa.eu/regulation-and-policy/model-validation/guidelines-on-pd-lgd-estimation-and-treatment-of-defaulted-assets) - EU IRB standards
- [Gordy (2003) "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules"](https://www.federalreserve.gov/pubs/feds/2003/200347/200347pap.pdf) - IRB formula derivation

---
**Status:** Core Basel regulatory framework | **Complements:** Credit Risk (PD/LGD/EAD), Model Validation, Basel Accords
