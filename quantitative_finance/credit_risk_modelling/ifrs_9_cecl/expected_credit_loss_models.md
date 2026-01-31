# Expected Credit Loss (ECL) Models

## 1. Concept Skeleton
**Definition:** Quantitative frameworks estimating probability-weighted present value of credit losses over instrument lifetime or 12 months; integrate PD (probability of default), LGD (loss given default), EAD (exposure at default), and macroeconomic scenarios  
**Purpose:** Calculate IFRS 9 / CECL provisions; forward-looking loss estimation; scenario-weighted risk quantification; regulatory capital calculation (aligns with Basel IRB)  
**Prerequisites:** Credit risk components (PD, LGD, EAD), survival analysis, logistic regression, macroeconomic scenario analysis, discounting, segmentation

## 2. Comparative Framing
| Model Type | Complexity | Data Requirements | Forward-Looking | Granularity | Use Case |
|------------|------------|-------------------|-----------------|-------------|----------|
| **Historical Loss Rate** | Low | Historical defaults | No (backward) | Portfolio-level | Simple portfolios; limited data |
| **Roll Rate (Migration)** | Medium | Payment status transitions | Implicit | Vintage/delinquency bucket | Consumer credit; arrears-based |
| **PD-LGD-EAD (IRB)** | High | Individual exposures; defaults | Yes (scenarios) | Loan-level | Corporate; Basel IRB; IFRS 9 |
| **Discounted Cash Flow** | Very High | Payment schedules; recovery timing | Yes | Instrument-level | Complex structures; Stage 3 |
| **Machine Learning** | Very High | Rich features (100+ variables) | Yes (if macro features) | Loan-level | Large datasets; non-linear patterns |

## 3. Examples + Counterexamples

**Simple Example:**  
Stage 1 loan: $100k, 12-month PD = 1%, LGD = 40%, EAD = $100k. ECL = $100k × 1% × 40% = $400. Stage 2 (lifetime): 5-year maturity, lifetime PD = 8%, ECL = $100k × 8% × 40% = $3,200.

**Scenario-Weighted ECL:**  
Base scenario (50% weight): PD = 2%, ECL = $800. Downturn (30% weight): PD = 5%, ECL = $2,000. Upturn (20% weight): PD = 1%, ECL = $400. Weighted ECL = 0.5×$800 + 0.3×$2,000 + 0.2×$400 = $1,080.

**Discounted Cash Flow (Stage 3):**  
Defaulted loan $100k, expected recovery $40k in 2 years. Discount rate (EIR) = 5%. ECL = $100k - $40k/(1.05)² = $100k - $36.3k = $63.7k.

**Roll Rate Model:**  
Consumer portfolio: 1,000 loans current (bucket 0), 50 transition to 30 DPD (bucket 1), 10 from bucket 1 to 60 DPD (bucket 2), 5 from bucket 2 to default. Roll rates: 5% (0→1), 20% (1→2), 50% (2→default). ECL = sum over buckets weighted by exposure.

**Failure Case (No Scenarios):**  
Bank uses point estimate PD (mean) without scenario weighting. Economic downturn hits: Actual losses 3× higher than ECL. Insufficient provisioning; regulatory criticism.

## 4. Layer Breakdown
```
Expected Credit Loss Model Framework:

├─ ECL Formula (General):
│   ECL = EAD × PD × LGD × Discount Factor
│   ├─ EAD: Exposure at Default (outstanding + undrawn commitments × CCF)
│   ├─ PD: Probability of Default (12-month or lifetime; scenario-adjusted)
│   ├─ LGD: Loss Given Default (1 - recovery rate; collateral-adjusted)
│   └─ Discount Factor: Present value (discounted at effective interest rate, EIR)
│
├─ 12-Month ECL (Stage 1):
│   ├─ Horizon: Next 12 months only
│   ├─ PD: 12-month probability of default
│   │   ├─ Point-in-Time (PIT): Current economic conditions
│   │   ├─ Through-the-Cycle (TTC): Average over cycle (Basel IRB)
│   │   └─ IFRS 9 requires PIT (forward-looking scenarios)
│   ├─ Formula: ECL = EAD × PD(12m) × LGD
│   ├─ Example: Loan $1M, PD(12m) = 0.5%, LGD = 45%
│   │   ECL = $1M × 0.5% × 45% = $2,250
│   └─ No discounting (materiality; 12m horizon short)
│
├─ Lifetime ECL (Stage 2 & 3):
│   ├─ Horizon: Remaining contractual life of instrument
│   ├─ PD: Cumulative probability of default over lifetime
│   │   ├─ Marginal PD(t): Default probability in period t (conditional on survival to t)
│   │   ├─ Survival probability: S(t) = ∏(1 - PD(τ)) for τ = 1..t-1
│   │   ├─ Cumulative PD: CPD = 1 - S(T) = 1 - ∏(1 - PD(t))
│   │   └─ Forward-looking: PD(t) varies by macroeconomic scenario path
│   ├─ Integration Over Time:
│   │   ECL = ∑[t=1 to T] { EAD(t) × PD(t) × LGD(t) × DF(t) }
│   │   ├─ EAD(t): Exposure at time t (amortization, prepayments, drawdowns)
│   │   ├─ PD(t): Marginal default probability at time t
│   │   ├─ LGD(t): Loss given default (may vary with economic scenario)
│   │   └─ DF(t): Discount factor = 1 / (1 + EIR)^t
│   ├─ Example: 5-year loan $1M, annual PD = [1%, 1.5%, 2%, 2.5%, 3%]
│   │   Survival: S(1) = 99%, S(2) = 97.5%, ..., S(5) = 91.8%
│   │   ECL ≈ $1M × 8.2% × 45% × avg DF ≈ $35,000 (lifetime)
│   └─ Stage 3: PD(default) = 100%; focus shifts to LGD estimation (recovery rate)
│
├─ PD Estimation Methods:
│   ├─ Credit Scoring Models:
│   │   ├─ Logistic Regression: log(PD / (1-PD)) = β₀ + β₁X₁ + ... + βₙXₙ
│   │   │   ├─ Features: Debt-to-income, credit score, LTV, payment history
│   │   │   └─ Output: Point-in-time PD (12-month or 1-year)
│   │   ├─ Calibration: Map scores to PD using historical default rates
│   │   └─ Segmentation: Separate models by product (mortgage, corporate, credit card)
│   ├─ Transition Matrices (Rating Migration):
│   │   ├─ Markov chain: P(Rating_t+1 | Rating_t)
│   │   ├─ Transition probabilities: AAA→AA, AA→A, ..., CCC→Default
│   │   ├─ Lifetime PD: Compound transitions over T periods
│   │   └─ Scenario adjustment: Stress rating migration matrix (downturn = higher default rates)
│   ├─ Survival Analysis (Hazard Models):
│   │   ├─ Hazard rate λ(t): Instantaneous default rate at time t
│   │   ├─ Survival function: S(t) = exp(-∫λ(τ)dτ)
│   │   ├─ PD(t) = λ(t) × S(t-1)
│   │   └─ Cox Proportional Hazards: λ(t) = λ₀(t) × exp(βX)
│   ├─ Structural Models (Merton):
│   │   ├─ Firm value follows geometric Brownian motion
│   │   ├─ Default when firm value < debt threshold
│   │   ├─ PD = N(-d₂) where d₂ = (ln(V/K) + (μ - σ²/2)T) / (σ√T)
│   │   └─ Calibrated to equity volatility, leverage
│   └─ Machine Learning (XGBoost, Neural Networks):
│       ├─ Non-linear relationships; interaction effects
│       ├─ Features: 100+ variables (payment history, macro, behavioral)
│       ├─ Calibration: Ensure monotonicity; align with historical default rates
│       └─ Challenge: Interpretability; regulatory acceptance
│
├─ LGD Estimation Methods:
│   ├─ Historical Recovery Rates:
│   │   ├─ LGD = 1 - (Recovery / EAD)
│   │   ├─ Segmentation: Secured vs unsecured; collateral type; seniority
│   │   ├─ Example: Secured mortgage LGD ≈ 20-30%; unsecured credit card LGD ≈ 70-90%
│   │   └─ Downturn LGD: Adjust for economic stress (lower collateral values)
│   ├─ Collateral Haircuts:
│   │   ├─ Market value of collateral × (1 - haircut) - costs to sell
│   │   ├─ Real estate: 20-30% haircut; equipment: 40-60%; intangibles: 80%+
│   │   └─ Stressed scenario: Higher haircuts (illiquid markets)
│   ├─ Discounted Cash Flow (Workout LGD):
│   │   ├─ Estimate recovery timing: Foreclosure 2-3 years; bankruptcy 1-5 years
│   │   ├─ Discount recoveries at EIR
│   │   └─ Example: Recovery $40k in 2 years, EIR 5% → PV = $36.3k
│   └─ Regression Models:
│       ├─ LGD = f(collateral value, seniority, industry, macro conditions)
│       └─ Calibrate to historical workout data
│
├─ EAD Estimation:
│   ├─ On-Balance Sheet (Term Loans):
│   │   ├─ EAD = Outstanding principal + accrued interest
│   │   └─ Amortization: EAD declines over time (scheduled repayments)
│   ├─ Off-Balance Sheet (Commitments):
│   │   ├─ Credit Conversion Factor (CCF): Proportion drawn at default
│   │   ├─ EAD = Drawn + Undrawn × CCF
│   │   ├─ Example: $50k drawn, $50k undrawn, CCF = 50% → EAD = $50k + $25k = $75k
│   │   └─ Stressed CCF: Higher drawdowns during crisis (liquidity stress)
│   └─ Derivatives (CVA):
│       ├─ Expected Exposure (EE): Forward simulation of MTM
│       └─ EAD = α × EE(t) (regulatory factor α)
│
├─ Forward-Looking Scenarios:
│   ├─ Scenario Design:
│   │   ├─ Base (50-60% weight): Most likely economic path (consensus forecast)
│   │   ├─ Adverse/Downturn (20-30%): Recession scenario (GDP -2%, unemployment +3%)
│   │   ├─ Severe Adverse (5-10%): Tail risk (financial crisis; GDP -5%)
│   │   └─ Upside (10-20%): Benign conditions (GDP +4%, low unemployment)
│   ├─ Scenario Variables:
│   │   ├─ Macroeconomic: GDP growth, unemployment, inflation, interest rates
│   │   ├─ Market: Equity indices, commodity prices, FX rates
│   │   └─ Sector-specific: Oil prices (energy loans), house prices (mortgages)
│   ├─ PD/LGD Sensitivity to Scenarios:
│   │   ├─ Recession → Higher PD (credit deterioration), Higher LGD (lower collateral values)
│   │   ├─ Boom → Lower PD (stronger borrower finances), Lower LGD (asset price appreciation)
│   │   └─ Econometric models: PD = f(GDP, unemployment); LGD = f(house prices)
│   └─ Probability Weighting:
│       ├─ ECL = ∑[s] w(s) × ECL(s)
│       ├─ Weights sum to 1; based on expert judgment or scenario probability
│       └─ IFRS 9 requires unbiased (not excessively prudent; not optimistic)
│
├─ Discounting:
│   ├─ Effective Interest Rate (EIR):
│   │   ├─ Discount rate that equates present value of cash flows to amortized cost
│   │   ├─ Includes origination fees, transaction costs (not market risk premium)
│   │   └─ Typical: 4-8% for corporate loans; 10-20% for credit cards
│   ├─ Stage 1 & 2: Discount at EIR (original effective rate)
│   ├─ Stage 3: Discount at EIR or credit-adjusted rate (debate; IFRS 9 allows both)
│   └─ Material Impact: Long maturity (10+ years) → Discounting reduces ECL by 20-40%
│
├─ Segmentation:
│   ├─ Why Segment:
│   │   ├─ Homogeneous risk within segment (similar PD/LGD drivers)
│   │   ├─ Reduces model complexity; improves calibration
│   │   └─ Aligns with business practices (product types)
│   ├─ Segmentation Dimensions:
│   │   ├─ Product: Mortgage, corporate term loan, revolving credit, credit card
│   │   ├─ Geography: Country, region (different economic conditions)
│   │   ├─ Industry: Energy, real estate, retail (sector risk)
│   │   ├─ Collateral: Secured vs unsecured; asset-backed
│   │   └─ Vintage: Origination year (cohort analysis)
│   └─ Example: Mortgage ECL model separate from corporate loan model (different PD/LGD drivers)
│
├─ Model Calibration & Validation:
│   ├─ Calibration:
│   │   ├─ Align modeled PD to historical default rates (by segment)
│   │   ├─ Central Tendency: Ensure long-run average PD = observed default rate
│   │   ├─ Adjust for economic cycle (TTC → PIT conversion)
│   │   └─ LGD: Map to historical recovery rates; adjust for downturn
│   ├─ Backtesting:
│   │   ├─ Out-of-sample validation: Test PD model on holdout data
│   │   ├─ AUC (Area Under Curve): Discriminatory power (>0.7 acceptable; >0.8 good)
│   │   ├─ Gini coefficient: 2 × AUC - 1 (alternative metric)
│   │   └─ Calibration plots: Predicted PD vs observed default rate by decile
│   ├─ Stress Testing:
│   │   ├─ Apply adverse scenarios; compare ECL to actual losses in crisis
│   │   └─ Regulatory stress tests (CCAR, EBA): Validate scenario sensitivity
│   └─ Model Risk:
│       ├─ Parameter uncertainty: PD/LGD estimates have confidence intervals
│       ├─ Model misspecification: Logistic regression may miss non-linearities
│       └─ Management overlay: Expert adjustments for model limitations (e.g., COVID-19 shock)
│
└─ Practical Implementation:
    ├─ Systems Architecture:
    │   ├─ Data warehouse: Loan-level exposures, payment history, collateral values
    │   ├─ ECL engine: Calculate 12m and lifetime ECL by instrument × scenario
    │   ├─ Scenario platform: Generate macro paths; map to PD/LGD
    │   └─ Reporting: IFRS 9 disclosures; regulatory capital (Basel IRB)
    ├─ Computational Efficiency:
    │   ├─ Monte Carlo simulation: For derivatives EAD (computationally expensive)
    │   ├─ Closed-form approximations: For large portfolios (analytical ECL formula)
    │   └─ Parallel processing: Distribute calculations across computing cluster
    ├─ Governance:
    │   ├─ Model documentation: Assumptions, data sources, validation results
    │   ├─ Model approval: Risk committee, Board oversight
    │   ├─ Annual review: Update PD/LGD models; recalibrate to new data
    │   └─ Audit trail: All ECL calculations; scenario assumptions logged
    └─ Regulatory Alignment:
        ├─ IFRS 9: Forward-looking ECL; probability-weighted scenarios
        ├─ CECL (US GAAP): Similar to IFRS 9 lifetime ECL (all exposures)
        └─ Basel IRB: PD/LGD/EAD models aligned with IFRS 9 (efficiency gain)
```

**Key Insight:** ECL = EAD × PD × LGD × DF; 12-month ECL for Stage 1 (low risk); lifetime ECL for Stage 2/3 (elevated/impaired); scenario-weighted forward-looking estimates; calibrated to historical data; validated via backtesting.

## 5. Mini-Project
ECL calculation comparing 12-month vs lifetime for loan portfolio:
```python
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
```

## 6. Challenge Round
When ECL models fail or introduce complexity:
- **Data Scarcity (Low Default Portfolios)**: Investment-grade corporate portfolio; 0.1% annual default rate → Insufficient defaults to calibrate PD; solution: External data (rating agency default rates); peer benchmarks; Bayesian priors
- **Scenario Weights Arbitrary**: Management assigns 50% base, 30% adverse, 20% upside → Subjective; changes provisions significantly; solution: Historical frequency of macro regimes; expert panel consensus; sensitivity analysis
- **Long Maturity (30-year Mortgages)**: PD term structure extends 30 years → High uncertainty; model risk; solution: Flatten PD curve after year 10; use TTC PD for long tail; sensitivity to maturity assumption
- **Revolving Credit (Credit Cards)**: EAD uncertain (drawdown behavior volatile); CCF model critical; solution: Behavioral scoring models; stress CCF (100% in crisis); vintage analysis
- **Model Risk (Overfitting)**: ML model achieves 95% AUC on historical data → Overfit to noise; poor forward performance; solution: Regularization (L1/L2); cross-validation; simple model benchmark
- **Discount Rate Ambiguity (Stage 3)**: IFRS 9 allows original EIR or credit-adjusted rate → Choice impacts ECL by 20-40%; solution: Consistent policy; disclose choice; sensitivity analysis

## 7. Key References
- [IFRS 9 Expected Credit Losses (EY Guide, 2020)](https://www.ey.com/en_gl/ifrs-technical-resources/ifrs-9-expected-credit-loss) - Practical implementation; ECL calculation methodologies; worked examples
- [Basel II: International Convergence of Capital Measurement (BIS, 2006)](https://www.bis.org/publ/bcbs128.pdf) - IRB approach; PD, LGD, EAD definitions; aligns with IFRS 9 ECL models
- [Loan Loss Provisioning and Economic Slowdowns (IMF Working Paper, 2018)](https://www.imf.org/en/Publications/WP/Issues/2018/07/23/Loan-Loss-Provisioning-and-Economic-Slowdowns-Too-Little-Too-Late-46053) - Empirical analysis; forward-looking ECL vs incurred loss; procyclicality

---
**Status:** IFRS 9 Core Methodology | **Complements:** Three-Stage Approach, SICR, Forward-Looking Information, PD/LGD/EAD Models
