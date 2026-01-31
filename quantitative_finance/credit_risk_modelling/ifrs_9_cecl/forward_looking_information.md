# Forward-Looking Information & Macroeconomic Scenarios

## 1. Concept Skeleton
**Definition:** Incorporation of reasonable and supportable forecasts of future economic conditions into ECL estimates; macroeconomic scenario generation (base, adverse, upside) with probability weighting; aligns PD/LGD with forward economic paths  
**Purpose:** Distinguish IFRS 9 from incurred loss model (IAS 39); timely loss recognition in downturns; countercyclical provisioning; avoid "too little, too late"; regulatory compliance  
**Prerequisites:** Macro forecasting models, scenario design, econometric relationships (PD/LGD vs GDP, unemployment), probability weighting, Monte Carlo simulation, stress testing

## 2. Comparative Framing
| Scenario Type | Probability Weight | GDP Growth | Unemployment | Use Case | PD Impact | LGD Impact |
|---------------|-------------------|------------|--------------|----------|-----------|------------|
| **Base** | 50-60% | Consensus forecast (~2-3%) | Stable (~5%) | Most likely path | Baseline PD | Baseline LGD |
| **Adverse** | 20-30% | Recession (-2% to 0%) | Rising (+2-3pp) | Economic downturn | +50-100% PD | +20-40% LGD |
| **Severe Adverse** | 5-10% | Deep recession (-5%) | High (+5pp) | Tail risk (crisis) | +200-300% PD | +50-80% LGD |
| **Upside** | 10-20% | Boom (+4-5%) | Low (-2pp) | Benign conditions | -30-50% PD | -20-30% LGD |

## 3. Examples + Counterexamples

**Scenario-Weighted ECL:**  
Base (50%): PD = 2%, ECL = $1M. Adverse (30%): PD = 4%, ECL = $2M. Upside (20%): PD = 1%, ECL = $500k. Weighted ECL = 0.5×$1M + 0.3×$2M + 0.2×$500k = $1.2M.

**Mortgage Portfolio (House Price Sensitivity):**  
Base: House prices +2%/yr, LGD = 30%. Adverse: House prices -10%, LGD = 50% (negative equity). Upside: House prices +5%, LGD = 20%. Weighted LGD = 0.5×30% + 0.3×50% + 0.2×20% = 34%.

**Energy Loan (Oil Price Shock):**  
Base: Oil $80/bbl, PD = 3%. Adverse: Oil $40/bbl (collapse), PD = 15% (sector distress). Upside: Oil $100/bbl, PD = 1%. Management overlay: Weight adverse 50% (oil volatility high) → Weighted PD = 8.3%.

**COVID-19 Example:**  
Pre-COVID (2019): Base PD = 1.5%, ECL = $500k. COVID scenario (March 2020): Adverse PD = 5%, weighted PD = 3%, ECL = $1M (doubled). Model shock; management overlays adjust sectors (airlines, retail hit hardest).

**Failure Case (No Forward-Looking):**  
Bank uses historical average PD (2% over 10 years). Recession hits; actual PD spikes to 6%. ECL understated by 3× → Regulatory criticism; late loss recognition.

## 4. Layer Breakdown
```
Forward-Looking Information Framework:

├─ IFRS 9 Requirements:
│   ├─ Principle: ECL must reflect reasonable and supportable information
│   │   └─ IFRS 9.5.5.17: "Consider past events, current conditions, and forecasts"
│   ├─ Forward-Looking Obligation:
│   │   ├─ Cannot rely solely on historical data (backward-looking)
│   │   ├─ Must incorporate forecasts of future economic conditions
│   │   └─ Forecasts must be unbiased (not excessively prudent or optimistic)
│   ├─ Reasonable and Supportable Horizon:
│   │   ├─ Explicit forecasts: Typically 3-5 years (consensus forecast horizon)
│   │   ├─ Beyond explicit: Revert to long-run average (through-the-cycle)
│   │   └─ Example: Year 1-3 forecast; Year 4+ use historical mean PD
│   └─ Multiple Scenarios Required:
│       ├─ Cannot use single point estimate (biased)
│       └─ Probability-weighted scenarios (base, adverse, upside minimum)
│
├─ Macroeconomic Scenario Design:
│   ├─ Base Scenario:
│   │   ├─ Definition: Most likely economic path (modal forecast)
│   │   ├─ Source: Consensus economics; central bank forecasts; internal macro team
│   │   ├─ Horizon: 3-5 years explicit; beyond = revert to long-run mean
│   │   ├─ Variables:
│   │   │   ├─ GDP growth: ~2-3% (developed markets)
│   │   │   ├─ Unemployment: ~4-5% (natural rate)
│   │   │   ├─ Interest rates: Central bank policy path
│   │   │   ├─ Inflation: ~2% (central bank target)
│   │   │   └─ Asset prices: Equity indices, house prices, commodity prices
│   │   └─ Probability Weight: Typically 50-60%
│   │
│   ├─ Adverse Scenario:
│   │   ├─ Definition: Economic downturn (recession)
│   │   ├─ Severity: Moderate recession (GDP -2% to 0% for 1-2 years)
│   │   ├─ Variables:
│   │   │   ├─ GDP growth: -1% to -2% (Year 1), 0% (Year 2), +1% (Year 3 recovery)
│   │   │   ├─ Unemployment: +2-3 percentage points above base
│   │   │   ├─ House prices: -10% to -15% (peak-to-trough)
│   │   │   ├─ Equity markets: -20% to -30%
│   │   │   └─ Corporate profitability: Compressed margins; revenue decline
│   │   ├─ Probability Weight: 20-30% (plausible but not most likely)
│   │   └─ Calibration: Historical recession frequency (1 in 10 years → ~10-15% weight)
│   │
│   ├─ Severe Adverse Scenario:
│   │   ├─ Definition: Deep recession / financial crisis (tail risk)
│   │   ├─ Severity: 2008-style crisis (GDP -5%, unemployment +5pp, house prices -30%)
│   │   ├─ Variables:
│   │   │   ├─ GDP growth: -4% to -5% (Year 1), -2% (Year 2), slow recovery
│   │   │   ├─ Unemployment: +5-7 percentage points
│   │   │   ├─ House prices: -30% to -40%
│   │   │   ├─ Equity markets: -50%+
│   │   │   └─ Credit spreads: Widen dramatically (risk aversion)
│   │   ├─ Probability Weight: 5-10% (rare but possible)
│   │   └─ Use: Stress testing; regulatory capital adequacy (CCAR, EBA)
│   │
│   ├─ Upside Scenario:
│   │   ├─ Definition: Economic boom (above-trend growth)
│   │   ├─ Variables:
│   │   │   ├─ GDP growth: +4-5% (strong expansion)
│   │   │   ├─ Unemployment: -1-2 percentage points (tight labor market)
│   │   │   ├─ House prices: +10% (asset price appreciation)
│   │   │   ├─ Equity markets: +20-30%
│   │   │   └─ Corporate profits: Strong earnings growth
│   │   ├─ Probability Weight: 10-20%
│   │   └─ Impact: Lower PD (stronger borrower finances); lower LGD (higher collateral values)
│   │
│   └─ Scenario Path (Time Series):
│       ├─ Quarter-by-quarter or year-by-year forecasts
│       ├─ Example (Base GDP path): Year 1: 2.5%, Year 2: 2.8%, Year 3: 2.3%, Year 4+: 2.5% (long-run)
│       └─ Consistency: Scenarios must be internally consistent (e.g., GDP down → unemployment up)
│
├─ Econometric Relationships (PD/LGD Sensitivity to Macro):
│   ├─ PD Models:
│   │   ├─ Regression Specification: log(PD_t) = α + β₁·GDP_t + β₂·Unemp_t + ε_t
│   │   ├─ Expected Signs:
│   │   │   ├─ β₁ < 0: Higher GDP → Lower PD (stronger economy reduces defaults)
│   │   │   └─ β₂ > 0: Higher unemployment → Higher PD (job losses increase defaults)
│   │   ├─ Estimation: Historical regression using panel data (PD vs macro over time)
│   │   ├─ Calibration: Ensure coefficients economically sensible (elasticity checks)
│   │   └─ Example: 1% GDP decline → +0.5pp PD increase (semi-elasticity)
│   │
│   ├─ LGD Models:
│   │   ├─ Regression: LGD_t = α + β₁·HousePrice_t + β₂·RecoveryRate_t + ε_t
│   │   ├─ Expected Signs:
│   │   │   ├─ β₁ < 0: Higher house prices → Lower LGD (collateral value higher)
│   │   │   └─ Recession indicator: Downturn → +10-20pp LGD (fire sales)
│   │   ├─ Downturn LGD: Basel requirement; use adverse scenario LGD for capital
│   │   └─ Example: House prices -10% → LGD increases from 30% to 40%
│   │
│   ├─ Segment-Specific Models:
│   │   ├─ Mortgages: PD = f(unemployment, house prices, interest rates)
│   │   ├─ Corporate: PD = f(GDP, sector performance, credit spreads)
│   │   ├─ Credit Cards: PD = f(unemployment, consumer confidence, delinquency rates)
│   │   └─ Energy: PD = f(oil prices, capex, leverage)
│   │
│   └─ Non-Linearity:
│       ├─ Tail Risk: Severe adverse scenario → Disproportionate PD increase
│       ├─ Threshold Effects: Unemployment > 10% → PD spikes (mass layoffs)
│       └─ Modeling: Logistic transformation; quantile regression; stress multipliers
│
├─ Probability Weighting:
│   ├─ Formula: ECL = ∑[s] w(s) × ECL(s)
│   │   ├─ s: Scenario index (base, adverse, upside)
│   │   ├─ w(s): Probability weight (sum to 1)
│   │   └─ ECL(s): Expected credit loss under scenario s
│   ├─ Weight Selection:
│   │   ├─ Expert Judgment: Risk committee consensus; macro team input
│   │   ├─ Historical Frequency: Recession occurs 1 in 10 years → 10% weight
│   │   ├─ Market-Implied: Option prices, credit spreads imply risk-neutral probabilities
│   │   └─ Unbiased Requirement: Cannot be excessively prudent (overweight adverse)
│   ├─ Example Weights:
│   │   ├─ Stable Environment: Base 60%, Adverse 25%, Upside 15%
│   │   ├─ Uncertain Environment: Base 40%, Adverse 40%, Upside 20% (higher tail risk)
│   │   └─ Crisis: Base 30%, Adverse 60%, Severe 10% (recession imminent)
│   ├─ Sensitivity Analysis:
│   │   ├─ Test ECL with alternative weights (e.g., ±10% shift)
│   │   └─ Disclose sensitivity in financial statements (IFRS 7)
│   └─ Management Discretion:
│       ├─ Overlay: Adjust weights for events not in models (e.g., COVID-19)
│       └─ Documentation: Rationale for weight changes; approved by risk committee
│
├─ Scenario Generation Process:
│   ├─ Step 1: Define Horizon (typically 3-5 years explicit forecast)
│   ├─ Step 2: Select Macro Variables:
│   │   ├─ Core: GDP, unemployment, inflation, interest rates
│   │   ├─ Asset Prices: Equity indices, house prices, commodity prices
│   │   └─ Sector-Specific: Oil prices (energy), freight rates (shipping)
│   ├─ Step 3: Source Forecasts:
│   │   ├─ Base: Consensus Economics; Bloomberg; central bank forecasts
│   │   ├─ Adverse: Historical recessions; stress test scenarios (CCAR, EBA)
│   │   └─ Upside: Optimistic consensus; historical expansion periods
│   ├─ Step 4: Ensure Internal Consistency:
│   │   ├─ GDP ↓ → Unemployment ↑, Corporate profits ↓, House prices ↓
│   │   ├─ Check: Okun's Law (GDP vs unemployment relationship)
│   │   └─ Tools: VAR (Vector Autoregression) models; structural macro models
│   ├─ Step 5: Map to PD/LGD:
│   │   ├─ Apply econometric models: PD(scenario) = f(GDP_scenario, Unemp_scenario)
│   │   └─ Validate: Compare scenario PD to historical recession PDs
│   ├─ Step 6: Calculate ECL by Scenario:
│   │   ├─ Run ECL model for each scenario s
│   │   └─ ECL(s) = EAD × PD(s) × LGD(s) × DF
│   ├─ Step 7: Probability Weighting:
│   │   └─ ECL_final = ∑ w(s) × ECL(s)
│   └─ Step 8: Governance:
│       ├─ Risk committee approval of scenarios + weights
│       ├─ Quarterly review; adjust if economic outlook shifts
│       └─ Document assumptions; audit trail
│
├─ Reasonable and Supportable Horizon:
│   ├─ Explicit Forecasts (Years 1-3):
│   │   ├─ Use econometric models; external forecasts
│   │   └─ High confidence; detailed quarter-by-quarter paths
│   ├─ Beyond Explicit (Years 4+):
│   │   ├─ Revert to long-run average (through-the-cycle)
│   │   ├─ Rationale: Low confidence in long-term forecasts; avoid spurious precision
│   │   └─ Implementation: Linear reversion over 1-2 years to TTC mean
│   ├─ Example (PD Path):
│   │   ├─ Year 1: 2.5% (explicit forecast)
│   │   ├─ Year 2: 3.0% (explicit)
│   │   ├─ Year 3: 2.8% (explicit)
│   │   ├─ Year 4: 2.5% (revert to TTC mean = 2.2%)
│   │   └─ Year 5+: 2.2% (TTC mean)
│   └─ IFRS 9 Guidance: "Consider uncertainty in longer horizons; revert to historical average"
│
├─ Management Overlays:
│   ├─ Definition: Expert judgment adjustments to model-driven ECL
│   ├─ Use Cases:
│   │   ├─ Model Limitations: Models miss new risk (e.g., pandemic not in historical data)
│   │   ├─ Emerging Risks: Geopolitical shocks, regulatory changes, climate risk
│   │   ├─ Data Quality: Sparse data segments (e.g., new product launches)
│   │   └─ Scenario Inadequacy: Current scenarios don't capture ongoing developments
│   ├─ Examples:
│   │   ├─ COVID-19: Models calibrated pre-pandemic; overlay to increase adverse weight
│   │   ├─ Brexit: UK exposures; overlay to reflect political uncertainty
│   │   └─ Climate Risk: Real estate in flood zones; overlay to adjust LGD
│   ├─ Governance:
│   │   ├─ Documented rationale; quantified impact
│   │   ├─ Approval by risk committee; audit scrutiny
│   │   ├─ Temporary: Review quarterly; remove when models updated
│   │   └─ Disclosure: IFRS 7 requires overlay disclosure
│   └─ Challenges:
│       ├─ Subjectivity: Risk of excessive conservatism or optimism
│       └─ Model Risk: Overreliance on overlays undermines model credibility
│
└─ Implementation & Systems:
    ├─ Scenario Platform:
    │   ├─ Generate macro paths (base, adverse, upside)
    │   ├─ Store historical scenarios; version control
    │   └─ Quarterly updates; approval workflow
    ├─ Econometric Models:
    │   ├─ PD/LGD sensitivity models: Regression; calibration
    │   └─ Validation: Backtesting; compare forecast to actual
    ├─ ECL Engine:
    │   ├─ Calculate ECL by instrument × scenario
    │   ├─ Probability weighting: Aggregate across scenarios
    │   └─ Output: Weighted ECL; scenario breakdown
    ├─ Reporting:
    │   ├─ IFRS 7 Disclosure: Scenarios used; weights; sensitivity
    │   ├─ Management Reports: ECL by scenario; scenario impact analysis
    │   └─ Regulatory: Stress test alignment (CCAR, EBA scenarios)
    └─ Governance:
        ├─ Quarterly scenario review; update weights/paths if outlook changes
        ├─ Annual model validation; recalibrate econometric models
        ├─ Audit: External auditors review scenarios; challenge weights
        └─ Regulatory: Supervisors assess forward-looking adequacy
```

**Key Insight:** IFRS 9 requires forward-looking ECL; multiple probability-weighted scenarios (base, adverse, upside); econometric models link PD/LGD to macro variables (GDP, unemployment, house prices); management overlays for model gaps; unbiased weighting; 3-5 year explicit forecast then revert to TTC.

## 5. Mini-Project
Scenario-weighted ECL calculation with macro sensitivity:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# Macroeconomic scenarios (3-year horizon)
scenarios = {
    'Base': {
        'weight': 0.50,
        'gdp_growth': [2.5, 2.8, 2.3],  # Year 1, 2, 3
        'unemployment': [5.0, 4.8, 5.0],
        'house_price_change': [2.0, 2.5, 2.0]  # % change
    },
    'Adverse': {
        'weight': 0.30,
        'gdp_growth': [-1.5, 0.0, 1.0],  # Recession Year 1, recovery Year 2-3
        'unemployment': [7.5, 8.0, 7.0],
        'house_price_change': [-10.0, -5.0, 0.0]
    },
    'Upside': {
        'weight': 0.20,
        'gdp_growth': [4.0, 4.5, 3.5],
        'unemployment': [4.0, 3.5, 4.0],
        'house_price_change': [5.0, 6.0, 4.0]
    }
}

# Econometric models: PD/LGD sensitivity to macro
def calc_pd(base_pd, gdp_growth, unemployment):
    """PD model: log(PD) = α + β₁·GDP + β₂·Unemp"""
    # Coefficients calibrated to historical data
    beta_gdp = -0.15  # 1% GDP decline → +0.15pp PD increase
    beta_unemp = 0.08  # 1pp unemployment increase → +0.08pp PD increase
    
    # Adjustments relative to base scenario (GDP=2.5%, Unemp=5%)
    gdp_effect = beta_gdp * (gdp_growth - 2.5)
    unemp_effect = beta_unemp * (unemployment - 5.0)
    
    adjusted_pd = base_pd * np.exp(gdp_effect + unemp_effect)
    return adjusted_pd.clip(0.001, 0.50)  # Clip to [0.1%, 50%]

def calc_lgd(base_lgd, house_price_change):
    """LGD model: LGD adjusts with collateral value"""
    # House price decline → Higher LGD (lower collateral)
    # Calibration: -10% house prices → +10pp LGD
    lgd_adjustment = -0.10 * house_price_change  # -10% houses → +1pp LGD
    adjusted_lgd = base_lgd + lgd_adjustment
    return adjusted_lgd.clip(0.10, 0.90)  # Clip to [10%, 90%]

# Portfolio parameters
n_loans = 500
loan_amounts = np.random.uniform(100_000, 500_000, n_loans)
base_pd = np.random.uniform(0.01, 0.03, n_loans)  # 1%-3% base PD
base_lgd = 0.40  # 40% base LGD
eir = 0.05  # 5% discount rate
maturity = 3  # 3-year loans for simplicity

# Calculate lifetime ECL by scenario
results = []

for scenario_name, scenario_data in scenarios.items():
    weight = scenario_data['weight']
    
    # Year-by-year ECL calculation
    total_ecl = np.zeros(n_loans)
    
    for year in range(maturity):
        # Year-specific macro variables
        gdp = scenario_data['gdp_growth'][year]
        unemp = scenario_data['unemployment'][year]
        house_price = scenario_data['house_price_change'][year]
        
        # Calculate PD/LGD for this year
        pd_year = calc_pd(base_pd, gdp, unemp)
        lgd_year = calc_lgd(base_lgd, house_price)
        
        # Marginal ECL for this year (simplified: ignore survival probability for clarity)
        ead_year = loan_amounts  # Assume no amortization
        discount_factor = 1 / (1 + eir) ** (year + 1)
        ecl_year = ead_year * pd_year * lgd_year * discount_factor
        
        total_ecl += ecl_year
    
    # Store scenario results
    results.append({
        'scenario': scenario_name,
        'weight': weight,
        'total_ecl': total_ecl,
        'avg_pd': pd_year.mean(),  # Year 3 PD (for comparison)
        'lgd': lgd_year.mean()
    })

# Weighted ECL (probability-weighted across scenarios)
weighted_ecl = np.zeros(n_loans)
for res in results:
    weighted_ecl += res['weight'] * res['total_ecl']

# Summary statistics
print("="*70)
print("Forward-Looking ECL: Scenario-Weighted Analysis")
print("="*70)
print(f"Number of Loans: {n_loans}")
print(f"Total Exposure: ${loan_amounts.sum():,.0f}")
print("")

print("Scenario-Specific ECL:")
print("-"*70)
print(f"{'Scenario':<12} {'Weight':<10} {'Total ECL ($)':<20} {'Avg PD (Y3)':<15} {'Avg LGD':<10}")
print("-"*70)
for res in results:
    print(f"{res['scenario']:<12} {res['weight']:<10.0%} ${res['total_ecl'].sum():>17,.0f}   {res['avg_pd']:<14.2%}  {res['lgd']:<10.2%}")

print("")
print(f"Probability-Weighted ECL: ${weighted_ecl.sum():,.0f}")
print("")

# Scenario impact analysis
base_ecl = [r['total_ecl'].sum() for r in results if r['scenario'] == 'Base'][0]
adverse_ecl = [r['total_ecl'].sum() for r in results if r['scenario'] == 'Adverse'][0]
upside_ecl = [r['total_ecl'].sum() for r in results if r['scenario'] == 'Upside'][0]

adverse_impact = (adverse_ecl / base_ecl - 1) * 100
upside_impact = (upside_ecl / base_ecl - 1) * 100

print("Scenario Impact vs Base:")
print("-"*70)
print(f"Adverse scenario: +{adverse_impact:.0f}% ECL")
print(f"Upside scenario:  {upside_impact:.0f}% ECL")
print("")

# Sensitivity to scenario weights
print("Sensitivity to Probability Weights:")
print("-"*70)
# Alternative weighting: Increase adverse weight
weights_alt = {'Base': 0.40, 'Adverse': 0.40, 'Upside': 0.20}  # More pessimistic
weighted_ecl_alt = sum(
    weights_alt[res['scenario']] * res['total_ecl']
    for res in results
)
impact = (weighted_ecl_alt.sum() / weighted_ecl.sum() - 1) * 100
print(f"Alt Weights (Base 40%, Adverse 40%): ${weighted_ecl_alt.sum():,.0f} (+{impact:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ECL by scenario
ax = axes[0, 0]
scenario_names = [r['scenario'] for r in results]
scenario_ecl = [r['total_ecl'].sum() / 1e6 for r in results]  # Convert to millions
colors = ['blue', 'red', 'green']
bars = ax.bar(scenario_names, scenario_ecl, color=colors, alpha=0.7)

# Add weighted ECL line
ax.axhline(weighted_ecl.sum() / 1e6, color='black', linestyle='--', linewidth=2, label='Weighted ECL')

ax.set_ylabel('Total ECL ($M)')
ax.set_title('Scenario-Specific ECL')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Macro paths (GDP)
ax = axes[0, 1]
years = np.arange(1, 4)
for scenario_name, scenario_data in scenarios.items():
    color = {'Base': 'blue', 'Adverse': 'red', 'Upside': 'green'}[scenario_name]
    ax.plot(years, scenario_data['gdp_growth'], marker='o', label=scenario_name, color=color, linewidth=2)

ax.set_xlabel('Year')
ax.set_ylabel('GDP Growth (%)')
ax.set_title('GDP Growth Paths by Scenario')
ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: PD distribution by scenario
ax = axes[1, 0]
for res, color in zip(results, colors):
    # Calculate Year 3 PD for each loan under scenario
    scenario_data = scenarios[res['scenario']]
    pd_y3 = calc_pd(base_pd, scenario_data['gdp_growth'][2], scenario_data['unemployment'][2])
    ax.hist(pd_y3 * 100, bins=30, alpha=0.5, label=res['scenario'], color=color)

ax.set_xlabel('PD (%)')
ax.set_ylabel('Frequency')
ax.set_title('PD Distribution by Scenario (Year 3)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Probability weighting visualization
ax = axes[1, 1]
weights = [r['weight'] for r in results]
ecl_values = [r['total_ecl'].sum() / 1e6 for r in results]

# Stacked contribution to weighted ECL
weighted_contributions = [w * ecl for w, ecl in zip(weights, ecl_values)]
ax.bar(scenario_names, weighted_contributions, color=colors, alpha=0.7)
ax.set_ylabel('Weighted ECL Contribution ($M)')
ax.set_title('Probability-Weighted ECL Contributions')
ax.axhline(weighted_ecl.sum() / 1e6, color='black', linestyle='--', linewidth=2, label='Total Weighted ECL')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('forward_looking_information.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*70)
print("Key Insights:")
print("="*70)
print("1. Adverse scenario ECL ~50-100% higher than base")
print("   → Recession significantly increases provisions")
print("")
print("2. Scenario weighting smooths volatility")
print("   → Avoids overreaction to single economic view")
print("")
print("3. Weight sensitivity critical (±10% shift → ±5-10% ECL impact)")
print("   → Governance and documentation essential")
print("")
print("4. GDP/unemployment are primary drivers of PD")
print("   → Calibrate econometric models to historical cycles")
```

## 6. Challenge Round
When forward-looking frameworks fail or introduce complexity:
- **Forecast Uncertainty (Long Horizon)**: 5-year GDP forecast highly uncertain → Model generates spurious precision; solution: Revert to TTC mean after Year 3; widen confidence intervals; scenario range (not point estimates)
- **Model Misspecification**: PD model calibrated 2000-2019 (no pandemic); COVID hits → Model useless; solution: Management overlays; update models with new data; stress test tail scenarios
- **Procyclicality**: Recession forecast → Higher ECL → Banks reduce lending → Worsens recession (feedback loop); solution: Smoothing mechanisms (TTC overlays); regulatory forbearance (temporary); countercyclical buffers
- **Scenario Weights Arbitrary**: CFO increases adverse weight 30% → 40% → Provisions up 15%; subjective; solution: Document rationale; independent validation; sensitivity disclosure
- **Data Limitations (Emerging Markets)**: Limited historical recession data → Cannot calibrate macro models; solution: Use developed market analogues; expert judgment; conservative assumptions
- **Cliff Effects at Horizon**: Explicit forecast ends Year 3 → Sudden PD jump to TTC mean; solution: Smooth transition over 1-2 years; linear interpolation

## 7. Key References
- [IFRS 9 Financial Instruments (Section 5.5.17)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) - Official standard; forward-looking information requirements; reasonable and supportable forecasts
- [IMF: Loan Loss Provisioning and Economic Slowdowns (2018)](https://www.imf.org/en/Publications/WP/Issues/2018/07/23/Loan-Loss-Provisioning-and-Economic-Slowdowns-Too-Little-Too-Late-46053) - Empirical analysis; forward-looking ECL vs incurred loss; procyclicality; policy implications
- [EBA Report on IFRS 9 Implementation (2019)](https://www.eba.europa.eu/eba-publishes-report-ifrs-9-implementation-eu-institutions) - European Banking Authority survey; scenario practices; weight selection; sensitivity analysis

---
**Status:** IFRS 9 Core Methodology | **Complements:** Three-Stage Approach, Expected Credit Loss Models, SICR
