# Maximum Loss and Stress Testing

## 1. Concept Skeleton
**Definition:** Identification of worst-case loss scenarios through historical analysis or hypothetical stress events; risk management approach quantifying extreme loss potential beyond statistical models  
**Purpose:** Complement probabilistic measures (VaR, CVaR) with deterministic worst-case; identify portfolio vulnerabilities; prepare for model-breaking events; enforce discipline on tail risk awareness  
**Prerequisites:** Portfolio risk concepts, scenario analysis, market history, sensitivity analysis, extreme events, stress testing frameworks

## 2. Comparative Framing
| Aspect | VaR | CVaR | LPM | Max Loss | Stress Test | Scenario |
|--------|-----|------|-----|----------|-------------|----------|
| **Definition** | q-th percentile | Tail average | Downside moment | Worst historical | Hypothetical worst | Narrative scenario |
| **Basis** | Statistical (quantile) | Statistical (tail) | Statistical (moment) | Historical extreme | Model-based projection | Expert judgment |
| **Time Horizon** | Fixed period | Fixed period | Fixed period | Specific date | Under condition | Event description |
| **Frequency** | Probability-weighted | Probability-weighted | Probability-weighted | Once observed | Probability low | Not quantified |
| **Model Dependence** | Distribution assumed | Distribution | Distribution | None (empirical) | High (full model) | Moderate (structured) |
| **Forward-Looking** | Yes (forecast) | Yes (forecast) | Yes (forecast) | No (backward) | Yes (forward) | Depends |
| **Worst-Case Bound** | No (tail continues) | No (tail continues) | No (tail continues) | Yes (observed) | Yes (defined) | Yes (defined) |

## 3. Examples + Counterexamples

**Historical Max Loss:**  
S&P 500 daily returns (1950-2024): Worst day = -20.4% (Oct 19, 1987, Black Monday). Max annual loss = -37% (2008 financial crisis). Worst 5-year period = -50% (2000-2004 tech crash). Portfolio of 60/40 stock/bond: Max loss ≈ -27% (mostly stock downside during 2008).

**Stress Test Scenario:**  
"Geopolitical shock: Major war announcement, oil +100%, credit spreads +200 bps, equity vol +300 bps." Run historical analogy (e.g., 1990 Gulf War) or parametric shock. Result: Portfolio estimated loss -15% to -25% depending on positioning.

**Model Failure Case:**  
Pre-2008: VaR models estimated max daily loss 3-4%. Oct 2008 realized: -9% day. Max monthly loss predicted 15-20%, realized -33%. Model broke due to correlation spike (all assets down together).

**Leverage Amplification:**  
Hedge fund leveraged 3x. Stock position -20% per day limit. Leveraged: -60% → wipeout if circuit breaker trips. Stress test: "3 standard deviation day + leverage." Result: Total loss scenario + margin call spiral.

**Tail Risk Insurance:**  
Portfolio with long put protection. Max loss = strike level (capped). Without put: Unbounded loss potential. Stress test validates: "Worst case ≤ strike" assumption holds or fails (depends on liquidity, counterparty).

**Tail Correlation Breakdown:**  
Normally uncorrelated assets. Crisis: Correlation → 1. Max loss higher than historical expected. Stress scenario: "All correlations = +1" → portfolio vol tripled. Max loss bounds become critical.

**Hidden Risk (VaR→Max Loss Gap):**  
1% VaR = $10M loss. Historical analysis shows single day could be -$50M (maximum observed). Gap suggests either: (1) VaR underestimates, (2) Regime shift, (3) Black swan. Stress testing bridges gap.

## 4. Layer Breakdown
```
Maximum Loss & Stress Testing Framework:

├─ Maximum Loss Definition:
│  ├─ Historical Maximum:
│  │   Max_Loss = min(returns) × portfolio_value
│  │   Worst observed return over period
│  │   Deterministic (no distribution assumed)
│  ├─ Period Selection:
│  │   Full available history: More comprehensive, regime shifts
│  │   Recent window (3-5 years): Regime-relevant, ignores ancient crises
│  │   Crisis-only window: Captures tail, biased pessimism
│  ├─ Calculation:
│  │   Portfolio Daily Returns: Rₚ,ₜ = Σᵢ wᵢ Rᵢ,ₜ
│  │   Maximum Loss = -min(Rₚ,ₜ) × Portfolio Value
│  │   E.g., if worst day = -5%, then Max_Loss = 5% of AUM
│  ├─ Expected Tail Loss:
│  │   Rarer than max observed (next tail event)
│  │   EVT: Extrapolate beyond sample
│  │   Estimated via extreme value distribution fit
│  ├─ Annual vs Daily:
│  │   Max daily: Highest single-day loss
│  │   Max monthly/annual: Multi-period compounding
│  │   Annual typically more severe
│  └─ Portfolio-Specific:
│      Different portfolio → different max loss
│      Leverage multiplies max loss
│      Currency exposure adds dimension
├─ Calculation Methodologies:
│  ├─ Historical Observation (Non-Parametric):
│  │   Data: Historical returns
│  │   Method: Sort, find minimum
│  │   Advantages: No model, purely empirical
│  │   Disadvantages: Limited by history, rare events missed
│  ├─ Parametric EVT (Extreme Value Theory):
│  │   1. Fit Generalized Pareto Distribution to tail
│  │   2. Extrapolate beyond sample
│  │   3. Estimate tail quantile (e.g., 1000-year event)
│  │   Advantages: Scientifically estimates unseen extremes
│  │   Disadvantages: Model assumptions, tail calibration risk
│  ├─ Monte Carlo Simulation:
│  │   1. Model portfolio (stock, bond, derivatives)
│  │   2. Generate scenarios (10k to 1M paths)
│  │   3. Compute portfolio value path
│  │   4. Find minimum value across scenarios
│  │   Advantages: Flexible, captures complex dynamics
│  │   Disadvantages: Model risk, computational
│  ├─ Historical Scenario:
│  │   Find worst portfolio loss over past crisis
│  │   Back-test portfolio against Black Monday, 2008, COVID, etc.
│  │   Limitations: Market structure changes (circuit breakers, margin rules)
│  ├─ Composite Approach:
│  │   Combine multiple methods
│  │   Take highest estimate (conservative)
│  │   Hedge against single-method failure
│  └─ Sensitivity Analysis:
│      Vary correlations, volatilities
│      Find portfolio max loss under different assumptions
├─ Stress Testing Framework:
│  ├─ Definition:
│  │   Hypothetical scenario analysis
│  │   Quantify portfolio impact if specific events occur
│  │   Not necessarily based on historical frequency
│  ├─ Scenario Types:
│  │   1. Historical: "Repeat 2008 crisis"
│  │   2. Parametric: "Vol +100%, correlations +0.2"
│  │   3. Narrative: "Trade war, capital flight to USD"
│  │   4. Reverse: "What would cause -30% loss?"
│  │   5. Sensitivity: "Each 1% rate move = X loss"
│  ├─ Typical Stress Scenarios:
│  │   Equity Crash: -20%, vol spike, spreads widen
│  │   Credit Crisis: Credit spreads +300 bps, defaults +10%
│  │   Rates Shock: Yield curve +/- 200 bps
│  │   Currency: USD +/-20% vs major currencies
│  │   Commodity: Oil +/-50%, metals +/-30%
│  │   Volatility: All vol +200%, correlations +0.5
│  │   Systemic: Combination (all bad simultaneously)
│  ├─ Stress Test Mechanics:
│  │   1. Define scenario (inputs)
│  │   2. Map market moves to asset prices
│  │   3. Revalue portfolio under scenario
│  │   4. Compare to baseline
│  │   5. Calculate portfolio impact
│  ├─ Asset-Specific Sensitivities:
│  │   Bonds: -5 × Δ(Yields) (duration effect)
│  │   Equities: β × Δ(Market) + α (beta times market move)
│  │   Options: Vega × Δ(Vol) + Gamma × (Δ(Underlying))²
│  │   FX: Δ(Exchange Rate) directional impact
│  ├─ Second-Order Effects:
│  │   Gamma: Convexity, non-linear P&L
│  │   Correlation breakdown: Cross-asset spillovers
│  │   Feedback loops: Losses trigger margin calls, forced sales
│  │   Fire sales: Illiquidity, widening spreads
│  ├─ Reverse Stress Testing:
│  │   Start with max loss tolerance: "Can't lose >$X"
│  │   Work backward: What market moves cause this?
│  │   Identifies portfolio vulnerabilities
│  ├─ Aggregation:
│  │   Combine component losses
│  │   Worst-case: Additive (all losses simultaneous)
│  │   Realistic: Portfolio diversification reduces combined loss
│  └─ Scenario Probability:
│      Historical frequency (if scenario occurred before)
│      Expert judgment (if novel scenario)
│      Not used for stress test itself (deterministic)
├─ Portfolio Vulnerabilities:
│  ├─ Concentration Risk:
│  │   Single security, sector, geography
│  │   Max loss if concentrated asset crashes
│  │   Stress test: Single position -20%, -50%, -100%
│  ├─ Leverage Risk:
│  │   Amplifies all losses
│  │   Max loss = leverage × underlying max loss
│  │   Stress: Margin call spiral (forced liquidation)
│  ├─ Illiquidity Risk:
│  │   Can't exit position at crisis prices
│  │   Bid-ask widens, depth disappears
│  │   Max loss = fundamental loss + illiquidity discount
│  ├─ Counterparty Risk:
│  │   OTC derivatives, repo
│  │   If counterparty defaults, lose collateral
│  │   Stress: Counterparty credit event + market move
│  ├─ Tail Correlation:
│  │   Diversifier fails in crisis
│  │   All correlated during extreme stress
│  │   Max loss higher than normal period suggests
│  ├─ Model Risk:
│  │   Greeks (delta, gamma) wrong during extreme move
│  │   Jump risk (gaps overnight)
│  │   Smile/skew changes
│  └─ Basis Risk:
│      Hedge doesn't perfectly offset exposure
│      Stress test: Hedge breaks down scenario
├─ Regulatory Stress Testing:
│  ├─ Banking (Fed Stress Tests):
│  │   Defined scenarios: Baseline, adverse, severe
│  │   Major banks must maintain capital above minimum
│  │   Public reporting (capital adequacy)
│  ├─ Dodd-Frank Requirements:
│  │   Large banks: Annual stress tests (Fed + banks)
│  │   Insurance companies: Own risk/solvency assessment
│  │   Transparency on max loss under scenarios
│  ├─ Basel III Guidance:
│  │   Scenarios cover: Interest rates, spreads, FX, equity, commodity
│  │   Joint stress (all move together)
│  │   Feedback loops included
│  ├─ Solvency II (Insurance):
│  │   Own Risk Solvency Assessment (ORSA)
│  │   Standardized scenarios + company-specific
│  │   Capital requirement calibrated to stress outcomes
│  └─ Dodd-Frank Volcker Rule:
│      Proprietary trading desks: VaR + stress P&L
│      Comprehensive risk factor coverage
│      Public disclosure thresholds
├─ Historical Crisis Benchmarks:
│  ├─ Black Monday (Oct 19, 1987):
│  │   S&P -22.6% single day (largest single-day drop)
│  │   Portfolio stress: Equity-heavy portfolio -15-20%
│  ├─ LTCM Crisis (1998):
│  │   Flight to quality, credit spreads +300 bps
│  │   Convertible bonds, emerging markets hit hard
│  │   Systemic risk (Fed intervention required)
│  ├─ Dot-com Crash (2000-2002):
│  │   NASDAQ -78% cumulative
│  │   Growth stock portfolio -60-70%
│  │   Long duration period (not single-day shock)
│  ├─ 2008 Financial Crisis:
│  │   S&P -50% annual return, -57% peak-to-trough
│  │   Credit spreads +400 bps, liquidity dried up
│  │   Leverage amplified losses
│  ├─ COVID Crash (March 2020):
│  │   S&P -34% in 23 days (fastest bear market)
│  │   VIX spiked 400%+, correlation spike to 1.0
│  │   Volatility ETN products broke, forced liquidations
│  └─ Lessons:
│      Max loss can exceed VaR by 10x
│      Tail correlation breaks diversification
│      Liquidity evaporates in crises
├─ Implementation Considerations:
│  ├─ Data:
│  │   Historical returns (minimally 5-10 years)
│  │   Crisis period coverage essential
│  │   Survivorship bias (companies removed from history)
│  ├─ Governance:
│  │   Stress testing committee (risk, business, finance)
│  │   Scenario approval process (regulators involved)
│  │   Regular review and update (annual minimum)
│  ├─ Documentation:
│  │   Scenario definitions (repeatable)
│  │   Methodology (transparent)
│  │   Assumptions (clearly stated)
│  │   Results (tracked over time)
│  ├─ Escalation:
│  │   If stress test loss > threshold → escalate to management
│  │   Hedging decision, position reduction
│  │   Public disclosure (regulatory requirement)
│  ├─ Backtesting:
│  │   Did hypothetical scenarios actually occur?
│  │   How did actual P&L compare to stress prediction?
│  │   Refine model based on outcomes
│  └─ Technology:
│      Scenario server (compute portfolio revaluation)
│      Risk data warehouse
│      Reporting dashboards
├─ Stress Test Interpretation:
│  ├─ Conservative:
│  │   Take worst outcome ("stressed max loss")
│  │   Accumulate all losses additively
│  │   No benefit of diversification
│  ├─ Realistic:
│  │   Apply historical relationships
│  │   Account for diversification benefits
│  │   Typical market stress scenario
│  ├─ Expected Shortfall:
│  │   Average loss across scenarios
│  │   Bridge between VaR and worst-case
│  └─ Risk Limit:
│      "Max daily loss ≤ $X"
│      "VaR ≤ $Y"
│      "Stress test loss ≤ $Z" (tightest constraint)
└─ Integration with Portfolio Management:
   ├─ Hedging Decisions:
   │   If stress loss unacceptable → buy protection
   │   Cost benefit: Premium vs. peace of mind
   ├─ Leverage Management:
   │   Reduce leverage if stress scenarios unacceptable
   │   Trade-off: Return vs. risk
   ├─ Diversification:
   │   Add uncorrelated assets if stress loss high
   │   Verify correlation holds in stress
   ├─ Position Sizing:
   │   Reduce concentrated positions
   │   Align max loss with risk budget
   ├─ Strategy Evaluation:
   │   Compare strategy max loss under stress
   │   Rank strategies by stress resilience
   └─ Investor Communication:
       "Worst case loss under X scenario: Y%"
       More tangible than VaR for most investors
```

**Interaction:** Collect data → Identify worst historical scenarios → Define stress scenarios → Revalue portfolio → Compare impacts → Set risk limits → Monitor and escalate.

## 5. Mini-Project
Stress test portfolio across historical crises and hypothetical scenarios:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Portfolio definition
portfolio = {
    'US Equities': {'weight': 0.60, 'volatility': 0.15},
    'Bonds': {'weight': 0.30, 'volatility': 0.05},
    'Commodities': {'weight': 0.10, 'volatility': 0.12}
}

# Historical scenarios (actual crises + worst days)
historical_scenarios = {
    'Black Monday 1987': {
        'US Equities': -0.226,
        'Bonds': 0.02,
        'Commodities': -0.05
    },
    'LTCM Crisis 1998': {
        'US Equities': -0.15,
        'Bonds': -0.02,
        'Commodities': -0.10
    },
    '2008 Financial Crisis': {
        'US Equities': -0.37,
        'Bonds': 0.05,
        'Commodities': -0.35
    },
    'COVID Crash March 2020': {
        'US Equities': -0.34,
        'Bonds': 0.02,
        'Commodities': -0.20
    },
    'Dot-com Peak (2000-2002)': {
        'US Equities': -0.78,
        'Bonds': 0.12,
        'Commodities': -0.10
    }
}

# Hypothetical stress scenarios
hypothetical_scenarios = {
    'Mild Correction': {
        'US Equities': -0.10,
        'Bonds': 0.01,
        'Commodities': -0.05
    },
    'Moderate Downturn': {
        'US Equities': -0.20,
        'Bonds': 0.03,
        'Commodities': -0.15
    },
    'Severe Crisis (All Down)': {
        'US Equities': -0.40,
        'Bonds': -0.05,
        'Commodities': -0.30
    },
    'Stagflation Scenario': {
        'US Equities': -0.25,
        'Bonds': -0.20,
        'Commodities': +0.40
    },
    'Rate Shock (+200 bps)': {
        'US Equities': -0.15,
        'Bonds': -0.12,
        'Commodities': +0.05
    },
    'Geopolitical Crisis': {
        'US Equities': -0.18,
        'Bonds': +0.02,
        'Commodities': +0.25
    }
}

# Calculate portfolio impact for each scenario
def calculate_portfolio_loss(portfolio, scenario):
    """Calculate portfolio loss under scenario"""
    loss = 0
    for asset, asset_info in portfolio.items():
        weight = asset_info['weight']
        asset_move = scenario[asset]
        loss += weight * asset_move
    return loss

# Evaluate all scenarios
all_scenarios = {**historical_scenarios, **hypothetical_scenarios}
results = []

for scenario_name, moves in all_scenarios.items():
    loss = calculate_portfolio_loss(portfolio, moves)
    scenario_type = 'Historical' if scenario_name in historical_scenarios else 'Hypothetical'
    results.append({
        'Scenario': scenario_name,
        'Type': scenario_type,
        'Portfolio Loss': loss,
        'Loss %': f'{loss*100:.2f}%',
        'USD Loss ($M)': loss * 100  # Assuming $100M AUM
    })

results_df = pd.DataFrame(results).sort_values('Portfolio Loss')

# Print results
print("="*100)
print("STRESS TEST RESULTS")
print("="*100)
print(f"\nPortfolio Composition: ", end='')
for asset, info in portfolio.items():
    print(f"{asset} {info['weight']*100:.0f}%, ", end='')
print("\n")

print(results_df.to_string(index=False))

print(f"\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"Worst-case loss: {results_df['Portfolio Loss'].min()*100:.2f}% ({results_df['Scenario'].iloc[0]})")
print(f"Best-case loss: {results_df['Portfolio Loss'].max()*100:.2f}% ({results_df['Scenario'].iloc[-1]})")
print(f"Average loss (all scenarios): {results_df['Portfolio Loss'].mean()*100:.2f}%")
print(f"Median loss: {results_df['Portfolio Loss'].median()*100:.2f}%")
print(f"Standard deviation: {results_df['Portfolio Loss'].std()*100:.2f}%")

# Calculate VaR approximation for comparison
daily_vol_portfolio = np.sqrt(
    sum(portfolio[asset]['weight']**2 * portfolio[asset]['volatility']**2 
        for asset in portfolio)
)
var_95 = daily_vol_portfolio * 1.645  # 95% confidence
var_99 = daily_vol_portfolio * 2.326  # 99% confidence

print(f"\nTraditional Risk Metrics (Daily):")
print(f"  Portfolio Volatility: {daily_vol_portfolio*100:.2f}%")
print(f"  VaR (95%, 1-day): {var_95*100:.2f}%")
print(f"  VaR (99%, 1-day): {var_99*100:.2f}%")
print(f"\nMax Loss vs VaR Gap:")
print(f"  Worst Stress Loss: {results_df['Portfolio Loss'].min()*100:.2f}%")
print(f"  VaR(99%): {var_99*100:.2f}%")
print(f"  Gap: {(results_df['Portfolio Loss'].min() - (-var_99))*100:.2f}% (stress worse)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss comparison (historical vs hypothetical)
ax = axes[0, 0]
colors = ['red' if t == 'Historical' else 'orange' for t in results_df['Type']]
sorted_results = results_df.sort_values('Portfolio Loss')
ax.barh(range(len(sorted_results)), sorted_results['Portfolio Loss']*100, color=colors, alpha=0.7)
ax.set_yticks(range(len(sorted_results)))
ax.set_yticklabels(sorted_results['Scenario'], fontsize=9)
ax.axvline(x=-var_95*100, color='blue', linestyle='--', linewidth=2, label=f'VaR(95%)={-var_95*100:.1f}%')
ax.axvline(x=-var_99*100, color='green', linestyle='--', linewidth=2, label=f'VaR(99%)={-var_99*100:.1f}%')
ax.set_xlabel('Portfolio Loss (%)')
ax.set_title('Stress Test Results: All Scenarios')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Plot 2: Historical vs Hypothetical box plot
ax = axes[0, 1]
hist_losses = results_df[results_df['Type'] == 'Historical']['Portfolio Loss'].values * 100
hyp_losses = results_df[results_df['Type'] == 'Hypothetical']['Portfolio Loss'].values * 100
ax.boxplot([hist_losses, hyp_losses], labels=['Historical', 'Hypothetical'])
ax.set_ylabel('Portfolio Loss (%)')
ax.set_title('Loss Distribution: Historical vs Hypothetical')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Asset contribution to losses
ax = axes[1, 0]
scenario_worst = historical_scenarios['2008 Financial Crisis']
contributions = []
assets_list = list(portfolio.keys())
for asset in assets_list:
    weight = portfolio[asset]['weight']
    move = scenario_worst[asset]
    contribution = weight * move * 100  # Convert to percentage
    contributions.append(contribution)

colors_contrib = ['red' if c < 0 else 'green' for c in contributions]
ax.bar(assets_list, contributions, color=colors_contrib, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Contribution to Loss (%)')
ax.set_title('Asset Contribution: 2008 Crisis Scenario')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Risk metrics comparison
ax = axes[1, 1]
risk_metrics = {
    'Daily Vol': daily_vol_portfolio * 100,
    'VaR(95%)': var_95 * 100,
    'VaR(99%)': var_99 * 100,
    'Worst Stress': -results_df['Portfolio Loss'].min() * 100,
    'Avg Stress': -results_df['Portfolio Loss'].mean() * 100,
}
x = np.arange(len(risk_metrics))
ax.bar(x, list(risk_metrics.values()), color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(list(risk_metrics.keys()), rotation=45, ha='right')
ax.set_ylabel('Loss (%)')
ax.set_title('Risk Metrics Summary')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
- Derive tail loss estimate via extreme value theory (Pickands-Balkema-de Haan)
- Design reverse stress test: "What scenarios cause 20% loss?"
- Implement cascading stress: Single asset failure → contagion impact
- Compare max loss: Historical vs EVT vs Monte Carlo estimates
- Explain stress test limitations: Model risk, scenario incompleteness, tail dependence

## 7. Key References
- [Jorion, "Value at Risk: The New Benchmark for Managing Financial Risk" (2007)](https://www.amazon.com/Value-Risk-Benchmark-Managing-Financial/dp/0071464956) — Stress testing comprehensive
- [Fed Stress Test Methodology (Annual)](https://www.federalreserve.gov/bankinforeg/stress-tests.htm) — Regulatory framework
- [McNeil et al, "Quantitative Risk Management" (2015)](https://www.cambridge.org/core/books/quantitative-risk-management/) — EVT for max loss
- [Basel Committee, "Stress Testing Guidance" (2012)](https://www.bis.org/bcbs/) — Best practices

---
**Status:** Deterministic worst-case complement to probabilistic measures | **Complements:** VaR, CVaR, Scenario Analysis, Risk Limits
