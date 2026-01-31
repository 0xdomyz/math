# Home Bias & Familiarity Bias in Global Portfolios

## 1. Concept Skeleton
**Definition:** Empirical phenomenon where investors significantly overweight domestic equities in portfolios despite international diversification benefits; deviation from CAPM prediction of world market portfolio weights; driven by familiarity, information asymmetries, behavioral preferences, and transaction costs  
**Purpose:** Quantify home bias magnitude, explain behavioral/structural drivers, analyze whether home bias rational or purely behavioral, determine optimal international allocation  
**Prerequisites:** International diversification, CAPM market portfolio theory, behavioral finance, information asymmetries

---

## 2. Comparative Framing

| Characteristic | CAPM Theoretical Prediction | Actual US Investor Practice | Developed Markets Average | Emerging Markets |
|----------------|---------------------------|---------------------------|---------------------------|-----------------|
| **US Stock Allocation** | 35-45% (US share of global market cap) | 70-90% (home bias) | 40-60% domestic bias | Extreme: 95%+ domestic |
| **International Allocation** | 55-65% (rest of world) | 10-30% (underweight) | 40-60% | <5% |
| **Explanations** | Market cap weighting | Behavioral + information gaps | Taxes + language + politics | Capital controls + illiquidity |
| **Efficiency Loss** | 0% (optimal) | ~0.5-1.5% p.a. variance drag | ~0.2-0.5% p.a. | ~2-3% p.a. (extreme) |
| **Documented Since** | CAPM theory (1960s) | French & Poterba (1991) found extreme bias | Tesar & Werner (1995) | Emerging 1990s-2000s |
| **Recent Trend** | Unchanged (theory doesn't explain) | Declining slightly with ETFs | Stable (~40% home bias) | Increasing with growth |
| **Cost to Investors** | 0 (market price is fair) | $2-5 trillion underallocation | ~$500B globally | ~$100B in EM |

**Key Insight:** Home bias is near-universal but varies with country size (larger countries have larger bias), development (emerging markets show extreme bias), and investor sophistication (institutions less biased than retail).

---

## 3. Examples + Counterexamples

**Example 1: The "Extreme" US Home Bias (Classic Finding)**
- US market cap: ~$40T (2024) = ~35% of global market
- CAPM theory: US investors should hold 35% US stocks
- Actual US investor holdings: ~75% US stocks (2x CAPM weight)
- Allocation underweight: International 45% underweight (holding 20% instead of 35%)
- Return consequence: If international stocks outperform by 2% (2010-2020 did not happen), 0.9% annual drag
- Actual outcome: US equities outperformed 2010-2024; home bias was lucky (not rational decision)

**Example 2: Japanese Home Bias During Lost Decade (1990s)**
- Japan's market cap: ~15% of global (1989 peak)
- Japanese investors held: ~95% domestic stocks
- Nikkei peak 1989: 40,000; Nikkei 2000: 20,000 (-50%)
- Japanese investors stuck with losing bets; zero international diversification benefit
- If they had 35% international: Minimized drawdown, captured global recovery
- Implication: Home bias most costly when home country underperforms (when diversification would help most)

**Example 3: UK "Home Bias" Actually Rational (Pound Liability Matching)**
- UK pension fund: £100M obligations in GBP
- CAPM suggests: 60% UK stocks, 40% international
- Actual allocation: Often 70-80% UK assets
- Rationale: Liabilities in GBP → FX risk if international assets decline + pound strengthens
- Liability-hedging perspective: Home bias rational (reduces liability-driven risk)
- Implication: Home bias not always irrational if liabilities domestic-currency denominated

**Example 4: Familiarity Effect (Investor Knows Home Companies)**
- Investor holds:
  - 50% Apple (US tech giant; knows via media)
  - 20% US bonds
  - 5% ASML (Dutch semiconductor; unfamiliar)
  - 2% Toyota (Japanese; less known)
  - 3% Tata Consultancy (Indian; unfamiliar to most)
- Familiarity correlation with allocation: Strong (familiar = overweight, unfamiliar = underweight)
- Research finding: Investors overweight stocks they are most familiar with (even within US: overweight local companies)
- Consequence: Non-diversified concentrated holdings; higher volatility

**Counterexample: Hedge Funds & Sophisticated Investors (Less Home Bias)**
- Institutional investors: 40-50% international (closer to optimal)
- Retail investors: 80-90% domestic (extreme bias)
- Reason: Professional managers encouraged to find best opportunities globally
- Implications: Home bias primarily retail phenomenon; information access crucial

**Edge Case: Small Country Home Bias (Canada, Israel)**
- Canada: Market cap ~2% of global; Canadian investors hold ~70% domestic
- Israel: Market cap ~0.1% of global; Israeli investors hold ~80% domestic
- Expected (by market cap): Canada 2%, Israel 0.1%
- Home bias ratio: Canada 35×, Israel 800× overweight (!!)
- Explanation: Capital controls, limited international market access historically, currency concerns
- Implication: Home bias extreme in small countries; reflects isolation more than pure behavioral bias

---

## 4. Layer Breakdown

```
Home Bias Architecture:

├─ Magnitude Measurement:
│   ├─ Gross home bias: (Domestic allocation) - (Market cap weight)
│   │   └─ US: 75% - 35% = 40 percentage points overweight domestic
│   │
│   ├─ Relative home bias: (Domestic allocation) / (Market cap weight)
│   │   └─ US: 75% / 35% = 2.14× (114% overweight)
│   │   └─ Japan: 95% / 8% = 11.9× (1,090% overweight)
│   │
│   ├─ By investor type:
│   │   ├─ Retail: 75-90% home bias (extreme)
│   │   ├─ Institutional: 45-60% home bias (moderate)
│   │   ├─ Hedge funds: 30-40% home bias (near-rational)
│   │   └─ Implications: Information access & costs explain much of bias
│   │
│   └─ Geographic pattern:
│       ├─ Large developed (US, Europe, Japan): 20-40% gross bias
│       ├─ Medium-sized (Canada, Australia): 30-50% bias
│       ├─ Small developed (Denmark, Singapore): 40-60% bias
│       └─ Emerging (India, Brazil): 70-90% bias
│
├─ Rational Explanations (NOT purely behavioral):
│   ├─ Market Microstructure Costs:
│   │   ├─ Trading costs higher internationally (wider bid-ask spreads, lower liquidity)
│   │   ├─ FX conversion: 0.1-0.3% cost per round trip
│   │   ├─ Market impact: Large orders move prices more in foreign markets
│   │   └─ Implication: 0.5-1% roundtrip cost for international position = rational deterrent
│   │
│   ├─ Information Asymmetries:
│   │   ├─ Home advantage: Know home market better (local news, companies, investors)
│   │   ├─ Information access: International companies harder to analyze
│   │   ├─ Language barriers: Non-English markets require translation/expertise
│   │   └─ Implication: Home bias reduces info risk; rational if info is costly
│   │
│   ├─ Taxes & Regulations:
│   │   ├─ Domestic tax treatment: Often more favorable (401k access, deductions)
│   │   ├─ International withholding: 10-30% on foreign dividends (depending on treaty)
│   │   ├─ Regulatory complexity: Foreign tax credits, PFIC rules (complexity tax)
│   │   └─ Implication: Tax drag on international holdings 1-3% p.a. after-tax return gap
│   │
│   ├─ Currency Risk:
│   │   ├─ Unhedged international: Adds 5-10% volatility from FX
│   │   ├─ Hedging cost: 1-3% p.a. (high for developed markets)
│   │   └─ Consequence: International allocation seems riskier than domestic (is it?)
│   │       Answer: Partially rational; FX volatility real but natural hedge exists
│   │
│   ├─ Liability Matching (Pension Funds):
│   │   ├─ Domestic liabilities (pension payments in domestic currency)
│   │   ├─ Home bias reduces FX mismatch risk
│   │   └─ May be rational if obligations are domestic-currency-denominated
│   │
│   ├─ Restricted Capital Flows (Emerging Markets):
│   │   ├─ Capital controls: Can't easily move money internationally
│   │   ├─ Non-resident taxation: Punitive for foreign holdings
│   │   └─ Consequence: Extreme home bias forced, not optional
│   │
│   └─ Expected Return Premium (Hidden):
│       ├─ Possibility: Home investors know home market better → expect higher returns
│       ├─ Rational to overweight if higher return justified
│       └─ Evidence: Mixed; hard to prove superior forecasting ability
│
├─ Behavioral Explanations (Pure bias):
│   ├─ Familiarity Bias:
│   │   ├─ Investors overweight familiar companies (know Apple, not Alibaba)
│   │   ├─ Familiarity ≠ lower risk (psychological comfort, not rational risk reduction)
│   │   ├─ Shefrin & Statman: Familiarity reduces perceived risk (not actual risk)
│   │   └─ Consequence: Overweight concentrated local positions
│   │
│   ├─ Availability Heuristic:
│   │   ├─ Media coverage: Home companies get more local news (availability)
│   │   ├─ Investors assume available = low risk (news = transparent; foreign = opaque)
│   │   └─ Implication: Home bias from news bias (media emphasizes local)
│   │
│   ├─ Overconfidence:
│   │   ├─ Investors believe they understand home market better (true)
│   │   ├─ Extrapolate: Misallocate based on confidence (overweight)
│   │   └─ Consequence: Concentrated bets on overconfidence
│   │
│   ├─ Status Quo Bias:
│   │   ├─ Default: Don't rebalance into international (too much effort)
│   │   ├─ Consequence: Inherited portfolios stay domestic
│   │   └─ Implication: Path-dependent; early allocation sticky
│   │
│   ├─ Endowment Effect:
│   │   ├─ Investors overvalue what they own (home stocks = "endowment")
│   │   ├─ Reluctant to sell at true market price
│   │   └─ Consequence: Home bias persistent despite underperformance
│   │
│   └─ Loss Aversion + Regret:
│       ├─ Foreign investment failure: Feel regret ("should have known better")
│       ├─ Home market failure: Feel less regret (understandable, others also fooled)
│       └─ Consequence: Bias toward home (emotional insurance against regret)
│
├─ Empirical Evidence (French & Poterba 1991, Tesar & Werner 1995):
│   ├─ Time-series: Home bias stable since 1980s (structural, not transient)
│   ├─ Cross-country: Larger countries show larger absolute bias (US 40%, Japan 30%, Canada 45%)
│   ├─ Within-country: Local companies overweighted (US investors overweight local stocks 5-10%)
│   ├─ Emerging markets: Extreme bias (90%+ domestic); reflects capital controls, illiquidity
│   └─ Decline hypothesis: Expected to decline with globalization; mostly false (bias persists)
│
├─ Cost of Home Bias:
│   ├─ Portfolio variance: 0.5-1.5% higher due to concentration risk
│   ├─ Diversification gap: Lose 5-10% of diversification benefit (correlation not optimized)
│   ├─ In dollar terms: $500B-$1T globally misallocated
│   ├─ Return drag: Depends on relative performance; 0-2% p.a. range
│   │   └─ Can be negative (home underperforms) or positive (home outperforms luck)
│   │
│   └─ Time-varying cost: Most expensive when home market underperforming (Japanese 1990s)
│       Most beneficial when home market outperforming (US 2010-2024)
│
└─ Gradient of Home Bias (Country Size Effect):
    ├─ Small open economy (Netherlands): Less home bias (30% domestic, market cap ~1%)
    │   └─ Forced to go international; population < Sweden; capital markets tiny
    │
    ├─ Medium developed (Canada, Australia): Moderate home bias (40% domestic, market cap ~2%)
    │   └─ Natural borders; distinct markets; some forced international exposure
    │
    ├─ Large developed (US, Japan, Germany): Significant bias (60-75% domestic)
    │   └─ Large domestic markets; can build diversified portfolio domestically
    │   └─ Less forced international (different from constraints)
    │
    └─ Emerging markets (India, Brazil, Mexico): Extreme bias (80%+ domestic)
        └─ Capital controls + illiquidity + currency risk all amplify home bias
```

**Mathematical Formulas:**

Home Bias Index:
$$HB_i = \frac{w_{i,domestic}}{w_{market,domestic}} - 1$$

Where:
- $w_{i,domestic}$ = investor i's domestic allocation
- $w_{market,domestic}$ = domestic market share of global market cap

Positive value = overweight domestic (home bias), negative = underweight (foreign bias)

For US: $HB = \frac{0.75}{0.35} - 1 = 1.14$ (114% overweight)

Expected return impact (assuming domestic outperformance of α):
$$\Delta R = HB \times \alpha$$

If home bias index = 1.14 and home outperforms by 1% p.a.: Gain = 1.14% p.a.
If home underperforms by 1% p.a.: Loss = 1.14% p.a.

---

## 5. Mini-Project: Home Bias Backtest & Decomposition

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Analyze home bias by downloading US and international equity data

def get_market_cap_data():
    """
    Approximate market cap weights based on index constituents.
    """
    # Approximate market weights (2024 Q1)
    weights_market = {
        'US': 0.35,
        'Developed ex-US': 0.28,
        'Emerging Markets': 0.22,
        'Other': 0.15
    }
    return weights_market


def get_investor_allocations():
    """
    Typical investor home bias allocations.
    """
    allocations = {
        'Rational (Theory)': {'US': 0.35, 'Developed ex-US': 0.28, 'EM': 0.22, 'Other': 0.15},
        'Average Retail': {'US': 0.75, 'Developed ex-US': 0.15, 'EM': 0.08, 'Other': 0.02},
        'Institutional': {'US': 0.55, 'Developed ex-US': 0.25, 'EM': 0.15, 'Other': 0.05},
        'Extreme Home Bias': {'US': 0.90, 'Developed ex-US': 0.07, 'EM': 0.02, 'Other': 0.01},
        'Globally Diversified': {'US': 0.35, 'Developed ex-US': 0.35, 'EM': 0.25, 'Other': 0.05},
    }
    return allocations


def get_regional_returns(start_date, end_date):
    """
    Fetch returns for US, International, and EM equities.
    """
    indices = {
        'SPY': 'US (S&P 500)',
        'EFA': 'Developed ex-US',
        'EEM': 'Emerging Markets',
    }
    
    data = yf.download(list(indices.keys()), start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns, indices


def calculate_portfolio_metrics(returns, allocation):
    """
    Compute risk, return, Sharpe ratio for given allocation.
    """
    # Map allocation to index returns
    portfolio_return_daily = returns['SPY'] * allocation['US'] + \
                            returns['EFA'] * allocation['Developed ex-US'] + \
                            returns['EEM'] * allocation['EM']
    
    annual_return = portfolio_return_daily.mean() * 252
    annual_vol = portfolio_return_daily.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Max drawdown
    cum_returns = (1 + portfolio_return_daily).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'Annual Return': annual_return,
        'Annual Vol': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown
    }


def home_bias_index(allocation, market_weights):
    """
    Calculate home bias index for US investor.
    """
    us_home_bias = allocation['US'] / market_weights['US']
    return us_home_bias


# Main Analysis
print("=" * 100)
print("HOME BIAS ANALYSIS & COST QUANTIFICATION")
print("=" * 100)

# Get data
returns, indices = get_regional_returns('2010-01-01', '2024-01-01')
market_weights = get_market_cap_data()
allocations = get_investor_allocations()

# 1. Market cap weights
print("\n1. MARKET CAPITALIZATION WEIGHTS (Global Portfolio)")
print("-" * 100)
for region, weight in market_weights.items():
    print(f"  {region}: {weight:.1%}")

# 2. Portfolio comparison
print("\n2. PORTFOLIO RISK-RETURN COMPARISON (2010-2024)")
print("-" * 100)

results = {}
for portfolio_name, allocation in allocations.items():
    metrics = calculate_portfolio_metrics(returns, allocation)
    results[portfolio_name] = metrics
    
    hb_index = home_bias_index(allocation, market_weights)
    
    print(f"\n{portfolio_name}:")
    print(f"  Allocation: US={allocation['US']:.1%}, DevEx-US={allocation['Developed ex-US']:.1%}, EM={allocation['EM']:.1%}")
    print(f"  Annual Return: {metrics['Annual Return']:.2%}")
    print(f"  Annual Volatility: {metrics['Annual Vol']:.2%}")
    print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['Max Drawdown']:.2%}")
    print(f"  Home Bias Index: {hb_index:.2f}x (1.0 = rational, >1.0 = home bias)")

# 3. Home bias decomposition
print("\n3. HOME BIAS INDEX DECOMPOSITION")
print("-" * 100)

for portfolio_name, allocation in allocations.items():
    hb_index = home_bias_index(allocation, market_weights)
    gross_bias = allocation['US'] - market_weights['US']
    
    print(f"{portfolio_name}:")
    print(f"  Home Bias Ratio: {hb_index:.2f}x")
    print(f"  Gross Bias: {gross_bias:+.1%} percentage points")
    print(f"  Interpretation: {'Underweight' if hb_index < 1 else 'Overweight'} domestic by {abs(hb_index - 1)*100:.0f}%")

# 4. Cost of home bias (US exposure)
print("\n4. COST OF HOME BIAS (Return differential 2010-2024)")
print("-" * 100)

us_cumulative_return = (1 + returns['SPY']).prod() ** (252 / len(returns)) - 1
international_cumulative_return = (1 + returns['EFA']).prod() ** (252 / len(returns)) - 1

print(f"US Annual Return: {us_cumulative_return:.2%}")
print(f"International Annual Return: {international_cumulative_return:.2%}")
print(f"Outperformance: {(us_cumulative_return - international_cumulative_return):.2%} p.a. (US won 2010-2024)")

# Opportunity cost calculation
retail_allocation = allocations['Average Retail']
rational_allocation = allocations['Rational (Theory)']

us_overweight = retail_allocation['US'] - rational_allocation['US']
return_diff = us_cumulative_return - international_cumulative_return

opportunity_cost = us_overweight * return_diff

print(f"\nRetail home bias cost:")
print(f"  Overweight US: {us_overweight:+.1%}")
print(f"  Return differential: {return_diff:.2%} p.a.")
print(f"  Opportunity gain (US outperformed): {opportunity_cost:.2%} p.a.")
print(f"  NOTE: This is luck! In 1990s EM outperformed; cost would be negative.")

# 5. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Allocation comparison
ax = axes[0, 0]
portfolio_names = list(allocations.keys())
us_allocs = [allocations[p]['US'] for p in portfolio_names]
colors = ['green' if alloc == 0.35 else 'red' if alloc > 0.75 else 'orange' for alloc in us_allocs]

ax.barh(portfolio_names, us_allocs, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(market_weights['US'], color='blue', linestyle='--', linewidth=2, label=f'Market weight ({market_weights["US"]:.1%})')
ax.set_xlabel('US Allocation (%)')
ax.set_title('Home Bias: US Equity Allocation by Portfolio Type', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Plot 2: Home bias index
ax = axes[0, 1]
hb_indices = [home_bias_index(allocations[p], market_weights) for p in portfolio_names]
colors = ['green' if hb < 1.1 else 'orange' if hb < 1.5 else 'red' for hb in hb_indices]

ax.barh(portfolio_names, hb_indices, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(1.0, color='blue', linestyle='--', linewidth=2, label='Rational (1.0x)')
ax.set_xlabel('Home Bias Index (1.0 = rational)')
ax.set_title('Home Bias Magnitude', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Plot 3: Risk-return scatter
ax = axes[0, 2]
for portfolio_name, metrics in results.items():
    color = 'green' if 'Rational' in portfolio_name else 'red' if 'Extreme' in portfolio_name else 'orange'
    ax.scatter(metrics['Annual Vol'], metrics['Annual Return'], s=200, alpha=0.6, label=portfolio_name, color=color)

ax.set_xlabel('Annual Volatility')
ax.set_ylabel('Annual Return')
ax.set_title('Risk-Return Frontier by Portfolio Type', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Sharpe ratios
ax = axes[1, 0]
sharpe_ratios = [results[p]['Sharpe Ratio'] for p in portfolio_names]
colors = ['green' if sr > 0.45 else 'orange' if sr > 0.40 else 'red' for sr in sharpe_ratios]

ax.bar(range(len(portfolio_names)), sharpe_ratios, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(portfolio_names)))
ax.set_xticklabels(portfolio_names, rotation=15, ha='right', fontsize=8)
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Risk-Adjusted Performance', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 5: Cumulative returns by region
ax = axes[1, 1]
cum_us = (1 + returns['SPY']).cumprod()
cum_efa = (1 + returns['EFA']).cumprod()
cum_eem = (1 + returns['EEM']).cumprod()

ax.plot(cum_us.index, cum_us, label='US (SPY)', linewidth=2)
ax.plot(cum_efa.index, cum_efa, label='Dev ex-US (EFA)', linewidth=2)
ax.plot(cum_eem.index, cum_eem, label='EM (EEM)', linewidth=2)

ax.set_title('Cumulative Returns by Region (2010-2024)', fontweight='bold')
ax.set_ylabel('Cumulative Return (1.0 = start)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Volatility comparison
ax = axes[1, 2]
volatilities = [results[p]['Annual Vol'] for p in portfolio_names]
colors = ['green' if vol < 0.13 else 'orange' if vol < 0.14 else 'red' for vol in volatilities]

ax.bar(range(len(portfolio_names)), [v*100 for v in volatilities], color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(portfolio_names)))
ax.set_xticklabels(portfolio_names, rotation=15, ha='right', fontsize=8)
ax.set_ylabel('Annual Volatility (%)')
ax.set_title('Portfolio Volatility Comparison', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('home_bias_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: home_bias_analysis.png")
plt.show()

# 6. Decomposition of home bias costs
print("\n5. HOME BIAS COST DECOMPOSITION")
print("-" * 100)
print(f"""
RATIONAL ALLOCATION (per CAPM):
├─ US: {rational_allocation['US']:.1%}
├─ Int'l Dev: {rational_allocation['Developed ex-US']:.1%}
└─ EM: {rational_allocation['EM']:.1%}

AVERAGE RETAIL ALLOCATION:
├─ US: {retail_allocation['US']:.1%} (overweight by {(retail_allocation['US'] - rational_allocation['US']):.1%})
├─ Int'l Dev: {retail_allocation['Developed ex-US']:.1%} (underweight by {(rational_allocation['Developed ex-US'] - retail_allocation['Developed ex-US']):.1%})
└─ EM: {retail_allocation['EM']:.1%} (underweight by {(rational_allocation['EM'] - retail_allocation['EM']):.1%})

HISTORICAL IMPACT (2010-2024, favorable to US):
├─ US outperformed Int'l by: {(us_cumulative_return - international_cumulative_return):.2%} p.a.
├─ Retail overweight US: {(retail_allocation['US'] - rational_allocation['US']):.1%}
├─ Luck factor: Retail gained {opportunity_cost:.2%} p.a. from outperformance
└─ IMPORTANT: This was luck! In other periods (1990s, 2000-2010), Int'l outperformed

COSTS OF HOME BIAS (Structural):
├─ Diversification loss: 0.5-1.0% higher volatility
├─ Transaction costs: 0.2-0.5% higher (illiquid international positions)
├─ FX hedging/costs: 0.5-1.0% if unhedged volatility considered
├─ Tax inefficiency: 0.2-0.5% from suboptimal tax placement
└─ TOTAL: 1.5-3.0% annual drag (even when US outperforms, home bias costly)

BEHAVIORAL CONTRIBUTORS:
├─ Familiarity bias: ~50% of home bias effect
├─ Information asymmetry: ~25% of effect
├─ Transaction costs: ~15% of effect
├─ Liability matching/currency: ~10% of effect

RECOMMENDATIONS:
├─ Target allocation: 40% US, 35% Dev ex-US, 25% EM (vs current 75/15/8)
├─ Phased approach: Annual 5% increase in international holdings
├─ Use low-cost index funds: Reduces implementation costs
├─ Tax-efficient placement: International in tax-deferred accounts
└─ Periodic rebalancing: Mechanical (not emotional) discipline
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Home Bias Rationality Test:** Given that US transactions cost 0.05% while international cost 0.30%, international FX costs 0.20%, and withholding taxes cost 0.15%, total international friction = 0.65% annually. International equities expected return only 0.5% higher than US. Is home bias rational? At what return differential does international allocation make sense?

2. **Emerging Markets Extreme Bias:** India has extreme home bias (90%+ domestic), yet IPO capital outflows suggest sophisticated capital allocation. Is extreme EM home bias rational (capital controls + illiquidity) or behavioral? Can you design a framework distinguishing the two?

3. **Liability-Driven Home Bias:** Pension fund with €50M obligations (25 years), assets €60M. CAPM says 60% equity globally. Fund holds 75% Euro-denominated assets (home bias). Is this rational or not? How would you advise reallocation?

4. **Familiarity Premium Measurement:** Suppose US investors can invest in either (A) Tata Consultancy (Indian, unfamiliar) or (B) equal-weight mix of 20 Indian companies (more familiar via diversification). Both have identical expected returns/risk. Would investors prefer (B)? Why does familiarity create premium?

5. **Home Bias Correction Path:** Portfolio currently 85% US / 15% international. Cost of shifting to 60/40 is 0.5% one-time transaction cost, but saves 1.5% p.a. ongoing. Should rebalance immediately or gradually over 3 years? What's the break-even timing?

---

## 7. Key References

- **French, K.R. & Poterba, J.M. (1991).** "Investor Diversification and International Equity Markets" – Foundational paper documenting extreme home bias; shows French investors 90% domestic, similar for US and Japan.

- **Tesar, L.L. & Werner, I.M. (1995).** "Home Bias and High Turnover" – Extends evidence; shows home bias greatest among young countries, smaller countries, countries with capital controls.

- **Solnik, B. (1974).** "Why Not Diversify Internationally Rather Than Domestically?" – Shows theoretical gains from international diversification despite home bias costs.

- **Shefrin, H. & Statman, M. (1985).** "The Disposition to Sell Winners Too Early and Ride Losers Too Long: Theory and Evidence" – Familiar (home) positions trigger disposition effect; hold longer when familiar.

- **Coval, J.D. & Moskowitz, T.J. (1999).** "Home Bias at Home: Local Equity Preference in Domestic Portfolios" – Shows US investors overweight local companies even within US (home bias gradient exists within country).

- **Bekaert, G. & Harvey, C.R. (1995).** "Foreign Speculators and Emerging Equity Markets" – EM home bias partly rational (capital controls) but also behavioral (familiarity, information gaps).

- **Investopedia: Home Bias** – https://www.investopedia.com/terms/h/homebias.asp – Accessible overview with examples.

- **Federal Reserve: International Portfolio Investment** – https://www.federalreserve.gov – Data on international portfolio allocation trends.

