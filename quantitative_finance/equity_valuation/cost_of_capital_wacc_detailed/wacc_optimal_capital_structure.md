# Cost of Capital (WACC): Blended Financing Cost Framework

## I. Concept Skeleton

**Definition:** Weighted Average Cost of Capital (WACC) is the blended discount rate reflecting the cost of all capital sources (debt + equity) weighted by their market values. It represents the minimum return a company must earn to satisfy both creditors (debt) and shareholders (equity).

**Purpose:** Discount cash flows in DCF valuation, evaluate investment returns (hurdle rate for projects), assess financial leverage impact, benchmark capital structure efficiency, and compare firms across different financing mixes.

**Prerequisites:** Risk-free rate (Rf), equity risk premium, beta estimation, cost of debt, tax effects, capital structure optimization, CAPM framework.

---

## II. Comparative Framing

| **Component** | **Cost of Equity** | **Cost of Debt** | **WACC (Blended)** | **Hurdle Rate** | **IRR** |
|-----------|----------|----------|----------|----------|-----------|
| **Definition** | Required return for equity holders | Interest rate on borrowing | Blended cost of all capital | Min required return for projects | Actual return achieved |
| **Calculation** | CAPM: Rf + β(Rm - Rf) | Interest expense / Debt | (E/V)×Re + (D/V)×Rd×(1-T) | Project-specific, risk-adjusted | Solve for IRR in NPV=0 |
| **Inputs** | Risk-free rate, beta, market premium | Interest rates, spreads | Cost of equity, cost of debt, tax, leverage | WACC + risk premium | Cash flows |
| **Range** | 8-15% typical | 2-8% typical | 6-12% typical | 15-25% for high-risk | Variable per project |
| **Use** | DCF discount rate (primary) | Debt pricing, refinancing | Project evaluation, firm valuation | High-risk project screening | Performance measurement |
| **Leverage Sensitive** | Yes (increases with debt) | No (direct borrowing cost) | Yes (debt weight + tax shield) | Yes (risk-adjusted) | No (realized return) |
| **Tax Impact** | No explicit tax (in Rd) | Yes (tax shield: interest deductible) | Yes (tax shield reduces cost) | Varies (after-tax assumed) | After-tax IRR vs pre-tax |

---

## III. Examples & Counterexamples

### Example 1: WACC Impact on Valuation - The Leverage Trap
**Setup:**
- Company: $100M EBITDA, 20% tax rate
- Two capital structures:
  - Structure A (Conservative): 30% debt, 70% equity
  - Structure B (Aggressive): 60% debt, 40% equity
- Task: Calculate WACC and resulting valuations

**Scenario A: Conservative Capital Structure**

```
Inputs:
├─ Risk-free rate: 3%
├─ Beta (unlevered): 1.0
├─ Market risk premium: 6%
├─ Beta (levered at 30% debt): 1.2 (increased leverage slightly)
├─ Cost of equity: 3% + 1.2 × 6% = 10.2%
├─ Cost of debt: 4% (good credit, low rates)
└─ Tax rate: 20%

Capital Structure:
├─ Equity weight (E/V): 70%
├─ Debt weight (D/V): 30%
└─ Debt value: $100M / 0.9 × 0.3 = $33M

WACC Calculation:
WACC = 0.70 × 10.2% + 0.30 × 4% × (1 - 0.20)
     = 7.14% + 0.96%
     = 8.1%

Valuation (perpetuity):
├─ Operating CF: $100M × (1 - 20%) = $80M (approx, NOPAT)
├─ Enterprise Value: $80M / 8.1% = $987M
├─ Less debt: $33M
└─ Equity value: $954M
```

**Scenario B: Aggressive Capital Structure**

```
Inputs:
├─ Risk-free rate: 3%
├─ Beta (unlevered): 1.0
├─ Market risk premium: 6%
├─ Beta (levered at 60% debt): 1.8 (much higher leverage)
├─ Cost of equity: 3% + 1.8 × 6% = 13.8%
├─ Cost of debt: 6% (riskier, higher rates due to leverage)
└─ Tax rate: 20%

Capital Structure:
├─ Equity weight (E/V): 40%
├─ Debt weight (D/V): 60%
└─ Debt value: $100M / 0.4 × 0.6 = $150M

WACC Calculation:
WACC = 0.40 × 13.8% + 0.60 × 6% × (1 - 0.20)
     = 5.52% + 2.88%
     = 8.4%

Valuation (perpetuity):
├─ Operating CF: $80M (same NOPAT)
├─ Enterprise Value: $80M / 8.4% = $952M
├─ Less debt: $150M
└─ Equity value: $802M
```

**Comparison:**
- Conservative structure: Equity value $954M, debt $33M, total $987M
- Aggressive structure: Equity value $802M, debt $150M, total $952M
- Issue: WACC slightly higher with leverage (8.4% vs 8.1%), but equity returns drop sharply (debt holders capture value)

**Key insight:**
- Increasing debt reduces WACC slightly (tax shield benefit)
- BUT equity value drops (equity holders bear higher risk)
- Optimal structure = minimize WACC, not maximize equity value

---

### Example 2: Beta Levering/Unlevering - Risk Adjustment for Different Capital Structures
**Setup:**
- Comparable company analysis: Peer has 40% debt
- Peer's levered beta: 1.2
- Target company has 50% debt (more leveraged)
- Task: Adjust beta for target's leverage

**Scenario A: Wrong Approach (Use Peer Beta Directly)**

```
Peer beta (at 40% debt): 1.2
Cost of equity for target: 3% + 1.2 × 6% = 10.2%
```

**Problem:** Doesn't account for target's different leverage (50% vs 40%).

**Scenario B: Correct Approach (Unlever, Re-lever)**

```
Step 1: Unlever peer's beta (remove debt effect)
Formula: βunlevered = βlevered / [1 + (1-T) × D/E]

Peer's D/E ratio: 40% / 60% = 0.667
βunlevered = 1.2 / [1 + (1 - 0.20) × 0.667]
          = 1.2 / [1 + 0.534]
          = 1.2 / 1.534
          = 0.78

Step 2: Re-lever for target's capital structure
Target's D/E ratio: 50% / 50% = 1.0

βlevered(target) = 0.78 × [1 + (1 - 0.20) × 1.0]
                 = 0.78 × [1 + 0.80]
                 = 0.78 × 1.80
                 = 1.40

Cost of equity (target): 3% + 1.40 × 6% = 11.4%
```

**Comparison:**
- Wrong approach: 10.2% cost of equity
- Correct approach: 11.4% cost of equity
- Difference: 120 bps! (significant impact on valuation)

**Lesson:** Must adjust beta for different leverage; unlevering/relevering essential for proper comparables.

---

### Example 3: Tax Shield Effect - Debt Benefit Illusion
**Setup:**
- Company considering debt issuance
- Current: All equity, WACC = 10%
- Proposal: Issue $50M debt at 5%, use to buy back equity
- Question: Does WACC improve?

**Before Debt:**
```
Market cap: $500M (all equity)
WACC = Cost of equity = 10%
```

**After Debt Issuance:**
```
Debt: $50M at 5%
Equity (post-buyback): $450M
Tax rate: 20%

New cost of equity (higher leverage, higher beta):
Old beta: 1.0
New D/E: $50M / $450M = 0.111
New beta = 1.0 × [1 + 0.8 × 0.111] = 1.089

Cost of equity: 3% + 1.089 × 6% = 9.534%

New WACC:
WACC = (450/500) × 9.534% + (50/500) × 5% × (1 - 0.20)
     = 0.90 × 9.534% + 0.10 × 4%
     = 8.58% + 0.40%
     = 8.98%
```

**Tax Shield Benefit:**
```
Before: WACC = 10%
After: WACC = 8.98%
Savings: 102 bps

But watch: Cost of equity rose from 10% to 9.534% (actually fell because less equity risk),
WACC fell due to:
1. Tax shield (interest tax-deductible): +(interest benefit)
2. Leverage effect on beta: +/- (mixed)
Net: WACC improved by ~100 bps
```

**Lesson:** Debt tax shield DOES improve WACC (up to optimal point), but doesn't increase total firm value—it redistributes value from equity to debt holders.

---

## IV. Layer Breakdown

```
WACC CALCULATION FRAMEWORK

┌──────────────────────────────────────────────────┐
│          WEIGHTED AVERAGE COST OF CAPITAL         │
│                                                   │
│ Core: WACC = (E/V)×Re + (D/V)×Rd×(1-T)          │
│       Where E,D = market values of equity, debt   │
│             V = E + D (total capital)            │
│             Re = cost of equity                  │
│             Rd = cost of debt                    │
│             T = tax rate                         │
└────────────────────┬─────────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  1. COST OF EQUITY (Re) - CAPM            │
    │                                            │
    │  Formula: Re = Rf + β × (Rm - Rf)         │
    │  where:                                    │
    │  ├─ Rf = Risk-free rate (gov bonds)       │
    │  ├─ β = Beta (systematic risk)            │
    │  ├─ Rm = Market return                    │
    │  └─ (Rm - Rf) = Market risk premium       │
    │                                            │
    │  Step 1: Risk-Free Rate (Rf)               │
    │  ├─ Use 10-year Treasury yield             │
    │  ├─ Current example: 3%                    │
    │  ├─ Why 10Y? Matches DCF horizon (~10yr)  │
    │  └─ Varies: 2-5% historically              │
    │                                            │
    │  Step 2: Market Risk Premium (MRP)         │
    │  ├─ Historical (1926-2024): 6-7%          │
    │  ├─ Forward-looking: 5-6%                  │
    │  ├─ Academically: 5-6% consensus          │
    │  └─ Sensitivity: +1% MRP = +1% Re         │
    │                                            │
    │  Step 3: Beta (β) - Measure of Volatility  │
    │  ├─ Beta = Cov(Stock, Market) / Var(Mkt)  │
    │  ├─ Market beta: 1.0 (by definition)       │
    │  ├─ Low beta (0.5-0.8): Defensive         │
    │  │  └─ Utilities, staples (stable)         │
    │  ├─ Medium beta (1.0-1.3): Typical        │
    │  │  └─ Industrials, financials             │
    │  └─ High beta (1.3-2.0+): Volatile        │
    │     └─ Tech, biotech (high growth)         │
    │                                            │
    │  Levered vs Unlevered Beta:                 │
    │  ├─ Levered = includes debt effect        │
    │  ├─ Unlevered = pure business risk        │
    │  ├─ Unlevered β = βlevered / [1+(D/E)×(1-T)]
    │  └─ Need to adjust for different capital   │
    │     structures (see Example 2)             │
    │                                            │
    │  Example CAPM Calculation:                  │
    │  ├─ Rf = 3%                                │
    │  ├─ β = 1.2 (company is 20% riskier)      │
    │  ├─ MRP = 6%                               │
    │  └─ Re = 3% + 1.2 × 6% = 10.2%            │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  2. COST OF DEBT (Rd) - Interest Rates     │
    │                                            │
    │  Definition: Effective interest rate       │
    │  paid on company's debt                    │
    │                                            │
    │  Calculation Methods:                       │
    │  ├─ Direct: Interest expense / Total debt  │
    │  ├─ Yield-to-maturity (YTM) on bonds      │
    │  ├─ Credit spread + risk-free rate        │
    │  └─ Average blended rate (weighted avg)    │
    │                                            │
    │  Determinants of Rd:                       │
    │  ├─ Risk-free rate (base): 3%             │
    │  ├─ + Credit spread (company risk):        │
    │  │  ├─ AAA rated: +50-100 bps             │
    │  │  ├─ A rated: +100-150 bps              │
    │  │  ├─ BBB rated: +150-250 bps            │
    │  │  ├─ B rated: +250-500 bps              │
    │  │  └─ CCC rated: +500-1000+ bps          │
    │  ├─ = Total Rd                             │
    │                                            │
    │  Example:                                   │
    │  ├─ A-rated company                        │
    │  ├─ Risk-free: 3%                          │
    │  ├─ Credit spread: 125 bps                 │
    │  └─ Cost of debt: 4.25%                    │
    │                                            │
    │  Tax Shield Effect:                         │
    │  ├─ Interest is tax-deductible             │
    │  ├─ Effective cost = Rd × (1 - Tax rate)  │
    │  ├─ Example: 4.25% × (1 - 0.20) = 3.4%   │
    │  │  (20% tax rate reduces cost by ~1%)     │
    │  └─ Key: After-tax cost is what matters    │
    │                                             │
    │  Note on Municipal Bonds:                  │
    │  └─ Tax-exempt, so Rd already after-tax    │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  3. MARKET VALUE WEIGHTS (E/V, D/V)        │
    │                                            │
    │  Critical: Use MARKET values, NOT          │
    │  book values!                              │
    │                                            │
    │  Market values:                             │
    │  ├─ Equity (E): Current stock price ×     │
    │  │             shares outstanding         │
    │  │  └─ Changes daily (market sentiment)    │
    │  ├─ Debt (D): Fair value of bonds/loans  │
    │  │  └─ Usually close to book value        │
    │  └─ V = E + D                              │
    │                                            │
    │  Common MISTAKE: Use book equity           │
    │  ├─ Book value = historical cost basis     │
    │  ├─ Vastly different from market value    │
    │  ├─ Example: Apple book equity ~$50B,     │
    │  │           market cap ~$3T (60x!)       │
    │  └─ Always use market value for WACC      │
    │                                            │
    │  Example:                                   │
    │  ├─ Market cap (E): $500M                  │
    │  ├─ Market value debt (D): $200M          │
    │  ├─ Total (V): $700M                      │
    │  ├─ E/V: 500/700 = 71.4%                  │
    │  └─ D/V: 200/700 = 28.6%                  │
    └────────────────┬──────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────┐
    │  4. FINAL WACC CALCULATION                │
    │                                            │
    │  All components assembled:                 │
    │  ├─ Re (cost of equity): 10.2%             │
    │  ├─ Rd (cost of debt): 4.25%               │
    │  ├─ Tax rate: 20%                          │
    │  ├─ E/V: 71.4%                             │
    │  ├─ D/V: 28.6%                             │
    │  │                                          │
    │  │ WACC = (E/V)×Re + (D/V)×Rd×(1-T)       │
    │  │ WACC = 0.714 × 10.2% + 0.286 × 4.25%   │
    │  │        × (1 - 0.20)                     │
    │  │ WACC = 7.28% + 0.97%                    │
    │  │ WACC = 8.25%                            │
    │  │                                          │
    │  └─ This is the discount rate for DCF!     │
    │                                            │
    │  Sensitivity Analysis:                     │
    │  If Rf rises 1%:                           │
    │  ├─ Re = 3% + 1% + 1.2 × 6% = 11.2%       │
    │  ├─ New WACC = 0.714 × 11.2% + 0.286 ×    │
    │  │  4.25% × 0.80 = 8.99%                  │
    │  └─ WACC rises ~74 bps (significant!)     │
    │                                            │
    │  If Tax rate rises to 25%:                 │
    │  ├─ New tax shield = 4.25% × (1 - 0.25) = │
    │  │  3.19%                                  │
    │  ├─ New WACC = 7.28% + 0.286 × 3.19% =   │
    │  │  = 8.19%                               │
    │  └─ WACC falls ~6 bps (minor)             │
    └──────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### WACC Formula

$$\text{WACC} = \left(\frac{E}{V}\right) R_e + \left(\frac{D}{V}\right) R_d (1-T)$$

where:
- $E$ = market value of equity
- $D$ = market value of debt
- $V = E + D$ = total enterprise value
- $R_e$ = cost of equity (CAPM)
- $R_d$ = cost of debt (before-tax)
- $T$ = corporate tax rate

### Cost of Equity (CAPM)

$$R_e = R_f + \beta (R_m - R_f)$$

where:
- $R_f$ = risk-free rate
- $\beta$ = systematic risk (levered)
- $R_m - R_f$ = market risk premium

### Levered vs Unlevered Beta

$$\beta_{\text{levered}} = \beta_{\text{unlevered}} \left[1 + (1-T) \frac{D}{E}\right]$$

$$\beta_{\text{unlevered}} = \frac{\beta_{\text{levered}}}{1 + (1-T) \frac{D}{E}}$$

---

## VI. Python Mini-Project: WACC Calculation & Sensitivity Engine

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# WACC CALCULATION ENGINE
# ============================================================================

class WACCCalculator:
    """
    Calculate WACC and perform sensitivity analysis
    """
    
    def __init__(self, market_cap, debt_value, rf, beta, mrp, 
                 cost_of_debt, tax_rate):
        """
        market_cap: Equity market value ($M)
        debt_value: Debt market value ($M)
        rf: Risk-free rate (%)
        beta: Levered beta
        mrp: Market risk premium (%)
        cost_of_debt: Cost of debt before tax (%)
        tax_rate: Corporate tax rate (%)
        """
        self.market_cap = market_cap
        self.debt_value = debt_value
        self.rf = rf / 100.0
        self.beta = beta
        self.mrp = mrp / 100.0
        self.cost_of_debt = cost_of_debt / 100.0
        self.tax_rate = tax_rate / 100.0
    
    def calc_cost_of_equity(self):
        """Calculate cost of equity using CAPM"""
        re = self.rf + self.beta * self.mrp
        return re
    
    def calc_enterprise_value(self):
        """Calculate total enterprise value"""
        return self.market_cap + self.debt_value
    
    def calc_weights(self):
        """Calculate market value weights"""
        v = self.calc_enterprise_value()
        e_weight = self.market_cap / v
        d_weight = self.debt_value / v
        return {'E/V': e_weight, 'D/V': d_weight}
    
    def calc_wacc(self):
        """Calculate WACC"""
        re = self.calc_cost_of_equity()
        weights = self.calc_weights()
        
        rd_aftertax = self.cost_of_debt * (1 - self.tax_rate)
        
        wacc = weights['E/V'] * re + weights['D/V'] * rd_aftertax
        
        return {
            're': re,
            'rd_pretax': self.cost_of_debt,
            'rd_aftertax': rd_aftertax,
            'e_weight': weights['E/V'],
            'd_weight': weights['D/V'],
            'wacc': wacc
        }
    
    def leverage_beta(self, unlevered_beta):
        """Relever beta for given capital structure"""
        de_ratio = self.debt_value / self.market_cap
        levered = unlevered_beta * (1 + (1 - self.tax_rate) * de_ratio)
        return levered
    
    def unlever_beta(self, levered_beta=None):
        """Unlever beta to get asset beta"""
        if levered_beta is None:
            levered_beta = self.beta
        
        de_ratio = self.debt_value / self.market_cap
        unlevered = levered_beta / (1 + (1 - self.tax_rate) * de_ratio)
        return unlevered


class SensitivityAnalysis:
    """
    WACC sensitivity to key variables
    """
    
    def __init__(self, base_wacc_calc):
        self.base = base_wacc_calc
    
    def sensitivity_to_rf(self, rf_range=np.array([1, 2, 3, 4, 5])):
        """Sensitivity to risk-free rate"""
        results = []
        
        for rf in rf_range:
            calc = WACCCalculator(
                self.base.market_cap, self.base.debt_value,
                rf, self.base.beta, self.base.mrp * 100,
                self.base.cost_of_debt * 100, self.base.tax_rate * 100
            )
            result = calc.calc_wacc()
            results.append({
                'Variable': f'Rf = {rf}%',
                'WACC': result['wacc'] * 100,
                'Re': result['re'] * 100
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_to_beta(self, beta_range=np.array([0.8, 1.0, 1.2, 1.4, 1.6])):
        """Sensitivity to beta"""
        results = []
        
        for beta in beta_range:
            calc = WACCCalculator(
                self.base.market_cap, self.base.debt_value,
                self.base.rf * 100, beta, self.base.mrp * 100,
                self.base.cost_of_debt * 100, self.base.tax_rate * 100
            )
            result = calc.calc_wacc()
            results.append({
                'Variable': f'β = {beta}',
                'WACC': result['wacc'] * 100,
                'Re': result['re'] * 100
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_to_leverage(self, debt_range=np.array([0, 100, 200, 300, 400])):
        """Sensitivity to debt level (impacts beta)"""
        results = []
        unlevered_beta = self.base.unlever_beta()
        
        for debt in debt_range:
            calc = WACCCalculator(
                self.base.market_cap, debt,
                self.base.rf * 100, unlevered_beta, self.base.mrp * 100,
                self.base.cost_of_debt * 100, self.base.tax_rate * 100
            )
            # Relever beta for new leverage
            calc.beta = calc.leverage_beta(unlevered_beta)
            
            result = calc.calc_wacc()
            results.append({
                'Debt($M)': debt,
                'D/V': result['d_weight'],
                'Beta': calc.beta,
                'WACC': result['wacc'] * 100,
                'Re': result['re'] * 100
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_to_tax(self, tax_range=np.array([15, 20, 25, 30])):
        """Sensitivity to tax rate (affects tax shield)"""
        results = []
        
        for tax in tax_range:
            calc = WACCCalculator(
                self.base.market_cap, self.base.debt_value,
                self.base.rf * 100, self.base.beta, self.base.mrp * 100,
                self.base.cost_of_debt * 100, tax
            )
            result = calc.calc_wacc()
            results.append({
                'Tax Rate': f'{tax}%',
                'Tax Shield': (self.base.cost_of_debt * (1 - tax/100)) * 100,
                'WACC': result['wacc'] * 100
            })
        
        return pd.DataFrame(results)


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("WACC CALCULATION & SENSITIVITY ANALYSIS")
print("="*80)

# Base case company
market_cap = 500  # $500M equity
debt_value = 200  # $200M debt
rf = 3.0  # 3% risk-free
beta = 1.2  # 20% more volatile than market
mrp = 6.0  # 6% market risk premium
cost_of_debt = 4.25  # 4.25% before tax
tax_rate = 20.0  # 20% corporate tax

print(f"\nBase Case Inputs:")
print(f"├─ Market cap (equity): ${market_cap}M")
print(f"├─ Debt value: ${debt_value}M")
print(f"├─ Enterprise value: ${market_cap + debt_value}M")
print(f"├─ Risk-free rate: {rf}%")
print(f"├─ Beta (levered): {beta}")
print(f"├─ Market risk premium: {mrp}%")
print(f"├─ Cost of debt (pre-tax): {cost_of_debt}%")
print(f"└─ Tax rate: {tax_rate}%")

# Calculate WACC
wacc_calc = WACCCalculator(market_cap, debt_value, rf, beta, mrp,
                           cost_of_debt, tax_rate)
result = wacc_calc.calc_wacc()

print(f"\nWACC Calculation:")
print(f"├─ Cost of equity (CAPM):")
print(f"│  Re = {rf}% + {beta} × {mrp}% = {result['re']*100:.2f}%")
print(f"├─ Cost of debt (after-tax):")
print(f"│  Rd(after-tax) = {cost_of_debt}% × (1 - {tax_rate}%) = {result['rd_aftertax']*100:.2f}%")
print(f"├─ Market value weights:")
print(f"│  E/V = {result['e_weight']:.1%}")
print(f"│  D/V = {result['d_weight']:.1%}")
print(f"├─")
print(f"└─ WACC = {result['e_weight']:.1%} × {result['re']*100:.2f}% + {result['d_weight']:.1%} × {result['rd_aftertax']*100:.2f}%")
print(f"        = {result['wacc']*100:.2f}%")

# Sensitivity analysis
sensitivity = SensitivityAnalysis(wacc_calc)

print(f"\n" + "="*80)
print(f"SENSITIVITY ANALYSIS")
print(f"="*80)

print(f"\n1. Sensitivity to Risk-Free Rate:")
rf_sens = sensitivity.sensitivity_to_rf(np.array([1, 2, 3, 4, 5]))
print(rf_sens.to_string(index=False))

print(f"\n2. Sensitivity to Beta:")
beta_sens = sensitivity.sensitivity_to_beta(np.array([0.6, 0.9, 1.2, 1.5, 1.8]))
print(beta_sens.to_string(index=False))

print(f"\n3. Sensitivity to Leverage (Debt Level):")
debt_sens = sensitivity.sensitivity_to_leverage(np.array([0, 100, 200, 300, 400, 500]))
print(debt_sens.to_string(index=False))

print(f"\n4. Sensitivity to Tax Rate:")
tax_sens = sensitivity.sensitivity_to_tax(np.array([15, 20, 25, 30, 35]))
print(tax_sens.to_string(index=False))

# Beta unlever/relever example
print(f"\n" + "="*80)
print(f"BETA UNLEVER/RELEVER EXAMPLE")
print(f"="*80)

unlevered_beta = wacc_calc.unlever_beta()
print(f"\nUnlevered beta (asset beta):")
print(f"├─ Current levered beta: {beta}")
print(f"├─ Current D/E ratio: {debt_value / market_cap:.2f}")
print(f"├─ Tax rate: {tax_rate}%")
print(f"└─ Unlevered beta = {beta} / [1 + (1 - {tax_rate/100}) × {debt_value/market_cap:.2f}]")
print(f"                 = {unlevered_beta:.3f}")

# Re-lever for different capital structures
print(f"\nRe-lever for different capital structures:")
capital_structures = [
    ("Conservative (20% debt)", market_cap * 1.2, market_cap * 0.2),
    ("Current (40% debt)", market_cap * 0.6, market_cap * 0.4),
    ("Aggressive (60% debt)", market_cap * 0.4, market_cap * 0.6),
]

for desc, cap, debt in capital_structures:
    de_ratio = debt / cap
    levered_new = unlevered_beta * (1 + (1 - tax_rate/100) * de_ratio)
    
    calc_new = WACCCalculator(cap, debt, rf, levered_new, mrp,
                              cost_of_debt, tax_rate)
    result_new = calc_new.calc_wacc()
    
    print(f"\n{desc}:")
    print(f"├─ D/E ratio: {de_ratio:.2f}")
    print(f"├─ Levered beta: {levered_new:.3f}")
    print(f"├─ Cost of equity: {result_new['re']*100:.2f}%")
    print(f"└─ WACC: {result_new['wacc']*100:.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: Sensitivity to Rf
ax1 = axes[0, 0]
rf_range = np.array([1, 2, 3, 4, 5, 6])
rf_sens_full = sensitivity.sensitivity_to_rf(rf_range)

ax1.plot(rf_range, rf_sens_full['WACC'], marker='o', linewidth=2.5, 
        markersize=8, color='blue', label='WACC')
ax1.plot(rf_range, rf_sens_full['Re'], marker='s', linewidth=2.5,
        markersize=8, color='red', label='Cost of Equity')
ax1.axvline(x=3, color='green', linestyle='--', alpha=0.5, label='Base case (3%)')
ax1.set_xlabel('Risk-Free Rate (%)')
ax1.set_ylabel('Rate (%)')
ax1.set_title('Panel 1: Sensitivity to Risk-Free Rate')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Sensitivity to Beta
ax2 = axes[0, 1]
beta_range = np.array([0.6, 0.9, 1.2, 1.5, 1.8])
beta_sens_full = sensitivity.sensitivity_to_beta(beta_range)

ax2.plot(beta_range, beta_sens_full['WACC'], marker='o', linewidth=2.5,
        markersize=8, color='blue', label='WACC')
ax2.plot(beta_range, beta_sens_full['Re'], marker='s', linewidth=2.5,
        markersize=8, color='red', label='Cost of Equity')
ax2.axvline(x=1.2, color='green', linestyle='--', alpha=0.5, label='Base case (1.2)')
ax2.set_xlabel('Beta')
ax2.set_ylabel('Rate (%)')
ax2.set_title('Panel 2: Sensitivity to Beta')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Sensitivity to Leverage
ax3 = axes[1, 0]
debt_range_full = np.array([0, 100, 200, 300, 400, 500])
debt_sens_full = sensitivity.sensitivity_to_leverage(debt_range_full)

ax3_twin = ax3.twinx()

ax3.plot(debt_range_full, debt_sens_full['WACC'], marker='o', linewidth=2.5,
        markersize=8, color='blue', label='WACC')
ax3_twin.plot(debt_range_full, debt_sens_full['Beta'], marker='s', linewidth=2.5,
             markersize=8, color='red', label='Levered Beta')

ax3.axvline(x=200, color='green', linestyle='--', alpha=0.5, label='Base case ($200M)')
ax3.set_xlabel('Debt ($M)')
ax3.set_ylabel('WACC (%)', color='blue')
ax3_twin.set_ylabel('Beta', color='red')
ax3.set_title('Panel 3: Sensitivity to Leverage')
ax3.tick_params(axis='y', labelcolor='blue')
ax3_twin.tick_params(axis='y', labelcolor='red')
ax3.grid(True, alpha=0.3)

# Panel 4: Waterfall - WACC components
ax4 = axes[1, 1]

components = ['Risk-Free\nRate', '+ Market\nRisk\n(Beta×MRP)', '= Cost of\nEquity', 
              'Tax Shield\nBenefit', '= WACC']
values = [
    result['re'] * 100,
    result['wacc'] * 100,
    0,
    0,
    result['wacc'] * 100
]

y_pos = 0
cumulative = 0

# Rf
ax4.bar(0, result['rf'] * 100, color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.text(0, result['rf']*100/2, f'{result["rf"]*100:.2f}%', ha='center', va='center', 
        fontweight='bold', color='white', fontsize=9)

# +Beta*MRP
rf_level = result['rf'] * 100
ax4.bar(1, result['re']*100 - result['rf']*100, bottom=rf_level, color='red', 
       alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.text(1, rf_level + (result['re']*100 - result['rf']*100)/2, 
        f'+{(result["re"]*100 - result["rf"]*100):.2f}%', 
        ha='center', va='center', fontweight='bold', color='white', fontsize=9)

# =Re
ax4.bar(2, result['re'] * 100, color='darkblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.text(2, result['re']*100/2, f'{result["re"]*100:.2f}%', ha='center', va='center',
        fontweight='bold', color='white', fontsize=9)

# Tax shield
tax_benefit = (result['rd_pretax'] - result['rd_aftertax']) * 100 * result['d_weight']
ax4.bar(3, -tax_benefit, color='green', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.text(3, -tax_benefit/2, f'-{tax_benefit:.2f}%', ha='center', va='center',
        fontweight='bold', color='white', fontsize=9)

# =WACC
ax4.bar(4, result['wacc'] * 100, color='darkgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.text(4, result['wacc']*100/2, f'{result["wacc"]*100:.2f}%', ha='center', va='center',
        fontweight='bold', color='white', fontsize=9)

ax4.set_xticks(range(5))
ax4.set_xticklabels(components, fontsize=9)
ax4.set_ylabel('Rate (%)')
ax4.set_title('Panel 4: WACC Component Waterfall')
ax4.set_ylim(0, 12)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('wacc_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• WACC most sensitive to cost of equity (higher beta = higher WACC)")
print("• Tax shield lowers WACC: 1% tax rate increase = ~0.5-1 bps WACC reduction")
print("• Optimal leverage minimizes WACC but increases equity risk (beta rises)")
print("• Risk-free rate changes flow through CAPM to Re and ultimately WACC")
print("• Market value weights matter: equity weight dominates WACC (typically 60-80%)")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Damodaran, A. (2012).** "Investment Valuation: Tools and Techniques." 3rd ed.
   - WACC calculation; beta estimation; optimal capital structure

2. **Brealey, R., Myers, S., & Allen, F. (2020).** "Principles of Corporate Finance." 13th ed.
   - Cost of capital framework; CAPM; leverage effects

3. **Hamada, R. S. (1972).** "The effect of the firm's capital structure on the systematic risk of common stocks." Journal of Finance, 27(2), 435-452.
   - Beta and leverage relationship; unlevering/relevering mechanics

**Key Design Concepts:**

- **Market Values Essential:** Always use current market values (not book), as they represent real capital contributors.
- **Tax Shield Benefit:** Debt tax deductibility reduces WACC, but doesn't increase firm value (redistributes to debt holders).
- **Beta Adjustment Critical:** Comps must adjust betas for different capital structures; ignore this = valuation error.
- **Optimal WACC:** Balances debt (lower cost but higher risk) with equity (higher cost but lower risk); minimum typically 30-50% debt.

