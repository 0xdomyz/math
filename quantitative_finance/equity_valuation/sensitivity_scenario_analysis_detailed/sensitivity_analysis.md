# Sensitivity & Scenario Analysis: Testing Valuation Robustness

## I. Concept Skeleton

**Definition:** Sensitivity analysis quantifies how valuation changes with small variations in key assumptions (discount rate, growth rate, terminal margin). Scenario analysis combines multiple assumption changes (base case, upside, downside) to assess valuation range under different business outcomes.

**Purpose:** Identify which assumptions most impact valuation, stress-test DCF robustness, communicate valuation uncertainty to investors, develop decision frameworks under uncertainty, and validate whether assumptions are realistic or driving valuation.

**Prerequisites:** DCF model fundamentals, probability distributions, enterprise value calculation, terminal value concepts, cost of capital, operating margins, capex requirements.

---

## II. Comparative Framing

| **Analysis Type** | **Method** | **Use Case** | **Output** | **Interpretation** |
|-----------|----------|----------|----------|----------|
| **One-Way Sensitivity** | Change 1 variable, hold others constant | Identify key value drivers | Table/chart showing EV by variable | Which 1-2 assumptions dominate? |
| **Two-Way Sensitivity** | Change 2 variables simultaneously | Analyze interdependencies | Matrix showing EV combinations | Which pairs create distress scenarios? |
| **Scenario Analysis** | Define named scenarios (bear/base/bull) | Communicate risk to board | Range of EV with probabilities | Most likely outcome range? |
| **Monte Carlo Simulation** | Randomly sample assumptions, run 1000+ times | Quantify tail risk | Distribution of outcomes with percentiles | What's probability of loss? |
| **Tornado Chart** | Rank assumptions by impact | Prioritize due diligence | Sorted bar chart of EV sensitivity | Which risks to hedge/investigate? |

---

## III. Examples & Counterexamples

### Example 1: One-Way Sensitivity - Discount Rate Impact

**Setup:**
- Base case valuation: Enterprise Value = $500M
- WACC assumption: 8% (cost of capital)
- Terminal growth rate: 3% (perpetual)
- Key question: How much does valuation change if WACC is wrong?

**Sensitivity Calculation:**

```
Perpetuity formula: EV = FCF / (WACC - g)

Base case (WACC 8%):
├─ EV = $100M / (8% - 3%) = $100M / 5% = $2,000M
└─ Valuation at 8%: $2,000M

WACC sensitivity:

WACC 6%:
├─ EV = $100M / (6% - 3%) = $100M / 3% = $3,333M
├─ Change vs base: +67%
└─ Interpretation: Lower risk → higher value (more stable cash flow)

WACC 7%:
├─ EV = $100M / (7% - 3%) = $100M / 4% = $2,500M
├─ Change vs base: +25%
└─ Why: Only 100 bps lower, but creates larger multiple

WACC 8% (base):
├─ EV = $2,000M
└─ Change vs base: 0%

WACC 9%:
├─ EV = $100M / (9% - 3%) = $100M / 6% = $1,667M
├─ Change vs base: -17%
└─ Why: Only 100 bps higher, cuts value by ~$330M

WACC 10%:
├─ EV = $100M / (10% - 3%) = $100M / 7% = $1,429M
├─ Change vs base: -29%
└─ Higher discount rate reduces present value significantly
```

**Key Insight:**

```
Valuation sensitivity is ASYMMETRIC around WACC

Why?
├─ Formula: EV = FCF / (WACC - g)
│  └─ Denominator shrinks as WACC approaches g
├─ As WACC → 3% (close to g), EV → ∞ (infinite sensitivity)
├─ As WACC increases (away from g), sensitivity decreases
│
└─ Practical: ±1% WACC change creates ±20-30% valuation range
   (but asymmetric: downside worse than upside)
```

**Counterexample: Over-Reliance on Base Case**

```
Analyst presents: "Fair value $2,000M"

Problem:
├─ Ignores that WACC has 200 bps range (6-10%)
├─ Actual valuation could be $1,429M - $3,333M
├─ 230% spread, not point estimate
│
└─ Better presentation:
   ├─ Base case (8% WACC): $2,000M
   ├─ Bull case (6% WACC): $3,333M
   ├─ Bear case (10% WACC): $1,429M
   └─ Fair value range: $1,400M - $3,300M
```

---

### Example 2: Two-Way Sensitivity - WACC × Growth Rate Matrix

**Setup:**
- Base case: WACC 8%, Terminal growth 3%
- Question: How does changing both simultaneously affect valuation?
- Output: Sensitivity matrix showing all combinations

**Sensitivity Matrix:**

```
                Terminal Growth Rate
           2.0%   2.5%   3.0%   3.5%   4.0%
WACC 6%   $5,000  $4,000 $3,333 $2,857 $2,500
WACC 7%   $3,333  $2,857 $2,500 $2,222 $2,000
WACC 8%   $2,500  $2,222 $2,000 $1,818 $1,667  ← Base case
WACC 9%   $2,000  $1,818 $1,667 $1,538 $1,429
WACC 10%  $1,667  $1,538 $1,429 $1,333 $1,250

Key observations:
├─ Base case (8%, 3%): $2,000M
├─ Best case (6%, 4%): $2,500M (+25%)
├─ Worst case (10%, 2%): $1,667M (-17%)
│
└─ Most sensitive to WACC (rows vary more than columns)
   Conclusion: Cost of capital assumption more critical than growth
```

**Realistic Scenario Mapping:**

```
Scenario 1: Bull Case (Best Case Outcome)
├─ Lower cost of capital (6%): Market rates fall or company improves
├─ Higher sustainable growth (4%): Market share gain, pricing power
├─ Valuation: $2,500M
└─ Probability: 20%

Scenario 2: Base Case
├─ WACC 8%, growth 3%
├─ Valuation: $2,000M
└─ Probability: 60%

Scenario 3: Bear Case (Stress Scenario)
├─ Higher cost of capital (10%): Market rates rise or company weakens
├─ Lower growth (2%): Market saturation, competition
├─ Valuation: $1,667M
└─ Probability: 20%

Expected Value:
├─ = 0.20 × $2,500M + 0.60 × $2,000M + 0.20 × $1,667M
├─ = $500M + $1,200M + $333M
└─ = $2,033M (very close to base case, as intended)
```

---

### Example 3: Scenario Analysis - Complete Business Cases

**Setup:**
- Company: Mature software platform
- Historical growth: 8% annually
- Current margin: 25% operating margin
- Question: What's valuation under different competitive outcomes?

**Scenario 1: Bull Case - Market Leadership Maintained**

```
Assumptions:
├─ Revenue growth: Years 1-3: 10%, Years 4-5: 8%, Terminal: 4%
│  └─ Rationale: Market share gains from strong competitive position
├─ Operating margin: 25% → 28% (scale benefits)
│  └─ Rationale: R&D leverage as revenue grows
├─ WACC: 7% (lower risk from stable market leadership)
│  └─ Rationale: Demonstrated pricing power, customer stickiness
├─ Capex needs: 4% of revenue growth
│  └─ Rationale: Asset-light software, high FCF conversion
│
└─ Result: Enterprise Value = $3,500M
   ├─ Sensitivity: Most optimistic 20% of cases
   └─ Decision: Buy thesis validated if price <$3,500M
```

**Scenario 2: Base Case - Stable Competition**

```
Assumptions:
├─ Revenue growth: Years 1-3: 7%, Years 4-5: 5%, Terminal: 3%
│  └─ Rationale: Market maturity, competitive equilibrium
├─ Operating margin: 25% maintained (efficiency stable)
│  └─ Rationale: Scale offsets competitive pricing pressure
├─ WACC: 8% (current market cost of capital)
│  └─ Rationale: Moderate risk profile
├─ Capex: 5% of revenue growth
│  └─ Rationale: Incremental capex for capacity
│
└─ Result: Enterprise Value = $2,000M
   ├─ Sensitivity: Most likely 60% of cases
   └─ Decision: Fair value reference point
```

**Scenario 3: Bear Case - Competitive Disruption**

```
Assumptions:
├─ Revenue growth: Years 1-3: 3%, Years 4-5: 1%, Terminal: 1%
│  └─ Rationale: New competitor with superior product
├─ Operating margin: 25% → 18% (price competition erodes margins)
│  └─ Rationale: Forced to reduce prices or increase marketing spend
├─ WACC: 10% (higher risk from competitive vulnerability)
│  └─ Rationale: Customer churn risk, margin pressure
├─ Capex: 8% of revenue growth (increased investment required)
│  └─ Rationale: Competitive R&D spend to defend position
│
└─ Result: Enterprise Value = $900M
   ├─ Sensitivity: Pessimistic 20% of cases
   └─ Decision: Downside risk if disruptive competitor enters
```

**Probability-Weighted Valuation:**

```
Expected Enterprise Value:
├─ = 0.20 × $3,500M + 0.60 × $2,000M + 0.20 × $900M
├─ = $700M + $1,200M + $180M
└─ = $2,080M

Interpretation:
├─ Base case $2,000M: Single point estimate
├─ Probability-weighted $2,080M: More realistic central estimate
├─ Range: $900M - $3,500M (2.8x spread)
│
└─ Investment decision:
   ├─ If price <$1,500M: Attractive (large margin of safety)
   ├─ If price $1,500-$2,500M: Fair value territory
   ├─ If price >$2,500M: Rich (limited upside, downside risk)
   └─ If price >$3,500M: Avoid (no margin of safety)
```

---

## IV. Layer Breakdown

```
SENSITIVITY & SCENARIO ANALYSIS FRAMEWORK

┌─────────────────────────────────────────────────────┐
│  1. ONE-WAY SENSITIVITY ANALYSIS                    │
│                                                     │
│  Methodology:                                       │
│  ├─ Base case valuation established first          │
│  ├─ Vary 1 assumption ±10%, ±20%, ±30% etc.       │
│  ├─ Hold all other assumptions constant            │
│  ├─ Calculate resulting enterprise value           │
│  └─ Plot EV vs assumption (creates curves)         │
│                                                     │
│  Key assumptions to test:                          │
│  ├─ Discount rate (WACC)                           │
│  │  ├─ Range: ±200 bps typical (6-10%)            │
│  │  ├─ Impact: Highest sensitivity                 │
│  │  └─ Reason: Appears in denominator              │
│  │                                                 │
│  ├─ Terminal growth rate                           │
│  │  ├─ Range: ±100 bps typical (2-4%)             │
│  │  ├─ Impact: Highest sensitivity (as approaches │
│  │  │  WACC)                                       │
│  │  └─ Reason: Terminal value often 60-80% of EV  │
│  │                                                 │
│  ├─ Revenue growth (forecast period)                │
│  │  ├─ Range: ±200 bps typical (5-10%)            │
│  │  ├─ Impact: Moderate sensitivity                │
│  │  └─ Reason: Fewer years than terminal value    │
│  │                                                 │
│  ├─ Operating margins (normalized)                  │
│  │  ├─ Range: ±200 bps typical (20-30%)           │
│  │  ├─ Impact: Moderate-high sensitivity           │
│  │  └─ Reason: Directly multiplies revenue        │
│  │                                                 │
│  ├─ Capex as % of revenue                          │
│  │  ├─ Range: ±2 pct points typical (3-7%)        │
│  │  ├─ Impact: Low-moderate sensitivity            │
│  │  └─ Reason: Reduces FCF but not by large amount│
│  │                                                 │
│  ├─ Tax rate                                        │
│  │  ├─ Range: ±300 bps typical (19-28%)           │
│  │  ├─ Impact: Moderate sensitivity                │
│  │  └─ Reason: Affects net cash flow               │
│  │                                                 │
│  └─ Working capital changes                         │
│     ├─ Range: ±2 pct points typical (2-6%)        │
│     ├─ Impact: Low sensitivity                     │
│     └─ Reason: Often small relative to revenue    │
│                                                     │
│  Interpretation of sensitivity:                     │
│  ├─ Steep curve = High sensitivity (small change   │
│  │  in assumption = large value change)            │
│  ├─ Flat curve = Low sensitivity (assumption       │
│  │  doesn't matter)                                │
│  ├─ Non-linear curve = "Cliff risk" (at certain   │
│  │  assumption levels, valuation collapses)       │
│  └─ Example: As terminal growth → WACC, curve     │
│     becomes vertical (infinite sensitivity)        │
│                                                     │
│  Common output formats:                             │
│  ├─ Sensitivity table (rows: assumption levels,    │
│  │  columns: resulting EV)                         │
│  ├─ Line chart (x-axis: assumption, y-axis: EV)   │
│  ├─ Tornado chart (sorted by impact magnitude)     │
│  └─ Waterfall chart (how changes compound)         │
│                                                     │
└──────────────────┬────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │  2. TWO-WAY SENSITIVITY ANALYSIS             │
    │                                              │
    │  Methodology:                                │
    │  ├─ Create matrix with 2 assumptions         │
    │  ├─ Vary each independently (5-7 values)    │
    │  ├─ Calculate EV for each combination        │
    │  └─ Matrix cell = EV for assumption pair    │
    │                                              │
    │  Most common pairs:                          │
    │  ├─ WACC × Terminal growth                  │
    │  │  └─ Most important drivers in valuation  │
    │  ├─ Terminal growth × Terminal margin       │
    │  │  └─ Long-term competitive outcomes       │
    │  ├─ Revenue growth × Operating margin       │
    │  │  └─ Business model quality               │
    │  └─ WACC × Operating margin                 │
    │     └─ Risk vs profitability tradeoff       │
    │                                              │
    │  Interpretation:                             │
    │  ├─ Color coding: Red (low EV), Green       │
    │  │  (high EV) helps visualization            │
    │  ├─ Identify "danger zones" (low EV cells)  │
    │  ├─ Best/worst case combinations clear      │
    │  ├─ Base case cell highlighted              │
    │  └─ Realistic ranges shown with borders     │
    │                                              │
    │  Example: WACC vs Terminal Growth Matrix    │
    │  ├─ Columns: 2%, 3%, 4%, 5%                 │
    │  ├─ Rows: 6%, 7%, 8%, 9%, 10%              │
    │  ├─ Each cell: calculated EV                │
    │  ├─ Base case (8%, 3%): $2,000M (middle)   │
    │  ├─ Pattern: Higher right corner (higher    │
    │  │  growth, lower WACC) = highest EV       │
    │  └─ Pattern: Lower left corner = lowest EV │
    │                                              │
    │  When to use 2-way:                          │
    │  ├─ Communicating to management/board       │
    │  ├─ Investment committee decisions            │
    │  ├─ M&A pricing negotiation (showing range) │
    │  └─ Stress-testing key interdependencies    │
    │                                              │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │  3. SCENARIO ANALYSIS - NAMED CASES           │
    │                                              │
    │  Typical structure:                          │
    │  ├─ 3-5 named scenarios (Bull, Base, Bear   │
    │  │  + sometimes Deep Bear, Moon Shot)       │
    │  ├─ Each scenario: Internal consistency      │
    │  │  (not arbitrary numbers)                  │
    │  ├─ Probability assigned to each scenario   │
    │  └─ Weighted expected value calculated      │
    │                                              │
    │  Scenario design principles:                 │
    │  ├─ Realistic: Based on actual market       │
    │  │  research, competitive analysis           │
    │  ├─ Internally consistent: Growth aligned   │
    │  │  with margins (fast growth usually lower  │
    │  │  margins due to competition)              │
    │  ├─ Differentiated: Not symmetric           │
    │  │  (downside usually worse than upside)    │
    │  ├─ Mutually exclusive: Cannot happen       │
    │  │  simultaneously                           │
    │  └─ Collectively exhaustive: All possible   │
    │     outcomes covered (probabilities sum 1)  │
    │                                              │
    │  Bull case characteristics:                  │
    │  ├─ Higher growth (market expansion, share  │
    │  │  gains)                                   │
    │  ├─ Margin expansion (scale, pricing power) │
    │  ├─ Lower WACC (de-risking through success) │
    │  ├─ Longer forecast period (multiyear       │
    │  │  advantage)                               │
    │  └─ Probability: Often 15-30%               │
    │                                              │
    │  Base case characteristics:                  │
    │  ├─ Consensus industry growth               │
    │  ├─ Normalized margins (no competitive      │
    │  │  advantage)                               │
    │  ├─ Current market WACC                      │
    │  ├─ Moderate competitive intensity           │
    │  └─ Probability: Usually 50-60%             │
    │                                              │
    │  Bear case characteristics:                  │
    │  ├─ Lower growth (new competitors,          │
    │  │  commoditization)                        │
    │  ├─ Margin compression (price competition)  │
    │  ├─ Higher WACC (increased risk)            │
    │  ├─ Potential structural decline             │
    │  └─ Probability: Usually 15-30%             │
    │                                              │
    │  Weighted EV calculation:                    │
    │  ├─ EV = Σ (Prob_i × EV_i)                  │
    │  ├─ Provides single number incorporating    │
    │  │  downside/upside risk                     │
    │  ├─ More realistic than base case alone     │
    │  └─ Communicates board risk tolerance       │
    │                                              │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │  4. MONTE CARLO SIMULATION                   │
    │                                              │
    │  Methodology:                                │
    │  ├─ Define probability distributions for    │
    │  │  each assumption (normal, uniform,       │
    │  │  triangular)                              │
    │  ├─ Randomly sample from each distribution  │
    │  ├─ Run DCF model with sampled values       │
    │  ├─ Repeat 10,000 times                      │
    │  ├─ Collect resulting EV distribution       │
    │  └─ Calculate percentiles, mean, std dev   │
    │                                              │
    │  Output metrics:                             │
    │  ├─ Mean EV: Average valuation              │
    │  ├─ Std deviation: Valuation uncertainty    │
    │  ├─ Percentiles (10th, 50th, 90th):        │
    │  │  Downside, median, upside EV            │
    │  ├─ Probability of loss: P(EV < purchase   │
    │  │  price)                                  │
    │  └─ Value at Risk (VaR): 5% worst case EV  │
    │                                              │
    │  When to use:                                │
    │  ├─ Complex models with many assumptions   │
    │  ├─ Communicating tail risk to investors    │
    │  ├─ M&A diligence (downside protection)     │
    │  ├─ Portfolio risk aggregation               │
    │  └─ Private equity underwriting              │
    │                                              │
    │  Advantages over scenarios:                  │
    │  ├─ Captures all possible combinations      │
    │  ├─ Quantifies exact probabilities          │
    │  ├─ Identifies tail risks (1% chance)       │
    │  ├─ More sophisticated for complex deals    │
    │  └─ Difficult to model manually              │
    │                                              │
    │  Disadvantages:                              │
    │  ├─ Black box (hard to explain to board)    │
    │  ├─ Sensitive to distribution assumptions   │
    │  ├─ Misused if correlations not modeled     │
    │  └─ Complexity can hide input errors        │
    │                                              │
    └────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### One-Way Sensitivity Formula

$$\text{Sensitivity} = \frac{\partial \text{EV}}{\partial x}$$

For perpetuity: $\text{EV} = \frac{\text{FCF}}{\text{WACC} - g}$

$$\frac{\partial \text{EV}}{\partial \text{WACC}} = -\frac{\text{FCF}}{(\text{WACC} - g)^2}$$

### Weighted Expected Value

$$\text{EV}_{\text{weighted}} = \sum_{i=1}^{n} p_i \times \text{EV}_i$$

Where $p_i$ = probability of scenario $i$, $\sum p_i = 1$

### Elasticity (% Change in EV per 1% Change in Variable)

$$\text{Elasticity} = \frac{\% \text{ change in EV}}{\% \text{ change in assumption}} = \frac{\partial \ln(\text{EV})}{\partial \ln(x)}$$

---

## VI. Python Mini-Project: Sensitivity & Scenario Analyzer

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# ============================================================================
# SENSITIVITY & SCENARIO ANALYSIS TOOLKIT
# ============================================================================

class SensitivityAnalyzer:
    """One-way and two-way sensitivity analysis"""
    
    @staticmethod
    def perpetuity_valuation(fcf, wacc, terminal_growth):
        """Calculate enterprise value using perpetuity formula"""
        if wacc <= terminal_growth:
            return np.inf
        return fcf / (wacc - terminal_growth)
    
    @staticmethod
    def one_way_sensitivity(base_fcf, base_wacc, base_growth, 
                           variable, var_range):
        """
        Calculate one-way sensitivity
        variable: 'wacc', 'growth', or 'fcf'
        var_range: array of values for the variable
        """
        evs = []
        
        for val in var_range:
            if variable == 'wacc':
                ev = SensitivityAnalyzer.perpetuity_valuation(base_fcf, val, base_growth)
            elif variable == 'growth':
                ev = SensitivityAnalyzer.perpetuity_valuation(base_fcf, base_wacc, val)
            elif variable == 'fcf':
                ev = SensitivityAnalyzer.perpetuity_valuation(val, base_wacc, base_growth)
            
            evs.append(min(ev, 1e10))  # Cap at extreme values
        
        return evs
    
    @staticmethod
    def two_way_sensitivity(base_fcf, base_wacc, base_growth,
                           var1_name, var1_range, var2_name, var2_range):
        """
        Create two-way sensitivity matrix
        """
        matrix = []
        
        for val2 in var2_range:
            row = []
            for val1 in var1_range:
                if var1_name == 'wacc' and var2_name == 'growth':
                    ev = SensitivityAnalyzer.perpetuity_valuation(base_fcf, val1, val2)
                elif var1_name == 'growth' and var2_name == 'wacc':
                    ev = SensitivityAnalyzer.perpetuity_valuation(base_fcf, val2, val1)
                elif var1_name == 'fcf' and var2_name == 'wacc':
                    ev = SensitivityAnalyzer.perpetuity_valuation(val1, val2, base_growth)
                else:
                    ev = 0
                
                row.append(min(ev, 1e10))
            matrix.append(row)
        
        return np.array(matrix)


class ScenarioAnalyzer:
    """Scenario-based valuation with probabilities"""
    
    @staticmethod
    def define_scenarios(scenarios_dict):
        """
        scenarios_dict: {
            'scenario_name': {
                'fcf': value,
                'wacc': value,
                'growth': value,
                'probability': value
            }
        }
        """
        results = {}
        total_prob = 0
        
        for name, params in scenarios_dict.items():
            ev = SensitivityAnalyzer.perpetuity_valuation(
                params['fcf'], params['wacc'], params['growth']
            )
            results[name] = {
                'ev': min(ev, 1e10),
                'fcf': params['fcf'],
                'wacc': params['wacc'],
                'growth': params['growth'],
                'probability': params['probability']
            }
            total_prob += params['probability']
        
        # Normalize probabilities
        for name in results:
            results[name]['probability'] /= total_prob
        
        return results
    
    @staticmethod
    def weighted_ev(scenarios):
        """Calculate probability-weighted expected value"""
        weighted_sum = sum(s['ev'] * s['probability'] for s in scenarios.values())
        return weighted_sum
    
    @staticmethod
    def scenario_statistics(scenarios):
        """Calculate statistics across scenarios"""
        evs = [s['ev'] for s in scenarios.values()]
        probs = [s['probability'] for s in scenarios.values()]
        
        mean = np.sum(np.array(evs) * np.array(probs))
        variance = np.sum(np.array(probs) * (np.array(evs) - mean)**2)
        std_dev = np.sqrt(variance)
        
        # Percentiles
        sorted_evs = sorted(evs)
        p10 = sorted_evs[int(len(sorted_evs) * 0.1)]
        p50 = sorted_evs[int(len(sorted_evs) * 0.5)]
        p90 = sorted_evs[int(len(sorted_evs) * 0.9)]
        
        return {
            'mean': mean,
            'std_dev': std_dev,
            'min': min(evs),
            'max': max(evs),
            'p10': p10,
            'p50': p50,
            'p90': p90,
            'cv': std_dev / mean if mean > 0 else 0  # Coefficient of variation
        }


class MonteCarlo:
    """Monte Carlo simulation for valuation"""
    
    @staticmethod
    def simulate_valuation(fcf_mean, fcf_std, wacc_mean, wacc_std,
                          growth_mean, growth_std, num_sims=10000):
        """
        Run Monte Carlo with normal distributions
        """
        evs = []
        
        for _ in range(num_sims):
            fcf = np.random.normal(fcf_mean, fcf_std)
            wacc = np.random.normal(wacc_mean, wacc_std)
            growth = np.random.normal(growth_mean, growth_std)
            
            # Ensure growth < WACC
            if growth >= wacc:
                growth = wacc - 0.005
            
            ev = SensitivityAnalyzer.perpetuity_valuation(fcf, wacc, growth)
            evs.append(min(ev, 1e10))
        
        return np.array(evs)
    
    @staticmethod
    def mc_statistics(evs):
        """Calculate statistics from simulation"""
        return {
            'mean': np.mean(evs),
            'std_dev': np.std(evs),
            'median': np.median(evs),
            'p5': np.percentile(evs, 5),
            'p25': np.percentile(evs, 25),
            'p75': np.percentile(evs, 75),
            'p95': np.percentile(evs, 95),
            'skewness': (np.mean(evs) - np.median(evs)) / np.std(evs)
        }


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SENSITIVITY & SCENARIO ANALYSIS")
print("="*80)

# 1. One-way sensitivity
print(f"\n1. ONE-WAY SENSITIVITY - DISCOUNT RATE IMPACT")
print(f"{'-'*80}")

base_fcf = 100  # $100M annual FCF
base_wacc = 0.08
base_growth = 0.03

base_ev = SensitivityAnalyzer.perpetuity_valuation(base_fcf, base_wacc, base_growth)
print(f"\nBase case: FCF ${base_fcf}M, WACC {base_wacc*100:.0f}%, Growth {base_growth*100:.0f}%")
print(f"Base EV: ${base_ev:.0f}M")

wacc_range = np.array([0.06, 0.07, 0.08, 0.09, 0.10])
wacc_evs = SensitivityAnalyzer.one_way_sensitivity(base_fcf, base_wacc, base_growth, 
                                                   'wacc', wacc_range)

print(f"\nWACC Sensitivity:")
for wacc_val, ev in zip(wacc_range, wacc_evs):
    change = (ev - base_ev) / base_ev * 100
    print(f"  WACC {wacc_val*100:.0f}%: EV ${ev:.0f}M ({change:+.0f}%)")

# 2. Two-way sensitivity
print(f"\n2. TWO-WAY SENSITIVITY - WACC × TERMINAL GROWTH")
print(f"{'-'*80}")

wacc_matrix_range = np.array([0.06, 0.07, 0.08, 0.09, 0.10])
growth_matrix_range = np.array([0.02, 0.025, 0.03, 0.035, 0.04])

matrix = SensitivityAnalyzer.two_way_sensitivity(
    base_fcf, base_wacc, base_growth,
    'wacc', wacc_matrix_range,
    'growth', growth_matrix_range
)

print(f"\nEV Matrix (WACC rows, Growth columns):")
print(f"            Growth: ", end="")
for g in growth_matrix_range:
    print(f"{g*100:.1f}%   ", end="")
print()

for i, w in enumerate(wacc_matrix_range):
    print(f"WACC {w*100:.0f}%:    ", end="")
    for j in range(len(growth_matrix_range)):
        val = matrix[i, j]
        marker = " ← " if (w == base_wacc and growth_matrix_range[j] == base_growth) else ""
        print(f"${val:.0f}M{marker}  ", end="")
    print()

# 3. Scenario analysis
print(f"\n3. SCENARIO ANALYSIS - 3 CASES")
print(f"{'-'*80}")

scenarios = {
    'Bull': {
        'fcf': 120,  # Higher FCF from growth
        'wacc': 0.07,  # Lower risk
        'growth': 0.04,  # Higher sustainable growth
        'probability': 0.25
    },
    'Base': {
        'fcf': 100,
        'wacc': 0.08,
        'growth': 0.03,
        'probability': 0.50
    },
    'Bear': {
        'fcf': 80,  # Lower FCF from margin pressure
        'wacc': 0.10,  # Higher risk
        'growth': 0.02,  # Lower growth
        'probability': 0.25
    }
}

scenario_results = ScenarioAnalyzer.define_scenarios(scenarios)

print(f"\nScenario Details:")
for name, params in scenario_results.items():
    print(f"\n{name}:")
    print(f"  FCF: ${params['fcf']}M")
    print(f"  WACC: {params['wacc']*100:.0f}%")
    print(f"  Terminal growth: {params['growth']*100:.0f}%")
    print(f"  EV: ${params['ev']:.0f}M")
    print(f"  Probability: {params['probability']*100:.0f}%")

weighted_ev = ScenarioAnalyzer.weighted_ev(scenario_results)
stats = ScenarioAnalyzer.scenario_statistics(scenario_results)

print(f"\nProbability-Weighted Valuation:")
print(f"  Expected EV: ${weighted_ev:.0f}M")
print(f"  Range: ${stats['min']:.0f}M (bear) to ${stats['max']:.0f}M (bull)")
print(f"  Spread: ${stats['max'] - stats['min']:.0f}M ({(stats['max']/stats['min']-1)*100:.0f}% spread)")

# 4. Monte Carlo simulation
print(f"\n4. MONTE CARLO SIMULATION (10,000 iterations)")
print(f"{'-'*80}")

fcf_std = 10  # $10M std dev
wacc_std = 0.015  # 1.5% std dev
growth_std = 0.010  # 1% std dev

mc_evs = MonteCarlo.simulate_valuation(
    base_fcf, fcf_std, base_wacc, wacc_std,
    base_growth, growth_std, num_sims=10000
)

mc_stats = MonteCarlo.mc_statistics(mc_evs)

print(f"\nMonte Carlo Results:")
print(f"  Mean EV: ${mc_stats['mean']:.0f}M")
print(f"  Std Dev: ${mc_stats['std_dev']:.0f}M")
print(f"  Median: ${mc_stats['median']:.0f}M")
print(f"  5th percentile (downside): ${mc_stats['p5']:.0f}M")
print(f"  95th percentile (upside): ${mc_stats['p95']:.0f}M")
print(f"  Coefficient of variation: {mc_stats['std_dev']/mc_stats['mean']:.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: One-way sensitivity
ax1 = axes[0, 0]

wacc_pct = wacc_range * 100
ax1.plot(wacc_pct, wacc_evs, linewidth=2.5, marker='o', markersize=8, color='blue')
ax1.axvline(x=base_wacc*100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Base case')
ax1.axhline(y=base_ev, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.fill_between(wacc_pct, wacc_evs, alpha=0.2, color='blue')

ax1.set_xlabel('WACC (%)')
ax1.set_ylabel('Enterprise Value ($M)')
ax1.set_title('Panel 1: One-Way Sensitivity (WACC Impact)')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Panel 2: Two-way sensitivity heatmap
ax2 = axes[0, 1]

im = ax2.imshow(matrix, cmap='RdYlGn', aspect='auto')
ax2.set_xticks(range(len(growth_matrix_range)))
ax2.set_yticks(range(len(wacc_matrix_range)))
ax2.set_xticklabels([f'{g*100:.1f}%' for g in growth_matrix_range])
ax2.set_yticklabels([f'{w*100:.0f}%' for w in wacc_matrix_range])
ax2.set_xlabel('Terminal Growth (%)')
ax2.set_ylabel('WACC (%)')
ax2.set_title('Panel 2: Two-Way Sensitivity Heatmap')

# Add value labels
for i in range(len(wacc_matrix_range)):
    for j in range(len(growth_matrix_range)):
        text = ax2.text(j, i, f'${matrix[i, j]:.0f}',
                       ha='center', va='center', color='black', fontweight='bold', fontsize=8)

plt.colorbar(im, ax=ax2, label='EV ($M)')

# Panel 3: Scenario analysis
ax3 = axes[1, 0]

scenario_names = list(scenario_results.keys())
scenario_evs = [scenario_results[name]['ev'] for name in scenario_names]
scenario_probs = [scenario_results[name]['probability']*100 for name in scenario_names]
colors = ['green', 'blue', 'red']

bars = ax3.bar(scenario_names, scenario_evs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add probability labels
for bar, prob in zip(bars, scenario_probs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 50,
            f'{prob:.0f}%', ha='center', fontweight='bold', fontsize=10)

ax3.axhline(y=weighted_ev, color='black', linestyle='--', linewidth=2, label=f'Weighted EV: ${weighted_ev:.0f}M')
ax3.set_ylabel('Enterprise Value ($M)')
ax3.set_title('Panel 3: Scenario Analysis (Probability-Weighted)')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Monte Carlo distribution
ax4 = axes[1, 1]

ax4.hist(mc_evs, bins=50, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
ax4.axvline(x=mc_stats['mean'], color='red', linestyle='--', linewidth=2.5, label=f"Mean: ${mc_stats['mean']:.0f}M")
ax4.axvline(x=mc_stats['median'], color='blue', linestyle='--', linewidth=2.5, label=f"Median: ${mc_stats['median']:.0f}M")
ax4.axvline(x=mc_stats['p5'], color='orange', linestyle='--', linewidth=2, label=f"5th %ile: ${mc_stats['p5']:.0f}M")
ax4.axvline(x=mc_stats['p95'], color='green', linestyle='--', linewidth=2, label=f"95th %ile: ${mc_stats['p95']:.0f}M")

ax4.set_xlabel('Enterprise Value ($M)')
ax4.set_ylabel('Frequency')
ax4.set_title('Panel 4: Monte Carlo Distribution (10,000 sims)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sensitivity_scenario_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• Valuation highly sensitive to WACC (small changes = large EV impact)")
print("• Terminal value often dominates NPV; growth assumption critical")
print("• Scenario analysis provides realistic valuation range with probabilities")
print("• Monte Carlo captures tail risks not visible in 3-scenario approach")
print("• Always present range, not point estimate, to boards & investors")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Damodaran, A. (2012).** "Investment Valuation: Tools and Techniques for Determining Any Asset's Value," 3rd ed.
   - Comprehensive sensitivity framework, scenario design principles

2. **Copeland, T., Koller, T., & Murrin, J. (2000).** "Valuation: Measuring and Managing the Value of Companies," 3rd ed.
   - DCF sensitivities, scenario planning integration

3. **Mauboussin, M. J. (2012).** "Expectations Investing: Reading Stock Prices for Better Returns."
   - Using sensitivity analysis to test investment hypotheses

**Key Design Concepts:**

- **Sensitivity is Non-Linear:** Small WACC changes create large EV changes; asymmetry increases as terminal growth approaches WACC.
- **Scenario Consistency Matters:** Bull case must have realistic margin/growth correlation (e.g., growth doesn't occur without capex).
- **Perpetuity Dominance:** Terminal value often 60-80% of enterprise value; terminal assumptions deserve equal diligence to forecast period.
- **Communicate Ranges, Not Points:** Single valuation estimate false precision; range with probabilities reflects actual uncertainty.

