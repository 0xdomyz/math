# Dynamic Programming & Lifecycle Asset Allocation

## 1. Concept Skeleton
**Definition:** Dynamic Programming: Multi-period optimization framework solving optimal decision sequences via backward recursion (Bellman equation). Lifecycle Allocation: Age-dependent portfolio strategy recognizing human capital as implicit bond; de-risk as approach retirement (target-date glide path).  
**Purpose:** Optimize portfolios over investor lifetime considering changing risk capacity, human capital, consumption needs, mortality risk. Extends static MPT to realistic multi-period context.  
**Prerequisites:** Mean-variance optimization, utility theory, present value, compound returns

---

## 2. Comparative Framing

| Aspect | Static MPT (Single-Period) | Dynamic Programming (Multi-Period) | Lifecycle Glide Path | Buy-and-Hold | Constant Mix Rebalancing |
|--------|---------------------------|-----------------------------------|---------------------|--------------|-------------------------|
| **Time Horizon** | One period (fixed) | Multiple periods (sequential) | Age-dependent (40+ years) | Entire lifetime (no trades) | Ongoing (periodic rebalancing) |
| **Allocation** | Fixed weights | State-dependent weights | Age-based rule (stocks decrease) | Initial allocation only | Mechanical (back to target) |
| **Rebalancing** | None (single-period) | Optimal rebalancing per period | Annual glide path adjustment | Never rebalance | Periodic (e.g., quarterly) |
| **Human Capital** | Ignored | Can incorporate as state variable | Central (implicit bond-like asset) | Ignored | Ignored |
| **Costs** | Not applicable | Include transaction costs in optimization | Minimal (infrequent trades) | Zero costs | Transaction costs eat returns |
| **Risk Tolerance** | Constant λ | Time-varying λ(t, wealth) | Decreases with age | Constant (implicit) | Constant (target weights) |
| **Complexity** | Simple (closed-form) | High (numerical Bellman recursion) | Moderate (heuristic rule) | Trivial (one-time decision) | Low (mechanical rebalancing) |
| **Empirical Evidence** | Suboptimal over long horizons | Theoretically optimal but hard to implement | Strong empirical support | Concentration risk; suboptimal | Good performance if low costs |
| **Suitability** | Academic benchmark | Institutional investors with models | Individual retirement savers | Passive investors; lazy portfolios | Active investors; taxable accounts |

**Key Insight:** Static MPT inadequate for lifetime investing; dynamic programming optimal but complex; lifecycle heuristic (glide path) practical approximation with strong empirical support.

---

## 3. Examples + Counterexamples

**Example 1: Lifecycle Allocation (Age-Based Rule)**

Investor age 25 (40 years to retirement):
- Human capital: High (future earnings stream worth $2M PV)
- Financial capital: Low ($50k portfolio)
- Human capital is bond-like (stable salary; low correlation to stocks)

**Optimal allocation:**
- Age 25: 90% stocks, 10% bonds (high risk capacity due to human capital cushion)
- Age 40: 70% stocks, 30% bonds (human capital declining; portfolio growing)
- Age 55: 50% stocks, 50% bonds (near retirement; human capital exhausted)
- Age 70: 30% stocks, 70% bonds (preservation mode; living off portfolio)

**Rationale:**
- Young investor has "implicit bonds" (future earnings)
- Can take more equity risk (long horizon to recover from crashes)
- As age increases, human capital depletes → need more explicit bonds
- Classic rule: Stock allocation = 100 - age (e.g., age 30 → 70% stocks)

**Implication:** Lifecycle allocation materially different from static MPT (which would recommend same allocation regardless of age).

---

**Example 2: Dynamic Programming Solution (Simplified 2-Period)**

Investor maximizes lifetime utility:
$$\max E[U(C_1) + \beta U(C_2)]$$

Where:
- C₁ = consumption period 1
- C₂ = consumption period 2
- β = discount factor (patience; typically 0.95-0.99)
- U(C) = log(C) (CRRA utility)

**Bellman equation (backward recursion):**

Period 2 (terminal):
$$V_2(W_2) = \max_{C_2} U(C_2) \quad \text{s.t.} \quad C_2 \leq W_2$$
- Consume all remaining wealth

Period 1 (working backward):
$$V_1(W_1) = \max_{C_1, w} U(C_1) + \beta E[V_2(W_2)]$$
$$\text{s.t.} \quad W_2 = (W_1 - C_1) \times R_p(w)$$

Where:
- w = portfolio weights (stocks vs bonds)
- R_p(w) = portfolio return (random)
- E[·] = expectation over return distribution

**Solution:**
- Optimal consumption: Smooth consumption over time (consume ~10-20% of wealth per period)
- Optimal allocation: Depends on wealth, age, risk aversion
- Young (period 1): High stocks (70-90%)
- Old (period 2): Lower stocks (30-50%)

**Implication:** Dynamic solution gives age-dependent allocation automatically (emerges from optimization, not imposed).

---

**Example 3: Human Capital as Implicit Bond**

Worker age 30:
- Salary: $100k/year (expected to grow 3% p.a.)
- Years to retirement: 35
- Present value of future earnings (discount at 4%):
  $$PV = 100k \times \frac{1 - (1.03/1.04)^{35}}{0.04 - 0.03} \approx \$2.5M$$

- Financial portfolio: $200k (stocks + bonds)
- Total wealth: $2.5M (human capital) + $200k (financial) = $2.7M

**Implicit bond allocation:**
- If human capital is bond-like (stable salary; low stock correlation):
  - Human capital as "bonds": $2.5M / $2.7M = 92% bonds (implicit)
  - Financial capital should be mostly stocks to balance overall allocation
  
**Optimal financial portfolio:**
- Target overall allocation: 70% stocks, 30% bonds
- Implicit bonds: 92% × ($2.5M / $2.7M) = 85% of total wealth
- Need stocks in financial portfolio: (70% - 8%) / (200k / 2.7M) = Need ~100% stocks in financial portfolio

**Implication:** Young workers with stable salaries should hold nearly 100% stocks in financial portfolios (human capital already "bonds").

---

**Example 4: Target-Date Fund Glide Path**

Target-Date 2060 Fund (for investors retiring in 2060):
- Age 25 (2025): 90% stocks, 10% bonds
- Age 40 (2040): 75% stocks, 25% bonds
- Age 55 (2055): 55% stocks, 45% bonds
- Age 65 (2065, retirement): 30% stocks, 70% bonds
- Age 75 (2075, 10 years into retirement): 20% stocks, 80% bonds

**Mechanics:**
- Automatic rebalancing along glide path (annual adjustment)
- "To retirement" glide path: Reduces stocks until retirement, then stabilizes
- "Through retirement" glide path: Continues reducing stocks for 10+ years post-retirement

**Empirical performance:**
- Vanguard Target Retirement 2060: ~8-9% annualized (2015-2024)
- Better than static 60/40 for young investors (captured equity upside early)
- Automatic discipline (prevents behavioral errors)

**Implication:** Target-date funds implement lifecycle allocation; popular in 401(k) plans (now ~40% of assets).

---

**COUNTEREXAMPLE 5: Buy-and-Hold vs Rebalancing (Concentration Risk)**

Investor age 25 starts with 80% stocks, 20% bonds ($100k total).

**Strategy 1: Buy-and-Hold (never rebalance)**
- After 10 years: Stocks +150%, bonds +40%
- Portfolio: Stocks = $80k × 2.5 = $200k; Bonds = $20k × 1.4 = $28k
- New allocation: $200k / $228k = 88% stocks (drifted up!)
- Risk increased without intention (market gains concentrated portfolio)

**Strategy 2: Annual Rebalancing (back to 80/20)**
- After 10 years: Same total wealth, but periodically sold stocks, bought bonds
- Allocation: Still 80% stocks, 20% bonds (maintained target)
- Benefit: Disciplined "sell high, buy low" (contrarian)
- Cost: Transaction costs (0.1-0.5% annually)

**Comparison:**
- Buy-and-hold: Higher return if stocks continue rising (more equity exposure)
- Rebalanced: Lower return but more stable (controlled risk)
- Historical studies: Rebalancing adds ~0.5-1% p.a. after costs (debated)

**Implication:** Buy-and-hold can create unintended concentration; rebalancing maintains risk control.

---

**COUNTEREXAMPLE 6: Static 60/40 Portfolio for 25-Year-Old (Suboptimal)**

Traditional advice: "60% stocks, 40% bonds for moderate risk."

Applied to age 25 investor:
- Human capital: $2M (implicit bonds)
- Financial portfolio: $50k
- Static 60/40: $30k stocks, $20k bonds

**Total allocation (including human capital):**
- Stocks: $30k / $2.05M total wealth = 1.5% (FAR TOO CONSERVATIVE!)
- Bonds: ($20k + $2M human capital) / $2.05M = 98.5%

**Implication:** Static 60/40 allocation ignores human capital; massively suboptimal for young investors (foregoes decades of equity returns).

**Better allocation:** 95% stocks, 5% bonds in financial portfolio (appropriate given human capital cushion).

---

## 4. Layer Breakdown

```
Dynamic Programming & Lifecycle Allocation Architecture:

├─ Dynamic Programming Framework (Bellman Equation):
│   ├─ Multi-Period Optimization:
│   │   │ Maximize lifetime utility: max Σ β^t E[U(C_t)]
│   │   │
│   │   ├─ Decision variables:
│   │   │   ├─ Consumption C_t (how much to spend each period)
│   │   │   ├─ Portfolio weights w_t (stocks vs bonds)
│   │   │   └─ Labor supply L_t (work vs leisure; optional)
│   │   │
│   │   ├─ State variables:
│   │   │   ├─ Wealth W_t (financial assets)
│   │   │   ├─ Human capital H_t (PV of future earnings)
│   │   │   ├─ Age/time t (deterministic)
│   │   │   └─ Other: Health status, family size, etc.
│   │   │
│   │   ├─ Transition equation:
│   │   │   W_{t+1} = (W_t - C_t) × R_p,t+1(w_t) + Y_{t+1}
│   │   │   Where: R_p = portfolio return, Y = labor income
│   │   │
│   │   └─ Constraint:
│   │       C_t ≥ 0, W_t ≥ 0 (no borrowing beyond limit)
│   │
│   ├─ Bellman Recursion (Backward Induction):
│   │   │ Start at terminal period T; work backward to t=0
│   │   │
│   │   ├─ Terminal period (T):
│   │   │   V_T(W_T) = max U(W_T)  (consume all remaining wealth)
│   │   │
│   │   ├─ Period T-1:
│   │   │   V_{T-1}(W_{T-1}) = max_{C,w} [U(C) + β E[V_T(W_T)]]
│   │   │   Subject to: W_T = (W_{T-1} - C) × R_p(w)
│   │   │
│   │   ├─ General recursion:
│   │   │   V_t(W_t) = max_{C_t,w_t} [U(C_t) + β E[V_{t+1}(W_{t+1}) | W_t]]
│   │   │
│   │   └─ Policy functions (solution):
│   │       ├─ Consumption rule: C_t^* = g_c(W_t, t)
│   │       ├─ Portfolio rule: w_t^* = g_w(W_t, t)
│   │       └─ Value function: V_t(W_t) stores expected lifetime utility
│   │
│   ├─ Solution Methods:
│   │   ├─ Analytical (rare):
│   │   │   ├─ Merton (1969): Continuous-time; log utility; closed-form
│   │   │   │   └─ Optimal w* = (μ - r) / (γ σ²) (constant allocation!)
│   │   │   └─ Limited cases; requires strong assumptions
│   │   │
│   │   ├─ Numerical (common):
│   │   │   ├─ Discretize state space (W grid: 0, 1k, 2k, ..., 10M)
│   │   │   ├─ Discretize controls (C, w on grid)
│   │   │   ├─ Value function iteration:
│   │   │   │   └─ For each (W, t): Try all (C, w); pick max E[U + βV]
│   │   │   ├─ Interpolation: Between grid points (cubic spline)
│   │   │   └─ Convergence: Iterate until V_t converges
│   │   │
│   │   └─ Simulation:
│   │       ├─ Monte Carlo: Simulate many paths; average outcomes
│   │       ├─ Calibrate: Match historical data (returns, income)
│   │       └─ Policy evaluation: Compare strategies
│   │
│   └─ Key Results from Dynamic Programming:
│       ├─ Optimal allocation time-varying (not constant)
│       ├─ Young: High stocks (80-100%); long horizon
│       ├─ Old: Lower stocks (30-50%); short horizon
│       ├─ Wealthy: More conservative (less need for risk)
│       ├─ Poor: More aggressive (need growth to meet goals)
│       └─ Consumption smoothing: Avoid sharp changes in spending
│
├─ Lifecycle Asset Allocation (Practical Implementation):
│   ├─ Human Capital as Implicit Asset:
│   │   │ PV of future labor income = bond-like asset
│   │   │
│   │   ├─ Calculation:
│   │   │   HC_t = Σ_{s=t+1}^{T} E[Y_s] / (1 + r)^{s-t}
│   │   │   Where: Y_s = expected income year s; r = discount rate
│   │   │
│   │   ├─ Properties:
│   │   │   ├─ Declines with age (fewer working years remaining)
│   │   │   ├─ Bond-like if stable salary (low β to stocks)
│   │   │   ├─ Stock-like if commission-based income (high β)
│   │   │   └─ Largest asset for young workers (exceeds financial wealth)
│   │   │
│   │   ├─ Implications:
│   │   │   ├─ Young: Human capital >> financial wealth
│   │   │   │   └─ Already heavily "bond" allocated (implicit)
│   │   │   │   └─ Financial portfolio should be ~100% stocks
│   │   │   ├─ Middle-aged: Human capital = financial wealth
│   │   │   │   └─ Balanced allocation (60/40 stocks/bonds)
│   │   │   └─ Retired: No human capital
│   │   │       └─ Financial portfolio is total wealth
│   │   │       └─ Conservative allocation (30/70 stocks/bonds)
│   │   │
│   │   └─ Exceptions:
│   │       ├─ Entrepreneur: Human capital correlated with business (stock-like)
│   │       │   └─ Should hold more bonds (hedge specific risk)
│   │       ├─ Finance professional: Income correlated with stock market
│   │       │   └─ Also hold more bonds (avoid double exposure)
│   │       └─ Gig worker: Volatile income (stock-like)
│   │           └─ More bonds for stability
│   │
│   ├─ Age-Based Glide Path (Heuristic Rule):
│   │   │ Simple formula: Stock allocation = f(age)
│   │   │
│   │   ├─ Classic rules:
│   │   │   ├─ "100 - age" rule:
│   │   │   │   └─ Age 30 → 70% stocks; Age 60 → 40% stocks
│   │   │   ├─ "110 - age" (modern; longer lifespans):
│   │   │   │   └─ Age 30 → 80% stocks; Age 60 → 50% stocks
│   │   │   └─ "120 - age" (aggressive; long retirements):
│   │   │       └─ Age 30 → 90% stocks; Age 60 → 60% stocks
│   │   │
│   │   ├─ Target-Date Fund Implementation:
│   │   │   ├─ Vanguard Target Retirement 2060:
│   │   │   │   ├─ 2025 (35 years out): 90% stocks
│   │   │   │   ├─ 2045 (15 years out): 70% stocks
│   │   │   │   ├─ 2060 (retirement): 50% stocks
│   │   │   │   └─ 2070 (10 years after): 30% stocks
│   │   │   │
│   │   │   ├─ Fidelity Freedom 2055:
│   │   │   │   └─ Similar; slightly more conservative
│   │   │   │
│   │   │   └─ TIAA-CREF Lifecycle:
│   │   │       └─ Even more conservative (40% stocks at retirement)
│   │   │
│   │   ├─ Glide Path Types:
│   │   │   ├─ "To retirement": De-risk until retirement, then stabilize
│   │   │   │   └─ Pros: Simple; easy to understand
│   │   │   │   └─ Cons: Ignores post-retirement risk capacity
│   │   │   ├─ "Through retirement": Continue de-risking post-retirement
│   │   │   │   └─ Pros: Accounts for longevity risk; sequence risk
│   │   │   │   └─ Cons: May be overly conservative early in retirement
│   │   │   └─ Empirical evidence: "Through" slightly better historically
│   │   │
│   │   └─ Customization by individual:
│   │       ├─ Risk tolerance: High λ → Lower stock allocation
│   │       ├─ Wealth: Wealthy → More conservative (less need for growth)
│   │       ├─ Pension: Defined benefit pension → More stocks (implicit bonds)
│   │       ├─ Social Security: Similar to pension (bond-like income stream)
│   │       └─ Longevity: Expect to live longer → More stocks (longer horizon)
│   │
│   ├─ Rebalancing Along Glide Path:
│   │   ├─ Frequency:
│   │   │   ├─ Annual: Common for target-date funds
│   │   │   ├─ Semi-annual: Balance between cost and discipline
│   │   │   └─ Quarterly: Too frequent (costs outweigh benefits)
│   │   │
│   │   ├─ Mechanics:
│   │   │   ├─ Check current allocation vs target glide path
│   │   │   ├─ If drift > 5%: Rebalance to target
│   │   │   ├─ Sell winners (stocks if outperformed)
│   │   │   ├─ Buy laggards (bonds if underperformed)
│   │   │   └─ Pay transaction costs (0.1-0.3% per rebalancing)
│   │   │
│   │   ├─ Tax considerations:
│   │   │   ├─ Tax-deferred accounts (401k, IRA): Rebalance freely
│   │   │   │   └─ No immediate tax consequences
│   │   │   ├─ Taxable accounts: Tax-loss harvest when rebalancing
│   │   │   │   └─ Sell losers; buy similar (not identical) assets
│   │   │   └─ After-tax: May prefer less frequent rebalancing
│   │   │
│   │   └─ Dollar-cost averaging (implicit rebalancing):
│   │       ├─ Contribute fixed amount each paycheck
│   │       ├─ Buy into underweight asset (automatic rebalancing)
│   │       └─ Zero explicit cost; natural discipline
│   │
│   ├─ Longevity Risk & Sequence Risk:
│   │   ├─ Longevity risk:
│   │   │   ├─ Live longer than expected → Outlive savings
│   │   │   ├─ Solution: Annuities (transfer risk to insurance company)
│   │   │   ├─ Or: Hold more stocks (growth to fund longer retirement)
│   │   │   └─ Social Security: Inflation-indexed lifetime income (valuable!)
│   │   │
│   │   ├─ Sequence risk (sequence-of-returns risk):
│   │   │   ├─ Bad returns early in retirement → Depletes portfolio
│   │   │   │   └─ Forced to sell at low prices (crystallizes losses)
│   │   │   ├─ Example: Retire 2007; crash 2008 → -40% immediately
│   │   │   │   └─ Recovery took 5+ years; meanwhile withdrawing
│   │   │   ├─ Mitigation:
│   │   │   │   ├─ Hold cash bucket (2-3 years expenses)
│   │   │   │   ├─ Reduce withdrawal rate during bear markets
│   │   │   │   ├─ Flexible spending (cut discretionary in downturns)
│   │   │   │   └─ Work part-time if feasible (reduce withdrawals)
│   │   │   └─ Glide path helps: Lower stocks near retirement → Less exposed
│   │   │
│   │   └─ Safe withdrawal rate:
│   │       ├─ 4% rule (Bengen 1994): Withdraw 4% of initial balance
│   │       │   └─ Adjust for inflation annually
│   │       │   └─ Historical success: ~95% (survived 30 years)
│   │       ├─ Conservative: 3-3.5% (safer; lower failure risk)
│   │       ├─ Aggressive: 5-6% (higher risk; may run out)
│   │       └─ Dynamic: Adjust based on portfolio performance
│   │
│   └─ Behavioral Considerations:
│       ├─ Automatic enrollment: Default to target-date fund
│       │   └─ 401(k) plans: 40% of new participants choose target-date
│       ├─ Inertia benefit: Once enrolled, rarely change (good!)
│       │   └─ Prevents panic selling during crashes
│       ├─ Simplicity: One-fund solution (reduces decision fatigue)
│       └─ Fee awareness: Expense ratios 0.1-0.5% (low-cost index-based best)
│
└─ Dynamic vs Static Comparison:
    ├─ Static MPT (constant allocation):
    │   ├─ Pros: Simple; easy to implement; low turnover
    │   ├─ Cons: Ignores age, human capital, changing risk capacity
    │   └─ Suitable: Short horizon; sophisticated investors with stable situations
    │
    ├─ Dynamic Programming (optimal multi-period):
    │   ├─ Pros: Theoretically optimal; accounts for all state variables
    │   ├─ Cons: Complex; requires numerical methods; parameter uncertainty
    │   └─ Suitable: Researchers; institutions with modeling capabilities
    │
    ├─ Lifecycle Glide Path (heuristic):
    │   ├─ Pros: Simple approximation; empirically validated; behaviorally sound
    │   ├─ Cons: Not fully optimal; one-size-fits-all (ignores heterogeneity)
    │   └─ Suitable: Most individual investors; 401(k) plans
    │
    └─ Empirical Evidence:
        ├─ Lifecycle beats static for young investors (captures equity premium)
        ├─ Dynamic programming marginally better than lifecycle (1-2% lifetime wealth)
        ├─ Costs matter: Simple glide path often beats complex dynamic if costs high
        └─ Behavioral benefits: Automatic discipline prevents emotional errors
```

---

## 5. Mini-Project: Simulating Lifecycle Asset Allocation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate lifecycle allocation strategies and compare outcomes

def simulate_lifecycle_path(start_age, retirement_age, death_age, initial_wealth, 
                            annual_contribution, stock_return, bond_return, 
                            stock_vol, bond_vol, correlation, glide_path_type='linear'):
    """
    Simulate wealth accumulation and decumulation over lifetime.
    """
    
    ages = np.arange(start_age, death_age + 1)
    n_periods = len(ages)
    
    # Initialize
    wealth = np.zeros(n_periods)
    wealth[0] = initial_wealth
    stock_allocation = np.zeros(n_periods)
    
    # Glide path rule
    for i, age in enumerate(ages):
        if glide_path_type == 'linear':
            # Linear: 110 - age
            stock_allocation[i] = max(0.2, min(0.95, (110 - age) / 100))
        elif glide_path_type == 'aggressive':
            # Aggressive: 120 - age
            stock_allocation[i] = max(0.3, min(1.0, (120 - age) / 100))
        elif glide_path_type == 'conservative':
            # Conservative: 100 - age
            stock_allocation[i] = max(0.1, min(0.8, (100 - age) / 100))
        else:  # 'static'
            stock_allocation[i] = 0.6  # Fixed 60/40
    
    # Simulate returns
    np.random.seed(42)
    
    for i in range(1, n_periods):
        age = ages[i]
        
        # Portfolio return (with correlation)
        z1 = np.random.normal(0, 1)
        z2 = correlation * z1 + np.sqrt(1 - correlation**2) * np.random.normal(0, 1)
        
        stock_ret = stock_return + stock_vol * z1
        bond_ret = bond_return + bond_vol * z2
        
        portfolio_return = (stock_allocation[i-1] * stock_ret + 
                          (1 - stock_allocation[i-1]) * bond_ret)
        
        # Update wealth
        if age < retirement_age:
            # Accumulation: Add contributions
            wealth[i] = wealth[i-1] * (1 + portfolio_return) + annual_contribution
        else:
            # Decumulation: Withdraw 4% of initial retirement wealth
            if age == retirement_age:
                retirement_wealth = wealth[i-1]
                annual_withdrawal = 0.04 * retirement_wealth
            
            wealth[i] = wealth[i-1] * (1 + portfolio_return) - annual_withdrawal
            wealth[i] = max(0, wealth[i])  # Can't go negative
    
    return ages, wealth, stock_allocation


def monte_carlo_lifecycle(n_simulations, start_age, retirement_age, death_age,
                          initial_wealth, annual_contribution, 
                          stock_return, bond_return, stock_vol, bond_vol, correlation,
                          glide_path_type):
    """
    Run Monte Carlo simulation for lifecycle wealth paths.
    """
    
    ages = np.arange(start_age, death_age + 1)
    n_periods = len(ages)
    
    all_wealth = np.zeros((n_simulations, n_periods))
    
    for sim in range(n_simulations):
        np.random.seed(sim)
        _, wealth, _ = simulate_lifecycle_path(
            start_age, retirement_age, death_age, initial_wealth,
            annual_contribution, stock_return, bond_return,
            stock_vol, bond_vol, correlation, glide_path_type
        )
        all_wealth[sim, :] = wealth
    
    # Statistics
    median_wealth = np.median(all_wealth, axis=0)
    percentile_10 = np.percentile(all_wealth, 10, axis=0)
    percentile_90 = np.percentile(all_wealth, 90, axis=0)
    
    # Terminal wealth at death
    terminal_wealth = all_wealth[:, -1]
    
    return {
        'ages': ages,
        'median_wealth': median_wealth,
        'p10': percentile_10,
        'p90': percentile_90,
        'terminal_wealth': terminal_wealth,
        'all_paths': all_wealth
    }


def dynamic_programming_allocation(age, wealth, human_capital, risk_aversion=4):
    """
    Simplified dynamic programming allocation rule.
    Accounts for human capital and wealth level.
    """
    
    # Total wealth = financial + human capital
    total_wealth = wealth + human_capital
    
    # Optimal allocation (simplified Merton rule with human capital adjustment)
    # If human capital is bond-like, need more stocks in financial portfolio
    
    if total_wealth > 0:
        implicit_bond_fraction = human_capital / total_wealth
        
        # Target overall stock allocation (decreases with age)
        target_overall_stock = max(0.3, min(0.9, (110 - age) / 100))
        
        # Financial portfolio stock allocation
        # Need to offset implicit bonds from human capital
        financial_stock_alloc = (target_overall_stock - implicit_bond_fraction * 0) / (1 - implicit_bond_fraction)
        financial_stock_alloc = max(0.2, min(1.0, financial_stock_alloc))
    else:
        financial_stock_alloc = 0.6  # Default
    
    return financial_stock_alloc


# Main Analysis
print("=" * 100)
print("DYNAMIC PROGRAMMING & LIFECYCLE ASSET ALLOCATION")
print("=" * 100)

# 1. Parameters
print("\n1. SIMULATION PARAMETERS")
print("-" * 100)

start_age = 25
retirement_age = 65
death_age = 90

initial_wealth = 50000
annual_contribution = 10000

stock_return = 0.08
bond_return = 0.04
stock_vol = 0.18
bond_vol = 0.05
correlation = 0.3

print(f"\nInvestor Profile:")
print(f"  Starting age: {start_age}")
print(f"  Retirement age: {retirement_age}")
print(f"  Life expectancy: {death_age}")
print(f"  Initial wealth: ${initial_wealth:,}")
print(f"  Annual contribution: ${annual_contribution:,}")

print(f"\nMarket Assumptions:")
print(f"  Stock return: {stock_return*100:.1f}% (vol: {stock_vol*100:.1f}%)")
print(f"  Bond return: {bond_return*100:.1f}% (vol: {bond_vol*100:.1f}%)")
print(f"  Correlation: {correlation:.2f}")

# 2. Single Path Examples
print("\n2. GLIDE PATH COMPARISON (Single Simulation)")
print("-" * 100)

glide_types = ['linear', 'aggressive', 'conservative', 'static']
glide_results = {}

for glide_type in glide_types:
    ages, wealth, stock_alloc = simulate_lifecycle_path(
        start_age, retirement_age, death_age, initial_wealth,
        annual_contribution, stock_return, bond_return,
        stock_vol, bond_vol, correlation, glide_type
    )
    glide_results[glide_type] = {
        'ages': ages,
        'wealth': wealth,
        'stock_alloc': stock_alloc
    }

print(f"\nTerminal Wealth Comparison (Single Path):")
print(f"{'Strategy':<20} {'Terminal Wealth':<20} {'Retirement Wealth':<20}")
print("-" * 60)

for glide_type in glide_types:
    terminal = glide_results[glide_type]['wealth'][-1]
    retirement_idx = retirement_age - start_age
    retirement_wealth = glide_results[glide_type]['wealth'][retirement_idx]
    
    label = glide_type.capitalize()
    print(f"{label:<20} ${terminal:>15,.0f} ${retirement_wealth:>18,.0f}")

# 3. Monte Carlo Simulation
print("\n3. MONTE CARLO ANALYSIS (1000 Simulations)")
print("-" * 100)

n_sims = 1000
mc_results = {}

for glide_type in ['linear', 'static']:
    mc_results[glide_type] = monte_carlo_lifecycle(
        n_sims, start_age, retirement_age, death_age,
        initial_wealth, annual_contribution,
        stock_return, bond_return, stock_vol, bond_vol, correlation,
        glide_type
    )

print(f"\nTerminal Wealth Statistics:")
print(f"{'Strategy':<20} {'Median':<18} {'10th %ile':<18} {'90th %ile':<18} {'Shortfall Risk':<15}")
print("-" * 89)

for glide_type in ['linear', 'static']:
    terminal = mc_results[glide_type]['terminal_wealth']
    median = np.median(terminal)
    p10 = np.percentile(terminal, 10)
    p90 = np.percentile(terminal, 90)
    shortfall = np.mean(terminal < 500000) * 100  # % below $500k
    
    label = 'Lifecycle' if glide_type == 'linear' else 'Static 60/40'
    print(f"{label:<20} ${median:>15,.0f} ${p10:>15,.0f} ${p90:>15,.0f} {shortfall:>13.1f}%")

# 4. Human Capital Analysis
print("\n4. HUMAN CAPITAL IMPACT")
print("-" * 100)

ages_hc = [25, 35, 45, 55, 65]
salary = 100000
growth_rate = 0.03
discount_rate = 0.04

print(f"\nHuman Capital by Age (Salary: ${salary:,}, Growth: {growth_rate*100:.1f}%):")
print(f"{'Age':<10} {'Years to Retire':<18} {'Human Capital':<20} {'HC as % Total Wealth':<25}")
print("-" * 73)

for age in ages_hc:
    years_remaining = max(0, retirement_age - age)
    
    if years_remaining > 0:
        # PV of growing annuity
        human_capital = salary * ((1 - ((1 + growth_rate) / (1 + discount_rate)) ** years_remaining) 
                                 / (discount_rate - growth_rate))
    else:
        human_capital = 0
    
    # Assume financial wealth grows over time
    financial_wealth = initial_wealth * (1.06 ** (age - start_age)) + annual_contribution * (age - start_age)
    
    total_wealth = financial_wealth + human_capital
    hc_fraction = human_capital / total_wealth * 100 if total_wealth > 0 else 0
    
    print(f"{age:<10} {years_remaining:<18} ${human_capital:>18,.0f} {hc_fraction:>23.1f}%")

# 5. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Glide Paths
ax = axes[0, 0]

for glide_type, label in [('linear', 'Linear (110-age)'), 
                           ('aggressive', 'Aggressive (120-age)'),
                           ('conservative', 'Conservative (100-age)'),
                           ('static', 'Static 60/40')]:
    ages = glide_results[glide_type]['ages']
    stock_alloc = glide_results[glide_type]['stock_alloc']
    ax.plot(ages, stock_alloc * 100, linewidth=2.5, label=label, alpha=0.8)

ax.axvline(x=retirement_age, color='red', linestyle='--', alpha=0.5, label='Retirement')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Stock Allocation (%)', fontsize=12)
ax.set_title('Lifecycle Glide Paths: Stock Allocation Over Time', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Wealth Accumulation Paths
ax = axes[0, 1]

for glide_type, label, color in [('linear', 'Lifecycle', '#2ecc71'),
                                  ('static', 'Static 60/40', '#3498db')]:
    ages = glide_results[glide_type]['ages']
    wealth = glide_results[glide_type]['wealth']
    ax.plot(ages, wealth / 1e6, linewidth=2.5, label=label, color=color, alpha=0.8)

ax.axvline(x=retirement_age, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Wealth ($ Millions)', fontsize=12)
ax.set_title('Wealth Accumulation Over Lifetime (Single Path)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Monte Carlo Fan Chart (Lifecycle)
ax = axes[1, 0]

ages = mc_results['linear']['ages']
median = mc_results['linear']['median_wealth']
p10 = mc_results['linear']['p10']
p90 = mc_results['linear']['p90']

ax.plot(ages, median / 1e6, linewidth=2.5, label='Median', color='#2ecc71')
ax.fill_between(ages, p10 / 1e6, p90 / 1e6, alpha=0.3, color='#2ecc71', label='10-90th percentile')

ax.axvline(x=retirement_age, color='red', linestyle='--', alpha=0.5, label='Retirement')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Wealth ($ Millions)', fontsize=12)
ax.set_title('Lifecycle Strategy: Monte Carlo Uncertainty (1000 Paths)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Terminal Wealth Distribution
ax = axes[1, 1]

terminal_lifecycle = mc_results['linear']['terminal_wealth'] / 1e6
terminal_static = mc_results['static']['terminal_wealth'] / 1e6

ax.hist(terminal_lifecycle, bins=50, alpha=0.6, label='Lifecycle', color='#2ecc71', edgecolor='black')
ax.hist(terminal_static, bins=50, alpha=0.6, label='Static 60/40', color='#3498db', edgecolor='black')

ax.axvline(x=np.median(terminal_lifecycle), color='#2ecc71', linestyle='--', linewidth=2)
ax.axvline(x=np.median(terminal_static), color='#3498db', linestyle='--', linewidth=2)

ax.set_xlabel('Terminal Wealth ($ Millions)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Terminal Wealth Distribution at Age 90', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lifecycle_allocation_simulation.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: lifecycle_allocation_simulation.png")
plt.show()

# 6. Key Insights
print("\n5. KEY INSIGHTS")
print("=" * 100)
print("""
LIFECYCLE ALLOCATION SUPERIORITY:
├─ Young investors: Lifecycle allocates 80-90% stocks (captures equity premium)
├─ Static 60/40: Foregoes ~2-3% annual return during accumulation phase
├─ Terminal wealth: Lifecycle median 15-25% higher than static (compound effect)
└─ Risk management: De-risks automatically as approach retirement

HUMAN CAPITAL CRITICAL:
├─ Age 25: Human capital 85%+ of total wealth (implicit bonds)
├─ Age 55: Human capital ~30% of total wealth (depleting)
├─ Implication: Young should hold 90%+ stocks in financial portfolio
└─ Ignoring human capital creates massive under-allocation to equities

GLIDE PATH COMPARISON:
├─ Aggressive (120-age): Best for high-earners, long horizons
├─ Linear (110-age): Standard recommendation; good balance
├─ Conservative (100-age): Suitable for low risk tolerance, health issues
└─ Static 60/40: Suboptimal for young; acceptable for near-retirees

SEQUENCE RISK MITIGATION:
├─ Lifecycle reduces stocks at retirement → Less exposed to early bear market
├─ Cash bucket (2-3 years expenses) provides buffer
├─ Dynamic withdrawal rates better than fixed 4% rule
└─ Part-time work in early retirement valuable (reduces withdrawals)

BEHAVIORAL BENEFITS:
├─ Target-date funds: Automatic; prevents panic selling
├─ Inertia: Once set, investors rarely change (good discipline)
├─ Simplicity: One-fund solution; reduces decision fatigue
└─ Empirically validated: ~40% of 401(k) participants choose target-date

PRACTICAL RECOMMENDATIONS:
├─ Age 25-40: 80-90% stocks (human capital cushion)
├─ Age 40-55: 60-75% stocks (balanced growth/preservation)
├─ Age 55-65: 40-60% stocks (de-risking for retirement)
├─ Age 65+: 30-40% stocks (longevity risk; maintain some growth)
└─ Adjust for individual: risk tolerance, wealth, pension, health
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Human Capital Valuation:** How do you value human capital for someone with volatile income (entrepreneur, commission-based sales)? Should it be treated as bond-like or stock-like? How does this change optimal allocation?

2. **Dynamic Programming Complexity:** Solve simple 3-period Bellman equation with log utility, two assets (stocks/bonds). What is optimal allocation in each period as function of wealth? Does it match lifecycle heuristic?

3. **Glide Path Customization:** 50-year-old with $2M portfolio, $100k defined-benefit pension, high risk tolerance. Should follow standard glide path (60% stocks) or deviate? Why?

4. **Rebalancing Frequency Trade-off:** Annual rebalancing costs 0.2%; adds 0.8% benefit. Should rebalance? What if costs 0.5%? At what cost threshold does rebalancing hurt?

5. **Sequence Risk Mitigation:** Retiree with $1M portfolio, needs $40k/year (4%). Market crashes -30% year 1. Better to: (a) sell stocks at loss to meet withdrawal, (b) reduce spending to $30k, (c) work part-time, (d) other?

---

## 7. Key References

- **Merton, R.C. (1969).** "Lifetime Portfolio Selection Under Uncertainty" – Dynamic programming solution; continuous-time optimization.

- **Merton, R.C. (1973).** "Intertemporal Capital Asset Pricing Model" – ICAPM; hedging demands for changing investment opportunities.

- **Bodie, Z., Merton, R.C., & Samuelson, W.F. (1992).** "Labor Supply Flexibility and Portfolio Choice" – Human capital as implicit bond.

- **Cocco, J.F., Gomes, F.J., & Maenhout, P.J. (2005).** "Consumption and Portfolio Choice over the Life Cycle" – Numerical dynamic programming; lifecycle results.

- **Bengen, W.P. (1994).** "Determining Withdrawal Rates Using Historical Data" – 4% safe withdrawal rate; sequence risk.

- **Vanguard Research (2013).** "Target-Date Funds: A Decade of Growth and Change" – Empirical evidence on glide paths; investor behavior.

- **Pfau, W. (2012).** "A Broader Framework for Determining an Efficient Frontier for Retirement" – Retirement income strategies; dynamic withdrawal.

