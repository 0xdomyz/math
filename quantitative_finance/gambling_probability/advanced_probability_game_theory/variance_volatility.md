# Variance & Volatility

## 1. Concept Skeleton
**Definition:** Measure of outcome dispersion around mean; quantifies risk, swings, and unpredictability of betting results  
**Purpose:** Assess bankroll fluctuations, guide risk tolerance, compare strategies with same EV, predict drawdowns  
**Prerequisites:** Expected value, probability, standard deviation, distribution theory

## 2. Comparative Framing
| Metric | Variance | Standard Deviation | Sharpe Ratio | Risk of Ruin |
|--------|----------|------------------|-------------|-------------|
| **Measures** | Squared dispersion | Dispersion in units | Risk-adjusted return | Bankruptcy probability |
| **Formula** | Var = E[(X-μ)²] | σ = √Var | (return - rf) / σ | Depends on variance + edge |
| **Interpretation** | Spread magnitude | Typical deviation | Return per unit risk | Extreme outcome |
| **Sign** | Always ≥ 0 | Always ≥ 0 | Can be negative | 0 to 100% |

## 3. Examples + Counterexamples

**Simple Example:**  
Bet A: 55% to win \$100, 45% to lose \$100. Bet B: 55% to win \$1, 45% to lose \$1. Both have same positive EV, but B is lower variance.

**Failure Case:**  
Ignoring variance with favorable EV. High-variance bet can bankrupt before law of large numbers (LLN) materialized.

**Edge Case:**  
Lottery: Negative EV (-50%) but massive variance. Some win big; most lose everything. Variance obscures poor expected value.

## 4. Layer Breakdown
```
Variance & Volatility Framework:
├─ Basic Formulas:
│   ├─ Variance: Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
│   ├─ Standard Dev: σ = √Var
│   ├─ Coefficient of Variation: CV = σ / μ (risk per unit return)
│   └─ Example: Bet with 55% to win $110, 45% to lose $100
│       E(X) = +$9.50
│       Var = 0.55(110-9.5)² + 0.45(-100-9.5)² ≈ $10,300
│       σ ≈ $101.50
├─ Volatility Types:
│   ├─ Session volatility: Variance within single session
│   ├─ Long-term volatility: Variance over many sessions
│   ├─ Clustering: Streaks (not randomly distributed variance)
│   ├─ Autocorrelation: Outcomes dependent on prior results
│   └─ Tail risk: Extreme outcomes beyond typical distribution
├─ Impact on Bankroll:
│   ├─ High variance + positive EV: Large swings, slower convergence to mean
│   │   Need large bankroll to survive downswings
│   ├─ Low variance + positive EV: Smooth growth, predictable
│   │   Can operate with smaller bankroll
│   ├─ High variance + negative EV: Ruin certain, just slower
│   ├─ Low variance + negative EV: Slow bleed, predictable loss
│   └─ Trade-off: Cannot minimize both variance and maximize EV
├─ Measuring Volatility:
│   ├─ Standard deviation: Single-game volatility
│   ├─ Drawdown: Peak-to-trough decline (max loss from high)
│   ├─ Win rate + avg win/loss: Components of total variance
│   ├─ Skewness: Asymmetry (heavy left tail vs right tail)
│   └─ Kurtosis: Tail fatness (extreme outcomes frequency)
├─ Relating EV to Variance:
│   ├─ Sharpe Ratio: Return / σ (efficiency metric)
│   ├─ Sortino Ratio: Return / downside σ (ignores upside)
│   ├─ Calmar Ratio: Return / max drawdown
│   └─ Superior strategy: High EV AND low variance
└─ Kelly Criterion Connection:
    ├─ f* = (edge) / variance (approximation)
    ├─ Higher variance → smaller Kelly fraction → lower bet
    ├─ Fractional Kelly: Reduce variance by betting less
    └─ Trade-off: Less variance reduces ruin risk but slows growth
```

## 5. Mini-Project
Analyze variance in different betting scenarios:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_variance_stats(prob_win, payout, bet=1, num_trials=100000):
    """
    Calculate variance and related stats for betting scenario
    """
    outcomes = []
    for _ in range(num_trials):
        if np.random.random() < prob_win:
            outcomes.append(payout - bet)
        else:
            outcomes.append(-bet)
    
    outcomes = np.array(outcomes)
    
    ev = np.mean(outcomes)
    var = np.var(outcomes)
    std_dev = np.std(outcomes)
    skewness = stats.skew(outcomes)
    kurtosis_val = stats.kurtosis(outcomes)
    
    # Drawdown analysis
    cumsum = np.cumsum(outcomes)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_drawdown = np.max(drawdown)
    
    return {
        'ev': ev,
        'var': var,
        'std': std_dev,
        'skew': skewness,
        'kurt': kurtosis_val,
        'max_dd': max_drawdown,
        'outcomes': outcomes,
        'cumsum': cumsum
    }

# Example 1: Compare bets with same EV, different variance
print("=== Same EV, Different Variance ===\n")

np.random.seed(42)

# Bet A: 55% to win $100, 45% to lose $100 (High variance)
bet_a_stats = calculate_variance_stats(0.55, 100, 100)

# Bet B: 51% to win $10, 49% to lose $10 (Low variance, similar EV)
bet_b_stats = calculate_variance_stats(0.51, 10, 10)

# Bet C: 60% to win $500, 40% to lose $300 (Very high variance)
bet_c_stats = calculate_variance_stats(0.60, 500, 300)

print(f"{'Bet':<10} {'EV ($)':<12} {'Variance':<15} {'Std Dev':<12} {'Sharpe':<12}")
print("-" * 60)

for name, stats_obj in [('A', bet_a_stats), ('B', bet_b_stats), ('C', bet_c_stats)]:
    sharpe = stats_obj['ev'] / stats_obj['std'] if stats_obj['std'] > 0 else 0
    print(f"{name:<10} {stats_obj['ev']:<12.2f} {stats_obj['var']:<15.0f} {stats_obj['std']:<12.2f} {sharpe:<12.4f}")

# Example 2: Drawdown analysis
print("\n\n=== Drawdown Analysis ===\n")

print(f"{'Bet':<10} {'Max Drawdown ($)':<20} {'Max Drawdown %':<15} {'Prob of Ruin':<15}")
print("-" * 60)

for name, stats_obj in [('A', bet_a_stats), ('B', bet_b_stats), ('C', bet_c_stats)]:
    outcomes = stats_obj['outcomes']
    max_dd = stats_obj['max_dd']
    max_dd_pct = (max_dd / 1000) * 100 if max_dd > 0 else 0
    ruin_rate = np.sum(np.cumsum(outcomes) <= 0) / len(outcomes) * 100
    print(f"{name:<10} ${max_dd:<19.0f} {max_dd_pct:<15.1f}% {ruin_rate:<15.2f}%")

# Example 3: Volatility scaling
print("\n\n=== Volatility Scaling with Number of Bets ===\n")

# As you place more bets, variance compounds
num_bets_range = np.array([1, 10, 100, 1000, 10000])
ev_per_bet = 9.50  # Bet A EV

print(f"{'Num Bets':<15} {'Expected Total':<20} {'Std Dev Total':<20} {'EV/SD Ratio':<15}")
print("-" * 70)

for n in num_bets_range:
    # For independent bets: Var_total = n * Var_single
    var_total = n * bet_a_stats['var']
    std_total = np.sqrt(var_total)
    ev_total = n * ev_per_bet
    ratio = ev_total / std_total if std_total > 0 else 0
    print(f"{n:<15} ${ev_total:<19,.0f} ${std_total:<19,.0f} {ratio:<15.2f}")

# Example 4: Volatility clustering
print("\n\n=== Win Rate Impact on Variance ===\n")

win_rates = [0.4, 0.5, 0.55, 0.6, 0.7]
payout = 2.0  # Even odds for simplicity

print(f"{'Win %':<15} {'EV ($)':<15} {'Variance':<15} {'Std Dev':<15} {'EV/SD':<15}")
print("-" * 75)

for wr in win_rates:
    stats_wr = calculate_variance_stats(wr, payout, 1, 10000)
    print(f"{wr*100:<14.0f}% {stats_wr['ev']:<15.4f} {stats_wr['var']:<15.4f} {stats_wr['std']:<15.4f} {stats_wr['ev']/stats_wr['std']:<15.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution comparison
axes[0, 0].hist(bet_a_stats['outcomes'], bins=50, alpha=0.5, label='Bet A (High Var)', color='red', density=True)
axes[0, 0].hist(bet_b_stats['outcomes'], bins=50, alpha=0.5, label='Bet B (Low Var)', color='green', density=True)
axes[0, 0].axvline(bet_a_stats['ev'], color='darkred', linestyle='--', linewidth=2)
axes[0, 0].axvline(bet_b_stats['ev'], color='darkgreen', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Outcome ($)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution: Same EV, Different Variance')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Cumulative outcomes (sample path)
sample_idx = np.arange(min(1000, len(bet_a_stats['cumsum'])))
axes[0, 1].plot(sample_idx, bet_a_stats['cumsum'][:1000], alpha=0.7, label='Bet A', color='red', linewidth=1)
axes[0, 1].plot(sample_idx, bet_b_stats['cumsum'][:1000], alpha=0.7, label='Bet B', color='green', linewidth=1)
axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[0, 1].set_xlabel('Bet Number')
axes[0, 1].set_ylabel('Cumulative Outcome ($)')
axes[0, 1].set_title('Sample Paths: High vs Low Variance')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Variance growth with sample size
num_bets_plot = np.logspace(0, 4, 50)
std_growth = np.sqrt(bet_a_stats['var'] * num_bets_plot)

axes[1, 0].loglog(num_bets_plot, std_growth, linewidth=2, color='darkblue')
axes[1, 0].set_xlabel('Number of Bets (log scale)')
axes[1, 0].set_ylabel('Standard Deviation (log scale)')
axes[1, 0].set_title('Variance Grows with √n')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Coefficient of Variation (risk per unit EV)
bets_comparison = ['Bet A\nHigh Var', 'Bet B\nLow Var', 'Bet C\nVery High', 'Fair Coin']
evs_comp = [bet_a_stats['ev'], bet_b_stats['ev'], bet_c_stats['ev'], 0]
stds_comp = [bet_a_stats['std'], bet_b_stats['std'], bet_c_stats['std'], 1.0]
cv_comp = [std / abs(ev) if abs(ev) > 0.01 else float('inf') for ev, std in zip(evs_comp, stds_comp)]

colors_cv = ['red', 'green', 'darkred', 'gray']
axes[1, 1].bar(bets_comparison, cv_comp, color=colors_cv, alpha=0.7)
axes[1, 1].set_ylabel('Coefficient of Variation (σ/EV)')
axes[1, 1].set_title('Risk Per Unit Return')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is variance analysis insufficient?
- Non-normal distributions (skew/kurtosis matter more)
- Correlated outcomes (variance compounds differently)
- Time-dependent volatility (calm periods then spikes)
- Changing game conditions (variance not constant)
- Rare catastrophic events (tail risk beyond typical variance)

## 7. Key References
- [Wikipedia: Variance](https://en.wikipedia.org/wiki/Variance)
- [Standard Deviation in Finance](https://www.investopedia.com/terms/s/standarddeviation.asp)
- [Sharpe Ratio Explained](https://www.investopedia.com/terms/s/sharperatio.asp)

---
**Status:** Risk quantification metric | **Complements:** Expected Value, Kelly Criterion, Bankroll Management
