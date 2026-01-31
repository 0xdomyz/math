# Risk of Ruin

## 1. Concept Skeleton
**Definition:** Probability of losing entire bankroll before achieving profit target through adverse outcomes and variance  
**Purpose:** Quantify catastrophic loss risk, determine minimum bankroll for strategy, guide bet sizing  
**Prerequisites:** Probability, expected value, variance, random walks

## 2. Comparative Framing
| Concept | Risk of Ruin | Expected Value | Variance | Optimal f |
|---------|-------------|-----------------|----------|-----------|
| **Measures** | Disaster probability | Long-term average | Outcome spread | Bet sizing |
| **Time Horizon** | Until bust | Per bet | Per bet | Portfolio |
| **Formula** | (q/p)^B or numerical | EV = Σ(p×outcome) | Var = E[X²] - E[X]² | Kelly Criterion |
| **Interpretation** | Risk tolerance | Profitability | Volatility/swings | Bankroll growth |

## 3. Examples + Counterexamples

**Simple Example:**  
Positive EV bet (+5%) with edge. If bankroll $1000, single $100 bet has Risk of Ruin ~0.5% over sequence.

**Failure Case:**  
Ignoring variance in favorable games. +5% EV bet with high variance can bankrupt underfunded player before law of large numbers kicks in.

**Edge Case:**  
Negative EV game (e.g., casino). With any strategy, Risk of Ruin = 100% eventually (negative drift always wins).

## 4. Layer Breakdown
```
Risk of Ruin Framework:
├─ Mathematical Foundation:
│   ├─ Gambler's Ruin (simple): p and q fixed, random walk
│   │   P(ruin | starting B) = (q/p)^B if p ≠ 0.5
│   │   P(ruin | starting B) = 1 if p ≤ 0.5 (unfavorable)
│   ├─ Parameters:
│   │   B = current bankroll
│   │   p = win probability per bet
│   │   q = 1 - p (loss probability)
│   │   W = goal/target amount
│   └─ Adjustment for goal: P(ruin before reaching B+W)
├─ Advanced Models:
│   ├─ With variable stakes: Numerical integration needed
│   ├─ With Kelly sizing: Lower ruin risk than fixed bet
│   ├─ With multiple bets: Correlation affects ruin probability
│   └─ Continuous time: Brownian motion approximation
├─ Factors Affecting Ruin:
│   ├─ Edge (p > 0.5): Favorable outcomes, ↓ ruin risk
│   ├─ Bet size: Larger bets → ↑ ruin risk (quadratic effect)
│   ├─ Variance: Higher variance → ↑ ruin risk
│   ├─ Bankroll: Larger bankroll → ↓ ruin risk (exponential)
│   └─ Time horizon: More bets → Law of Large Numbers dominates
├─ Practical Application:
│   ├─ Minimum bankroll: B_min = -log(acceptable_ruin_prob) / edge
│   ├─ Session limits: Stop-loss prevents catastrophic ruin
│   ├─ Compounding: Reinvest profits → exponential growth/ruin
│   └─ Diversification: Multiple strategies → reduced correlation risk
└─ Simulation Approach:
    ├─ Monte Carlo: Simulate thousands of bankroll paths
    ├─ Track: How many paths hit $0 before reaching goal
    ├─ Percentage: (Paths to ruin) / (Total paths) ≈ RoR
    └─ Refine: Increase simulations for better estimates
```

## 5. Mini-Project
Calculate and visualize risk of ruin:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Risk of Ruin Formula (simple case)
def risk_of_ruin_formula(p, q, bankroll):
    """
    Closed-form solution for gambler's ruin problem
    p: probability of winning each bet
    q: probability of losing each bet (1-p)
    bankroll: starting bankroll in units of bet size
    """
    if p == 0.5:
        return 1 / (1 + bankroll)
    else:
        return (q/p) ** bankroll

def risk_of_ruin_with_goal(p, q, bankroll, goal):
    """Risk of ruin before reaching goal (bankroll + goal)"""
    if p == 0.5:
        return (goal) / (bankroll + goal)
    else:
        ratio = (q/p)
        ruin = (ratio**(bankroll + goal) - ratio**goal) / (ratio**(bankroll + goal) - 1)
        return ruin

# Example 1: Simple comparison
print("=== Risk of Ruin: Formula Calculations ===\n")

# Scenario: 52% win rate (2% edge)
p = 0.52
q = 0.48

bankrolls = [10, 20, 50, 100, 200]
print(f"Win probability: {p:.1%}, Loss probability: {q:.1%}\n")
print(f"{'Bankroll (units)':<20} {'Risk of Ruin':<15} {'Probability':<15}")
print("-" * 50)

for b in bankrolls:
    ror = risk_of_ruin_formula(p, q, b)
    print(f"{b:<20} {ror:<15.4f} ({ror*100:<6.2f}%)")

# Example 2: Kelly Criterion comparison
print("\n\n=== Edge Effect on Risk of Ruin ===\n")

edges = [0.01, 0.02, 0.05, 0.10]
bankroll = 50

print(f"Bankroll: {bankroll} units\n")
print(f"{'Edge':<10} {'Win Prob':<12} {'Risk of Ruin':<15}")
print("-" * 40)

for edge in edges:
    p_edge = 0.5 + edge/2
    q_edge = 1 - p_edge
    ror = risk_of_ruin_formula(p_edge, q_edge, bankroll)
    print(f"{edge*100:<6.1f}% {p_edge:<12.1%} {ror:<15.4f}")

# Example 3: Monte Carlo Simulation
print("\n\n=== Monte Carlo Simulation ===\n")

def simulate_bankroll_paths(p, initial_bankroll, num_bets, num_simulations=10000):
    """Simulate bankroll paths and return percentage reaching ruin"""
    paths = np.zeros((num_simulations, num_bets + 1))
    paths[:, 0] = initial_bankroll
    
    for sim in range(num_simulations):
        for bet in range(num_bets):
            if paths[sim, bet] <= 0:
                paths[sim, bet+1:] = 0
            else:
                outcome = np.random.binomial(1, p)
                if outcome == 1:
                    paths[sim, bet + 1] = paths[sim, bet] + 1
                else:
                    paths[sim, bet + 1] = paths[sim, bet] - 1
    
    ruin_count = np.sum(np.min(paths, axis=1) <= 0)
    ruin_prob = ruin_count / num_simulations
    
    return paths, ruin_prob

# Parameters
p_sim = 0.51  # 51% win rate (1% edge)
initial_bank = 30
num_bets_sim = 100
num_sims = 10000

paths, ror_sim = simulate_bankroll_paths(p_sim, initial_bank, num_bets_sim, num_sims)

# Compare to formula
ror_formula = risk_of_ruin_formula(p_sim, 1-p_sim, initial_bank)

print(f"Win probability: {p_sim:.1%}")
print(f"Initial bankroll: {initial_bank} units")
print(f"Number of bets: {num_bets_sim}")
print(f"Simulations: {num_sims}\n")
print(f"Formula RoR: {ror_formula:.4f}")
print(f"Simulated RoR: {ror_sim:.4f}")
print(f"Difference: {abs(ror_formula - ror_sim):.4f}")

# Example 4: Minimum bankroll recommendation
print("\n\n=== Minimum Bankroll Recommendation ===\n")

target_ruin_prob = 0.01  # 1% acceptable risk
edge_pct = 0.02  # 2% edge

# Estimate required bankroll
p_req = 0.5 + edge_pct / 2
q_req = 1 - p_req
ratio = q_req / p_req
min_bankroll = -np.log(target_ruin_prob) / np.log(ratio)

print(f"Target acceptable Risk of Ruin: {target_ruin_prob:.1%}")
print(f"Edge: {edge_pct:.1%}")
print(f"Recommended minimum bankroll: {min_bankroll:.0f} bet units")
print(f"Verification: RoR = {risk_of_ruin_formula(p_req, q_req, min_bankroll):.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RoR vs Bankroll for different edges
bankrolls_range = np.arange(5, 201, 5)
edges_plot = [0.01, 0.02, 0.05, 0.10]
colors = ['red', 'orange', 'yellow', 'green']

for edge, color in zip(edges_plot, colors):
    p_plot = 0.5 + edge/2
    q_plot = 1 - p_plot
    rors = [risk_of_ruin_formula(p_plot, q_plot, b) for b in bankrolls_range]
    axes[0, 0].semilogy(bankrolls_range, rors, linewidth=2, label=f'{edge*100:.1f}% edge', color=color)

axes[0, 0].axhline(0.01, color='black', linestyle='--', alpha=0.5, label='1% threshold')
axes[0, 0].set_xlabel('Bankroll (bet units)')
axes[0, 0].set_ylabel('Risk of Ruin (log scale)')
axes[0, 0].set_title('Risk of Ruin vs Bankroll Size')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Simulated paths
sample_paths = np.random.choice(num_sims, 100, replace=False)
for path_idx in sample_paths:
    color = 'red' if np.min(paths[path_idx]) <= 0 else 'green'
    alpha = 0.1 if np.min(paths[path_idx]) <= 0 else 0.05
    axes[0, 1].plot(paths[path_idx], color=color, alpha=alpha, linewidth=0.5)

axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Bet Number')
axes[0, 1].set_ylabel('Bankroll')
axes[0, 1].set_title(f'Simulated Bankroll Paths\n(Red = Ruined, Green = Survived)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Distribution of final outcomes
final_bankrolls = paths[:, -1]
axes[1, 0].hist(final_bankrolls[final_bankrolls > 0], bins=50, alpha=0.7, color='green', label='Surviving')
axes[1, 0].hist(final_bankrolls[final_bankrolls <= 0], bins=10, alpha=0.7, color='red', label='Ruined')
axes[1, 0].set_xlabel('Final Bankroll')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Distribution of Final Bankrolls\n(p={p_sim:.1%})')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: RoR sensitivity to edge
edges_sensitivity = np.linspace(0.001, 0.10, 50)
bankroll_sensitivity = 50
rors_sensitivity = []

for edge_s in edges_sensitivity:
    p_s = 0.5 + edge_s / 2
    q_s = 1 - p_s
    rors_sensitivity.append(risk_of_ruin_formula(p_s, q_s, bankroll_sensitivity))

axes[1, 1].plot(edges_sensitivity * 100, rors_sensitivity, linewidth=2, color='darkblue')
axes[1, 1].fill_between(edges_sensitivity * 100, rors_sensitivity, alpha=0.3)
axes[1, 1].set_xlabel('Edge (%)')
axes[1, 1].set_ylabel('Risk of Ruin')
axes[1, 1].set_title(f'RoR Sensitivity to Edge\n(Bankroll = {bankroll_sensitivity} units)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is risk of ruin analysis insufficient?
- Highly correlated bets (assumed independence violated)
- Skill-based games (win rate not constant)
- Regression to mean (if edge shifts over time)
- Catastrophic events (rare but larger losses)
- Stopping rules not accounted for (emotional stop-loss, etc.)

## 7. Key References
- [Wikipedia: Gambler's Ruin](https://en.wikipedia.org/wiki/Gambler%27s_ruin)
- [Risk of Ruin Calculator](https://www.betfairblog.com/calculator/)
- [Catalan Numbers and Ruin Probability](https://en.wikipedia.org/wiki/Catalan_number)

---
**Status:** Catastrophic loss quantifier | **Complements:** Kelly Criterion, Bankroll Management, Variance
