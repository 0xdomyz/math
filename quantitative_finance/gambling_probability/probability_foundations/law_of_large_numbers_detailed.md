# Law of Large Numbers

## 1. Concept Skeleton
**Definition:** As sample size increases, empirical frequency converges to theoretical probability  
**Purpose:** Guarantee that casino edge and player disadvantage becomes realized long-term  
**Prerequisites:** Probability, variance, convergence concepts

## 2. Comparative Framing
| Concept | Law of Large Numbers | Central Limit Theorem | Regression to Mean | Gambler's Fallacy |
|---------|---------------------|----------------------|-------------------|------------------|
| **Driver** | Sample mean → population mean | Sample mean distribution normalizes | Extremes revert average | Misconception of LLN |
| **Time Horizon** | Very long-term (N→∞) | Applicable at all scales | Medium-term | Misapplies LLN |
| **Application** | Casino guarantee | Bet sizing, confidence | Streak ending expectation | False belief in trend reversal |
| **Insight** | Math guarantees house wins | Volatility predictable | Temporary deviations expected | Past streaks don't affect future |

## 3. Examples + Counterexamples

**Example (Roulette):**  
1,000 spins ≈ 48.6% red (near true 48.65%); 1 million spins → even closer.

**Example (House Edge Realization):**  
Martingale bettor loses →0.52% over millions of bets despite occasional wins.

**Counterexample (Short Term):**  
100 spins can be 55% red by chance, violating LLN temporarily.

## 4. Layer Breakdown
```
Law of Large Numbers:
├─ Mathematical Statement:
│  ├─ Weak LLN: Sample mean converges in probability
│  ├─ Strong LLN: Sample mean converges almost surely
│  └─ Rate of convergence: Depends on variance
├─ Gambling Implications:
│  ├─ House edge WILL be realized with enough volume
│  ├─ Variance decreases as √N (slower than you think)
│  ├─ Short-term luck can dominate for extended periods
│  └─ Professional requires massive capital to survive variance
├─ Statistical Measurement:
│  ├─ Convergence rate: Standard error = σ / √N
│  ├─ 95% confidence interval shrinks with N
│  ├─ Sample size needed grows quadratically with accuracy
│  └─ Small edge = need enormous N to detect
├─ Practical Limits:
│  ├─ Human lifetime: 50 years, ~500 million card hands
│  ├─ If edge = 0.5%, LLN takes centuries to guarantee realization
│  ├─ Variance can exceed lifetime earning capacity
│  └─ Bankroll must sustain variance until LLN manifests
├─ Misapplications:
│  ├─ Gambler's fallacy: LLN does NOT imply short-term balancing
│  ├─ "Due for a loss": False; past does not affect future probability
│  ├─ Trend persistence: LLN contradicts mean reversion beliefs
│  └─ Skill mastery: LLN requires massive sample sizes to establish edge
└─ Distinction from Gambler's Fallacy:
   ├─ LLN: Long-term frequencies approach true probability
   ├─ Fallacy: Belief short-term deviations must correct immediately
   ├─ Math supports LLN, contradicts fallacy
   └─ Real-world: LLN dominates, fallacy misleads over short horizons
```

**Interaction:** More hands dealt → sample outcomes → closer to house edge → casino certainty.

## 5. Mini-Project
Simulate and visualize convergence to house edge:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Non-obvious: small edge requires massive N to detect with confidence
games = {
    "Roulette (2.7%)": 0.027,
    "Blackjack (0.5%)": 0.005,
    "Craps (1.4%)": 0.014,
    "Slots (10%)": 0.10
}

n_simulations = 100
max_hands = 1000000

results = {}

for game_name, house_edge in games.items():
    trajectories = []
    
    for sim in range(n_simulations):
        outcomes = np.random.binomial(1, 1 - house_edge, max_hands)
        cumsum = np.cumsum(outcomes - (1 - house_edge))
        trajectories.append(cumsum)
    
    results[game_name] = trajectories

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (game_name, house_edge) in enumerate(games.items()):
    trajectories = results[game_name]
    
    # Plot sample paths
    for traj in trajectories[:10]:
        axes[idx].plot(traj, alpha=0.1, color='blue')
    
    # Plot mean trajectory
    mean_traj = np.mean(trajectories, axis=0)
    axes[idx].plot(mean_traj, color='red', linewidth=2, label='Mean')
    
    # Theoretical expectation
    theoretical = np.arange(max_hands) * (-house_edge)
    axes[idx].plot(theoretical, color='green', linewidth=2, linestyle='--', label='Theory')
    
    axes[idx].set_title(f'{game_name}')
    axes[idx].set_xlabel('Number of Hands')
    axes[idx].set_ylabel('Cumulative Profit')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)
    axes[idx].set_xscale('log')

plt.tight_layout()
plt.show()

# Convergence rate
print("Sample sizes for 95% confidence in detecting house edge:")
print("(Using standard error = σ / √N with σ ≈ 1)")
for game_name, house_edge in games.items():
    z_score = 1.96
    acceptable_error = house_edge * 0.1  # 10% of edge
    n_required = (z_score / acceptable_error) ** 2
    print(f"{game_name}: {n_required:,.0f} hands")
```

## 6. Challenge Round
If LLN guarantees house wins long-term, why do professional gamblers exist?
- LLN requires enormous N; human lifetimes insufficient for negative-edge games
- Variance dominates before LLN effect realized (bankrupt before convergence)
- Professionals have positive edge (skill > house) → LLN favors them
- Small edge + large bankroll can sustain long enough for LLN to work

## 7. Key References
- [Law of Large Numbers (Wikipedia)](https://en.wikipedia.org/wiki/Law_of_large_numbers)
- [Gambler's Fallacy (Wikipedia)](https://en.wikipedia.org/wiki/Gambler%27s_fallacy)
- [House Edge (Wizard of Odds)](https://www.wizardofodds.com/gambling/)

---
**Status:** Foundational probability theorem | **Complements:** House Edge, Expected Value, Gambler's Fallacy
