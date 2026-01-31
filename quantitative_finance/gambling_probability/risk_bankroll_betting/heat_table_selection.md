# Heat & Table Selection

## 1. Concept Skeleton
**Definition:** Choosing games and tables based on rules, limits, and casino “heat” (scrutiny of advantage players)  
**Purpose:** Maximize expected value and longevity by selecting favorable conditions and avoiding detection  
**Prerequisites:** House edge, variance, bankroll sizing, basic advantage play concepts

## 2. Comparative Framing
| Factor | Favorable Table | Unfavorable Table | EV Impact | Practical Risk |
|--------|-----------------|-------------------|-----------|----------------|
| **Rules** | Player-friendly (e.g., S17, DAS) | Dealer-friendly (H17, no DAS) | High | None |
| **Limits** | Wide limits | Low max bet | Medium | None |
| **Penetration** | Deep shoe | Shallow shoe | High | High for counters |
| **Heat** | Low surveillance | High scrutiny | Indirect | High (ban risk) |

## 3. Examples + Counterexamples

**Example (Good Table):**  
Blackjack: S17, DAS, late surrender, 3:2 payout → lower house edge.

**Example (Bad Table):**  
6:5 blackjack pays less, increases house edge by ~1.4%.

**Counterexample:**  
Best rules but heavy heat → short session, bans erase long-run EV.

## 4. Layer Breakdown
```
Heat & Table Selection:
├─ Rule-Based EV Drivers:
│  ├─ Blackjack payout (3:2 vs 6:5)
│  ├─ Dealer hits/stands on soft 17
│  ├─ Double after split (DAS)
│  ├─ Surrender availability
│  └─ Number of decks and penetration
├─ Table Limits & Bankroll Fit:
│  ├─ Minimum bet must match bankroll plan
│  ├─ Maximum bet must allow optimal spread
│  ├─ Volatility vs bankroll size
│  └─ Session length control
├─ Heat Dynamics:
│  ├─ Surveillance monitoring bet spread and play speed
│  ├─ Pit attention to unusual patterns
│  ├─ Backoffs (no blackjack) vs bans
│  └─ Heat increases with long sessions and large spreads
├─ Practical Selection Strategy:
│  ├─ Prefer lower-traffic tables for longevity
│  ├─ Avoid peak hours to reduce detection
│  ├─ Mix play styles to reduce profile
│  └─ Rotate casinos to avoid repeated scrutiny
└─ Ethical/Legal Boundaries:
   ├─ Counting is legal but casinos can refuse service
   ├─ Cheating (devices, collusion) illegal
   ├─ Respect posted rules and policies
   └─ Responsible gambling principles still apply
```

**Interaction:** Better rules → higher EV; higher EV → more scrutiny; table choice balances both.

## 5. Mini-Project
Compare blackjack rule sets and simulate long-run EV:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simplified EV adjustments (illustrative)
# Non-obvious: use rule deltas to approximate EV without full blackjack simulator
base_edge = -0.005  # -0.5% for good rules

rule_sets = {
    "Good Rules (3:2, S17, DAS)": base_edge,
    "Average Rules (3:2, H17, no DAS)": base_edge - 0.004,
    "Bad Rules (6:5, H17, no DAS)": base_edge - 0.014,
}

n_hands = 50000
bet = 10

results = {}
for name, edge in rule_sets.items():
    # simulate profit as EV + noise
    outcomes = np.random.normal(loc=edge * bet, scale=bet * 1.15, size=n_hands)
    results[name] = np.cumsum(outcomes)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative profit
for name, series in results.items():
    axes[0, 0].plot(series, label=name)
axes[0, 0].set_title('Cumulative Profit by Rule Set')
axes[0, 0].set_xlabel('Hand')
axes[0, 0].set_ylabel('Profit')
axes[0, 0].legend()

# Plot 2: Distribution of final profit
finals = [series[-1] for series in results.values()]
axes[0, 1].bar(list(results.keys()), finals)
axes[0, 1].set_title('Final Profit (50,000 Hands)')
axes[0, 1].set_ylabel('Profit')
axes[0, 1].tick_params(axis='x', rotation=20)

# Plot 3: Rule-set EV bars
axes[1, 0].bar(list(rule_sets.keys()), [e*100 for e in rule_sets.values()])
axes[1, 0].set_title('Approx House Edge by Rule Set')
axes[1, 0].set_ylabel('House Edge (%)')
axes[1, 0].tick_params(axis='x', rotation=20)

# Plot 4: Heat vs EV trade-off (illustrative)
heat = [2, 5, 1]  # arbitrary heat score
axes[1, 1].scatter([e*100 for e in rule_sets.values()], heat)
for i, name in enumerate(rule_sets.keys()):
    axes[1, 1].annotate(name, (list(rule_sets.values())[i]*100, heat[i]))
axes[1, 1].set_title('Heat vs EV Trade-off')
axes[1, 1].set_xlabel('House Edge (%)')
axes[1, 1].set_ylabel('Heat Score (lower is better)')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
Why can a slightly worse table be better in practice than the best-rule table?
- Lower heat extends playtime and preserves long-run EV
- Limits may better match bankroll, reducing ruin risk
- Lower attention reduces forced bet spread constraints
- Practical access outweighs small theoretical edge differences

## 7. Key References
- [Blackjack (Wikipedia)](https://en.wikipedia.org/wiki/Blackjack)
- [Wizard of Odds: Blackjack Rules](https://www.wizardofodds.com/games/blackjack/rule-variations/)
- [House Edge (Wikipedia)](https://en.wikipedia.org/wiki/House_edge)

---
**Status:** Table selection & longevity | **Complements:** Bankroll Management, Risk of Ruin, Expected Value
