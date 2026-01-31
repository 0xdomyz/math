# Time Limits

## 1. Concept Skeleton
**Definition:** Pre-set session duration after which play stops, regardless of outcomes  
**Purpose:** Prevent escalating losses through extended play and protect against tilt  
**Prerequisites:** Bankroll management, session budgeting, loss aversion

## 2. Comparative Framing
| Control Type | Time Limits | Stop-Loss | Betting Limits | Emotional Control |
|-------------|------------|-----------|----------------|-------------------|
| **Trigger** | Clock-based | Amount-based | Bet-size-based | Subjective feeling |
| **Enforcement** | External (rules) | External (math) | External (rules) | Internal (discipline) |
| **Advantage** | Prevents streaks | Caps single-session loss | Prevents escalation | Self-awareness required |
| **Weakness** | Easy to ignore | Requires tracking | Felt as frustrating | Most unreliable |

## 3. Examples + Counterexamples

**Example (Protective):**  
Gambler arrives at 2 PM, leaves at 4 PM regardless of score.

**Example (Disciplined):**  
Player sets 1-hour limit, forced stop prevents chasing losses.

**Counterexample (Ignored):**  
"One more hour" repeatedly extended → time limit becomes meaningless.

## 4. Layer Breakdown
```
Time Limits:
├─ Psychological Drivers for Violation:
│  ├─ "Hot hand" feeling justifies extension
│  ├─ Sunk cost: "One more hour recovers losses"
│  ├─ Trance state: Loses perception of time
│  └─ Excitement: Wants to ride momentum
├─ Loss Prevention Mechanism:
│  ├─ Shorter sessions → fewer hands → less variance realization
│  ├─ Breaks interrupt trance state (casino design fights this)
│  ├─ Natural stopping point reduces escalation
│  └─ Fatigue effects: Longer sessions → worse decision-making
├─ Implementation Strategies:
│  ├─ Set alarm: Physical reminder of elapsed time
│  ├─ Buddy system: Another person enforces limit
│  ├─ Casino features: Some offer time-based warnings
│  └─ Appointment approach: Schedule other commitment after
├─ Interaction with Other Limits:
│  ├─ Time + stop-loss: Dual protection (tightest wins)
│  ├─ Time + betting limit: Prevents streak chasing
│  ├─ Time alone: Weaker than combined limits
│  └─ Enforcement: External > internal (hard to self-enforce)
└─ Effectiveness Measurement:
   ├─ Compliance rate: % of sessions respecting limit
   ├─ Extension frequency: How often is limit violated?
   └─ Outcome impact: Compare loss rates pre/post-limit
```

**Interaction:** Set session time → play → alarm rings → stop (even if winning or losing) → cool-down period breaks trance.

## 5. Mini-Project
Simulate impact of time limits on session outcomes:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Non-obvious concept: shorter sessions reduce variance realization AND cap losses
def simulate_session(duration_hands, house_edge=0.02, bet=1):
    """Simulate game outcomes within time limit (hands allowed)"""
    outcomes = []
    for _ in range(duration_hands):
        outcome = np.random.rand() < (1 - house_edge)
        outcomes.append(bet if outcome else -bet)
    return outcomes

# Compare different time limits
time_limits = [10, 50, 100, 500, 1000]
n_sessions = 500

results = {}
for limit in time_limits:
    finals = []
    for _ in range(n_sessions):
        outcomes = simulate_session(limit)
        finals.append(sum(outcomes))
    results[limit] = finals

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of final outcomes by time limit
for limit in time_limits:
    axes[0, 0].hist(results[limit], bins=30, alpha=0.5, label=f'{limit} hands')
axes[0, 0].set_title('Distribution of Session Outcomes')
axes[0, 0].set_xlabel('Profit')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Plot 2: Average loss by time limit
avg_loss = [np.mean(results[limit]) for limit in time_limits]
axes[0, 1].plot(time_limits, avg_loss, 'o-', linewidth=2, markersize=8)
axes[0, 1].set_title('Expected Loss by Session Duration')
axes[0, 1].set_xlabel('Number of Hands')
axes[0, 1].set_ylabel('Average Loss ($)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Risk of ruin (lose all $100 session bankroll)
session_bankroll = 100
ruin_rates = []
for limit in time_limits:
    ruin_count = np.sum(np.array(results[limit]) < -session_bankroll)
    ruin_rates.append(ruin_count / n_sessions)
axes[1, 0].bar(time_limits, ruin_rates, color=['green','yellow','orange','red','darkred'])
axes[1, 0].set_title('Risk of Ruin by Session Duration')
axes[1, 0].set_xlabel('Number of Hands')
axes[1, 0].set_ylabel('Ruin Probability')

# Plot 4: Distribution of session lengths (compliance)
# Simulate compliance: 80% stop on time, 20% extend
compliance_rate = 0.8
session_lengths = []
for _ in range(n_sessions):
    if np.random.rand() < compliance_rate:
        session_lengths.append(1.0)  # Stopped on time
    else:
        session_lengths.append(1.5)  # Extended 50%

axes[1, 1].hist(session_lengths, bins=[0.9, 1.1, 1.6], color=['green','red'])
axes[1, 1].set_title('Session Compliance (Stop on Time vs Extend)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_xticks([1.0, 1.5])
axes[1, 1].set_xticklabels(['On Time', 'Extended'])

plt.tight_layout()
plt.show()

print(f"Expected loss (100 hands, 2% edge): ${abs(avg_loss[0]):.2f}")
print(f"Expected loss (1000 hands, 2% edge): ${abs(avg_loss[-1]):.2f}")
print(f"Ruin risk (100 hands): {ruin_rates[0]:.1%}")
print(f"Ruin risk (1000 hands): {ruin_rates[-1]:.1%}")
```

## 6. Challenge Round
If time limits prevent losses, why don't all casinos enforce them for players?
- Casinos profit from extended play (more hands = more edge realization)
- Player retention > player protection from casino perspective
- Some jurisdictions now require optional time-limit features
- Voluntary compliance rates low; forced limits reduce revenue

## 7. Key References
- [Responsible Gambling Features (UK Gambling Commission)](https://www.gamblingcommission.gov.uk)
- [Session Time Limits (NCPG)](https://www.ncpg.org/)
- [Problem Gambling Treatment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4541962/)

---
**Status:** Risk mitigation tool | **Complements:** Bankroll Management, Stop-Loss Limits, Responsible Gambling
