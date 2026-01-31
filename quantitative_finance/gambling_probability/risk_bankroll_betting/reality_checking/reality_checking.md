# Reality Checking

## 1. Concept Skeleton
**Definition:** Periodic displays of session time/money spent to interrupt trance and increase awareness  
**Purpose:** Combat dissociation and "flow state" that masks losses  
**Prerequisites:** Behavioral psychology, casino design understanding, loss aversion

## 2. Comparative Framing
| Awareness Trigger | Frequency | Effectiveness | User Cooperation | Casino Adoption |
|------------------|-----------|--------------|------------------|-----------------|
| **Reality Check** | Every 30 min | Medium | Voluntary | Low |
| **Mandatory Break** | Every 1-2 hr | High | Forced | Rare |
| **Balance Display** | Continuous | High | Passive | Medium |
| **Loss Limit Alert** | On threshold | High | Reactive | Medium |

## 3. Examples + Counterexamples

**Example (UK Standard):**  
Online gambling interface shows "Time played: 3 hours 45 minutes | Money wagered: £450."

**Example (Interrupting):**  
Casino forces 15-minute break after 3 hours of play.

**Counterexample (Ignored):**  
Mandatory popup ignored with quick "X" click, no behavioral change.

## 4. Layer Breakdown
```
Reality Checking:
├─ Psychological Mechanism:
│  ├─ Trance state: Dissociation from reality
│  ├─ Time distortion: Hours feel like minutes
│  ├─ Money abstraction: Chips feel meaningless
│  ├─ Arousal manipulation: Lights/sounds suppress reflection
│  └─ Break in attention: Reality check re-anchors to reality
├─ Implementation Methods:
│  ├─ Pop-up dialogs: Mandatory pause with info
│  ├─ Balance display: Continuous on-screen total
│  ├─ Time stamps: Show elapsed time at regular intervals
│  ├─ Verbal announcements: Periodic "time checks"
│  └─ Buddy notifications: Friend sends reminder
├─ Effectiveness Factors:
│  ├─ Frequency: More frequent checks more effective
│  ├─ Obtrusiveness: Harder to ignore = more impact
│  ├─ Content: Clear data vs vague reminder
│  ├─ Timing: During losing streaks more impactful
│  └─ User intention: Motivated players more responsive
├─ Casino Resistance:
│  ├─ Reduces play duration (fewer hands/bets)
│  ├─ Breaks momentum and emotional escalation
│  ├─ Encourages walk-away decisions
│  ├─ Regulatory requirement (UK, Australia, EU)
│  └─ Revenue negative (plays vs profits)
├─ Technical Implementation:
│  ├─ Online gambling: Pop-up at intervals
│  ├─ Slot machines: Screen display option
│  ├─ Live gaming: Dealer announcement capability
│  └─ App-based: Push notifications to player
└─ Evidence & Outcomes:
   ├─ Studies show 20-40% reduction in continued play after reality check
   ├─ More effective with personalized loss data
   ├─ Timing (after losses) more impactful than time-based
   └─ Repeated exposure builds habit (but efficacy declines over time)
```

**Interaction:** Trance state → lose awareness → reality check → brief awakening → decision to continue or stop.

## 5. Mini-Project
Simulate reality check impact on session duration and loss:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Non-obvious: reality check at key moments (losses) more effective than time-based
def simulate_without_check(n_hands=500, house_edge=0.02, bet=10):
    outcomes = []
    for _ in range(n_hands):
        win = np.random.rand() < (1 - house_edge)
        outcomes.append(bet if win else -bet)
    return outcomes

def simulate_with_check(n_hands=500, house_edge=0.02, bet=10, check_frequency=50, stop_threshold=0.7):
    """
    Reality check: every check_frequency hands or after consecutive losses
    Assume: 30% of players stop after reality check (stop_threshold)
    """
    outcomes = []
    checks = 0
    consecutive_losses = 0
    
    for i in range(n_hands):
        if i % check_frequency == 0 and i > 0:
            checks += 1
            # Reality check: 30% of players stop
            if np.random.rand() < stop_threshold:
                break  # Player stops
        
        win = np.random.rand() < (1 - house_edge)
        outcome = bet if win else -bet
        outcomes.append(outcome)
        
        if not win:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        
        # Additional check after 3+ consecutive losses
        if consecutive_losses >= 3:
            checks += 1
            if np.random.rand() < stop_threshold:
                break
    
    return outcomes, checks

# Run simulations
n_sessions = 300
sessions_no_check = []
sessions_with_check = []
check_counts = []

for _ in range(n_sessions):
    outcomes1 = simulate_without_check()
    outcomes2, checks = simulate_with_check()
    
    sessions_no_check.append(sum(outcomes1))
    sessions_with_check.append(sum(outcomes2))
    check_counts.append(checks)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of losses
axes[0, 0].hist(sessions_no_check, bins=40, alpha=0.6, label='No Reality Check', color='red')
axes[0, 0].hist(sessions_with_check, bins=40, alpha=0.6, label='With Reality Check', color='green')
axes[0, 0].set_title('Session Loss Distribution')
axes[0, 0].set_xlabel('Total Loss ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Plot 2: Average loss
avg_no_check = np.mean(sessions_no_check)
avg_with_check = np.mean(sessions_with_check)
axes[0, 1].bar(['No Check', 'With Check'], [avg_no_check, avg_with_check], color=['red', 'green'])
axes[0, 1].set_title('Average Session Loss')
axes[0, 1].set_ylabel('Loss ($)')

# Plot 3: Stop rate (how many ended session early due to check)
stop_rate = np.mean(np.array(check_counts) > 0)
axes[1, 0].bar(['No Check\n(Full Duration)', 'With Check\n(Some Early Stop)'], 
              [0, stop_rate], color=['red', 'green'])
axes[1, 0].set_title('Session Interruption Rate')
axes[1, 0].set_ylabel('Probability')
axes[1, 0].set_ylim([0, 1])

# Plot 4: Cumulative loss by session order
axes[1, 1].plot(np.cumsum(sessions_no_check[:100]), label='No Check', linewidth=2)
axes[1, 1].plot(np.cumsum(sessions_with_check[:100]), label='With Check', linewidth=2)
axes[1, 1].set_title('Cumulative Loss Over Sessions')
axes[1, 1].set_xlabel('Session Number')
axes[1, 1].set_ylabel('Cumulative Loss ($)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Average loss (no check): ${avg_no_check:.2f}")
print(f"Average loss (with check): ${avg_with_check:.2f}")
print(f"Loss reduction: {(avg_no_check - avg_with_check)/avg_no_check:.1%}")
```

## 6. Challenge Round
If reality checks reduce losses, why don't players demand them?
- Casino resistance embedded (feature disabled by default or ignored)
- Psychological dissociation so strong that even visible data doesn't penetrate
- Sunk cost mentality overrides awareness ("Already lost $200, keep playing")
- Regulatory requirement, not market demand (players profit-seeking, not loss-limiting)
- Efficacy declines with repeated exposure (habituation to alerts)

## 7. Key References
- [Reality Check (UK Gambling Commission)](https://www.gamblingcommission.gov.uk)
- [Responsible Gambling Features](https://www.ncpg.org/)
- [Dissociation in Problem Gambling](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4541962/)

---
**Status:** Awareness & interruption tool | **Complements:** Responsible Gambling, Trance State Prevention, Loss Control
