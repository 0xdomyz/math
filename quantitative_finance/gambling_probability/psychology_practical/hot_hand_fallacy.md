# Hot-Hand Fallacy

## 1. Concept Skeleton
**Definition:** Belief that recent success increases the probability of future success in a random process  
**Purpose:** Explain streak-chasing behavior, overbetting after wins, and misinterpretation of randomness  
**Prerequisites:** Probability basics, independence, randomness vs pattern perception

## 2. Comparative Framing
| Concept | Hot-Hand Fallacy | Gambler’s Fallacy | Skill-Based Streaks | Regression to Mean |
|--------|------------------|------------------|---------------------|--------------------|
| **Core Belief** | “Winning continues” | “Winning must end” | Past skill predicts future | Extremes revert toward average |
| **Process** | Random or weakly skill-based | Random | Strongly skill-based | Statistical tendency |
| **Typical Error** | Overconfidence after wins | Overbetting after losses | Underestimating variance | Misreading noise as signal |
| **Gambling Impact** | Bet escalation | Loss chasing | Advantage play when real | Conservative sizing |

## 3. Examples + Counterexamples

**Example (Fallacy):**  
A roulette player hits red 4 times and increases bets, believing “red is hot.”

**Example (Borderline):**  
A blackjack player wins 5 hands and assumes skill improved, despite independent hands.

**Counterexample (Skill Streak):**  
A professional poker player with verified edge shows true positive autocorrelation in performance.

**Edge Case:**  
Short-term streaks in small samples can occur naturally without any underlying change.

## 4. Layer Breakdown
```
Hot-Hand Fallacy:
├─ Cognitive Mechanism:
│  ├─ Pattern detection bias: Humans seek patterns in noise
│  ├─ Recency bias: Recent outcomes overweighted
│  ├─ Availability heuristic: Vivid wins dominate memory
│  └─ Illusory correlation: Chance alignment seen as causal
├─ Statistical Reality:
│  ├─ Independence: Outcomes do not affect future probability
│  ├─ Variance: Streaks are expected in random sequences
│  ├─ Regression to mean: Extremes tend to revert
│  └─ Small sample error: Short streaks exaggerate belief
├─ Gambling Behaviors:
│  ├─ Bet escalation: Raising stakes after wins
│  ├─ Session extension: Playing longer due to “hot” belief
│  ├─ Risk tolerance shift: Increased risk appetite
│  └─ Overconfidence: Misattributing luck to skill
├─ Measurement:
│  ├─ Streak length distribution: Expected runs in Bernoulli trials
│  ├─ Conditional probability tests: P(win | k wins) vs baseline
│  ├─ Runs test: Detect non-random streak patterns
│  └─ Autocorrelation: Check dependence in outcome sequence
└─ Mitigation:
   ├─ Pre-commitment: Fixed bet sizing rules
   ├─ Education: Explain run distributions
   ├─ Cooldown periods: Pause after big wins
   └─ Data tracking: Compare perceived vs actual edge
```

**Interaction:** Wins → perceived momentum → higher bets → increased variance → larger losses despite unchanged odds

## 5. Mini-Project
Simulate streaks and show why “hot hands” appear in random sequences:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate coin flips (fair)
num_trials = 1000
flips = np.random.binomial(1, 0.5, num_trials)

# Identify streaks
streak_lengths = []
current = 1
for i in range(1, num_trials):
    if flips[i] == flips[i-1]:
        current += 1
    else:
        streak_lengths.append(current)
        current = 1
streak_lengths.append(current)

# Conditional probability after streaks
max_k = 6
cond_probs = []
for k in range(1, max_k + 1):
    indices = []
    count = 0
    for i in range(k, num_trials):
        if np.all(flips[i-k:i] == 1):  # k wins in a row
            count += 1
            indices.append(i)
    if count > 0:
        cond_prob = flips[indices].mean()
    else:
        cond_prob = np.nan
    cond_probs.append(cond_prob)

# Simulate skill-based streaks (slight edge)
skill_flips = np.random.binomial(1, 0.55, num_trials)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sequence of flips
axes[0, 0].plot(flips[:200], drawstyle='steps-post')
axes[0, 0].set_title('Random Outcomes (First 200)')
axes[0, 0].set_xlabel('Trial')
axes[0, 0].set_ylabel('Outcome')

# Plot 2: Streak length distribution
axes[0, 1].hist(streak_lengths, bins=range(1, 15), alpha=0.7, color='orange')
axes[0, 1].set_title('Streak Length Distribution')
axes[0, 1].set_xlabel('Streak Length')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Conditional probability after k wins
axes[1, 0].plot(range(1, max_k + 1), cond_probs, marker='o')
axes[1, 0].axhline(0.5, color='red', linestyle='--', label='True p=0.5')
axes[1, 0].set_title('P(Win | k Wins in a Row)')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('Conditional Probability')
axes[1, 0].legend()

# Plot 4: Random vs skill-based win rates
window = 50
random_rate = np.convolve(flips, np.ones(window)/window, mode='valid')
skill_rate = np.convolve(skill_flips, np.ones(window)/window, mode='valid')

axes[1, 1].plot(random_rate, label='Random (p=0.5)')
axes[1, 1].plot(skill_rate, label='Skill (p=0.55)')
axes[1, 1].set_title('Rolling Win Rate (50-trial window)')
axes[1, 1].set_xlabel('Trial')
axes[1, 1].set_ylabel('Win Rate')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Average streak length: {np.mean(streak_lengths):.2f}")
print(f"Max streak length observed: {max(streak_lengths)}")
print(f"Conditional probs after k wins: {cond_probs}")
```

## 6. Challenge Round
If hot hands are mostly illusion in random games, why do casinos still encourage “winning streak” narratives?
- Narratives increase betting intensity and session length without changing odds
- Streak framing shifts attention from negative EV to emotional momentum
- Social proof (crowds, announcements) amplifies perceived skill and keeps players engaged
- Even if rare true edges exist, most players misidentify them and overbet

## 7. Key References
- [Hot-Hand Fallacy (Wikipedia)](https://en.wikipedia.org/wiki/Hot_hand)
- [Gambler’s Fallacy (Wikipedia)](https://en.wikipedia.org/wiki/Gambler%27s_fallacy)
- [Runs Test (Statistics)](https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test)

---
**Status:** Behavioral bias in gambling | **Complements:** Gambler’s Fallacy, Randomness & Fairness, Bankroll Management
