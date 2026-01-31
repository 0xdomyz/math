# Illusion of Control

## 1. Concept Skeleton
**Definition:** Belief that one can influence outcomes of random or chance-determined events  
**Purpose:** Explain overconfidence, excessive betting, and mistaken attribution of skill in gambling  
**Prerequisites:** Randomness, independence, basic probability

## 2. Comparative Framing
| Concept | Illusion of Control | Skill Advantage | Superstition | Information Edge |
|--------|---------------------|----------------|-------------|------------------|
| **Driver** | Perceived personal influence | Real statistical edge | Rituals, lucky cues | Asymmetric data access |
| **Reality** | Random outcomes | Measurable EV > 0 | No causal effect | Positive EV possible |
| **Risk** | Overbetting | Underbetting possible | Persistence in bad games | Edge decay if public |
| **Evidence** | No performance lift | Repeatable profit | None | Replicable results |

## 3. Examples + Counterexamples

**Example (Fallacy):**  
Dice players blow on dice or throw harder to “force” high rolls.

**Example (Casino Design):**  
Interactive slots and “skill-stop” buttons create agency illusion without changing odds.

**Counterexample (Real Skill):**  
Skilled poker players improve EV through opponent exploitation, not luck control.

**Edge Case:**  
Blackjack card counting is real edge, but does not control randomness of individual cards.

## 4. Layer Breakdown
```
Illusion of Control:
├─ Cognitive Mechanisms:
│  ├─ Agency bias: Desire to feel in control
│  ├─ Outcome attribution: Wins credited to skill
│  ├─ Selective memory: Forget random losses
│  └─ Pattern imposition: Imposing structure on noise
├─ Gambling Manifestations:
│  ├─ Rituals: Lucky seats, charms, or routines
│  ├─ “Skill” buttons: Stop reels, tap screens
│  ├─ Table behavior: Aggressive betting after perceived control
│  └─ Overconfidence: Mistaking variance for skill
├─ Casino Design Amplifiers:
│  ├─ Interactivity: Buttons and animations
│  ├─ Near-miss effects: Outcomes that feel close
│  ├─ Personalized feedback: “You almost won!”
│  └─ Social cues: Crowds, applause, announcements
├─ Statistical Reality:
│  ├─ RNG dominance: Random outcomes dominate results
│  ├─ House edge: Fixed negative EV
│  ├─ Independence: Prior actions do not affect next outcome
│  └─ Variance: Random streaks create false confidence
└─ Mitigation:
   ├─ Education: Distinguish skill vs chance games
   ├─ Pre-commitment: Fixed bet sizing
   ├─ Tracking: Record outcomes vs expectations
   └─ Regulatory design: Limit misleading “skill” features
```

**Interaction:** Perceived control → larger bets → higher variance exposure → larger losses

## 5. Mini-Project
Test whether “control actions” change random outcomes:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate slot outcomes with and without "control" action
n_spins = 10000
p_win = 0.45  # illustrative (still negative EV with payouts)

# Control action does not change probability
wins_control = np.random.binomial(1, p_win, n_spins)
wins_no_control = np.random.binomial(1, p_win, n_spins)

# Rolling win rate
window = 200
roll_control = np.convolve(wins_control, np.ones(window)/window, mode='valid')
roll_no_control = np.convolve(wins_no_control, np.ones(window)/window, mode='valid')

# Simulate perceived control effect (subjective)
perceived_boost = roll_control - roll_no_control

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Win outcomes
axes[0, 0].plot(wins_control[:200], drawstyle='steps-post', label='Control')
axes[0, 0].plot(wins_no_control[:200], drawstyle='steps-post', label='No Control', alpha=0.6)
axes[0, 0].set_title('Outcome Sequences (First 200 Spins)')
axes[0, 0].set_xlabel('Spin')
axes[0, 0].set_ylabel('Win (1) / Loss (0)')
axes[0, 0].legend()

# Plot 2: Rolling win rates
axes[0, 1].plot(roll_control, label='Control')
axes[0, 1].plot(roll_no_control, label='No Control', alpha=0.7)
axes[0, 1].axhline(p_win, color='red', linestyle='--', label='True p')
axes[0, 1].set_title('Rolling Win Rate')
axes[0, 1].set_xlabel('Spin')
axes[0, 1].set_ylabel('Win Rate')
axes[0, 1].legend()

# Plot 3: Difference in rolling rates
axes[1, 0].plot(perceived_boost, color='purple')
axes[1, 0].axhline(0, color='black', linestyle='--')
axes[1, 0].set_title('Perceived Control Effect (Difference)')
axes[1, 0].set_xlabel('Spin')
axes[1, 0].set_ylabel('Rate Difference')

# Plot 4: Distribution of win rates
axes[1, 1].hist(roll_control, bins=30, alpha=0.7, label='Control')
axes[1, 1].hist(roll_no_control, bins=30, alpha=0.7, label='No Control')
axes[1, 1].set_title('Win Rate Distribution (Rolling Window)')
axes[1, 1].set_xlabel('Win Rate')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Mean win rate (control): {wins_control.mean():.3f}")
print(f"Mean win rate (no control): {wins_no_control.mean():.3f}")
```

## 6. Challenge Round
If the illusion of control is harmful, why do gambling interfaces intentionally add “interactive” features?
- Interactivity increases engagement and time-on-device
- Players equate agency with skill and tolerate negative EV longer
- Near-miss and feedback loops reinforce perceived control
- Behavioral design nudges are profitable even without changing odds

## 7. Key References
- [Illusion of Control (Wikipedia)](https://en.wikipedia.org/wiki/Illusion_of_control)
- [Gambling Disorder (Wikipedia)](https://en.wikipedia.org/wiki/Gambling_disorder)
- [Near-Miss Effect in Gambling](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3089191/)

---
**Status:** Core cognitive bias | **Complements:** Casino Design, Gambler’s Fallacy, Responsible Gambling
