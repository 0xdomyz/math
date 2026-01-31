# Emotional Control & Tilt Management

## 1. Concept Skeleton
**Definition:** Managing emotional decisions (tilt) during losing streaks and high-stress situations  
**Purpose:** Maintain optimal strategy and avoid impulsive escalation  
**Prerequisites:** Bankroll management, self-awareness, behavioral psychology basics

## 2. Comparative Framing
| State | Optimal | Slight Tilt | Severe Tilt | Tilted-Out |
|-------|---------|------------|-----------|-----------|
| **Decision Quality** | Game-theoretic | Slightly off | Poor | Reckless |
| **Bet Sizing** | Per plan | +25% stakes | 2-3x normal | All-in mentality |
| **Frequency** | Controlled | Increased | Rapid fire | Continuous |
| **Outcome** | Long-term edge preserved | Minor leak | Severe losses | Bankrupt |

## 3. Examples + Counterexamples

**Example (Tilt Spiral):**  
Lose 3 hands → bet larger → lose more → frustration → abandon strategy → catastrophic loss.

**Example (Optimal Recovery):**  
Lose 3 hands → recognize tilt → take break → return focused → execute strategy.

**Counterexample (No Tilt):**  
Emotionless robot player maintains EV (unrealistic for humans).

## 4. Layer Breakdown
```
Emotional Control & Tilt:
├─ Tilt Triggers:
│  ├─ Bad beats (unlucky losses on good hands)
│  ├─ Downswings (multiple losses in sequence)
│  ├─ Pressure situations (large stakes)
│  ├─ Sleep deprivation (fatigue impairs judgment)
│  └─ Personal factors (external stress, anger)
├─ Physical Manifestations:
│  ├─ Accelerated betting pace
│  ├─ Bet sizing escalation
│  ├─ Reduced pause between decisions
│  ├─ Aggressive body language
│  └─ Visible frustration (signals to opponents)
├─ Cognitive Effects:
│  ├─ Narrowed focus (tunnel vision)
│  ├─ Loss of objectivity (emotional weight on decisions)
│  ├─ Impaired risk assessment (overestimate edge)
│  └─ Strategy abandonment (chase losses vs play optimal)
├─ Prevention Strategies:
│  ├─ Pre-session: Good sleep, clear mind, set limits
│  ├─ During session: Breathing exercises, micro-breaks
│  ├─ Trigger response: Recognize and pause immediately
│  ├─ Cool-down: Walk away if tilting detected
│  └─ Post-tilt: Review what happened, adjust plan
├─ Detection Signs:
│  ├─ Self-awareness: Can you describe your emotional state?
│  ├─ Behavior change: Betting patterns shifted?
│  ├─ Decision speed: Are you rushing?
│  └─ Buddy feedback: Do others notice you're off?
└─ Long-term Management:
   ├─ Study: Understand game theory, reduce "bad beat" surprises
   ├─ Variance acceptance: Expect swings as normal
   ├─ Bankroll sizing: Enough capital to weather tilts
   ├─ Support system: Confide in trusted players/advisors
   └─ Mindfulness: Meditation, stress management training
```

**Interaction:** Loss → emotional reaction → tilt → poor decisions → larger losses → recovery required.

## 5. Mini-Project
Model tilt impact on strategy and outcomes:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Non-obvious: small increase in bet size during tilt compounds losses exponentially
def simulate_with_tilt(n_rounds=200, house_edge=0.02, base_bet=10, tilt_threshold=3):
    """
    Simulate: consecutive losses trigger tilt, bet size increases
    """
    outcomes = []
    bet_sizes = []
    tilted_rounds = []
    
    bankroll = 500
    consecutive_losses = 0
    tilt_multiplier = 1.0
    
    for round_num in range(n_rounds):
        if bankroll <= 0:
            break
        
        bet = min(base_bet * tilt_multiplier, bankroll)
        bet_sizes.append(bet)
        
        # Outcome: win with prob (1 - house_edge)
        win = np.random.rand() < (1 - house_edge)
        
        if win:
            outcome = bet
            consecutive_losses = 0
            tilt_multiplier = 1.0
        else:
            outcome = -bet
            consecutive_losses += 1
            
            if consecutive_losses >= tilt_threshold:
                tilt_multiplier = min(2.0, tilt_multiplier + 0.5)  # Escalate
        
        outcomes.append(outcome)
        bankroll += outcome
        tilted = 1 if consecutive_losses >= tilt_threshold else 0
        tilted_rounds.append(tilted)
    
    return np.cumsum(outcomes), bet_sizes, tilted_rounds, bankroll

# Compare: with vs without tilt control
with_tilt, bets_with, tilt_signal, final_with = simulate_with_tilt()
without_tilt_signal = [0] * len(with_tilt)
without_tilt, bets_without, _, final_without = simulate_with_tilt(tilt_threshold=999)  # Never tilts

# Better version: tilt with corrective action
def simulate_with_tilt_control(n_rounds=200, house_edge=0.02, base_bet=10):
    outcomes = []
    bankroll = 500
    consecutive_losses = 0
    
    for _ in range(n_rounds):
        if bankroll <= 0:
            break
        
        if consecutive_losses >= 3:
            break  # Stop to avoid tilt spiral (CONTROL)
        
        win = np.random.rand() < (1 - house_edge)
        outcome = base_bet if win else -base_bet
        
        if not win:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        
        outcomes.append(outcome)
        bankroll += outcome
    
    return np.cumsum(outcomes), bankroll

controlled, final_controlled = simulate_with_tilt_control()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bankroll paths
axes[0, 0].plot(with_tilt, label='With Tilt (escalate)', linewidth=2)
axes[0, 0].plot(without_tilt, label='No Tilt (constant)', linewidth=2, alpha=0.7)
axes[0, 0].plot(controlled, label='Tilt Control (stop)', linewidth=2, alpha=0.7)
axes[0, 0].set_title('Bankroll Trajectory: Impact of Tilt')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Bankroll ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Bet sizes over time
axes[0, 1].plot(bets_with, label='With Tilt', alpha=0.7)
axes[0, 1].plot(bets_without, label='Without Tilt', alpha=0.7)
axes[0, 1].set_title('Bet Size Escalation (Tilt Effect)')
axes[0, 1].set_xlabel('Round')
axes[0, 1].set_ylabel('Bet Size ($)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Tilt signal over time
axes[1, 0].fill_between(range(len(tilt_signal)), tilt_signal, alpha=0.3, color='red', label='Tilted')
axes[1, 0].plot(tilt_signal, color='red', linewidth=2)
axes[1, 0].set_title('Tilt Detection (Consecutive Losses)')
axes[1, 0].set_xlabel('Round')
axes[1, 0].set_ylabel('Tilted (Yes/No)')
axes[1, 0].set_ylim(-0.1, 1.1)

# Plot 4: Final outcomes comparison
strategies = ['With Tilt\n(Escalate)', 'Without Tilt\n(Constant)', 'Tilt Control\n(Stop)']
finals = [final_with, final_without, final_controlled]
colors = ['red', 'blue', 'green']
axes[1, 1].bar(strategies, finals, color=colors)
axes[1, 1].set_title('Final Bankroll by Strategy')
axes[1, 1].set_ylabel('Final Bankroll ($)')
axes[1, 1].axhline(500, color='black', linestyle='--', alpha=0.5, label='Starting')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Final with tilt (no control): ${final_with:.0f}")
print(f"Final without tilt: ${final_without:.0f}")
print(f"Final with tilt control: ${final_controlled:.0f}")
```

## 6. Challenge Round
If emotional control is so important, why can't players just "stay calm"?
- Emotional systems operate faster than conscious thought
- High-stakes stress triggers amygdala (fight-or-flight), overriding prefrontal judgment
- Prolonged fatigue and sleep deprivation reduce impulse control neurologically
- Even professional athletes/traders experience tilt despite training
- Requires external controls (limits, breaks, buddy system) beyond willpower alone

## 7. Key References
- [Psychology of Tilt (Poker Research)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4541962/)
- [Emotional Regulation (Psychology Today)](https://www.psychologytoday.com/basics/emotion-regulation)
- [Stress & Decision Making (Harvard Business Review)](https://hbr.org/)

---
**Status:** Behavioral risk management | **Complements:** Bankroll Management, Responsible Gambling, Mental Health
