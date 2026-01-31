"""
Extracted from: emotional_control_tilt.md
"""

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
