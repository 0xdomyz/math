"""
Extracted from: law_of_large_numbers.md
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

np.random.seed(42)

print("="*70)
print("LAW OF LARGE NUMBERS: CONVERGENCE TO EXPECTED VALUE")
print("="*70)

# ============================================================================
# 1. SINGLE GAME SIMULATION: ROULETTE
# ============================================================================

print("\n" + "="*70)
print("1. ROULETTE: CONVERGENCE WITH CONFIDENCE BANDS")
print("="*70)

class RouletteGame:
    """European roulette: -1 loss with prob 19/37, +1 win with prob 18/37."""
    
    def __init__(self):
        self.p_win = 18/37
        self.p_loss = 19/37
        self.ev = 1 * self.p_win + (-1) * self.p_loss  # -1/37
        self.variance = (1 - self.ev)**2 * self.p_win + (-1 - self.ev)**2 * self.p_loss
        self.std = np.sqrt(self.variance)
    
    def simulate_bets(self, n_bets):
        """Simulate n consecutive $1 bets; return running average."""
        outcomes = np.random.choice([1, -1], size=n_bets, p=[self.p_win, self.p_loss])
        cumsum = np.cumsum(outcomes)
        average = cumsum / np.arange(1, n_bets + 1)
        return average, outcomes, cumsum

# Theoretical values
game = RouletteGame()
print(f"\nRoulette Theoretical Values:")
print(f"   E[X] = {game.ev:.6f} (≈ -$0.027 per bet)")
print(f"   Variance = {game.variance:.6f}")
print(f"   Std Dev = {game.std:.6f}")

# Simulate many paths
n_bets = 10000
n_simulations = 100

all_averages = []
for sim in range(n_simulations):
    avg, _, _ = game.simulate_bets(n_bets)
    all_averages.append(avg)

all_averages = np.array(all_averages)

# Compute statistics along the path
mean_average = np.mean(all_averages, axis=0)
std_average = np.std(all_averages, axis=0, ddof=1)

# Confidence bands (±1.96 SD for 95% CI)
z_score = 1.96
upper_band = mean_average + z_score * std_average
lower_band = mean_average - z_score * std_average

# Theoretical SE
theoretical_se = game.std / np.sqrt(np.arange(1, n_bets + 1))
theoretical_upper = game.ev + z_score * theoretical_se
theoretical_lower = game.ev - z_score * theoretical_se

print(f"\nSimulation Results ({n_simulations} paths, {n_bets} bets each):")
print(f"   At n=100: Average ± SE = {mean_average[99]:.4f} ± {std_average[99]:.4f}")
print(f"   At n=1000: Average ± SE = {mean_average[999]:.4f} ± {std_average[999]:.4f}")
print(f"   At n=10000: Average ± SE = {mean_average[9999]:.4f} ± {std_average[9999]:.4f}")
print(f"   Theoretical EV = {game.ev:.6f}")
print(f"   Convergence? {abs(mean_average[9999] - game.ev) < 0.01}")

# Probability of profit at different n
prob_profit = np.mean(all_averages > 0, axis=0)

print(f"\n   P(profit | n bets):")
for n_test in [100, 500, 1000, 5000, 10000]:
    idx = n_test - 1
    print(f"      n={n_test}: {prob_profit[idx]:.3f} ({int(prob_profit[idx]*100)}%)")

# ============================================================================
# 2. MULTIPLE GAME COMPARISON
# ============================================================================

print("\n" + "="*70)
print("2. COMPARING CONVERGENCE ACROSS GAMES")
print("="*70)

class CasinoGame:
    """Generic casino game with specified EV and variance."""
    
    def __init__(self, name, ev, std):
        self.name = name
        self.ev = ev
        self.std = std
    
    def simulate_bets(self, n_bets, n_sims=100):
        """Simulate n_bets for n_sims independent paths."""
        paths = []
        for _ in range(n_sims):
            outcomes = np.random.normal(self.ev, self.std, n_bets)
            cumsum = np.cumsum(outcomes)
            average = cumsum / np.arange(1, n_bets + 1)
            paths.append(average)
        return np.array(paths)

# Different games
games = [
    CasinoGame("Roulette (EU)", ev=-1/37, std=game.std),
    CasinoGame("Blackjack", ev=-0.005, std=1.2),
    CasinoGame("Craps", ev=-0.014, std=1.1),
    CasinoGame("Slots", ev=-0.05, std=2.0)
]

n_bets_comparison = 1000
results_comparison = []

for game_obj in games:
    paths = game_obj.simulate_bets(n_bets_comparison, n_sims=50)
    mean_path = np.mean(paths, axis=0)
    results_comparison.append({
        'game': game_obj.name,
        'ev': game_obj.ev,
        'mean_path': mean_path
    })

print(f"\nGame Comparison ({n_bets_comparison} bets, 50 simulations each):")
for res in results_comparison:
    final_avg = res['mean_path'][-1]
    print(f"   {res['game']:20s}: EV={res['ev']:7.4f}, Avg after {n_bets_comparison} bets={final_avg:7.4f}")

# ============================================================================
# 3. BANKROLL DYNAMICS
# ============================================================================

print("\n" + "="*70)
print("3. BANKROLL EVOLUTION UNDER LAW OF LARGE NUMBERS")
print("="*70)

def simulate_bankroll(ev, std, n_bets, initial_bankroll, bet_size=1.0):
    """Simulate bankroll evolution; stop if bankruptcy."""
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_placed = 0
    
    for bet in range(n_bets):
        # Bet
        if bankroll <= 0:
            break  # Bankrupt
        
        # Outcome
        outcome = np.random.normal(ev, std)
        bankroll += bet_size * outcome
        bankroll_history.append(bankroll)
        bets_placed += 1
    
    return np.array(bankroll_history), bets_placed

# Scenarios
scenarios = [
    {'name': 'Roulette, $1 bet', 'ev': -1/37, 'std': game.std, 'initial': 100, 'bet': 1},
    {'name': 'Blackjack, $5 bet', 'ev': -0.005, 'std': 1.2, 'initial': 500, 'bet': 5},
    {'name': 'Slots, $0.25 bet', 'ev': -0.05, 'std': 2.0, 'initial': 100, 'bet': 0.25}
]

print(f"\nBankroll Trajectories (representative paths):")
for scenario in scenarios:
    bankroll, bets = simulate_bankroll(scenario['ev'], scenario['std'], 5000, 
                                       scenario['initial'], scenario['bet'])
    final_bankroll = bankroll[-1]
    expected_loss = scenario['ev'] * scenario['bet'] * bets
    print(f"   {scenario['name']}:")
    print(f"      Bets completed: {bets}")
    print(f"      Final bankroll: ${final_bankroll:.2f} (started ${scenario['initial']:.2f})")
    print(f"      Expected loss: ${-expected_loss:.2f}")

# ============================================================================
# 4. CONVERGENCE RATE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("4. CONVERGENCE RATE: HOW QUICKLY DOES LLN KICK IN?")
print("="*70)

def compute_convergence_n_for_accuracy(ev, std, target_accuracy, confidence=0.95):
    """
    Compute n needed for convergence to ev within ±target_accuracy at given confidence.
    
    From normal approximation:
    P(|average - ev| ≤ target_accuracy) ≈ Φ(target_accuracy × √n / std)
    
    For confidence level (e.g., 0.95), find z such that Φ(z) = (1+confidence)/2
    Then: z = target_accuracy × √n / std
          n = (z × std / target_accuracy)²
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    n = (z * std / target_accuracy) ** 2
    return int(np.ceil(n))

# For different accuracies
print(f"\nSample Size Needed for Convergence (Roulette):")
target_accuracies = [0.01, 0.05, 0.1, 0.2]

for accuracy in target_accuracies:
    n_needed = compute_convergence_n_for_accuracy(game.ev, game.std, accuracy, confidence=0.95)
    hours_of_play = n_needed / 60  # Assuming 1 bet/minute
    print(f"   Accuracy ±${accuracy:.2f} (95% CI): n={n_needed:6d} bets (~{hours_of_play:6.0f} hours)")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Roulette convergence with confidence bands
ax1 = axes[0, 0]
bets_plot = np.arange(1, n_bets + 1)
ax1.plot(bets_plot, mean_average, 'b-', linewidth=2, label='Mean of 100 paths')
ax1.fill_between(bets_plot, lower_band, upper_band, alpha=0.3, label='95% Confidence Band')
ax1.axhline(y=game.ev, color='red', linestyle='--', linewidth=2, label='True EV')
ax1.set_xlabel('Number of Bets')
ax1.set_ylabel('Average Outcome ($)')
ax1.set_title('Roulette: LLN Convergence (100 simulations)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Multiple games convergence
ax2 = axes[0, 1]
for res in results_comparison:
    ax2.plot(np.arange(1, len(res['mean_path'])+1), res['mean_path'], 
            linewidth=2, label=res['game'])
ax2.set_xlabel('Number of Bets')
ax2.set_ylabel('Average Outcome ($)')
ax2.set_title('Convergence Comparison: Different Games')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Probability of profit vs n
ax3 = axes[1, 0]
ax3.plot(np.arange(1, len(prob_profit)+1), prob_profit * 100, 'o-', linewidth=2, markersize=3)
ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% break-even')
ax3.fill_between(np.arange(1, len(prob_profit)+1), 0, prob_profit*100, alpha=0.3)
ax3.set_xlabel('Number of Bets')
ax3.set_ylabel('P(Profit) %')
ax3.set_title('Roulette: Probability of Positive Return')
ax3.set_ylim(0, 100)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Sample size for accuracy
ax4 = axes[1, 1]
accuracy_range = np.linspace(0.01, 0.3, 30)
n_required = []

for acc in accuracy_range:
    n_needed = compute_convergence_n_for_accuracy(game.ev, game.std, acc, confidence=0.95)
    n_required.append(n_needed)

ax4.semilogy(accuracy_range, n_required, 'o-', linewidth=2, markersize=5, color='purple')
ax4.set_xlabel('Target Accuracy (±$)')
ax4.set_ylabel('Sample Size Required (log scale)')
ax4.set_title('Convergence: Required Sample Size vs Accuracy')
ax4.grid(True, alpha=0.3, which='both')
ax4.invert_xaxis()  # Higher accuracy on right

plt.tight_layout()
plt.savefig('law_of_large_numbers.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: law_of_large_numbers.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY: Law of Large Numbers in Gambling")
print("="*70)
print("✓ Average converges to EV as n→∞ (mathematical certainty)")
print("✓ Convergence rate: SD decreases as 1/√n")
print("✓ House edge GUARANTEES long-term loss for negative EV games")
print("✓ Time horizon critical: Need ~1000s of bets for edge to materialize")
print("✓ Bankroll size must sustain variance until convergence (else ruin before convergence)")
