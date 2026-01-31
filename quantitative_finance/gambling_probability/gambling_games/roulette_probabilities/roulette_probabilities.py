"""
Extracted from: roulette_probabilities.md
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_roulette(bet_type='red', num_spins=10000, num_simulations=1000, european=True):
    """
    Simulate roulette spins and track bankroll
    bet_type: 'red', 'black', 'straight', 'dozen', 'split'
    """
    wheels_size = 37 if european else 38
    results = []
    
    for _ in range(num_simulations):
        bankroll = 1000  # Starting bankroll
        spins = np.random.randint(0, wheels_size, size=num_spins)
        
        for spin in spins:
            bet_unit = 10
            
            if bet_type == 'red':
                red_numbers = set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36])
                if spin in red_numbers:
                    bankroll += bet_unit  # Win 1:1
                else:
                    bankroll -= bet_unit  # Lose
                    
            elif bet_type == 'dozen':
                if 1 <= spin <= 12:
                    bankroll += 2 * bet_unit  # Payout 2:1, keep original
                    bankroll -= bet_unit
                else:
                    bankroll -= bet_unit
                    
            elif bet_type == 'straight':
                winning_number = 17  # Arbitrary
                if spin == winning_number:
                    bankroll += 35 * bet_unit
                    bankroll -= bet_unit
                else:
                    bankroll -= bet_unit
        
        results.append(bankroll)
    
    return np.array(results)

# Example 1: Verify house edge
print("=== House Edge Verification ===\n")

np.random.seed(42)

# Simulate different bet types
bet_types_data = []

for bet_type in ['red', 'dozen', 'straight']:
    results = simulate_roulette(bet_type=bet_type, num_spins=100000, num_simulations=100)
    avg_outcome = np.mean(results)
    initial_bankroll = 1000
    expected_outcome = initial_bankroll - initial_bankroll * 0.0270 * (100000 * 10) / 1000
    
    bet_types_data.append({
        'bet': bet_type,
        'avg': avg_outcome,
        'std': np.std(results),
        'min': np.min(results),
        'max': np.max(results)
    })

print(f"{'Bet Type':<15} {'Avg Bankroll':<20} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
print("-" * 80)

for data in bet_types_data:
    print(f"{data['bet']:<15} ${data['avg']:<19,.0f} ${data['std']:<14,.0f} ${data['min']:<14,.0f} ${data['max']:<14,.0f}")

# Example 2: Probability of streaks
print("\n\n=== Streak Probabilities ===\n")

red_numbers = set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36])
p_red = len(red_numbers) / 37
p_black = 18 / 37

streak_lengths = [3, 5, 10, 15, 20]

print(f"{'Streak Length':<20} {'Red/Black Prob':<20} {'Expected Frequency':<25}")
print("-" * 65)

for streak in streak_lengths:
    prob_streak = p_red ** streak
    freq = 1 / prob_streak
    print(f"{streak:<20} {prob_streak:<20.8f} 1 in {freq:<24,.0f}")

# Example 3: Martingale system failure
print("\n\n=== Martingale Strategy Failure Analysis ===\n")

def simulate_martingale(num_rounds=100, p_win=18/37):
    """Simulate martingale doubling strategy"""
    bankroll = 1000
    bet = 10
    losses_in_row = 0
    max_bet_needed = 0
    
    for _ in range(num_rounds):
        if np.random.random() < p_win:
            # Win
            bankroll += bet
            losses_in_row = 0
            bet = 10  # Reset
        else:
            # Loss
            bankroll -= bet
            losses_in_row += 1
            bet *= 2  # Double
            max_bet_needed = max(max_bet_needed, bet)
        
        if bankroll <= 0:
            return bankroll, max_bet_needed, losses_in_row
    
    return bankroll, max_bet_needed, losses_in_row

martingale_results = []
for _ in range(1000):
    final_bank, max_bet, streak = simulate_martingale(num_rounds=100)
    martingale_results.append({
        'final': final_bank,
        'max_bet': max_bet,
        'streak': streak
    })

martingale_results = np.array([r['final'] for r in martingale_results])
max_bets = np.array([r['max_bet'] for r in martingale_results])

print(f"Starting bankroll: $1000")
print(f"Average final: ${np.mean(martingale_results):,.0f}")
print(f"Median final: ${np.median(martingale_results):,.0f}")
print(f"Avg max bet during sequence: ${np.mean(max_bets):,.0f}")
print(f"Bankrupt rate: {np.sum(martingale_results <= 0) / len(martingale_results) * 100:.1f}%")

# Example 4: Wheel bias detection
print("\n\n=== Wheel Bias Detection ===\n")

# Simulate fair wheel
fair_spins = np.random.randint(0, 37, 10000)
fair_counts = np.bincount(fair_spins, minlength=37)

# Simulate biased wheel (number 17 slightly favored)
biased_probs = np.ones(37) / 37
biased_probs[17] *= 2  # Double frequency for 17
biased_probs /= np.sum(biased_probs)
biased_spins = np.random.choice(37, size=10000, p=biased_probs)
biased_counts = np.bincount(biased_spins, minlength=37)

# Chi-square test for fairness
expected_freq = np.ones(37) * (10000 / 37)
chi_square_fair = np.sum((fair_counts - expected_freq) ** 2 / expected_freq)
chi_square_biased = np.sum((biased_counts - expected_freq) ** 2 / expected_freq)
p_value_fair = 1 - stats.chi2.cdf(chi_square_fair, df=36)
p_value_biased = 1 - stats.chi2.cdf(chi_square_biased, df=36)

print(f"Fair wheel chi-square: {chi_square_fair:.2f}, p-value: {p_value_fair:.4f}")
print(f"Biased wheel chi-square: {chi_square_biased:.2f}, p-value: {p_value_biased:.4f}")
print(f"\nInterpretation: p-value < 0.05 suggests bias")
print(f"Fair wheel passes fairness test: {p_value_fair > 0.05}")
print(f"Biased wheel fails fairness test: {p_value_biased > 0.05}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bankroll simulation (red bet)
results_red = simulate_roulette(bet_type='red', num_spins=1000, num_simulations=500)
percentiles = [np.percentile(results_red, p) for p in range(0, 101, 5)]

axes[0, 0].hist(results_red, bins=40, alpha=0.7, color='red', edgecolor='black')
axes[0, 0].axvline(np.mean(results_red), color='darkred', linestyle='--', linewidth=2, label=f"Mean: ${np.mean(results_red):.0f}")
axes[0, 0].axvline(1000, color='green', linestyle='--', linewidth=2, label='Starting: $1000')
axes[0, 0].set_xlabel('Final Bankroll ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Bankroll After 1000 Red Bets (500 simulations)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Streak probability decay
streaks = np.arange(1, 21)
probs_red = (18/37) ** streaks
freqs = 1 / probs_red

axes[0, 1].semilogy(streaks, freqs, 'o-', linewidth=2, markersize=8, color='red')
axes[0, 1].set_xlabel('Streak Length')
axes[0, 1].set_ylabel('Expected Frequency (log scale)')
axes[0, 1].set_title('Red Streak Probability')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Bet type comparison
bet_names = ['Straight\n(1 num)', 'Split\n(2 num)', 'Street\n(3 num)', 'Dozen\n(12 num)', 'Red/Black\n(18 num)']
probabilities = [1/37, 2/37, 3/37, 12/37, 18/37]
payouts = [35, 17, 11, 2, 1]
evs = [prob * payout - (1-prob) for prob, payout in zip(probabilities, payouts)]

x_pos = np.arange(len(bet_names))
axes[1, 0].bar(x_pos, evs, color=['darkred', 'red', 'orange', 'yellow', 'lightcoral'], alpha=0.7)
axes[1, 0].axhline(-0.027, color='black', linestyle='--', linewidth=2, label='Uniform edge')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(bet_names)
axes[1, 0].set_ylabel('EV per $1 bet')
axes[1, 0].set_title('House Edge by Bet Type')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Fair vs biased wheel distribution
axes[1, 1].bar(range(0, 37, 2), fair_counts[::2], alpha=0.5, label='Fair wheel', color='blue')
axes[1, 1].bar(range(1, 37, 2), biased_counts[1::2], alpha=0.5, label='Biased wheel (17 favored)', color='red')
axes[1, 1].axhline(10000/37, color='green', linestyle='--', linewidth=2, label='Expected fair')
axes[1, 1].set_xlabel('Number Pairs')
axes[1, 1].set_ylabel('Frequency (10,000 spins)')
axes[1, 1].set_title('Wheel Fairness Check')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
