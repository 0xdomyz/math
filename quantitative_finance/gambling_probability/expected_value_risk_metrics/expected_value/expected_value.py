"""
Extracted from: expected_value.md
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)

print("="*70)
print("EXPECTED VALUE: GAME COMPARISON & SIMULATION")
print("="*70)

# ============================================================================
# 1. CALCULATE EV FOR DIFFERENT GAMES
# ============================================================================

print("\n" + "="*70)
print("1. EXPECTED VALUE CALCULATION ACROSS GAMES")
print("="*70)

class CasinoGame:
    """Represents a casino game with outcomes and probabilities."""
    
    def __init__(self, name, outcomes, probabilities, bet_amount=1.0):
        """
        Args:
            name: Game name
            outcomes: List of outcome values (profit/loss)
            probabilities: List of outcome probabilities (must sum to 1)
            bet_amount: Default bet size ($)
        """
        self.name = name
        self.outcomes = np.array(outcomes)
        self.probabilities = np.array(probabilities)
        self.bet_amount = bet_amount
        
        assert np.isclose(np.sum(probabilities), 1.0), "Probabilities must sum to 1"
        
        # Calculate EV
        self.ev = np.sum(self.outcomes * self.probabilities)
        self.ev_percentage = self.ev / bet_amount * 100
        
        # Calculate variance and std dev
        self.variance = np.sum((self.outcomes - self.ev)**2 * self.probabilities)
        self.std_dev = np.sqrt(self.variance)
    
    def display_summary(self):
        """Print game summary."""
        print(f"\n{self.name}:")
        print(f"   Outcomes: {self.outcomes}")
        print(f"   Probabilities: {self.probabilities}")
        print(f"   Expected Value: ${self.ev:.4f} per ${self.bet_amount:.2f} bet")
        print(f"   EV %: {self.ev_percentage:.2f}%")
        print(f"   Std Dev: ${self.std_dev:.4f}")
        print(f"   Risk-Adjusted: EV/σ = {self.ev/self.std_dev:.4f}")

# Define games
games = []

# Roulette (European)
roulette_ev = CasinoGame(
    "Roulette (European, bet red)",
    outcomes=[1, -1],  # Win $1 or lose $1
    probabilities=[18/37, 19/37],  # Red prob, not red prob
    bet_amount=1.0
)
games.append(roulette_ev)

# Blackjack (basic strategy)
blackjack_ev = CasinoGame(
    "Blackjack (basic strategy)",
    outcomes=[1, -1.05, 0.5],  # Win $1, lose, or push (tie)
    probabilities=[0.48, 0.49, 0.03],
    bet_amount=1.0
)
games.append(blackjack_ev)

# Craps (Pass line)
craps_ev = CasinoGame(
    "Craps (Pass line)",
    outcomes=[1, -1],
    probabilities=[49/100, 51/100],  # Simplified
    bet_amount=1.0
)
games.append(craps_ev)

# Slots (typical)
slots_ev = CasinoGame(
    "Slot Machine (typical)",
    outcomes=[9, 99, 999, -1],  # Various payouts
    probabilities=[0.04, 0.005, 0.0001, 0.9549],
    bet_amount=1.0
)
games.append(slots_ev)

# Fair coin (for reference)
fair_coin = CasinoGame(
    "Fair Coin Flip",
    outcomes=[1, -1],
    probabilities=[0.5, 0.5],
    bet_amount=1.0
)
games.append(fair_coin)

print("\nGame Comparison:")
for game in games:
    game.display_summary()

# ============================================================================
# 2. RANK GAMES BY EV & RISK-ADJUSTED RETURNS
# ============================================================================

print("\n" + "="*70)
print("2. RANKING GAMES BY EXPECTED VALUE")
print("="*70)

df_games = pd.DataFrame({
    'Game': [g.name for g in games],
    'EV ($)': [g.ev for g in games],
    'EV (%)': [g.ev_percentage for g in games],
    'Std Dev ($)': [g.std_dev for g in games],
    'Sharpe Ratio (EV/σ)': [g.ev / g.std_dev for g in games],
    'Sortino Ratio': [g.ev / g.std_dev for g in games]  # Simplified
})

df_games_sorted = df_games.sort_values('EV ($)', ascending=False)
print("\nRanked by Expected Value (best to worst):")
print(df_games_sorted.to_string(index=False))

# ============================================================================
# 3. SIMULATE BETTING OUTCOMES
# ============================================================================

print("\n" + "="*70)
print("3. SIMULATING BETTING OVER TIME")
print("="*70)

def simulate_game(game, n_bets, bet_size=1.0, n_simulations=100):
    """
    Simulate n independent bets; return cumulative outcomes.
    
    Returns:
        paths: (n_simulations, n_bets) array of cumulative profits
        expected_path: Average path across simulations
    """
    paths = []
    
    for sim in range(n_simulations):
        # Generate outcomes based on probabilities
        bets = np.random.choice(
            game.outcomes,
            size=n_bets,
            p=game.probabilities
        )
        
        # Multiply by bet size and compute cumulative sum
        profits = bets * bet_size
        cumsum = np.cumsum(profits)
        paths.append(cumsum)
    
    paths = np.array(paths)
    expected_path = np.mean(paths, axis=0)
    
    return paths, expected_path

# Simulate each game
n_bets = 1000
simulations_per_game = {}

print(f"\nSimulating {n_bets} bets per game ({100} simulations each):")
for game in games:
    paths, expected = simulate_game(game, n_bets, bet_size=game.bet_amount, n_simulations=100)
    simulations_per_game[game.name] = {'paths': paths, 'expected': expected}
    
    final_avg = np.mean(paths[:, -1])
    final_std = np.std(paths[:, -1])
    theoretical_loss = game.ev * n_bets
    
    print(f"\n{game.name}:")
    print(f"   Theoretical total EV: ${theoretical_loss:.2f}")
    print(f"   Simulated avg outcome: ${final_avg:.2f} ± ${final_std:.2f}")
    print(f"   % of bets profitable: {100 * np.mean(paths[:, -1] > 0):.1f}%")

# ============================================================================
# 4. CALCULATING EV FOR SPECIFIC BETS
# ============================================================================

print("\n" + "="*70)
print("4. EV CALCULATION FOR CUSTOM BETS")
print("="*70)

# Sports betting scenario
print("\nSports Betting Scenario:")
print("   Bet: $100 on Team A to win")
print("   Your estimated P(Team A wins): 0.55")
print("   Sportsbook odds: -110 (American)")
print("   ")
print("   Implied probability: 110 / (110 + 100) = 0.524")
print("   ")

your_prob = 0.55
implied_prob = 110 / (110 + 100)
bet_amount = 100
win_payout = 100  # Net gain if win

ev_sports = (your_prob * win_payout) - ((1 - your_prob) * bet_amount)
print(f"   Your EV: (0.55 × $100) - (0.45 × $100) = ${ev_sports:.2f}")
print(f"   Your EV %: {(ev_sports / bet_amount) * 100:.2f}%")
print(f"   Decision: {'✓ BET' if ev_sports > 0 else '✗ PASS'}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: EV comparison (bar chart)
ax1 = axes[0, 0]
colors_ev = ['green' if ev > 0 else 'red' for ev in df_games['EV (%)']]
bars = ax1.barh(df_games['Game'], df_games['EV (%)'], color=colors_ev, alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Expected Value (%)')
ax1.set_title('Expected Value Comparison Across Games')
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (game, ev) in enumerate(zip(df_games['Game'], df_games['EV (%)'])):
    ax1.text(ev + 0.1 if ev > 0 else ev - 0.1, i, f'{ev:.2f}%', 
            va='center', ha='left' if ev > 0 else 'right', fontsize=9)

# Plot 2: Roulette simulation (cumulative profit)
ax2 = axes[0, 1]
roulette_paths = simulations_per_game['Roulette (European, bet red)']['paths']
roulette_expected = simulations_per_game['Roulette (European, bet red)']['expected']

# Plot sample paths
for path in roulette_paths[:20]:  # Plot first 20 simulations
    ax2.plot(path, alpha=0.1, color='red')

ax2.plot(roulette_expected, color='red', linewidth=3, label='Expected Path')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Bet Number')
ax2.set_ylabel('Cumulative Profit ($)')
ax2.set_title('Roulette: Cumulative Outcomes (20 simulations)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of outcomes by game
ax3 = axes[1, 0]
final_outcomes_all = []
game_labels_all = []

for game_name, sim_data in simulations_per_game.items():
    final_outcomes = sim_data['paths'][:, -1]
    final_outcomes_all.append(final_outcomes)
    game_labels_all.append(game_name.split(' (')[0])  # Shorten label

bp = ax3.boxplot(final_outcomes_all, labels=game_labels_all, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_ev):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_ylabel('Final Outcome ($) after 1000 bets')
ax3.set_title('Distribution of Final Outcomes by Game')
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Risk vs Reward (scatter)
ax4 = axes[1, 1]
ax4.scatter(df_games['Std Dev ($)'], df_games['EV ($)'], s=200, alpha=0.7, c=colors_ev)

for idx, row in df_games.iterrows():
    ax4.annotate(row['Game'].split(' (')[0], 
                (row['Std Dev ($)'], row['EV ($)']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Break-even')
ax4.set_xlabel('Risk (Std Dev, $)')
ax4.set_ylabel('Expected Value ($)')
ax4.set_title('Risk-Return Profile')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('expected_value_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: expected_value_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ EV metric ranks games by long-term profitability")
print("✓ Negative EV games → guaranteed loss over time")
print("✓ Only play +EV games or minimize -EV exposure")
print("✓ Risk-return trade-off: High std dev with negative EV is worst")
