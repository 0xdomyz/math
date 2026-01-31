"""
Extracted from: variance_volatility.md
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)

print("="*70)
print("VARIANCE AND VOLATILITY: RISK ANALYSIS")
print("="*70)

# ============================================================================
# 1. COMPUTE VARIANCE FOR DIFFERENT GAMES
# ============================================================================

print("\n" + "="*70)
print("1. VARIANCE CALCULATION ACROSS GAMES")
print("="*70)

class Game:
    """Represents a gambling game with variance properties."""
    
    def __init__(self, name, outcomes, probabilities, bet_amount=1.0):
        self.name = name
        self.outcomes = np.array(outcomes)
        self.probabilities = np.array(probabilities)
        self.bet_amount = bet_amount
        
        # EV
        self.ev = np.sum(self.outcomes * self.probabilities)
        
        # Variance & Std Dev
        self.variance = np.sum((self.outcomes - self.ev)**2 * self.probabilities)
        self.std_dev = np.sqrt(self.variance)
        
        # CV
        self.cv = self.std_dev / abs(self.ev) if self.ev != 0 else np.inf
    
    def display(self):
        print(f"\n{self.name}:")
        print(f"   EV: ${self.ev:.4f}")
        print(f"   Variance: {self.variance:.4f}")
        print(f"   Std Dev (σ): ${self.std_dev:.4f}")
        print(f"   CV (Risk per $ return): {self.cv:.2f}")

# Define games
games = [
    Game("Roulette (EU, red)", [1, -1], [18/37, 19/37]),
    Game("Blackjack (basic)", [1, -1, 0.5], [0.48, 0.49, 0.03]),
    Game("Craps (pass)", [1, -1], [0.49, 0.51]),
    Game("Slots", [9, 99, 999, -1], [0.04, 0.005, 0.0001, 0.9549]),
    Game("Poker", [100, -100], [0.55, 0.45]),  # Hypothetical +EV
]

print("\nVariance Comparison:")
for game in games:
    game.display()

# ============================================================================
# 2. BANKROLL SIZING BASED ON VARIANCE
# ============================================================================

print("\n" + "="*70)
print("2. SAFE BET SIZING & BANKROLL REQUIREMENTS")
print("="*70)

def compute_bet_sizing(game, bankroll, sessions=1, hands_per_session=100, safety_factor=3):
    """
    Compute safe bet size and ruin risk.
    
    safety_factor: k where bankroll ≥ k × σ × √(n hands)
                   k=3 → 99.7% safety (3 sigma)
    """
    total_hands = sessions * hands_per_session
    
    # Variance scales as n
    total_std_dev = game.std_dev * np.sqrt(total_hands)
    
    # Safe bet = bankroll / (safety_factor × total_std_dev)
    safe_bet = bankroll / (safety_factor * total_std_dev)
    
    # Expected outcome
    expected_outcome = game.ev * total_hands * safe_bet
    
    # Simple ruin probability (approximation)
    z_score = (expected_outcome) / total_std_dev
    ruin_prob = stats.norm.sf(-z_score)  # P(X < -bankroll)
    
    return {
        'safe_bet': safe_bet,
        'expected_outcome': expected_outcome,
        'total_std_dev': total_std_dev,
        'ruin_prob': ruin_prob
    }

bankroll = 500
results = []

print(f"\nBankroll: ${bankroll}, 5 sessions × 100 hands each:")
for game in games:
    sizing = compute_bet_sizing(game, bankroll, sessions=5, hands_per_session=100)
    results.append({
        'Game': game.name,
        'Safe Bet ($)': sizing['safe_bet'],
        'Expected ($)': sizing['expected_outcome'],
        'Total σ': sizing['total_std_dev'],
        'Ruin %': sizing['ruin_prob'] * 100
    })

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# ============================================================================
# 3. CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*70)
print("3. CONFIDENCE INTERVALS FOR OUTCOMES")
print("="*70)

def compute_ci(game, n_bets, bet_amount=1.0, confidence=0.95):
    """Compute confidence interval for outcome distribution."""
    total_ev = game.ev * n_bets * bet_amount
    total_std = game.std_dev * np.sqrt(n_bets) * bet_amount
    
    z = stats.norm.ppf((1 + confidence) / 2)
    lower = total_ev - z * total_std
    upper = total_ev + z * total_std
    
    return lower, upper

n_test = 1000
bet = 1.0

print(f"\n{n_test} bets of ${bet} each, 95% confidence interval:")
for game in games:
    lower, upper = compute_ci(game, n_test, bet)
    midpoint = game.ev * n_test * bet
    print(f"\n{game.name}:")
    print(f"   Expected: ${midpoint:.2f}")
    print(f"   95% CI: [${lower:.2f}, ${upper:.2f}]")
    print(f"   Range: ${upper - lower:.2f}")

# ============================================================================
# 4. VARIANCE IMPACT ON OUTCOMES (SIMULATION)
# ============================================================================

print("\n" + "="*70)
print("4. SIMULATING VARIANCE IMPACT")
print("="*70)

def simulate_outcomes(game, n_bets, bet_amount=1.0, n_sims=1000):
    """Simulate many independent plays."""
    outcomes = []
    
    for _ in range(n_sims):
        bets = np.random.choice(game.outcomes, size=n_bets, p=game.probabilities)
        total = np.sum(bets) * bet_amount
        outcomes.append(total)
    
    return np.array(outcomes)

n_bets_test = 100
bet_amount_test = 5

print(f"\n{n_bets_test} bets of ${bet_amount_test} ({1000} simulations):")
for game in games:
    outcomes = simulate_outcomes(game, n_bets_test, bet_amount_test)
    
    mean_outcome = np.mean(outcomes)
    std_outcome = np.std(outcomes)
    min_outcome = np.min(outcomes)
    max_outcome = np.max(outcomes)
    prob_profit = np.mean(outcomes > 0)
    
    print(f"\n{game.name}:")
    print(f"   Mean: ${mean_outcome:.2f}")
    print(f"   Std: ${std_outcome:.2f}")
    print(f"   Range: [${min_outcome:.2f}, ${max_outcome:.2f}]")
    print(f"   P(profit): {prob_profit*100:.1f}%")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variance comparison
ax1 = axes[0, 0]
names = [g.name.split('(')[0].strip() for g in games]
stds = [g.std_dev for g in games]
evs = [g.ev for g in games]
colors = ['green' if ev > 0 else 'red' for ev in evs]

ax1.bar(names, stds, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Standard Deviation ($)')
ax1.set_title('Volatility Comparison (σ)')
ax1.grid(True, alpha=0.3, axis='y')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Safe bet sizing
ax2 = axes[0, 1]
ax2.barh(df_results['Game'], df_results['Safe Bet ($)'], color='blue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Safe Bet Size ($)')
ax2.set_title(f'Safe Bet Sizing (Bankroll: ${bankroll})')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Distribution of outcomes (roulette vs poker)
ax3 = axes[1, 0]
roulette_outcomes = simulate_outcomes(games[0], 100, 5)
poker_outcomes = simulate_outcomes(games[4], 100, 5)

ax3.hist(roulette_outcomes, bins=30, alpha=0.6, label='Roulette', color='red', edgecolor='black')
ax3.hist(poker_outcomes, bins=30, alpha=0.6, label='Poker (+EV)', color='green', edgecolor='black')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('Final Outcome ($)')
ax3.set_ylabel('Frequency')
ax3.set_title('Outcome Distributions (100 bets)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: EV vs Variance (risk-return plot)
ax4 = axes[1, 1]
evs_plot = [g.ev for g in games]
stds_plot = [g.std_dev for g in games]
colors_plot = ['green' if e > 0 else 'red' for e in evs_plot]

ax4.scatter(stds_plot, evs_plot, s=300, c=colors_plot, alpha=0.7, edgecolor='black')
for name, std, ev in zip(names, stds_plot, evs_plot):
    ax4.annotate(name, (std, ev), xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Risk (Std Dev, σ)')
ax4.set_ylabel('Expected Value ($)')
ax4.set_title('Risk-Return Profile')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('variance_volatility_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: variance_volatility_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Variance measures risk (spread of outcomes)")
print("✓ High variance → larger bankroll needed for same bet size")
print("✓ Safe bet size inversely proportional to √(variance)")
print("✓ Combine EV and variance for complete risk assessment")
