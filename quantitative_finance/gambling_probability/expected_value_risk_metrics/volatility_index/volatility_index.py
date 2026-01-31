"""
Extracted from: volatility_index.md
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)

print("="*70)
print("VOLATILITY INDEX: GAME CLASSIFICATION & BANKROLL OPTIMIZATION")
print("="*70)

# ============================================================================
# 1. GAME VOLATILITY DEFINITIONS
# ============================================================================

print("\n" + "="*70)
print("1. VOLATILITY INDEX BY GAME")
print("="*70)

games = {
    'Blackjack': {'sigma': 0.95, 'ev': -0.005, 'category': 'Low'},
    'Roulette EU': {'sigma': 0.99, 'ev': -0.027, 'category': 'Medium'},
    'Baccarat': {'sigma': 0.96, 'ev': -0.011, 'category': 'Low'},
    'Craps': {'sigma': 0.94, 'ev': -0.014, 'category': 'Low'},
    'Slots': {'sigma': 1.15, 'ev': -0.05, 'category': 'Medium-High'},
    'Video Poker': {'sigma': 1.00, 'ev': -0.01, 'category': 'Medium'},
    'Keno': {'sigma': 2.8, 'ev': -0.30, 'category': 'Extreme'},
}

df_volatility = pd.DataFrame(games).T
df_volatility['CV (σ/|EV|)'] = df_volatility['sigma'] / df_volatility['ev'].abs()

print("\nGame Volatility Classification:")
print(df_volatility[['sigma', 'ev', 'category']].to_string())

# ============================================================================
# 2. VOLATILITY CATEGORIES & CHARACTERISTICS
# ============================================================================

print("\n" + "="*70)
print("2. VOLATILITY CATEGORIES")
print("="*70)

volatility_bands = {
    'Low': {'min': 0.0, 'max': 0.5, 'desc': 'Tight outcomes', 'examples': ['Blackjack', 'Baccarat']},
    'Medium': {'min': 0.5, 'max': 1.5, 'desc': 'Moderate swings', 'examples': ['Roulette', 'Video Poker']},
    'High': {'min': 1.5, 'max': 3.0, 'desc': 'Large swings', 'examples': ['Slots', 'Poker']},
    'Extreme': {'min': 3.0, 'max': 10.0, 'desc': 'Wild outcomes', 'examples': ['Keno', 'Lotteries']},
}

print(f"\n{'Category':<12} {'Range':<15} {'Characteristics':<30} {'Duration':>12}")
print("=" * 70)

duration_map = {'Low': 'Very long', 'Medium': 'Long', 'High': 'Short', 'Extreme': 'Very short'}

for cat, data in volatility_bands.items():
    range_str = f"{data['min']:.1f}-{data['max']:.1f}"
    print(f"{cat:<12} {range_str:<15} {data['desc']:<30} {duration_map[cat]:>12}")

# ============================================================================
# 3. RISK OF RUIN BY VOLATILITY
# ============================================================================

print("\n" + "="*70)
print("3. RISK OF RUIN: IMPACT OF VOLATILITY")
print("="*70)

def risk_of_ruin(ev, sigma, bankroll, bet_size=1):
    """
    Approximate RoR formula: RoR ≈ exp(-2 × |EV| × B / σ²)
    """
    if sigma == 0 or ev >= 0:
        return 0
    
    ror = np.exp(-2 * abs(ev) * (bankroll / bet_size) / (sigma**2))
    return min(ror, 1.0)

print("\nRisk of Ruin for 100-unit bankroll, $1 bet:")
print(f"{'Game':<20} {'Volatility':>12} {'RoR %':>12}")
print("-" * 45)

for game, params in games.items():
    ror = risk_of_ruin(params['ev'], params['sigma'], 100, bet_size=1)
    print(f"{game:<20} {params['sigma']:>12.2f} {ror*100:>11.1f}%")

# ============================================================================
# 4. SIMULATED SESSION OUTCOMES BY VOLATILITY
# ============================================================================

print("\n" + "="*70)
print("4. SIMULATED SESSIONS: OUTCOME DISTRIBUTION")
print("="*70)

def simulate_session(sigma, ev, n_bets=100, bet_size=1):
    """
    Simulate a gambling session.
    Outcomes roughly normal with mean EV*bet and std dev sigma*bet.
    """
    outcomes = np.random.normal(loc=ev*bet_size, scale=sigma*bet_size, size=n_bets)
    total_return = np.sum(outcomes)
    return total_return

# Simulate 1000 sessions for each game
print("\nSession Results (1000 simulations, 100 bets, $1 bet each):")
print(f"{'Game':<20} {'Mean':<10} {'Std Dev':>10} {'Min':>10} {'Max':>10} {'Win %':>10}")
print("=" * 70)

session_results = {}

for game, params in games.items():
    results = [simulate_session(params['sigma'], params['ev'], n_bets=100, bet_size=1) 
              for _ in range(1000)]
    
    session_results[game] = results
    mean_return = np.mean(results)
    std_return = np.std(results)
    min_return = np.min(results)
    max_return = np.max(results)
    win_pct = np.sum([r > 0 for r in results]) / len(results) * 100
    
    print(f"{game:<20} {mean_return:>9.2f} {std_return:>10.2f} {min_return:>10.2f} {max_return:>10.2f} {win_pct:>9.1f}%")

# ============================================================================
# 5. OPTIMAL BET SIZING BY VOLATILITY
# ============================================================================

print("\n" + "="*70)
print("5. OPTIMAL BET SIZING: KELLY CRITERION ADJUSTED")
print("="*70)

print("\nOptimal bet size as % of bankroll (1% RoR target):")
print(f"{'Game':<20} {'σ':>8} {'EV':>10} {'Kelly %':>12} {'$ per $100B':>15}")
print("=" * 65)

for game, params in games.items():
    # Simplified Kelly: f* = EV / σ²
    # But need to ensure RoR doesn't exceed target
    kelly_fraction = params['ev'] / (params['sigma']**2) if params['sigma'] > 0 else 0
    kelly_fraction = max(0, min(kelly_fraction, 0.02))  # Cap at 2%
    
    bet_per_100 = kelly_fraction * 100
    
    # Adjust down if RoR still too high
    if risk_of_ruin(params['ev'], params['sigma'], 100, bet_size=bet_per_100) > 0.01:
        kelly_fraction *= 0.5  # Reduce by 50%
        bet_per_100 = kelly_fraction * 100
    
    print(f"{game:<20} {params['sigma']:>8.2f} {params['ev']:>10.4f} {kelly_fraction*100:>11.2f}% {bet_per_100:>14.2f}")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Volatility by game (bar chart)
ax1 = axes[0, 0]
game_names = list(games.keys())
sigmas = [games[g]['sigma'] for g in game_names]
colors = ['green' if s < 0.5 else 'yellow' if s < 1.5 else 'orange' if s < 3 else 'red' 
         for s in sigmas]

ax1.bar(range(len(game_names)), sigmas, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Low cutoff')
ax1.axhline(y=1.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='High cutoff')
ax1.axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Extreme cutoff')
ax1.set_xticks(range(len(game_names)))
ax1.set_xticklabels(game_names, rotation=45, ha='right')
ax1.set_ylabel('Volatility (σ)')
ax1.set_title('Game Volatility Classification')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Session outcome distributions
ax2 = axes[0, 1]

for game in ['Blackjack', 'Slots', 'Keno']:
    results = session_results[game]
    ax2.hist(results, bins=50, alpha=0.5, label=game, edgecolor='black')

ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Session Profit/Loss ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Session Outcome Distributions (100 bets)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Risk of Ruin vs Bankroll
ax3 = axes[1, 0]

bankrolls = np.linspace(10, 500, 50)
ror_low = [risk_of_ruin(games['Blackjack']['ev'], games['Blackjack']['sigma'], b) for b in bankrolls]
ror_med = [risk_of_ruin(games['Slots']['ev'], games['Slots']['sigma'], b) for b in bankrolls]
ror_high = [risk_of_ruin(games['Keno']['ev'], games['Keno']['sigma'], b) for b in bankrolls]

ax3.semilogy(bankrolls, ror_low, linewidth=2, label='Blackjack (low vol)', marker='.')
ax3.semilogy(bankrolls, ror_med, linewidth=2, label='Slots (medium vol)', marker='.')
ax3.semilogy(bankrolls, ror_high, linewidth=2, label='Keno (extreme vol)', marker='.')
ax3.axhline(y=0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='1% target')
ax3.set_xlabel('Bankroll (Units)')
ax3.set_ylabel('Risk of Ruin')
ax3.set_title('RoR vs Bankroll by Volatility')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: Coefficient of Variation (risk per unit EV)
ax4 = axes[1, 1]

cv_values = []
for game in game_names:
    cv = games[game]['sigma'] / abs(games[game]['ev'])
    cv_values.append(cv)

colors_cv = ['green' if cv < 50 else 'yellow' if cv < 150 else 'red' for cv in cv_values]
ax4.bar(range(len(game_names)), cv_values, color=colors_cv, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(game_names)))
ax4.set_xticklabels(game_names, rotation=45, ha='right')
ax4.set_ylabel('CV (σ / |EV|)')
ax4.set_title('Coefficient of Variation: Risk per Unit Loss')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('volatility_index_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: volatility_index_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Low volatility: Steady, predictable losses; longer play duration")
print("✓ High volatility: Wild swings; shorter effective duration; requires smaller bets")
print("✓ RoR scales with σ²: Doubling volatility = 4× ruin risk")
print("✓ Bet sizing inverse to volatility: Higher σ → lower bet %")
print("✓ Volatility category selection = risk tolerance match")
