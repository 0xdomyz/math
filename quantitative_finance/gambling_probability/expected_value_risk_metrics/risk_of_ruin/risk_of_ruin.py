"""
Extracted from: risk_of_ruin.md
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

np.random.seed(42)

print("="*70)
print("RISK OF RUIN: BANKRUPTCY PROBABILITY ANALYSIS")
print("="*70)

# ============================================================================
# 1. ROR FORMULA & CALCULATION
# ============================================================================

print("\n" + "="*70)
print("1. COMPUTING RISK OF RUIN")
print("="*70)

def compute_ror_exponential(ev, sigma2, bankroll):
    """
    Compute risk of ruin using exponential approximation.
    RoR ≈ exp(-2 × |EV| × B / σ²)
    """
    if ev >= 0:
        return 0.0  # +EV → no ruin risk (asymptotically)
    
    exponent = -2 * abs(ev) * bankroll / sigma2
    ror = np.exp(exponent)
    
    return min(ror, 1.0)  # Cap at 100%

# Games and scenarios
scenarios = [
    {'name': 'Roulette, $100 bankroll', 'ev': -1/37, 'sigma2': 1.0, 'b': 100},
    {'name': 'Roulette, $500 bankroll', 'ev': -1/37, 'sigma2': 1.0, 'b': 500},
    {'name': 'Blackjack, $200 bankroll', 'ev': -0.005, 'sigma2': 0.8, 'b': 200},
    {'name': 'Card Counter, $1000 bankroll', 'ev': 0.01, 'sigma2': 1.2, 'b': 1000},
    {'name': 'Poker, $2000 bankroll', 'ev': 0.015, 'sigma2': 2.0, 'b': 2000},
]

print("\nRisk of Ruin for Different Scenarios:")
print(f"{'Scenario':<40} {'EV':>8} {'B':>7} {'σ²':>6} {'RoR':>7}")
print("=" * 70)

ror_results = []
for scenario in scenarios:
    ror = compute_ror_exponential(scenario['ev'], scenario['sigma2'], scenario['b'])
    ror_results.append(ror)
    print(f"{scenario['name']:<40} {scenario['ev']:>8.4f} {scenario['b']:>7} {scenario['sigma2']:>6.2f} {ror*100:>6.2f}%")

# ============================================================================
# 2. BANKROLL SENSITIVITY
# ============================================================================

print("\n" + "="*70)
print("2. BANKROLL IMPACT ON ROR")
print("="*70)

# Roulette scenario: vary bankroll
bankrolls = np.array([50, 100, 200, 500, 1000, 2000, 5000])
ev_roulette = -1/37
sigma2_roulette = 1.0

rors_roulette = [compute_ror_exponential(ev_roulette, sigma2_roulette, b) for b in bankrolls]

print("\nRoulette (p=18/37): RoR vs Bankroll")
for b, ror in zip(bankrolls, rors_roulette):
    print(f"  Bankroll ${b:5d}: RoR = {ror*100:6.2f}%")

# ============================================================================
# 3. BET SIZING ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("3. BET SIZING & RISK OF RUIN")
print("="*70)

# Card counter: how bet size affects RoR
ev_counter = 0.01  # +1% per hand
sigma2_counter = 1.2
bankroll_counter = 500
bet_sizes = np.array([1, 2, 5, 10, 20])

print("\nCard Counter ($500 bankroll, +1% EV):")
print(f"{'Bet Size':>10} {'Effective EV':>15} {'Effective σ²':>15} {'RoR':>8}")
print("-" * 50)

for bet in bet_sizes:
    # Scaling: larger bets = larger effective EV and variance per bet
    eff_ev = ev_counter * bet
    eff_sigma2 = sigma2_counter * bet**2
    ror = compute_ror_exponential(eff_ev, eff_sigma2, bankroll_counter)
    print(f"${bet:>9} {eff_ev:>15.4f} {eff_sigma2:>15.2f} {ror*100:>7.2f}%")

# ============================================================================
# 4. SIMULATING ROR (MONTE CARLO)
# ============================================================================

print("\n" + "="*70)
print("4. MONTE CARLO SIMULATION: ROR VERIFICATION")
print("="*70)

def simulate_ruin(ev, sigma, initial_bankroll, n_simulations=10000, max_bets=10000):
    """
    Simulate betting sequences and track ruin events.
    """
    ruin_count = 0
    
    for _ in range(n_simulations):
        bankroll = initial_bankroll
        
        for bet_num in range(max_bets):
            # Outcome is normally distributed approximation
            outcome = np.random.normal(ev, sigma)
            bankroll += outcome
            
            if bankroll <= 0:
                ruin_count += 1
                break  # Bankrupted
    
    return ruin_count / n_simulations

# Test for a few scenarios
print("\nVerifying formula vs simulation (10,000 Monte Carlo paths):")
print(f"{'Scenario':<40} {'Formula':>10} {'Simulation':>12} {'Diff':>8}")
print("=" * 70)

for scenario in scenarios[:3]:  # Test first 3
    formula_ror = compute_ror_exponential(scenario['ev'], scenario['sigma2'], scenario['b'])
    simulated_ror = simulate_ruin(scenario['ev'], np.sqrt(scenario['sigma2']), scenario['b'], n_simulations=1000)
    diff = abs(formula_ror - simulated_ror)
    print(f"{scenario['name']:<40} {formula_ror*100:>9.2f}% {simulated_ror*100:>11.2f}% {diff*100:>7.2f}%")

# ============================================================================
# 5. MINIMUM BANKROLL CALCULATION
# ============================================================================

print("\n" + "="*70)
print("5. MINIMUM BANKROLL FOR TARGET ROR")
print("="*70)

def compute_min_bankroll(ev, sigma2, target_ror):
    """Solve for B given target RoR."""
    if ev >= 0:
        return 0  # +EV doesn't need bankroll for RoR
    
    b = -np.log(target_ror) * sigma2 / (2 * abs(ev))
    return b

# For different games, compute bankroll needed for 1% RoR
target_ror = 0.01  # 1%

print(f"\nBankroll Required for {target_ror*100:.1f}% Risk of Ruin:")
print(f"{'Game':<30} {'EV per hand':>15} {'σ² per hand':>15} {'Min Bankroll':>15}")
print("=" * 75)

games_minb = [
    ('Roulette', -1/37, 1.0),
    ('Blackjack (basic)', -0.005, 0.8),
    ('Card Counter', 0.01, 1.2),
    ('Poker Pro', 0.02, 2.0),
]

for game_name, ev, sigma2 in games_minb:
    if ev < 0:
        min_b = compute_min_bankroll(ev, sigma2, target_ror)
        print(f"{game_name:<30} {ev:>15.5f} {sigma2:>15.2f} ${min_b:>14.0f}")
    else:
        # +EV: compute for illustrative RoR (say, 10% over different horizon)
        min_b = compute_min_bankroll(ev, sigma2, 0.01)
        print(f"{game_name:<30} {ev:>15.5f} {sigma2:>15.2f} ${min_b:>14.0f} (for 1% RoR)")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bankroll vs RoR (roulette)
ax1 = axes[0, 0]
bankrolls_plot = np.logspace(1.5, 4, 50)  # 30 to 10,000
rors_plot = [compute_ror_exponential(ev_roulette, sigma2_roulette, b) for b in bankrolls_plot]

ax1.semilogy(bankrolls_plot, np.array(rors_plot)*100, linewidth=2, color='red')
ax1.fill_between(bankrolls_plot, np.array(rors_plot)*100, 100, alpha=0.3, color='red')
ax1.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='1% RoR threshold')
ax1.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='10% RoR threshold')
ax1.set_xlabel('Bankroll ($, log scale)')
ax1.set_ylabel('Risk of Ruin (%)')
ax1.set_title('Roulette: Bankroll Impact on RoR')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend()
ax1.set_ylim(0.01, 100)

# Plot 2: Scenario comparison
ax2 = axes[0, 1]
scenario_names = [s['name'].split(',')[0] for s in scenarios]
colors = ['red' if r > 0.5 else 'orange' if r > 0.1 else 'yellow' if r > 0.01 else 'green' for r in ror_results]
bars = ax2.bar(range(len(scenario_names)), np.array(ror_results)*100, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Risk of Ruin (%)')
ax2.set_title('RoR Across Different Games')
ax2.set_xticks(range(len(scenario_names)))
ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y', which='both')

# Add value labels
for bar, ror in zip(bars, ror_results):
    height = bar.get_height()
    if ror > 0.001:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ror*100:.2f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., 0.1,
                f'{ror*100:.4f}%', ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

# Plot 3: Bet sizing impact (card counter)
ax3 = axes[1, 0]
rors_bet = []
for bet in bet_sizes:
    eff_ev = ev_counter * bet
    eff_sigma2 = sigma2_counter * bet**2
    ror = compute_ror_exponential(eff_ev, eff_sigma2, bankroll_counter)
    rors_bet.append(ror)

ax3.plot(bet_sizes, np.array(rors_bet)*100, marker='o', linewidth=2, markersize=10, color='blue')
ax3.fill_between(bet_sizes, 0, np.array(rors_bet)*100, alpha=0.3, color='blue')
ax3.set_xlabel('Bet Size ($)')
ax3.set_ylabel('Risk of Ruin (%)')
ax3.set_title('Card Counter: Bet Size Impact on RoR')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Simulated ruin paths
ax4 = axes[1, 1]
# Roulette: simulate a few paths
n_paths = 20
n_bets = 500
for path_num in range(n_paths):
    bankroll = 100
    path = [bankroll]
    
    for _ in range(n_bets):
        outcome = np.random.choice([1, -1], p=[18/37, 19/37])
        bankroll += outcome
        path.append(bankroll)
        
        if bankroll <= 0:
            break
    
    color = 'red' if bankroll <= 0 else 'green'
    ax4.plot(path, alpha=0.6, color=color)

ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax4.set_xlabel('Bet Number')
ax4.set_ylabel('Bankroll ($)')
ax4.set_title('Roulette: Sample Bankruptcy Paths (red=ruined, green=survived)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('risk_of_ruin_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: risk_of_ruin_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Risk of Ruin exponentially related to bankroll size")
print("✓ Larger bankroll → dramatically lower RoR")
print("✓ Negative EV games → RoR approaches 100% with extended play")
print("✓ +EV games → RoR decreases with more capital and time")
