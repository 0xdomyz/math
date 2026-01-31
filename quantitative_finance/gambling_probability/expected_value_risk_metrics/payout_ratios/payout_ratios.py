"""
Extracted from: payout_ratios.md
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

print("="*70)
print("PAYOUT RATIOS: LONG-TERM IMPACT ANALYSIS")
print("="*70)

# ============================================================================
# 1. PAYOUT RATIOS BY GAME
# ============================================================================

print("\n" + "="*70)
print("1. PAYOUT RATIOS (RTP) BY GAME")
print("="*70)

games_payout = {
    'Blackjack (basic strategy)': 0.995,
    'Craps (Pass/odds)': 0.9859,
    'Baccarat (Banker)': 0.9894,
    'European Roulette': 0.973,
    'American Roulette': 0.9474,
    'Slots (typical)': 0.95,
    'Video Poker (good table)': 0.99,
    'Keno': 0.70,
}

df_payout = pd.DataFrame({
    'Game': list(games_payout.keys()),
    'Payout Ratio (RTP)': list(games_payout.values()),
})

df_payout['House Edge %'] = (1 - df_payout['Payout Ratio (RTP)']) * 100
df_payout = df_payout.sort_values('Payout Ratio (RTP)', ascending=False)

print("\nGame Rankings by Payout Ratio:")
print(df_payout.to_string(index=False))

# ============================================================================
# 2. CUMULATIVE LOSS CALCULATION
# ============================================================================

print("\n" + "="*70)
print("2. CUMULATIVE IMPACT: LOSSES OVER TIME")
print("="*70)

bet_amounts = [1, 5, 10, 100]
n_bets_list = [100, 1000, 10000]

print(f"\nExpected Loss by Game & Number of Bets:")
print(f"(Bet amount = $1)\n")

cumulative_results = []

for n_bets in n_bets_list:
    print(f"\n{n_bets} bets:")
    for game, rtp in games_payout.items():
        house_edge = 1 - rtp
        expected_loss = n_bets * house_edge * 1.0  # $1 bet
        cumulative_results.append({
            'Game': game,
            'N_bets': n_bets,
            'Expected_loss': expected_loss
        })
        print(f"  {game:30s}: ${expected_loss:7.2f} loss")

# ============================================================================
# 3. COMPARISON: PAYOUT RATIO IMPACT
# ============================================================================

print("\n" + "="*70)
print("3. COMPARING TWO STRATEGIES")
print("="*70)

print("\nStrategy A: Play Blackjack (99.5% RTP)")
print("Strategy B: Play Keno (70% RTP)")
print("Same bet: $1/hand, 1000 hands")

n_hands = 1000
bet_size = 1
loss_blackjack = (1 - 0.995) * n_hands * bet_size
loss_keno = (1 - 0.70) * n_hands * bet_size
difference = loss_keno - loss_blackjack

print(f"\nBlackjack expected loss: ${loss_blackjack:.2f}")
print(f"Keno expected loss: ${loss_keno:.2f}")
print(f"Additional loss from worse RTP: ${difference:.2f}")
print(f"Multiple: {difference / loss_blackjack:.1f}x worse")

# ============================================================================
# 4. HOURLY LOSS RATE
# ============================================================================

print("\n" + "="*70)
print("4. HOURLY LOSS RATES")
print("="*70)

# Assume hands per hour
hands_per_hour = {
    'Blackjack': 60,
    'Craps': 100,
    'Roulette': 30,
    'Slots': 600,
}

bet_size_hourly = 1

print(f"\nHourly loss rate for different games (${bet_size_hourly} bet):")
for game_short, hands in hands_per_hour.items():
    for game_full, rtp in games_payout.items():
        if game_short.lower() in game_full.lower():
            house_edge = 1 - rtp
            hourly_loss = hands * bet_size_hourly * house_edge
            print(f"  {game_full:30s}: ${hourly_loss:6.2f}/hour ({hands} hands/hr)")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Payout ratio by game
ax1 = axes[0, 0]
colors_payout = ['green' if rtp > 0.97 else 'orange' if rtp > 0.92 else 'red' 
                 for rtp in df_payout['Payout Ratio (RTP)']]
ax1.barh(df_payout['Game'], df_payout['Payout Ratio (RTP)'], color=colors_payout, alpha=0.7, edgecolor='black')
ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Break-even (100%)')
ax1.set_xlabel('Payout Ratio (RTP)')
ax1.set_xlim(0.65, 1.01)
ax1.set_title('Payout Ratios by Game')
ax1.grid(True, alpha=0.3, axis='x')
ax1.legend()

# Plot 2: Cumulative loss over bets
ax2 = axes[0, 1]
for game, rtp in list(games_payout.items())[:5]:  # Top 5 games
    house_edge = 1 - rtp
    losses = np.array(n_bets_list) * house_edge * 1.0
    ax2.plot(n_bets_list, losses, marker='o', linewidth=2, markersize=8, label=game.split('(')[0].strip())

ax2.set_xlabel('Number of Bets')
ax2.set_ylabel('Expected Loss ($)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Cumulative Loss vs Bets (log scale)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: House edge comparison
ax3 = axes[1, 0]
he_values = df_payout['House Edge %']
colors_he = ['green' if he < 2 else 'orange' if he < 5 else 'red' for he in he_values]
ax3.bar(range(len(df_payout)), he_values, color=colors_he, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(df_payout)))
ax3.set_xticklabels([g.split('(')[0].strip() for g in df_payout['Game']], rotation=45, ha='right')
ax3.set_ylabel('House Edge (%)')
ax3.set_title('House Edge by Game')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, he in enumerate(he_values):
    ax3.text(i, he + 0.3, f'{he:.1f}%', ha='center', fontsize=9)

# Plot 4: Loss comparison: Blackjack vs Keno (1000 bets)
ax4 = axes[1, 1]
games_compare = ['Blackjack\n(99.5% RTP)', 'Keno\n(70% RTP)']
losses_compare = [loss_blackjack, loss_keno]
colors_compare = ['green', 'red']

bars = ax4.bar(games_compare, losses_compare, color=colors_compare, alpha=0.7, edgecolor='black', width=0.5)
ax4.set_ylabel('Expected Loss ($)')
ax4.set_title(f'Loss Comparison: 1000 bets of $1')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels and multiplier
for i, (bar, loss) in enumerate(zip(bars, losses_compare)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'${loss:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add multiplier annotation
mid_x = 0.5
mid_y = (losses_compare[0] + losses_compare[1]) / 2
ax4.text(mid_x, mid_y, f'{difference / loss_blackjack:.0f}x\nworse', 
        ha='center', va='center', fontsize=12, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('payout_ratios_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: payout_ratios_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Payout ratio (RTP) directly determines long-term profitability")
print("✓ 1% difference compounds: 100 bets = $1 difference, 10,000 bets = $100 difference")
print("✓ Game selection critical: Play highest RTP available")
print("✓ Blackjack (99.5%) vastly superior to Keno (70%)")
