"""
Extracted from: card_counting.md
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)

print("="*70)
print("CARD COUNTING: TRACKING BLACKJACK ADVANTAGE")
print("="*70)

# ============================================================================
# 1. HI-LO COUNTING SYSTEM
# ============================================================================

print("\n" + "="*70)
print("1. HI-LO SYSTEM MECHANICS")
print("="*70)

card_values_hiLo = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,  # Low cards
    '7': 0, '8': 0, '9': 0,                   # Neutral
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1  # High cards
}

def create_deck(num_decks=6):
    """Create a shoe with num_decks standard decks."""
    cards = []
    for _ in range(num_decks):
        cards.extend(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4)
    np.random.shuffle(cards)
    return cards

def calculate_true_count(running_count, cards_remaining):
    """Convert running count to true count."""
    decks_remaining = cards_remaining / 52  # 52 cards per deck
    if decks_remaining > 0:
        return running_count / decks_remaining
    return running_count

def get_bet_size(true_count, base_bet=10):
    """Determine bet size based on true count."""
    # Betting strategy: $10 base, increase with positive count
    if true_count < 0:
        return base_bet  # Minimum bet on negative count
    elif true_count < 1:
        return base_bet
    elif true_count < 2:
        return base_bet * 1.5
    elif true_count < 3:
        return base_bet * 2
    elif true_count < 4:
        return base_bet * 3
    else:
        return base_bet * 4  # Max spread (4x bet)

# ============================================================================
# 2. SIMULATE 1000 HANDS OF COUNTING
# ============================================================================

print("\nSimulating 1000 hands with Hi-Lo card counting:")

deck = create_deck(num_decks=6)
running_count = 0
hands_played = 0
deck_position = 0
penetration_threshold = int(len(deck) * 0.75)  # 75% penetration before reshuffle

hand_results = []
bet_sizes = []
true_counts = []
counts = []

# Simulate 1000 hands
for hand in range(1000):
    # Check if reshuffle needed (penetration)
    if deck_position >= penetration_threshold:
        deck = create_deck(num_decks=6)
        running_count = 0
        deck_position = 0
    
    # Draw 2 cards for dealer, 2 for player (simplified)
    cards_in_hand = []
    for _ in range(4):
        if deck_position < len(deck):
            card = deck[deck_position]
            cards_in_hand.append(card)
            running_count += card_values_hiLo[card]
            deck_position += 1
    
    cards_remaining = len(deck) - deck_position
    true_count = calculate_true_count(running_count, cards_remaining)
    bet_size = get_bet_size(true_count)
    
    # Simulate hand outcome (simplified: random win/loss)
    # In reality, outcome depends on player/dealer hands
    player_wins = np.random.random() > 0.51  # ~49% player win rate (accounting for push, BJ)
    
    if player_wins:
        profit = bet_size  # Simplified (ignore blackjack 3:2 payout)
    else:
        profit = -bet_size
    
    hand_results.append(profit)
    bet_sizes.append(bet_size)
    true_counts.append(true_count)
    counts.append(running_count)
    hands_played += 1

# ============================================================================
# 3. ANALYSIS: COUNTING METRICS
# ============================================================================

print(f"\nSimulation Results (1000 hands):")
print(f"{'Metric':<30} {'Value':>20}")
print("=" * 50)

total_wagered = np.sum(bet_sizes)
total_profit = np.sum(hand_results)
avg_profit_per_hand = total_profit / hands_played
win_rate = np.sum([1 for x in hand_results if x > 0]) / hands_played

print(f"{'Total hands played':<30} {hands_played:>20}")
print(f"{'Total wagered':<30} ${total_wagered:>19,.2f}")
print(f"{'Total profit/loss':<30} ${total_profit:>19,.2f}")
print(f"{'Avg profit per hand':<30} ${avg_profit_per_hand:>19,.2f}")
print(f"{'Win rate':<30} {win_rate*100:>19.1f}%")
print(f"{'Hands with +TC':<30} {np.sum([1 for tc in true_counts if tc > 0]):>20}")
print(f"{'Max true count reached':<30} {max(true_counts):>19.2f}")
print(f"{'Min true count reached':<30} {min(true_counts):>19.2f}")
print(f"{'Avg true count':<30} {np.mean(true_counts):>19.2f}")

# ============================================================================
# 4. BET SIZING ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("2. BET SIZING BY TRUE COUNT")
print("="*70)

bet_by_count = defaultdict(list)
for tc, bet in zip(true_counts, bet_sizes):
    tc_bucket = int(tc * 2) / 2  # Round to nearest 0.5
    bet_by_count[tc_bucket].append(bet)

print(f"\n{'True Count Range':<20} {'Avg Bet':<15} {'Frequency':>15}")
print("=" * 50)

for tc in sorted(bet_by_count.keys())[-10:]:  # Show top 10 TC values
    avg_bet = np.mean(bet_by_count[tc])
    freq = len(bet_by_count[tc])
    print(f"{tc:>5.1f} to {tc+0.5:>5.1f}       ${avg_bet:>12.2f}   {freq:>15}")

# ============================================================================
# 5. CUMULATIVE PROFIT TRACKING
# ============================================================================

cumulative_profit = np.cumsum(hand_results)
cumulative_wagered = np.cumsum(bet_sizes)

print("\n" + "="*70)
print("3. PROFIT/LOSS EVOLUTION")
print("="*70)

milestones = [250, 500, 750, 1000]
print(f"\n{'Hands':<10} {'Cumulative P/L':<20} {'Avg Bet':<15} {'ROI %':>15}")
print("=" * 60)

for milestone in milestones:
    cum_pl = cumulative_profit[milestone - 1]
    cum_bet = cumulative_wagered[milestone - 1]
    roi = (cum_pl / cum_bet * 100) if cum_bet > 0 else 0
    avg_bet = cum_bet / milestone
    print(f"{milestone:<10} ${cum_pl:>17,.2f}   ${avg_bet:>12.2f}   {roi:>14.2f}%")

# ============================================================================
# 6. DETECTION RISK ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("4. BETTING PATTERN ANALYSIS (Detection Risk)")
print("="*70)

# Analyze betting correlation to true count (suspicious pattern)
bet_correlation = np.corrcoef(true_counts, bet_sizes)[0, 1]

print(f"\nBet-Count correlation: {bet_correlation:.3f}")
print(f"(>0.7 = suspicious pattern that casinos detect)")

# Count instances of max bet spread
max_bet = max(bet_sizes)
min_bet = min(bet_sizes)
spread_ratio = max_bet / min_bet

print(f"\nBet spread ratio: {spread_ratio:.1f}:1")
print(f"(>4:1 = very suspicious; typical casino tolerance ~2:1)")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative profit vs hands
ax1 = axes[0, 0]
hands_x = np.arange(1, len(cumulative_profit) + 1)
ax1.plot(hands_x, cumulative_profit, linewidth=2, color='blue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.fill_between(hands_x, cumulative_profit, 0, alpha=0.3)
ax1.set_xlabel('Hand Number')
ax1.set_ylabel('Cumulative Profit/Loss ($)')
ax1.set_title('Profit Evolution Over 1000 Hands')
ax1.grid(True, alpha=0.3)

# Plot 2: True count distribution
ax2 = axes[0, 1]
ax2.hist(true_counts, bins=50, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral (TC=0)')
ax2.axvline(x=np.mean(true_counts), color='blue', linestyle='--', linewidth=2, label=f'Mean ({np.mean(true_counts):.2f})')
ax2.set_xlabel('True Count')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of True Counts')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Bet size vs true count (should show correlation)
ax3 = axes[1, 0]
ax3.scatter(true_counts[::10], bet_sizes[::10], alpha=0.5, s=30)  # Plot every 10th to reduce clutter
ax3.set_xlabel('True Count')
ax3.set_ylabel('Bet Size ($)')
ax3.set_title(f'Bet Size vs True Count (Correlation: {bet_correlation:.3f})')
ax3.grid(True, alpha=0.3)

# Plot 4: Running count progression
ax4 = axes[1, 1]
ax4.plot(counts, linewidth=1, alpha=0.7, color='purple')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Hand Number')
ax4.set_ylabel('Running Count')
ax4.set_title('Running Count Progression (with reshuffles)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('card_counting_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: card_counting_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Hi-Lo counting achieves 0.5-1.5% edge in favorable shoes")
print("✓ True count fluctuates; affects betting and strategy decisions")
print("✓ Bet correlation high → casino surveillance detects pattern")
print("✓ Long-term: Edge + variance requires 100-200× minimum bet bankroll")
print("✓ Career duration: Average 2-5 years before detection/barring")
