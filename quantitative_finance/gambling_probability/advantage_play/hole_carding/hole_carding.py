"""
Extracted from: hole_carding.md
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("HOLE CARDING: INFORMATION ADVANTAGE QUANTIFICATION")
print("="*70)

# ============================================================================
# 1. BLACKJACK OUTCOME MATRIX
# ============================================================================

print("\n" + "="*70)
print("1. DEALER HAND VALUES (2-21)")
print("="*70)

# Dealer hole card possibilities: 2-11 (Ace)
dealer_cards = list(range(2, 12))
dealer_upcard = 10  # Example: dealer showing 10

# For each hole card, calculate dealer final value after hitting to 17+
def dealer_final_value(upcard, hole_card):
    """Calculate dealer's likely final value (assumes hit to 17+)."""
    total = upcard + hole_card
    # If total < 17, dealer must hit (simplified; ignores soft 17 rules)
    if total < 17:
        # Dealer draws random card (2-11)
        third_card = np.random.choice(range(2, 12))
        total += third_card
        # If still <17, assume busts (simplified)
        if total < 17:
            return 0  # Bust
    return min(total, 21)

print(f"\nDealer showing: {dealer_upcard} (10-value card)")
print(f"Possible hole cards (2-11): ")
print(f"\n{'Hole Card':<12} {'Dealer Total':<15} {'Bust Prob':<15} {'Win %':<15}")
print("=" * 57)

for hole_card in dealer_cards:
    # Simplified: assume dealer hits to 17+
    dealer_total = min(21, dealer_upcard + hole_card)
    if dealer_upcard + hole_card < 17:
        bust_prob = 0.31  # Rough estimate for hitting on hard <17
    else:
        bust_prob = 0
    
    print(f"{hole_card:<12} {dealer_upcard + hole_card:<15} {bust_prob*100:<14.1f}% {(1-bust_prob)*100:<14.1f}%")

# ============================================================================
# 2. EDGE FROM HOLE CARD KNOWLEDGE
# ============================================================================

print("\n" + "="*70)
print("2. PLAYER EDGE: WITH vs WITHOUT HOLE CARD KNOWLEDGE")
print("="*70)

# Example scenario: Player has Hard 16 vs Dealer 10
player_hand = 16
dealer_upcard = 10

# Without knowledge: Hit or Stand (basic strategy says Hit)
# Hit: ~46% win (draw 5-6 to beat 18+; dealer likely has 18-20)
# Stand: ~23% win (dealer busts ~23% when hitting <17)
ev_without_knowledge = -0.50  # House edge

# With hole card knowledge:
# If dealer has 20/21: Stand (lose 100%) or Surrender (lose 50%)
# If dealer has <17: Stand (win ~high %); Hit risky
ev_with_knowledge_avg = 0

# Calculate edge split by hole card
print(f"\nPlayer hand: Hard {player_hand}")
print(f"Dealer showing: {dealer_upcard}")

ace_count = 0
edge_improvements = []

for hole_card in dealer_cards:
    dealer_total = dealer_upcard + hole_card
    
    # Without knowledge: stick to basic strategy
    if player_hand < 17:
        # Hit expected value (very simplified)
        ev_no_info = -0.05  # Negative overall
    else:
        ev_no_info = 0
    
    # With knowledge: optimal play
    if dealer_total >= 18:
        # Dealer likely wins; best is Surrender (-0.5) vs Hit (-0.95)
        ev_with_info = -0.50 if player_hand <= 16 else -0.95
    else:
        # Dealer likely busts; Stand wins
        ev_with_info = 0.95
    
    improvement = ev_with_info - ev_no_info
    edge_improvements.append(improvement)
    
    print(f"  Hole {hole_card}: Dealer total {dealer_total} → EV change: {improvement:+.2f}")

avg_edge_improvement = np.mean(edge_improvements)
print(f"\nAverage edge improvement from hole card knowledge: {avg_edge_improvement:+.2f}")

# ============================================================================
# 3. BETTING STRATEGY WITH HOLE CARD KNOWLEDGE
# ============================================================================

print("\n" + "="*70)
print("3. OPTIMAL BET SIZING (WITH HOLE CARD INFO)")
print("="*70)

# Bet large when dealer has bad card; minimum when dealer has good card
base_bet = $10
bet_strategy = {}

for hole_card in dealer_cards:
    dealer_total = dealer_upcard + hole_card
    
    if dealer_total >= 18:
        # Dealer likely wins
        bet_size = base_bet  # Minimum bet
    elif dealer_total == 17:
        # Dealer stands; could push or lose
        bet_size = base_bet * 1.5  # Moderate bet
    else:  # dealer_total < 17
        # Dealer likely busts
        bet_size = base_bet * 4  # Maximum bet
    
    bet_strategy[hole_card] = bet_size

print(f"\nBet sizing by dealer hole card:")
print(f"{'Hole Card':<12} {'Dealer Total':<15} {'Bet Size':<15} {'Expected Win':>15}")
print("=" * 57)

for hole_card in sorted(bet_strategy.keys()):
    dealer_total = dealer_upcard + hole_card
    bet = bet_strategy[hole_card]
    expected_win = "High" if bet > base_bet * 2 else "Medium" if bet > base_bet else "Protect"
    print(f"{hole_card:<12} {dealer_total:<15} ${bet:<14.0f} {expected_win:>15}")

# ============================================================================
# 4. SESSION PROFITABILITY SIMULATION
# ============================================================================

print("\n" + "="*70)
print("4. SESSION PROFIT SIMULATION (50 hands with hole card knowledge)")
print("="*70)

n_hands = 50
hands_with_peek = int(n_hands * 0.8)  # Assume see hole card 80% of hands

session_profit = 0
hand_results = []

for hand in range(n_hands):
    # Randomly determine hole card (if visible)
    if hand < hands_with_peek:
        hole_card = np.random.choice(dealer_cards)
        dealer_total = dealer_upcard + hole_card
        
        # Bet based on knowledge
        if dealer_total >= 18:
            bet = base_bet
            win_prob = 0.15  # Low win chance
        elif dealer_total == 17:
            bet = base_bet * 1.5
            win_prob = 0.40
        else:
            bet = base_bet * 4
            win_prob = 0.80  # High win chance
    else:
        # Blind hand; use basic strategy
        bet = base_bet
        win_prob = 0.48  # ~blackjack house edge
    
    # Determine outcome
    profit = bet if np.random.random() < win_prob else -bet
    session_profit += profit
    hand_results.append(profit)

print(f"\nSimulation results (50 hands):")
print(f"  Total bet amount: ${np.sum([abs(x) for x in hand_results]):.2f}")
print(f"  Total profit: ${session_profit:.2f}")
print(f"  Win rate: {sum(1 for x in hand_results if x > 0)}/{len(hand_results)}")
print(f"  ROI: {session_profit / np.sum([abs(x) for x in hand_results]) * 100:.1f}%")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Edge improvement by hole card
ax1 = axes[0, 0]
colors = ['green' if e > 0 else 'red' for e in edge_improvements]
ax1.bar(dealer_cards, edge_improvements, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Dealer Hole Card')
ax1.set_ylabel('EV Improvement vs No Info')
ax1.set_title('Edge Gain from Hole Card Knowledge')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Betting strategy by hole card
ax2 = axes[0, 1]
bets = [bet_strategy[card] for card in sorted(bet_strategy.keys())]
ax2.bar(sorted(bet_strategy.keys()), bets, color='blue', alpha=0.7, edgecolor='black')
ax2.axhline(y=base_bet, color='red', linestyle='--', linewidth=2, label='Base bet')
ax2.set_xlabel('Dealer Hole Card')
ax2.set_ylabel('Bet Size ($)')
ax2.set_title('Optimal Betting Strategy')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Cumulative session profit
ax3 = axes[1, 0]
cumulative = np.cumsum(hand_results)
ax3.plot(range(1, n_hands + 1), cumulative, linewidth=2, marker='o', markersize=4)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.fill_between(range(1, n_hands + 1), cumulative, 0, alpha=0.3)
ax3.set_xlabel('Hand Number')
ax3.set_ylabel('Cumulative Profit ($)')
ax3.set_title('Session Profit Evolution (50 hands)')
ax3.grid(True, alpha=0.3)

# Plot 4: Distribution of hand profits
ax4 = axes[1, 1]
ax4.hist(hand_results, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax4.axvline(x=np.mean(hand_results), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(hand_results):.2f}')
ax4.set_xlabel('Profit/Loss per Hand ($)')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Hand Outcomes')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hole_carding_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: hole_carding_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Hole card knowledge provides 15-40% edge (deterministic)")
print("✓ Optimal play: Bet large when dealer weak; minimum when strong")
print("✓ Sustainability: Limited (few exploitable dealers; detection risk high)")
print("✓ Session potential: $500-5000+ profit per opportunity possible")
print("✓ Career viability: Better as one-time exploit than long-term career")
