# Poker Probabilities

## 1. Concept Skeleton
**Definition:** Mathematical analysis of poker hand rankings, probabilities, and equity; decision-making framework for betting  
**Purpose:** Calculate precise hand odds, compare hand strength, determine value betting decisions  
**Prerequisites:** Combinatorics, poker hand rankings, conditional probability, pot odds

## 2. Comparative Framing
| Hand Strength | Examples | Probability | Rank | Decision |
|---------------|----------|------------|------|----------|
| Premium hands | AA, KK, AK | 1.35% combined | Top 5% | Raise |
| Strong hands | QQ, TT, AJ | 4.5% | Top 15% | Raise/call |
| Medium hands | 99, AT, KQ | 12% | Top 40% | Call/fold |
| Weak hands | 72o, 63o | 50%+ | Bottom 50% | Fold |

## 3. Examples + Counterexamples

**Simple Example:**  
Flush draw (4 outs): 9 spades remain, 47 unknown. Probability: 9/47 = 19.1% on river alone.

**Failure Case:**  
Overestimating hand equity. Pair of 7s vs opponent's hidden hand is 45-55% at best; not guaranteed.

**Edge Case:**  
Reverse implied odds. Open-ended straight draw seems +EV, but losing to flush; true equity less favorable.

## 4. Layer Breakdown
```
Poker Probability Framework:
├─ Hand Rankings (highest to lowest):
│   ├─ Royal Flush: A-K-Q-J-10 same suit (1 in 649,740)
│   ├─ Straight Flush: Five consecutive, same suit (40 combinations)
│   ├─ Four of a Kind: Four same rank (624 combinations)
│   ├─ Full House: Three + Pair (3,744 combinations)
│   ├─ Flush: Five same suit (5,108 combinations)
│   ├─ Straight: Five consecutive ranks (10,200 combinations)
│   ├─ Three of a Kind: Three same rank (54,912 combinations)
│   ├─ Two Pair: Two different pairs (123,552 combinations)
│   ├─ One Pair: Two same rank (1,098,240 combinations)
│   └─ High Card: No combinations (1,302,540 combinations)
├─ Starting Hand Probabilities (Texas Hold'em):
│   ├─ Pocket Aces (AA): 0.45% (1 in 220)
│   ├─ Pocket Kings (KK): 0.45%
│   ├─ Any pocket pair: 5.88%
│   ├─ AK (same/different suits): 1.21%
│   ├─ AQ, AJ, AT: 3.63%
│   └─ Total "playable" premium: 5.9%
├─ Draw Probabilities (Post-flop):
│   ├─ Flush draw (4 out): 19.1% (turn), 32.0% (river)
│   ├─ Straight draw (8 out): 32.0% (turn), 51.0% (river)
│   ├─ Open-ended straight (8 out, both sides): Same as straight
│   ├─ Two overs (6 out): 24.5% (turn), 41.6% (river)
│   ├─ Set mining (small pair to trips): 12.5% on flop
│   └─ Backdoor draw (3 out): 4.3% (turn), 11.3% (river)
├─ Hand Equity (2-player scenarios):
│   ├─ AA vs KK: 82% vs 18% (pre-flop)
│   ├─ AA vs KQ: 67% vs 33%
│   ├─ Pocket pair vs two overs: 55% vs 45%
│   ├─ Flush draw vs pair: 45% vs 55% typically
│   ├─ Open straight draw vs high card: 50-50 or better
│   └─ Equity changes as cards revealed
├─ Pot Odds & Calling Decisions:
│   ├─ Pot odds = (amount to call) / (total pot + call + future bets)
│   ├─ Break-even: Equity ≥ Pot odds
│   ├─ Example: Call $10 into $30 pot → 3:1 pot odds
│   │   Need 25% equity to break-even; flush draw (32%) is profitable
│   ├─ Implied odds: Factor in future winnings if hand hits
│   └─ Reverse implied odds: Factor in future losses
├─ Implied Probability from Odds:
│   ├─ 2:1 odds → 33% implied probability
│   ├─ 3:1 odds → 25% implied probability
│   ├─ 4:1 odds → 20% implied probability
│   └─ Compare to actual hand equity for +EV decisions
└─ Advanced Concepts:
    ├─ Range vs. range equity (both hands distributions)
    ├─ Card removal (knowing one card affects others)
    ├─ Position impact on equity realization
    ├─ Multi-way pot equity (complex, usually diminishes)
    └─ Tournament dynamics (ICM chip equity differs from hand equity)
```

## 5. Mini-Project
Calculate poker hand probabilities and equity:
```python
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def calculate_hand_equity_simulation(hero_cards, villain_cards, board, num_simulations=100000):
    """
    Simulate poker equity between two hands
    Cards format: ['As', 'Kd', '2h'] etc
    """
    deck = [f"{r}{s}" for r in '23456789TJQKA' for s in 'hcds']
    
    # Remove known cards
    for card in hero_cards + villain_cards + board:
        deck.remove(card)
    
    hero_wins = 0
    villain_wins = 0
    ties = 0
    
    for _ in range(num_simulations):
        sample_deck = deck.copy()
        np.random.shuffle(sample_deck)
        
        # Complete boards (5 community cards)
        cards_needed = 5 - len(board)
        remaining_board = board + sample_deck[:cards_needed]
        remaining_deck = sample_deck[cards_needed:]
        
        # Evaluate hands (simplified: value by combinations)
        hero_best = evaluate_hand(hero_cards + remaining_board)
        villain_best = evaluate_hand(villain_cards + remaining_board)
        
        if hero_best > villain_best:
            hero_wins += 1
        elif villain_best > hero_best:
            villain_wins += 1
        else:
            ties += 1
    
    total = hero_wins + villain_wins + ties
    hero_equity = hero_wins / total if total > 0 else 0
    villain_equity = villain_wins / total if total > 0 else 0
    tie_equity = ties / total if total > 0 else 0
    
    return hero_equity, villain_equity, tie_equity

def evaluate_hand(seven_cards):
    """
    Simple hand evaluator - counts hand strength
    Returns numeric value for 5-card best hand
    Simplified: just checking pairs, straights, flushes
    """
    ranks = [card[0] for card in seven_cards]
    suits = [card[1] for card in seven_cards]
    
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    rank_nums = sorted([rank_values[r] for r in ranks])
    
    from collections import Counter
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    # Hand strength (simplified)
    is_flush = max(suit_counts.values()) >= 5
    is_straight = any(rank_nums[i:i+5] == list(range(rank_nums[i], rank_nums[i]+5)) for i in range(len(rank_nums)-4)) or set(rank_nums[-5:]) == {10, 11, 12, 13, 14}
    
    counts_sorted = sorted(rank_counts.values(), reverse=True)
    
    if is_straight and is_flush:
        return 800  # Straight flush
    elif counts_sorted == [4, 1, 1, 1]:
        return 700  # Four of a kind
    elif counts_sorted == [3, 2, 1, 1]:
        return 600  # Full house
    elif is_flush:
        return 500  # Flush
    elif is_straight:
        return 400  # Straight
    elif counts_sorted == [3, 1, 1, 1, 1]:
        return 300  # Three of a kind
    elif counts_sorted == [2, 2, 1, 1, 1]:
        return 200  # Two pair
    elif counts_sorted == [2, 1, 1, 1, 1, 1]:
        return 100  # One pair
    else:
        return rank_nums[-1]  # High card

# Example 1: Starting hand probabilities
print("=== Starting Hand Probabilities (Texas Hold'em) ===\n")

starting_hands = [
    ("AA (Pocket Aces)", 1/220),
    ("KK (Pocket Kings)", 1/220),
    ("QQ (Pocket Queens)", 1/220),
    ("Any Pocket Pair", 1/17),
    ("AK (same suit)", 1/331),
    ("AK (any)", 4/1326),
    ("AQ, AJ, AT (any)", 12/1326),
    ("Total premium hands", 2.4/100),
]

print(f"{'Hand':<30} {'Probability':<15} {'Frequency':<20}")
print("-" * 65)

for hand, prob in starting_hands:
    freq = 1 / prob if prob > 0 else float('inf')
    print(f"{hand:<30} {prob:<15.4f} 1 in {freq:<19.0f}")

# Example 2: Hand equity matchups (pre-flop)
print("\n\n=== Hand Equity Matchups (Pre-flop) ===\n")

matchups = [
    ("AA", "KK", "82% vs 18%"),
    ("AA", "AK", "87% vs 13%"),
    ("KK", "AQ", "69% vs 31%"),
    ("TT", "AK", "55% vs 45%"),
    ("AK", "QQ", "45% vs 55%"),
]

print(f"{'Hand 1':<15} {'Hand 2':<15} {'Approximate Equity':<30}")
print("-" * 60)

for h1, h2, equity in matchups:
    print(f"{h1:<15} {h2:<15} {equity:<30}")

# Example 3: Draw probabilities
print("\n\n=== Draw Probabilities (Post-Flop) ===\n")

# Assume flop seen, turn/river unknown
draws = [
    ("Flush draw (4 out)", 4),
    ("Straight draw (8 out)", 8),
    ("Open-ended straight", 8),
    ("Two pair to full house", 4),
    ("Pair to trips (set mining)", 2),
    ("Backdoor flush (3 out)", 3),
    ("Pair of overs (6 out)", 6),
]

print(f"{'Draw Type':<30} {'Outs':<10} {'Turn %':<12} {'River %':<12} {'Either %':<12}")
print("-" * 76)

for draw_name, outs in draws:
    turn_prob = (outs / 47) * 100
    river_prob = (outs / 46) * 100
    either_prob = (1 - ((47-outs)/47 * (46-outs)/46)) * 100
    print(f"{draw_name:<30} {outs:<10} {turn_prob:<12.1f} {river_prob:<12.1f} {either_prob:<12.1f}")

# Example 4: Pot odds vs equity
print("\n\n=== Pot Odds Analysis ===\n")

scenarios = [
    ("Flush draw", 9, 10, 30),
    ("Straight draw", 8, 10, 30),
    ("Pair of overs", 6, 5, 25),
    ("Backdoor draw", 3, 15, 30),
]

print(f"{'Draw':<20} {'Outs':<8} {'To Call':<10} {'Pot Size':<10} {'Pot Odds':<12} {'Win %':<10} {'Call?':<10}")
print("-" * 80)

for draw_name, outs, to_call, pot in scenarios:
    pot_odds = (pot + to_call) / to_call
    win_pct = (outs / 47) * 100  # River only
    breakeven = 100 / (pot_odds + 100) if pot_odds > 0 else 0
    should_call = "YES" if win_pct > breakeven else "NO"
    print(f"{draw_name:<20} {outs:<8} ${to_call:<9} ${pot:<9} {pot_odds:<12.2f}:1 {win_pct:<10.1f} {should_call:<10}")

# Example 5: Hand ranges vs equity
print("\n\n=== Range Analysis ===\n")

# Simple ranges
print("If opponent has top 10% of hands (AA, KK, QQ, AK):")
print("You have: AJ")
print("Approximate equity vs range: 35-40%\n")

print("If opponent has middle 30% of hands (pairs, broadway):")
print("You have: AJ")
print("Approximate equity vs range: 45-50%\n")

print("If opponent has wide range (50% of hands):")
print("You have: AJ")
print("Approximate equity vs range: 52-55%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Pocket pair probabilities
pair_probs = [1/220 * 100 for _ in range(13)]
pair_names = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']

axes[0, 0].bar(range(13), pair_probs, alpha=0.7, color='darkblue')
axes[0, 0].set_xticks(range(13))
axes[0, 0].set_xticklabels(pair_names, rotation=45)
axes[0, 0].set_ylabel('Probability (%)')
axes[0, 0].set_title('Pocket Pair Starting Probabilities')
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Draw probabilities turn vs river
draw_names = ['Flush\n(9 out)', 'Straight\n(8 out)', 'Pair\n(6 out)', 'Pair→TT\n(2 out)', 'Backdoor\n(3 out)']
outs = [9, 8, 6, 2, 3]
turn_probs = [(o / 47) * 100 for o in outs]
river_probs = [(o / 46) * 100 for o in outs]

x_pos = np.arange(len(draw_names))
axes[0, 1].bar(x_pos - 0.2, turn_probs, 0.4, label='Turn', alpha=0.7, color='green')
axes[0, 1].bar(x_pos + 0.2, river_probs, 0.4, label='River', alpha=0.7, color='orange')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(draw_names)
axes[0, 1].set_ylabel('Probability (%)')
axes[0, 1].set_title('Draw Probability Comparison')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Pot odds breakeven line
pot_odds_range = np.logspace(-0.5, 2, 50)
breakeven_equity = 100 / (pot_odds_range + 100) * 100

# Overlay draw equity
draw_options = [(9, 'Flush draw', 'green'), (8, 'Straight', 'blue'), (6, 'Overs', 'orange')]

axes[1, 0].plot(pot_odds_range, breakeven_equity, 'k-', linewidth=2, label='Break-even equity')

for outs, name, color in draw_options:
    equity = (outs / 47) * 100
    axes[1, 0].axhline(equity, color=color, linestyle='--', alpha=0.5, label=f'{name}: {equity:.0f}%')

axes[1, 0].fill_between(pot_odds_range, breakeven_equity, 50, alpha=0.2, color='green')
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('Pot Odds')
axes[1, 0].set_ylabel('Equity Required (%)')
axes[1, 0].set_title('Pot Odds vs Hand Equity')
axes[1, 0].set_ylim([0, 50])
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Hand strength cumulative distribution
hands_ranked = [
    ('AA', 99), ('KK', 98), ('AK', 95), ('QQ', 94), ('JJ', 92), ('TT', 88),
    ('AQ', 85), ('AJ', 82), ('99', 80), ('88', 77), ('77', 74), ('KQ', 72),
    ('AT', 70), ('KJ', 68), ('66', 65), ('A9', 62), ('KT', 60), ('QJ', 58),
    ('55', 55), ('A8', 52), ('JT', 50),
]

percentile = np.linspace(100, 0, len(hands_ranked))
strength = [h[1] for h in hands_ranked]

axes[1, 1].plot(percentile, strength, 'o-', linewidth=2, markersize=6)
axes[1, 1].fill_between(percentile, strength, alpha=0.2)
axes[1, 1].set_xlabel('Starting Hand Percentile')
axes[1, 1].set_ylabel('Approximate Strength (%)')
axes[1, 1].set_title('Hand Strength Distribution')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does poker probability analysis break down?
- Hidden information (don't know opponent's hand, only range)
- Player skill variations (decisions beyond probability)
- Bankroll impact on risk tolerance (fold margins change)
- Multi-way pots (equity distribution complex, blockers matter)
- Non-standard rules (antes, blind structure changes)

## 7. Key References
- [Wikipedia: Poker Hand Probabilities](https://en.wikipedia.org/wiki/Poker_probability)
- [Pot Odds Calculator](https://www.cardschat.com/poker-odds-calculator.php)
- [Poker Equity Reference](https://www.icmpoker.com/calculator/)

---
**Status:** Strategic game analysis | **Complements:** Combinatorics, Hand Rankings, Pot Odds
