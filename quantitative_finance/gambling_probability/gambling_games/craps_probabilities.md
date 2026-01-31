# Craps Probabilities

## 1. Concept Skeleton
**Definition:** Mathematical analysis of craps outcomes; dice mechanics, bet types, and probability calculations  
**Purpose:** Calculate exact probabilities for all craps bets, determine true odds vs payouts, identify best bets  
**Prerequisites:** Probability, two-die outcomes, conditional probability, house edge

## 2. Comparative Framing
| Bet Type | Probability | Payout | True Odds | House Edge |
|----------|------------|--------|-----------|-----------|
| Pass line | 49.29% | 1:1 | 251:244 | 1.41% |
| Don't pass | 50.70% | 1:1 | 244:251 | 1.36% |
| Come | 49.29% | 1:1 | 251:244 | 1.41% |
| Odds (Pass) | Varies | True | 0% | 0% |
| Field bet | 44.44% | 1:1 or 2:1 | Varies | 5.56% |

## 3. Examples + Counterexamples

**Simple Example:**  
Rolling 7 or 11 on come-out: 8/36 = 22.22%. Rolling 2, 3, 12 (craps): 4/36 = 11.11%.

**Failure Case:**  
"7 hasn't rolled in 20 rolls, so it's due." Each roll independent; streak doesn't change 7's probability.

**Edge Case:**  
Odds bets: True odds with no house edge, but require establishing point first. Maximize odds bets.

## 4. Layer Breakdown
```
Craps Probability Framework:
├─ Dice Mechanics:
│   ├─ Two dice, 36 total outcomes
│   ├─ Each outcome 1/36 probability (if fair)
│   ├─ Sum distribution: 7 most common (6 ways), 2 and 12 rarest (1 way each)
│   ├─ Probability of each sum:
│   │   2: 1/36 = 2.78%
│   │   3: 2/36 = 5.56%
│   │   4: 3/36 = 8.33%
│   │   5: 4/36 = 11.11%
│   │   6: 5/36 = 13.89%
│   │   7: 6/36 = 16.67% (most common)
│   │   8: 5/36 = 13.89%
│   │   9: 4/36 = 11.11%
│   │   10: 3/36 = 8.33%
│   │   11: 2/36 = 5.56%
│   │   12: 1/36 = 2.78%
│   └─ Standard deviation increases with spread
├─ Come-Out Roll Outcomes:
│   ├─ Natural (7 or 11): 8/36 = 22.22% → Pass wins, Don't pass loses
│   ├─ Craps (2, 3, 12): 4/36 = 11.11% → Pass loses, Don't pass varies
│   ├─ Point established (4, 5, 6, 8, 9, 10): 24/36 = 66.67% → Continue
│   ├─ Point probabilities:
│   │   4 or 10: 3/36 = 8.33%
│   │   5 or 9: 4/36 = 11.11%
│   │   6 or 8: 5/36 = 13.89%
│   └─ Combinations matter for multi-roll games
├─ Pass Line Mechanics:
│   ├─ Come-out: 7 or 11 wins immediately; 2, 3, 12 loses; 4-10 sets point
│   ├─ Point established: Roll point before 7 to win, 7 loses
│   ├─ Probability of winning given point 4: 3/(3+6) = 33.3%
│   ├─ Probability of winning given point 6: 5/(5+6) = 45.5%
│   ├─ Overall pass line: 49.29% win rate
│   └─ House edge: 1.41%
├─ Don't Pass Mechanics:
│   ├─ Come-out: 2 or 3 wins; 12 is push; 7 or 11 loses; 4-10 sets point
│   ├─ Point established: 7 wins before point repeats
│   ├─ Probability 44.44% (lower due to 12 push)
│   └─ House edge: 1.36%
├─ Odds Bets (True Probability Payouts):
│   ├─ No house edge; payout equals true probability
│   ├─ Point 4 or 10: 2:1 odds (2 ways to make 4, 6 ways to make 7)
│   ├─ Point 5 or 9: 3:2 odds (4 ways to make 5, 6 ways to make 7)
│   ├─ Point 6 or 8: 6:5 odds (5 ways to make 6, 6 ways to make 7)
│   ├─ Lay odds (don't pass): Reverse (bet more, win less)
│   └─ Recommended: Maximize odds bet size with bankroll
├─ Field Bets:
│   ├─ Covers: 2, 3, 4, 5, 9, 10, 11, 12 (16/36 = 44.44%)
│   ├─ Misses: 6, 7, 8 (20/36 = 55.56%)
│   ├─ Typical payouts: 2 and 12 pay 2:1 or 3:1, others 1:1
│   ├─ House edge: 5.56% or higher
│   └─ Avoid: Worse than pass/don't pass
├─ Come/Don't Come Bets:
│   ├─ Same as pass/don't pass but after point established
│   ├─ Can establish multiple come points
│   ├─ Same odds, same house edge as main line
│   └─ Variance higher (more bets in action)
└─ Practical Implications:
    ├─ Best bets: Pass/don't pass + odds (1.4% edge)
    ├─ Avoid: Field bets, proposition bets (high edge)
    ├─ Odds bets: Only true odds available, mandatory to maximize
    ├─ Multiple come bets: Increase action but variance too
    └─ Bankroll planning: Protect against point losses
```

## 5. Mini-Project
Simulate craps games and analyze bet performance:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def roll_dice():
    """Return sum of two dice"""
    return np.random.randint(1, 7) + np.random.randint(1, 7)

def simulate_craps_session(num_rolls=1000, bets=['pass'], odds_multiplier=1):
    """
    Simulate craps session
    bets: list of bet types
    odds_multiplier: how much to back pass bet with odds (1 = 1x odds)
    """
    bankroll = 1000
    point = 0
    
    for _ in range(num_rolls):
        roll = roll_dice()
        
        if point == 0:
            # Come-out roll
            if roll in [7, 11]:
                bankroll += 10  # Pass wins
            elif roll in [2, 3, 12]:
                bankroll -= 10  # Pass loses
            else:
                point = roll  # Establish point
                # Place odds bet
                if 'pass' in bets:
                    bankroll -= 10 * odds_multiplier
        else:
            # Point established
            if roll == point:
                # Point wins
                bankroll += 10  # Pass wins
                # Pay odds
                if roll in [4, 10]:
                    bankroll += 10 * odds_multiplier * 2  # 2:1 odds
                elif roll in [5, 9]:
                    bankroll += 10 * odds_multiplier * 1.5  # 3:2 odds
                elif roll in [6, 8]:
                    bankroll += 10 * odds_multiplier * 1.2  # 6:5 odds
                point = 0
            elif roll == 7:
                # 7 out
                bankroll -= 10  # Pass loses
                bankroll -= 10 * odds_multiplier  # Odds lost
                point = 0
    
    return bankroll

def calculate_craps_probabilities():
    """Calculate key craps probabilities"""
    outcomes = Counter()
    
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            sum_dice = die1 + die2
            outcomes[sum_dice] += 1
    
    return outcomes

# Example 1: Dice roll probabilities
print("=== Dice Roll Probabilities ===\n")

outcomes = calculate_craps_probabilities()

print(f"{'Sum':<6} {'Count':<8} {'Probability':<15} {'Interpretation':<30}")
print("-" * 60)

for s in range(2, 13):
    count = outcomes[s]
    prob = count / 36
    if s == 7:
        interp = "Most common (16.67%)"
    elif s in [2, 12]:
        interp = "Least common (2.78%)"
    else:
        interp = ""
    print(f"{s:<6} {count:<8} {prob:<15.4f} {interp:<30}")

# Example 2: Come-out roll outcomes
print("\n\n=== Come-Out Roll Outcomes ===\n")

natural = outcomes[7] + outcomes[11]
craps = outcomes[2] + outcomes[3] + outcomes[12]
point_est = sum(outcomes[i] for i in [4, 5, 6, 8, 9, 10])

print(f"Natural (7 or 11): {natural}/36 = {natural/36:.1%} → Pass wins")
print(f"Craps (2, 3, 12): {craps}/36 = {craps/36:.1%} → Pass loses")
print(f"Point (4-10): {point_est}/36 = {point_est/36:.1%} → Continue")

# Example 3: Pass line win probability by point
print("\n\n=== Pass Line Win Probability by Point ===\n")

points = [4, 5, 6, 8, 9, 10]
sevens = 6

print(f"{'Point':<10} {'Ways to Make':<18} {'Ways to Make 7':<18} {'Win %':<15}")
print("-" * 60)

for point in points:
    ways_to_make = outcomes[point]
    win_prob = ways_to_make / (ways_to_make + sevens)
    print(f"{point:<10} {ways_to_make:<18} {sevens:<18} {win_prob:<15.1%}")

# Example 4: Odds bet payouts
print("\n\n=== Odds Bet Payouts (True Probability) ===\n")

print(f"{'Point':<10} {'Odds':<15} {'Payout on $10':<15}")
print("-" * 40)

odds_dict = {4: 2, 5: 1.5, 6: 1.2, 8: 1.2, 9: 1.5, 10: 2}

for point in points:
    odds = odds_dict[point]
    payout = 10 * odds
    print(f"{point:<10} {odds:.1f}:1 {payout:<15.1f}")

# Example 5: Session simulation
print("\n\n=== Session Simulation Comparison ===\n")

np.random.seed(42)

strategies = [
    ("Pass line only", ['pass'], 0),
    ("Pass + 1x odds", ['pass'], 1),
    ("Pass + 2x odds", ['pass'], 2),
    ("Pass + 5x odds", ['pass'], 5),
]

results_summary = []

for strat_name, bets, odds_mult in strategies:
    results = []
    for _ in range(100):
        final_bank = simulate_craps_session(num_rolls=1000, bets=bets, odds_multiplier=odds_mult)
        results.append(final_bank)
    
    avg = np.mean(results)
    std = np.std(results)
    min_val = np.min(results)
    max_val = np.max(results)
    
    results_summary.append({
        'strategy': strat_name,
        'avg': avg,
        'std': std,
        'min': min_val,
        'max': max_val
    })

print(f"{'Strategy':<20} {'Avg Bankroll':<18} {'Std Dev':<15} {'Min':<12} {'Max':<12}")
print("-" * 77)

for r in results_summary:
    print(f"{r['strategy']:<20} ${r['avg']:<17,.0f} ${r['std']:<14,.0f} ${r['min']:<11,.0f} ${r['max']:<11,.0f}")

# Example 6: Field bet vs pass line
print("\n\n=== Field Bet vs Pass Line Over 100 Rolls ===\n")

field_covered = set([2, 3, 4, 5, 9, 10, 11, 12])
uncovered = set([6, 7, 8])

field_hits = sum(1 for s in range(2, 13) if s in field_covered)
field_losses = sum(1 for s in range(2, 13) if s in uncovered)

print(f"Field covered: {field_covered} ({field_hits}/36 = {field_hits/36:.1%})")
print(f"Field uncovered: {uncovered} ({field_losses}/36 = {field_losses/36:.1%})")
print(f"House edge on field: ~5.56%")
print(f"House edge on pass: ~1.41%")
print(f"\nRecommendation: Avoid field bet; use pass + odds")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Dice roll distribution
sums = list(range(2, 13))
counts = [outcomes[s] for s in sums]
colors_dist = ['red' if s == 7 else 'orange' if s in [2, 12] else 'blue' for s in sums]

axes[0, 0].bar(sums, counts, color=colors_dist, alpha=0.7)
axes[0, 0].set_xlabel('Sum')
axes[0, 0].set_ylabel('Frequency out of 36')
axes[0, 0].set_title('Dice Sum Distribution')
axes[0, 0].set_xticks(sums)
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Come-out roll outcomes
come_out = ['Natural\n(7,11)', 'Craps\n(2,3,12)', 'Point\n(4-10)']
come_out_counts = [natural, craps, point_est]
colors_come = ['green', 'red', 'blue']

axes[0, 1].bar(come_out, come_out_counts, color=colors_come, alpha=0.7)
axes[0, 1].set_ylabel('Frequency out of 36')
axes[0, 1].set_title('Come-Out Roll Outcomes')
axes[0, 1].set_ylim([0, 30])
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Pass line win rate by point
point_win_rates = [outcomes[p] / (outcomes[p] + 6) * 100 for p in points]

axes[1, 0].plot(points, point_win_rates, 'o-', linewidth=2, markersize=8, color='darkblue')
axes[1, 0].axhline(50, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Point Value')
axes[1, 0].set_ylabel('Pass Line Win Rate (%)')
axes[1, 0].set_title('Pass Line Win Probability by Point')
axes[1, 0].set_xticks(points)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Strategy comparison (bankroll after 1000 rolls)
strats = [r['strategy'] for r in results_summary]
avgs = [r['avg'] for r in results_summary]
stds = [r['std'] for r in results_summary]

x_pos = np.arange(len(strats))
axes[1, 1].bar(x_pos, avgs, yerr=stds, capsize=5, alpha=0.7, color=['red', 'orange', 'yellow', 'green'])
axes[1, 1].axhline(1000, color='black', linestyle='--', linewidth=2, label='Starting')
axes[1, 1].set_ylabel('Final Bankroll ($)')
axes[1, 1].set_title('Strategy Performance (1000 rolls, 100 sims)')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(strats, rotation=15, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does craps probability analysis break down?
- Biased dice detection (requires large sample, casino checks regularly)
- Dealer bias (controlled outcome difficult in craps)
- Bankroll inadequacy (swings can be large)
- Fatigue from extended play (concentration lapses)
- Betting proposition mistakes (sucker bets offer 10%+ edge)

## 7. Key References
- [Wikipedia: Craps](https://en.wikipedia.org/wiki/Craps)
- [Craps House Edge Analysis](https://www.casinogamespro.com/craps-odds.html)
- [Craps Odds Calculator](https://www.gamblingsites.org/casino/craps/odds/)

---
**Status:** Casino dice game analysis | **Complements:** Dice Probabilities, Odds Calculation, House Edge
