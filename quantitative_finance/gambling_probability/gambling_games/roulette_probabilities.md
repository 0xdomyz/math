# Roulette Probabilities

## 1. Concept Skeleton
**Definition:** Analysis of outcome likelihoods in roulette; wheel mechanics, bet types, and mathematical edge  
**Purpose:** Calculate exact probabilities for all roulette bets, compare bet efficiency, understand house advantage  
**Prerequisites:** Probability, expected value, basic statistics, counting

## 2. Comparative Framing
| Bet Type | Numbers | Probability | Payout | House Edge |
|----------|---------|------------|--------|-----------|
| Straight | 1 | 1/37 (2.70%) | 35:1 | 2.70% |
| Split | 2 | 2/37 (5.41%) | 17:1 | 2.70% |
| Street | 3 | 3/37 (8.11%) | 11:1 | 2.70% |
| Dozen | 12 | 12/37 (32.43%) | 2:1 | 2.70% |
| Red/Black | 18 | 18/37 (48.65%) | 1:1 | 2.70% |

## 3. Examples + Counterexamples

**Simple Example:**  
Red bet: Win \$100, payout 1:1 for \$100 bet. True probability 18/37; EV = (18/37)(100) - (19/37)(100) = -\$2.70 per \$100.

**Failure Case:**  
"Red hasn't hit in 10 spins, so it's due." Each spin independent; past doesn't affect future (representativeness heuristic).

**Edge Case:**  
European vs American roulette: European has 37 numbers (0-36); American has 38 (0, 00, 1-36). American edge = 5.26% on even-money bets.

## 4. Layer Breakdown
```
Roulette Probability Framework:
├─ Wheel Mechanics:
│   ├─ European: 37 pockets (0-36), house edge = 2.70%
│   ├─ American: 38 pockets (0, 00, 1-36), house edge = 5.26%
│   ├─ Numbers: Red (18), Black (18), Green (1 or 2)
│   ├─ Spin outcome: Uniformly random if fair wheel
│   └─ True probability: 1/37 per number (European)
├─ Bet Type Probabilities:
│   ├─ Straight (single number): 1/37 = 2.70%, pays 35:1
│   │   EV per $1: (1/37)(35) + (36/37)(-1) = -$0.027
│   ├─ Split (two adjacent numbers): 2/37 = 5.41%, pays 17:1
│   │   EV per $1: (2/37)(17) + (35/37)(-1) = -$0.027
│   ├─ Street (three numbers): 3/37 = 8.11%, pays 11:1
│   ├─ Corner (four numbers): 4/37 = 10.81%, pays 8:1
│   ├─ Six line (six numbers): 6/37 = 16.22%, pays 5:1
│   ├─ Red/Black (18 numbers): 18/37 = 48.65%, pays 1:1
│   │   EV per $1: (18/37)(1) + (19/37)(-1) = -$0.027
│   ├─ Odd/Even: 18/37 = 48.65%, pays 1:1
│   ├─ High/Low (1-18 or 19-36): 18/37 = 48.65%, pays 1:1
│   ├─ Dozens (1-12, 13-24, 25-36): 12/37 = 32.43%, pays 2:1
│   └─ Columns: 12/37 = 32.43%, pays 2:1
├─ House Edge Calculation:
│   ├─ Formula: (true payout + 1 - probability × payout) / original bet
│   ├─ All bets have same edge on single-zero: 2.70%
│   ├─ American 0/00 bet: 5.26% (worst bet)
│   ├─ Edge = (payout - true payout) / true payout
│   └─ Conversion: Edge % ≈ 2.70% (European), 5.26% (American)
├─ Expected Value by Bet:
│   ├─ For any €1 bet over infinite spins: -€0.027 (European)
│   ├─ Cumulative EV = num_bets × (-€0.027) per €1
│   ├─ Longer play → larger absolute loss
│   └─ "Safer" bets still have same house edge
├─ Clustering & Streaks:
│   ├─ Probability of seeing 5 reds in row: (18/37)^5 = 1.28%
│   ├─ Happens about 1 in 78 sets of 5 spins
│   ├─ Not evidence of bias; within normal variance
│   ├─ Gambler's fallacy: Mistaking long-run distribution for short-term guarantee
│   └─ Hot hand fallacy: Betting on "hot" numbers (representativeness)
└─ Practical Implications:
    ├─ No strategy beats 2.70% edge (European)
    ├─ Martingale, progression systems still lose in expectation
    ├─ Table limits prevent infinite doubling strategies
    ├─ Bankroll depletion timeline: 1000 spins, 2.70% edge → ~27 unit loss
    └─ Only edge comes from wheel bias (rare) or biased dealing (very rare)
```

## 5. Mini-Project
Simulate roulette spins and verify house edge:
```python
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
```

## 6. Challenge Round
When does roulette analysis break down?
- Biased physical wheels (rare, detected quickly)
- Dealer bias (impossible if ball is randomized)
- Automated betting (same edge applies)
- Table limits (prevent infinite progression recovery)
- Betting systems (cannot overcome constant negative EV)

## 7. Key References
- [Wikipedia: Roulette](https://en.wikipedia.org/wiki/Roulette)
- [Roulette House Edge](https://www.investopedia.com/terms/h/house-edge.asp)
- [Probability of Roulette Outcomes](https://www.calculatorsoup.com/calculators/games/roulette.php)

---
**Status:** Foundational casino game analysis | **Complements:** House Edge, Expected Value, Martingale Fallacy
