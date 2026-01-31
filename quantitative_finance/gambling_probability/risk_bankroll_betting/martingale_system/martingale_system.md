# Martingale System

## 1. Concept Skeleton
**Definition:** Betting strategy doubling wager after each loss, attempting to recover losses with single win at doubled stake  
**Purpose:** Recover losses quickly; mathematical illusion of guaranteed win  
**Prerequisites:** Betting fundamentals, probability, expected value, bankroll limits

## 2. Comparative Framing
| Strategy | Martingale | Anti-Martingale | Kelly | Fixed Bet |
|----------|-----------|-----------------|--------|-----------|
| **After Loss** | Double bet | Decrease bet | Adjust by edge | No change |
| **After Win** | Reset to base | Increase bet | Adjust by edge | No change |
| **Goal** | Recover loss | Compound wins | Optimal growth | Steady play |
| **Outcome** | Ruin probable | Grows on streak | Balanced | Predictable |

## 3. Examples + Counterexamples

**Simple Example:**  
Roulette red/black (50% each, ignore house edge). Bet \$1, lose. Bet \$2, lose. Bet \$4, win → recover losses + \$1 profit.

**Failure Case:**  
Consecutive losses exceed bankroll. After 10 losses: \$1 → \$2 → \$4 → ... → \$512. Need \$1023 total to win \$1.

**Edge Case:**  
House limits. Bet limit (e.g., \$10,000 max) stops doubling sequence, breaking strategy.

## 4. Layer Breakdown
```
Martingale Framework:
├─ Basic Mechanism:
│   ├─ Bet 1: $b (lose)
│   ├─ Bet 2: $2b (lose)
│   ├─ Bet 3: $4b (lose)
│   ├─ ...
│   └─ Bet n: $2^(n-1)b (WIN) → recover all + profit $b
├─ Mathematical Analysis:
│   ├─ Total wagered after n losses: $b(2^n - 1)
│   ├─ Profit if win at bet n: $b(2^(n-1))
│   ├─ Net result if win: $b (guaranteed)
│   ├─ BUT: Eventual loss sequence → total loss $b(2^m - 1)
│   └─ Expected value: E[result] = -house_edge × expected_total_wagered
├─ Key Assumptions (usually violated):
│   ├─ Infinite bankroll (impossible)
│   ├─ No bet limits (casinos enforce limits)
│   ├─ Fair game (casinos have house edge)
│   ├─ No correlations (independence)
│   └─ Immediate access to capital (liquidity)
├─ Variants:
│   ├─ Classic Martingale: Double bet after loss
│   ├─ Grand Martingale: Double + add base unit after loss
│   ├─ Labouchère (cancellation): Cross out on win, add on loss
│   ├─ D'Alembert: +1 unit on loss, -1 unit on win (less aggressive)
│   └─ Fibonacci: Bet sequence follows Fibonacci numbers
├─ The Illusion:
│   ├─ Seems to guarantee profit: "Eventually I'll win"
│   ├─ Ignores: Gambler's Ruin (infinite sequence ends in loss)
│   ├─ Ignores: House edge (negative expectation compounds)
│   ├─ Ignores: Bankroll limits (finite resources)
│   └─ Reality: E[profit] = -(house_edge) × E[amount_wagered]
└─ Practical Failure Modes:
    ├─ Losing streak: 10 losses → $1023 wagered for $1 profit
    ├─ Bet limit hit: Can't double further, lose entire sequence
    ├─ Bankroll exhaustion: Run out of money mid-sequence
    ├─ Margin calls: Borrow money → compound losses
    └─ Psychological: Tilt, desperation increase actual losses
```

## 5. Mini-Project
Simulate Martingale and compare to alternatives:
```python
import numpy as np
import matplotlib.pyplot as plt

def martingale_strategy(initial_bet, max_bets, prob_win=0.5):
    """
    Simulate Martingale strategy
    Returns: (final_profit, max_bet_reached, bets_before_loss)
    """
    bankroll = 1000
    current_bet = initial_bet
    total_wagered = 0
    consecutive_losses = 0
    sequence_number = 0
    
    for bet_num in range(max_bets):
        # Check if can afford bet
        if bankroll < current_bet:
            return bankroll - 1000, consecutive_losses, sequence_number
        
        # Place bet
        total_wagered += current_bet
        
        # Determine outcome
        if np.random.random() < prob_win:
            # WIN: Reset to base bet
            bankroll += current_bet
            current_bet = initial_bet
            consecutive_losses = 0
            sequence_number += 1
        else:
            # LOSS: Double bet
            bankroll -= current_bet
            current_bet *= 2
            consecutive_losses += 1
        
        # Stop if bust
        if bankroll <= 0:
            return bankroll, consecutive_losses, sequence_number
    
    return bankroll - 1000, consecutive_losses, sequence_number

def fixed_bet_strategy(bet_size, num_bets, prob_win=0.5):
    """Simple fixed bet strategy"""
    profit = 0
    for _ in range(num_bets):
        if np.random.random() < prob_win:
            profit += bet_size
        else:
            profit -= bet_size
    return profit

# Example 1: Single sequence of Martingale
print("=== Martingale Sequence Example ===\n")

np.random.seed(42)
base_bet = 10
sequence_bets = [base_bet]
cumulative_loss = 0

print(f"Base bet: ${base_bet}")
print(f"Probability of win: 50%\n")
print(f"{'Bet':<8} {'Bet Size':<12} {'Outcome':<10} {'Profit/Loss':<15} {'Cumulative':<15}")
print("-" * 60)

for i in range(10):
    bet_size = base_bet * (2 ** i)
    
    # Simulate outcome
    if np.random.random() < 0.5:
        outcome = "WIN"
        pnl = bet_size
    else:
        outcome = "LOSS"
        pnl = -bet_size
    
    cumulative_loss += pnl
    
    if outcome == "WIN":
        print(f"{i+1:<8} ${bet_size:<11} {outcome:<10} +${bet_size:<14} ${cumulative_loss:<14}")
        break
    else:
        print(f"{i+1:<8} ${bet_size:<11} {outcome:<10} -${bet_size:<14} ${cumulative_loss:<14}")

print(f"\nNote: After loss sequence, net profit = ${-cumulative_loss:.0f}")
print(f"Total wagered to win: ${sum([base_bet * (2**j) for j in range(i+1)]):.0f}")

# Example 2: Multiple simulations
print("\n\n=== Martingale vs Fixed Bet (10,000 simulations) ===\n")

num_simulations = 10000
num_bets_per_session = 200
initial_bank = 1000
base_bet_martingale = 10

martingale_results = []
fixed_bet_results = []

for sim in range(num_simulations):
    # Martingale
    profit_m, _, _ = martingale_strategy(base_bet_martingale, num_bets_per_session, prob_win=0.5)
    martingale_results.append(profit_m)
    
    # Fixed bet
    profit_f = fixed_bet_strategy(base_bet_martingale, num_bets_per_session, prob_win=0.5)
    fixed_bet_results.append(profit_f)

martingale_array = np.array(martingale_results)
fixed_array = np.array(fixed_bet_results)

print(f"Strategy Comparison (50% win probability, 200 bets):\n")
print(f"{'Metric':<25} {'Martingale':<20} {'Fixed Bet':<20}")
print("-" * 65)
print(f"{'Mean Profit':<25} ${martingale_array.mean():<19,.0f} ${fixed_array.mean():<19,.0f}")
print(f"{'Median Profit':<25} ${np.median(martingale_array):<19,.0f} ${np.median(fixed_array):<19,.0f}")
print(f"{'Std Deviation':<25} ${martingale_array.std():<19,.0f} ${fixed_array.std():<19,.0f}")
print(f"{'Min (Best)':<25} ${martingale_array.min():<19,.0f} ${fixed_array.min():<19,.0f}")
print(f"{'Max (Worst)':<25} ${martingale_array.max():<19,.0f} ${fixed_array.max():<19,.0f}")
print(f"{'Win Rate (Profit > 0)':<25} {(martingale_array > 0).sum()/num_simulations:<20.1%} {(fixed_array > 0).sum()/num_simulations:<20.1%}")

# Example 3: House edge impact
print("\n\n=== Martingale with House Edge (e.g., Roulette 5.26%) ===\n")

house_edge_prob = 0.4737  # American roulette (18/38 chance of win)

martingale_with_edge = []
fixed_with_edge = []

for sim in range(num_simulations):
    profit_m, _, _ = martingale_strategy(base_bet_martingale, num_bets_per_session, prob_win=house_edge_prob)
    martingale_with_edge.append(profit_m)
    
    profit_f = fixed_bet_strategy(base_bet_martingale, num_bets_per_session, prob_win=house_edge_prob)
    fixed_with_edge.append(profit_f)

martingale_edge_array = np.array(martingale_with_edge)
fixed_edge_array = np.array(fixed_with_edge)

print(f"With 5.26% house edge (American Roulette):\n")
print(f"Martingale expected loss: ${martingale_edge_array.mean():<,.0f}")
print(f"Fixed bet expected loss: ${fixed_edge_array.mean():<,.0f}")
print(f"Martingale ruin rate: {(martingale_edge_array < -initial_bank).sum()/num_simulations:.2%}")
print(f"Fixed bet ruin rate: {(fixed_edge_array < -1000).sum()/num_simulations:.2%}")

# Example 4: Losing streak probability
print("\n\n=== Probability of Long Losing Streaks ===\n")

print(f"{"Consecutive Losses":<25} {"Probability":<15} {"Bet Size ($)":<15}")
print("-" * 55)

for n_losses in [5, 10, 15, 20]:
    prob_streak = (0.5) ** n_losses
    bet_size = base_bet_martingale * (2 ** n_losses)
    print(f"{n_losses:<25} {prob_streak:<15.6f} (1/{1/prob_streak:.0f}) ${bet_size:<14,}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Fair game comparison (50-50)
axes[0, 0].hist(martingale_array, bins=50, alpha=0.5, label='Martingale', color='red')
axes[0, 0].hist(fixed_array, bins=50, alpha=0.5, label='Fixed Bet', color='green')
axes[0, 0].axvline(martingale_array.mean(), color='darkred', linestyle='--', linewidth=2)
axes[0, 0].axvline(fixed_array.mean(), color='darkgreen', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Final Profit ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Fair Game (50% win): Final Profit Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: House edge impact
axes[0, 1].hist(martingale_edge_array, bins=50, alpha=0.5, label='Martingale', color='red')
axes[0, 1].hist(fixed_edge_array, bins=50, alpha=0.5, label='Fixed Bet', color='green')
axes[0, 1].axvline(martingale_edge_array.mean(), color='darkred', linestyle='--', linewidth=2)
axes[0, 1].axvline(fixed_edge_array.mean(), color='darkgreen', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Final Profit ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('With House Edge (5.26%): Final Profit Distribution')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Losing streak exponential growth
streak_lengths = np.arange(1, 21)
bet_multiplier = 2 ** streak_lengths
total_wagered = (2 ** streak_lengths - 1) * base_bet_martingale

axes[1, 0].plot(streak_lengths, total_wagered, 'o-', linewidth=2, color='darkred', markersize=6)
axes[1, 0].fill_between(streak_lengths, total_wagered, alpha=0.3, color='red')
axes[1, 0].set_xlabel('Consecutive Losses')
axes[1, 0].set_ylabel('Total Amount Wagered ($)')
axes[1, 0].set_title('Capital Required by Losing Streak')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Probability vs bet size
max_losses = np.arange(1, 25)
probabilities = 0.5 ** max_losses
bet_sizes = base_bet_martingale * (2 ** max_losses)

axes[1, 1].semilogy(max_losses, probabilities, 'o-', linewidth=2, color='darkblue', label='Streak Probability')
axes[1, 1].set_xlabel('Consecutive Losses to Sustain')
axes[1, 1].set_ylabel('Probability (log scale)')
axes[1, 1].set_title('Risk of Long Losing Streaks')
axes[1, 1].grid(alpha=0.3)
for i in [5, 10, 15, 20]:
    prob = 0.5 ** i
    axes[1, 1].plot(i, prob, 'r*', markersize=15)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does Martingale fail catastrophically?
- Expected value unchanged (EV = -house_edge × total_wagered, always negative)
- Losing streak wiping bankroll before recovery
- Bet limits preventing doubling sequence
- Margin calls requiring funds not available
- Psychological desperation increasing irrational betting

## 7. Key References
- [Wikipedia: Martingale (Betting System)](https://en.wikipedia.org/wiki/Martingale_(betting_system))
- [Why Martingale Fails](https://www.investopedia.com/terms/m/martingalesystem.asp)
- [Historical Martingale Failures](https://en.wikipedia.org/wiki/Roulette#Betting_systems)

---
**Status:** Cautionary betting tale | **Complements:** Expected Value, Risk of Ruin, Bankroll Management
