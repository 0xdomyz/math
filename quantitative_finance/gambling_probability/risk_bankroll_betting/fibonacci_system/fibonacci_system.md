# Fibonacci System

## 1. Concept Skeleton
**Definition:** Progressive betting system using Fibonacci sequence (1,1,2,3,5,8,...) after losses, step back after a win  
**Purpose:** Smooth recovery versus Martingale while limiting bet escalation speed  
**Prerequisites:** Expected value, variance, bankroll constraints

## 2. Comparative Framing
| System | Progression Rule | Risk of Ruin | Recovery Speed | House Edge Impact |
|--------|------------------|--------------|----------------|-------------------|
| **Fibonacci** | Move forward after loss, back two after win | High | Moderate | None (unchanged EV) |
| **Martingale** | Double after loss | Very high | Fast | None |
| **Flat Betting** | Same stake each bet | Lower | None | None |
| **D’Alembert** | +1 unit after loss, -1 after win | Medium | Slow | None |

## 3. Examples + Counterexamples

**Example:**  
Lose 3 in a row: bets 1 → 1 → 2 → 3. Win → move back two steps to bet 1.

**Counterexample:**  
Roulette with limits: sequence hits table max before recovery, system fails.

**Edge Case:**  
Short sessions with early wins can appear “profitable,” but EV remains negative.

## 4. Layer Breakdown
```
Fibonacci System:
├─ Mechanics:
│  ├─ Sequence: 1,1,2,3,5,8,13,...
│  ├─ After loss: move one step forward
│  ├─ After win: move two steps back
│  └─ Goal: recover prior losses + 1 unit
├─ Risk Dynamics:
│  ├─ Slower escalation than Martingale
│  ├─ Still unbounded in long loss streaks
│  ├─ Bankroll limit defines maximum survivable streak
│  └─ Table max caps recovery
├─ Statistical Reality:
│  ├─ Expected value unchanged by staking system
│  ├─ Variance increased versus flat betting
│  ├─ Long streaks are inevitable
│  └─ Short-term wins are selection bias
├─ Practical Implications:
│  ├─ Appears safer, but edge remains negative
│  ├─ Requires large bankroll to survive bad runs
│  ├─ Works only in short samples by chance
│  └─ Psychological comfort vs mathematical edge
└─ Responsible Use:
   ├─ Pre-set loss limit
   ├─ Fixed session length
   ├─ Avoid escalation past bankroll plan
   └─ Treat as entertainment, not strategy
```

**Interaction:** Loss streak → Fibonacci escalation → temporary recovery if a win arrives before limits.

## 5. Mini-Project
Simulate Fibonacci betting versus flat betting:
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Non-obvious: use even-money game with small negative edge to show system fragility
p_win = 0.486  # roulette red (18/37)

fibo = [1, 1]
for _ in range(12):
    fibo.append(fibo[-1] + fibo[-2])


def simulate_fibonacci(n_rounds=200, bankroll=200):
    idx = 0
    profit = []
    b = bankroll
    for _ in range(n_rounds):
        bet = fibo[idx]
        if b < bet:
            break
        win = np.random.rand() < p_win
        if win:
            b += bet
            idx = max(idx - 2, 0)
            profit.append(b - bankroll)
        else:
            b -= bet
            idx = min(idx + 1, len(fibo) - 1)
            profit.append(b - bankroll)
    return profit, b


def simulate_flat(n_rounds=200, bankroll=200, bet=1):
    profit = []
    b = bankroll
    for _ in range(n_rounds):
        if b < bet:
            break
        win = np.random.rand() < p_win
        b += bet if win else -bet
        profit.append(b - bankroll)
    return profit, b

n_sessions = 300
final_fibo = []
final_flat = []

for _ in range(n_sessions):
    _, b1 = simulate_fibonacci()
    _, b2 = simulate_flat()
    final_fibo.append(b1)
    final_flat.append(b2)

sample_fibo, _ = simulate_fibonacci()
sample_flat, _ = simulate_flat()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(sample_fibo, label='Fibonacci')
axes[0, 0].plot(sample_flat, label='Flat', alpha=0.7)
axes[0, 0].set_title('Sample Session Profit')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Profit')
axes[0, 0].legend()

axes[0, 1].hist(final_fibo, bins=30, alpha=0.7, label='Fibonacci')
axes[0, 1].hist(final_flat, bins=30, alpha=0.7, label='Flat')
axes[0, 1].set_title('Final Bankroll Distribution')
axes[0, 1].set_xlabel('Bankroll')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

axes[1, 0].boxplot([final_fibo, final_flat], labels=['Fibonacci', 'Flat'])
axes[1, 0].set_title('Bankroll Dispersion')
axes[1, 0].set_ylabel('Bankroll')

ruin_fibo = np.mean(np.array(final_fibo) <= 0)
ruin_flat = np.mean(np.array(final_flat) <= 0)
axes[1, 1].bar(['Fibonacci', 'Flat'], [ruin_fibo, ruin_flat], color=['red','blue'])
axes[1, 1].set_title('Probability of Ruin')
axes[1, 1].set_ylabel('Probability')

plt.tight_layout()
plt.show()

print(f"Ruin probability (Fibonacci): {ruin_fibo:.2%}")
print(f"Ruin probability (Flat): {ruin_flat:.2%}")
```

## 6. Challenge Round
Why does Fibonacci betting feel “safer” than Martingale but still fail long-term?
- Loss recovery is slower, but still requires a win before bankroll exhaustion
- House edge persists and dominates in long sequences
- Table limits cap recovery before the sequence resets
- Variance increases relative to flat betting, raising ruin risk

## 7. Key References
- [Martingale (betting system)](https://en.wikipedia.org/wiki/Martingale_(betting_system))
- [Fibonacci number](https://en.wikipedia.org/wiki/Fibonacci_number)
- [Gambler’s Ruin](https://en.wikipedia.org/wiki/Gambler%27s_ruin)

---
**Status:** Progressive betting system | **Complements:** Martingale System, Bankroll Management, Risk of Ruin
