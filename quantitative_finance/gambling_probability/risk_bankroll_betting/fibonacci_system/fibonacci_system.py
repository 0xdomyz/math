"""
Extracted from: fibonacci_system.md
"""

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
