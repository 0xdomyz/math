# Bankroll Management

## 1. Concept Skeleton
**Definition:** Strategic allocation and protection of capital across bets to maximize longevity and profitability while minimizing catastrophic loss  
**Purpose:** Ensure sustainable play, adapt to variance, scale with success, prevent emotional decisions  
**Prerequisites:** Risk of ruin, Kelly criterion, expected value, variance

## 2. Comparative Framing
| Approach | Conservative | Balanced | Aggressive | Reckless |
|----------|-------------|----------|-----------|----------|
| **Bet Size** | <1% bankroll | 1-5% bankroll | 5-10% bankroll | >10% bankroll |
| **Stop Loss** | 10-20% loss limit | 25% loss limit | 50% loss limit | None |
| **Risk of Ruin** | <1% | 1-5% | 5-20% | >20% |
| **Recovery Time** | Long | Moderate | Short | Impossible |
| **Frequency** | Long-term grinders | Professional | Short-term | Gamblers |

## 3. Examples + Counterexamples

**Simple Example:**  
\$10,000 bankroll, 1% unit sizing = \$100 per bet. After loss, bankroll \$9,900 → next bet \$99 (dynamic).

**Failure Case:**  
Ignoring variance and betting fixed amounts. Downswing hits, can't adapt, forced to quit mid-sequence.

**Edge Case:**  
Winnings reinvestment. Grow bankroll to \$12,000 → adjust bet unit to \$120. Compounding success.

## 4. Layer Breakdown
```
Bankroll Management Framework:
├─ Core Principles:
│   ├─ Preservation: Never risk entire capital on single bet
│   ├─ Scalability: Bet size grows with bankroll, shrinks with losses
│   ├─ Sustainability: Last long enough for edge to materialize
│   ├─ Adaptation: Adjust to variance and changing conditions
│   └─ Discipline: Follow rules even when losing (hardest part)
├─ Unit-Based System:
│   ├─ 1 unit: Base bet amount (e.g., $100)
│   ├─ Stake per bet: 1-5 units typically
│   ├─ Dynamic: Recalculate unit based on current bankroll
│   │   Bankroll / Base_Units = New_Unit
│   │   Example: $10K / 100 = $100 per unit
│   └─ Flexibility: Change units as bankroll grows/shrinks
├─ Stop Loss & Win Goals:
│   ├─ Daily stop loss: Stop after X% down in session
│   │   Typical: 10-25% of daily bankroll
│   ├─ Monthly stop loss: Reassess if down X% in month
│   │   Typical: 20-50% of monthly allocation
│   ├─ Win goal: Stop after X% up (lock in gains)
│   │   Typical: 20-50% of session bankroll
│   └─ Purpose: Prevent emotional tilt, compounding losses
├─ Allocation Strategy:
│   ├─ Full bankroll: Total capital available for gambling
│   ├─ Monthly allocation: Spend X per month, not total
│   │   Preserves capital for future months
│   ├─ Session bankroll: Play with X, stop when depleted
│   │   Psychological: Accept loss boundary
│   ├─ Variance reserve: Keep 20-30% untouched for swings
│   │   Buffer for downside scenarios
│   └─ Reinvestment: Add profits back to grow bankroll
├─ Kelly Fraction Application:
│   ├─ Full Kelly: f* from Kelly criterion
│   ├─ Fractional Kelly: 25-50% of f* (safer)
│   │   Reduces variance, keeps profitability
│   ├─ Calculation: Bankroll × (Kelly %) = Bet size
│   │   Example: $10K × 2% Kelly = $200 per bet
│   └─ Adjustment: Reduce if underestimating risk
├─ Compounding & Growth:
│   ├─ Reinvest profits: Grow bankroll for larger bets
│   ├─ Exponential growth: Doubling time = ln(2) / edge
│   ├─ Diminishing variance: Larger bankroll → more secure
│   └─ Long-term: Month-to-month vs session-to-session
└─ Record Keeping:
    ├─ Track every bet: Amount, odds, outcome, P&L
    ├─ Session stats: Win rate, average win/loss, streaks
    ├─ Variance analysis: Compare actual vs expected outcomes
    ├─ Bankroll curve: Chart growth/decline over time
    └─ Identify leaks: Where are losses concentrated?
```

## 5. Mini-Project
Build bankroll management simulator:
```python
import numpy as np
import matplotlib.pyplot as plt

class BankrollManager:
    def __init__(self, initial_bankroll, bet_strategy, stop_loss_pct=0.25, win_goal_pct=0.25):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.bet_strategy = bet_strategy
        self.stop_loss_pct = stop_loss_pct
        self.win_goal_pct = win_goal_pct
        self.history = [initial_bankroll]
        self.num_bets = 0
        self.num_wins = 0
        self.session_bets = 0
        self.session_high = initial_bankroll
        
    def place_bet(self, prob_win, odds_ratio=1.0):
        """Place single bet and update bankroll"""
        if self.bankroll <= 0:
            return None
        
        # Determine bet size
        bet_size = self.bet_strategy(self.bankroll)
        
        # Check stop loss
        session_loss = self.session_high - self.bankroll
        if session_loss > self.session_high * self.stop_loss_pct:
            return "stop_loss"
        
        # Check win goal
        session_gain = self.bankroll - self.session_high
        if session_gain > self.session_high * self.win_goal_pct:
            return "win_goal"
        
        # Outcome
        if np.random.random() < prob_win:
            self.bankroll += bet_size * odds_ratio
            self.num_wins += 1
        else:
            self.bankroll -= bet_size
        
        self.num_bets += 1
        self.session_bets += 1
        self.history.append(self.bankroll)
        
        return self.bankroll
    
    def run_session(self, num_bets, prob_win):
        """Run single session"""
        for _ in range(num_bets):
            result = self.place_bet(prob_win)
            if result in ["stop_loss", "win_goal"]:
                return result
        return "completed"
    
    def reset_session(self):
        """Reset session stats, keep bankroll"""
        self.session_high = self.bankroll
        self.session_bets = 0

# Betting strategies
def fixed_unit_strategy(unit_size):
    return lambda bankroll: unit_size

def percentage_strategy(percentage):
    return lambda bankroll: bankroll * percentage

def kelly_strategy(prob_win, kelly_frac=1.0):
    def kelly_bet(bankroll):
        q = 1 - prob_win
        f = kelly_frac * (prob_win - q)  # Simplified for even odds
        return max(bankroll * f, 1)
    return kelly_bet

# Example 1: Compare strategies
print("=== Bankroll Management Strategy Comparison ===\n")

np.random.seed(42)
initial_bank = 10000
prob_win = 0.55  # 55% win rate (5% edge)
sessions = 50

# Strategy 1: Fixed $100 units
manager_fixed = BankrollManager(initial_bank, fixed_unit_strategy(100), 
                               stop_loss_pct=0.25, win_goal_pct=0.25)

# Strategy 2: 1% of bankroll
manager_percent = BankrollManager(initial_bank, percentage_strategy(0.01),
                                 stop_loss_pct=0.25, win_goal_pct=0.25)

# Strategy 3: 1/2 Kelly
manager_kelly = BankrollManager(initial_bank, kelly_strategy(prob_win, kelly_frac=0.5),
                               stop_loss_pct=0.25, win_goal_pct=0.25)

for session in range(sessions):
    for strategy in [manager_fixed, manager_percent, manager_kelly]:
        strategy.run_session(100, prob_win)
        strategy.reset_session()

print(f"After {sessions} sessions of 100 bets each:\n")
print(f"{'Strategy':<20} {'Final Bankroll':<20} {'Win Rate':<15} {'Total Bets':<15}")
print("-" * 70)
print(f"{'Fixed $100':<20} ${manager_fixed.bankroll:<19,.0f} {manager_fixed.num_wins/manager_fixed.num_bets:<15.1%} {manager_fixed.num_bets:<15}")
print(f"{'1% Bankroll':<20} ${manager_percent.bankroll:<19,.0f} {manager_percent.num_wins/manager_percent.num_bets:<15.1%} {manager_percent.num_bets:<15}")
print(f"{'1/2 Kelly':<20} ${manager_kelly.bankroll:<19,.0f} {manager_kelly.num_wins/manager_kelly.num_bets:<15.1%} {manager_kelly.num_bets:<15}")

# Example 2: Variance impact
print("\n\n=== Variance Impact on Bankroll ===\n")

# High variance scenario
num_simulations = 1000
final_banks_fixed = []
final_banks_percent = []

for sim in range(num_simulations):
    m_fixed = BankrollManager(10000, fixed_unit_strategy(100))
    m_percent = BankrollManager(10000, percentage_strategy(0.01))
    
    for _ in range(300):  # 300 bets
        m_fixed.place_bet(0.52)
        m_percent.place_bet(0.52)
    
    final_banks_fixed.append(m_fixed.bankroll)
    final_banks_percent.append(m_percent.bankroll)

print(f"After 300 bets per simulation ({num_simulations} simulations):\n")
print(f"{'Metric':<25} {'Fixed $100':<20} {'1% Bankroll':<20}")
print("-" * 65)
print(f"{'Mean Final Bankroll':<25} ${np.mean(final_banks_fixed):<19,.0f} ${np.mean(final_banks_percent):<19,.0f}")
print(f"{'Std Dev':<25} ${np.std(final_banks_fixed):<19,.0f} ${np.std(final_banks_percent):<19,.0f}")
print(f"{'Min (Worst Case)':<25} ${np.min(final_banks_fixed):<19,.0f} ${np.min(final_banks_percent):<19,.0f}")
print(f"{'Ruin Rate (Bust)':<25} {(np.array(final_banks_fixed)<=0).sum()/num_simulations:<20.2%} {(np.array(final_banks_percent)<=0).sum()/num_simulations:<20.2%}")

# Example 3: Reinvestment analysis
print("\n\n=== Reinvestment vs Fixed Capital ===\n")

def simulate_with_reinvestment(initial, years, edge_pct):
    """Simulate growing bankroll with reinvestment"""
    capital = initial
    for year in range(years):
        annual_growth = (capital * edge_pct) * (1 + np.random.normal(0, 0.2))
        capital += annual_growth
        if capital <= 0:
            return 0
    return capital

# Compare: reinvest vs fixed
initial = 10000
years = 5
edge = 0.05  # 5% annual edge

capital_growth = []
for sim in range(1000):
    final = simulate_with_reinvestment(initial, years, edge)
    capital_growth.append(final)

print(f"Starting capital: ${initial:,}")
print(f"Edge: {edge*100:.1f}% annual")
print(f"Time horizon: {years} years\n")
print(f"Average final capital: ${np.mean(capital_growth):,.0f}")
print(f"Median final capital: ${np.median(capital_growth):,.0f}")
print(f"Best case (90th %ile): ${np.percentile(capital_growth, 90):,.0f}")
print(f"Worst case (10th %ile): ${np.percentile(capital_growth, 10):,.0f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bankroll curves
axes[0, 0].plot(manager_fixed.history, label='Fixed $100', linewidth=2)
axes[0, 0].plot(manager_percent.history, label='1% Bankroll', linewidth=2)
axes[0, 0].plot(manager_kelly.history, label='1/2 Kelly', linewidth=2)
axes[0, 0].axhline(initial_bank, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Bet Number')
axes[0, 0].set_ylabel('Bankroll ($)')
axes[0, 0].set_title('Bankroll Curves: Strategy Comparison')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Distribution of final values
axes[0, 1].hist(final_banks_fixed, bins=50, alpha=0.5, label='Fixed $100', color='red')
axes[0, 1].hist(final_banks_percent, bins=50, alpha=0.5, label='1% Bankroll', color='green')
axes[0, 1].axvline(np.mean(final_banks_fixed), color='darkred', linestyle='--', linewidth=2)
axes[0, 1].axvline(np.mean(final_banks_percent), color='darkgreen', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Final Bankroll ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Final Bankroll Distribution (300 bets)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Long-term growth
years_range = np.arange(0, 6)
growth_paths = []

for sim in range(100):
    capital_path = [10000]
    capital = 10000
    for year in range(5):
        annual_growth = (capital * 0.05) * (1 + np.random.normal(0, 0.2))
        capital += annual_growth
        capital_path.append(capital)
    growth_paths.append(capital_path)

growth_paths = np.array(growth_paths)
axes[1, 0].plot(years_range, growth_paths.T, alpha=0.1, color='blue')
axes[1, 0].plot(years_range, np.mean(growth_paths, axis=0), color='darkblue', linewidth=2, label='Mean')
axes[1, 0].plot(years_range, np.percentile(growth_paths, 90, axis=0), color='green', linewidth=2, label='90th %ile')
axes[1, 0].plot(years_range, np.percentile(growth_paths, 10, axis=0), color='red', linewidth=2, label='10th %ile')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Capital ($)')
axes[1, 0].set_title('Long-Term Compounding (5% edge, 5 years)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Unit size vs bankroll
bankrolls_range = np.arange(5000, 50001, 5000)
fixed_units = [100] * len(bankrolls_range)
percent_units = bankrolls_range * 0.01

axes[1, 1].plot(bankrolls_range, fixed_units, 'o-', label='Fixed $100', linewidth=2, markersize=6)
axes[1, 1].plot(bankrolls_range, percent_units, 's-', label='1% Bankroll', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Bankroll ($)')
axes[1, 1].set_ylabel('Bet Size ($)')
axes[1, 1].set_title('Bet Sizing Strategies')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does bankroll management break down?
- Severe losing streak exceeds expected variance
- Estimation error in win rate (actual lower than estimated)
- Correlation in bets violates independence
- Changing game conditions (rules, opposition quality shifts)
- Emotional discipline failure (chasing losses, tilting)

## 7. Key References
- [Professional Gambling Bankroll](https://en.wikipedia.org/wiki/Bankroll_management)
- [Kelly Criterion Applied](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Stop Loss Strategies](https://www.investopedia.com/terms/s/stoploss.asp)

---
**Status:** Operational discipline framework | **Complements:** Kelly Criterion, Risk of Ruin, Expected Value
