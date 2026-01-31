# Payout Ratios

## 1. Concept Skeleton
**Definition:** Ratio of winnings to bet amount, expressed as odds format; contracts between true odds and offered odds encode house edge  
**Purpose:** Standardize gambling payouts, compare fairness across betting formats, identify value bets  
**Prerequisites:** Basic probability, odds notation, expected value

## 2. Comparative Framing
| Format | Decimal Odds | Fractional Odds | American Odds | Payout Ratio |
|--------|-------------|-----------------|---------------|--------------|
| **Form** | 2.50 | 3/2 | -200 or +200 | 1:2 (win:bet) |
| **Meaning** | Return $2.50 per $1 | Win $3 per $2 bet | Favorite: -200 or Underdog: +200 | Win $1 per $2 bet |
| **Implied P** | 1/2.50 = 40% | 2/5 = 40% | 200/(200+100)=66.7% | Depends on true P |
| **Region** | Europe, Asia | UK, Ireland | USA | Casinos |

## 3. Examples + Counterexamples

**Simple Example:**  
Coin flip: true odds 1:1. Fair payout: \$1 return per \$1 bet (decimal 2.0). Casino pays 1.9 → player gets 5% less.

**Failure Case:**  
Confusing decimal odds with payout ratio. Decimal 2.0 = return \$2 total (including \$1 bet), not \$2 profit.

**Edge Case:**  
Negative American odds (favorite, e.g., -200): Must bet \$200 to win \$100. Positive odds (underdog, e.g., +200): Bet \$100 to win \$200.

## 4. Layer Breakdown
```
Payout Ratio Framework:
├─ Basic Concepts:
│   ├─ True Odds: Correct ratio based on true probability
│   ├─ Offered Odds: Payout casino actually gives
│   ├─ Overround: Total implied probabilities > 100% (house margin)
│   └─ Vig/Juice: Betting cost (typically 2-5% in sports)
├─ Odds Formats & Conversions:
│   ├─ Decimal → Fractional: (D - 1) = Numerator/Denominator
│   │   Example: 2.5 decimal = 1.5 fractional = 3/2
│   ├─ Decimal → American: If D > 2: +(D-1)×100; if D < 2: -100/(D-1)
│   │   Example: 2.5 decimal = +150 American
│   ├─ American → Decimal: If positive: 1+(A/100); if negative: 1+(100/|A|)
│   │   Example: -200 American = 1.50 decimal
│   └─ Probability implied: 1/Decimal or 100/(American+100)
├─ Fair vs Unfair Payouts:
│   ├─ Fair Payout: Return = Bet / True_Probability
│   ├─ Overround (overask): Implied P > true P (house advantage)
│   ├─ Underround: Implied P < true P (favor bettor, rare in casinos)
│   └─ Example: Roulette red 18/38 true; payout 1:1; fair would be 10:11
├─ Expected Value Calculation:
│   ├─ EV = (Win_Probability × Payout) - Lose_Probability
│   ├─ EV = Win_Prob × (Odds - 1) - Lose_Prob × 1
│   ├─ Positive EV: Bet has edge (rare in casinos)
│   └─ Negative EV: House has edge (typical)
└─ Reading the Market:
    ├─ Odds movement: Line shifts based on action/information
    ├─ Sharp action: Sophisticated bets move lines
    ├─ Closing line value: Return vs original posted odds
    └─ Arbitrage: Differences across sportsbooks create opportunities
```

## 5. Mini-Project
Compare payouts and identify value:
```python
import numpy as np
import matplotlib.pyplot as plt

# Odds conversion functions
def decimal_to_american(decimal):
    """Convert decimal to American odds"""
    if decimal > 2:
        return int((decimal - 1) * 100)
    else:
        return -int(100 / (decimal - 1))

def american_to_decimal(american):
    """Convert American to decimal odds"""
    if american > 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)

def decimal_to_fractional(decimal):
    """Convert decimal to fractional odds"""
    from fractions import Fraction
    frac = Fraction(decimal - 1).limit_denominator(100)
    return f"{frac.numerator}/{frac.denominator}"

def implied_probability(decimal_odds):
    """Calculate implied probability from decimal odds"""
    return 1 / decimal_odds

# Example 1: Odds conversion
print("=== Odds Conversion ===\n")

decimal_odds = [1.50, 2.00, 2.50, 3.50, 5.00]
print(f"{'Decimal':<10} {'American':<12} {'Fractional':<12} {'Implied P':<12}")
print("-" * 48)

for d in decimal_odds:
    american = decimal_to_american(d)
    fractional = decimal_to_fractional(d)
    impl_p = implied_probability(d)
    print(f"{d:<10.2f} {american:<12} {fractional:<12} {impl_p:<12.2%}")

# Example 2: Fair vs Casino Payouts
print("\n\n=== Fair vs Casino Payouts ===\n")

# Coin flip: true probability 50%
true_p = 0.5
fair_decimal = 1 / true_p
print(f"Coin flip (true P = {true_p:.1%}):")
print(f"  Fair decimal odds: {fair_decimal:.2f}")
print(f"  Casino decimal odds: 1.95")
print(f"  Fair EV (no edge): 0")
print(f"  Casino EV: {(0.5 * 1.95) - 0.5:.4f} per $1 bet (negative for player)")

# Roulette: betting on red
print(f"\nRoulette red (true P = {18/38:.4f}):")
true_odds_roulette = 1 / (18/38)
fair_decimal_roulette = true_odds_roulette
casino_decimal_roulette = 2.0  # Pays 1:1

print(f"  Fair decimal odds: {fair_decimal_roulette:.4f}")
print(f"  Casino decimal odds: {casino_decimal_roulette:.2f}")
player_ev_roulette = (18/38 * casino_decimal_roulette) - 1
print(f"  Player EV: {player_ev_roulette:.4f} per $1 bet")
house_edge_roulette = -player_ev_roulette * 100
print(f"  House edge: {house_edge_roulette:.2f}%")

# Example 3: Finding value in sports betting
print("\n\n=== Sports Betting Value ===\n")

# Example: Team A vs Team B
teams = ['Team A', 'Team B']
true_probs = [0.45, 0.55]  # Your assessment
offered_decimals = [2.20, 1.70]  # Sportsbook odds

print(f"{'Team':<10} {'True P':<10} {'Offered Odds':<15} {'Implied P':<12} {'EV/dollar':<12}")
print("-" * 60)

for team, true_p, offered_d in zip(teams, true_probs, offered_decimals):
    impl_p = implied_probability(offered_d)
    ev = true_p * offered_d - 1
    print(f"{team:<10} {true_p:<10.1%} {offered_d:<15.2f} {impl_p:<12.1%} {ev:+.4f}")

print("\n→ Team A has positive EV (value bet); implied P < true P")
print("→ Team B has negative EV (no value); implied P > true P")

# Example 4: Overround in multiple outcomes
print("\n\n=== Overround Detection ===\n")

# Three-way bet (win, draw, loss)
decimal_odds_3way = [2.50, 3.00, 2.30]
implied_probs = [1/d for d in decimal_odds_3way]
overround = sum(implied_probs)

print(f"Three outcomes with decimal odds: {decimal_odds_3way}")
print(f"Implied probabilities: {[f'{p:.1%}' for p in implied_probs]}")
print(f"Sum of implied probabilities: {overround:.1%}")
print(f"Overround: {(overround - 1) * 100:.2f}%")
print(f"House margin: ~{(overround - 1) * 100 / overround:.2f}% of handle")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Decimal to Probability
decimals = np.linspace(1.1, 10, 100)
probs = 1 / decimals

axes[0, 0].plot(decimals, probs * 100, linewidth=2, color='steelblue')
axes[0, 0].scatter([1.5, 2.0, 2.5, 3.5, 5.0], 
                   [1/d * 100 for d in [1.5, 2.0, 2.5, 3.5, 5.0]], 
                   color='red', s=100, zorder=5)
axes[0, 0].set_xlabel('Decimal Odds')
axes[0, 0].set_ylabel('Implied Probability (%)')
axes[0, 0].set_title('Decimal Odds to Probability')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Overround visualization
outcomes = ['Outcome 1\n(2.50)', 'Outcome 2\n(3.00)', 'Outcome 3\n(2.30)']
impl_probs_3way = [1/2.50, 1/3.00, 1/2.30]
colors_3way = ['lightgreen', 'lightblue', 'lightcoral']

axes[0, 1].bar(outcomes, [p*100 for p in impl_probs_3way], color=colors_3way, alpha=0.7)
axes[0, 1].axhline(100/3, color='gray', linestyle='--', linewidth=2, label='Fair (33.3%)')
axes[0, 1].set_ylabel('Implied Probability (%)')
axes[0, 1].set_title(f'Overround Example (Total: {overround*100:.1f}%)')
axes[0, 1].set_ylim([0, 50])
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: EV of different payouts
payouts = np.linspace(1, 5, 100)
true_p = 0.5
ev = true_p * payouts - 1

axes[1, 0].plot(payouts, ev, linewidth=2, color='darkred')
axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].axvline(2.0, color='green', linestyle='--', alpha=0.5, label='Fair (2.0)')
axes[1, 0].axvline(1.95, color='orange', linestyle='--', alpha=0.5, label='Casino (1.95)')
axes[1, 0].fill_between(payouts, ev, 0, where=(ev>0), alpha=0.2, color='green', label='Positive EV')
axes[1, 0].fill_between(payouts, ev, 0, where=(ev<=0), alpha=0.2, color='red', label='Negative EV')
axes[1, 0].set_xlabel('Payout Ratio (Decimal Odds)')
axes[1, 0].set_ylabel('EV per $1 bet')
axes[1, 0].set_title('Expected Value vs Payout (50% true probability)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Odds format comparison
formats = ['Decimal\n2.50', 'American\n+150', 'Fractional\n3/2']
returns = [2.5, 2.5, 2.5]  # Same odds, different formats
colors_format = ['steelblue', 'coral', 'lightgreen']

axes[1, 1].bar(formats, returns, color=colors_format, alpha=0.7)
axes[1, 1].set_ylabel('Return on $1 bet')
axes[1, 1].set_title('Same Odds, Different Formats')
axes[1, 1].set_ylim([0, 3])
for i, ret in enumerate(returns):
    axes[1, 1].text(i, ret + 0.05, f'${ret:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is payout ratio analysis insufficient?
- Bookmakers adjust odds based on liability, not just probability
- Line movements reflect smart money, not truth
- Liquidity varies; exotic bets have wider margins
- Correlation in multi-leg bets creates compounding errors
- Vigorish not constant across all outcomes (middled bets)

## 7. Key References
- [Wikipedia: Odds Formats](https://en.wikipedia.org/wiki/Odds)
- [Wizard of Odds: Payout and Odds Conversion](https://wizardofodds.com)
- [Overround Explained](https://en.wikipedia.org/wiki/Overround)

---
**Status:** Standardized payout expression | **Complements:** House Edge, Expected Value, Fair Games
