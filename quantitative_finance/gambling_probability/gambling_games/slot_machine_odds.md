# Slot Machine Odds

## 1. Concept Skeleton
**Definition:** Analysis of slot machine mechanics, probabilities, and return-to-player (RTP) calculations  
**Purpose:** Understand RNG mechanics, calculate expected losses, compare machine variance, identify slot design  
**Prerequisites:** Probability, combinations, expected value, RNG concepts

## 2. Comparative Framing
| Metric | Meaning | Range | Impact |
|--------|---------|-------|--------|
| RTP (Return to Player) | Percentage returned to players | 85-99% | Lower RTP = faster loss |
| Volatility | Variance of outcomes | Low, Med, High | High volatility = big swings |
| House Edge | Complement of RTP | 1-15% | Determines loss rate |
| Hit Frequency | % of spins that win | 20-50% | More frequent but smaller wins |
| Max Win | Largest single payout | 500x-10,000x bet | Attracts players |

## 3. Examples + Counterexamples

**Simple Example:**  
Slot with 96% RTP and $1 bet. Expected loss per spin: $0.04. After 1000 spins: -$40.

**Failure Case:**  
"I'm overdue for a big win." RNG has no memory; each spin independent. Streaks are random variance.

**Edge Case:**  
Cascading symbols: Winning symbols disappear, new ones fall. Might trigger mini-sequences. Still RNG-controlled, no true advantage.

## 4. Layer Breakdown
```
Slot Machine Probability Framework:
├─ RNG Mechanics:
│   ├─ Random Number Generator: Determines every spin outcome
│   ├─ Constantly running: Generates numbers even between spins
│   ├─ Seeded by time: Difficult to predict without internal clock access
│   ├─ Regulations: Tested by independent labs (eCOGRA, GLI)
│   ├─ Fair distribution: Numbers should be uniformly distributed
│   └─ Manipulation detection: Casinos audit RNG regularly
├─ Reel Mechanics:
│   ├─ Virtual reels: Modern slots use software, not physical reels
│   ├─ Weighting: Symbols can have different frequencies on virtual reel
│   ├─ Example: 22 stops on virtual reel, but symbol appears 3 times
│   │   Probability of symbol: 3/22 = 13.6%
│   ├─ Payline combinations: Number of ways to win
│   ├─ Bet levels: Multiplies payouts; affects RTP calculation
│   └─ Bonus features: Free spins, multipliers, special events
├─ RTP Calculation:
│   ├─ RTP = Total payouts / Total wagers over long run
│   ├─ Example: $1M wagered, $960K paid out → 96% RTP
│   ├─ Regulations: Casinos set RTP (85-99% typical)
│   ├─ Player-friendly ranges: 94-99% (better than 85%)
│   ├─ Progressive slots: RTP may exclude progressive jackpot portion
│   └─ Variation: RTP certified but short-term variance can be extreme
├─ House Edge Relationship:
│   ├─ House Edge = 100% - RTP
│   ├─ 96% RTP = 4% house edge
│   ├─ 90% RTP = 10% house edge
│   ├─ Expected loss per spin: (bet × house edge)
│   └─ Example: $1 bet, 4% edge → $0.04 expected loss
├─ Volatility Categories:
│   ├─ Low volatility: Frequent wins, smaller payouts
│   │   Hit frequency 40-50%, max win 200-500x bet
│   │   Suitable for: Extended play on limited budget
│   ├─ Medium volatility: Balanced wins and payouts
│   │   Hit frequency 25-40%, max win 500-2000x bet
│   │   Suitable for: Moderate players, mix of action
│   ├─ High volatility: Rare wins, large payouts
│   │   Hit frequency 10-25%, max win 5000-10000x bet
│   │   Suitable for: Risk takers, larger bankrolls
│   └─ Variance formula: Var = Σ(outcome - mean)² × probability
├─ Betting Strategy Ineffectiveness:
│   ├─ Changing bet size: Doesn't change RTP, just adjusts loss rate
│   │   Higher bet = faster expected loss
│   │   Lower bet = slower expected loss
│   ├─ Chasing losses: Increases expected total loss
│   ├─ "Hot/cold" machines: Fallacy; each spin independent
│   ├─ Bonus features: Pre-determined by RNG, not skill-dependent
│   └─ Timing: When you play doesn't affect outcome
├─ Progressive Jackpots:
│   ├─ Accumulates portion of each bet across machines/time
│   ├─ RTP may be 92% base + 2-3% to progressive pool
│   ├─ Odds of hitting progressive: Typically 0.001% - 0.01%
│   ├─ Contribution to pool: Incentivizes continued play
│   └─ Strategy: Only play progressive at high enough amount to justify
└─ Practical Implications:
    ├─ Slots are pure gambling: No skill, only luck
    ├─ RTP determines long-term loss: Non-negotiable by player
    ├─ Volatility affects short-term experience: Matters for budget
    ├─ Expected loss is certain: Play time × bet × house edge
    ├─ Breakeven is impossible: RTP < 100% guarantees negative EV
    └─ Bankroll depletion timeline: Bank / (bet × house edge × spins/min)
```

## 5. Mini-Project
Simulate slot machine spins and analyze outcomes:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class SlotMachine:
    def __init__(self, rtp=0.96, volatility='medium', num_paylines=20, bet_per_line=1):
        self.rtp = rtp
        self.house_edge = 1 - rtp
        self.volatility = volatility
        self.num_paylines = num_paylines
        self.bet_per_line = bet_per_line
        self.total_bet = num_paylines * bet_per_line
        
        # Define outcome distribution based on volatility
        if volatility == 'low':
            self.outcomes = self._low_volatility_outcomes()
        elif volatility == 'medium':
            self.outcomes = self._medium_volatility_outcomes()
        else:  # high
            self.outcomes = self._high_volatility_outcomes()
    
    def _low_volatility_outcomes(self):
        """Frequent small wins"""
        return {
            0: 0.50,      # Loss (50% of spins)
            1: 0.25,      # Win 1x
            2: 0.15,      # Win 2x
            5: 0.08,      # Win 5x
            10: 0.02,     # Win 10x
        }
    
    def _medium_volatility_outcomes(self):
        """Balanced"""
        return {
            0: 0.65,      # Loss (65%)
            1: 0.15,      # Win 1x
            2: 0.10,      # Win 2x
            5: 0.06,      # Win 5x
            20: 0.03,     # Win 20x
            100: 0.01,    # Win 100x
        }
    
    def _high_volatility_outcomes(self):
        """Rare big wins"""
        return {
            0: 0.80,      # Loss (80%)
            1: 0.10,      # Win 1x
            5: 0.06,      # Win 5x
            50: 0.03,     # Win 50x
            500: 0.01,    # Win 500x
        }
    
    def spin(self):
        """Return multiplier for this spin"""
        multiplier = np.random.choice(
            list(self.outcomes.keys()),
            p=list(self.outcomes.values())
        )
        return multiplier * self.total_bet
    
    def simulate_session(self, num_spins):
        """Run session of spins"""
        bankroll = 1000
        spins_data = []
        
        for _ in range(num_spins):
            payout = self.spin()
            bankroll += payout - self.total_bet
            spins_data.append(bankroll)
            
            if bankroll <= 0:
                break
        
        return np.array(spins_data)

# Example 1: Expected loss per spin
print("=== Expected Loss Analysis ===\n")

machines = [
    ("Low-end strip casino", 0.85),
    ("Mid-range casino", 0.92),
    ("Player-friendly casino", 0.96),
    ("Best online casino", 0.98),
]

bet_amount = 1.0

print(f"{'Casino Type':<30} {'RTP':<10} {'House Edge':<15} {'Loss per $1 bet':<20}")
print("-" * 75)

for name, rtp in machines:
    house_edge = 1 - rtp
    loss_per_bet = bet_amount * house_edge
    print(f"{name:<30} {rtp:<10.1%} {house_edge:<15.1%} ${loss_per_bet:<19.3f}")

# Example 2: Bankroll depletion timeline
print("\n\n=== Time to Bust (1000 spins/hour) ===\n")

scenarios = [
    ("Low volatility, 96% RTP, $1 bet", 0.96, 'low', 1.0),
    ("Medium volatility, 96% RTP, $1 bet", 0.96, 'medium', 1.0),
    ("High volatility, 96% RTP, $1 bet", 0.96, 'high', 1.0),
    ("Medium volatility, 92% RTP, $5 bet", 0.92, 'medium', 5.0),
]

print(f"{'Scenario':<50} {'Expected Minutes to Bust':<30}")
print("-" * 80)

for scenario_name, rtp, volatility, bet in scenarios:
    house_edge = 1 - rtp
    expected_loss_per_spin = bet * house_edge
    starting_bank = 1000
    spins_to_bust = starting_bank / expected_loss_per_spin if expected_loss_per_spin > 0 else float('inf')
    minutes_to_bust = spins_to_bust / 1000  # 1000 spins per hour
    
    if minutes_to_bust > 60:
        time_str = f"{minutes_to_bust/60:.1f} hours"
    else:
        time_str = f"{minutes_to_bust:.1f} minutes"
    
    print(f"{scenario_name:<50} {time_str:<30}")

# Example 3: Hit frequency
print("\n\n=== Hit Frequency Analysis ===\n")

np.random.seed(42)

volatilities = ['low', 'medium', 'high']
results_freq = {}

for vol in volatilities:
    machine = SlotMachine(rtp=0.96, volatility=vol)
    hits = 0
    losses = 0
    
    for _ in range(10000):
        payout = machine.spin()
        if payout > 0:
            hits += 1
        else:
            losses += 1
    
    hit_freq = hits / (hits + losses)
    results_freq[vol] = hit_freq

print(f"{'Volatility':<20} {'Hit Frequency':<15}")
print("-" * 35)

for vol in volatilities:
    print(f"{vol.capitalize():<20} {results_freq[vol]:<15.1%}")

# Example 4: Bankroll erosion over time
print("\n\n=== Session Simulation ===\n")

np.random.seed(42)

machine_low = SlotMachine(rtp=0.96, volatility='low')
machine_high = SlotMachine(rtp=0.96, volatility='high')

session_low = machine_low.simulate_session(1000)
session_high = machine_high.simulate_session(1000)

print(f"Low volatility, 96% RTP:")
print(f"  Starting: $1000")
print(f"  Final: ${session_low[-1]:.0f}")
print(f"  Loss: ${1000 - session_low[-1]:.0f}")
print(f"  Win rate: {np.sum(session_low > 1000) / len(session_low) * 100:.1f}%\n")

print(f"High volatility, 96% RTP:")
print(f"  Starting: $1000")
print(f"  Final: ${session_high[-1]:.0f}")
print(f"  Loss: ${1000 - session_high[-1]:.0f}")
print(f"  Win rate: {np.sum(session_high > 1000) / len(session_high) * 100:.1f}%")

# Example 5: Max win potential
print("\n\n=== Max Win Potential by Volatility ===\n")

print(f"{'Volatility':<20} {'Typical Max Win':<20} {'Odds':<15}")
print("-" * 55)

print(f"{'Low':<20} {'200-500x bet':<20} {'~1 in 2000':<15}")
print(f"{'Medium':<20} {'500-2000x bet':<20} {'~1 in 10000':<15}")
print(f"{'High':<20} {'5000-10000x bet':<20} {'~1 in 50000':<15}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RTP vs house edge impact
rtp_range = np.linspace(0.85, 0.99, 50)
expected_loss_per_1000 = 1000 * (1 - rtp_range)

axes[0, 0].plot(rtp_range * 100, expected_loss_per_1000, linewidth=2, color='darkred')
axes[0, 0].fill_between(rtp_range * 100, expected_loss_per_1000, alpha=0.2, color='red')
axes[0, 0].set_xlabel('RTP (%)')
axes[0, 0].set_ylabel('Expected Loss on $1000 Wagered ($)')
axes[0, 0].set_title('RTP Impact on Expected Loss')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axvline(92, color='orange', linestyle='--', alpha=0.5, label='92% RTP')
axes[0, 0].axvline(96, color='green', linestyle='--', alpha=0.5, label='96% RTP')
axes[0, 0].legend()

# Plot 2: Bankroll erosion paths
np.random.seed(42)

spin_range = np.arange(0, 1001)
colors_vol = {'low': 'green', 'medium': 'orange', 'high': 'red'}

for vol in ['low', 'medium', 'high']:
    machine = SlotMachine(rtp=0.96, volatility=vol)
    session = machine.simulate_session(1000)
    axes[0, 1].plot(spin_range[:len(session)], session, label=f'{vol.capitalize()} vol', 
                   color=colors_vol[vol], alpha=0.7, linewidth=1.5)

axes[0, 1].axhline(1000, color='black', linestyle='--', linewidth=2, label='Starting')
axes[0, 1].set_xlabel('Spin Number')
axes[0, 1].set_ylabel('Bankroll ($)')
axes[0, 1].set_title('Bankroll Erosion: 96% RTP, Different Volatilities')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Hit frequency distribution
vols = list(results_freq.keys())
freqs = [results_freq[v] * 100 for v in vols]

axes[1, 0].bar(vols, freqs, color=['green', 'orange', 'red'], alpha=0.7)
axes[1, 0].set_ylabel('Hit Frequency (%)')
axes[1, 0].set_title('Hit Frequency by Volatility (10k spins)')
axes[1, 0].set_ylim([0, 60])
axes[1, 0].grid(alpha=0.3, axis='y')

for i, (vol, freq) in enumerate(zip(vols, freqs)):
    axes[1, 0].text(i, freq + 1, f'{freq:.1f}%', ha='center', fontweight='bold')

# Plot 4: Payout distribution (medium volatility)
machine_med = SlotMachine(rtp=0.96, volatility='medium')
payouts = []

for _ in range(10000):
    payout = machine_med.spin()
    payouts.append(payout)

axes[1, 1].hist(payouts, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[1, 1].axvline(np.mean(payouts), color='red', linestyle='--', linewidth=2, label=f"Mean: ${np.mean(payouts):.2f}")
axes[1, 1].set_xlabel('Payout ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Payout Distribution (Medium Volatility, 10k spins)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does slot analysis break down?
- Regulatory compliance (RTP certified but varies by jurisdiction)
- Regional differences (Nevada vs Atlantic City vs online)
- Bonus round complexity (triggers not fully disclosed)
- Live updating (newer games may use skill-based elements)
- Psychological pricing (bet structures designed to confuse)

## 7. Key References
- [Wikipedia: Slot Machine](https://en.wikipedia.org/wiki/Slot_machine)
- [RTP Explanation](https://www.gamblersanonymous.org/ga/understanding-rtp)
- [Gaming Regulations & Testing](https://www.gaminglabsinternational.com/)

---
**Status:** Pure chance game analysis | **Complements:** RNG Verification, Expected Value, House Edge
