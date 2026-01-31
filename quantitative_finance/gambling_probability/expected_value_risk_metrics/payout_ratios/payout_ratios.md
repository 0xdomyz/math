# Payout Ratios: Return per Dollar Wagered

## 1. Concept Skeleton
**Definition:** Ratio of money returned to players vs total wagered; includes house edge (inverse relationship)  
**Purpose:** Compare game profitability; identify which games retain more player capital; standardize across different bet structures  
**Prerequisites:** Expected value, house edge, probability basics

## 2. Comparative Framing
| Concept | Payout Ratio | Return to Player (RTP) | House Edge | Payback % |
|---------|-------------|----------------------|-----------|----------|
| **Definition** | Total return / total wagered | Same as payout ratio | House profit % | Same as RTP |
| **Formula** | Payout $ / Wagered $ | (Wagered - House edge) / Wagered | 1 - Payout ratio | 100% - House edge % |
| **Example (EU Roulette)** | 0.973 (per $1 wagered) | 97.3% | 2.7% | 97.3% |
| **Range** | 0.85-0.99 (most games) | 85%-99% | 1%-15% | 85%-99% |
| **Better** | Higher (more to players) | Higher (fairer) | Lower (less advantage) | Higher (fairer) |
| **Application** | Quick game comparison | Long-term planning | Strategy evaluation | Regulatory compliance |

## 3. Examples + Counterexamples

**Simple Example:**  
Roulette: You wager $100 on red, win 50% of bets (hypothetically, fair). Payout ratio = $100 / $100 = 1.0. (Actually 0.973 with house edge)

**Failure Case:**  
Slots advertise 95% RTP; player believes each $1 wagered returns $0.95. Actually true only over millions of spins; short-term can be $0 or $5

**Edge Case:**  
Game A: 98% RTP, Game B: 97% RTP. Seem similar. But over 10,000 spins at $1 each: Game A loses $200, Game B loses $300. Cumulative difference matters.

## 4. Layer Breakdown
```
Payout Ratios Framework:
├─ I. DEFINITIONS & RELATIONSHIPS:
│   ├─ Payout Ratio:
│   │   ├─ Definition: Total money returned / total money wagered
│   │   ├─ Formula: Payout ratio = (Wagered - Loss) / Wagered
│   │   ├─ Example: $1000 wagered, $973 returned → ratio = 0.973
│   │   ├─ Interpretation: Keep 2.7¢ per $1 wagered (house)
│   │   └─ Range: 0.85 (slots) to 0.99+ (advantage play)
│   ├─ House Edge Relationship:
│   │   ├─ House edge = 1 - Payout ratio
│   │   ├─ Example: 0.973 payout → 0.027 = 2.7% house edge
│   │   ├─ Inverse relationship: Higher payout → lower edge
│   │   └─ Key insight: Payout ratio is player-centric version
│   ├─ Return to Player (RTP):
│   │   ├─ Definition: Same as payout ratio (alternative term)
│   │   ├─ Usage: Common in slots, video poker, online gambling
│   │   ├─ Regulatory: Often displayed for compliance
│   │   └─ Example: Slot machine labeled 95% RTP
│   ├─ Expected Value Link:
│   │   ├─ EV = (Payout ratio - 1) × bet
│   │   ├─ Example: 97.3% RTP, $1 bet → EV = -2.7¢
│   │   ├─ Interpretation: Average loss per bet determined by payout ratio
│   │   └─ Scaling: n bets → Total EV = n × (payout ratio - 1) × bet
│   └─ Variance Interaction:
│       ├─ Low payout ratio + high variance = fast capital drain
│       ├─ High payout ratio + high variance = slower drain
│       ├─ Low payout ratio + low variance = predictable slow drain
│       └─ Strategy: Maximize payout ratio and minimize variance
├─ II. PAYOUT RATIOS BY GAME:
│   ├─ Roulette:
│   │   ├─ European (single 0): 97.3% RTP, 2.7% HE
│   │   ├─ American (0, 00): 94.74% RTP, 5.26% HE
│   │   ├─ Implication: European superior; avoid American
│   │   └─ All bets same: Red/black, odd/even, single numbers all identical
│   ├─ Blackjack:
│   │   ├─ Basic strategy: 99.5% RTP, 0.5% HE
│   │   ├─ Average player: 95-98% RTP (poor play)
│   │   ├─ Card counter: 100%+ RTP, positive EV
│   │   └─ Implication: Skill dramatically affects payout ratio
│   ├─ Craps:
│   │   ├─ Pass/Don't Pass: 98.59% RTP, 1.41% HE
│   │   ├─ Come/Don't Come: Same as Pass/Don't Pass
│   │   ├─ Place bets: 95-97% RTP (worse)
│   │   ├─ Odds bets: 100% RTP (fair, zero house edge)
│   │   └─ Strategy: Stick to Pass/Don't Pass with Odds
│   ├─ Baccarat:
│   │   ├─ Banker bet: 98.94% RTP (better due to higher win rate)
│   │   ├─ Player bet: 98.76% RTP
│   │   ├─ Tie bet: 85.64% RTP (avoid!)
│   │   └─ Implication: Slight banker advantage worth considering
│   ├─ Slots:
│   │   ├─ Typical range: 92-96% RTP
│   │   ├─ Loose slots: 96-98% RTP
│   │   ├─ Tight slots: 85-90% RTP
│   │   ├─ Progressive slots: Often 90-94% (lower than regular)
│   │   └─ Regulation: Nevada average ≈ 95% (theoretical, actual varies)
│   ├─ Video Poker:
│   │   ├─ Good pay table: 99%+ RTP (possible +EV)
│   │   ├─ Average pay table: 95-97% RTP
│   │   ├─ Poor pay table: 91-93% RTP
│   │   ├─ Volatility: Extreme (rare jackpots)
│   │   └─ Strategy: ONLY play good pay tables; skill matters (hand selection)
│   ├─ Keno:
│   │   ├─ Typical RTP: 60-75% (worst casino game!)
│   │   ├─ Very tight game: 30-40% house edge
│   │   └─ Verdict: Avoid entirely; unbeatable
│   ├─ Poker (with rake):
│   │   ├─ Rake structure: Typically 5-10% of pot (or $1-5 per hand)
│   │   ├─ RTP for average player: 90-95% (rake eats profit)
│   │   ├─ Skilled player: Can overcome rake, achieve 100%+ return
│   │   └─ Strategy: Only play if skilled; rakeback helps
│   └─ Sports Betting:
│       ├─ Sportsbook margin (vig): 4-5% typical
│       ├─ RTP for casual bettor: 94-96%
│       ├─ RTP for sharp bettor: 101%+ (profitable)
│       └─ Strategy: Model-based approach essential to overcome vig
├─ III. CUMULATIVE IMPACT OF PAYOUT RATIOS:
│   ├─ Single Session Impact:
│   │   ├─ Game A: 99% RTP, 100 bets of $1 → avg loss $1
│   │   ├─ Game B: 95% RTP, 100 bets of $1 → avg loss $5
│   │   └─ Difference: $4 (4× worse in Game B)
│   ├─ Extended Play Impact:
│   │   ├─ 1,000 bets: A loses $10, B loses $50
│   │   ├─ 10,000 bets: A loses $100, B loses $500
│   │   ├─ 100,000 bets: A loses $1,000, B loses $5,000
│   │   └─ Insight: Small RTP difference compounds dramatically
│   ├─ Scaling with Bet Size:
│   │   ├─ RTP is percentage, so scales with bet
│   │   ├─ 1% RTP difference: $1 bet → $0.01/hand, $100 bet → $1/hand
│   │   ├─ Implication: High rollers lose (or win) more absolutely
│   │   └─ Strategy: Scale bet size proportional to acceptable loss
│   └─ Break-Even Calculation:
│       ├─ Hours needed to overcome downswing:
│       ├─ 99% RTP, $10/hour loss, need +$100 swing
│       ├─ Expected time: ~10 hours (if no variance)
│       ├─ Reality: Variance means unpredictable break-even
│       └─ Lesson: Don't chase to recover; accept losses and quit
├─ IV. USING PAYOUT RATIOS FOR DECISION-MAKING:
│   ├─ Game Selection:
│   │   ├─ Rule: Play highest RTP available
│   │   ├─ Ranking: Craps (98.6%) > Blackjack (99.5%) > Roulette EU (97.3%)
│   │   ├─ Problem: Don't always know RTP (unlisted in casinos)
│   │   ├─ Solution: Research game rules or test over time
│   │   └─ Long-term: Seek advantage play (+RTP) over time
│   ├─ Bet Structure Optimization:
│   │   ├─ Strategy 1: Use high-RTP bets within game
│   │   ├─ Example (Craps): Odds bets (100% RTP) > field bets (94.4% RTP)
│   │   ├─ Strategy 2: Skip low-RTP bets entirely
│   │   ├─ Example: Avoid roulette Ties (14.4% RTP)
│   │   └─ Compound: Multiple high-RTP decisions beat few mistakes
│   ├─ Bankroll Scaling:
│   │   ├─ Lower RTP → smaller bets to preserve capital
│   │   ├─ Higher RTP → can sustain larger bets
│   │   ├─ Rule: Max bet ∝ 1 / House edge
│   │   └─ Example: 1% HE allows 10× larger bets than 10% HE (same ruin risk)
│   └─ Multi-Game Strategy:
│       ├─ Mix high-RTP games (losses slower)
│       ├─ If +RTP found, concentrate play there
│       ├─ Example: Mix 99.5% blackjack (hedge) + 96% slots (entertainment)
│       └─ Weighted average: Better than worst game alone
├─ V. PAYOUT RATIO PITFALLS & MISCONCEPTIONS:
│   ├─ Short-Term Variance:
│   │   ├─ RTP applies only long-term (asymptotic)
│   │   ├─ Short-term: Anything possible (win or lose)
│   │   ├─ Misconception: "I'm owed a win after losses" (false)
│   │   └─ Truth: Convergence slow; bankroll must sustain variance
│   ├─ Ignoring Rake & Commissions:
│   │   ├─ Listed RTP sometimes excludes rake (poker) or commissions
│   │   ├─ Actual RTP lower than advertised
│   │   ├─ Example: Poker "95% RTP" + 5% rake = 90% actual
│   │   └─ Solution: Always check rake structure separately
│   ├─ Confusing RTP with Win Probability:
│   │   ├─ 95% RTP ≠ 95% chance to win
│   │   ├─ Example: Roulette is ~50/50 win/lose but 97.3% RTP
│   │   ├─ Implication: You lose just as often; net loss smaller with high RTP
│   │   └─ Lesson: RTP affects magnitude of loss, not probability
│   ├─ Assuming Constant RTP:
│   │   ├─ Reality: RTP varies by conditions
│   │   ├─ Example: Blackjack RTP depends on deck composition (card counting)
│   │   ├─ Example: Poker rake as % decreases at higher stakes
│   │   ├─ Implication: Adapt strategy as conditions change
│   │   └─ Solution: Re-calculate RTP periodically
│   └─ RTP ≠ Fairness:
│       ├─ High RTP still means house advantage
│       ├─ 99% RTP = house advantage
│       ├─ Only 100% RTP is fair (rare outside poker odds bets)
│       └─ Lesson: No casino game is "fair"; minimize disadvantage
├─ VI. CALCULATING PAYOUT RATIOS:
│   ├─ From Expected Value:
│   │   ├─ EV = (1 - Payout ratio) × bet
│   │   ├─ Payout ratio = 1 - EV / bet
│   │   └─ Example: EV = -$0.027, bet = $1 → RTP = 1 - 0.027 = 0.973
│   ├─ From House Edge:
│   │   ├─ RTP = 1 - House edge
│   │   └─ Example: 2.7% HE → 97.3% RTP
│   ├─ From Actual Results:
│   │   ├─ Observed RTP = total returned / total wagered
│   │   ├─ Calculation: Track all wins/losses over many hands
│   │   ├─ Statistical test: Compare to theoretical using chi-square
│   │   └─ Interpretation: If observed << theoretical, game unfair or rigged
│   └─ Time-Based Scaling:
│       ├─ Loss per hour = hourly wages wagered × house edge
│       ├─ Example: $100/hour bet, 2.7% HE → $2.70/hour loss
│       └─ Planning: Hours played × loss rate = total expected loss
├─ VII. REGULATORY PERSPECTIVE:
│   ├─ Casinos Required to Disclose:
│   │   ├─ Nevada: RTP must be posted (theoretical, not guaranteed)
│   │   ├─ EU: RTP by law, often > 95%
│   │   ├─ Online: RTP certified by third parties
│   │   └─ Slots: Typically 92-96% (varies by machine)
│   ├─ Verification:
│   │   ├─ Casino audit: Independent firm tests machines
│   │   ├─ Long-term play: Compare observed to theoretical
│   │   ├─ Red flag: Observed RTP << theoretical (rigged?)
│   │   └─ Solution: Report to gaming commission if suspicious
│   └─ Player Leverage:
│       ├─ If RTP not disclosed, request it
│       ├─ If refused, vote with feet (go elsewhere)
│       ├─ If rigged, report to regulators
│       └─ Collective action improves industry transparency
└─ VIII. PRACTICAL GUIDANCE:
    ├─ Ranking Decision:
    │   ├─ Tier 1 (98%+): Blackjack, Craps with odds → play these
    │   ├─ Tier 2 (95-97%): European Roulette, Baccarat → acceptable
    │   ├─ Tier 3 (90-95%): Slots, Video Poker → only entertainment budget
    │   ├─ Tier 4 (85-90%): American Roulette, most video poker → avoid
    │   └─ Tier 5 (<85%): Keno, progressive slots → strictly avoid
    ├─ Bankroll Allocation:
    │   ├─ Allocate by inverse RTP
    │   ├─ High RTP → larger allocation (slower loss)
    │   ├─ Low RTP → tiny allocation (entertainment only)
    │   └─ Example: 60% to 99% RTP, 30% to 96% RTP, 10% to 92% RTP
    ├─ Session Management:
    │   ├─ Track actual vs theoretical RTP
    │   ├─ If actual >> theoretical, quit while ahead
    │   ├─ If actual << theoretical, reduce bet size or switch games
    │   └─ Rule: Never extend session to "recover" (wrong math)
    └─ Long-Term Strategy:
        ├─ Find +RTP opportunities (card counting, poker skill, sports modeling)
        ├─ Only +RTP games justify extended play
        ├─ Minimize -RTP exposure via game selection
        └─ Track ROI: (Net profit) / (Total wagered) → should match RTP
```

**Core Insight:** Payout ratio (RTP) directly determines long-term profitability. 1% difference compounds to massive amounts over thousands of bets.

## 5. Mini-Project
Compare payout ratios across games and compute cumulative impact:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

print("="*70)
print("PAYOUT RATIOS: LONG-TERM IMPACT ANALYSIS")
print("="*70)

# ============================================================================
# 1. PAYOUT RATIOS BY GAME
# ============================================================================

print("\n" + "="*70)
print("1. PAYOUT RATIOS (RTP) BY GAME")
print("="*70)

games_payout = {
    'Blackjack (basic strategy)': 0.995,
    'Craps (Pass/odds)': 0.9859,
    'Baccarat (Banker)': 0.9894,
    'European Roulette': 0.973,
    'American Roulette': 0.9474,
    'Slots (typical)': 0.95,
    'Video Poker (good table)': 0.99,
    'Keno': 0.70,
}

df_payout = pd.DataFrame({
    'Game': list(games_payout.keys()),
    'Payout Ratio (RTP)': list(games_payout.values()),
})

df_payout['House Edge %'] = (1 - df_payout['Payout Ratio (RTP)']) * 100
df_payout = df_payout.sort_values('Payout Ratio (RTP)', ascending=False)

print("\nGame Rankings by Payout Ratio:")
print(df_payout.to_string(index=False))

# ============================================================================
# 2. CUMULATIVE LOSS CALCULATION
# ============================================================================

print("\n" + "="*70)
print("2. CUMULATIVE IMPACT: LOSSES OVER TIME")
print("="*70)

bet_amounts = [1, 5, 10, 100]
n_bets_list = [100, 1000, 10000]

print(f"\nExpected Loss by Game & Number of Bets:")
print(f"(Bet amount = $1)\n")

cumulative_results = []

for n_bets in n_bets_list:
    print(f"\n{n_bets} bets:")
    for game, rtp in games_payout.items():
        house_edge = 1 - rtp
        expected_loss = n_bets * house_edge * 1.0  # $1 bet
        cumulative_results.append({
            'Game': game,
            'N_bets': n_bets,
            'Expected_loss': expected_loss
        })
        print(f"  {game:30s}: ${expected_loss:7.2f} loss")

# ============================================================================
# 3. COMPARISON: PAYOUT RATIO IMPACT
# ============================================================================

print("\n" + "="*70)
print("3. COMPARING TWO STRATEGIES")
print("="*70)

print("\nStrategy A: Play Blackjack (99.5% RTP)")
print("Strategy B: Play Keno (70% RTP)")
print("Same bet: $1/hand, 1000 hands")

n_hands = 1000
bet_size = 1
loss_blackjack = (1 - 0.995) * n_hands * bet_size
loss_keno = (1 - 0.70) * n_hands * bet_size
difference = loss_keno - loss_blackjack

print(f"\nBlackjack expected loss: ${loss_blackjack:.2f}")
print(f"Keno expected loss: ${loss_keno:.2f}")
print(f"Additional loss from worse RTP: ${difference:.2f}")
print(f"Multiple: {difference / loss_blackjack:.1f}x worse")

# ============================================================================
# 4. HOURLY LOSS RATE
# ============================================================================

print("\n" + "="*70)
print("4. HOURLY LOSS RATES")
print("="*70)

# Assume hands per hour
hands_per_hour = {
    'Blackjack': 60,
    'Craps': 100,
    'Roulette': 30,
    'Slots': 600,
}

bet_size_hourly = 1

print(f"\nHourly loss rate for different games (${bet_size_hourly} bet):")
for game_short, hands in hands_per_hour.items():
    for game_full, rtp in games_payout.items():
        if game_short.lower() in game_full.lower():
            house_edge = 1 - rtp
            hourly_loss = hands * bet_size_hourly * house_edge
            print(f"  {game_full:30s}: ${hourly_loss:6.2f}/hour ({hands} hands/hr)")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Payout ratio by game
ax1 = axes[0, 0]
colors_payout = ['green' if rtp > 0.97 else 'orange' if rtp > 0.92 else 'red' 
                 for rtp in df_payout['Payout Ratio (RTP)']]
ax1.barh(df_payout['Game'], df_payout['Payout Ratio (RTP)'], color=colors_payout, alpha=0.7, edgecolor='black')
ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Break-even (100%)')
ax1.set_xlabel('Payout Ratio (RTP)')
ax1.set_xlim(0.65, 1.01)
ax1.set_title('Payout Ratios by Game')
ax1.grid(True, alpha=0.3, axis='x')
ax1.legend()

# Plot 2: Cumulative loss over bets
ax2 = axes[0, 1]
for game, rtp in list(games_payout.items())[:5]:  # Top 5 games
    house_edge = 1 - rtp
    losses = np.array(n_bets_list) * house_edge * 1.0
    ax2.plot(n_bets_list, losses, marker='o', linewidth=2, markersize=8, label=game.split('(')[0].strip())

ax2.set_xlabel('Number of Bets')
ax2.set_ylabel('Expected Loss ($)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Cumulative Loss vs Bets (log scale)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: House edge comparison
ax3 = axes[1, 0]
he_values = df_payout['House Edge %']
colors_he = ['green' if he < 2 else 'orange' if he < 5 else 'red' for he in he_values]
ax3.bar(range(len(df_payout)), he_values, color=colors_he, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(df_payout)))
ax3.set_xticklabels([g.split('(')[0].strip() for g in df_payout['Game']], rotation=45, ha='right')
ax3.set_ylabel('House Edge (%)')
ax3.set_title('House Edge by Game')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, he in enumerate(he_values):
    ax3.text(i, he + 0.3, f'{he:.1f}%', ha='center', fontsize=9)

# Plot 4: Loss comparison: Blackjack vs Keno (1000 bets)
ax4 = axes[1, 1]
games_compare = ['Blackjack\n(99.5% RTP)', 'Keno\n(70% RTP)']
losses_compare = [loss_blackjack, loss_keno]
colors_compare = ['green', 'red']

bars = ax4.bar(games_compare, losses_compare, color=colors_compare, alpha=0.7, edgecolor='black', width=0.5)
ax4.set_ylabel('Expected Loss ($)')
ax4.set_title(f'Loss Comparison: 1000 bets of $1')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels and multiplier
for i, (bar, loss) in enumerate(zip(bars, losses_compare)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'${loss:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add multiplier annotation
mid_x = 0.5
mid_y = (losses_compare[0] + losses_compare[1]) / 2
ax4.text(mid_x, mid_y, f'{difference / loss_blackjack:.0f}x\nworse', 
        ha='center', va='center', fontsize=12, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('payout_ratios_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: payout_ratios_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Payout ratio (RTP) directly determines long-term profitability")
print("✓ 1% difference compounds: 100 bets = $1 difference, 10,000 bets = $100 difference")
print("✓ Game selection critical: Play highest RTP available")
print("✓ Blackjack (99.5%) vastly superior to Keno (70%)")
```

## 6. Challenge Round
**When do payout ratios mislead?**
- Non-linear: 95% RTP over 1000 bets ≠ 95% RTP over 1 bet (variance huge for single)
- Rake structure: Actual RTP may be lower after rake/commission included
- Game variations: Different rules within game → different RTP (must verify)
- Adaptive play: With skill (card counting, poker), true RTP changes (static RTP invalid)
- Sample size: Observed RTP from limited plays unreliable (need thousands of trials)

## 7. Key References
- [Wizard of Odds - Casino House Edge](https://www.wizardofodds.com/gambling/) - Comprehensive RTP database
- [Nevada Gaming Control Board](https://gaming.nv.gov/) - Actual RTP percentages by machine
- [Wikipedia - Return to Player](https://en.wikipedia.org/wiki/Return_to_player) - Regulatory definitions

---
**Status:** Quick comparison metric for game selection | **Complements:** Expected value, House edge | **Enables:** Game ranking, loss planning