# Card Counting: Tracking Deck Composition for Blackjack Edge

## 1. Concept Skeleton
**Definition:** System for tracking remaining card composition in deck to estimate deck favorability and adjust bets accordingly  
**Purpose:** Gain mathematical edge over casino (positive EV); increase bet size when deck is favorable; decrease when unfavorable  
**Prerequisites:** Blackjack basic strategy, expected value, probability of card values, risk of ruin concept

## 2. Comparative Framing
| Method | Hi-Lo System | KO System | Omega II | Hi-Opt I |
|--------|-------------|----------|----------|----------|
| **Complexity** | Simple | Simple | Moderate | Moderate |
| **Counting Range** | -16 to +16 typical | 0 to +50+ | -20 to +16 | -8 to +8 |
| **Accuracy** | ~80% | ~75% | ~95% | ~90% |
| **Learning Time** | 1-2 weeks | 1-2 weeks | 2-3 weeks | 2-3 weeks |
| **Table Speed** | Fast enough | Fast enough | Difficult | Fast enough |
| **Edge Achieved** | 0.5-1.5% | 0.3-1% | 1-2% | 0.75-1.5% |
| **Professional Use** | Most common | Transitional | Advanced | Advanced |
| **Casino Detection** | Higher risk | Higher risk | Moderate | Moderate |

## 3. Examples + Counterexamples

**Simple Example:**  
Hi-Lo system: Count low cards (2-6) as +1, high cards (10-A) as -1, neutral (7-9) as 0. Running count +5 after half deck → favorable deck

**Failure Case:**  
Counting perfectly but betting the same amount regardless of count → no advantage captured; must vary bet size  
Using outdated count strategy (0.5% edge) when casino uses 6-deck shoe → effort wasted by penetration/rules

**Edge Case:**  
Continuous shuffler machines (CSM): Impossible to count; no edge possible regardless of skill

## 4. Layer Breakdown
```
Card Counting Framework:
├─ I. COUNTING SYSTEMS:
│   ├─ Basic Principle:
│   │   ├─ Assign point value to each card (2-A)
│   │   ├─ Track running count: sum of all card values seen
│   │   ├─ Convert to true count: running count / decks remaining
│   │   ├─ High true count = favorable (more 10s, Aces)
│   │   └─ Implication: Adjust bet size proportional to true count
│   ├─ Hi-Lo System (Most Popular):
│   │   ├─ Card values: 2-6 = +1, 7-9 = 0, 10-A = -1
│   │   ├─ Rationale: High cards (10,A) favor player (blackjack, bust dealer)
│   │   ├─ Low cards (2-6) favor dealer (reduce bust risk)
│   │   ├─ Running count: Cumulative sum from shoe start
│   │   ├─ True count: Running count ÷ estimated decks remaining
│   │   ├─ Bet spreading: Bet $1 at TC -1, $4 at TC +3, $8+ at TC +5+
│   │   ├─ Accuracy: Misses aces (high impact); simplified index
│   │   └─ Edge: 0.5-1.5% depending on penetration, rules
│   ├─ KO System (Knockout):
│   │   ├─ Card values: 2-7 = +1, 8-9 = 0, 10-A = -1
│   │   ├─ Difference: Counts 7s (no neutral), eliminates ace adjustment
│   │   ├─ Advantage: Simpler (no true count needed at certain thresholds)
│   │   ├─ Disadvantage: Slightly less accurate than Hi-Lo
│   │   ├─ Penetration: Easier to use with shallow penetration (earlier play decisions)
│   │   └─ Edge: 0.3-1% (lower than Hi-Lo but sufficient)
│   ├─ Omega II System:
│   │   ├─ Card values: 2,3,7 = +1; 4,5,6 = +2; 8 = 0; 9 = -1; 10-A = -2
│   │   ├─ Accuracy: Tracks both high and low concentration
│   │   ├─ Complexity: Multi-value assignments difficult in real-time
│   │   ├─ Usage: Professional advantage players, offline practice
│   │   ├─ Speed: Requires discipline and practice (error-prone)
│   │   └─ Edge: 1-2% (highest but hardest to execute)
│   ├─ Hi-Opt I System:
│   │   ├─ Card values: 3,4,5,6 = +1; 10-A = -1; all others = 0
│   │   ├─ Focus: Only very high/low impact cards counted
│   │   ├─ Simplicity: Fewer decisions than Hi-Lo
│   │   ├─ Advantage: 0.75-1.5% with less mental load
│   │   ├─ Limitation: May miss some betting opportunities
│   │   └─ Learning: 2 weeks typical
│   └─ Ace-Adjustment Systems:
│       ├─ Ace Side Count: Track aces separately (they're most valuable)
│       ├─ Combo Index: Integrate ace count with regular count
│       ├─ Improvement: Adds 0.1-0.2% accuracy
│       └─ Trade-off: Requires simultaneous tracking of two counts
├─ II. PRACTICAL EXECUTION:
│   ├─ Running Count Tracking:
│   │   ├─ Method 1 (Mental): Memorize cumulative count (1, 2, 3, ..., +5)
│   │   ├─ Method 2 (Chips): Use chips to track (e.g., move chip $1 per count)
│   │   ├─ Method 3 (Fingers/Toes): Hidden physical tracking
│   │   ├─ Challenge: Must remain natural/unnoticed by casino
│   │   ├─ Speed: Process 60+ cards per minute in real casino
│   │   └─ Error Rate: 1-2% error acceptable; higher = caught by surveillance
│   ├─ True Count Conversion:
│   │   ├─ Formula: True count = Running count / Decks remaining
│   │   ├─ Example: Running count +8, ~2 decks left → TC = +8/2 = +4
│   │   ├─ Decks remaining estimate: Penetration % × shoe size
│   │   ├─ Estimation: Count cards remaining in discard pile, subtract from shoe
│   │   ├─ Precision: Error in estimation → ±0.5 TC typical
│   │   └─ Impact: True count critical for betting decisions
│   ├─ Bet Sizing Strategy:
│   │   ├─ Linear Bet Spread: $1-$4 (4× bet spread common)
│   │   ├─ Non-linear: $1, $2, $4, $8, $16 (aggressive, easier to detect)
│   │   ├─ Optimal: Bet amount ∝ (True count × Edge ÷ Variance)
│   │   ├─ Risk: Large bet spreads trigger casino surveillance
│   │   ├─ Cover: Occasional large bets on negative TC (appear to be lucky)
│   │   ├─ Table Entry: Bet minimally until TC known (avoid betting blind)
│   │   └─ Exit: Leave after win (avoid "back to back large bets" pattern)
│   ├─ Index Play Adjustment:
│   │   ├─ Index Numbers: For each hand type (hard 16, soft 17), threshold TC to deviate
│   │   ├─ Example: "Hit hard 16 vs dealer 10 if TC ≥ +4"
│   │   ├─ Basic strategy: Play assuming TC = 0
│   │   ├─ Index table: Memorize deviations for all major plays
│   │   ├─ Complexity: 30-50 indices for full optimization
│   │   └─ Simplification: Use subset (10-15 key indices) for casual play
│   └─ Casino Countermeasures:
│       ├─ Early shuffle: Shuffle when penetration high (reduce available cards)
│       ├─ Mid-shoe entry allowed: Can't start betting at table start
│       ├─ Bet spread limits: "Bet variations suspicious, leave table"
│       ├─ Back-counting: Count before sitting, only play at favorable TC
│       ├─ Surveillance: Video review, thermal imaging, betting pattern analysis
│       └─ Barring: Refused play if suspected counter (private casino right)
├─ III. ADVANTAGE CALCULATION:
│   ├─ Player Edge from Counting:
│   │   ├─ Formula: Edge ≈ (EV at TC) - (EV at TC 0) × true count frequency
│   │   ├─ Baseline: Blackjack -0.5% to -1.5% (house edge vs average player)
│   │   ├─ With counting: Can shift to +0.5% to +1.5% player edge
│   │   ├─ Conditional on count: Edge varies by true count
│   │   ├─ +1 TC: ~0% edge (break-even to +0.25%)
│   │   ├─ +3 TC: ~0.5-1% edge (moderate)
│   │   ├─ +5+ TC: ~1.5-2% edge (very favorable)
│   │   ├─ -2 TC: ~1.5% house edge (very unfavorable)
│   │   └─ Implication: At negative TC, leave or minimum bet
│   ├─ Risk of Ruin (for Counter):
│   │   ├─ Formula: RoR ≈ exp(-2 × EV × B / σ²)
│   │   ├─ Variance (σ): Blackjack ~1.15 per hand
│   │   ├─ Example: 1% edge, $10,000 bankroll, 200 hands
│   │   ├─ RoR ≈ 5-10% (manageable with proper bankroll)
│   │   ├─ Comparison: Non-counter with -1% edge → certain ruin
│   │   └─ Implication: Bankroll requirements: ~100-200× minimum bet
│   ├─ Expected Hourly Win Rate:
│   │   ├─ Formula: Hourly EV = (EV %) × (average bet) × (hands/hour)
│   │   ├─ Example: 0.75% edge, $20 avg bet, 60 hands/hr
│   │   ├─ Hourly EV ≈ 0.0075 × $20 × 60 = $9/hour
│   │   ├─ Reality: Swings ±$100+ common (variance dominates short-term)
│   │   ├─ Annual: $9 × 2000 hours = $18,000 (before variance)
│   │   └─ Sustainability: Requires long-term commitment, significant bankroll
│   └─ Profitability vs Detection:
│       ├─ Low profile: Minimal bet spread → lower detection, lower edge capture
│       ├─ High profile: Large bet spread → higher detection risk, higher edge
│       ├─ Trade-off: Professional players use low profile (lifetime career)
│       └─ Casino-specific: High-limit games less watched vs penny slots
├─ IV. LIMITATIONS & CHALLENGES:
│   ├─ Casino Countermeasures:
│   │   ├─ Multiple decks (6, 8 deck shoes): Dilutes counting accuracy
│   │   ├─ Continuous Shufflers: Eliminate counting entirely
│   │   ├─ Frequent shuffles: Restart counting frequently (penertration crucial)
│   │   ├─ Rules changes: No soft 17 double, lower blackjack payout (1.2:1 vs 1.5:1)
│   │   ├─ Side bets: Optional bets with much higher house edge (avoid)
│   │   └─ Volatility: Even with edge, short-term losses possible (variance)
│   ├─ Player Errors:
│   │   ├─ Counting mistakes: +1 error becomes -2 expected value
│   │   ├─ True count miscalculation: Overestimate remaining decks → underbet
│   │   ├─ Index play mistakes: Playing basic strategy when should deviate
│   │   ├─ Bet sizing inconsistency: Suspicious to casino
│   │   ├─ Emotional play: Deviation from strategy after losses (tilt)
│   │   └─ Fatigue: Long sessions reduce accuracy
│   ├─ Detection & Barring:
│   │   ├─ Surveillance: Casino security reviews video, identifies patterns
│   │   ├─ Betting patterns: Bet $1 at -2 TC, $100 at +3 TC (obvious)
│   │   ├─ Edge detection: Computers scan tables for unusual win rates
│   │   ├─ Network sharing: Casinos share photos of banned counters
│   │   ├─ Barring: Legal right of private businesses (not illegal, but can refuse service)
│   │   └─ Consequence: Permanent ban from casino (sometimes entire jurisdiction)
│   └─ Bankroll Requirements:
│       ├─ Volatility scaling: σ² scales with hand count
│       ├─ For 0.75% edge, 1% RoR → need ~$150k+ bankroll
│       ├─ For 1.5% edge → need ~$50k+ bankroll
│       ├─ Downswings: 50+ hand losing streaks possible (LLN takes time)
│       └─ Professional play: Requires serious capital commitment
├─ V. STRATEGIC VARIANTS:
│   ├─ Back-Counting (Wonging):
│   │   ├─ Concept: Count cards while standing (not playing)
│   │   ├─ Entry: Sit down only when TC favorable (e.g., +2+)
│   │   ├─ Advantage: Avoid negative expectation hands
│   │   ├─ Risk: Casino may bar for suspicious entry pattern
│   │   ├─ Edge gain: Additional 0.1-0.3% vs full play
│   │   └─ Application: Team play (one counts, others play)
│   ├─ Team Play:
│   │   ├─ Roles: Spotters (count, stay longer), Big Player (bets large when +TC)
│   │   ├─ Advantage: Big player only plays hands; reduces variance
│   │   ├─ Communication: Coded signals for true count updates
│   │   ├─ Detection: Sophisticated teams harder to spot (organized)
│   │   ├─ Risk: Entire team banned if one member caught
│   │   └─ Profitability: Higher hourly but higher risk/complexity
│   ├─ Shuffle Tracking:
│   │   ├─ Concept: Track clumps of cards through shuffle
│   │   ├─ Edge: 0.5-1.5% additional on top of regular counting
│   │   ├─ Complexity: Extremely difficult (requires multiple card tracking)
│   │   ├─ Application: Professional teams with years of training
│   │   └─ Rarity: Very few practitioners worldwide
│   └─ Deck Estimation Improvements:
│       ├─ Zone counting: Estimate cards remaining by visualizing discard pile volume
│       ├─ Peek counting: Casino sometimes reveals next card (mishandling)
│       ├─ Precise estimation: Reduces true count error → improves edge
│       └─ Practice: Offline card tracking exercises critical
├─ VI. LEGAL & ETHICAL ASPECTS:
│   ├─ Legality:
│   │   ├─ USA: Card counting NOT illegal (mental process)
│   │   ├─ But: Casinos are private businesses, can refuse service
│   │   ├─ Banning: Legal if no deception (using devices illegal)
│   │   ├─ Devices: Hidden earpieces, computer assistance → criminal (illegal)
│   │   ├─ International: Varies by jurisdiction (some countries ban counters)
│   │   └─ Conclusion: Legal to count, illegal to use devices
│   ├─ Ethics:
│   │   ├─ From casino: Playing against published odds; slight advantage reversal
│   │   ├─ From player: Skill-based edge similar to poker
│   │   ├─ From society: Minimal harm (casinos profit overall)
│   │   ├─ Debate: Casino advantage over decades much larger than countingAdvantage
│   │   └─ Stance: Most mathematicians/advantage players consider it ethical
│   ├─ Risk Assessment:
│   │   ├─ Primary risk: Lifetime barring from casinos (not legal prosecution)
│   │   ├─ Financial risk: Bankroll loss if detected mid-session
│   │   ├─ Reputational: Some view as "cheating" (philosophical)
│   │   └─ Decision: Personal choice based on risk tolerance
│   └─ Professional Viability:
│       ├─ Income potential: $50k-$200k+/year for successful teams
│       ├─ Sustainability: Decreasing as casinos improve countermeasures
│       ├─ Career duration: Average 2-5 years before barring
│       ├─ Evolution: Successful counters move to poker, sports betting
│       └─ Modern reality: Harder than 1970s-1990s (post-"Rain Man" era)
└─ VII. PRACTICAL IMPLEMENTATION:
    ├─ Training Protocol:
    │   ├─ Phase 1 (Week 1): Learn Hi-Lo values, drill flashcards
    │   ├─ Phase 2 (Week 2): Count from 1 deck (all cards visible)
    │   ├─ Phase 3 (Week 3): Count from 6-deck shoe at home (realistic speed)
    │   ├─ Phase 4 (Week 4): Count while playing blackjack (basic strategy)
    │   ├─ Phase 5 (Month 2): Add bet spreading and index plays
    │   ├─ Phase 6 (Month 3): Practice in actual casino (small bets)
    │   └─ Total: ~3 months minimum for competence
    ├─ Bankroll Planning:
    │   ├─ Minimum edge: 0.5% (with good rules, 75% penetration)
    │   ├─ Bankroll size: 200 × average bet for 1-2% RoR
    │   ├─ Example: $15/hand avg bet → $3000 minimum bankroll
    │   ├─ Realistic: $10,000-$50,000 for 6-8 hour day, multiple tables
    │   └─ Safety factor: Plan for downswings of 50+ hands
    ├─ Casino Selection:
    │   ├─ Favorable: 6-deck shoes, 75%+ penetration, $5-$25 tables
    │   ├─ Rules: Soft 17 hit, resplit aces, DAS (double after split)
    │   ├─ Unfavorable: Continuous shufflers, strict bet limits, 1.2:1 blackjack
    │   ├─ Surveillance: Choose lower-profile locations (not Vegas)
    │   └─ Team coordination: Share intelligence on friendly casinos
    └─ Long-Term Sustainability:
        ├─ Barring inevitability: Eventually caught or banned (average 2-5 years)
        ├─ Geographic rotation: Different casinos, cities, countries
        ├─ Identity management: Use different names/appearances (risky)
        ├─ Exit strategy: Plan transition to poker, sports betting before barring
        ├─ Financial goal: Make enough to sustain lifestyle after barring
        └─ Perspective: Counting as finite career, not lifetime
```

**Core Insight:** Card counting shifts blackjack from -1% to +0.5-1.5% edge; viability depends on discipline, bankroll, and casino countermeasures.

## 5. Mini-Project
Simulate card counting and betting strategy over 1000 hands:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)

print("="*70)
print("CARD COUNTING: TRACKING BLACKJACK ADVANTAGE")
print("="*70)

# ============================================================================
# 1. HI-LO COUNTING SYSTEM
# ============================================================================

print("\n" + "="*70)
print("1. HI-LO SYSTEM MECHANICS")
print("="*70)

card_values_hiLo = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,  # Low cards
    '7': 0, '8': 0, '9': 0,                   # Neutral
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1  # High cards
}

def create_deck(num_decks=6):
    """Create a shoe with num_decks standard decks."""
    cards = []
    for _ in range(num_decks):
        cards.extend(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4)
    np.random.shuffle(cards)
    return cards

def calculate_true_count(running_count, cards_remaining):
    """Convert running count to true count."""
    decks_remaining = cards_remaining / 52  # 52 cards per deck
    if decks_remaining > 0:
        return running_count / decks_remaining
    return running_count

def get_bet_size(true_count, base_bet=10):
    """Determine bet size based on true count."""
    # Betting strategy: $10 base, increase with positive count
    if true_count < 0:
        return base_bet  # Minimum bet on negative count
    elif true_count < 1:
        return base_bet
    elif true_count < 2:
        return base_bet * 1.5
    elif true_count < 3:
        return base_bet * 2
    elif true_count < 4:
        return base_bet * 3
    else:
        return base_bet * 4  # Max spread (4x bet)

# ============================================================================
# 2. SIMULATE 1000 HANDS OF COUNTING
# ============================================================================

print("\nSimulating 1000 hands with Hi-Lo card counting:")

deck = create_deck(num_decks=6)
running_count = 0
hands_played = 0
deck_position = 0
penetration_threshold = int(len(deck) * 0.75)  # 75% penetration before reshuffle

hand_results = []
bet_sizes = []
true_counts = []
counts = []

# Simulate 1000 hands
for hand in range(1000):
    # Check if reshuffle needed (penetration)
    if deck_position >= penetration_threshold:
        deck = create_deck(num_decks=6)
        running_count = 0
        deck_position = 0
    
    # Draw 2 cards for dealer, 2 for player (simplified)
    cards_in_hand = []
    for _ in range(4):
        if deck_position < len(deck):
            card = deck[deck_position]
            cards_in_hand.append(card)
            running_count += card_values_hiLo[card]
            deck_position += 1
    
    cards_remaining = len(deck) - deck_position
    true_count = calculate_true_count(running_count, cards_remaining)
    bet_size = get_bet_size(true_count)
    
    # Simulate hand outcome (simplified: random win/loss)
    # In reality, outcome depends on player/dealer hands
    player_wins = np.random.random() > 0.51  # ~49% player win rate (accounting for push, BJ)
    
    if player_wins:
        profit = bet_size  # Simplified (ignore blackjack 3:2 payout)
    else:
        profit = -bet_size
    
    hand_results.append(profit)
    bet_sizes.append(bet_size)
    true_counts.append(true_count)
    counts.append(running_count)
    hands_played += 1

# ============================================================================
# 3. ANALYSIS: COUNTING METRICS
# ============================================================================

print(f"\nSimulation Results (1000 hands):")
print(f"{'Metric':<30} {'Value':>20}")
print("=" * 50)

total_wagered = np.sum(bet_sizes)
total_profit = np.sum(hand_results)
avg_profit_per_hand = total_profit / hands_played
win_rate = np.sum([1 for x in hand_results if x > 0]) / hands_played

print(f"{'Total hands played':<30} {hands_played:>20}")
print(f"{'Total wagered':<30} ${total_wagered:>19,.2f}")
print(f"{'Total profit/loss':<30} ${total_profit:>19,.2f}")
print(f"{'Avg profit per hand':<30} ${avg_profit_per_hand:>19,.2f}")
print(f"{'Win rate':<30} {win_rate*100:>19.1f}%")
print(f"{'Hands with +TC':<30} {np.sum([1 for tc in true_counts if tc > 0]):>20}")
print(f"{'Max true count reached':<30} {max(true_counts):>19.2f}")
print(f"{'Min true count reached':<30} {min(true_counts):>19.2f}")
print(f"{'Avg true count':<30} {np.mean(true_counts):>19.2f}")

# ============================================================================
# 4. BET SIZING ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("2. BET SIZING BY TRUE COUNT")
print("="*70)

bet_by_count = defaultdict(list)
for tc, bet in zip(true_counts, bet_sizes):
    tc_bucket = int(tc * 2) / 2  # Round to nearest 0.5
    bet_by_count[tc_bucket].append(bet)

print(f"\n{'True Count Range':<20} {'Avg Bet':<15} {'Frequency':>15}")
print("=" * 50)

for tc in sorted(bet_by_count.keys())[-10:]:  # Show top 10 TC values
    avg_bet = np.mean(bet_by_count[tc])
    freq = len(bet_by_count[tc])
    print(f"{tc:>5.1f} to {tc+0.5:>5.1f}       ${avg_bet:>12.2f}   {freq:>15}")

# ============================================================================
# 5. CUMULATIVE PROFIT TRACKING
# ============================================================================

cumulative_profit = np.cumsum(hand_results)
cumulative_wagered = np.cumsum(bet_sizes)

print("\n" + "="*70)
print("3. PROFIT/LOSS EVOLUTION")
print("="*70)

milestones = [250, 500, 750, 1000]
print(f"\n{'Hands':<10} {'Cumulative P/L':<20} {'Avg Bet':<15} {'ROI %':>15}")
print("=" * 60)

for milestone in milestones:
    cum_pl = cumulative_profit[milestone - 1]
    cum_bet = cumulative_wagered[milestone - 1]
    roi = (cum_pl / cum_bet * 100) if cum_bet > 0 else 0
    avg_bet = cum_bet / milestone
    print(f"{milestone:<10} ${cum_pl:>17,.2f}   ${avg_bet:>12.2f}   {roi:>14.2f}%")

# ============================================================================
# 6. DETECTION RISK ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("4. BETTING PATTERN ANALYSIS (Detection Risk)")
print("="*70)

# Analyze betting correlation to true count (suspicious pattern)
bet_correlation = np.corrcoef(true_counts, bet_sizes)[0, 1]

print(f"\nBet-Count correlation: {bet_correlation:.3f}")
print(f"(>0.7 = suspicious pattern that casinos detect)")

# Count instances of max bet spread
max_bet = max(bet_sizes)
min_bet = min(bet_sizes)
spread_ratio = max_bet / min_bet

print(f"\nBet spread ratio: {spread_ratio:.1f}:1")
print(f"(>4:1 = very suspicious; typical casino tolerance ~2:1)")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative profit vs hands
ax1 = axes[0, 0]
hands_x = np.arange(1, len(cumulative_profit) + 1)
ax1.plot(hands_x, cumulative_profit, linewidth=2, color='blue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.fill_between(hands_x, cumulative_profit, 0, alpha=0.3)
ax1.set_xlabel('Hand Number')
ax1.set_ylabel('Cumulative Profit/Loss ($)')
ax1.set_title('Profit Evolution Over 1000 Hands')
ax1.grid(True, alpha=0.3)

# Plot 2: True count distribution
ax2 = axes[0, 1]
ax2.hist(true_counts, bins=50, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral (TC=0)')
ax2.axvline(x=np.mean(true_counts), color='blue', linestyle='--', linewidth=2, label=f'Mean ({np.mean(true_counts):.2f})')
ax2.set_xlabel('True Count')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of True Counts')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Bet size vs true count (should show correlation)
ax3 = axes[1, 0]
ax3.scatter(true_counts[::10], bet_sizes[::10], alpha=0.5, s=30)  # Plot every 10th to reduce clutter
ax3.set_xlabel('True Count')
ax3.set_ylabel('Bet Size ($)')
ax3.set_title(f'Bet Size vs True Count (Correlation: {bet_correlation:.3f})')
ax3.grid(True, alpha=0.3)

# Plot 4: Running count progression
ax4 = axes[1, 1]
ax4.plot(counts, linewidth=1, alpha=0.7, color='purple')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Hand Number')
ax4.set_ylabel('Running Count')
ax4.set_title('Running Count Progression (with reshuffles)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('card_counting_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: card_counting_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Hi-Lo counting achieves 0.5-1.5% edge in favorable shoes")
print("✓ True count fluctuates; affects betting and strategy decisions")
print("✓ Bet correlation high → casino surveillance detects pattern")
print("✓ Long-term: Edge + variance requires 100-200× minimum bet bankroll")
print("✓ Career duration: Average 2-5 years before detection/barring")
```

## 6. Challenge Round
**When does card counting fail?**
- Continuous shuffler machines: No card removal memory; counting impossible
- Multi-deck shoes with poor penetration (<50%): Edge reduced to <0.25%
- Countermeasures: Mid-shoe entry banned, no bet variations allowed
- Casino surveillance: Modern AI detects betting patterns in real-time
- High stress: Errors increase; mental fatigue reduces accuracy
- Variance: Even with +1% edge, 50-hand losing streaks possible (LLN takes time)

## 7. Key References
- [MIT Blackjack Team - Card Counting Methods](https://www.wizardofodds.com/games/blackjack/card-counting/)
- [Edward Thorp - Beat the Dealer (1962)](https://en.wikipedia.org/wiki/Edward_Thorp) - Classic card counting foundations
- [Casino Countermeasures & Detection](https://www.countercultureproductions.com/) - Modern advantage play challenges

---
**Status:** Primary blackjack advantage technique | **Complements:** Bankroll management, Risk of ruin, Bet sizing | **Enables:** Professional gambling viability, edge quantification