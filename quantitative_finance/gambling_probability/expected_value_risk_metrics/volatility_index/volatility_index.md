# Volatility Index: Classifying Game Risk Profiles

## 1. Concept Skeleton
**Definition:** Categorization system for games by outcome variability; volatility index ranges from low (stable) to extreme (wild swings)  
**Purpose:** Bankroll management; risk tolerance matching; session planning; portfolio diversification  
**Prerequisites:** Variance, standard deviation, expected value, probability distributions

## 2. Comparative Framing
| Metric | Volatility Index | Volatility Coefficient | Sharpe Ratio | Beta |
|--------|-----------------|----------------------|-------------|------|
| **Scope** | Game outcome category | Numerical (σ or CV) | Risk-adjusted return | Market comparison |
| **Purpose** | Game selection | Bankroll sizing | Efficiency metric | Correlation |
| **Range** | Low/Medium/High/Extreme | 0.5-3.0+ | Any real number | 0.5-2.0+ |
| **Focus** | Outcome spread | Relative dispersion | Return per unit risk | Systematic risk |
| **Application** | Casual gambler | Portfolio manager | Hedge fund | Equities |
| **Calculation** | Qualitative + tables | σ / mean | (Return - rf) / σ | Covariance / market var |
| **Example** | Blackjack = Low | σ = 0.95 | 0.52 (BJ) | 0.8 (low correlation) |

## 3. Examples + Counterexamples

**Simple Example:**  
Roulette (medium-high volatility): Red/black bet has 50/50 odds, tight payout. Session results highly variable short-term.

**Failure Case:**  
Thinking "low volatility = guaranteed profit": Low volatility still loses money with negative EV; just more predictably.

**Edge Case:**  
Games with same variance but different RTP: Both risky, but one worse value. Choose high RTP regardless of volatility classification.

## 4. Layer Breakdown
```
Volatility Index: Game Risk Classification:
├─ I. DEFINITIONS & CATEGORIES:
│   ├─ Low Volatility (σ < 0.5):
│   │   ├─ Definition: Outcomes tightly clustered; little swing
│   │   ├─ Pattern: Consistent small losses or small wins
│   │   ├─ Examples: Blackjack, Baccarat, Craps (main bets)
│   │   ├─ Typical session: Lose/win 5-15% of bankroll
│   │   ├─ Characteristic: Slow, steady bleed if EV negative
│   │   ├─ Bankroll impact: Preserves capital longest
│   │   ├─ Duration: Allows extended play time
│   │   └─ Psychological: Less exciting but less stressful
│   ├─ Medium Volatility (0.5 < σ < 1.5):
│   │   ├─ Definition: Moderate outcome spread; noticeable variance
│   │   ├─ Pattern: Occasional larger wins/losses mixed with small
│   │   ├─ Examples: Roulette, Video poker, Standard slots
│   │   ├─ Typical session: Lose/win 10-50% of bankroll
│   │   ├─ Characteristic: Some lucky streaks, occasional bad runs
│   │   ├─ Bankroll impact: Moderate preservation
│   │   ├─ Duration: Allows reasonable play duration
│   │   └─ Psychological: Balanced excitement and control
│   ├─ High Volatility (1.5 < σ < 3.0):
│   │   ├─ Definition: Large outcome swings; significant variance
│   │   ├─ Pattern: Mix of big wins and losses in single session
│   │   ├─ Examples: Progressive slots, Keno, Poker (deep stacks)
│   │   ├─ Typical session: Lose/win 30-100%+ of bankroll
│   │   ├─ Characteristic: Wild rides; big wins possible but risky
│   │   ├─ Bankroll impact: Quick depletion possible
│   │   ├─ Duration: Short effective duration
│   │   └─ Psychological: Exciting but stressful; requires discipline
│   ├─ Extreme Volatility (σ > 3.0):
│   │   ├─ Definition: Massive swings; highly unpredictable outcomes
│   │   ├─ Pattern: All-or-nothing feel; binary near-total win/loss
│   │   ├─ Examples: Lottery-style games, Long-shot bets, Keno high
│   │   ├─ Typical session: Lose 100% or win 10x+ bankroll
│   │   ├─ Characteristic: Jackpot seeking; recreational entertainment
│   │   ├─ Bankroll impact: High ruin risk
│   │   ├─ Duration: Very short effective duration
│   │   └─ Psychological: Lottery-like fantasy; high emotional range
│   ├─ Classification Factors:
│   │   ├─ Payout structure: Narrow payouts → low volatility; Wide payouts → high
│   │   ├─ Probability: Even odds → lower volatility; Skewed odds → higher
│   │   ├─ Bet types: Single outcome → higher volatility; Aggregated → lower
│   │   ├─ House edge: Can compound: negative EV + high volatility = fastest ruin
│   │   └─ Skill factor: Skillful reduction can lower effective volatility
│   └─ Coefficient of Variation (CV):
│       ├─ Definition: σ / μ (normalized volatility)
│       ├─ Purpose: Compares volatility across games with different EV/scales
│       ├─ Formula: CV = (std dev of outcomes) / (expected value)
│       ├─ Interpretation: Higher CV = higher relative risk per unit return
│       ├─ Example: Roulette CV ≈ 4 (high relative risk despite -1/37 expected loss)
│       └─ Use: Helps rank games by risk-adjusted return
├─ II. VOLATILITY BY GAME:
│   ├─ Blackjack:
│   │   ├─ Volatility Index: LOW
│   │   ├─ σ per hand: ~0.95 times bet
│   │   ├─ Typical ±2σ range: Lose/win 2× bet in 95% of sessions
│   │   ├─ Session variance: Tight around -0.5% EV
│   │   ├─ Reason: 50/50-ish outcomes (win/lose/push); consistent payout
│   │   ├─ Implication: Slowest decline if playing perfect strategy
│   │   ├─ Volatility driver: Number of hands played (more hands = lower overall σ %)
│   │   └─ Strategy: Increase hand speed to accelerate loss if trying to reach goal
│   ├─ Roulette European:
│   │   ├─ Volatility Index: MEDIUM
│   │   ├─ σ per spin: ~0.99 (for even money bets)
│   │   ├─ Typical ±2σ range: Lose/win 2× bet in 95% of spins
│   │   ├─ Session variance: High variance per spin; but steady aggregate -2.7%
│   │   ├─ Reason: Binary outcome (win/lose); tight odds on red/black
│   │   ├─ Implication: Moderate bankroll preservation
│   │   ├─ Volatility driver: Sector bets vs single number (single # = extreme)
│   │   └─ Strategy: Sector bets (multiple numbers) reduce effective volatility
│   ├─ Craps:
│   │   ├─ Volatility Index: LOW-MEDIUM (depends on bet type)
│   │   ├─ σ (Pass line): ~0.94
│   │   ├─ σ (Field): ~1.2
│   │   ├─ σ (Hardways): ~2.5
│   │   ├─ Reason: Multiple correlated outcomes per round
│   │   ├─ Implication: Bet selection = volatility selection
│   │   ├─ Volatility driver: Bet type (simple pass vs complex odds)
│   │   └─ Strategy: Stick to pass/don't pass for lower volatility
│   ├─ Slots:
│   │   ├─ Volatility Index: MEDIUM-HIGH (varies by machine)
│   │   ├─ σ (tight slots): ~0.9
│   │   ├─ σ (loose slots): ~1.1
│   │   ├─ σ (progressive): ~1.5+
│   │   ├─ Typical range: ±1-3σ = massive swings per session
│   │   ├─ Reason: Discrete payout structure; occasional big jackpots
│   │   ├─ Implication: Quick bankroll depletion or jackpot
│   │   ├─ Volatility driver: Pay table (jackpot size = volatility)
│   │   └─ Strategy: Understand pay table before playing
│   ├─ Video Poker:
│   │   ├─ Volatility Index: MEDIUM
│   │   ├─ σ: Ranges 0.8-1.2 depending on pay table
│   │   ├─ Reason: Discrete hand payouts (high card, pair, flush, etc.)
│   │   ├─ Skill factor: Optimal play reduces effective variance (fewer poor outcomes)
│   │   ├─ Implication: Volatility reduced by skillful play
│   │   ├─ Volatility driver: Pay table (royal flush payout size)
│   │   └─ Strategy: Master strategy to reduce variance and improve RTP
│   ├─ Poker:
│   │   ├─ Volatility Index: HIGH (tournament) to EXTREME (cash game)
│   │   ├─ σ per hand: Varies wildly (1-10+) based on stack sizes
│   │   ├─ Reason: Continuous range of outcomes; all-in possibilities
│   │   ├─ Skill factor: Dominant; skilled players reduce volatility vs weak fields
│   │   ├─ Implication: Bankroll management critical; need large buffer
│   │   ├─ Volatility driver: Table stakes, field quality, variance of holdings
│   │   └─ Strategy: Tight play reduces volatility; loose games increase it
│   ├─ Baccarat:
│   │   ├─ Volatility Index: LOW-MEDIUM
│   │   ├─ σ: ~0.96 (banker), ~1.06 (player), ~1.4 (tie)
│   │   ├─ Reason: Simple binary outcome; payout adjustment (banker slightly less)
│   │   ├─ Implication: Lower volatility than roulette
│   │   ├─ Volatility driver: Bet selection (banker = lower, tie = higher)
│   │   └─ Strategy: Always bet banker for lowest house edge and volatility
│   ├─ Keno:
│   │   ├─ Volatility Index: EXTREME
│   │   ├─ σ: 2.0-4.0+ (varies by ticket type)
│   │   ├─ Reason: Very low probability of large payouts; lottery-like
│   │   ├─ Implication: Recreational only; expect rapid ruin
│   │   ├─ Volatility driver: Ticket structure (catching 6/6 numbers vs 3/6)
│   │   └─ Strategy: Avoid; worst odds in casino
│   └─ Sports Betting:
│       ├─ Volatility Index: MEDIUM-HIGH (depends on model)
│       ├─ σ: 1.0-2.0 typical
│       ├─ Reason: Continuous outcome range; correlated events possible
│       ├─ Skill factor: Critical; model quality = volatility reduction
│       ├─ Implication: Skilled bettors can achieve low volatility + positive EV
│       ├─ Volatility driver: Bet types (straight vs parlays; single vs multi-leg)
│       └─ Strategy: Straight bets lower volatility; parlays increase it
├─ III. VOLATILITY & BANKROLL INTERACTION:
│   ├─ Risk of Ruin Scaling:
│   │   ├─ Formula: RoR ≈ exp(-2 × |EV| × B / σ²)
│   │   ├─ σ² dominance: Doubling volatility = quadrupling ruin risk
│   │   ├─ Implication: Volatility impact huge on ruin probability
│   │   ├─ Example: Game A (σ=0.9) vs Game B (σ=1.8); Game B has 4× ruin risk
│   │   ├─ Bankroll compensation: Would need 16× bankroll to match Game A safety
│   │   └─ Strategy: Choose low volatility games to reduce required capital
│   ├─ Session Duration & Volatility:
│   │   ├─ Low volatility: Extended session (stay for hours)
│   │   ├─ High volatility: Short effective session (ruin or jackpot likely)
│   │   ├─ Example: Blackjack (low vol) sustainable for 8 hours vs Keno (extreme) likely 30min
│   │   ├─ Implication: Time limit depends on volatility category
│   │   └─ Strategy: Match session length to volatility; leave if variance spikes
│   ├─ Bet Sizing & Volatility:
│   │   ├─ Kelly Criterion Adjustment: f* = (EV/σ²) × payout ratio
│   │   ├─ High volatility: Smaller bet size (preserve capital)
│   │   ├─ Low volatility: Larger bet size (acceptable risk)
│   │   ├─ Example: Blackjack (σ≈0.95) → 2.5% bankroll bet vs Keno (σ≈2.5) → 0.4% bet
│   │   ├─ Ratio: ~6× difference in bet sizing
│   │   └─ Strategy: Bet size inversely proportional to volatility
│   └─ Portfolio Diversification:
│       ├─ Mixing volatility levels: Combination reduces aggregate volatility
│       ├─ 50% blackjack + 50% roulette: Intermediate volatility
│       ├─ Low vol hedges high vol: Smoother results
│       ├─ Implication: Diverse play = more predictable outcomes
│       └─ Strategy: Allocate bankroll across volatility tiers for stability
├─ IV. VOLATILITY MEASUREMENT:
│   ├─ Historical Volatility (Empirical):
│   │   ├─ Method: Collect n sessions; compute standard deviation
│   │   ├─ Formula: σ = √[Σ(outcome - mean)² / (n-1)]
│   │   ├─ Interpretation: Typical deviation from average outcome
│   │   ├─ Validation: Compare to theoretical volatility
│   │   └─ Challenge: Requires large sample size (100+ sessions)
│   ├─ Theoretical Volatility (Mathematical):
│   │   ├─ Method: From game rules, compute variance
│   │   ├─ Formula: σ² = E[X²] - E[X]²
│   │   ├─ Example (Roulette): σ = 0.9898 for single number bet
│   │   ├─ Accuracy: Exact if rules known; adjusts for skill deviations
│   │   └─ Use: Game comparison, planning
│   ├─ Implied Volatility (Market):
│   │   ├─ Method: Invert from actual bet data / odds
│   │   ├─ Formula: Solve for σ from payout options
│   │   ├─ Example: Sports betting lines imply volatility estimate
│   │   ├─ Interpretation: Market's uncertainty about outcome
│   │   └─ Use: Compare perception vs reality
│   └─ Annualized Volatility:
│       ├─ Definition: σ scaled to annual timeframe
│       ├─ Formula: Annual σ = Session σ × √(sessions/year)
│       ├─ Example: 0.95 per session, 100 sessions/year → 9.5% annual
│       ├─ Interpretation: Total annual outcome variability
│       └─ Use: Long-term planning, career evaluation
├─ V. STRATEGIC IMPLICATIONS:
│   ├─ Game Selection by Volatility Preference:
│   │   ├─ Risk-averse: Choose low volatility (blackjack, baccarat)
│   │   ├─ Risk-seeking: Choose high volatility (slots, progressive games)
│   │   ├─ Balanced: Choose medium volatility (roulette, video poker)
│   │   ├─ Consideration: Volatility + EV combined (high vol + negative EV = worst)
│   │   └─ Decision matrix: Match volatility tolerance to personality
│   ├─ Capital Management:
│   │   ├─ Low vol games: Can risk more per bet
│   │   ├─ High vol games: Must risk less per bet
│   │   ├─ Rebalancing: Increase bets as bankroll grows
│   │   ├─ Stop-loss: Set at -2σ from mean; quit if exceeded
│   │   └─ Scaling: Bankroll sizing follows: B ∝ 1/vol
│   ├─ Session Planning:
│   │   ├─ Duration: Inverse to volatility (low vol = longer sessions)
│   │   ├─ Target: For low vol games, can reach specific loss target (predictable)
│   │   ├─ Quit point: For high vol games, quit at +1σ (capture lucky swings)
│   │   ├─ Recovery: Don't chase losses in high vol; variance will shift
│   │   └─ Variance awareness: Accept session variance as normal
│   ├─ Advantage Seeking:
│   │   ├─ Card counting: Reduces blackjack volatility (concentrated good situations)
│   │   ├─ Position play: Reduces poker volatility (skill advantage over time)
│   │   ├─ Implication: Volatility can be managed via skill
│   │   ├─ Investment: Time spent learning = volatility reduction ROI
│   │   └─ Edge amplification: Low vol + positive EV = best scenario
│   └─ Risk Appetite Calibration:
│       ├─ Conservative: Low vol, low bet size, long duration, focus on capital preservation
│       ├─ Moderate: Medium vol, medium bet size, balanced results
│       ├─ Aggressive: High vol, large bet size, short sessions, accept ruin risk
│       ├─ Implication: Choose volatility matching risk personality
│       └─ Alert: Don't confuse "fun" with "sustainable"; high vol is entertainment
├─ VI. VOLATILITY-SPECIFIC BANKROLL FORMULAS:
│   ├─ General Formula:
│   │   ├─ Ruin probability: RoR ≈ exp(-2 × EV × B / σ²)
│   │   ├─ Rearrange: B ≥ -0.5 × σ² / EV × ln(RoR_target)
│   │   ├─ Example 1 - Low volatility (Blackjack):
│   │   │   ├─ σ = 0.95, EV = -0.005 per $1 bet, Target RoR = 1%
│   │   │   ├─ B ≥ 0.5 × (0.95)² / 0.005 × ln(0.01) = ~875 betting units
│   │   │   └─ Result: Need ~875 units (e.g., $8,750 for $10 bets)
│   │   ├─ Example 2 - High volatility (Keno):
│   │   │   ├─ σ = 3.0, EV = -0.30 per $1 bet, Target RoR = 1%
│   │   │   ├─ B ≥ 0.5 × (3.0)² / 0.30 × ln(0.01) = ~115 betting units
│   │   │   └─ Result: Need ~115 units (e.g., $1,150 for $10 bets); Higher ruin risk!
│   │   └─ Interpretation: Lower EV + higher vol = requires less capital but higher ruin %
│   ├─ Kelly Criterion Adaptation:
│   │   ├─ Standard: f* = EV / σ²
│   │   ├─ Adjustment: f* ∝ 1 / volatility (lower volatility = higher optimal bet %)
│   │   ├─ Example: Blackjack (σ=0.95) → f* = 0.5% vs Slots (σ=1.2) → f* = 0.3%
│   │   └─ Implication: Geometric growth optimal only achievable with skill/edge
│   └─ Multiperiod Planning:
│       ├─ N-session variance: σ_total = σ_single × √N
│       ├─ Cumulative ruin: Increases over time (risk compounds)
│       ├─ Example: Single session ruin 1%, but 100-session ruin 50%+
│       └─ Strategy: Set session limit before reaching true ruin probability
└─ VII. PRACTICAL VOLATILITY TABLES:
    ├─ Quick Reference (σ values per $1 bet):
    │   ├─ Blackjack (basic strat): 0.95
    │   ├─ Roulette (red/black): 0.99
    │   ├─ Baccarat (banker): 0.96
    │   ├─ Craps (pass line): 0.94
    │   ├─ Video Poker (full pay): 1.0
    │   ├─ Slots (typical): 1.1
    │   ├─ Keno: 2.5-3.5
    │   ├─ Poker (no rake): 1.5-5.0 (depends on skill gap)
    │   └─ Sports Betting (sharp): 1.0-1.5
    ├─ Bankroll Sizing Quick Table (1% ruin risk, assuming -1.5% EV):
    │   ├─ Low vol (σ<1): ~600 betting units
    │   ├─ Medium vol (σ=1-1.5): ~400 betting units
    │   ├─ High vol (σ=1.5-3): ~150 betting units
    │   ├─ Extreme vol (σ>3): ~50 betting units
    │   └─ Note: Absolute dollar amounts depend on bet size and EV
    └─ Volatility-Match Bet Sizes (Starting with $100 bankroll):
        ├─ Low vol game: $5-10 per bet (sustainable 10-20 bets)
        ├─ Medium vol game: $3-5 per bet (sustainable 20-30 bets)
        ├─ High vol game: $1-2 per bet (sustainable 50-100 bets)
        ├─ Extreme vol game: $0.50 per bet (sustainable 200 bets)
        └─ Adjustment: Double bet size if bankroll doubles
```

**Core Insight:** Volatility determines capital depletion rate and session length. Higher volatility requires proportionally smaller bets.

## 5. Mini-Project
Classify games by volatility and optimize bankroll allocation:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)

print("="*70)
print("VOLATILITY INDEX: GAME CLASSIFICATION & BANKROLL OPTIMIZATION")
print("="*70)

# ============================================================================
# 1. GAME VOLATILITY DEFINITIONS
# ============================================================================

print("\n" + "="*70)
print("1. VOLATILITY INDEX BY GAME")
print("="*70)

games = {
    'Blackjack': {'sigma': 0.95, 'ev': -0.005, 'category': 'Low'},
    'Roulette EU': {'sigma': 0.99, 'ev': -0.027, 'category': 'Medium'},
    'Baccarat': {'sigma': 0.96, 'ev': -0.011, 'category': 'Low'},
    'Craps': {'sigma': 0.94, 'ev': -0.014, 'category': 'Low'},
    'Slots': {'sigma': 1.15, 'ev': -0.05, 'category': 'Medium-High'},
    'Video Poker': {'sigma': 1.00, 'ev': -0.01, 'category': 'Medium'},
    'Keno': {'sigma': 2.8, 'ev': -0.30, 'category': 'Extreme'},
}

df_volatility = pd.DataFrame(games).T
df_volatility['CV (σ/|EV|)'] = df_volatility['sigma'] / df_volatility['ev'].abs()

print("\nGame Volatility Classification:")
print(df_volatility[['sigma', 'ev', 'category']].to_string())

# ============================================================================
# 2. VOLATILITY CATEGORIES & CHARACTERISTICS
# ============================================================================

print("\n" + "="*70)
print("2. VOLATILITY CATEGORIES")
print("="*70)

volatility_bands = {
    'Low': {'min': 0.0, 'max': 0.5, 'desc': 'Tight outcomes', 'examples': ['Blackjack', 'Baccarat']},
    'Medium': {'min': 0.5, 'max': 1.5, 'desc': 'Moderate swings', 'examples': ['Roulette', 'Video Poker']},
    'High': {'min': 1.5, 'max': 3.0, 'desc': 'Large swings', 'examples': ['Slots', 'Poker']},
    'Extreme': {'min': 3.0, 'max': 10.0, 'desc': 'Wild outcomes', 'examples': ['Keno', 'Lotteries']},
}

print(f"\n{'Category':<12} {'Range':<15} {'Characteristics':<30} {'Duration':>12}")
print("=" * 70)

duration_map = {'Low': 'Very long', 'Medium': 'Long', 'High': 'Short', 'Extreme': 'Very short'}

for cat, data in volatility_bands.items():
    range_str = f"{data['min']:.1f}-{data['max']:.1f}"
    print(f"{cat:<12} {range_str:<15} {data['desc']:<30} {duration_map[cat]:>12}")

# ============================================================================
# 3. RISK OF RUIN BY VOLATILITY
# ============================================================================

print("\n" + "="*70)
print("3. RISK OF RUIN: IMPACT OF VOLATILITY")
print("="*70)

def risk_of_ruin(ev, sigma, bankroll, bet_size=1):
    """
    Approximate RoR formula: RoR ≈ exp(-2 × |EV| × B / σ²)
    """
    if sigma == 0 or ev >= 0:
        return 0
    
    ror = np.exp(-2 * abs(ev) * (bankroll / bet_size) / (sigma**2))
    return min(ror, 1.0)

print("\nRisk of Ruin for 100-unit bankroll, $1 bet:")
print(f"{'Game':<20} {'Volatility':>12} {'RoR %':>12}")
print("-" * 45)

for game, params in games.items():
    ror = risk_of_ruin(params['ev'], params['sigma'], 100, bet_size=1)
    print(f"{game:<20} {params['sigma']:>12.2f} {ror*100:>11.1f}%")

# ============================================================================
# 4. SIMULATED SESSION OUTCOMES BY VOLATILITY
# ============================================================================

print("\n" + "="*70)
print("4. SIMULATED SESSIONS: OUTCOME DISTRIBUTION")
print("="*70)

def simulate_session(sigma, ev, n_bets=100, bet_size=1):
    """
    Simulate a gambling session.
    Outcomes roughly normal with mean EV*bet and std dev sigma*bet.
    """
    outcomes = np.random.normal(loc=ev*bet_size, scale=sigma*bet_size, size=n_bets)
    total_return = np.sum(outcomes)
    return total_return

# Simulate 1000 sessions for each game
print("\nSession Results (1000 simulations, 100 bets, $1 bet each):")
print(f"{'Game':<20} {'Mean':<10} {'Std Dev':>10} {'Min':>10} {'Max':>10} {'Win %':>10}")
print("=" * 70)

session_results = {}

for game, params in games.items():
    results = [simulate_session(params['sigma'], params['ev'], n_bets=100, bet_size=1) 
              for _ in range(1000)]
    
    session_results[game] = results
    mean_return = np.mean(results)
    std_return = np.std(results)
    min_return = np.min(results)
    max_return = np.max(results)
    win_pct = np.sum([r > 0 for r in results]) / len(results) * 100
    
    print(f"{game:<20} {mean_return:>9.2f} {std_return:>10.2f} {min_return:>10.2f} {max_return:>10.2f} {win_pct:>9.1f}%")

# ============================================================================
# 5. OPTIMAL BET SIZING BY VOLATILITY
# ============================================================================

print("\n" + "="*70)
print("5. OPTIMAL BET SIZING: KELLY CRITERION ADJUSTED")
print("="*70)

print("\nOptimal bet size as % of bankroll (1% RoR target):")
print(f"{'Game':<20} {'σ':>8} {'EV':>10} {'Kelly %':>12} {'$ per $100B':>15}")
print("=" * 65)

for game, params in games.items():
    # Simplified Kelly: f* = EV / σ²
    # But need to ensure RoR doesn't exceed target
    kelly_fraction = params['ev'] / (params['sigma']**2) if params['sigma'] > 0 else 0
    kelly_fraction = max(0, min(kelly_fraction, 0.02))  # Cap at 2%
    
    bet_per_100 = kelly_fraction * 100
    
    # Adjust down if RoR still too high
    if risk_of_ruin(params['ev'], params['sigma'], 100, bet_size=bet_per_100) > 0.01:
        kelly_fraction *= 0.5  # Reduce by 50%
        bet_per_100 = kelly_fraction * 100
    
    print(f"{game:<20} {params['sigma']:>8.2f} {params['ev']:>10.4f} {kelly_fraction*100:>11.2f}% {bet_per_100:>14.2f}")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Volatility by game (bar chart)
ax1 = axes[0, 0]
game_names = list(games.keys())
sigmas = [games[g]['sigma'] for g in game_names]
colors = ['green' if s < 0.5 else 'yellow' if s < 1.5 else 'orange' if s < 3 else 'red' 
         for s in sigmas]

ax1.bar(range(len(game_names)), sigmas, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Low cutoff')
ax1.axhline(y=1.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='High cutoff')
ax1.axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Extreme cutoff')
ax1.set_xticks(range(len(game_names)))
ax1.set_xticklabels(game_names, rotation=45, ha='right')
ax1.set_ylabel('Volatility (σ)')
ax1.set_title('Game Volatility Classification')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Session outcome distributions
ax2 = axes[0, 1]

for game in ['Blackjack', 'Slots', 'Keno']:
    results = session_results[game]
    ax2.hist(results, bins=50, alpha=0.5, label=game, edgecolor='black')

ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Session Profit/Loss ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Session Outcome Distributions (100 bets)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Risk of Ruin vs Bankroll
ax3 = axes[1, 0]

bankrolls = np.linspace(10, 500, 50)
ror_low = [risk_of_ruin(games['Blackjack']['ev'], games['Blackjack']['sigma'], b) for b in bankrolls]
ror_med = [risk_of_ruin(games['Slots']['ev'], games['Slots']['sigma'], b) for b in bankrolls]
ror_high = [risk_of_ruin(games['Keno']['ev'], games['Keno']['sigma'], b) for b in bankrolls]

ax3.semilogy(bankrolls, ror_low, linewidth=2, label='Blackjack (low vol)', marker='.')
ax3.semilogy(bankrolls, ror_med, linewidth=2, label='Slots (medium vol)', marker='.')
ax3.semilogy(bankrolls, ror_high, linewidth=2, label='Keno (extreme vol)', marker='.')
ax3.axhline(y=0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='1% target')
ax3.set_xlabel('Bankroll (Units)')
ax3.set_ylabel('Risk of Ruin')
ax3.set_title('RoR vs Bankroll by Volatility')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: Coefficient of Variation (risk per unit EV)
ax4 = axes[1, 1]

cv_values = []
for game in game_names:
    cv = games[game]['sigma'] / abs(games[game]['ev'])
    cv_values.append(cv)

colors_cv = ['green' if cv < 50 else 'yellow' if cv < 150 else 'red' for cv in cv_values]
ax4.bar(range(len(game_names)), cv_values, color=colors_cv, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(game_names)))
ax4.set_xticklabels(game_names, rotation=45, ha='right')
ax4.set_ylabel('CV (σ / |EV|)')
ax4.set_title('Coefficient of Variation: Risk per Unit Loss')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('volatility_index_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: volatility_index_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Low volatility: Steady, predictable losses; longer play duration")
print("✓ High volatility: Wild swings; shorter effective duration; requires smaller bets")
print("✓ RoR scales with σ²: Doubling volatility = 4× ruin risk")
print("✓ Bet sizing inverse to volatility: Higher σ → lower bet %")
print("✓ Volatility category selection = risk tolerance match")
```

## 6. Challenge Round
**When does volatility classification fail?**
- Tournament poker: Volatility extremely high short-term, but skilled players have positive EV (contradicts ruin formula)
- Bankroll-dependent volatility: Same game, different stacks = different effective volatility
- Correlated outcomes: Multi-leg bets create non-independent variance (Keno high volatility justified)
- Skill reduction: Card counting reduces blackjack volatility below theoretical (edge dominates)

## 7. Key References
- [Wizard of Odds - Variance/Volatility Guide](https://www.wizardofodds.com/) - Detailed volatility calculations per game
- [Kelly Criterion & Risk Management](https://en.wikipedia.org/wiki/Kelly_criterion) - Optimal betting with volatility
- [Gambler's Ruin - Classic Formula](https://en.wikipedia.org/wiki/Gambler%27s_ruin) - RoR derivation and scaling with σ²

---
**Status:** Comprehensive classification system for game risk profiles | **Complements:** Variance, Risk of Ruin, Bankroll sizing | **Enables:** Game selection by risk appetite, optimal bet sizing