# Bankroll Requirements for Advantage Play: Capital Sizing & Risk Management

## 1. Concept Skeleton
**Definition:** Minimum capital needed to survive variance while exploiting player-edge opportunities; prevents ruin from unlucky streaks  
**Purpose:** Calculate adequate bankroll to sustain advantage play career; maintain 1-2% risk of ruin despite negative variance sequences  
**Prerequisites:** Risk of ruin formula, variance calculations, edge estimation, expected value concepts

## 2. Comparative Framing
| Strategy | Kelly Criterion | 1% RoR | 2% RoR | Conservative |
|----------|-----------------|--------|--------|--------------|
| **Bankroll Multiplier** | 100-150× bet | 200× bet | 150× bet | 300+ × bet |
| **Risk Profile** | Aggressive | Moderate | Moderate | Safe |
| **Ruin Probability** | 5-10% | 1% | 2% | <0.5% |
| **Volatility** | High swings | Moderate | Moderate | Smooth |
| **Recovery Speed** | Fast if winning | Steady | Steady | Conservative |
| **Recommended For** | Teams/experts | Professionals | Professionals | Casual players |

## 3. Examples + Counterexamples

**Simple Example:**  
Blackjack counter with +0.75% edge, σ=0.95: For 1% RoR with $10 bets → B ≈ $20,000 needed

**Failure Case:**  
Underestimating variance: You have +0.75% edge, $3000 bankroll, $10 bets → 45% RoR (dangerous)  
You can expect 100+ hand losing streak on bad variance; bankroll inadequate

**Edge Case:**  
Team play with big player: Team bankroll $100k supports multiple big players simultaneously; individual risk spreads

## 4. Layer Breakdown
```
Bankroll Framework:
├─ I. FORMULA & CALCULATIONS:
│   ├─ Gambler's Ruin Formula:
│   │   ├─ RoR ≈ exp(-2 × |EV| × B / σ²)
│   │   ├─ Variables:
│   │   │   ├─ EV: Player edge (e.g., 0.0075 for 0.75%)
│   │   │   ├─ B: Bankroll in betting units
│   │   │   ├─ σ: Standard deviation of outcome per bet
│   │   │   └─ RoR: Probability of reaching $0 before target
│   │   ├─ Rearrange: B ≥ -(σ² / (2 × EV)) × ln(RoR_target)
│   │   ├─ Example: σ=0.95, EV=0.0075, RoR_target=1%
│   │   │   └─ B ≥ -(0.95² / (2 × 0.0075)) × ln(0.01) = ~1,829 units
│   │   ├─ Practical: For $10 bet → $18,290 bankroll for 1% RoR
│   │   └─ Validation: Rough approximation; actual depends on exact game
│   ├─ Volatility Scaling:
│   │   ├─ High variance → more bankroll needed
│   │   ├─ Low variance → less bankroll needed
│   │   ├─ Example: Poker (σ=3) vs Blackjack (σ=0.95)
│   │   │   ├─ Blackjack: B ≈ 1,829 units
│   │   │   ├─ Poker: B ≈ 19,600 units (10× more!)
│   │   │   └─ Reason: σ² factor dominates RoR formula
│   │   └─ Implication: High-variance games require larger bankroll
│   ├─ Edge Sensitivity:
│   │   ├─ EV appears as denominator in ruin formula
│   │   ├─ Small edge → requires massive bankroll
│   │   ├─ Example: 0.5% edge vs 1.5% edge
│   │   │   ├─ 0.5% edge: B ≈ 5,486 units
│   │   │   ├─ 1.5% edge: B ≈ 1,829 units
│   │   │   └─ 3× edge difference → 3× bankroll reduction
│   │   └─ Implication: Seek highest-edge games; bankroll scales inversely
│   └─ Risk Target Selection:
│       ├─ 0.5% RoR: Very conservative; longest career sustainability
│       ├─ 1% RoR: Industry standard for professionals
│       ├─ 2% RoR: Acceptable for teams with backup capital
│       ├─ 5%+ RoR: Risky; only for aggressive teams
│       └─ Trade-off: Lower RoR = larger bankroll needed
├─ II. GAME-SPECIFIC BANKROLL REQUIREMENTS:
│   ├─ Blackjack Card Counting:
│   │   ├─ Edge: 0.75% (good counter, favorable rules)
│   │   ├─ Variance: σ ≈ 0.95
│   │   ├─ RoR 1%: ≈ 200-250× minimum bet
│   │   ├─ Example: $10 minimum bet → $2000-2500 comfortable
│   │   ├─ Reality: Most counters use $5000-20000 (provides buffer)
│   │   ├─ Reasoning: Variance shocks; downswings >$1000 possible
│   │   └─ Professional standard: 500-1000× minimum bet
│   ├─ Shuffle Tracking:
│   │   ├─ Edge: 0.5-1.5% (overlaps with card counting uncertainty)
│   │   ├─ Variance: Higher than counting (σ ≈ 1.1)
│   │   ├─ RoR 1%: ≈ 300-400× minimum bet
│   │   ├─ Example: $10 bet → $3000-4000 bankroll
│   │   ├─ Reality: Less predictable than counting; larger buffer needed
│   │   └─ Team requirement: $50,000+ for sustained operation
│   ├─ Poker (by skill edge):
│   │   ├─ Casual skill (+1% edge): 300-500× buy-in
│   │   ├─ Strong skill (+2% edge): 200-400× buy-in
│   │   ├─ Professional (+3%+ edge): 150-300× buy-in
│   │   ├─ Example: $50 buy-in, +2% edge → $10,000-20000 bankroll
│   │   ├─ Variance: High (σ ≈ 2-4); large swings expected
│   │   └─ Professionals: Often manage $50,000-200,000 bankroll
│   ├─ Sports Betting:
│   │   ├─ Sharp edge: +2-3% (requires analytical model)
│   │   ├─ Casual edge: +0.5-1% (line shopping, basic analysis)
│   │   ├─ RoR 1%: ≈ 100-200× average bet
│   │   ├─ Example: $100 average bet, +2% edge → $10,000-20000
│   │   ├─ Variance: Moderate (σ ≈ 1-1.5) when diversified
│   │   └─ Scaling: Small bets early; increase as bankroll grows
│   └─ Video Poker:
│       ├─ Edge: +0.5-1.5% (with perfect play, good pay table)
│       ├─ Variance: Medium-high (σ ≈ 1.2)
│       ├─ RoR 1%: ≈ 250-350× average bet
│       ├─ Example: $25 average bet → $6000-9000 bankroll
│       └─ Advantage: Can earn while playing; no strategy required
├─ III. DOWNSWING SCENARIOS:
│   ├─ Law of Large Numbers:
│   │   ├─ Concept: Results converge to EV as n→∞, but slow
│   │   ├─ Example: +0.75% edge requires ~5000+ hands to be confident edge is real
│   │   ├─ Convergence: With 60 hands/hour, ~85 hours for reasonable confidence
│   │   ├─ Variance before: ±$500 swings easily possible with $10 bets
│   │   └─ Implication: Expect downswings; bankroll must absorb
│   ├─ Downswing Magnitude:
│   │   ├─ 1σ downswing: ~16% probability; -σ dollars
│   │   │   ├─ Example: -$1200 on blackjack session (bad luck, not edge loss)
│   │   │   └─ Recovery: Requires ~1-2 winning sessions
│   │   ├─ 2σ downswing: ~2.3% probability; -2σ dollars
│   │   │   ├─ Example: -$2400 losing streak
│   │   │   └─ Recovery: Requires ~3-5 winning sessions
│   │   ├─ 3σ downswing: ~0.1% probability; -3σ dollars
│   │   │   ├─ Example: -$3600 catastrophic loss
│   │   │   └─ Recovery: Requires ~7-10 winning sessions
│   │   └─ Implication: Bankroll must cover 2-3σ downswings comfortably
│   ├─ Multi-Session Ruin:
│   │   ├─ Consecutive losing sessions: Possible with variance
│   │   ├─ Streak probability: P(lose n sessions) = (1-win_rate)^n
│   │   ├─ Example: 55% win rate (good edge), P(lose 5 straight) ≈ 2%
│   │   ├─ Magnitude: 5 × ($500 loss) = $2500 gone if underfunded
│   │   └─ Bankroll needs to survive 3-5 losing sessions
│   └─ Downswing Duration:
│       ├─ Variance decay: σ/√n; variance reduces with hands played
│       ├─ Example: After 100 hands, σ_cumulative ≈ $95 (vs $95 per hand)
│       ├─ Implication: Extended play smooths results; lucky/unlucky streaks less likely
│       ├─ Session focus: Longer sessions better for advantage players
│       └─ Bankroll scaling: More hours played → less relative bankroll needed
├─ IV. BANKROLL GROWTH STRATEGY:
│   ├─ Conservative Growth:
│   │   ├─ Initial: Build to 300-500× minimum bet
│   │   ├─ Threshold 1: At $50k → increase bet size 1.5×
│   │   ├─ Threshold 2: At $100k → increase bet size 1.5×
│   │   ├─ Pattern: Geometric growth; bet size scales with capital
│   │   ├─ Advantage: Prevents ruin while capturing increasing returns
│   │   └─ Timescale: 2-5 years to scale from $5k to $100k
│   ├─ Kelly Criterion Approach:
│   │   ├─ Formula: f* = EV / σ²
│   │   ├─ Example: +0.75% edge, σ=0.95 → f* = 0.83% of bankroll per bet
│   │   ├─ Meaning: Bet 0.83% of capital per hand (maximizes geometric growth)
│   │   ├─ Advantage: Optimal growth rate; mathematically proven
│   │   ├─ Risk: Ruin probability = 0 (asymptotic safety)
│   │   └─ Drawback: Bet sizes fluctuate with bankroll (suspicious to casino)
│   ├─ Fractional Kelly:
│   │   ├─ Conservative: Bet 1/2 or 1/4 of Kelly
│   │   ├─ 1/2 Kelly: f* = 0.41% of bankroll (safer, slower growth)
│   │   ├─ Advantage: Reduced volatility; easier to sustain
│   │   ├─ Application: More practical for advantage players (less detection)
│   │   └─ Growth: Linear instead of geometric; steadier career
│   ├─ Bankroll Allocation:
│   │   ├─ Separate accounts: Playing bankroll vs personal savings
│   │   ├─ Discipline: Only play with designated advantage capital
│   │   ├─ Allocation: Diversify if multi-game (poker, blackjack, sports)
│   │   ├─ Flexibility: Allow reallocation between games based on opportunities
│   │   └─ Exit plan: Withdraw winnings; maintain core capital
│   └─ Downswing Recovery:
│       ├─ Conservative rule: If bankroll drops 30%, reduce bet size 50%
│       ├─ Rationale: Rebuilding requires smaller, lower-variance play
│       ├─ Threshold: If drop 50%, stop play; reassess strategy
│       ├─ Return: Only restart with full bankroll rebuilt
│       └─ Psychological: Prevents desperation betting (dangerous)
├─ V. TEAM BANKROLL DYNAMICS:
│   ├─ Team Pooling:
│   │   ├─ Combined capital: Team members pool funds
│   │   ├─ Advantage: Larger bankroll supports bigger bets
│   │   ├─ Edge: Multiple players = more hands; faster edge realization
│   │   ├─ Risk: All-or-nothing; if caught, entire bankroll at risk
│   │   └─ Profit sharing: Typically % of winnings or equal split
│   ├─ Role-Based Capital:
│   │   ├─ Big player: Needs largest bankroll (makes biggest bets)
│   │   ├─ Spotters: Smaller individual bankroll (low bets)
│   │   ├─ Coordinator: No bankroll risk (only manages)
│   │   ├─ Total: Sum of all individual requirements
│   │   └─ Scaling: Team bankroll = 5-10× individual player capital
│   ├─ Variance Reduction:
│   │   ├─ Benefit: Multiple players reduce aggregate variance
│   │   ├─ Math: Combined σ = √(individual σ²)
│   │   ├─ Example: 5 players with $5k each vs 1 with $25k
│   │   │   ├─ Team: Lower combined risk (variance spreads)
│   │   │   └─ Single: Higher risk (concentrated)
│   │   ├─ Implication: Teams can achieve better RoR with same capital
│   │   └─ Structure: Optimal team size 3-5 players
│   └─ Capital Redeployment:
│       ├─ Winning teams: May expand to new locations
│       ├─ Losing teams: May need to reduce scope
│       ├─ Flexibility: Reallocate capital to highest-edge opportunities
│       └─ Dynamic: Bankroll grows/shrinks with performance
├─ VI. RELATIONSHIP TO EDGE & SESSION LENGTH:
│   ├─ Edge-Bankroll Trade-off:
│   │   ├─ Higher edge → smaller bankroll needed
│   │   ├─ Example: +2% edge needs ~50% less capital than +0.75% edge
│   │   ├─ Implication: Seek games where you have skill advantage
│   │   └─ Decision: Play highest-edge games available
│   ├─ Session Length Impact:
│   │   ├─ Longer sessions → variance realizes more fully
│   │   ├─ Formula: RoR improves with √n (n = hands played)
│   │   ├─ Example: 100-hand session vs 500-hand session
│   │   │   ├─ 100 hands: Higher relative variance
│   │   │   ├─ 500 hands: Lower relative variance (4% improvement)
│   │   │   └─ Implication: Longer sessions reduce bankroll requirement
│   │   ├─ Time horizon: Lifetime play requires largest bankroll
│   │   └─ Planning: Professional careers need $50k-$200k+
│   └─ Seasonal Variation:
│       ├─ Opportunity windows: Better/worse games vary seasonally
│       ├─ Capital flexibility: Keep extra capital for high-edge opportunities
│       ├─ Deployment: Use more bankroll when edge highest
│       └─ Prudence: Never risk full bankroll on single opportunity
└─ VII. PRACTICAL BANKROLL PLANNING:
    ├─ Conservative Recommendation:
    │   ├─ Starting bankroll: 500-1000× minimum bet
    │   ├─ Example: $15 bets → $7,500-15,000 starting capital
    │   ├─ Safety margin: Accounts for conservative RoR estimation
    │   ├─ Rationale: Professional-grade sustainability
    │   └─ Alternative: Start with 250× if highly confident in edge
    ├─ Scaling Schedule:
    │   ├─ First $10k: Minimum bets ($5-15); build foundation
    │   ├─ $10-50k: Medium bets ($25-50); comfortable growth
    │   ├─ $50-100k: Large bets ($50-100); professional operations
    │   ├─ $100k+: Scalable operations; team-based potential
    │   └─ Timeline: 2-5 years to reach $100k (with good edge)
    ├─ Risk Management Checkpoints:
    │   ├─ Monthly: Review variance; confirm edge assumptions
    │   ├─ Quarterly: Rebalance bet sizes; adjust to capital changes
    │   ├─ Annually: Full strategy review; adapt to casino changes
    │   ├─ Career exit: Plan transition before burnout/detection
    │   └─ Flexibility: Adapt to changing opportunities
    └─ Emergency Protocols:
        ├─ Rapid drawdown: If lose 40%+ of bankroll, pause operations
        ├─ Investigation: Reassess edge assumptions; verify not playing -EV
        ├─ Recovery plan: Reduced stakes; build back to threshold
        ├─ Abandonment: If cannot recover, exit advantage play career
        └─ Lesson: Document what went wrong for future improvement
```

**Core Insight:** Bankroll = critical factor determining career sustainability; inadequate capital causes ruin despite positive edge through bad variance alone.

## 5. Mini-Project
(Python code for bankroll calculator and downswing simulation - abbreviated for space)

Calculate required bankroll for various advantage scenarios and simulate downswing scenarios demonstrating importance of adequate capital.

## 6. Challenge Round
**Bankroll management failures:**
- Underfunding: Positive edge but only $2000 bankroll → 20% RoR easily
- Overleveraging: Using 50%+ of net worth for gambling (personal financial ruin if variance)
- Ignoring variance: Assuming +1% edge = +1% returns every session (wrong; variance dominates short-term)
- No growth plan: Bankroll stagnates; never scales to optimal bet sizes
- Emotional betting: After losing streak, size bets up (desperate) instead of down

## 7. Key References
- [Gambler's Ruin Formula Derivation](https://en.wikipedia.org/wiki/Gambler%27s_ruin) - Mathematical foundation
- [Kelly Criterion Bankroll Optimization](https://en.wikipedia.org/wiki/Kelly_criterion) - Optimal growth strategies
- [MIT Blackjack Team - Bankroll Management](https://en.wikipedia.org/wiki/MIT_Blackjack_Team) - Real-world team capital sizing

---
**Status:** Essential infrastructure for advantage play viability | **Complements:** Risk of ruin, Edge quantification, Career planning | **Enables:** Sustainable professional gambling careers with acceptable ruin risk