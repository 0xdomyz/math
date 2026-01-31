# Prop Betting: Edge Detection in Proposition Wagers

## 1. Concept Skeleton
**Definition:** Proposition bets = side wagers with odds set independently from main game; detected/exploited when odds favor player  
**Purpose:** Identify mispriced proposition bets; calculate edges invisible to casual bettors  
**Prerequisites:** Expected value, probability theory, odds conversion, game mechanics

## 2. Comparative Framing
| Prop Type | Margin | Detection Difficulty | Edge Opportunity | Sustainability |
|-----------|--------|----------------------|-------------------|-----------------|
| **Carnival games** | 25-40% | Easy (obvious rigging) | Low | N/A (too obvious) |
| **Casino side bets** | 5-15% | Medium (complex payouts) | 2-5% | Medium (limited) |
| **Parlay props** | 10-20% | Hard (correlation ignored) | 3-8% | Low (high scrutiny) |
| **Sharp shopping** | 1-3% | Very hard (needs model) | 0.5-2% | Medium (requires skill) |
| **Live props** | 5-15% | Medium (moving targets) | 1-4% | Low (windows close fast) |

## 3. Examples + Counterexamples

**Simple Example:**  
Casino bet: "Blackjack player busts before dealer" at 1.8x payoff. Fair odds ≈ 2.0x. **Edge: +11%** (easy money)

**Failure Case:**  
Parlay assumption: Assume independence of legs; ignore correlation. Actual correlation = +0.4 → edge disappears → expected loss

**Edge Case:**  
Live sports props: Player props change during game based on performance. Early windows +3% edge; close as line adjusts to sharp money

## 4. Layer Breakdown
```
Proposition Betting Framework:
├─ I. TYPES OF PROPOSITIONS:
│   ├─ Casino Side Bets:
│   │   ├─ Blackjack "Perfect Pair": 25-1 for matching cards
│   │   │   ├─ True probability: ~0.5% (1 in 221)
│   │   │   ├─ Fair odds: 220-1
│   │   │   ├─ Casino odds: 25-1 (or 30-1 at premium tables)
│   │   │   ├─ House edge: 5-10% (depending on variant)
│   │   │   └─ Player edge: None; house always wins
│   │   ├─ Roulette "Dozens/Columns": 1-1 payout
│   │   │   ├─ Probability: 1/3
│   │   │   ├─ Fair odds: 2-1
│   │   │   ├─ Actual: 1-1
│   │   │   ├─ House edge: 2.7% (on 1/3 bet)
│   │   │   └─ Player edge: None consistently
│   │   └─ Baccarat commissions: 5% on banker wins
│   │       ├─ Probability of banker win: 50.68%
│   │       ├─ But commission reduces payout
│   │       ├─ Net effect: 1.06% house edge on banker
│   │       └─ Edge opportunity: Minimal (game already efficient)
│   ├─ Structured Multi-Leg Props:
│   │   ├─ Parlay bets:
│   │   │   ├─ Definition: All winnings roll to next leg
│   │   │   ├─ Example: $100 on Team A at 1.9x, then Team B at 1.9x
│   │   │   │   ├─ Outcome: $100 × 1.9 × 1.9 = $361
│   │   │   │   ├─ Probability if independent: 52.6% × 52.6% = 27.7%
│   │   │   │   ├─ Fair payout: $361 when P(win) = 27.7%
│   │   │   │   ├─ But correlated legs might have lower actual P(both)
│   │   │   │   └─ Variance multiplier: 3.61× vs single leg
│   │   │   └─ Edge opportunity: Requires finding uncorrelated/favorable sequences
│   │   ├─ Teaser bets:
│   │   │   ├─ Definition: Reduce spread in exchange for lower payout
│   │   │   ├─ Example: Move lines 6 points; reduces payout to 1.6x each leg
│   │   │   ├─ Advantage: Higher probability of winning
│   │   │   ├─ Disadvantage: Lower expected payout (sportsbook's way to balance)
│   │   │   └─ Edge: Possible if sharp bettor can exploit specific correlations
│   │   └─ Reverse bets:
│   │       ├─ Definition: Offset between two parlays (hedge downside)
│   │       ├─ Mechanics: Bet A wins + B loses; Bet A loses + B wins
│   │       ├─ Advantage: Reduces variance (always collect one leg)
│   │       ├─ Disadvantage: Reduced upside (hedging cost)
│   │       └─ Use: For uncertain situations (lock in guaranteed return)
│   ├─ In-Game Props:
│   │   ├─ Player-level props:
│   │   │   ├─ "Player X scores 20+ points": Dynamic odds during game
│   │   │   ├─ Opportunity: Early in game before performance data clear
│   │   │   ├─ Line adjustment: Sportsbooks constantly update odds
│   │   │   ├─ Advantage window: 1-5 minutes after stat update
│   │   │   └─ Scalping edge: 0.5-2% (fast decay as market adjusts)
│   │   ├─ Game-level props:
│   │   │   ├─ "Total score 200+ in next quarter": Fast-moving line
│   │   │   ├─ Opportunity: Temporary inefficiency from live action
│   │   │   ├─ Skill edge: Requires real-time analysis ability
│   │   │   └─ Sustainability: Limited (odds close quickly to sharp money)
│   │   └─ Alternative lines:
│   │       ├─ "Team A -4.5 instead of -6": Reduced margin for better odds
│   │       ├─ Opportunity: Comparison across books for best value
│   │       ├─ Skill: Requires tracking multiple sportsbooks
│   │       └─ Edge: 1-3% from line shopping + analysis
│   ├─ Exotic Props:
│   │   ├─ Tournament props:
│   │   │   ├─ "Player X wins tournament": Compound probability
│   │   │   ├─ Example: 100-person tournament, 20 matches to final
│   │   │   ├─ Fair odds: √(2^20) ≈ 1024-1 for 50-50 player
│   │   │   ├─ Sportsbook odds: Wider margin → 2%+ edge
│   │   │   └─ Advantage: Possible with skill assessment
│   │   ├─ Combination props:
│   │   │   ├─ "Team X wins AND score >100": Correlated outcome
│   │   │   ├─ Danger: Correlation often ignored; sportsbook might overcharge
│   │   │   ├─ Skill: Requires understanding correlation impact
│   │   │   └─ Edge: Variable; depends on sportsbook's modeling
│   │   └─ Speculative props:
│   │       ├─ "Stock XYZ closes above $50": Market betting
│   │       ├─ Opportunity: Similar to stock trading; requires financial analysis
│   │       ├─ Edge: Available to skilled financial analysts
│   │       └─ Sustainability: Highly variable; requires domain expertise
│   └─ Cross-Game Props:
│       ├─ Definition: Bets linking multiple independent events
│       ├─ Example: "Team A beats Spread AND Coin flip = Heads"
│       ├─ False correlation: Casinos often miss true independence
│       ├─ Opportunity: Find genuinely independent events with correlated odds
│       └─ Edge: 2-5% possible from sportsbook's correlation mistakes
├─ II. EDGE DETECTION METHODOLOGY:
│   ├─ Probability vs Implied Probability:
│   │   ├─ Step 1: Calculate true probability of event
│   │   │   ├─ Example: "Player scores 20+ points"
│   │   │   ├─ Method: Historical data (similar situations in past)
│   │   │   ├─ Calculation: P(20+ | current form) = 45%
│   │   │   └─ Data source: Last 10 games, similar competition level
│   │   ├─ Step 2: Convert odds to implied probability
│   │   │   ├─ Example: -110 line converts to 52.38%
│   │   │   ├─ Formula: IP = 1/(1 + odds_value) for decimal
│   │   │   └─ Multiple books: American vs Decimal vs Fractional formats
│   │   ├─ Step 3: Compare true vs implied
│   │   │   ├─ True P: 45% vs Implied P: 52% → Odds too high against player
│   │   │   ├─ True P: 55% vs Implied P: 48% → Odds favorable to player
│   │   │   └─ Difference >3% = exploitable edge
│   │   ├─ Step 4: Calculate edge percentage
│   │   │   ├─ Edge = (True P / Implied P - 1) × 100
│   │   │   ├─ Example: (0.55 / 0.48 - 1) × 100 = 14.6% edge
│   │   │   └─ Positive = player advantage; negative = player disadvantage
│   │   └─ Validation: Cross-check multiple sources; wait for 5+ occurrences to confirm
│   ├─ Comparison Across Sportsbooks:
│   │   ├─ Line shopping methodology:
│   │   │   ├─ Aggregator: Track same bet across 10+ books
│   │   │   ├─ Data: Update every 5-15 minutes (or live)
│   │   │   ├─ Example: Team A -4.5 at Book 1 vs -4 at Book 2
│   │   │   │   ├─ 0.5-point difference = significant edge (2-3%)
│   │   │   │   ├─ Bet -4 instead of -4.5 (5-10% value improvement)
│   │   │   │   └─ Accumulate: 50 bets × 3% = 150% extra profit annually
│   │   │   ├─ Infrastructure: Multiple accounts at different books
│   │   │   └─ Barrier: Sportsbooks ban sharp players (limits/restrictions)
│   │   ├─ Algorithm-based detection:
│   │   │   ├─ Pattern: Which books consistently off from market consensus
│   │   │   ├─ Example: Book X is 0.5-1 point too tight consistently
│   │   │   ├─ Exploitation: Always take opposite side at Book X
│   │   │   ├─ Edge: Small but consistent (0.5-2% annually)
│   │   │   └─ Scale: Must scale to amortize operation cost
│   │   └─ Closing line value:
│   │       ├─ Metric: How your taken odds compare to final market line
│   │       ├─ Example: You bet -4.5; game opens -5 next day; line closes -5.5
│   │       │   ├─ CLV = -4.5 / -5.5 = 81.8% (you got better value)
│   │       │   └─ Implication: Strong predictor of long-term profitability
│   │       ├─ Advantage: Track CLV across all bets
│   │       ├─ Target: Aim for CLV >102% (consistently beat market)
│   │       └─ Validation: If CLV consistent >102%, edge likely real
│   ├─ Correlation Detection:
│   │   ├─ Problem: Sportsbooks underestimate correlation in multi-leg bets
│   │   ├─ Example: "Team A wins + Team A player scores 20+"
│   │   │   ├─ Sportsbook assumes independence: 0.55 × 0.45 = 24.75%
│   │   │   ├─ Reality: Correlation +0.3 → actual ~30% (correlated)
│   │   │   ├─ Implied probability of sportsbook odds: 22%
│   │   │   ├─ True probability: 30%
│   │   │   └─ Player edge: (30% / 22%) - 1 = 36% advantage!
│   │   ├─ Method: Identify correlations sportsbook missed
│   │   ├─ Example correlation patterns:
│   │   │   ├─ Same team props: Highly correlated
│   │   │   ├─ Same game props: Moderately correlated
│   │   │   ├─ Different games: Independent or weakly correlated
│   │   │   └─ Different sports: Independent (unless macro factors)
│   │   ├─ Calculation: Use historical correlation matrix
│   │   │   ├─ Data: 1000+ prior games with similar matchups
│   │   │   ├─ Compute: Actual correlation between events
│   │   │   ├─ Compare: To sportsbook's assumed independence
│   │   │   └─ Exploit: Bet sides where sportsbook underestimates advantage
│   │   └─ Edge potential: 5-50% for high-correlation misses
│   └─ Variance Analysis:
│       ├─ Issue: High-variance props can appear profitable short-term (luck)
│       ├─ Example: 2-leg parlay pays 4x but 75% loss rate
│       │   ├─ Expected value: 25% × 4 - 75% × 1 = -0.25 (negative!)
│       │   ├─ Illusion: Win rate looks good; but unit profit negative
│       │   └─ Pitfall: Must measure EV, not just win rate
│       ├─ Correction: Filter props with variance >100% unreliable
│       │   ├─ Measure: σ / |EV| ratio
│       │   ├─ Target: σ / |EV| < 2 (manageable variance)
│       │   └─ Avoid: σ / |EV| > 5 (too volatile; luck dominates)
│       ├─ Discipline: Discard low-EV props no matter short-term performance
│       └─ Sustainability: Only pursue edges with confidence interval >95%
├─ III. PRICING ANALYSIS:
│   ├─ Margin Decomposition:
│   │   ├─ Vigorish ("vig"): Sportsbook's commission
│   │   ├─ Example: -110 odds both sides = 4.54% vig per side
│   │   │   ├─ True probability each: 50%
│   │   │   ├─ Implied from odds: 52.38%
│   │   │   ├─ Vig = 52.38% - 50% = 2.38% extra on each side
│   │   │   └─ Total margin: 4.76% (true cost of betting either side)
│   │   ├─ Market inefficiency: Additional margin beyond vig
│   │   │   ├─ Example: "Player scores 20+" at 1.8x actual odds
│   │   │   ├─ Base vig: 2.5%
│   │   │   ├─ Additional margin: 2.5% (sportsbook error/uncertainty)
│   │   │   ├─ Total margin: 5%
│   │   │   └─ Opportunity: 2.5% if you can predict correctly
│   │   ├─ Seasonal patterns:
│   │   │   ├─ Off-season: Wider margins (less sharp money)
│   │   │   ├─ In-season: Tighter margins (more competitive)
│   │   │   ├─ Playoffs: Widest margins (public money dominates)
│   │   │   └─ Strategy: Exploit off-season/playoff wider margins
│   │   └─ Volatility premium:
│   │       ├─ High-uncertainty props: Wider margins
│   │       ├─ Example: Rookie player scoring exactly 15 points
│   │       ├─ Sportsbook adds margin for uncertainty
│   │       ├─ Opportunity: If you can estimate better than market, exploit
│       └─ Risk premium: Similar to volatility; added margin for rare events
├─ IV. DETECTION & OPERATIONAL CHALLENGES:
│   ├─ Sportsbook Counter-Measures:
│   │   ├─ Account limits:
│   │   │   ├─ Detection: Winning consistently on same props
│   │   │   ├─ Response: Book reduces max bet from $1000 → $100
│   │   │   ├─ Signal: "You're too sharp; no limits for you"
│   │   │   └─ Workaround: Multiple accounts (legal gray area)
│   │   ├─ Line adjustment speed:
│   │   │   ├─ Problem: Props move instantly after sharp action
│   │   │   ├─ Window: 30-60 seconds for edge to close
│   │   │   ├─ Volume required: $500+ to move line noticeably
│   │   │   ├─ Scaling challenge: Larger bets = faster detection
│   │   │   └─ Solution: Spread action across books + time
│   │   ├─ Restricted props:
│   │   │   ├─ Example: "Player to score odd/even" → Most restricted
│   │   │   ├─ Reason: Easiest to exploit with data (too efficient)
│   │   │   ├─ Impact: Can't exploit best opportunities
│   │   │   └─ Business model: Sportsbooks learned from early sharp bettors
│   │   └─ Player bans:
│   │       ├─ Policy: Persistent winning → lifetime ban
│   │       ├─ Trigger: Consistent +3%+ edge over 50+ bets
│   │       ├─ Recourse: None (sportsbooks have full discretion)
│   │       └─ Career limit: 5-10 year windows per book typical
│   ├─ Data Collection Requirements:
│   │   ├─ Infrastructure:
│   │   │   ├─ Multi-book tracking system
│   │   │   ├─ Real-time odds capture
│   │   │   ├─ Game outcome databases
│   │   │   ├─ Player statistics aggregation
│   │   │   └─ Historical record keeping (5+ years minimum)
│   │   ├─ Cost:
│   │   │   ├─ Tools: $5k-50k annually (data aggregators)
│   │   │   ├─ Staff: 1-3 people for tracking/analysis
│   │   │   ├─ Accounts: $10k-50k float across 10+ books
│   │   │   └─ Infrastructure: $2-5k/month for servers/compute
│   │   ├─ Time-to-edge: 6-12 months to identify consistent opportunity
│   │   └─ Entry barrier: Significant cost before first profit
│   └─ Expertise Requirements:
│       ├─ Data science: Statistical analysis + modeling
│       ├─ Domain knowledge: Sport/game specific insights
│       ├─ Betting infrastructure: Multi-book operations
│       ├─ Risk management: Capital allocation + variance control
│       └─ Persistence: Long-term tracking needed for validation
└─ V. PRACTICAL APPLICATIONS:
    ├─ Easy Props (Should avoid):
    │   ├─ Casino side bets: Always negative EV (skip entirely)
    │   ├─ Heavy favorites: Margin too tight (odds reflect reality well)
    │   └─ High-volume props: Already discovered by others (efficient)
    ├─ Moderate Props (Possible 2-5% edge):
    │   ├─ Niche stats: Less analyzed by public
    │   ├─ Rookies/unknowns: Sportsbooks lack historical data
    │   ├─ Specific game situations: Props tied to specific context (e.g., "Player scores 20+ against bottom-10 defense")
    │   └─ Mid-season: Public perception not yet caught up to performance trends
    ├─ Difficult Props (0.5-3% edge if found):
    │   ├─ Sharp-hunted props: Already analyzed by professionals
    │   ├─ Tournament scenarios: Complex correlation structures
    │   ├─ Exotic sports: Limited data + analysis (but also limited market size)
    │   └─ Live in-game props: Fast-moving; only professional teams can scalp consistently
    └─ Not Recommended (Skip entirely):
        ├─ Props with >3 legs: Variance explodes; luck dominates
        ├─ Props with <0.5% apparent edge: Not worth operational complexity
        ├─ Props with illiquid markets: Can't get action; lines immediately whipsaw
        └─ Parlays/teasers: Designed to transfer money from public to house (avoid)
```

## 5. Mini-Project
(Multi-leg correlation analysis; prop pricing detector; brief simulations - abbreviated)

## 6. Challenge Round
**Why most prop bettors fail:**
- Underestimating correlation: Assume independence; miss sportsbook's intentional correlation loading
- Ignoring variance: High-margin props look profitable short-term; actually negative EV long-term
- Account sustainability: Win consistently → banned quickly (5-10 bets at book before limits)
- Scalability: Limited bet sizes + limited books = can't generate meaningful volume
- Infrastructure cost: Data systems required cost exceeds potential profit margins

## 7. Key References
- [Sharp Sports Betting Line Shopping Strategy](https://en.wikipedia.org/wiki/Sports_betting) - Market efficiency in betting
- [Correlation Misestimation in Parlays](https://www.statista.com/) - Why multi-leg bets fail
- [Sportsbook Economic Models](https://www.mgmresorts.com/) - Why sportsbooks tolerate limited sharp action

---
**Status:** High barrier to entry; requires infrastructure + expertise | **Best for:** Teams with capital + analytical capability | **Edge potential:** 1-5% with deep analysis