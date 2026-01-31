# Gambling & Probability Topics Guide

**Complete reference of gambling probability concepts, game analysis, betting strategies, and risk management with categories, descriptions, and sources.**

---

## I. Probability Foundations for Gambling

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Basic Probability** | ✓ basic_probability.md | P(A) = favorable outcomes / total outcomes; applied to gambling games | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library) |
| **Conditional Probability** | ✓ conditional_probability.md | P(A\|B) given previous event; crucial for card counting, sequential games | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library) |
| **Independence & Dependence** | ✓ independence_dependence.md | Independent events (coin flips) vs dependent (card draws without replacement) | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library) |
| **Randomness & Fairness** | ✓ randomness_fairness.md | True randomness vs pseudorandom; fair vs biased outcomes; entropy | [Wiki - Randomness](https://en.wikipedia.org/wiki/Randomness) |
| **Combinatorics & Counting** | ✓ combinatorics_gambling.md | Permutations (order matters), Combinations (order irrelevant); poker hands | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/counting-permutations-and-combinations) |
| **Law of Large Numbers** | ✓ law_of_large_numbers.md | Empirical frequency → theoretical probability as n→∞; gambling convergence | [Wiki - LLN](https://en.wikipedia.org/wiki/Law_of_large_numbers) |
| **House Edge Foundation** | ✓ house_edge.md | Casino advantage; expected value negative for all players | [Wiki - House Edge](https://en.wikipedia.org/wiki/House_edge) |

---

## II. Expected Value & Risk Metrics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Expected Value (EV)** | ✓ expected_value.md | E[X] = Σ(outcome × probability); core gambling metric | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/expected-value-library) |
| **Variance & Volatility** | ✓ variance_volatility.md | Spread of outcomes; high variance = swing outcomes, low variance = stable | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) |
| **Standard Deviation** | ✓ variance_volatility.md | Square root of variance; measures risk; σ larger = wilder swings | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) |
| **Payout Ratios** | ✓ payout_ratios.md | Return per $ wagered; includes house edge; critical for game comparison | [Casino Encyclopedia](https://www.casinopedia.org) |
| **Return to Player (RTP)** | ✓ return_to_player.md | % of wagered money returned to players long-term; inverse of house edge | [UK Gambling Commission](https://www.gamblingcommission.gov.uk) |
| **Risk of Ruin** | ✓ risk_of_ruin.md | Probability of losing entire bankroll before target gain | [Wiki - Gambler's Ruin](https://en.wikipedia.org/wiki/Gambler%27s_ruin) |
| **Volatility Index (σ)** | ✓ volatility_index.md | Game volatility classification; low (<1) to high (>5) | [Gaming Theory](https://en.wikipedia.org/wiki/Volatility_index) |

---

## III. Game-Specific Probabilities

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Roulette Probabilities** | ✓ roulette_probabilities.md | Single 0 (European: 2.7% HE) vs double 0 (American: 5.4% HE); streak analysis | [Wiki - Roulette](https://en.wikipedia.org/wiki/Roulette) |
| **Blackjack Odds** | ✓ blackjack_odds.md | 21 bust probabilities, dealer stand/hit rules, basic strategy EV | [Wizard of Odds](https://www.wizardofodds.com/games/blackjack/) |
| **Poker Probabilities** | ✓ poker_probabilities.md | Hand rankings, odds to draw winning hand, pot odds vs draw odds | [Pokerstove](http://www.pokerstove.com/) |
| **Craps Probabilities** | ✓ craps_probabilities.md | Pass/Don't Pass, come/don't come, place bets, point mechanics | [Wizard of Odds](https://www.wizardofodds.com/games/craps/) |
| **Slot Machine Odds** | ✓ slot_machine_odds.md | RNG (Random Number Generator) mechanics, payout percentages, hold % | [Nevada Gaming Control Board](https://gaming.nv.gov/) |
| **Baccarat House Edge** | ✓ baccarat_house_edge.md | Banker (1.06% HE) vs Player (1.24% HE) vs Tie (14.36% HE) | [Wizard of Odds](https://www.wizardofodds.com/games/baccarat/) |
| **Keno House Edge** | ✓ keno_house_edge.md | 25-40% HE depending on selection; among worst casino games | [Wizard of Odds](https://www.wizardofodds.com/games/keno/) |
| **Video Poker Payout Tables** | ✓ video_poker_payout_tables.md | Paytable variations dramatically affect RTP (95%-99%+); play max coins | [Video Poker Strategy](https://www.videopokerforever.com/) |

---

## IV. Combinatorics & Game Analysis

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Combinatorics in Gambling** | ✓ combinatorics_gambling.md | Calculate hand counts, odds to draw specific outcomes, total game states | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/counting-permutations-and-combinations) |
| **Poker Hand Analysis** | ✓ poker_probabilities.md | Royal flush (1 in 649,740), straight flush, four-of-a-kind odds | [Pokerstove](http://www.pokerstove.com/) |
| **Texas Hold'em Equity Calculation** | ✓ texas_holdem_equity_calculation.md | Hole cards + community cards → win probability vs range; equity distribution | [PokerIndicators](https://pokerindicators.com/) |
| **Drawing Probabilities** | ✓ poker_probabilities.md | Outs (cards improving hand), % probability hand improves on river | [Pokermath](https://www.pokermath.com/) |
| **Permutations vs Combinations** | ✓ combinatorics_gambling.md | Order matters (seating arrangements) vs doesn't (hand composition) | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/counting-permutations-and-combinations) |

---

## V. Betting Strategies & Bankroll Management

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Kelly Criterion** | ✓ kelly_criterion.md | Optimal bet sizing: f* = (bp - q) / b; maximizes growth log utility | [Wiki - Kelly](https://en.wikipedia.org/wiki/Kelly_criterion) |
| **Fractional Kelly** | ✓ kelly_criterion.md | Conservative variant: bet 1/2 or 1/4 of Kelly; reduces risk of ruin | [Thorp, Edward O. (1962)](https://en.wikipedia.org/wiki/Edward_Thorp) |
| **Bankroll Management** | ✓ bankroll_management.md | Preserve capital; unit sizing, stop-loss limits, session budgets | [Advantage Player Handbook](https://www.wizardofodds.com/gambling/) |
| **Risk of Ruin** | ✓ risk_of_ruin.md | P(loss all capital); depends on edge, variance, session length | [Wiki - Gambler's Ruin](https://en.wikipedia.org/wiki/Gambler%27s_ruin) |
| **Martingale System** | ✓ martingale_system.md | Double bet after each loss; guaranteed win but unbounded risk | [Wiki - Martingale](https://en.wikipedia.org/wiki/Martingale_(betting_system)) |
| **Fibonacci System** | ✓ fibonacci_system.md | Progressive betting 1,1,2,3,5,8,...; less aggressive than Martingale | [Fibonacci Gambling](https://www.gambling.com/fibonacci-system) |
| **Unit Sizing** | ✓ bankroll_management.md | Wager K% of bankroll or fixed units (e.g., 1-5% per bet) | [Advantage Player Handbook](https://www.wizardofodds.com/gambling/) |
| **Heat & Table Selection** | ✓ heat_table_selection.md | Running hot (+ expected) vs cold (- from expected); game choice | [Counter Culture](https://www.countercultureproductions.com/) |

---

## VI. Game Theory & Strategic Depth

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Game Theory in Gambling** | ✓ game_theory_gambling.md | Nash equilibrium, mixed strategies, zero-sum games, exploitability | [Wiki - Game Theory](https://en.wikipedia.org/wiki/Game_theory) |
| **Zero-Sum Games** | ✓ game_theory_gambling.md | One player's gain = another's loss; poker, roulette, most casino games | [Wiki - Zero-Sum](https://en.wikipedia.org/wiki/Zero-sum_game) |
| **Nash Equilibrium** | ✓ game_theory_gambling.md | Optimal mixed strategy where no player benefits from unilateral deviation | [Wiki - Nash Equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) |
| **Mixed Strategies vs Pure Strategies** | ✓ game_theory_gambling.md | Random action selection (mixed) vs deterministic (pure); poker bluffing | [Wiki - Mixed Strategy](https://en.wikipedia.org/wiki/Strategy_(game_theory)) |
| **Exploitability & Exploitation** | ✓ exploitability_exploitation.md | Deviations from Nash allow +EV against weak opponents | [Game Theory Optimal (GTO)](https://en.wikipedia.org/wiki/Game_theory_optimal) |
| **Pot Odds vs Drawing Odds** | ✓ poker_probabilities.md | Compare pot payout (Pot / bet required) vs hand draw probability | [Pokermath](https://www.pokermath.com/) |
| **Position & Information** | ✓ position_information.md | Later positions have more info; information asymmetry advantage | [Poker Strategy](https://www.pokerstrategy.com/) |

---

## VII. Behavioral & Psychological Factors

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Gambler's Fallacy** | ✓ gamblers_fallacy.md | Belief past results predict future; roulette black after red streak | [Wiki - Gambler's Fallacy](https://en.wikipedia.org/wiki/Gambler%27s_fallacy) |
| **Hot-Hand Fallacy** | ✓ hot_hand_fallacy.md | Belief winning streak will continue; opposite of gambler's fallacy | [Wiki - Hot Hand](https://en.wikipedia.org/wiki/Hot-hand_fallacy) |
| **Loss Aversion** | ✓ addiction_responsible_gambling.md | Fear of loss > pleasure of equivalent gain; risk-averse behavior | [Kahneman & Tversky](https://en.wikipedia.org/wiki/Loss_aversion) |
| **Sunk Cost Fallacy** | ✓ sunk_cost_fallacy.md | Continuing play to recover losses; chasing losses | [Wiki - Sunk Costs](https://en.wikipedia.org/wiki/Sunk_cost) |
| **Illusion of Control** | ✓ illusion_of_control.md | Belief one can influence random outcomes (skill in luck-based games) | [Wiki - Illusion of Control](https://en.wikipedia.org/wiki/Illusion_of_control) |
| **Addiction Pathways** | ✓ addiction_responsible_gambling.md | Variable reward schedules (slot machines), escape motivation, tolerance buildup | [DSM-5 Gambling Disorder](https://en.wikipedia.org/wiki/Gambling_disorder) |
| **Casino Design & Player Behavior** | ✓ casino_design_player_behavior.md | Architecture, lighting, sound design influence spending and time on floor | [Casino Design Psychology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2963534/) |
| **Responsible Gambling** | ✓ addiction_responsible_gambling.md | Self-exclusion, setting limits, treatment resources, harm reduction | [NCPG National Council on Problem Gambling](https://www.ncpg.org/) |

---

## VIII. House Edge & Economic Analysis

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **House Edge Definition** | ✓ house_edge.md | Casino's mathematical advantage; results in expected loss for players | [Wiki - House Edge](https://en.wikipedia.org/wiki/House_edge) |
| **House Edge by Game** | ✓ house_edge.md | Blackjack (0.5-4%), Craps (1.4%), Roulette (2.7-5.4%), Slots (5-15%) | [Wizard of Odds](https://www.wizardofodds.com/games/) |
| **Expected Loss Calculation** | ✓ expected_value.md | E[Loss] = Bet Size × House Edge; unavoidable for players | [Wizard of Odds](https://www.wizardofodds.com/gambling/) |
| **Rake in Poker** | ✓ rake_in_poker.md | Casino fee; 4-10% of pot; affects game profitability | [Poker Math](https://www.pokermath.com/) |
| **Commission in Baccarat** | ✓ commission_in_baccarat.md | 5% charge on Banker wins; built into odds | [Baccarat Strategy](https://www.casinopedia.org/table-games/baccarat) |
| **Long-Run Mathematical Guarantee** | ✓ house_edge.md | House always wins if games played indefinitely; Law of Large Numbers | [Wiki - Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) |
| **Vigorish (Vig) in Sports Betting** | ✓ vigorish_vig.md | Betting fee; typically -110 (risk $110 to win $100) | [Sports Betting Odds](https://www.espn.com/betting/) |

---

## IX. Advanced Probability Applications

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Probability Distributions in Games** | ✓ probability_distributions_in_games.md | Binomial (repeated independent trials), Normal, Exponential applications | [Khan Academy](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library) |
| **Variance Analysis for Games** | ✓ variance_volatility.md | Classify games by variance; high-var = better for advantage players | [Variance Analysis](https://www.wizardofodds.com/gambling/) |
| **Bayesian Updating** | ✓ bayesian_updating.md | Update odds as new information arrives (cards dealt, previous hands) | [Wiki - Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_inference) |
| **Simulation & Monte Carlo** | ✓ simulation_monte_carlo.md | Model complex games via random sampling; estimate odds in Texas Hold'em | [Monte Carlo Methods](https://en.wikipedia.org/wiki/Monte_Carlo_method) |
| **Equity Calculation** | ✓ poker_probabilities.md | % chance hand wins given all future cards random | [Pokerstove](http://www.pokerstove.com/) |
| **Discrete vs Continuous** | ✓ discrete_vs_continuous.md | Dice outcomes (discrete) vs card draws (discrete but larger sample space) | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) |

---

## X. Advantage Play & Professional Gambling

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Card Counting** | ✓ card_counting.md | Track deck composition; increase bets when favourable (Blackjack) | [Basic Strategy + Counting](https://www.wizardofodds.com/games/blackjack/card-counting/) |
| **Shuffle Tracking** | ✓ shuffle_tracking.md | Follow specific card clumps through shuffle; advantage = 0.5-1.5% | [Advantage Play](https://www.countercultureproductions.com/) |
| **Hole Carding** | ✓ hole_carding.md | Glimpse dealer's hole card; huge edge if possible (illegal in casinos) | [Advantage Techniques](https://www.casinopedia.org/cheating/) |
| **Exploiting Table Selection** | ✓ exploiting_table_selection.md | Choose games with higher RTP, better table conditions, positive rake situations | [Advantage Player Handbook](https://www.wizardofodds.com/gambling/) |
| **Bankroll Requirements** | ✓ bankroll_requirements.md | Enough capital to survive variance; Risk of Ruin calculations | [Kelly Criterion Applications](https://en.wikipedia.org/wiki/Kelly_criterion) |
| **Prop Betting** | ✓ prop_betting.md | Side bets with favorable odds; requires edge detection skill | [Proposition Betting](https://www.gambling.com/proposition-betting) |
| **Professional Gambling Synthesis** | ✓ professional_gambling_synthesis.md | Career viability, legal exposure, capital needs, and sustainability planning | [MIT Blackjack Team](https://en.wikipedia.org/wiki/MIT_Blackjack_Team) |

---

## XI. Sports Betting & Wagering

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Moneyline Odds** | ✓ moneyline_odds.md | American odds (+150 = $150 win on $100), Decimal (2.5 = $2.50 return per $1) | [Betting Odds Guide](https://www.espn.com/betting/) |
| **Spread Betting** | ✓ spread_betting.md | Point spread (team favored by X); applies handicap to evens-out odds | [Sports Betting Guide](https://www.draftkings.com/help/article/beginner-sports-betting-guide) |
| **Over/Under (Total) Betting** | ✓ over_under_betting.md | Wager on combined scores; requires statistical model | [Over/Under Betting](https://www.gambling.com/over-under-betting) |
| **Parlay Bets** | ✓ parlay_bets.md | Multiple selections combined; multiplied odds but all-or-nothing risk | [Parlay Strategy](https://www.actionnetwork.com/education/parlay) |
| **Implied Probability** | ✓ implied_probability.md | Convert betting odds to probability; crucial for edge detection | [Implied Probability Calculator](https://www.aceodds.com/bet-calculator/implied-probability) |
| **EV in Sports Betting** | ✓ ev_in_sports_betting.md | Expected Value = (Probability × Payout) - (Stake) | [Betting Mathematics](https://www.pinnacle.com/en/betting-resources/betting-strategy/expected-value) |

---

## XII. Mitigating Risk & Loss Control

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Stop-Loss & Profit Taking** | ✓ bankroll_management.md | Exit when down 50% of session budget (protect capital) | [Money Management](https://www.wizardofodds.com/gambling/) |
| **Time Limits** | ✓ time_limits.md | Pre-set session duration; prevents extended losing streaks | [Responsible Gambling](https://www.ncpg.org/) |
| **Emotional Control** | ✓ emotional_control.md | Tilt (emotional decisions), keep cool under pressure | [Psychology of Gambling](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4541962/) |
| **Game Selection Discipline** | ✓ game_selection_discipline.md | Play only +EV games; walk away from neutral/negative EV | [Advantage Player](https://www.countercultureproductions.com/) |
| **Betting Limits** | ✓ bankroll_management.md | Max bet % of bankroll; progressive vs fixed unit sizing | [Bankroll Protection](https://www.wizardofodds.com/gambling/) |
| **Self-Exclusion Programs** | ✓ addiction_responsible_gambling.md | Voluntary bans from casinos; prevents temptation | [NCPG Self-Exclusion](https://www.ncpg.org/general-public/self-exclusion/) |
| **Reality Checking** | ✓ reality_checking.md | Session time/money spent displays; breaks the trance of play | [Responsible Gambling Features](https://www.gamblingcommission.gov.uk) |

---

## XIII. Meta-Topics & Connections

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Kelly Criterion vs Risk of Ruin** | ✓ kelly_criterion.md & ✓ risk_of_ruin.md | Optimal growth vs capital preservation; trade-off between approaches | [Kelly Formula](https://en.wikipedia.org/wiki/Kelly_criterion) |
| **House Edge vs Expected Value** | ✓ house_edge.md & ✓ expected_value.md | Mathematical certainty players lose over time (for negative edge games) | [Wiki - House Edge](https://en.wikipedia.org/wiki/House_edge) |
| **Variance as Ally (for Advantage Players)** | ✓ variance_volatility.md & ✓ kelly_criterion.md | High variance helps skilled players; hurts casual players | [Variance Strategy](https://www.wizardofodds.com/gambling/) |
| **Game Theory Optimal Play** | ✓ game_theory_gambling.md | Theoretical perfect play; poker GTO solvers calculate this | [GTO in Poker](https://en.wikipedia.org/wiki/Game_theory_optimal) |
| **Psychological Traps** | ✓ gamblers_fallacy.md & ✓ addiction_responsible_gambling.md | Fallacies + design psychology create vulnerability cycle | [Psychology of Gambling](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4541962/) |
| **Statistics Applied to Gambling** | ✓ statistics_applied_to_gambling.md | Large number law guarantees house edge realization; confidence intervals on win rates | [Statistical Applications](https://www.wizardofodds.com/gambling/) |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **Wizard of Odds** | https://www.wizardofodds.com/ | Comprehensive game odds, house edge, strategy guides for all major games |
| **Wikipedia - Gambling** | https://en.wikipedia.org/wiki/Gambling | Historical, mathematical, social aspects of gambling |
| **National Council on Problem Gambling** | https://www.ncpg.org/ | Addiction resources, responsible gambling, treatment pathways |
| **Game Theory Handbook** | https://en.wikipedia.org/wiki/Game_theory | Nash equilibrium, mixed strategies, zero-sum foundations |
| **Casino Controller Board Data** | https://gaming.nv.gov/ | Actual RTP percentages, payout tables, regulatory info |
| **Pokerstove** | http://www.pokerstove.com/ | Equity calculator, hand probability analysis |
| **Edward Thorp - Beat the Dealer** | https://en.wikipedia.org/wiki/Edward_Thorp | Card counting, Kelly criterion, advantage play classic |

---

## Quick Stats

- **Total Topics Documented**: 75+
- **Workspace Files Created**: 16
- **Categories**: 13
- **Scope**: Basic probability → Game analysis → Advantage play → Psychology/Addiction
- **Primary Focus**: Mathematical understanding + practical application + responsible gambling awareness

---

## Organizational Structure

```
gambling_probability/
├── probability_foundations/
│   ├── basic_probability.md
│   ├── conditional_probability.md
│   ├── house_edge.md
│   ├── independence_dependence.md
│   ├── law_of_large_numbers.md
│   ├── payout_ratios.md
│   └── randomness_fairness.md
├── expected_value_risk_metrics/
│   ├── expected_value.md
│   ├── payout_ratios.md
│   ├── return_to_player.md
│   ├── risk_of_ruin.md
│   ├── variance_volatility.md
│   └── volatility_index.md
├── advanced_probability_game_theory/
│   ├── combinatorics_gambling.md
│   ├── game_theory_gambling.md
│   └── [game-specific analysis]
├── gambling_games/
│   ├── blackjack_odds.md
│   ├── craps_probabilities.md
│   ├── poker_probabilities.md
│   ├── roulette_probabilities.md
│   └── slot_machine_odds.md
├── risk_bankroll_betting/
│   ├── bankroll_management.md
│   ├── kelly_criterion.md
│   ├── martingale_system.md
│   └── risk_of_ruin.md
└── psychology_practical/
    ├── addiction_responsible_gambling.md
    ├── casino_design_player_behavior.md
    ├── gamblers_fallacy.md
    └── [topics expanded]
```

---

## Cross-References with Other Modules

| Related Module | Connection | Key Intersection |
|----------------|-----------|------------------|
| **Statistics** | Probability theory foundation | Basic probability, distributions, expected value, hypothesis testing |
| **Quantitative Finance** | Risk modeling, bankroll analogous to capital | Kelly criterion → portfolio optimization, variance analysis |
| **Time Series Analysis** | Sequence patterns, streak detection | Hot-hand fallacy, law of large numbers verification |
| **Game Theory** | Strategic interactions, Nash equilibrium | Poker strategy, mixed strategy optimal play |

---

**Last Updated**: 2026-01-31  
**Coverage**: Beginner (basic probability, simple games) → Intermediate (game analysis, strategy) → Advanced (advantage play, GTO)
