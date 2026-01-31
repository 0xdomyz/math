# Behavioral Economics & Responsible Gambling: Cognitive Biases & Addiction Mechanics

## I. Concept Skeleton

**Definition:** Behavioral economics applies psychological research to explain and predict gambling behavior beyond rational choice theory. This includes cognitive biases (gambler's fallacy, illusion of control), emotional drivers (hope, fear, frustration), casino design exploiting human psychology, and addiction as a medical/behavioral disorder requiring intervention.

**Purpose:** Understand why people gamble despite negative EV (mathematical irrationality), identify structural and behavioral addiction mechanisms, recognize casino design tactics that maximize engagement and spending, and design evidence-based harm reduction and responsible gambling programs.

**Prerequisites:** Expected value, probability theory, loss aversion, cognitive biases, dopamine/reward systems, diagnostic criteria (DSM-5), harm reduction frameworks.

---

## II. Comparative Framing

| **Bias/Factor** | **Mechanism** | **Impact on Behavior** | **Casino Exploitation** | **Harm Severity** |
|-----------|----------|----------|----------|----------|
| **Gambler's Fallacy** | "Streak must end" (regression illusion) | Increases bets after losses (chasing) | Design losing streaks to trigger escalation | High (loss acceleration) |
| **Hot-Hand Fallacy** | "Streak continues" (momentum illusion) | Increases bets after wins (overconfidence) | Celebrate wins → play longer → lose all | High (volatility trap) |
| **Illusion of Control** | "My skill/timing beats randomness" | Overestimate win probability | Skill-lite games (slots pretend skill) | Moderate-High (persistence) |
| **Loss Aversion** | Losing $100 hurts more than winning $100 feels good | Chases losses deeper | Deposit matching → catch-up betting | Very High (ruin acceleration) |
| **Near-Misses** | Almost-win triggers hope and continued play | Extends session duration | Slots display near-misses prominently | Moderate (engagement) |
| **Availability Bias** | Wins easier to recall than losses | Overestimate actual win rate | Celebrate wins loudly (social media, sounds) | Moderate (misperception) |
| **Sunk Cost Fallacy** | "Already lost $500, must recover it" | Justifies continued play despite losses | Deposit bonuses (create commitment) | Very High (entrapment) |

---

## III. Examples & Counterexamples

### Example 1: The Gambler's Fallacy - The "Due" Illusion

**Setup:**
- You play roulette: Red for 10 spins in a row (extremely unlikely: 1 in 1,024)
- Next spin approaching
- Question: What's the probability next spin is Black?

**Intuitive (Fallacious) Reasoning:**

```
"Red has come up 10 times! Black must be 'due'."

Logic (flawed):
├─ In a fair game, 50% of spins should be Red, 50% Black
├─ After 10 Red: "The universe owes us Black"
├─ Next spin is more likely Black (>50%) to "balance out"
└─ "If I keep playing, Black will come eventually"

Betting Behavior:
├─ Increase bet size on Black
├─ Emotional conviction: "This is MY moment"
└─ May bet life savings on next Black

Result:
├─ Next spin is Black 49.5%, Red 50.5% (virtually unchanged!)
├─ Fallacy causes overconfidence
└─ Bad luck (skew persists) → escalating losses
```

**Mathematical Reality:**

```
Roulette outcomes are INDEPENDENT:

P(Black on spin 11 | 10 Red before) = P(Black) = 18/37 ≈ 48.6%

Explanation:
├─ Previous spins DO NOT affect future spins
├─ Roulette wheel has no memory
├─ "Streak" is just randomness (variance)
├─ Past reds and future blacks are unrelated

Law of Large Numbers (Correct Interpretation):
├─ Over 10,000 spins, count will approach 50-50
├─ But short-term: Any sequence is possible
├─ 10 reds followed by 10 blacks is as "normal" as alternating
└─ No sequence is "owed" to balance short-term deficit

Key Distinction:
├─ SAMPLE will converge to population proportion (LLN is TRUE)
├─ But EACH spin is still independent
├─ Gambler confuses "sample will balance" with
│  "this spin is biased to balance"
└─ Category error: Population property ≠ individual event property
```

**Counterexample: Gambler Recognizes Fallacy**

```
Same setup: 10 Red spins in a row

Rational thinking:
├─ Probability unchanged: P(Black) = 48.6%
├─ But probability is not favorable: 51.4% to lose
├─ Expected value of betting: -2.7% (house edge)
├─ Correct decision: DON'T BET

Why no increased bet?
├─ Previous streak is irrelevant
├─ House edge remains unchanged
└─ Only rational basis for betting: None (negative EV)
```

**Psychological Damage of the Fallacy:**

```
Gambler's Loss Acceleration Cycle:

Initial loss (roulette bet, -$100):
├─ Frustration: "I was SO close"
├─ False belief: "Due for a win now"
├─ Bet increased: $200

Another loss (-$200):
├─ Desperation: "Surely now I'm due"
├─ Bet escalation: $500
├─ Emotional investment: "I can't leave a loser"

Continuing spiral:
├─ Loss count: $100, $300, $800 cumulative
├─ Bet size: $500 → $1000 → $5000
├─ Conviction: "NEXT spin is my money back"
├─ Loss of control: Can't walk away (feels "wrong")

Final outcome:
├─ Total loss: $3,500+
├─ Bankroll depleted
├─ "If I'd had more money, I'd have won"
└─ Gambler's fallacy + loss aversion = ruin
```

---

### Example 2: Near-Misses & The "Almost-Win" Illusion

**Setup:**
- Slot machine: 3 reels
- Your spin: 🍒 🍒 🔔 (two matching cherries, one cherry away from 3-match jackpot)
- Expected result: Lose, small payout
- Question: Why do casinos highlight this?

**Casino Design (Near-Miss Exploitation):**

```
Near-miss display mechanism:

Traditional (player sees):
├─ Reel 1: 🍒 (gold highlight) ✓
├─ Reel 2: 🍒 (gold highlight) ✓
├─ Reel 3: 🔔 (spinning)
│
└─ Design choice: SLOW DOWN reel 3, show how close you came

Psychological effect:
├─ Brain registers: "Almost won!"
├─ Dopamine response: Similar to near-wins in sports
├─ Player thinks: "The machine ALMOST paid me"
├─ Implicit belief: "I was SO close, next spin surely wins"
│
└─ Reality: Reel 3 was NEVER going to hit (predetermined)
```

**Brain Science (Reward System):**

```
Dopamine release (motivation/reward):

Normal win pattern:
├─ See 3 matching symbols
├─ Brain: "Success! I won!"
├─ Dopamine spike: Moderate
└─ Behavior: Satisfied, may stop playing

Near-miss pattern:
├─ See 2 matching + 1 miss
├─ Brain: "Almost! SO CLOSE!"
├─ Dopamine spike: Surprisingly HIGH (research shows ~70% of full-win dopamine)
├─ Behavior: "One more spin! I'm feeling it!"
└─ Result: Extended play → more losses

Why evolution wired this:
├─ In sports/hunting: Near-miss means "adjust & try again"
├─ Adaptive in real skills (improving technique)
├─ MALADAPTIVE in pure chance (randomness, no adjustment helps)
└─ Casino exploits this mismatched intuition
```

**Casino Near-Miss Implementation:**

```
Slot machine mathematics (realistic example):

Reel configuration: 20 positions each, 1 cherry on each
P(cherry on reel 1): 1/20
P(cherry on reel 2): 1/20
P(cherry on reel 3): 1/20

P(3 cherries = jackpot): (1/20)^3 = 1/8,000

But casino programs:
├─ Reel 3 occasionally lands on symbol ADJACENT to cherry
├─ Display: "Ooohhh, SO CLOSE! 🍒 🍒 🍒 (next to it!)"
│
├─ Reality: RNG chose result BEFORE reels spun
├─ Reel animation was FAKE (show closest symbol for drama)
│
└─ Player fallacy: "The machine almost gave me the win"
    (Illusion of control / luck)
```

**Counterexample: Player Recognizes the Illusion**

```
Same near-miss (🍒 🍒 🔔)

Rational interpretation:
├─ Predetermined result: Loss (90% of all spins lose)
├─ Near-miss: Just animation (reels spun AFTER RNG decision)
├─ Implication: "Almost close" is meaningless
│
├─ Odds next spin: Still -EV (house edge 2-15%)
├─ Decision: STOP PLAYING
│
└─ Actual behavior: Quit before the "nudge" trap
```

---

### Example 3: Addiction Mechanics - How Gambling Disorders Develop

**Setup:**
- Patient: 35-year-old, initially recreational poker player
- Timeline: 5-year progression to severe gambling disorder
- Question: How does addiction develop and escalate?

**Stage 1: Social/Recreational (Months 0-3)**

```
Initial experience:
├─ Friend invites to poker game ($20 buy-in)
├─ Win $150 on first night
├─ Brain: "This is fun AND I'm good at it!"
├─
├─ EV reality: Skill variance dominates short-term
├─ Lucky run ($1,200 total wins over 10 sessions)
│
└─ Outcome: 
   ├─ More frequent attendance (weekly)
   ├─ Romanticized as "skill" not luck
   └─ Early win bias (remember wins, minimize losses)
```

**Stage 2: Regular/Problem Gambling (Months 3-12)**

```
Escalation drivers:
├─ Initial winnings plateau/reverse
├─ Loss aversion kicks in: "Can't accept losing my $1,200"
│
├─ Behavioral changes:
│  ├─ Poker nights increase to 3x weekly
│  ├─ Bet sizes increase gradually
│  ├─ Playing "to recover losses" (chase behavior)
│  └─ Minimizing losses to spouse ("only $200 lost this week")
│
├─ Psychological mechanisms:
│  ├─ Illusion of control: "I've gotten better" (actually worse)
│  ├─ Gambler's fallacy: "My losing streak must end soon"
│  ├─ Cognitive distortions: "Professional players start here"
│  └─ Emotional drivers: Excitement peak > normal life dopamine
│
└─ Net result:
   ├─ Month 3: +$1,200 (lucky)
   ├─ Month 6: -$800 (skill limited, variance reverting)
   ├─ Month 12: -$5,000 cumulative (escalation + downswing)
   └─ Spending: $5,000 / 12 months = $417/month (significant for middle income)
```

**Stage 3: Compulsive Gambling (Year 1-3)**

```
Diagnostic patterns (DSM-5 Gambling Disorder):

Behavioral indicators:
├─ Preoccupation: Constant thinking about gambling
│  └─ Daydreaming about next game, calculating odds, planning bets
├─ Tolerance: Increasing bet sizes needed for excitement
│  └─ $20 game feels "boring," need $100+ game
├─ Withdrawal: Restlessness/irritability when not gambling
│  └─ Can't focus at work on non-poker days
├─ Escape motivation: Using gambling to escape stress/depression
│  └─ Wife mad? Gambling relieves emotional pain (temporarily)
└─ Chasing losses: Increasing bets to "recover"
   └─ Lost $500 tonight → "Double down tomorrow"

Life consequences:
├─ Financial:
│  ├─ Hiding losses from spouse
│  ├─ Maxing credit cards
│  ├─ Missing bills (mortgage late, car payment skip)
│  └─ Cumulative loss: $20,000+ over 2 years
│
├─ Relationship:
│  ├─ Spouse discovers credit card debt
│  ├─ Marriage on brink (threats of divorce)
│  ├─ Withdrawn from children (emotionally unavailable)
│  └─ Social isolation (avoids friends, ashamed)
│
├─ Employment:
│  ├─ Late to work (poker ran late)
│  ├─ Performance decline (distracted, tired)
│  ├─ Almost fired (supervisor's warning)
│  └─ Job security threatened
│
└─ Mental health:
   ├─ Anxiety: Constant debt worry
   ├─ Depression: Hopelessness ("Can't pay this back")
   ├─ Insomnia: Can't sleep (running numbers, guilt)
   └─ Suicidal ideation: "Only way out is..." (crisis thinking)
```

**Stage 4: Severe Disorder/Harm (Year 3+)**

```
Crisis escalation:

Financial ruin:
├─ Total losses: $100,000+
├─ Bankruptcy filing required
├─ House at risk (second mortgage taken)
├─ Wife files for divorce

Behavioral control loss:
├─ Promises to quit broken repeatedly
├─ "I'm done" → 2 days later back at table
├─ Unable to self-limit (says $200 limit, leaves $2,000 later)
├─ Gambling despite severe consequences

Legal/Criminal involvement:
├─ Embezzlement at work (to fund gambling)
├─ Criminal charges filed
├─ Felony on record
└─ Prison time possible

Health consequences:
├─ Stress-related health issues: High BP, ulcers
├─ Substance abuse escalation (alcohol during play)
├─ Neglected sleep, nutrition
└─ Suicidal attempt (or completed)

Outcome statistics (untreated):
├─ 15-20% suicide rate among severe gambling disorder patients
├─ 50%+ relationship breakdowns
├─ 30%+ job loss
├─ Financial recovery: Typically 5-10+ years post-treatment
└─ Relapse rate (untreated): 40-60% within first year
```

---

## IV. Layer Breakdown

```
BEHAVIORAL ECONOMICS & ADDICTION PSYCHOLOGY

┌─────────────────────────────────────────────────────┐
│  1. COGNITIVE BIASES IN GAMBLING                    │
│                                                     │
│  Gambler's Fallacy:                                 │
│  ├─ Definition: "Streak must reverse soon"         │
│  ├─ Origin: Misapplication of law of large numbers │
│  ├─ Impact: Escalates bets after losses (chasing)  │
│  ├─ Example: Red 10x → "Black is due" → bet more  │
│  │  Result: Amplifies losses                       │
│  └─ Prevalence: 70%+ of casual gamblers            │
│                                                     │
│  Hot-Hand Fallacy:                                  │
│  ├─ Definition: "Streak will continue"             │
│  ├─ Origin: Pattern-seeking (evolutionary)         │
│  ├─ Impact: Overconfidence, plays longer           │
│  ├─ Example: Win 3x → "I'm hot!" → bigger bets    │
│  │  Result: Regression to mean, amplified loss    │
│  └─ Adaptive for: Sports/competition (skill-based) │
│                                                     │
│  Illusion of Control:                               │
│  ├─ Definition: "My skill/luck beats randomness"   │
│  ├─ Origin: Overestimate ability to influence RNG  │
│  ├─ Impact: Persistent belief despite evidence     │
│  ├─ Example: "I have a system" (for roulette)     │
│  │  Reality: No system beats house edge            │
│  ├─ Manifestation: Special rituals, lucky objects  │
│  └─ Targeted by: Slots (skill-mimicking buttons)   │
│                                                     │
│  Loss Aversion:                                     │
│  ├─ Definition: Losing $X hurts >2x more than     │
│  │  winning $X feels good                          │
│  ├─ Origin: Evolutionary (avoid scarcity)          │
│  ├─ Impact: Aggressive chasing of losses           │
│  ├─ Example: Down $500 → Bet $1000 to recover    │
│  │  Result: Ruin acceleration (exponential loss)   │
│  └─ Magnitude: Loss hurts ~2.5x more for typical  │
│     person                                          │
└──────────────────┬────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │  2. REWARD SYSTEM & DOPAMINE                │
    │                                              │
    │  Brain Chemistry:                            │
    │  ├─ Dopamine: Motivation/reward transmitter │
    │  ├─ Released on: Winning (especially close) │
    │  ├─ Also released: Anticipation before bet │
    │  └─ Addiction cycles: Dopamine dysregulation│
    │                                              │
    │  Variable Ratio Reinforcement:               │
    │  ├─ Definition: Reward unpredictable         │
    │  ├─ Schedule type: Most addictive (psych)   │
    │  ├─ Applied in: Slot machines, lotteries    │
    │  ├─ Why addictive:                          │
    │  │  ├─ Unpredictability = high dopamine    │
    │  │  ├─ Brain can't predict payoff timing    │
    │  │  └─ Keeps repeating "maybe next time"    │
    │  │                                          │
    │  ├─ Comparison (less addictive):             │
    │  │  ├─ Fixed ratio (every 5th spin wins)   │
    │  │  └─ Predictable = lower dopamine         │
    │  │                                          │
    │  └─ Real-world examples:                     │
    │     ├─ Slots (variable ratio) = highly       │
    │     │  addictive                             │
    │     ├─ Lottery (variable ratio) = addictive │
    │     ├─ Fixed schedule = less addictive      │
    │     └─ Straight poker (skill) = less        │
    │        addictive (more predictable rewards)  │
    │                                              │
    │  Tolerance Development:                      │
    │  ├─ Initial: Small bet = high dopamine      │
    │  ├─ After weeks: Same bet = less dopamine   │
    │  ├─ Adaptation: Brain downregulates         │
    │  │  dopamine receptors                       │
    │  ├─ Result: Need bigger bets/bigger wins    │
    │  └─ Escalation spiral: Bets double/triple   │
    │                                              │
    │  Withdrawal Symptoms:                        │
    │  ├─ When unable to gamble:                  │
    │  │  ├─ Restlessness, irritability           │
    │  │  ├─ Anxiety, depression                  │
    │  │  ├─ Insomnia                             │
    │  │  └─ Difficulty concentrating              │
    │  ├─ Parallels: Opioid withdrawal (similar   │
    │  │  but less severe physically)              │
    │  └─ Duration: 2-4 weeks for acute           │
    │                                              │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │  3. CASINO DESIGN EXPLOITATION               │
    │                                              │
    │  Architecture:                               │
    │  ├─ No clocks/windows: Distort time         │
    │  │  → Gambler doesn't notice hours passing  │
    │  ├─ Layout: Slot machines near entrance     │
    │  │  → Draw in casual visitors                │
    │  ├─ Carpet pattern: Busy, stimulating       │
    │  │  → Hyperstimulation (harder to think)    │
    │  └─ Exit obscured: Difficult to find door   │
    │     → Encourage continued play              │
    │                                              │
    │  Sound Design:                               │
    │  ├─ Celebratory bells/chimes on EVERY win  │
    │  │  → Even -EV situations feel celebratory  │
    │  ├─ Loud jackpot sounds: Attract neighbors │
    │  │  → Social proof ("Someone won!")         │
    │  ├─ Silence on losses: Not announced        │
    │  │  → Attention/psychology mismatch          │
    │  └─ Music: Fast tempo, energetic            │
    │     → Psychological acceleration             │
    │                                              │
    │  Visual Design:                              │
    │  ├─ Bright lights: Hyperstimulation         │
    │  ├─ Slot machine reels: Slow-motion on      │
    │  │  near-misses (animated excitement)        │
    │  ├─ Graphics: Celebrate small wins like     │
    │  │  jackpots ("Oh no! 2 cherries!")         │
    │  └─ Color psychology: Red (arousal),        │
    │     gold (reward) throughout               │
    │                                              │
    │  Game Design:                                │
    │  ├─ Volatility: High-variance games         │
    │  │  → Frequent near-misses (dopamine)       │
    │  ├─ Betting options: All sizes available    │
    │  │  → Low barrier to high stakes             │
    │  ├─ Fast play: Slots spin every 2-3 sec   │
    │  │  → Rapid loss of bankroll (1000s/hour)   │
    │  └─ Multi-line slots: Play 50 lines at once│
    │     → Cognitive overload (can't track losses)│
    │                                              │
    │  Psychological Tactics:                      │
    │  ├─ Free drinks: Impair judgment + gratitude│
    │  ├─ Loyalty programs: Incentivize return    │
    │  ├─ Losses presented as "near-wins"         │
    │  ├─ Winnings celebrated (losses muted)      │
    │  └─ Staff: Friendly, encouraging attitude   │
    │                                              │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │  4. ADDICTION DISORDER PROGRESSION           │
    │                                              │
    │  DSM-5 Criteria (need 4+ for diagnosis):     │
    │  ├─ 1. Preoccupation with gambling           │
    │  ├─ 2. Tolerance (need escalation)           │
    │  ├─ 3. Withdrawal (irritable when not)      │
    │  ├─ 4. Loss of control (can't quit)         │
    │  ├─ 5. Escape motivation                    │
    │  ├─ 6. Chasing losses                       │
    │  ├─ 7. Lying about extent of gambling      │
    │  ├─ 8. Jeopardized/lost opportunities      │
    │  ├─ 9. Relied on others for financial help │
    │  └─ 10. Illegal acts to fund gambling       │
    │                                              │
    │  Progression Timeline:                       │
    │  ├─ Stage 1 (Social): Occasional,           │
    │  │  controlled, wins normative (~0-6 months)│
    │  ├─ Stage 2 (Problem): Regular play,        │
    │  │  chasing losses, minimizing             │
    │  │  (~6-18 months)                          │
    │  ├─ Stage 3 (Compulsive): Preoccupied,     │
    │  │  loss of control, life dysfunction      │
    │  │  (~18-36 months)                         │
    │  └─ Stage 4 (Severe): Crisis mode,         │
    │     suicidal ideation, criminalization     │
    │     (36+ months untreated)                 │
    │                                              │
    │  Co-morbidities (common):                    │
    │  ├─ Mood: Depression, bipolar (20-60%)     │
    │  ├─ Anxiety: GAD, OCD (15-45%)             │
    │  ├─ Substance: Alcohol abuse (20-50%)      │
    │  ├─ Personality: ADHD (10-30%)             │
    │  └─ Family history: Addiction heritability  │
    │     (60%+)                                  │
    │                                              │
    │  Risk Factors:                               │
    │  ├─ Biological: Dopamine sensitivity        │
    │  ├─ Psychological: Impulsivity, stress      │
    │  ├─ Environmental: Casino proximity,        │
    │  │  peer pressure, trauma history           │
    │  └─ Timing: Peak onset age 20-40            │
    │                                              │
    └────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Loss Aversion Function

$$\text{Value}(x) = \begin{cases} x^\alpha & x \geq 0 \\ -\lambda |x|^\alpha & x < 0 \end{cases}$$

Where $\lambda \approx 2.25$ (losses hurt ~2.25x more), $\alpha \approx 0.88$ (diminishing sensitivity)

### Gambler's Ruin (with Fallacy Betting)

Expected ruin probability when chasing losses (escalating bets):
$$P(\text{ruin} | \text{chase}) \approx \frac{1 - (1-2p)^{B/b_0}}{1 - (1-2p)^{B/b_0} + \text{exponential escalation}}$$

With fallacy, effective $p$ decreases and bet escalation accelerates convergence to ruin.

### Addiction Severity Index (DSM-5 Based)

$$\text{Severity} = \frac{\text{# criteria met}}{10} \times 100$$
- 40-50%: Mild
- 60-70%: Moderate  
- 80%+: Severe

---

## VI. Python Mini-Project: Cognitive Bias Simulation & Addiction Risk Analyzer

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# ============================================================================
# COGNITIVE BIAS SIMULATORS & ADDICTION MODELS
# ============================================================================

class GamblersFallacySimulator:
    """Simulate behavioral impact of gambler's fallacy"""
    
    @staticmethod
    def simulation_no_fallacy(num_spins=100, initial_bet=10, edge=-0.0526):
        """
        Rational player: Bet fixed amount regardless of streak
        """
        bankroll = 1000
        bets = []
        outcomes = []
        
        for spin in range(num_spins):
            # Fixed bet (rational)
            bet_amount = initial_bet
            bets.append(bet_amount)
            
            # Roulette outcome
            if np.random.random() < 0.5 + edge/2:
                outcome = bet_amount  # Win
            else:
                outcome = -bet_amount  # Lose
            
            outcomes.append(outcome)
            bankroll += outcome
            
            if bankroll <= 0:
                break
        
        return {
            'bankroll_history': np.cumsum([1000] + outcomes),
            'bets': bets,
            'final_bankroll': max(0, bankroll),
            'total_loss': 1000 - max(0, bankroll)
        }
    
    @staticmethod
    def simulation_with_fallacy(num_spins=100, initial_bet=10, edge=-0.0526):
        """
        Gambler with fallacy: Escalate bets after losses (chase)
        """
        bankroll = 1000
        bets = []
        outcomes = []
        loss_streak = 0
        
        for spin in range(num_spins):
            # Escalate bet based on loss streak
            bet_multiplier = 1 + (loss_streak * 0.5)  # 1x, 1.5x, 2x, 2.5x...
            bet_amount = min(initial_bet * bet_multiplier, bankroll * 0.25)  # Cap at 25% of bank
            bets.append(bet_amount)
            
            # Roulette outcome
            if np.random.random() < 0.5 + edge/2:
                outcome = bet_amount  # Win
                loss_streak = 0  # Reset streak
            else:
                outcome = -bet_amount  # Lose
                loss_streak += 1  # Increment streak (triggers escalation)
            
            outcomes.append(outcome)
            bankroll += outcome
            
            if bankroll <= 0:
                break
        
        return {
            'bankroll_history': np.cumsum([1000] + outcomes),
            'bets': bets,
            'final_bankroll': max(0, bankroll),
            'total_loss': 1000 - max(0, bankroll),
            'peak_bet': max(bets)
        }


class AddictionRiskModel:
    """Model gambling addiction progression"""
    
    @staticmethod
    def dsm5_severity_score(criteria_met):
        """
        Calculate DSM-5 gambling disorder severity
        criteria_met: 0-10 (number of DSM-5 criteria satisfied)
        """
        if criteria_met < 4:
            return {'severity': 'Not Disordered', 'percentage': 0}
        elif criteria_met < 6:
            return {'severity': 'Mild', 'percentage': (criteria_met - 3) * 20}
        elif criteria_met < 8:
            return {'severity': 'Moderate', 'percentage': 50 + (criteria_met - 6) * 15}
        else:
            return {'severity': 'Severe', 'percentage': 80 + (criteria_met - 8) * 10}
    
    @staticmethod
    def progression_model(initial_spend_monthly, months=36):
        """
        Model addiction progression over time
        """
        timelines = []
        spending = [initial_spend_monthly]
        stages = ['Social']
        
        for month in range(1, months + 1):
            # Escalation: Monthly spend increases over time
            if month < 6:
                # Social phase: gradual increase
                escalation_factor = 1.05
            elif month < 18:
                # Problem phase: moderate escalation
                escalation_factor = 1.08
            else:
                # Compulsive phase: rapid escalation
                escalation_factor = 1.15
            
            new_spend = spending[-1] * escalation_factor
            spending.append(new_spend)
            
            # Stage assignment
            if month < 6:
                stages.append('Social')
            elif month < 18:
                stages.append('Problem')
            elif month < 36:
                stages.append('Compulsive')
            else:
                stages.append('Severe')
        
        cumulative_loss = np.sum(spending)
        
        return {
            'months': list(range(months + 1)),
            'monthly_spending': spending,
            'cumulative_loss': cumulative_loss,
            'stages': stages,
            'final_monthly': spending[-1],
            'spending_ratio': spending[-1] / spending[0]
        }


class CasinoDesignImpactAnalyzer:
    """Quantify impact of casino design on gambling duration"""
    
    @staticmethod
    def near_miss_effect_on_play_duration(base_duration_minutes=30):
        """
        Model impact of near-miss frequency on play duration
        """
        near_miss_frequencies = np.array([0, 0.2, 0.4, 0.6, 0.8])  # 0%, 20%, 40%, etc.
        
        # Empirical finding: Each +1% near-miss frequency extends play ~1.5 minutes
        extended_durations = []
        
        for nmf in near_miss_frequencies:
            # Base duration + extension from near-misses
            extension = nmf * 100 * 1.5  # X% frequency * X * 1.5 min per %
            duration = base_duration_minutes + extension
            extended_durations.append(duration)
        
        return {
            'near_miss_frequencies': near_miss_frequencies * 100,
            'play_durations': extended_durations,
            'avg_loss_per_hour': np.array(extended_durations) / 60 * 100,  # Assuming $100/hour loss rate
        }
    
    @staticmethod
    def dopamine_response_simulation(num_bets=100, bet_type='slot_machine'):
        """
        Simulate dopamine response to different bet types
        """
        dopamine = []
        
        for bet_num in range(num_bets):
            if bet_type == 'slot_machine':
                # Variable ratio (unpredictable wins)
                if np.random.random() < 0.1:  # 10% win rate
                    # Dopamine: High for win
                    dp_hit = 0.8
                else:
                    # Dopamine: Small for loss or near-miss
                    dp_hit = 0.2 if np.random.random() < 0.2 else 0.0  # 20% near-miss chance
                
                # Anticipation: Present before every spin
                dopamine.append(0.3 + dp_hit)
            
            elif bet_type == 'sports_betting':
                # Outcome-dependent (predictable)
                if np.random.random() < 0.5:  # 50% win rate
                    dopamine.append(0.6)
                else:
                    dopamine.append(0.0)
            
            elif bet_type == 'poker':
                # Skill-based (less variable)
                skill_factor = min(1.0, bet_num / 100)  # Improve with experience
                outcome_prob = 0.5 + skill_factor * 0.1
                
                if np.random.random() < outcome_prob:
                    dopamine.append(0.5 + skill_factor * 0.2)
                else:
                    dopamine.append(0.0)
        
        # Tolerance: Dopamine requirement increases
        dopamine_adjusted = np.array(dopamine)
        for i in range(len(dopamine_adjusted)):
            tolerance_factor = 1 - (i / (num_bets * 0.8)) * 0.3  # 30% decrease in sensitivity
            dopamine_adjusted[i] *= max(0.4, tolerance_factor)
        
        return {
            'bet_type': bet_type,
            'dopamine_responses': dopamine_adjusted.tolist(),
            'avg_dopamine': np.mean(dopamine_adjusted),
            'dopamine_decline': (dopamine_adjusted[0] - dopamine_adjusted[-1]) / dopamine_adjusted[0] * 100
        }


# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("BEHAVIORAL ECONOMICS & ADDICTION MECHANICS")
print("="*80)

# 1. Gambler's Fallacy Simulation
print(f"\n1. GAMBLER'S FALLACY - CHASING LOSSES")
print(f"{'-'*80}")

sim_rational = GamblersFallacySimulator.simulation_no_fallacy(num_spins=50)
sim_fallacy = GamblersFallacySimulator.simulation_with_fallacy(num_spins=50)

print(f"\nRational Player (Fixed Bets):")
print(f"  Initial bankroll: $1,000")
print(f"  Final bankroll: ${sim_rational['final_bankroll']:.2f}")
print(f"  Total loss: ${sim_rational['total_loss']:.2f}")
print(f"  Max bet during run: ${max(sim_rational['bets']):.2f}")

print(f"\nGambler with Fallacy (Escalating Bets):")
print(f"  Initial bankroll: $1,000")
print(f"  Final bankroll: ${sim_fallacy['final_bankroll']:.2f}")
print(f"  Total loss: ${sim_fallacy['total_loss']:.2f}")
print(f"  Max bet during run: ${sim_fallacy['peak_bet']:.2f}")

print(f"\nImpact of Fallacy:")
print(f"  Additional loss from chasing: ${sim_fallacy['total_loss'] - sim_rational['total_loss']:.2f}")
print(f"  Escalation factor: {sim_fallacy['peak_bet'] / max(sim_rational['bets']):.1f}x")

# 2. Addiction Progression
print(f"\n2. ADDICTION PROGRESSION MODEL (36 Months)")
print(f"{'-'*80}")

addiction_prog = AddictionRiskModel.progression_model(initial_spend_monthly=100, months=36)

print(f"\nMonthly Spending Trajectory:")
print(f"  Month 1: ${addiction_prog['monthly_spending'][0]:.2f}")
print(f"  Month 12: ${addiction_prog['monthly_spending'][12]:.2f}")
print(f"  Month 24: ${addiction_prog['monthly_spending'][24]:.2f}")
print(f"  Month 36: ${addiction_prog['monthly_spending'][-1]:.2f}")

print(f"\nCumulative Impact:")
print(f"  Total loss over 36 months: ${addiction_prog['cumulative_loss']:.2f}")
print(f"  Spending increase factor: {addiction_prog['spending_ratio']:.1f}x")

print(f"\nStage Progression:")
for stage in set(addiction_prog['stages']):
    month_start = addiction_prog['stages'].index(stage) + 1
    stage_spending = [addiction_prog['monthly_spending'][i] for i, s in enumerate(addiction_prog['stages']) if s == stage]
    if stage_spending:
        print(f"  {stage}: Month ~{month_start}, avg spending ${np.mean(stage_spending):.2f}/month")

# 3. Casino Design Impact
print(f"\n3. CASINO DESIGN - NEAR-MISS IMPACT")
print(f"{'-'*80}")

nmiss_analysis = CasinoDesignImpactAnalyzer.near_miss_effect_on_play_duration(base_duration_minutes=30)

print(f"\nNear-Miss Frequency Impact on Play Duration:")
for freq, duration in zip(nmiss_analysis['near_miss_frequencies'], nmiss_analysis['play_durations']):
    additional_min = duration - 30
    print(f"  {freq:.0f}% near-misses: {duration:.0f} min play (+{additional_min:.0f} min)")

# 4. Dopamine Response Comparison
print(f"\n4. DOPAMINE RESPONSE - ADDICTION RISK BY GAME TYPE")
print(f"{'-'*80}")

games = ['slot_machine', 'sports_betting', 'poker']
dopamine_results = []

for game in games:
    result = CasinoDesignImpactAnalyzer.dopamine_response_simulation(num_bets=100, bet_type=game)
    dopamine_results.append(result)
    
    print(f"\n{game.upper().replace('_', ' ')}:")
    print(f"  Avg dopamine response: {result['avg_dopamine']:.2f} / 1.0")
    print(f"  Tolerance development: {result['dopamine_decline']:.1f}% decline over 100 bets")
    print(f"  Addiction risk: {'HIGH' if result['avg_dopamine'] > 0.5 else 'MODERATE' if result['avg_dopamine'] > 0.3 else 'LOW'}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Gambler's Fallacy Comparison
ax1 = axes[0, 0]

months_rational = range(len(sim_rational['bankroll_history']))
months_fallacy = range(len(sim_fallacy['bankroll_history']))

ax1.plot(months_rational, sim_rational['bankroll_history'], label='Rational (Fixed Bets)',
        linewidth=2.5, marker='o', markersize=4, color='green', alpha=0.7)
ax1.plot(months_fallacy, sim_fallacy['bankroll_history'], label='With Fallacy (Escalating)',
        linewidth=2.5, marker='s', markersize=4, color='red', alpha=0.7)

ax1.axhline(y=1000, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Starting Bankroll')
ax1.fill_between(months_rational, 0, 1000, alpha=0.1, color='green', label='Profit Zone')
ax1.fill_between(months_rational, 0, 500, alpha=0.1, color='red', label='Loss Zone')

ax1.set_xlabel('Number of Spins')
ax1.set_ylabel('Bankroll ($)')
ax1.set_title('Panel 1: Gambler\'s Fallacy Impact (Chasing Losses)')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1200)

# Panel 2: Addiction Progression
ax2 = axes[0, 1]

ax2.plot(addiction_prog['months'], addiction_prog['monthly_spending'], 
        linewidth=2.5, marker='o', markersize=5, color='darkred')
ax2.fill_between(addiction_prog['months'], 0, addiction_prog['monthly_spending'], alpha=0.3, color='darkred')

# Add stage background colors
stage_boundaries = {
    'Social': (0, 6, 'lightgreen'),
    'Problem': (6, 18, 'lightyellow'),
    'Compulsive': (18, 36, 'lightsalmon'),
}

for stage_name, (start, end, color) in stage_boundaries.items():
    ax2.axvspan(start, end, alpha=0.2, color=color)
    ax2.text((start + end) / 2, addiction_prog['cumulative_loss'] * 0.01, stage_name,
            ha='center', fontweight='bold', fontsize=9)

ax2.set_xlabel('Month')
ax2.set_ylabel('Monthly Spending ($)')
ax2.set_title('Panel 2: Addiction Progression - Monthly Spending Escalation')
ax2.grid(True, alpha=0.3)

# Panel 3: Near-Miss Effect
ax3 = axes[1, 0]

ax3.plot(nmiss_analysis['near_miss_frequencies'], nmiss_analysis['play_durations'],
        linewidth=2.5, marker='D', markersize=8, color='purple')
ax3.fill_between(nmiss_analysis['near_miss_frequencies'], 30, nmiss_analysis['play_durations'],
                alpha=0.3, color='purple', label='Extended Play Time')

ax3.axhline(y=30, color='black', linestyle='--', linewidth=1.5, label='Baseline Duration (30 min)')
ax3.set_xlabel('Near-Miss Frequency (%)')
ax3.set_ylabel('Average Play Duration (minutes)')
ax3.set_title('Panel 3: Casino Design - Near-Miss Engagement Effect')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Dopamine Response & Addiction Risk
ax4 = axes[1, 1]

game_names = [g.upper().replace('_', ' ') for g in games]
dopamine_avgs = [result['avg_dopamine'] for result in dopamine_results]
tolerance_develops = [result['dopamine_decline'] for result in dopamine_results]

colors_risk = ['red', 'orange', 'green']
bars = ax4.bar(game_names, dopamine_avgs, color=colors_risk, edgecolor='black', linewidth=1.5, alpha=0.7)

# Add tolerance decline as error bars
ax4.errorbar(range(len(game_names)), dopamine_avgs, 
            yerr=[t/100 * 0.3 for t in tolerance_develops],
            fmt='none', ecolor='black', linewidth=2, capsize=5, label='Tolerance Decline')

ax4.set_ylabel('Average Dopamine Response (0-1)')
ax4.set_title('Panel 4: Addiction Risk by Game Type\n(Dopamine & Tolerance Development)')
ax4.set_ylim(0, 0.8)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Add risk labels
for bar, game, dop in zip(bars, game_names, dopamine_avgs):
    risk = 'HIGH' if dop > 0.5 else 'MODERATE' if dop > 0.3 else 'LOW'
    ax4.text(bar.get_x() + bar.get_width()/2, dop + 0.05, risk,
            ha='center', fontweight='bold', fontsize=10, color=bar.get_facecolor())

plt.tight_layout()
plt.savefig('behavioral_economics_addiction.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("• Gambler's fallacy escalates losses exponentially (chasing behavior)")
print("• Addiction progression: 3-4 year trajectory from social to severe")
print("• Near-misses extend play 30-120% longer, amplifying losses")
print("• Slot machines trigger highest dopamine (variable ratio schedule)")
print("• Tolerance develops rapidly: 30%+ sensitivity loss within 100 bets")
print("• Intervention critical: Untreated progression nearly 100% to ruin")
print("="*80 + "\n")
```

---

## VII. References & Key Design Insights

1. **Kahneman, D., & Tversky, A. (1979).** "Prospect Theory: An Analysis of Decision under Risk."
   - Loss aversion, prospect theory foundations

2. **Natarajan, R. (2016).** "The Neuroscience of Preference and Choice," from *The Handbook of Neuroeconomics.*
   - Dopamine, reward systems, addiction neurobiology

3. **Gainsbury, S. M., & Blaszczynski, A. (2011).** "A Systematic Review of Internet-Based Therapy for Mental Disorders."
   - Online gambling harms, addiction treatment efficacy

4. **Schüll, N. D. (2012).** "Addiction by Design: Machine Gambling in Las Vegas."
   - Casino design psychology, slot machine mechanics

**Key Design Concepts:**

- **Cognitive Biases = Market Inefficiencies:** Gamblers systematically misperceive probability; casinos exploit these blind spots architecturally and mathematically.
- **Dopamine ≠ Value:** Brain's reward signal (dopamine) mismatches with financial reality (negative EV); variable ratio schedules create maximum misalignment.
- **Addiction = Progressive Neurological Change:** Not moral failure; dopamine dysregulation requires clinical intervention.
- **Responsible Gambling = Systems Approach:** Individual willpower insufficient; requires structural limits (deposit caps, time limits, loss limits), access to treatment, and community support.

