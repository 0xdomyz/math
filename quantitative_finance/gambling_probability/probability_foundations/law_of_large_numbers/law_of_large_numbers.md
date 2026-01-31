# Law of Large Numbers: Convergence to Expected Value

## 1. Concept Skeleton
**Definition:** As n→∞, empirical frequency of event converges to theoretical probability; average outcome converges to expected value  
**Purpose:** Justifies why casinos profit long-term (house edge materializes); explains why short-term variance doesn't disprove negative EV  
**Prerequisites:** Expected value, variance, probability basics, limits

## 2. Comparative Framing
| Concept | Law of Large Numbers | Central Limit Theorem | Law of Iterated Logarithm | Regression to Mean |
|---------|-------------------|---------------------|--------------------------|-------------------|
| **Statement** | Average → E[X] as n→∞ | Distribution → Normal as n→∞ | |Avg - E[X]| ~ √(ln ln n) | Extreme → average after time |
| **Rate** | 1/√n | 1/√n | ln ln n (slower) | Varies by domain |
| **Application** | House edge certainty | Profit margins, CI | Clustered outcomes | Regression effect |
| **Example** | Roulette EV -2.7% achieved | Win/loss distribution shapes | Streaks length bounds | Hot hand fades |
| **Implication** | More bets = more sure loss | Range of outcomes predictable | Long streaks possible | Don't extend after luck |
| **Gambling** | Casino certainty | Bankroll risk modeling | Tilt after bad streak | Tilt reversal (false) |

## 3. Examples + Counterexamples

**Simple Example:**  
Coin flips: 10 flips might give 8 heads (0.80), 100 flips 48 heads (0.48), 10,000 flips 5,047 heads (0.5047) → convergence to 0.5

**Failure Case:**  
Assuming convergence in short-term: Roulette loss of 1,000 USD → "Long-term I'll recover." False: Convergence is asymptotic; finite n has variance

**Edge Case:**  
Path of loss: −1000 → −1500 → −500 → −2000. Average still converging to −2.7% of total bet, but sequence highly volatile

## 4. Layer Breakdown
```
Law of Large Numbers Framework:
├─ I. DEFINITION & MATHEMATICAL STATEMENT:
│   ├─ Weak Law (WLLN):
│   │   ├─ Statement: P(|S_n/n - μ| > ε) → 0 as n→∞
│   │   ├─ Translation: Probability that average deviates from E[X] by ε vanishes
│   │   ├─ Meaning: For large n, empirical average ≈ theoretical mean
│   │   ├─ Example: 100,000 roulette bets average ≈ -2.7% (almost certainly)
│   │   ├─ Implication: House edge WILL materialize if play long enough
│   │   └─ Loophole: Still requires sufficient capital (can bust before convergence)
│   ├─ Strong Law (SLLN):
│   │   ├─ Statement: P(lim_{n→∞} S_n/n = μ) = 1
│   │   ├─ Translation: Almost surely (probability 1), average → mean
│   │   ├─ Stronger: Guarantees limit exists, not just convergence in probability
│   │   ├─ Math: Requires finite variance (well-defined mean)
│   │   ├─ Application: With infinite capital and no time constraint, convergence certain
│   │   └─ Gambling: Theoretical, not practical (infinite capital impossible)
│   ├─ Expected Value Definition:
│   │   ├─ E[X] = Σ x_i × P(X = x_i) (discrete)
│   │   ├─ Example: Roulette, bet $1 on red
│   │   │           Outcomes: +$1 (prob 18/37), -$1 (prob 19/37)
│   │   │           E[X] = 1×(18/37) + (-1)×(19/37) = -1/37 ≈ -$0.027
│   │   ├─ Interpretation: Average loss per bet
│   │   └─ Multiplier: n bets → Expected total loss ≈ n × E[X]
│   ├─ Rate of Convergence:
│   │   ├─ Standard error: SE = σ / √n
│   │   ├─ Meaning: Deviation from mean scales as 1/√n
│   │   ├─ Implication: Quadrupling bets → halve deviation range
│   │   ├─ Example: 100 bets SE=σ/10, 10,000 bets SE=σ/100 (10× smaller)
│   │   └─ Gambling: Doubling bets doesn't double certainty (only √2× more certain)
│   └─ Convergence Guarantees:
│       ├─ Non-asymptotic (finite n): Use Chebyshev, Chernoff bounds
│       ├─ Chebyshev: P(|S_n/n - μ| > ε) ≤ σ²/(n×ε²)
│       ├─ Chernoff: Exponential decay with n
│       └─ Practical: n=1000 → high confidence; n=10 → high variance still
├─ II. INTUITION & MECHANICS:
│   ├─ Averaging Process:
│   │   ├─ First few bets: Huge variance possible
│   │   ├─ Win sequence: +$200 from 5 bets (lucky early)
│   │   ├─ Mixed: Luck cancels over large n
│   │   ├─ Long-term: Luck component ≈ 0% of total profit/loss
│   │   ├─ System: (Luck + Skill×n) / n → Skill as n→∞ (if skill exists)
│   │   └─ Implication: Randomness becomes negligible; true edge determines result
│   ├─ Scaling with n:
│   │   ├─ Total profit/loss: S_n = X_1 + X_2 + ... + X_n
│   │   ├─ Average: S_n / n → E[X] (what LLN claims)
│   │   ├─ Variance scaling:
│   │   │   ├─ Var(S_n) = Σ Var(X_i) = n × Var(X) (if independent)
│   │   │   ├─ Std(S_n) = √n × σ
│   │   │   ├─ Std(S_n / n) = σ / √n (shrinks!)
│   │   │   └─ Implication: Average becomes more stable as n grows
│   │   └─ Example Scaling:
│   │       ├─ 1 bet: Range ±$37 typical (from ±1σ)
│   │       ├─ 100 bets: Average ±$3.7 typical
│   │       ├─ 10,000 bets: Average ±$0.37 typical (concentrated near -$27)
│   │       └─ Intuition: Big picture emerges from massive data
│   └─ Rare Events Become Common:
│       ├─ P(extreme outcome | single bet) = tiny
│       ├─ P(extreme outcome | 1000 independent events) = not negligible
│       ├─ Example: Lose 10 straight ≈ rare (1/1024 × bet)
│       ├─ But over 10,000 sequences: Expect ~10 such streaks
│       └─ Implication: Streaks are NORMAL in random data (not evidence of bias)
├─ III. CASINO ADVANTAGE & CONVERGENCE:
│   ├─ House Edge Definition:
│   │   ├─ From player perspective: E[payout] < $1 per $1 bet
│   │   ├─ Example Roulette (American, 0,00):
│   │   │   ├─ Bet $1 red: P(win)=18/38, P(lose)=20/38
│   │   │   ├─ E[profit] = 1×(18/38) - 1×(20/38) = -2/38 ≈ -5.26%
│   │   ├─ House Edge = |E[payout] - 1|
│   │   └─ House guarantees: Long enough play → guaranteed loss
│   ├─ Convergence Timeline:
│   │   ├─ 100 bets: High variance, possible win (20-30% chance)
│   │   ├─ 1,000 bets: Win possible but unlikely (< 5% chance with σ=1.5)
│   │   ├─ 10,000 bets: Win almost impossible (99.99%+ probability of loss)
│   │   ├─ 100,000 bets: Loss certain to within rounding (multiple σ away)
│   │   └─ Formula: P(profit after n bets) ≈ Φ(E[X]×√n / σ) (normal approx)
│   ├─ Scaling Trap:
│   │   ├─ Myth: "If I bet more, I'll escape the edge"
│   │   ├─ Truth: Expected loss scales linearly with n and bet size
│   │   ├─ If Edge = 5%, then E[loss] = 5% × n × bet_size
│   │   ├─ Doubling bets → double expected loss
│   │   ├─ 10× more bets → 10× expected loss
│   │   └─ Implication: Scaling up accelerates convergence to loss
│   └─ Bankroll vs Convergence Time:
│       ├─ Ruin before convergence: Possible if bankroll small
│       ├─ Example: 10,000 needed for convergence, bankroll 1,000
│       ├─ With variance σ²=100, ruin possible before convergence
│       ├─ Risk of Ruin formula: R = [1-(E[X]/σ)²]^(B/E[X]²) where B=bankroll
│       └─ Implication: Negative EV games → certain ruin given enough time
├─ IV. MATHEMATICAL TREATMENT:
│   ├─ Formal Statement (Weak LLN):
│   │   ├─ Let X_i ~ iid with E[X_i]=μ, Var(X_i)=σ² < ∞
│   │   ├─ Define S_n = X_1 + X_2 + ... + X_n
│   │   ├─ Then: ∀ε > 0, lim_{n→∞} P(|S_n/n - μ| > ε) = 0
│   │   ├─ Proof sketch: Use Chebyshev inequality
│   │   │   ├─ P(|S_n/n - μ| > ε) ≤ Var(S_n/n) / ε²
│   │   │   ├─ = σ² / (n×ε²) → 0 as n→∞
│   │   └─ Rigor: Requires finite variance
│   ├─ Chebyshev Inequality (Quantifies Convergence):
│   │   ├─ Formula: P(|X - μ| ≥ k×σ) ≤ 1/k²
│   │   ├─ For average: P(|S_n/n - μ| ≥ ε) ≤ σ² / (n×ε²)
│   │   ├─ Example (Roulette σ ≈ 1.5, μ ≈ -0.027):
│   │   │   ├─ n=1000: P(|avg - μ| ≥ 0.1) ≤ 1.5² / (1000×0.1²) ≈ 2.25
│   │   │   ├─ Too loose! (can't bound probability > 1)
│   │   │   └─ Better: Use normal approximation or tighter bounds
│   │   └─ Utility: Worst-case guarantee, often conservative
│   ├─ Berry-Esseen Theorem (Error Bounds):
│   │   ├─ Statement: | F_n(x) - Φ(x) | ≤ C × ρ / σ³ / √n
│   │   ├─ Components: ρ = E[|X - μ|³] (3rd absolute moment)
│   │   ├─ C = 0.56 (universal constant)
│   │   ├─ Implication: Error in normal approximation shrinks at rate 1/√n
│   │   └─ Application: Estimate how quickly CLT kicks in
│   └─ Strong Law (Almost Sure Convergence):
│       ├─ Statement: P(lim_{n→∞} S_n/n = μ) = 1
│       ├─ Meaning: With probability 1, limit exists
│       ├─ Requirement: Only need finite mean (variance not required!)
│       ├─ Application: Even heavy-tailed distributions converge almost surely
│       └─ Implication: House edge WILL materialize (probability 1)
├─ V. VARIANCE & CONFIDENCE INTERVALS:
│   ├─ Confidence Interval Construction:
│   │   ├─ 95% CI for average after n bets:
│   │   │   ├─ Sample average ± 1.96 × σ / √n
│   │   │   ├─ Example: n=10,000, σ≈1 (roulette)
│   │   │   ├─ CI: avg ± 1.96/100 ≈ avg ± 0.02
│   │   │   └─ Interpretation: 95% confident true avg in range
│   │   ├─ Shrinking range with n:
│   │   │   ├─ n=100: CI ≈ ±0.2
│   │   │   ├─ n=1,000: CI ≈ ±0.06
│   │   │   ├─ n=10,000: CI ≈ ±0.02
│   │   │   └─ Implication: Larger sample → narrower range, higher certainty
│   │   └─ Accuracy vs confidence trade-off:
│   │       ├─ 95% vs 99% CI: 99% wider (more conservative)
│   │       ├─ n=1,000 vs n=10,000: 1,000 wider
│   │       └─ Choice: Balance between certainty and precision
│   ├─ Practical Implications for Gambling:
│   │   ├─ Session (100 bets): ±10-20% expected variation (realistic for short session)
│   │   ├─ Daily (1,000 bets): ±3-6% expected variation (more predictable)
│   │   ├─ Weekly (10,000 bets): ±1-2% expected variation (nearly certain loss)
│   │   └─ Formula: EV ± z × σ / √n (where z=1.96 for 95%)
│   └─ Illusion of Control Trap:
│       ├─ Belief: "I'm different; I won't experience variance"
│       ├─ Truth: Variance unavoidable; only magnitude shrinks
│       ├─ Implication: Even with perfect strategy, short-term swings expected
│       └─ Solution: Budget for variance; don't expect smooth convergence
├─ VI. APPLICATIONS IN GAMBLING STRATEGY:
│   ├─ Bankroll Management:
│   │   ├─ Risk of Ruin (Gambler's Ruin):
│   │   │   ├─ Formula: R = exp(-2×EV×B/σ²) (for small edge)
│   │   │   ├─ Components: EV = expected value per bet
│   │   │   ├─ B = bankroll size
│   │   │   ├─ σ² = variance per bet
│   │   │   ├─ Example: EV = -1%, σ² = 1, B = $1000
│   │   │   ├─ R = exp(-2×(-0.01)×1000/1) = exp(20) ≈ 0.9999+ (almost certain ruin)
│   │   │   └─ Implication: Negative EV guarantees ruin; only question is when
│   │   ├─ Kelly Criterion (Positive EV):
│   │   │   ├─ Optimal bet size: f = EV / σ²
│   │   │   ├─ Application: When EV > 0, maximize growth
│   │   │   ├─ Example: +2% EV, σ²=4 → bet 0.5% of bankroll per wager
│   │   │   ├─ Properties: Maximizes expected log wealth (long-term growth)
│   │   │   └─ Caution: Overbetting leads to ruin; conservative Kelly (f/2) recommended
│   │   └─ Fixed Bet vs Proportional:
│   │       ├─ Fixed: Bet same amount always (simplest)
│   │       ├─ Proportional: Bet % of current bankroll (Kelly, conservative)
│   │       ├─ Martingale: Double after loss (false security, leads to ruin)
│   │       └─ Lesson: LLN implies fixed bet works for positive EV; martingale doesn't fix negative
│   ├─ Bet Duration & Time Horizon:
│   │   ├─ Short-term (100s of bets): Variance dominates, luck can overcome edge
│   │   ├─ Medium-term (1000s): Edge becomes apparent, variance ±5-10%
│   │   ├─ Long-term (10000s+): Edge dominates, variance ±1-2%
│   │   ├─ Strategy: If positive EV, increase time horizon (more bets = more certain profit)
│   │   ├─ Inverse: If negative EV, quit early (don't play long enough for LLN)
│   │   └─ Practical: Casino has positive EV, plays billions of hands → guaranteed profit
│   ├─ Detecting Bias/Cheating:
│   │   ├─ Method: Track long-term frequency vs theoretical
│   │   ├─ Example: Roulette should hit red 18/37 times
│   │   ├─ Observation: 10,000 spins, red appeared 5,200 times (52%)
│   │   ├─ Analysis: Expected ≈ 4,865, observed = 5,200 → SE ≈ 70
│   │   ├─ Z-score: (5,200 - 4,865) / 70 ≈ 4.8 → p < 0.0001 (highly significant)
│   │   └─ Conclusion: Likely biased wheel (or cheating)
│   └─ Simulations for Exploration:
│       ├─ Monte Carlo: Generate thousands of independent games
│       ├─ Track: Average outcome, range, distribution
│       ├─ Compare: To theoretical predictions
│       ├─ Benefit: Visualize convergence; see variance effects
│       └─ Application: Understand risk before real money exposure
├─ VII. CAVEATS & LIMITATIONS:
│   ├─ Requires Independence:
│   │   ├─ LLN assumes iid samples
│   │   ├─ Violation: Correlated outcomes (feedback, trending)
│   │   ├─ Example: Tilt after loss → future bets riskier (correlation)
│   │   ├─ Effect: LLN may not apply or convergence much slower
│   │   └─ Lesson: Ensure independence; control for psychological factors
│   ├─ Finite Variance Requirement:
│   │   ├─ LLN needs finite variance (weak law) or just finite mean (strong law)
│   │   ├─ Exception: Heavy-tailed distributions (e.g., Cauchy) - no mean
│   │   ├─ Gambling: Most games have finite variance, OK
│   │   └─ Caution: Exotic bets (e.g., arbitrage with explosive gains) need care
│   ├─ Time-Horizon Problem:
│   │   ├─ LLN is asymptotic (n→∞)
│   │   ├─ Practical question: How large is n?
│   │   ├─ Answer: Depends on variance and required accuracy
│   │   ├─ Example: σ ≈ 1, want 95% confidence edge materializes
│   │   │   ├─ n ≈ (1.96/edge)² ≈ (1.96/0.027)² ≈ 5,300 bets
│   │   │   ├─ At 1 bet/min: ~90 hours of play
│   │   │   └─ For casinos: Billions of hands/year, converged
│   │   └─ Implication: Player may never live long enough; casino will
│   ├─ Ruin Before Convergence:
│   │   ├─ Classic problem: Negative EV guarantees ruin
│   │   ├─ But if bankroll too small, ruin before long-term plays out
│   │   ├─ Example: Edge = -2.7%, need ~10,000 bets for convergence
│   │   ├─ With bankroll $100, average loss/bet = $2.70
│   │   ├─ Expected total loss = $27,000 → bankrupt after ~37 bets
│   │   └─ Lesson: Insufficient capital prevents convergence
│   ├─ Non-Stationarity:
│   │   ├─ LLN assumes distribution unchanged (stationary)
│   │   ├─ Real gambling: Dealer changes, table changes, psychology shifts
│   │   ├─ Effect: Violations compound error
│   │   ├─ Example: Strong play (edge +1%) then drunk (edge -2%)
│   │   ├─ Average: Close to zero despite changing conditions
│   │   └─ Lesson: Track for changes; LLN assumes stable system
│   └─ Regression to the Mean (Misleading):
│       ├─ Observation: After lucky streak, performance regresses
│       ├─ Misinterpretation: "Luck ran out; predict future regression"
│       ├─ Truth: Expected regression due to randomness, not mystical
│       ├─ Math: E[2nd measurement | 1st extreme] < 1st measurement (regression coefficient < 1)
│       ├─ Example: Extremely lucky first 100 bets (+10%) → expect closer to 0% (convergence, not reversal)
│       └─ Lesson: Regression is automatic; no predictive power
└─ VIII. FORMULAS & CALCULATIONS:
    ├─ Expected Value:
    │   ├─ E[X] = Σ x_i P(x_i)
    │   ├─ EV after n bets: E[S_n] = n × E[X]
    │   └─ Average: E[S_n/n] = E[X]
    ├─ Variance & Standard Deviation:
    │   ├─ Var(X) = E[(X - μ)²]
    │   ├─ Var(S_n) = n × Var(X) (independent)
    │   ├─ SD(S_n) = √n × σ
    │   └─ SD(average) = σ / √n
    ├─ Chebyshev Bound:
    │   ├─ P(|S_n/n - μ| ≥ ε) ≤ σ² / (n × ε²)
    │   └─ Rewrite: P(|S_n/n - μ| ≤ ε) ≥ 1 - σ² / (n × ε²)
    ├─ Normal Approximation (CLT):
    │   ├─ S_n/n ≈ N(μ, σ²/n)
    │   ├─ Standardize: Z = (S_n/n - μ) / (σ/√n) ≈ N(0,1)
    │   └─ Confidence: P(|Z| ≤ 1.96) ≈ 0.95 (95% CI)
    └─ House Edge Scaling:
        ├─ Expected loss: E[loss] = |edge| × bet × n
        ├─ Example: 2.7% edge, $1 bet, 1000 hands → E[loss] = 0.027 × 1 × 1000 = $27
        └─ Generalization: Scale linearly with bet size and number of hands
```

**Core Insight:** LLN guarantees edge materializes with time; understanding convergence rate explains bankroll needs and time horizons.

## 5. Mini-Project
Simulate convergence to expected value across games with confidence bands:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

np.random.seed(42)

print("="*70)
print("LAW OF LARGE NUMBERS: CONVERGENCE TO EXPECTED VALUE")
print("="*70)

# ============================================================================
# 1. SINGLE GAME SIMULATION: ROULETTE
# ============================================================================

print("\n" + "="*70)
print("1. ROULETTE: CONVERGENCE WITH CONFIDENCE BANDS")
print("="*70)

class RouletteGame:
    """European roulette: -1 loss with prob 19/37, +1 win with prob 18/37."""
    
    def __init__(self):
        self.p_win = 18/37
        self.p_loss = 19/37
        self.ev = 1 * self.p_win + (-1) * self.p_loss  # -1/37
        self.variance = (1 - self.ev)**2 * self.p_win + (-1 - self.ev)**2 * self.p_loss
        self.std = np.sqrt(self.variance)
    
    def simulate_bets(self, n_bets):
        """Simulate n consecutive $1 bets; return running average."""
        outcomes = np.random.choice([1, -1], size=n_bets, p=[self.p_win, self.p_loss])
        cumsum = np.cumsum(outcomes)
        average = cumsum / np.arange(1, n_bets + 1)
        return average, outcomes, cumsum

# Theoretical values
game = RouletteGame()
print(f"\nRoulette Theoretical Values:")
print(f"   E[X] = {game.ev:.6f} (≈ -$0.027 per bet)")
print(f"   Variance = {game.variance:.6f}")
print(f"   Std Dev = {game.std:.6f}")

# Simulate many paths
n_bets = 10000
n_simulations = 100

all_averages = []
for sim in range(n_simulations):
    avg, _, _ = game.simulate_bets(n_bets)
    all_averages.append(avg)

all_averages = np.array(all_averages)

# Compute statistics along the path
mean_average = np.mean(all_averages, axis=0)
std_average = np.std(all_averages, axis=0, ddof=1)

# Confidence bands (±1.96 SD for 95% CI)
z_score = 1.96
upper_band = mean_average + z_score * std_average
lower_band = mean_average - z_score * std_average

# Theoretical SE
theoretical_se = game.std / np.sqrt(np.arange(1, n_bets + 1))
theoretical_upper = game.ev + z_score * theoretical_se
theoretical_lower = game.ev - z_score * theoretical_se

print(f"\nSimulation Results ({n_simulations} paths, {n_bets} bets each):")
print(f"   At n=100: Average ± SE = {mean_average[99]:.4f} ± {std_average[99]:.4f}")
print(f"   At n=1000: Average ± SE = {mean_average[999]:.4f} ± {std_average[999]:.4f}")
print(f"   At n=10000: Average ± SE = {mean_average[9999]:.4f} ± {std_average[9999]:.4f}")
print(f"   Theoretical EV = {game.ev:.6f}")
print(f"   Convergence? {abs(mean_average[9999] - game.ev) < 0.01}")

# Probability of profit at different n
prob_profit = np.mean(all_averages > 0, axis=0)

print(f"\n   P(profit | n bets):")
for n_test in [100, 500, 1000, 5000, 10000]:
    idx = n_test - 1
    print(f"      n={n_test}: {prob_profit[idx]:.3f} ({int(prob_profit[idx]*100)}%)")

# ============================================================================
# 2. MULTIPLE GAME COMPARISON
# ============================================================================

print("\n" + "="*70)
print("2. COMPARING CONVERGENCE ACROSS GAMES")
print("="*70)

class CasinoGame:
    """Generic casino game with specified EV and variance."""
    
    def __init__(self, name, ev, std):
        self.name = name
        self.ev = ev
        self.std = std
    
    def simulate_bets(self, n_bets, n_sims=100):
        """Simulate n_bets for n_sims independent paths."""
        paths = []
        for _ in range(n_sims):
            outcomes = np.random.normal(self.ev, self.std, n_bets)
            cumsum = np.cumsum(outcomes)
            average = cumsum / np.arange(1, n_bets + 1)
            paths.append(average)
        return np.array(paths)

# Different games
games = [
    CasinoGame("Roulette (EU)", ev=-1/37, std=game.std),
    CasinoGame("Blackjack", ev=-0.005, std=1.2),
    CasinoGame("Craps", ev=-0.014, std=1.1),
    CasinoGame("Slots", ev=-0.05, std=2.0)
]

n_bets_comparison = 1000
results_comparison = []

for game_obj in games:
    paths = game_obj.simulate_bets(n_bets_comparison, n_sims=50)
    mean_path = np.mean(paths, axis=0)
    results_comparison.append({
        'game': game_obj.name,
        'ev': game_obj.ev,
        'mean_path': mean_path
    })

print(f"\nGame Comparison ({n_bets_comparison} bets, 50 simulations each):")
for res in results_comparison:
    final_avg = res['mean_path'][-1]
    print(f"   {res['game']:20s}: EV={res['ev']:7.4f}, Avg after {n_bets_comparison} bets={final_avg:7.4f}")

# ============================================================================
# 3. BANKROLL DYNAMICS
# ============================================================================

print("\n" + "="*70)
print("3. BANKROLL EVOLUTION UNDER LAW OF LARGE NUMBERS")
print("="*70)

def simulate_bankroll(ev, std, n_bets, initial_bankroll, bet_size=1.0):
    """Simulate bankroll evolution; stop if bankruptcy."""
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_placed = 0
    
    for bet in range(n_bets):
        # Bet
        if bankroll <= 0:
            break  # Bankrupt
        
        # Outcome
        outcome = np.random.normal(ev, std)
        bankroll += bet_size * outcome
        bankroll_history.append(bankroll)
        bets_placed += 1
    
    return np.array(bankroll_history), bets_placed

# Scenarios
scenarios = [
    {'name': 'Roulette, $1 bet', 'ev': -1/37, 'std': game.std, 'initial': 100, 'bet': 1},
    {'name': 'Blackjack, $5 bet', 'ev': -0.005, 'std': 1.2, 'initial': 500, 'bet': 5},
    {'name': 'Slots, $0.25 bet', 'ev': -0.05, 'std': 2.0, 'initial': 100, 'bet': 0.25}
]

print(f"\nBankroll Trajectories (representative paths):")
for scenario in scenarios:
    bankroll, bets = simulate_bankroll(scenario['ev'], scenario['std'], 5000, 
                                       scenario['initial'], scenario['bet'])
    final_bankroll = bankroll[-1]
    expected_loss = scenario['ev'] * scenario['bet'] * bets
    print(f"   {scenario['name']}:")
    print(f"      Bets completed: {bets}")
    print(f"      Final bankroll: ${final_bankroll:.2f} (started ${scenario['initial']:.2f})")
    print(f"      Expected loss: ${-expected_loss:.2f}")

# ============================================================================
# 4. CONVERGENCE RATE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("4. CONVERGENCE RATE: HOW QUICKLY DOES LLN KICK IN?")
print("="*70)

def compute_convergence_n_for_accuracy(ev, std, target_accuracy, confidence=0.95):
    """
    Compute n needed for convergence to ev within ±target_accuracy at given confidence.
    
    From normal approximation:
    P(|average - ev| ≤ target_accuracy) ≈ Φ(target_accuracy × √n / std)
    
    For confidence level (e.g., 0.95), find z such that Φ(z) = (1+confidence)/2
    Then: z = target_accuracy × √n / std
          n = (z × std / target_accuracy)²
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    n = (z * std / target_accuracy) ** 2
    return int(np.ceil(n))

# For different accuracies
print(f"\nSample Size Needed for Convergence (Roulette):")
target_accuracies = [0.01, 0.05, 0.1, 0.2]

for accuracy in target_accuracies:
    n_needed = compute_convergence_n_for_accuracy(game.ev, game.std, accuracy, confidence=0.95)
    hours_of_play = n_needed / 60  # Assuming 1 bet/minute
    print(f"   Accuracy ±${accuracy:.2f} (95% CI): n={n_needed:6d} bets (~{hours_of_play:6.0f} hours)")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Roulette convergence with confidence bands
ax1 = axes[0, 0]
bets_plot = np.arange(1, n_bets + 1)
ax1.plot(bets_plot, mean_average, 'b-', linewidth=2, label='Mean of 100 paths')
ax1.fill_between(bets_plot, lower_band, upper_band, alpha=0.3, label='95% Confidence Band')
ax1.axhline(y=game.ev, color='red', linestyle='--', linewidth=2, label='True EV')
ax1.set_xlabel('Number of Bets')
ax1.set_ylabel('Average Outcome ($)')
ax1.set_title('Roulette: LLN Convergence (100 simulations)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Multiple games convergence
ax2 = axes[0, 1]
for res in results_comparison:
    ax2.plot(np.arange(1, len(res['mean_path'])+1), res['mean_path'], 
            linewidth=2, label=res['game'])
ax2.set_xlabel('Number of Bets')
ax2.set_ylabel('Average Outcome ($)')
ax2.set_title('Convergence Comparison: Different Games')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Probability of profit vs n
ax3 = axes[1, 0]
ax3.plot(np.arange(1, len(prob_profit)+1), prob_profit * 100, 'o-', linewidth=2, markersize=3)
ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% break-even')
ax3.fill_between(np.arange(1, len(prob_profit)+1), 0, prob_profit*100, alpha=0.3)
ax3.set_xlabel('Number of Bets')
ax3.set_ylabel('P(Profit) %')
ax3.set_title('Roulette: Probability of Positive Return')
ax3.set_ylim(0, 100)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Sample size for accuracy
ax4 = axes[1, 1]
accuracy_range = np.linspace(0.01, 0.3, 30)
n_required = []

for acc in accuracy_range:
    n_needed = compute_convergence_n_for_accuracy(game.ev, game.std, acc, confidence=0.95)
    n_required.append(n_needed)

ax4.semilogy(accuracy_range, n_required, 'o-', linewidth=2, markersize=5, color='purple')
ax4.set_xlabel('Target Accuracy (±$)')
ax4.set_ylabel('Sample Size Required (log scale)')
ax4.set_title('Convergence: Required Sample Size vs Accuracy')
ax4.grid(True, alpha=0.3, which='both')
ax4.invert_xaxis()  # Higher accuracy on right

plt.tight_layout()
plt.savefig('law_of_large_numbers.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: law_of_large_numbers.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY: Law of Large Numbers in Gambling")
print("="*70)
print("✓ Average converges to EV as n→∞ (mathematical certainty)")
print("✓ Convergence rate: SD decreases as 1/√n")
print("✓ House edge GUARANTEES long-term loss for negative EV games")
print("✓ Time horizon critical: Need ~1000s of bets for edge to materialize")
print("✓ Bankroll size must sustain variance until convergence (else ruin before convergence)")
```

## 6. Challenge Round
**When does the Law of Large Numbers fail or give false confidence?**
- Non-independent draws: Correlated outcomes violate iid assumption → convergence slower or non-existent
- Infinite variance: Heavy-tailed distributions → LLN might not apply to average (only to median)
- Non-stationarity: Distribution changes over time → LLN computes average of changing parameters (misleading)
- Ruin before convergence: Negative EV with finite bankroll → bankruptcy before LLN kicks in
- Misinterpretation: Seeing convergence starting doesn't mean trend will continue (regression to mean confusion)

## 7. Key References
- [Feller (1968), "An Introduction to Probability Theory"](https://www.amazon.com/Introduction-Probability-Theory-Applications-Vol/dp/0471257087) - Rigorous mathematical treatment
- [Blackwell & Girshick (1954), "Theory of Games and Statistical Decisions"](https://www.amazon.com/Theory-Games-Statistical-Decisions-Blackwell/dp/0486639170) - Connections to decision theory
- [Thorp (1962), "Beat the Dealer"](https://en.wikipedia.org/wiki/Edward_Thorp) - Practical exploitation of dependence despite LLN pressure

---
**Status:** Explains certainty of house edge and importance of time horizon | **Complements:** Expected value, Variance, Central Limit Theorem | **Enables:** Bankroll management, risk of ruin calculations