# Market Efficiency Tests

## 1. Concept Skeleton
**Definition:** Empirical tests of whether asset prices fully reflect all available information (Efficient Market Hypothesis)  
**Purpose:** Assess market predictability, identify anomalies, evaluate active management vs indexing, test behavioral biases  
**Prerequisites:** EMH theory (Fama 1970), event study methodology, statistical hypothesis testing, return predictability regressions

## 2. Comparative Framing
| EMH Form | Information Set | Implication | Test Method | Typical Result |
|----------|----------------|-------------|-------------|----------------|
| **Weak** | Past prices/returns | Technical analysis useless | Autocorrelation, runs tests | Mostly holds; small momentum |
| **Semi-strong** | All public information | Fundamental analysis useless | Event studies, announcement effects | Mixed; some anomalies persist |
| **Strong** | All information (incl. private) | Insider trading unprofitable | Insider returns, mutual fund performance | Rejected; insiders profit |

## 3. Examples + Counterexamples

**Simple Example:**  
Earnings announcement: Stock jumps 5% within 1 minute (semi-strong efficiency); no further drift over next week (fully incorporated)

**Failure Case:**  
Post-earnings announcement drift (PEAD): Stock drifts 3% over 60 days after surprise; semi-strong efficiency violated (delayed incorporation)

**Edge Case:**  
January effect: Small caps outperform in January; but transaction costs (bid-ask + commissions) exceed 2-3% gain; efficiency after costs

## 4. Layer Breakdown
```
Market Efficiency Tests Structure:
├─ Weak-Form Efficiency (Past Price Information):
│   ├─ Random Walk Hypothesis:
│   │   ├─ Definition: r_t = μ + ε_t where ε_t ~ iid(0, σ²)
│   │   │   ├─ Implication: Returns unpredictable; E[r_{t+1}|r_t, r_{t-1},...] = μ
│   │   │   ├─ Martingale: E[P_{t+1}|P_t] = P_t(1+μ) (expected return constant)
│   │   │   └─ Rationale: All past info in current price; only new info moves price
│   │   ├─ Tests for predictability:
│   │   │   ├─ Autocorrelation test:
│   │   │   │   ├─ H₀: ρ_k = Corr(r_t, r_{t-k}) = 0 for all k>0
│   │   │   │   ├─ Test statistic: t = ρ̂_k / SE(ρ̂_k) ~ N(0,1) under H₀
│   │   │   │   ├─ SE(ρ̂_k) ≈ 1/√n (for large n)
│   │   │   │   ├─ Example: S&P 500 daily ρ̂₁ = 0.03, n=2500 → t = 1.5 (p=0.13; cannot reject)
│   │   │   │   └─ Result: Weak evidence against random walk
│   │   │   ├─ Ljung-Box test (joint test):
│   │   │   │   ├─ Q(m) = n(n+2)Σ[ρ̂²_k/(n-k)] ~ χ²(m)
│   │   │   │   ├─ Tests H₀: ρ₁=ρ₂=...=ρ_m=0 jointly
│   │   │   │   ├─ Example: Q(10) = 18.3 vs χ²(10, 0.05) = 18.3 (borderline)
│   │   │   │   └─ Interpretation: Slight serial correlation but economically small
│   │   │   └─ Variance ratio test (Lo & MacKinlay 1988):
│   │   │       ├─ VR(k) = Var(r_t+...+r_{t-k+1}) / [k·Var(r_t)]
│   │   │       ├─ Random walk: VR(k) = 1 for all k
│   │   │       ├─ Mean reversion: VR(k) < 1; Momentum: VR(k) > 1
│   │   │       ├─ Example: Monthly VR(12) = 0.75 (significant mean reversion at 1-year)
│   │   │       └─ Implication: Weak-form efficiency rejected at long horizons
│   │   ├─ Runs test (sign persistence):
│   │   │   ├─ Count runs: Sequences of same-sign returns
│   │   │   ├─ Expected runs: E[R] = 1 + (n-1)/2 for n observations
│   │   │   ├─ Too few runs → persistence; Too many → reversal
│   │   │   └─ Test: z = (R - E[R])/SE(R) ~ N(0,1)
│   │   └─ Technical analysis profitability:
│   │       ├─ Moving average crossovers: Buy when MA(50)>MA(200)
│   │       ├─ Momentum: Buy past winners, sell past losers
│   │       ├─ Evidence: Small positive returns pre-1990; disappeared post-2000
│   │       └─ Interpretation: Data mining; strategies arbitraged away
│   ├─ Observed Violations:
│   │   ├─ Momentum effect (Jegadeesh & Titman 1993):
│   │   │   ├─ 12-month winners outperform losers by 1% monthly (significant)
│   │   │   ├─ Persistence: Continues 3-12 months; reverses after 3-5 years
│   │   │   ├─ Mechanism: Underreaction to firm-specific news
│   │   │   └─ Profitability: After transaction costs, ~0.5% monthly for institutions
│   │   ├─ Short-term reversals (1-week):
│   │   │   ├─ Weekly winners underperform next week by 0.2%
│   │   │   ├─ Mechanism: Overreaction, bid-ask bounce, liquidity provision
│   │   │   └─ Profitability: Eroded by transaction costs for retail
│   │   └─ Long-term mean reversion (3-5 years):
│   │       ├─ Variance ratio <1 at 3-5 year horizons
│   │       ├─ Contrarian strategies profitable (DeBondt & Thaler 1985)
│   │       └─ Debate: Risk premium variation vs overreaction?
│   └─ Implications:
│       ├─ Technical analysis: Limited value; strategies compete away excess returns
│       ├─ Indexing: Hard to beat via past price patterns
│       └─ Transaction costs: Critical; small predictability ≠ profitable
├─ Semi-Strong Form Efficiency (Public Information):
│   ├─ Event Study Methodology:
│   │   ├─ Framework: Measure abnormal returns around information events
│   │   │   ├─ Events: Earnings announcements, M&A, stock splits, IPOs, regulatory changes
│   │   │   ├─ Event window: [-1, +1] days around announcement (t=0)
│   │   │   ├─ Estimation window: [-120, -21] days (estimate normal returns)
│   │   │   └─ Test: Are abnormal returns = 0 post-event?
│   │   ├─ Abnormal return calculation:
│   │   │   ├─ Normal return: E[R_{it}] = α_i + β_i R_{mt} (market model)
│   │   │   ├─ Estimate α̂_i, β̂_i from estimation window
│   │   │   ├─ Abnormal return: AR_{it} = R_{it} - (α̂_i + β̂_i R_{mt})
│   │   │   └─ Cumulative: CAR(t₁,t₂) = Σ AR_{it} for t∈[t₁,t₂]
│   │   ├─ Statistical testing:
│   │   │   ├─ Cross-sectional test: t = CAR̄ / SE(CAR̄) ~ N(0,1)
│   │   │   ├─ SE(CAR̄) = σ(CAR_i) / √N (N = number of events)
│   │   │   ├─ Example: N=500 earnings announcements; CAR̄[-1,+1] = 2.5%, t=5.2 (p<0.001)
│   │   │   └─ Interpretation: Significant immediate response (efficiency)
│   │   └─ Post-event drift test:
│   │       ├─ Examine CAR[+2, +60] (should be zero if efficient)
│   │       ├─ PEAD (Post-Earnings Announcement Drift): CAR[+2,+60] = 3% (surprise earnings)
│   │       ├─ t = 3.8 (p<0.001) → Drift significant (violation of semi-strong efficiency)
│   │       └─ Mechanism: Underreaction; investors slowly incorporate earnings info
│   ├─ Earnings Announcement Effects:
│   │   ├─ Immediate response (Day 0):
│   │   │   ├─ Positive surprise: +4% average (within 1 minute of release)
│   │   │   ├─ Negative surprise: -3% average
│   │   │   ├─ Volume spike: 5× normal (information absorption)
│   │   │   └─ Consistent with semi-strong efficiency (rapid incorporation)
│   │   ├─ Post-Earnings Announcement Drift (PEAD):
│   │   │   ├─ Discovered: Ball & Brown (1968); persists in modern data
│   │   │   ├─ Magnitude: 2-4% additional drift over 60 days (large surprises)
│   │   │   ├─ Mechanism theories:
│   │   │   │   ├─ Underreaction: Investors anchored to prior beliefs
│   │   │   │   ├─ Risk-based: Omitted risk factor compensation
│   │   │   │   └─ Limits to arbitrage: Small stocks, transaction costs deter arbitrage
│   │   │   ├─ Profitability: Hedge fund strategies exploit; retail unlikely (costs)
│   │   │   └─ Academic consensus: Anomaly persists; challenges semi-strong EMH
│   │   └─ Analyst forecast revisions:
│   │       ├─ Upward revisions → stock rises gradually over months
│   │       ├─ Suggests market underweights analyst information
│   │       └─ Momentum effect partially explained by revision drift
│   ├─ Merger & Acquisition Announcements:
│   │   ├─ Target firm:
│   │   │   ├─ Average premium: 20-40% above pre-announcement price
│   │   │   ├─ Response time: 90%+ gain within 1 day (efficient)
│   │   │   ├─ Post-announcement: Small drift if deal uncertain
│   │   │   └─ Interpretation: Efficient incorporation of takeover premium
│   │   ├─ Acquirer firm:
│   │   │   ├─ Average return: -1% to 0% (slight negative)
│   │   │   ├─ Large acquisitions: -3% (overpayment concerns)
│   │   │   ├─ Post-announcement: Gradual -5% drift over 2 years
│   │   │   └─ Interpretation: Market initially underestimates value destruction
│   │   └─ Arbitrage spread:
│   │       ├─ Deal announced: Target trades 2-5% below offer price (risk arbitrage)
│   │       ├─ Spread reflects: Deal break risk, time value, regulatory uncertainty
│   │       └─ Convergence: Narrows as deal approaches close (informational efficiency)
│   ├─ Calendar Anomalies:
│   │   ├─ January effect:
│   │   │   ├─ Small caps outperform large caps by 5-8% in January
│   │   │   ├─ Mechanism: Tax-loss selling in December; repurchase in January
│   │   │   ├─ Evidence: Strong pre-1990; weakened post-2000 (arbitraged)
│   │   │   └─ Transaction costs: Bid-ask + commissions eat most gain
│   │   ├─ Weekend effect:
│   │   │   ├─ Monday returns: -0.1% average (negative)
│   │   │   ├─ Friday returns: +0.2% average (positive)
│   │   │   ├─ Mechanism: Weekend news pessimism?
│   │   │   └─ Modern data: Effect disappeared post-1990
│   │   └─ Turn-of-the-month effect:
│   │       ├─ Last day + first 3 days of month: +0.5% (significant)
│   │       ├─ Mechanism: Fund flows, window dressing
│   │       └│ Profitability: Small; index funds capture automatically
│   ├─ Value vs Growth Anomaly:
│   │   ├─ Fama-French (1992): Low P/E, P/B stocks outperform high by 7-10% annually
│   │   ├─ Risk-based explanation: Value stocks riskier (distress risk)
│   │   ├─ Behavioral explanation: Overreaction to past growth; undervaluation
│   │   └─ Debate: Risk premium or mispricing? (ongoing controversy)
│   └─ Mutual Fund Performance:
│       ├─ Average active fund: Underperforms index by 1-2% annually (after fees)
│       ├─ Persistence: Past winners ≈ 50/50 future winners (no skill)
│       ├─ Top decile: Outperform by 2-3% (but hard to identify ex-ante)
│       └─ Interpretation: Supports semi-strong efficiency on average
├─ Strong-Form Efficiency (All Information):
│   ├─ Insider Trading Studies:
│   │   ├─ Legal insider trades (Form 4 filings):
│   │   │   ├─ Purchases: Subsequent 6-month return +8% abnormal
│   │   │   ├─ Sales: Subsequent 6-month return -3% abnormal
│   │   │   ├─ Interpretation: Insiders profit from private information
│   │   │   └─ SEC regulation: Must file within 2 days; cannot trade during blackouts
│   │   ├─ Illegal insider trading:
│   │   │   ├─ Pre-announcement purchases: +20-50% returns (massive abnormal)
│   │   │   ├─ Detection: SEC monitors unusual volume/price spikes before news
│   │   │   └─ Penalties: Criminal charges, disgorgement, fines
│   │   └─ Implication: Strong-form efficiency clearly violated (as expected)
│   ├─ Analyst Recommendations:
│   │   ├─ Upgrades: Stock rises 1-2% on announcement day
│   │   ├─ Post-upgrade drift: Additional 3-5% over next 3 months
│   │   ├─ Mechanism: Analysts have superior information (channel checks, models)
│   │   └─ Caveat: Conflicts of interest (investment banking relationships)
│   ├─ Hedge Fund Performance:
│   │   ├─ Average hedge fund: Outperforms by 3-5% annually (pre-fee)
│   │   ├─ After fees (2% + 20%): Slight outperformance 1-2%
│   │   ├─ Top quartile: 8-10% alpha (sophisticated strategies, information edge)
│   │   └─ Interpretation: Some managers have informational/skill advantages
│   └─ Corporate Board Trades:
│       ├─ Board members: Similar abnormal returns to insiders
│       ├─ Independent directors: 4-6% abnormal (slightly less than executives)
│       └─ Timing: Often trade before material events (fiduciary info access)
├─ Joint Hypothesis Problem:
│   ├─ Definition: Tests of market efficiency jointly test EMH + asset pricing model
│   │   ├─ Abnormal return: AR_t = R_t - E[R_t]
│   │   ├─ E[R_t] depends on model (CAPM, Fama-French, etc.)
│   │   ├─ Rejection: Is market inefficient OR model wrong?
│   │   └─ Cannot disentangle without knowing "true" model
│   ├─ Example: Value premium (low P/B outperforms)
│   │   ├─ Efficiency advocate: Value stocks riskier → higher expected return (not mispricing)
│   │   ├─ Inefficiency advocate: Overreaction → value underpriced (behavioral bias)
│   │   └─ Resolution impossible without measuring "true risk"
│   ├─ Fama's response:
│   │   ├─ Most "anomalies" disappear with better risk adjustment (Fama-French factors)
│   │   ├─ Transaction costs eliminate small patterns
│   │   └─ Data mining: 1 in 20 tests significant by chance (p<0.05)
│   └─ Behavioral response:
│       ├─ Patterns persist out-of-sample (momentum, PEAD decades after discovery)
│       ├─ Magnitude too large for risk alone
│       └─ Consistent with known biases (overconfidence, anchoring)
└─ Modern Perspectives:
    ├─ Adaptive Market Hypothesis (Lo 2004):
    │   ├─ Markets approximately efficient most of the time
    │   ├─ Inefficiencies arise and disappear (evolutionary competition)
    │   ├─ Opportunities exist but require innovation
    │   └─ Explains: Anomaly decay after publication (momentum weaker post-2000)
    ├─ Limits to Arbitrage (Shleifer & Vishny 1997):
    │   ├─ Mispricing exists but risky to exploit
    │   ├─ Arbitrage requires capital, short horizon, low leverage
    │   ├─ Example: LTCM 1998 (correct thesis, liquidated early)
    │   └─ Implication: Inefficiencies can persist if arbitrage constrained
    ├─ High-Frequency Trading Impact:
    │   ├─ Price discovery: Faster incorporation (milliseconds vs seconds)
    │   ├─ Liquidity: Tighter spreads (HFT market making)
    │   └─ Flash crashes: Occasional breakdowns (liquidity withdrawal)
    └─ Machine Learning Era:
        ├─ Pattern discovery: ML finds complex predictive signals
        ├─ Decay: Signals weaken as algorithms compete
        └─ Arms race: Continuous innovation required (consistent with adaptive EMH)
```

**Key Insight:** Weak-form efficiency mostly holds (random walk approximation); semi-strong violations persist (PEAD, momentum) but hard to exploit; strong-form clearly fails (insiders profit)

## 5. Mini-Project
Event study: Test semi-strong efficiency with earnings announcements:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulate earnings announcement event study
np.random.seed(42)
n_events = 200  # Number of earnings announcements
window_length = 121  # -60 to +60 days

# Generate market returns
market_returns = np.random.normal(0.0005, 0.01, window_length)

# Generate individual stock parameters (varying beta)
betas = np.random.uniform(0.8, 1.2, n_events)
alphas = np.random.normal(0.0002, 0.0003, n_events)

# Generate stock returns (normal + earnings surprise on day 0)
stock_returns = np.zeros((n_events, window_length))
earnings_surprises = np.random.choice([-1, 1], n_events, p=[0.5, 0.5])  # Positive/negative surprise

for i in range(n_events):
    # Normal returns (market model)
    base_returns = alphas[i] + betas[i] * market_returns + np.random.normal(0, 0.015, window_length)
    
    # Add earnings announcement effect (day 60 = event day 0)
    event_day = 60
    immediate_response = 0.04 * earnings_surprises[i]  # 4% immediate response
    base_returns[event_day] += immediate_response
    
    # Add post-earnings drift (violation of efficiency)
    drift_period = range(event_day + 1, min(event_day + 41, window_length))
    drift_per_day = 0.0005 * earnings_surprises[i]  # 0.05% per day drift
    for j in drift_period:
        base_returns[j] += drift_per_day
    
    stock_returns[i, :] = base_returns

# Calculate abnormal returns
abnormal_returns = np.zeros((n_events, window_length))
for i in range(n_events):
    # Use estimation window (-60 to -11) to estimate alpha, beta
    estimation_window = range(0, 50)
    X = np.column_stack([np.ones(len(estimation_window)), market_returns[estimation_window]])
    y = stock_returns[i, estimation_window]
    
    # OLS regression
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_hat, beta_market = beta_hat[0], beta_hat[1]
    
    # Calculate abnormal returns across full window
    expected_returns = alpha_hat + beta_market * market_returns
    abnormal_returns[i, :] = stock_returns[i, :] - expected_returns

# Average abnormal returns (AAR) and cumulative (CAR)
AAR = abnormal_returns.mean(axis=0)
AAR_se = abnormal_returns.std(axis=0) / np.sqrt(n_events)
CAR = np.cumsum(AAR)

# Event days relative to announcement (day 60 = 0)
event_days = np.arange(-60, 61)

# Statistical tests
# Test 1: Immediate response [-1, +1]
immediate_window = range(59, 62)  # Days -1, 0, +1
CAR_immediate = CAR[61] - CAR[58]  # CAR from -1 to +1
se_immediate = np.sqrt(np.sum(AAR_se[immediate_window]**2))
t_stat_immediate = CAR_immediate / se_immediate

print("="*70)
print("Event Study: Earnings Announcement Efficiency Test")
print("="*70)
print(f"Number of events: {n_events}")
print(f"Event window: Day -60 to +60")
print(f"Estimation window: Day -60 to -11")
print(f"\nImmediate Response [-1, +1]:")
print(f"  CAR: {CAR_immediate*100:>8.3f}%")
print(f"  t-statistic: {t_stat_immediate:>8.3f}")
print(f"  p-value: {stats.t.sf(abs(t_stat_immediate), n_events-1)*2:>8.6f}")
print(f"  Result: {'Significant immediate response' if abs(t_stat_immediate) > 2 else 'No significant response'}")

# Test 2: Post-event drift [+2, +40]
drift_window = range(62, 100)
CAR_drift = CAR[100] - CAR[61]
se_drift = np.sqrt(np.sum(AAR_se[drift_window]**2))
t_stat_drift = CAR_drift / se_drift

print(f"\nPost-Earnings Announcement Drift [+2, +40]:")
print(f"  CAR: {CAR_drift*100:>8.3f}%")
print(f"  t-statistic: {t_stat_drift:>8.3f}")
print(f"  p-value: {stats.t.sf(abs(t_stat_drift), n_events-1)*2:>8.6f}")
print(f"  Result: {'PEAD detected - EFFICIENCY VIOLATED' if abs(t_stat_drift) > 2 else 'No drift - Efficient'}")

# Test 3: Pre-event drift (should be zero)
pre_window = range(20, 59)
CAR_pre = CAR[58] - CAR[19]
se_pre = np.sqrt(np.sum(AAR_se[pre_window]**2))
t_stat_pre = CAR_pre / se_pre

print(f"\nPre-Event Drift [-40, -2]:")
print(f"  CAR: {CAR_pre*100:>8.3f}%")
print(f"  t-statistic: {t_stat_pre:>8.3f}")
print(f"  p-value: {stats.t.sf(abs(t_stat_pre), n_events-1)*2:>8.6f}")
print(f"  Result: {'Information leakage possible' if abs(t_stat_pre) > 2 else 'No pre-event drift'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Average Abnormal Returns (AAR)
axes[0, 0].bar(event_days, AAR*100, color='blue', alpha=0.6)
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Event Day')
axes[0, 0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0, 0].fill_between(event_days, -2*AAR_se*100, 2*AAR_se*100, alpha=0.2, color='gray', label='±2 SE')
axes[0, 0].set_title('Average Abnormal Returns (AAR)')
axes[0, 0].set_xlabel('Event Day')
axes[0, 0].set_ylabel('AAR (%)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xlim(-60, 60)

# Plot 2: Cumulative Abnormal Returns (CAR)
axes[0, 1].plot(event_days, CAR*100, linewidth=2, color='darkblue')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Event Day')
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.8)
# Highlight drift period
axes[0, 1].axvspan(2, 40, alpha=0.2, color='orange', label='Drift Period [+2,+40]')
axes[0, 1].set_title('Cumulative Abnormal Returns (CAR)')
axes[0, 1].set_xlabel('Event Day')
axes[0, 1].set_ylabel('CAR (%)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim(-60, 60)

# Plot 3: Distribution of individual CARs at day +40
CAR_individual_40 = np.cumsum(abnormal_returns[:, 59:100], axis=1)[:, -1]
axes[1, 0].hist(CAR_individual_40*100, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(CAR_individual_40.mean()*100, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {CAR_individual_40.mean()*100:.2f}%')
axes[1, 0].set_title('Distribution of CAR[0,+40] Across Events')
axes[1, 0].set_xlabel('CAR (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: t-statistics over time (statistical significance)
t_stats_rolling = AAR / AAR_se
axes[1, 1].plot(event_days, t_stats_rolling, linewidth=1.5, color='purple')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Event Day')
axes[1, 1].axhline(2, color='orange', linestyle=':', linewidth=1.5, label='Significance (t=2)')
axes[1, 1].axhline(-2, color='orange', linestyle=':', linewidth=1.5)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1, 1].set_title('Statistical Significance of AAR')
axes[1, 1].set_xlabel('Event Day')
axes[1, 1].set_ylabel('t-statistic')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(-60, 60)

plt.tight_layout()
plt.savefig('market_efficiency_event_study.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("Interpretation:")
print(f"{'='*70}")
print("1. Immediate Response: Stock price reacts quickly to earnings surprise")
print("   → Consistent with semi-strong efficiency (public info incorporated)")
print("")
print("2. Post-Earnings Drift: Continued abnormal returns over 40 days")
print("   → VIOLATION of semi-strong efficiency (delayed incorporation)")
print("")
print("3. Economic Significance: ~2% drift (annualized ~18%)")
print("   → Substantial; but transaction costs may reduce profitability")
```

## 6. Challenge Round
When efficiency tests mislead or fail:
- **Data mining bias**: Testing 100 variables; 5 show p<0.05 by chance (spurious anomalies); need out-of-sample validation
- **Look-ahead bias**: Using future data unknowingly (survivor-biased indices; restated earnings); overstates returns
- **Transaction costs ignored**: 0.5% monthly alpha - 0.4% costs = 0.1% profit (not actionable); many "anomalies" disappear after costs
- **Risk adjustment uncertainty**: Value premium 7% - is it mispricing or compensation for distress risk? Cannot definitively answer (joint hypothesis problem)
- **Sample selection**: Testing 1950-2020 excludes Great Depression; post-WWII regime unique (low inflation, stable growth); results may not generalize
- **Decay after publication**: Momentum effect 1.2% monthly pre-1993; 0.5% post-2000 (arbitraged); documented anomalies weaken

## 7. Key References
- [Fama: Efficient Capital Markets Review (1970)](https://www.jstor.org/stable/2325486) - Original EMH framework
- [Ball & Brown: Empirical Evaluation of Accounting Income (1968)](https://www.jstor.org/stable/2490232) - First event study, PEAD discovery
- [MacKinlay: Event Studies in Economics & Finance (1997)](https://www.jstor.org/stable/2729691) - Event study methodology

---
**Status:** Core market microstructure | **Complements:** Asset Return Properties, Behavioral Finance, Portfolio Performance, Anomalies
