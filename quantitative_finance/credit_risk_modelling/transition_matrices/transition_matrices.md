# Transition Matrices (Rating Migration)

## 1. Concept Skeleton
**Definition:** Matrix showing probability that borrower migrates from one credit rating to another within specified period (typically one year)  
**Purpose:** Forecast credit quality changes, estimate multi-year default probability, model rating-based PD trends  
**Prerequisites:** Markov chains, rating systems (S&P, Moody's, internal), historical default data, stochastic modeling

## 2. Comparative Framing
| Framework | Time Orientation | Data Source | Update Frequency | Use |
|-----------|-----------------|------------|------------------|-----|
| **Transition Matrix** | Historical | Rating agencies | Annual | Portfolio dynamics |
| **PIT-PD** | Current conditions | Scorecards, models | Real-time | Immediate underwriting |
| **TTC-PD** | Through-cycle | Smoothed history | Rare updates | Capital budgeting |
| **Forward-Looking** | Projected | Analyst forecasts | Quarterly | Stress testing |

## 3. Examples + Counterexamples

**Simple Example:**  
BBB rating: 90% stay BBB, 5% upgrade to A, 3% downgrade to BB, 2% default. Track 100 BBB bonds over 1 year

**Failure Case:**  
Using pre-2008 transition matrix during crisis; downgrade rates spike 10x. Historical assumptions break in regime change

### 3B. Technical Counterexample: Stationarity Violation and Regime-Shift Default Clustering

**Common Misconception:** "Historical transition matrix for investment-grade corporates shows 2% annual default rate (average 2010-2019). I'll use this to forecast 5-year cumulative default probability: 1-(1-0.02)⁵ = 9.6%. Portfolio capital should cover this tail."

**Why This Fails:** Transition matrices assume stationarity (probabilities constant over time). In reality, default rates spike 5-10× during downturns. Pre-crisis matrix (normal) is worthless for crisis predictions. Multi-year cumulative probability becomes 20-30% (not 9.6%) if cycle included recession.

**Quantitative Example:**

**Normal Times Matrix (2010-2019 data):**
- BBB-rated corporates: 2% annual default rate (estimate from 10 years data)
- 1-year default: 2%
- 2-year cumulative: 1-(1-0.02)² = 3.96%
- 5-year cumulative: 1-(1-0.02)⁵ = 9.57%
- Capital reserved (assuming 8% ratio): $20M per $1B portfolio

**2008-2009 Crisis Actual (included recession):**
- BBB-rated defaults: 8-10% annual (4-5× normal)
- 1-year default: 8%
- 2-year cumulative: 1-(1-0.08)² = 15.4%
- 5-year cumulative: 1-(1-0.08)⁵ = 34% (vs model 9.6%)

**Why Stationarity Breaks:**
1. **Regime Mix:** 2010-2019 was mostly recovery (no recession); BBB default rate ~2% only because economy stable
2. **Through-Cycle vs Point-in-Time:** Historical matrix is Point-in-Time (reflecting current cycle phase). Cumulative PD requires Through-the-Cycle (average across all phases).
3. **Rating Heterogeneity:** BBB-rated corporates in 2019 are safer (lower-risk issuers retained BBB; high-risk downgraded to B). 2008 BBB cohort included riskier names → higher defaults than 2019 BBB cohort.

**Markov Chain Extension:**
- Standard matrix: Assumes probabilities constant
- Regime-switching matrix: Two matrices
  - Normal times: 2% default rate
  - Crisis times: 8% default rate
- Transition between regimes: P(crisis | normal) = 5% per year
- 5-year default: Calculated through regime paths
  - Path 1: Stay normal 5 years: (1-0.02)⁵ = 90.4% (no default)
  - Path 2: Crisis occurs in year 3: (1-0.02)² × (1-0.08)³ = 60% (higher default)
  - Cumulative: Weighted average including all regime paths ≈ 15-20% (vs 9.6% stationary)

**Real-World Case - Moody's Historical Transition Matrix:**
- Pre-2008 Moody's matrix (using 1983-2007 data): BBB 5-year default ≈ 9-10%
- Actual 2008-2012 BBB defaults (with recession): ~20-22%
- Stationarity error: 2.0-2.2×

**Correct Approach:**
1. **Report Through-Cycle PD:** Average PD across full cycle (boom + recession), not just recent period
2. **Regime-Switching Model:** Explicitly model two or more states (healthy economy vs recession)
3. **Stress Testing:** Show 5-year defaults under high-stress regime (unemployment 9-10%, spreads +300bps)
4. **Rating Migration:** Document that same issuer migrates down-grade more quickly in crisis (transition probabilities themselves change)
5. **Marginal vs Cumulative:** Report annual defaults separately from cumulative; cumulative requires regime assumptions

## 4. Layer Breakdown
```
Transition Matrix Framework:
â”œâ”€ Matrix Structure:
â”‚   â”œâ”€ Rows: Current rating (AAA, AA, A, BBB, BB, B, CCC, D)
â”‚   â”œâ”€ Columns: Ending rating (same set + default)
â”‚   â”œâ”€ Entry P[i,j]: Probability rating i â†’ rating j in 1 year
â”‚   â”œâ”€ Row sum = 1 (all outcomes exhaustive)
â”‚   â””â”€ Example: P[BBBâ†’D] = 0.02 (2% default rate)
â”œâ”€ Mathematical Properties:
â”‚   â”œâ”€ Markov chain: Future state depends only on current state
â”‚   â”œâ”€ Stationarity assumption: Matrix stays constant over time
â”‚   â”œâ”€ Multi-period: T-year transition = M^T (matrix power)
â”‚   â”œâ”€ Eigenvalues determine long-run behavior
â”‚   â””â”€ Absorbing state: D (default) is absorbing (P[D,D]=1)
â”œâ”€ Construction Methods:
â”‚   â”œâ”€ Cohort: Track fixed group over time (accurate but slow)
â”‚   â”œâ”€ Duration: Treats each rating duration separately
â”‚   â”œâ”€ Hazard rate: Continuous-time intensity approach
â”‚   â””â”€ Adjusted: Point-in-time vs through-the-cycle
â”œâ”€ Key Patterns:
â”‚   â”œâ”€ Rating drift: Migration from investment â†’ speculative
â”‚   â”œâ”€ Default clustering: In downturns, all ratings see more defaults
â”‚   â”œâ”€ Upgrades rarer: Downgrades > Upgrades in cycle
â”‚   â””â”€ Non-homogeneity: Bank ratings differ from corporate
â”œâ”€ Multi-Year Predictions:
â”‚   â”œâ”€ 2-year default: Use MÂ² to find cumulative PD
â”‚   â”œâ”€ Rating path: Most likely path through intermediate states
â”‚   â””â”€ Distribution: Calculate probability of any ending rating
â””â”€ Practical Applications:
    â”œâ”€ Portfolio analysis: Forecast concentration by rating
    â”œâ”€ Provision calculation: Estimate rating-based EL
    â”œâ”€ Capital modeling: Stress rating migration
    â””â”€ Loss distribution: Combine with LGD for loss scenarios
```

## 5. Challenge Round
When are transition matrices problematic?
- **Stationarity violation**: Matrix changes dramatically with economic cycle; 2008 vs 2019 matrices incomparable
- **Limited data**: Some transitions rare (e.g., AAA â†’ Default); estimates unreliable
- **Rating action lag**: Agencies slow to downgrade; matrix reflects late recognition
- **Cohort effects**: Different cohorts may have different migration (e.g., bonds vs loans)
- **Default definition**: Varies by source (payment vs restructuring vs rating trigger); incomparable matrices

## 6. Key References
- [Markov Chain Rating Dynamics - Wikipedia](https://en.wikipedia.org/wiki/Markov_chain) - Mathematical properties of Markov processes; stationarity assumption; eigenvalue analysis; application to credit rating transitions.

- [Moody's Rating Transitions Report](https://www.moodysanalytics.com/research/insight/2022/rating-transitions) - Historical transition matrices by rating and industry; empirical default rates; through-the-cycle vs point-in-time; documentation of regime changes.

- [Basel III Multi-year PD Framework](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Transition matrix application to multi-year PD estimation; cumulative default probability formula; migration across rating grades.

- Israel, R. B., Rosenthal, J. S., & Wei, J. Z. (2001). "Finding Generators for Markov Chains via Empirical Transition Matrices, with Applications to Credit Ratings." Mathematical Finance, 11(2), 245-265. Advanced methodology for constructing transition matrices from observed migration data; estimation error bounds; low-default portfolio adjustments.

- Christensen, J. H., Hansen, E., & Lando, D. (2004). "Confidence Sets for Continuous-Time Rating Transitions." Journal of Banking & Finance, 28(11), 2575-2602. Statistical methods for transition matrix estimation; confidence intervals; regime-switching dynamics under economic conditions; stress scenario matrices.

- Kadam, A., & Lenk, P. (2008). "Bayesian Inference for Markov Chains Based on Low-Frequency Data." Journal of Business & Economic Statistics, 26(3), 369-380. Bayesian approach to transition matrix estimation with limited default observations; informative priors from structural models; posterior uncertainty quantification.

---
**Status:** Historical and forward-looking rating dynamics tool | **Complements:** Ratings, PD forecasting, stress testing
