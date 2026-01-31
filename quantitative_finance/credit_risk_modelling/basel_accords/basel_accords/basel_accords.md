# Basel Accords for Credit Risk

## 1. Concept Skeleton
**Definition:** International regulatory frameworks establishing minimum capital requirements for banks to absorb losses from credit risk, operational risk, and market risk  
**Purpose:** Promote financial stability; ensure banks hold adequate capital cushion; standardize risk measurement; prevent systemic crises; level playing field globally  
**Prerequisites:** Credit risk fundamentals, probability of default, loss given default, risk-weighted assets, capital adequacy concepts

## 2. Comparative Framing
| Basel Accord | Basel I (1988) | Basel II (2004) | Basel III (2010) | Basel IV (2017) |
|--------------|---------------|-----------------|------------------|-----------------|
| **Capital Definition** | Tier 1 + Tier 2 | Tier 1 + Tier 2 + Tier 3 | CET1 + AT1 + Tier 2 | Enhanced quality |
| **Risk Coverage** | Credit only | Credit + Operational + Market | + Liquidity + Leverage | Standardized floor |
| **Approach** | Standardized only | + IRB Advanced | + Countercyclical buffers | Output floor 72.5% |
| **Minimum Ratio** | 8% (total) | 8% (total) | 10.5% (CET1 4.5% + buffers) | Stricter |
| **Risk Sensitivity** | Crude (5 buckets) | Granular (ratings) | + Systemic risk | Reduced IRB variability |

| IRB Approach | Foundation IRB (F-IRB) | Advanced IRB (A-IRB) | Standardized Approach | Internal Models |
|--------------|----------------------|---------------------|----------------------|-----------------|
| **Bank Estimates** | PD only | PD, LGD, EAD, M | None (supervisor values) | Full internal model |
| **Supervisor Provides** | LGD, EAD, M | Nothing | Risk weights | Approval required |
| **Risk Sensitivity** | Moderate | High | Low | Highest |
| **Capital Impact** | Lower than standardized | Lowest (typically) | Baseline | Variable |
| **Validation Burden** | Moderate | Very high | Low | Extreme |

## 3. Examples + Counterexamples

**Simple Example:**  
Bank has $100M loan to AAA corporate. Basel I: 100% risk weight → RWA=$100M, capital=$8M. Basel II Standardized: 20% weight → RWA=$20M, capital=$1.6M (80% reduction).

**Perfect Fit:**  
Large international bank with diverse portfolio: Uses A-IRB for corporates (lower capital), Standardized for retail (simpler). Basel III buffers protect against crisis (2008 prevented).

**Foundation IRB:**  
Bank estimates PD=2% for SME borrower. Supervisor sets LGD=45%, EAD=100%, M=2.5 years. Calculate RWA via Basel formula, multiply by 8% for capital.

**Countercyclical Buffer:**  
Economy booming, credit growth excessive. Regulator activates 2.5% buffer → banks must hold 13% capital (vs 10.5% normally). Cools lending, prevents bubble.

**Poor Fit:**  
Small regional bank: A-IRB compliance costs $10M+ annually. Validation, data systems, models. Better to use Standardized (simpler, less optimal capital).

**2008 Crisis Lesson:**  
Banks met 8% Basel II capital but failed (Lehman, Bear Stearns). Why? Low-quality capital (hybrid debt), pro-cyclical RWA (all AAA suddenly defaulted), no liquidity rules. Basel III fixes.

## 4. Layer Breakdown
```
Basel Accords Framework:

├─ Basel I (1988 - "The Beginning"):
│  ├─ Original Objective:
│  │   Stop undercapitalization of international banks
│  │   Level playing field (Japanese vs US banks)
│  ├─ Core Principle:
│  │   Capital Ratio = Capital / Risk-Weighted Assets ≥ 8%
│  ├─ Capital Definition:
│  │   ├─ Tier 1 (Core Capital):
│  │   │   Common stock, retained earnings
│  │   │   Non-redeemable preferred stock
│  │   │   Minimum 4% of RWA
│  │   └─ Tier 2 (Supplementary):
│  │       Subordinated debt (<50% of Tier 1)
│  │       Hybrid instruments, loan loss reserves
│  ├─ Risk Weights (Crude Buckets):
│  │   ├─ 0%: Cash, OECD government bonds
│  │   ├─ 20%: OECD bank debt, municipal bonds
│  │   ├─ 50%: Residential mortgages
│  │   ├─ 100%: Corporate loans, non-OECD sovereign
│  │   └─ Off-Balance Sheet: Credit conversion factors
│  │       Letters of credit: 20-50%, Guarantees: 100%
│  ├─ Calculation Example:
│  │   $100M corporate loan × 100% = $100M RWA
│  │   Required capital: $100M × 8% = $8M
│  ├─ Criticisms:
│  │   ├─ Not risk-sensitive (all corporates 100%)
│  │   ├─ AAA and B-rated same weight → arbitrage
│  │   ├─ Ignores operational risk
│  │   ├─ No distinction for collateral, maturity
│  │   └─ Gaming: Securitize to lower RWA
│  └─ Legacy:
│      Simple, transparent, but too crude
│      Led to "regulatory arbitrage"
├─ Basel II (2004 - "Three Pillars"):
│  ├─ Pillar 1: Minimum Capital Requirements:
│  │   ├─ Credit Risk:
│  │   │   ├─ Standardized Approach:
│  │   │   │   ├─ External ratings-based (Moody's, S&P)
│  │   │   │   ├─ Risk Weights by Rating:
│  │   │   │   │   AAA-AA: 20%, A: 50%, BBB: 100%
│  │   │   │   │   BB: 100%, B and below: 150%
│  │   │   │   │   Unrated: 100% (or 150% if high risk)
│  │   │   │   ├─ Corporates, Banks, Sovereigns
│  │   │   │   └─ Retail: 75% (residential mortgage 35%)
│  │   │   ├─ Foundation IRB (F-IRB):
│  │   │   │   ├─ Bank estimates: PD (Probability of Default)
│  │   │   │   ├─ Supervisor provides: LGD, EAD, M
│  │   │   │   │   LGD (Loss Given Default): 45% senior, 75% sub
│  │   │   │   │   EAD (Exposure at Default): Drawn + %Undrawn
│  │   │   │   │   M (Maturity): Typically 2.5 years
│  │   │   │   ├─ Risk Weight Formula:
│  │   │   │   │   RW = LGD × N[(1-R)^(-0.5) × G(PD) + (R/(1-R))^0.5 × G(0.999)] × (1+(M-2.5)×b)/(1-1.5×b) × 1.06
│  │   │   │   │   N = cumulative normal, G = inverse normal
│  │   │   │   │   R = correlation (depends on PD, asset class)
│  │   │   │   │   b = maturity adjustment factor
│  │   │   │   ├─ Correlation R:
│  │   │   │   │   Corporate: 0.12×(1-e^(-50×PD))/(1-e^(-50)) + 0.24×[1-(1-e^(-50×PD))/(1-e^(-50))]
│  │   │   │   │   Retail: Lower (0.03-0.15 range)
│  │   │   │   │   Reflects systematic vs idiosyncratic risk
│  │   │   │   └─ Example Calculation:
│  │   │   │       PD=2%, LGD=45%, EAD=$100M, M=3
│  │   │   │       → RW ≈ 50% → RWA=$50M → Capital=$4M
│  │   │   └─ Advanced IRB (A-IRB):
│  │   │       ├─ Bank estimates ALL parameters: PD, LGD, EAD, M
│  │   │       ├─ Requires:
│  │   │       │   5+ years PD data, 7+ years LGD/EAD
│  │   │       │   Robust models, validation, stress testing
│  │   │       │   Supervisor approval (intensive review)
│  │   │       ├─ Benefit:
│  │   │       │   Most risk-sensitive → lowest capital (typically)
│  │   │       │   Especially for low-risk portfolios (retail, mortgage)
│  │   │       ├─ Downsides:
│  │   │       │   Expensive to implement ($10-100M+)
│  │   │       │   Pro-cyclical (PDs spike in crisis)
│  │   │       │   Model risk, gaming potential
│  │   │       └─ Used by: Large global banks only
│  │   ├─ Operational Risk:
│  │   │   ├─ Basic Indicator Approach:
│  │   │   │   Capital = 15% × 3-year avg gross income
│  │   │   ├─ Standardized Approach:
│  │   │   │   Different % for business lines (12-18%)
│  │   │   └─ Advanced Measurement Approach (AMA):
│  │   │       Internal models, loss distributions
│  │   │       Requires 5+ years loss data
│  │   └─ Market Risk:
│  │       VaR-based capital (99%, 10-day holding)
│  │       Standardized or internal models
│  ├─ Pillar 2: Supervisory Review:
│  │   ├─ ICAAP (Internal Capital Adequacy Assessment):
│  │   │   Bank's own risk assessment
│  │   │   Beyond Pillar 1 minimums
│  │   ├─ SREP (Supervisory Review and Evaluation):
│  │   │   Regulator evaluates bank's ICAAP
│  │   │   Can impose additional buffers
│  │   ├─ Risks Beyond Pillar 1:
│  │   │   Concentration risk, interest rate risk (banking book)
│  │   │   Liquidity risk, business risk, reputational
│  │   └─ Stress Testing:
│  │       Scenario analysis, reverse stress tests
│  │       Ensure viability under adverse conditions
│  └─ Pillar 3: Market Discipline:
│      ├─ Disclosure Requirements:
│      │   Capital structure, RWA by asset class
│      │   Credit risk exposures, PD/LGD (IRB banks)
│      │   Remuneration policies
│      ├─ Frequency: Quarterly (large banks)
│      └─ Purpose: Enable market to assess risk
│          Stakeholders can price risk accurately
├─ Basel III (2010 - "Crisis Response"):
│  ├─ Enhanced Capital Quality & Quantity:
│  │   ├─ Common Equity Tier 1 (CET1):
│  │   │   ├─ Highest quality: Common stock + retained earnings
│  │   │   ├─ Minimum: 4.5% of RWA (up from ~2% Basel II)
│  │   │   ├─ Deductions:
│  │   │   │   Goodwill, intangibles, deferred tax assets
│  │   │   │   Significant investments in financial entities
│  │   │   │   Expected losses > provisions (IRB)
│  │   │   └─ Purpose: Loss-absorbing, permanent, no obligations
│  │   ├─ Additional Tier 1 (AT1):
│  │   │   Perpetual bonds with loss absorption (CoCos - Contingent Convertibles)
│  │   │   Convert to equity or write-down if CET1 falls below trigger
│  │   │   CET1 + AT1 ≥ 6% (Tier 1 minimum)
│  │   ├─ Tier 2:
│  │   │   Subordinated debt (≥5 year maturity)
│  │   │   Total Capital (CET1 + AT1 + T2) ≥ 8%
│  │   └─ Capital Conservation Buffer (CCB):
│  │       ├─ Additional 2.5% CET1 above minimum
│  │       ├─ Total: 4.5% + 2.5% = 7% CET1
│  │       ├─ Purpose: Build cushion in good times
│  │       └─ Breach consequences:
│  │           Restrictions on dividends, bonuses, buybacks
│  │           Must rebuild buffer
│  ├─ Countercyclical Capital Buffer (CCyB):
│  │   ├─ Variable: 0-2.5% (typically, can go higher)
│  │   ├─ Activated by regulators when credit growth excessive
│  │   ├─ Released in downturn to support lending
│  │   ├─ Country-specific: Varies by jurisdiction
│  │   └─ Example: Sweden 2% (2015), UK 1% (2018)
│  ├─ Systemically Important Banks (G-SIBs, D-SIBs):
│  │   ├─ Additional buffer: 1-3.5% CET1 (based on systemic importance)
│  │   ├─ G-SIBs: Globally systemically important (top 30 banks)
│  │   │   JPMorgan, HSBC, Deutsche, Citi, etc.
│  │   │   Bucket 1-5 based on score (size, interconnectedness, complexity, substitutability, cross-border)
│  │   ├─ D-SIBs: Domestic systemically important
│  │   │   National champions (1-2% buffer typically)
│  │   └─ Total: Can reach 13-16% CET1 (4.5% + 2.5% CCB + 0-2.5% CCyB + 1-3.5% G-SIB)
│  ├─ Leverage Ratio:
│  │   ├─ Non-Risk-Based Backstop:
│  │   │   Leverage Ratio = Tier 1 Capital / Total Exposure ≥ 3%
│  │   │   Exposure = On-balance + Off-balance (no risk weights)
│  │   ├─ Purpose:
│  │   │   Prevent gaming of risk weights
│  │   │   Limit absolute leverage (Lehman had 30× leverage)
│  │   └─ Binding constraint for low-risk portfolios
│  │       (e.g., government bonds with 0% RW)
│  ├─ Liquidity Coverage Ratio (LCR):
│  │   ├─ Formula: LCR = HQLA / Net Cash Outflows (30 days) ≥ 100%
│  │   ├─ HQLA (High-Quality Liquid Assets):
│  │   │   Level 1: Cash, central bank reserves, sovereigns (100% weight)
│  │   │   Level 2A: Agency debt, covered bonds (85%)
│  │   │   Level 2B: Corporate bonds, equities (50-65%)
│  │   │   Max Level 2: 40% of total
│  │   ├─ Net Cash Outflows:
│  │   │   Expected outflows (deposits × run-off rate)
│  │   │   - Expected inflows (75% cap)
│  │   │   Stress scenario (30 days)
│  │   └─ Purpose: Survive acute stress (1 month)
│  ├─ Net Stable Funding Ratio (NSFR):
│  │   ├─ Formula: NSFR = Available Stable Funding / Required Stable Funding ≥ 100%
│  │   ├─ ASF:
│  │   │   Equity, long-term debt (≥1 year): 100%
│  │   │   Stable retail deposits: 90-95%
│  │   │   Less stable deposits: 80-90%
│  │   ├─ RSF:
│  │   │   Illiquid assets (loans): 85-100%
│  │   │   Liquid assets: 0-15%
│  │   └─ Purpose: Structural liquidity (1 year horizon)
│  │       Avoid maturity mismatch (borrow short, lend long)
│  └─ Phase-In (2013-2019):
│      Gradual implementation to avoid credit crunch
│      Full compliance by 2019
├─ Basel IV (Finalized 2017, Implementation 2023-2028):
│  ├─ "Basel 3.1" or "Endgame":
│  │   Finalize post-crisis reforms
│  │   Address excessive variability in RWA
│  ├─ Revised Standardized Approach:
│  │   ├─ More risk-sensitive than Basel II
│  │   ├─ Corporate exposures:
│  │   │   Revenue, leverage-based risk weights
│  │   │   No longer purely ratings-dependent
│  │   ├─ Real estate:
│  │   │   LTV-based risk weights (35-70%)
│  │   │   No flat 35% anymore
│  │   └─ Subordinated positions: Higher weights
│  ├─ IRB Restrictions:
│  │   ├─ Output Floor:
│  │   │   RWA_IRB ≥ 72.5% × RWA_Standardized
│  │   │   Prevents IRB from reducing capital too much
│  │   │   Phase-in: 50% (2023) → 55% (2024) → ... → 72.5% (2028)
│  │   ├─ Removal of A-IRB for certain exposures:
│  │   │   Large corporates (€500M+ revenue): Must use F-IRB or Standardized
│  │   │   Bank exposures, equity exposures
│  │   └─ Rationale: Reduce model variability
│  │       Same obligor, different banks → 300% RWA difference (pre-Basel IV)
│  ├─ Operational Risk:
│  │   ├─ Single Standardized Approach (replaces AMA):
│  │   │   Capital = Business Indicator Component (BIC) × Internal Loss Multiplier (ILM)
│  │   │   BIC = Function of income (interest, fees, trading)
│  │   │   ILM = Based on historical losses (15-year lookback)
│  │   └─ No more internal models (AMA removed)
│  │       Too much variability, gaming
│  ├─ Credit Valuation Adjustment (CVA) Risk:
│  │   ├─ Counterparty credit risk for derivatives
│  │   ├─ Standardized approach mandatory
│  │   └─ Sensitive to rating, maturity, hedging
│  └─ Impact:
│      ├─ Average: 10-20% increase in RWA (varies by bank)
│      ├─ IRB banks hit hardest (output floor)
│      ├─ Operational risk: Mixed (depends on loss history)
│      └─ Implementation: 2023-2028 (long phase-in)
├─ PD, LGD, EAD Modeling (IRB Requirements):
│  ├─ Probability of Default (PD):
│  │   ├─ Definition: Likelihood of default over 1 year
│  │   ├─ Rating Scale:
│  │   │   Minimum 7 borrower grades (performing)
│  │   │   1 defaulted grade
│  │   │   More granular = better risk sensitivity
│  │   ├─ Estimation Methods:
│  │   │   ├─ Historical default rates: # defaults / # obligors
│  │   │   ├─ Cohort analysis: Track rating migration
│  │   │   ├─ Logistic regression: Score → PD mapping
│  │   │   └─ Merton model: Structural (equity → default)
│  │   ├─ Through-the-Cycle (TTC) vs Point-in-Time (PIT):
│  │   │   TTC: Long-run average (less volatile, Basel prefers)
│  │   │   PIT: Current conditions (more volatile)
│  │   ├─ Long-Run Average:
│  │   │   Must reflect economic cycle (boom + bust)
│  │   │   Minimum 5 years data (ideally cover cycle)
│  │   ├─ Floor: 0.03% (corporate, bank, sovereign)
│  │   │   Prevents unrealistically low PD
│  │   └─ Margin of Conservatism (MoC):
│  │       Add buffer for estimation uncertainty
│  ├─ Loss Given Default (LGD):
│  │   ├─ Definition: % of exposure lost upon default
│  │   ├─ Formula: LGD = (Exposure - Recoveries) / Exposure
│  │   ├─ Components:
│  │   │   ├─ Exposure: EAD at default
│  │   │   ├─ Recoveries: Collateral sale + cash flows
│  │   │   ├─ Costs: Legal, workout, time value
│  │   │   └─ Time: Discounted to default date
│  │   ├─ A-IRB Requirements:
│  │   │   ├─ 7+ years data (including downturn)
│  │   │   ├─ Downturn LGD: Stressed collateral values
│  │   │   │   Hair cut: Property values in recession
│  │   │   ├─ Cure rates: Defaults that cure (self-cure, restructure)
│  │   │   └─ Realized LGD: Post-workout, fully resolved
│  │   ├─ F-IRB (Supervisory LGD):
│  │   │   Senior unsecured: 45%
│  │   │   Subordinated: 75%
│  │   │   Secured: Adjusted for collateral
│  │   ├─ Floor: 0% (but downturn LGD typically 10-20% minimum)
│  │   └─ Collateral Adjustments:
│  │       C* = C × (1 - Hc)
│  │       Hc = haircut (vol, type, liquidity)
│  │       LGD = max(0, (E - C*) / E)
│  ├─ Exposure at Default (EAD):
│  │   ├─ On-Balance Sheet: Current exposure
│  │   ├─ Off-Balance Sheet:
│  │   │   EAD = Drawn + CCF × Undrawn
│  │   │   CCF = Credit Conversion Factor
│  │   ├─ CCF by Facility:
│  │   │   Committed lines: 75% (A-IRB can estimate)
│  │   │   Unconditionally cancellable: 0%
│  │   │   Guarantees: 100%
│  │   │   Derivatives: Add-on for potential future exposure
│  │   ├─ A-IRB EAD Modeling:
│  │   │   Usage-given-default: How much drawn at default?
│  │   │   Stressed utilization (crisis → max out lines)
│  │   └─ F-IRB: Supervisory CCFs
│  └─ Maturity (M):
│  │   ├─ Corporate/Bank: 1-5 years (cap)
│  │   │   Effective maturity = CF-weighted average
│  │   ├─ Retail: Fixed 1 year (no maturity adjustment)
│  │   └─ Maturity adjustment in RW formula:
│  │       Longer maturity → higher risk weight
│  │       Reflects rating migration risk
├─ Capital Planning & Stress Testing:
│  ├─ ICAAP (Internal Capital Adequacy Assessment Process):
│  │   ├─ Bank's self-assessment of capital needs
│  │   ├─ Forward-looking (3-year horizon)
│  │   ├─ Covers all material risks:
│  │   │   Credit, market, operational, liquidity
│  │   │   Concentration, interest rate (banking book)
│  │   │   Business risk, strategic risk
│  │   ├─ Economic Capital:
│  │   │   Amount needed to absorb unexpected losses
│  │   │   VaR at 99.9% (insolvency probability 0.1%)
│  │   │   May differ from regulatory capital
│  │   └─ Capital Plan:
│  │       Baseline, stress scenarios
│  │       Management actions (dividends, issuance)
│  ├─ CCAR (US) / EBA Stress Tests (EU):
│  │   ├─ Supervisory stress testing
│  │   ├─ Scenarios:
│  │   │   Baseline: Expected economic path
│  │   │   Adverse: Moderate recession
│  │   │   Severely Adverse: Deep recession (2008-like)
│  │   ├─ Projections:
│  │   │   RWA evolution, PnL, capital ratios
│  │   │   9-quarter horizon
│  │   ├─ Pass/Fail:
│  │   │   Must remain above minimums in stress
│  │   │   CET1 ≥ 4.5% (+ buffers) even in severe stress
│  │   └─ Consequences:
│  │       Pass: Can pay dividends, buybacks
│  │       Fail: Restrictions, remediation plan
│  └─ Resolution Planning ("Living Wills"):
│      How to wind down bank without taxpayer bailout
│      Credible resolution strategy
└─ Practical Implementation:
   ├─ Data Requirements:
   │   ├─ Loan-level data: Obligor, facility, rating, collateral
   │   ├─ Historical: 5-7+ years (full cycle)
   │   ├─ Default definition: 90 days past due or unlikely to pay
   │   ├─ Resolution: Full recovery process tracking
   │   └─ Quality: Clean, consistent, auditable
   ├─ Model Development:
   │   ├─ PD Models: Logistic regression, scorecards
   │   ├─ LGD Models: Beta regression, OLS, hurdle models
   │   ├─ EAD Models: Linear regression, time-series
   │   ├─ Backtesting: Out-of-sample, out-of-time
   │   └─ Calibration: Long-run average, MoC
   ├─ Validation:
   │   ├─ Independent team (not model developers)
   │   ├─ Annual review (minimum)
   │   ├─ Tests:
   │   │   Discriminatory power (AUC, Gini)
   │   │   Calibration (observed vs predicted)
   │   │   Stability (PSI - Population Stability Index)
   │   ├─ Benchmarking: Compare to external data
   │   └─ Documentation: Detailed methodology
   ├─ Governance:
   │   ├─ Board oversight (capital adequacy)
   │   ├─ Model risk committee
   │   ├─ Separation: Development vs validation vs use
   │   └─ Regular reporting to senior management
   ├─ Regulatory Approval:
   │   ├─ IRB: 12-18 month approval process
   │   ├─ On-site inspection, data quality review
   │   ├─ Ongoing compliance (annual certification)
   │   └─ Remediation: Fix deficiencies or revert to Standardized
   └─ Challenges:
      ├─ Pro-cyclicality: PDs spike in crisis → higher capital when least affordable
      ├─ Data scarcity: Especially for low-default portfolios (LDP)
      ├─ Model risk: Wrong model → too little capital
      ├─ Regulatory arbitrage: Shop for best regulator
      └─ Complexity: Basel compliance costs billions for large banks
```

**Interaction:** Measure credit exposure → Estimate PD/LGD/EAD → Calculate RWA (Standardized or IRB) → Apply capital ratio (8-13%+) → Add buffers → Validate models → Stress test → Report to regulators.

## 5. Mini-Project
Implement Basel capital calculations with IRB and Standardized approaches:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("BASEL ACCORDS: CAPITAL REQUIREMENTS CALCULATION")
print("="*70)

class BaselCapitalCalculator:
    """Calculate regulatory capital under Basel frameworks"""
    
    def __init__(self):
        self.minimum_cet1 = 0.045  # 4.5%
        self.ccb = 0.025  # Capital conservation buffer
        self.minimum_total = 0.08  # 8%
        
    def basel_i_rwa(self, exposure, asset_class='corporate'):
        """Basel I risk-weighted assets (crude buckets)"""
        weights = {
            'sovereign_oecd': 0.0,
            'bank_oecd': 0.2,
            'mortgage': 0.5,
            'corporate': 1.0,
            'other': 1.0
        }
        
        rw = weights.get(asset_class, 1.0)
        return exposure * rw
    
    def basel_ii_standardized_rwa(self, exposure, rating='unrated', asset_class='corporate'):
        """Basel II Standardized Approach"""
        # Corporate risk weights by external rating
        corporate_weights = {
            'AAA': 0.20, 'AA': 0.20,
            'A': 0.50,
            'BBB': 1.00,
            'BB': 1.00,
            'B': 1.50,
            'CCC': 1.50,
            'unrated': 1.00
        }
        
        # Asset class specific
        if asset_class == 'corporate':
            rw = corporate_weights.get(rating, 1.0)
        elif asset_class == 'retail':
            rw = 0.75
        elif asset_class == 'mortgage':
            rw = 0.35
        elif asset_class == 'sovereign':
            sovereign_weights = {'AAA': 0.0, 'AA': 0.0, 'A': 0.20, 
                               'BBB': 0.50, 'BB': 1.00, 'B': 1.00, 'unrated': 1.00}
            rw = sovereign_weights.get(rating, 1.0)
        else:
            rw = 1.0
        
        return exposure * rw
    
    def basel_ii_irb_correlation(self, pd, asset_class='corporate'):
        """Asset correlation R in Basel IRB formula"""
        if asset_class == 'corporate':
            # Corporate correlation formula
            R = (0.12 * (1 - np.exp(-50*pd))/(1 - np.exp(-50)) +
                 0.24 * (1 - (1 - np.exp(-50*pd))/(1 - np.exp(-50))))
        elif asset_class == 'retail':
            # Retail (simplified)
            R = 0.15
        else:
            R = 0.12
        
        return R
    
    def basel_ii_irb_rw(self, pd, lgd, ead, maturity=2.5, asset_class='corporate'):
        """
        Basel II IRB risk weight formula
        
        Parameters:
        - pd: Probability of default (annual)
        - lgd: Loss given default (as fraction)
        - ead: Exposure at default
        - maturity: Effective maturity in years
        """
        # Floor for PD
        pd = max(pd, 0.0003)  # 0.03% floor
        
        # Correlation
        R = self.basel_ii_irb_correlation(pd, asset_class)
        
        # Maturity adjustment factor
        b = (0.11852 - 0.05478 * np.log(pd))**2
        
        # Basel IRB formula
        # RW = LGD × N[(1-R)^(-0.5) × G(PD) + (R/(1-R))^0.5 × G(0.999)] × [1 + (M-2.5)×b]/(1-1.5×b) × 1.06
        
        # Calculate components
        G_pd = norm.ppf(pd)  # Inverse normal of PD
        G_999 = norm.ppf(0.999)  # 99.9th percentile
        
        # Capital requirement (before LGD scaling)
        K = norm.cdf(
            ((1-R)**(-0.5)) * G_pd + ((R/(1-R))**0.5) * G_999
        )
        
        # Maturity adjustment
        if asset_class == 'corporate':
            MA = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
        else:
            MA = 1.0  # No maturity adjustment for retail
        
        # Risk weight
        RW = lgd * K * MA * 1.06  # 1.06 is scaling factor
        
        # RWA
        rwa = ead * RW
        
        return rwa, RW
    
    def calculate_capital_ratios(self, rwa, cet1_capital, tier1_capital, total_capital):
        """Calculate capital ratios"""
        cet1_ratio = cet1_capital / rwa if rwa > 0 else 0
        tier1_ratio = tier1_capital / rwa if rwa > 0 else 0
        total_ratio = total_capital / rwa if rwa > 0 else 0
        
        return {
            'cet1_ratio': cet1_ratio,
            'tier1_ratio': tier1_ratio,
            'total_ratio': total_ratio
        }
    
    def check_compliance(self, cet1_ratio, ccyb=0.0, gsib_buffer=0.0):
        """Check Basel III compliance"""
        required_cet1 = self.minimum_cet1 + self.ccb + ccyb + gsib_buffer
        
        compliant = cet1_ratio >= required_cet1
        
        return {
            'required_cet1': required_cet1,
            'actual_cet1': cet1_ratio,
            'excess': cet1_ratio - required_cet1,
            'compliant': compliant
        }

class LiquidityCalculator:
    """Calculate LCR and NSFR"""
    
    def calculate_lcr(self, hqla_level1, hqla_level2a, hqla_level2b,
                      cash_outflows, cash_inflows):
        """
        Liquidity Coverage Ratio
        LCR = HQLA / Net Cash Outflows (30 days) ≥ 100%
        """
        # HQLA with weights
        hqla = hqla_level1 * 1.0 + hqla_level2a * 0.85 + hqla_level2b * 0.5
        
        # Cap level 2 at 40%
        level2 = hqla_level2a * 0.85 + hqla_level2b * 0.5
        if level2 > 0.4 * hqla:
            # Adjust
            adjustment = level2 - 0.4 * hqla
            hqla -= adjustment
        
        # Net cash outflows (cap inflows at 75% of outflows)
        capped_inflows = min(cash_inflows, 0.75 * cash_outflows)
        net_outflows = cash_outflows - capped_inflows
        
        lcr = hqla / net_outflows if net_outflows > 0 else np.inf
        
        return {
            'hqla': hqla,
            'net_outflows': net_outflows,
            'lcr': lcr,
            'compliant': lcr >= 1.0
        }
    
    def calculate_nsfr(self, asf_components, rsf_components):
        """
        Net Stable Funding Ratio
        NSFR = Available Stable Funding / Required Stable Funding ≥ 100%
        """
        # ASF: {amount: weight} dictionary
        asf = sum(amount * weight for amount, weight in asf_components)
        
        # RSF: {amount: weight} dictionary
        rsf = sum(amount * weight for amount, weight in rsf_components)
        
        nsfr = asf / rsf if rsf > 0 else np.inf
        
        return {
            'asf': asf,
            'rsf': rsf,
            'nsfr': nsfr,
            'compliant': nsfr >= 1.0
        }

# Scenario 1: Basel I vs Basel II Comparison
print("\n" + "="*70)
print("SCENARIO 1: Basel I vs Basel II Comparison")
print("="*70)

calc = BaselCapitalCalculator()

# Portfolio of loans
exposures = [
    {'amount': 100e6, 'rating': 'AAA', 'asset_class': 'corporate'},
    {'amount': 50e6, 'rating': 'BBB', 'asset_class': 'corporate'},
    {'amount': 200e6, 'rating': 'unrated', 'asset_class': 'corporate'},
    {'amount': 150e6, 'rating': None, 'asset_class': 'mortgage'},
    {'amount': 75e6, 'rating': None, 'asset_class': 'retail'},
]

print(f"\n{'Exposure':<15} {'Asset Class':<15} {'Rating':<10} {'Basel I RWA':<15} {'Basel II RWA':<15}")
print("-" * 70)

total_rwa_basel_i = 0
total_rwa_basel_ii = 0

for exp in exposures:
    # Basel I
    rwa_i = calc.basel_i_rwa(exp['amount'], exp['asset_class'])
    
    # Basel II Standardized
    rwa_ii = calc.basel_ii_standardized_rwa(
        exp['amount'], 
        exp.get('rating', 'unrated'), 
        exp['asset_class']
    )
    
    total_rwa_basel_i += rwa_i
    total_rwa_basel_ii += rwa_ii
    
    print(f"${exp['amount']/1e6:<14.0f}M {exp['asset_class']:<15} {str(exp.get('rating', 'N/A')):<10} "
          f"${rwa_i/1e6:<14.0f}M ${rwa_ii/1e6:<14.0f}M")

print(f"\n{'Total RWA:':<40} ${total_rwa_basel_i/1e6:.0f}M ${total_rwa_basel_ii/1e6:.0f}M")

capital_basel_i = total_rwa_basel_i * 0.08
capital_basel_ii = total_rwa_basel_ii * 0.08

print(f"{'Required Capital (8%):':<40} ${capital_basel_i/1e6:.1f}M ${capital_basel_ii/1e6:.1f}M")
print(f"Capital Reduction: {(capital_basel_i - capital_basel_ii)/capital_basel_i*100:.1f}%")

# Scenario 2: IRB vs Standardized
print("\n" + "="*70)
print("SCENARIO 2: Foundation IRB vs Standardized Approach")
print("="*70)

# Single exposure with different PDs
exposure_amount = 100e6
lgd = 0.45  # 45% LGD (F-IRB supervisor value)
maturity = 3.0

print(f"\nExposure: ${exposure_amount/1e6:.0f}M, LGD: {lgd*100:.0f}%, Maturity: {maturity} years")
print(f"\n{'PD':<10} {'Rating':<10} {'F-IRB RWA':<15} {'F-IRB RW%':<12} {'Std RWA':<15} {'Std RW%':<12}")
print("-" * 72)

pds = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
ratings = ['AAA', 'AA', 'A', 'BBB', 'BBB', 'BB', 'B']

for pd, rating in zip(pds, ratings):
    # F-IRB
    rwa_irb, rw_irb = calc.basel_ii_irb_rw(pd, lgd, exposure_amount, maturity)
    
    # Standardized
    rwa_std = calc.basel_ii_standardized_rwa(exposure_amount, rating, 'corporate')
    rw_std = rwa_std / exposure_amount
    
    print(f"{pd*100:<10.2f}% {rating:<10} ${rwa_irb/1e6:<14.1f}M {rw_irb*100:<11.1f}% "
          f"${rwa_std/1e6:<14.1f}M {rw_std*100:<11.1f}%")

print(f"\nF-IRB provides capital relief for low-PD exposures")

# Scenario 3: Basel III Capital Requirements
print("\n" + "="*70)
print("SCENARIO 3: Basel III Capital Requirements & Buffers")
print("="*70)

# Bank portfolio
total_rwa = 500e9  # $500B RWA

# Capital structure
cet1_capital = 45e9  # $45B common equity
at1_capital = 5e9    # $5B AT1 (CoCos)
tier2_capital = 10e9  # $10B Tier 2 debt

tier1_capital = cet1_capital + at1_capital
total_capital = tier1_capital + tier2_capital

ratios = calc.calculate_capital_ratios(total_rwa, cet1_capital, tier1_capital, total_capital)

print(f"\nBank Capital Structure:")
print(f"  CET1: ${cet1_capital/1e9:.1f}B")
print(f"  AT1: ${at1_capital/1e9:.1f}B")
print(f"  Tier 2: ${tier2_capital/1e9:.1f}B")
print(f"  Total Capital: ${total_capital/1e9:.1f}B")
print(f"  RWA: ${total_rwa/1e9:.1f}B")

print(f"\nCapital Ratios:")
print(f"  CET1 Ratio: {ratios['cet1_ratio']*100:.2f}%")
print(f"  Tier 1 Ratio: {ratios['tier1_ratio']*100:.2f}%")
print(f"  Total Capital Ratio: {ratios['total_ratio']*100:.2f}%")

# Check compliance scenarios
scenarios = [
    {'name': 'Non-Systemic Bank', 'ccyb': 0.0, 'gsib': 0.0},
    {'name': 'With Countercyclical Buffer', 'ccyb': 0.025, 'gsib': 0.0},
    {'name': 'G-SIB (Bucket 1)', 'ccyb': 0.01, 'gsib': 0.01},
    {'name': 'G-SIB (Bucket 3)', 'ccyb': 0.01, 'gsib': 0.02},
]

print(f"\n{'Scenario':<30} {'Required CET1':<15} {'Actual':<12} {'Excess':<12} {'Status':<10}")
print("-" * 79)

for scenario in scenarios:
    compliance = calc.check_compliance(
        ratios['cet1_ratio'], 
        scenario['ccyb'], 
        scenario['gsib']
    )
    
    status = "✓ Compliant" if compliance['compliant'] else "✗ BREACH"
    
    print(f"{scenario['name']:<30} {compliance['required_cet1']*100:<14.2f}% "
          f"{compliance['actual_cet1']*100:<11.2f}% {compliance['excess']*100:<11.2f}% {status:<10}")

# Scenario 4: Liquidity Coverage Ratio
print("\n" + "="*70)
print("SCENARIO 4: Liquidity Coverage Ratio (LCR)")
print("="*70)

liq_calc = LiquidityCalculator()

# HQLA
hqla_l1 = 50e9   # Cash, central bank reserves, sovereigns
hqla_l2a = 20e9  # Agency debt, covered bonds
hqla_l2b = 15e9  # Corporate bonds, equities

# Cash flows (30-day stress)
outflows = 100e9  # Deposit runoff, wholesale funding
inflows = 40e9    # Loan repayments

lcr_result = liq_calc.calculate_lcr(hqla_l1, hqla_l2a, hqla_l2b, outflows, inflows)

print(f"\nHigh-Quality Liquid Assets:")
print(f"  Level 1 (100% weight): ${hqla_l1/1e9:.1f}B")
print(f"  Level 2A (85% weight): ${hqla_l2a/1e9:.1f}B")
print(f"  Level 2B (50% weight): ${hqla_l2b/1e9:.1f}B")
print(f"  Total HQLA (weighted): ${lcr_result['hqla']/1e9:.1f}B")

print(f"\n30-Day Stressed Cash Flows:")
print(f"  Expected Outflows: ${outflows/1e9:.1f}B")
print(f"  Expected Inflows: ${inflows/1e9:.1f}B (capped at 75% of outflows)")
print(f"  Net Outflows: ${lcr_result['net_outflows']/1e9:.1f}B")

print(f"\nLCR: {lcr_result['lcr']*100:.1f}%")
print(f"Minimum Required: 100%")
print(f"Status: {'✓ Compliant' if lcr_result['compliant'] else '✗ BREACH'}")

# Scenario 5: Net Stable Funding Ratio
print("\n" + "="*70)
print("SCENARIO 5: Net Stable Funding Ratio (NSFR)")
print("="*70)

# Available Stable Funding (amount, weight)
asf_components = [
    (100e9, 1.0),   # Equity
    (50e9, 1.0),    # Long-term debt (>1 year)
    (200e9, 0.90),  # Stable retail deposits
    (100e9, 0.50),  # Less stable deposits
]

# Required Stable Funding (amount, weight)
rsf_components = [
    (300e9, 0.85),  # Corporate loans
    (100e9, 0.65),  # Retail loans (mortgages)
    (50e9, 0.15),   # HQLA Level 1
    (30e9, 0.50),   # Other liquid assets
]

nsfr_result = liq_calc.calculate_nsfr(asf_components, rsf_components)

print(f"\nAvailable Stable Funding: ${nsfr_result['asf']/1e9:.1f}B")
print(f"Required Stable Funding: ${nsfr_result['rsf']/1e9:.1f}B")
print(f"\nNSFR: {nsfr_result['nsfr']*100:.1f}%")
print(f"Minimum Required: 100%")
print(f"Status: {'✓ Compliant' if nsfr_result['compliant'] else '✗ BREACH'}")

# Scenario 6: IRB Parameter Sensitivity
print("\n" + "="*70)
print("SCENARIO 6: IRB Risk Weight Sensitivity Analysis")
print("="*70)

base_pd = 0.02
base_lgd = 0.45
base_ead = 100e6

print(f"\nBase Case: PD={base_pd*100:.0f}%, LGD={base_lgd*100:.0f}%, EAD=${base_ead/1e6:.0f}M")

# PD sensitivity
print(f"\n{'PD %':<10} {'RWA ($M)':<15} {'RW %':<10} {'Change vs Base':<15}")
print("-" * 50)

base_rwa, base_rw = calc.basel_ii_irb_rw(base_pd, base_lgd, base_ead)

for pd_mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
    pd = base_pd * pd_mult
    rwa, rw = calc.basel_ii_irb_rw(pd, base_lgd, base_ead)
    change = (rwa - base_rwa) / base_rwa * 100 if pd_mult != 1.0 else 0
    
    print(f"{pd*100:<10.1f} ${rwa/1e6:<14.1f} {rw*100:<9.1f} {change:+14.1f}%")

# LGD sensitivity
print(f"\n{'LGD %':<10} {'RWA ($M)':<15} {'RW %':<10} {'Change vs Base':<15}")
print("-" * 50)

for lgd in [0.20, 0.35, 0.45, 0.60, 0.75]:
    rwa, rw = calc.basel_ii_irb_rw(base_pd, lgd, base_ead)
    change = (rwa - base_rwa) / base_rwa * 100 if lgd != base_lgd else 0
    
    print(f"{lgd*100:<10.0f} ${rwa/1e6:<14.1f} {rw*100:<9.1f} {change:+14.1f}%")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Basel I vs II capital comparison
ax = axes[0, 0]
categories = ['Basel I', 'Basel II Std']
capitals = [capital_basel_i/1e6, capital_basel_ii/1e6]

bars = ax.bar(categories, capitals, color=['#d62728', '#2ca02c'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Required Capital ($M)')
ax.set_title('Basel I vs Basel II: Capital Requirements')
ax.grid(alpha=0.3, axis='y')

for bar, capital in zip(bars, capitals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${capital:.1f}M', ha='center', va='bottom')

# Plot 2: IRB vs Standardized by PD
ax = axes[0, 1]
pds_plot = np.linspace(0.001, 0.10, 50)
rws_irb = []
rws_std = []

for pd in pds_plot:
    _, rw = calc.basel_ii_irb_rw(pd, lgd, 100)
    rws_irb.append(rw * 100)
    
    # Approximate standardized mapping
    if pd < 0.01:
        rws_std.append(50)
    elif pd < 0.03:
        rws_std.append(100)
    else:
        rws_std.append(150)

ax.plot(pds_plot * 100, rws_irb, 'b-', linewidth=2.5, label='F-IRB')
ax.plot(pds_plot * 100, rws_std, 'r--', linewidth=2, label='Standardized')
ax.set_xlabel('PD (%)')
ax.set_ylabel('Risk Weight (%)')
ax.set_title('IRB vs Standardized: Risk Sensitivity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Basel III capital structure
ax = axes[0, 2]
components = ['CET1', 'AT1', 'Tier 2']
amounts = [cet1_capital/1e9, at1_capital/1e9, tier2_capital/1e9]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars = ax.bar(components, amounts, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=total_rwa * 0.045 / 1e9, color='r', linestyle='--', linewidth=2, label='Min CET1 (4.5%)')
ax.axhline(y=total_rwa * 0.07 / 1e9, color='orange', linestyle=':', linewidth=2, label='Min CET1 + CCB (7%)')
ax.set_ylabel('Capital ($B)')
ax.set_title('Basel III Capital Structure')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar, amount in zip(bars, amounts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${amount:.1f}B', ha='center', va='bottom')

# Plot 4: Capital requirements by bank type
ax = axes[1, 0]
bank_types = ['Regional', 'Large Bank', 'D-SIB', 'G-SIB (Low)', 'G-SIB (High)']
cet1_requirements = [7.0, 7.0, 8.0, 8.5, 10.0]  # Including buffers

bars = ax.barh(bank_types, cet1_requirements, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=4.5, color='r', linestyle='--', linewidth=1.5, label='Minimum (4.5%)')
ax.set_xlabel('CET1 Requirement (%)')
ax.set_title('Basel III: CET1 by Institution Type')
ax.legend()
ax.grid(alpha=0.3, axis='x')

for bar, req in zip(bars, cet1_requirements):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{req:.1f}%', ha='left', va='center', fontweight='bold')

# Plot 5: LCR components
ax = axes[1, 1]
components_lcr = ['Level 1\nHQLA', 'Level 2A\nHQLA', 'Level 2B\nHQLA', 'Net\nOutflows']
values_lcr = [hqla_l1/1e9, hqla_l2a/1e9 * 0.85, hqla_l2b/1e9 * 0.5, -lcr_result['net_outflows']/1e9]
colors_lcr = ['green', 'yellowgreen', 'gold', 'red']

bars = ax.bar(range(len(components_lcr)), values_lcr, color=colors_lcr, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
ax.set_xticks(range(len(components_lcr)))
ax.set_xticklabels(components_lcr)
ax.set_ylabel('Amount ($B, weighted)')
ax.set_title(f'LCR Components (LCR={lcr_result["lcr"]*100:.0f}%)')
ax.grid(alpha=0.3, axis='y')

# Plot 6: IRB RW sensitivity to PD and LGD
ax = axes[1, 2]
pds_grid = np.linspace(0.005, 0.10, 20)
lgds_grid = [0.20, 0.45, 0.75]

for lgd_val in lgds_grid:
    rws = []
    for pd in pds_grid:
        _, rw = calc.basel_ii_irb_rw(pd, lgd_val, 100)
        rws.append(rw * 100)
    
    ax.plot(pds_grid * 100, rws, linewidth=2.5, marker='o', markersize=4, label=f'LGD={lgd_val*100:.0f}%')

ax.set_xlabel('PD (%)')
ax.set_ylabel('Risk Weight (%)')
ax.set_title('IRB Risk Weight Sensitivity')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Basel IV Output Floor:** Implement 72.5% floor (RWA_IRB ≥ 72.5% × RWA_Standardized). For portfolio of 100 loans, calculate impact. Which banks affected most?

2. **Stress Testing:** Simulate recession: PDs double, LGDs increase 20%, RWA increases. Calculate new capital ratios. Does bank remain compliant? Need capital raise?

3. **Pro-Cyclicality:** Model economic cycle (boom-bust). Track PD estimates, RWA, capital requirements through cycle. Quantify procyclical amplification. How does countercyclical buffer help?

4. **Correlation Impact:** Vary correlation R in IRB formula (0.05 to 0.30). How does RW change for PD=2%? Plot sensitivity. What drives correlation differences (corporate vs retail)?

5. **LCR Optimization:** Given mix of assets and liabilities, optimize HQLA allocation to minimize funding cost while maintaining LCR≥100%. Integer programming problem?

## 7. Key References
- [Basel Committee on Banking Supervision, "Basel III: Finalising post-crisis reforms"](https://www.bis.org/bcbs/publ/d424.htm) - official Basel IV framework
- [Gordy, "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules" (2003)](https://www.federalreserve.gov/pubs/feds/2003/200347/200347pap.pdf) - theoretical foundation of IRB formula
- [Basel Committee, "The Basel Framework" (consolidated)](https://www.bis.org/basel_framework/) - complete regulatory text

---
**Status:** International banking regulation | **Complements:** Credit Risk Modeling, PD/LGD/EAD Estimation, Capital Management, Stress Testing, Risk-Weighted Assets
