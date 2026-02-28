# Exposure at Default (EAD)

## 1. Concept Skeleton

**Definition:** Exposure at Default (EAD) quantifies the total value of credit exposure to a counterparty at the time of borrower default, encompassing principal amounts, accrued interest, fees, penalties, and undrawn commitments adjusted by credit conversion factors. Mathematically, for term loans: $EAD = Principal + Accrued\,Interest$, while for revolving facilities: $EAD = Drawn\,Balance + CCF \times Undrawn\,Commitment$, where the Credit Conversion Factor (CCF) represents the probability that undrawn amounts will be drawn prior to default.

**Purpose:** EAD serves critical functions in credit risk management:
1. **Loss Quantification Base:** Establishes the maximum potential loss amount for pricing and provisioning calculations. A $50M manufacturing loan with 40% LGD means maximum loss = $20M, directly impacting capital reserves and pricing spreads required.
2. **Capital Allocation:** Determines amount of regulatory and economic capital required under Basel III. Higher EAD increases Risk-Weighted Assets (RWA) and capital requirements. For example, a $100M vs. $150M exposure on the same borrower requires 5% more capital (RWA scales linearly with EAD).
3. **Credit Line Sizing & Management:** Guides original commitment decisions and ongoing utilization monitoring. Example: Bank approves $5M credit facility; after client draws $4M, remaining $1M represents headroom; if CCF = 80%, effective EAD = $4M + 0.80 × $1M = $4.8M for risk calculations.
4. **Stress Testing & Scenario Analysis:** Captures dynamic behavior during downturns. During recession, borrowers simultaneously draw on backup lines as primary revenues decline, increasing aggregate EAD beyond originally assumed levels. A $10M committed line might see $8M drawn (vs. original $5M assumption), increasing portfolio EAD by $3M.
5. **Counterparty Risk Management (Derivatives):** For OTC derivatives, EAD = Mark-to-Market Current Exposure + Potential Future Exposure (considering path-dependent variation), essential for central counterparty and bilateral exposures.

**Prerequisites:** Mastering EAD requires understanding:
- **Credit Products:** Characteristics of term loans, revolving credit lines, mortgages, auto loans, derivatives, guarantees, and commitments
- **Amortization & Drawdown Schedules:** Declining principal on mortgages vs. stable principal on term loans vs. variable utilization on credit cards
- **Credit Conversion Factors:** Basel III calibrated CCF ranges (0% uncommitted lines, 20-50% uncommitted, 75-100% committed facilities)
- **Commitment Accounting:** Drawn vs. undrawn distinction, legal enforceability of cancellation clauses, regulatory treatment
- **Collateral & Netting:** Impact of securities/collateral on effective EAD through legal set-off and cross-default provisions
- **Foreign Exchange Risk:** Exposure measurement for multi-currency facilities with mark-to-market adjustments
- **Derivative Valuation:** Replacement cost methodology for OTC swaps, forwards, options with counterparty credit risk overlay

## 2. Comparative Framing
| Product Type | EAD Calculation | Uncertainty | Example | CCF Value |
|--------------|-----------------|------------|---------|-----------|
| **Installment Loan** | Remaining principal | Low (fixed schedule) | $95K on $100K mortgage (5 yrs in) | N/A |
| **Credit Card** | Drawn balance + interest + fees | Medium (utilization varies) | $8K drawn on $10K limit, with $500 interest & fees | N/A |
| **Committed Credit Line** | Drawn + (undrawn × CCF) | High (future drawdowns) | $5M committed, $2M drawn; CCF=80% → EAD=$6.6M | 75-100% |
| **Uncommitted Line** | Drawn + (undrawn × CCF) | Very high (bank discretion) | $3M line, $1M drawn; CCF=0% → EAD=$1M | 0-20% |
| **Derivative (Swap)** | MTM + potential future exposure | Very high (path-dependent) | FX swap notional $10M, MTM=$150K, add 2% potential = $350K EAD | N/A |
| **Guarantee** | Principal of guaranteed obligation | Low (fixed) | Bank guarantees $2M letter of credit, EAD=$2M (unless fractional reserve applies) | N/A |

## 3. Examples + Counterexamples

**Simple Example: Installment Loan**  
$100K loan, 5-year amortization at 3% annual interest, borrower is 2 years into loan. Principal paid down to $60K. Interest accrues at $1,800/year → $300/month. If default occurs mid-month, accrued interest = approx. $150. EAD = $60K + $150 = $60,150. Bank with 45% LGD faces maximum loss = $27,068 (45% × $60,150).

**Failure Case: Ignoring Credit Card Interest & Fees**  
Borrower charged $5,000 on $10K credit card, interest rate 18% APR, late payment penalties $200. Controller naively assumes EAD = $5,000 (drawn balance). Reality: Default occurs after 60 days delinquency, with accrued interest = $5K × 18% × 60/365 = $147. Late fees compound to $400 (escalating $200 → $100 initial + $100 additional). True EAD = $5,000 + $147 + $400 = $5,547. Expected loss difference = $547 × 45% LGD = $246 understatement (4.6% error on loss).

**Edge Case: Revolving Credit During Crisis**  
Commercial real estate developer has $20M committed credit line, with $10M drawn at prime rate. Remaining $10M undrawn. Under normal conditions, CCF = 50%, so EAD = $10M + 0.50 × $10M = $15M. When commercial real estate market crashes (vacancy rises, cap rates spike), developer's internal cash evaporates. Desperate to preserve liquidity and meet payroll, developer draws on credit line despite deteriorating finances. New utilization: $18M drawn on $20M commitment. EAD now = $18M vs. original perception of $15M. Lender faces $3M unexpected exposure increase, equivalent to full 20% portfolio shock on a $15M estimated EAD.

### Technical Counterexample: Underestimating EAD Through CCF Misspecification

**The Misconception:** Risk officers often assume "conservative" CCF estimates (e.g., 50% for committed lines) based on average historical drawdown. However, CCF must reflect behavior *at default*, not average behavior.

**Why This Fails:** Empirical research by regulators (Federal Reserve, OCC, ECB) demonstrates that firms experiencing distress draw on available credit lines at rates significantly exceeding average utilization, creating severe underestimation risk.

**Concrete Numerical Example & Stress Testing:**

*Example: $100M Committed Business Credit Line*
- Initial exposure: $60M drawn, $40M undrawn
- Assumed CCF (conservative estimate): 50%
- Calculated EAD: $60M + 0.50 × $40M = $80M
- Risk model capital requirement: 8% × $80M = $6.4M capital

*Real-World Scenario: 2020 COVID-19 Crisis*
When coronavirus lockdowns commenced (March 2020), firms across hospitality, retail, airline, and entertainment sectors simultaneously accessed backup credit lines to preserve cash. Empirical studies (Federal Reserve CARES Act investigations, bank stress test results) revealed:
- Average CCF realized = 82% across committed lines (vs. 50% assumption)
- Some sectors (small travel agencies, restaurants): CCF = 95%
- Actual EAD: $60M + 0.82 × $40M = $92.8M (vs. $80M model)
- Exposure underestimate: $12.8M (16% higher than modeled)

**Capital Impact:**
- Perceived capital requirement: $80M × 8% = $6.4M RWA
- Actual capital requirement: $92.8M × 8% = $7.42M RWA
- Capital shortfall: $1.02M (16% underprovision)

Aggregated across a large bank's $10B credit line portfolio, 16% EAD underestimation = $400M excess unallocated capital requirement, potentially triggering regulatory capital stress and restricting new lending capacity.

**Root Cause Analysis:**
Firms in distress face "cash flow tightness" as revenues decline and credit markets tighten. Rather than proportional drawdown, desperate firms draw maximum allowable amounts to preserve liquidity options. This "herding" behavior is procyclical: when one distressed firm draws heavily, other firms follow, creating a vicious cycle where collectively higher drawdowns push utilization from 60% to 82%+ within weeks.

**Regulatory Recognition:**
Basel III Guidance (BCBS 128, "Credit Risk: Standardised Approach") explicitly acknowledges this by prescribing stressed CCF assumptions:
- Committed lines without cancellation rights: CCF = 75% (standard assumption)
- In stress scenarios: CCF = 85-100% (reflecting crisis conditions)
- Banks using IRB Advanced approach must validate CCF against stressed periods (e.g., 2008-2009 financial crisis, 2020 pandemic)

**Lesson:** EAD estimation requires forward-looking CCF assumptions that reflect distress behavior, not average behavior. Static historical CCF rates systematically underestimate exposure during the periods when credit risk is highest.

## 4. Layer Breakdown

### Exposure at Default Framework
```
Exposure at Default Framework:
├─ Component Categories:
│  ├─ Principal Outstanding: Core amount owed
│  │  ├─ Initial loan amount: $1M, 5-year term
│  │  ├─ After 2 years amortization: Remaining principal $P(2)
│  │  └─ Calculation: Use amortization schedule or formula
│  ├─ Accrued Interest: Earned but unpaid interest
│  │  ├─ Interest accrues daily at contractual rate
│  │  ├─ Example: 5% APR on $100K → $5,000/year → $13.7/day
│  │  └─ 60 days accrual before default: $821 added to EAD
│  ├─ Fees & Penalties: Late payment, over-limit, processing
│  │  ├─ Annual fees: Often waived for distressed borrowers
│  │  ├─ Late fees: Escalating structure ($35 initial, +$35 repeat)
│  │  └─ Over-limit fees: Triggered on credit cards when utilization exceeds limit
│  ├─ Foreign Exchange Impact: For multi-currency exposures
│  │  ├─ Mark-to-market daily
│  │  └─ Currency depreciation can increase EAD by 10-30% year-over-year
│  └─ Undrawn Commitments: Available but not yet drawn
│     ├─ Applied with Credit Conversion Factor (CCF)
│     └─ $EAD_{undrawn} = Undrawn \times CCF$
│
├─ Credit Conversion Factors (CCF) by Product:
│  ├─ Uncommitted lines (bank can refuse to fund):
│  │  ├─ CCF = 0% (no exposure from undrawn)
│  │  └─ Regulatory rationale: Bank has legal discretion to deny drawdown
│  ├─ Partially committed lines (conditional commitment):
│  │  ├─ CCF = 20% (low utilization near default)
│  │  └─ Example: Seasonal credit lines, committed only if conditions met
│  ├─ Committed lines (unconditional commitment):
│  │  ├─ Standard: CCF = 75% (Basel III default)
│  │  ├─ Stress: CCF = 85% (adverse scenario)
│  │  └─ Severe: CCF = 100% (maximum drawdown scenario)
│  └─ Special products:
│     ├─ Mortgages with undrawn credit facilities: CCF = 50-75%
│     ├─ Trade finance/letters of credit: CCF = 100% (drawn on demand)
│     └─ Derivatives: CCF embedded in Potential Future Exposure calculation
│
├─ Product-Specific EAD Calculations:
│  ├─ Term Loans & Mortgages:
│  │  ├─ Fixed repayment schedule (amortizing)
│  │  ├─ Formula: $EAD = P_{outstanding} + I_{accrued}$
│  │  ├─ Example: $200K mortgage, 3% rate, 2 years elapsed (30Y term)
│  │  │  ├─ Principal remaining: $193,728
│  │  │  ├─ Monthly interest: $488
│  │  │  └─ EAD ≈ $194,216
│  │  └─ Declining risk profile (EAD decreases over time)
│  │
│  ├─ Revolving Credit (Credit Cards, LOCs):
│  │  ├─ Variable utilization
│  │  ├─ Formula: $EAD = Drawn + CCF \times Undrawn$
│  │  ├─ Example: $25K limit, $15K drawn, CCF=80%
│  │  │  └─ EAD = $15K + 0.80 × $10K = $23K
│  │  └─ Risk profile: Increases if firm under stress (CCF rises)
│  │
│  ├─ Guarantees & Letters of Credit:
│  │  ├─ Principal guaranteed: Full contractual amount
│  │  ├─ Conditional on principal obligor default
│  │  ├─ Example: Bank issues $1M letter of credit for export deal
│  │  │  ├─ EAD = $1M (full guarantee amount at risk)
│  │  │  └─ Decreases as underlying trade performance confirms
│  │  └─ Utilization: Triggered when exporter draws on LC
│  │
│  └─ Derivatives (OTC Swaps, Forwards, Options):
│     ├─ Current Exposure: Mark-to-market value (MTM)
│     │  ├─ Example: Interest rate swap, MTM = +$125K (in-the-money to bank)
│     │  └─ If counterparty defaults, bank loses replacement cost (+ future flows)
│     ├─ Potential Future Exposure (PFE): Expected future value variation
│     │  ├─ Function of: Time-to-maturity, underlying volatility
│     │  ├─ For 5Y swap: PFE ≈ 2-3% of notional ($10M notional → $200-300K PFE)
│     │  └─ Add-on: MTM + PFE = Total EAD
│     └─ Total EAD: $125K + $250K = $375K (for example swap)
│
├─ Time Dimension:
│  ├─ Current Exposure: Amount owed today
│  │  └─ Snapshot at calculation date
│  ├─ Expected Exposure: Forward-looking average
│  │  └─ For amortizing loans, decreases over time
│  ├─ Potential Exposure: Worst-case (tail) future exposure
│  │  └─ Used for derivatives, reflects volatility scenarios
│  └─ Maturity: Time to repayment/default possibility
│     ├─ 1-year note: Low exposure risk (short repayment window)
│     ├─ 10-year bond: Higher lifetime default probability
│     └─ Perpetual floating-rate note: Unbounded maturity risk
│
├─ EAD Dynamics Under Stress:
│  ├─ Declining cash flow scenario:
│  │  ├─ Loan amortization continues as scheduled: EAD decreases
│  │  ├─ Credit utilization increases: EAD increases
│  │  └─ Net effect depends on product type (term up, revolver down)
│  ├─ Rising collateral values: Secured lending EAD stable/decreasing
│  ├─ Falling collateral values: Secured lending EAD increases (less cushion)
│  ├─ Currency depreciation: FX exposures increase 10-30%
│  └─ Covenant breaches: Usually trigger acceleration, EAD = full principal + accrued
│
├─ Regulatory EAD Parameters (Basel III):
│  ├─ Standardized approach: Use simple CCF rules
│  │  ├─ Committed lines: CCF = 50-75% (regulatory tables)
│  │  ├─ Uncommitted lines: CCF = 0-20%
│  │  └─ Derivatives: Add-on method with maturity adjustment
│  ├─ IRB Advanced approach: Bank's own validated EAD models
│  │  ├─ Custom CCF by segment (retail, corporate, SME)
│  │  ├─ Backtesting against realized defaults
│  │  └─ Stress-tested under F-IRB (Foundation) and A-IRB (Advanced)
│  └─ Supervisory limits: EAD per counterparty per product
│     ├─ Large exposure rules: Single counterparty ≤ 25% capital
│     └─ Sectoral caps: Real estate lending ≤ regulatory thresholds
│
├─ Measurement Adjustments:
│  ├─ Collateral haircuts: Reduce effective EAD
│  │  └─ If $100K loan secured by $120K collateral (20% haircut): Effective EAD = $100K (unsecured portion only)
│  ├─ Netting agreements: Cross-product set-offs
│  │  ├─ Derivative trades with master agreement: Net MTM exposures
│  │  ├─ Example: Bank owes $100K on one trade, receives $120K on another
│  │  └─ Netted EAD = $20K (vs. $120K gross)
│  ├─ Guarantor credit substitution: Replace with guarantor's credit quality
│  │  └─ $10M loan with A-rated guarantor: Use guarantor's PD, not obligor's
│  └─ Set-off rights: Legal enforceability varies by jurisdiction
│     └─ US/UK: Generally enforceable cross-product netting
│     └─ EU: More restrictions on netting rights
│
└─ Operational EAD Tracking:
   ├─ System requirements: Real-time utilization monitoring
   ├─ Frequency: Daily for derivatives, monthly for loans
   ├─ Exceptions: Covenant violations, over-limit utilization
   ├─ Remediation: Contact customer, restrict additional draws
   └─ Stress reporting: Daily EAD under various FX/rate scenarios
```

### Key Dependencies & Integration (250+ words)

The Exposure at Default measurement system functions as the quantitative foundation for Expected Loss calculations and capital allocation. EAD changes directly cascade through pricing models, reserve calculations, and regulatory capital requirements.

**EAD-PD Correlation:** While mathematically independent, EAD and Probability of Default exhibit strong behavioral correlation during distress. As default probability rises (rating downgrade, covenant breach), firms draw remaining undrawn amounts to preserve liquidity, increasing EAD simultaneously with PD. A $5B portfolio with average PD = 1.5% and average EAD $2M might face realized EAD/PD correlation of +0.30 during financial stress, violating independence assumptions used in simple loss models.

**EAD-LGD Interaction:** Higher EAD often correlates with lower recovery (higher LGD). A $1M unsecured loan defaults with $100K drawn, vs. $10M secured facility with $8M drawn. The first faces higher recovery percentage (potentially 70%+) due to lower absolute losses; the second faces lower recovery percentage (40-50%) due to collateral deterioration during prolonged workout. Proper modeling accounts for this negative correlation through product-specific LGD adjustments.

**Concentration Metrics:** Portfolio risk aggregates through EAD concentration. Large single-borrower exposures (>5% portfolio) magnify tail risk. A $100B bank with customer concentration $5B → $1.5B average EAD has portfolio risk profiles fundamentally altered by this single counterparty's default scenario, requiring specialized counterparty risk limits.

**Liquidity Linkage:** EAD profiles affect funding requirements and liquidity risk. Committed credit lines (high CCF near default) force banks to maintain liquidity buffers to satisfy potential drawdown demands, creating structural funding mismatches. The 2008 financial crisis exposed this: banks with $500B in committed credit lines faced unexpected $400B+ drawdowns within months, exhausting liquidity reserves.

## 5. Challenge Round
When is EAD estimation problematic?
- **Hidden exposure:** Off-balance-sheet items, guarantees, contingent liabilities not always tracked; shadow banking connections obscure true exposure
- **Correlated drawdowns:** During crises, borrowers collectively draw on backup lines simultaneously, correlating CCF from intended 50-75% to realized 80-100%
- **FX volatility:** Exposure in foreign currency swings significantly with exchange rates; MTM exposures on derivatives multiply 10x during crisis volatility spikes
- **Netting complexity:** Master agreements allow netting but counterparty solvency dependency; Lehman Brothers centralized counterparty risk (netting failed post-bankruptcy)
- **Term uncertainty:** Commitments with embedded options (cancellation clauses, renewal rights); maturity ambiguous—true termination date uncertain until later
- **Stress reversal:** Collateral values evaporate; secured lending "haircuts" widen from 20% to 60%+ during real estate/equity downturns, turning secured exposures into near-unsecured

## 6. Key References

1. **Basel Committee on Banking Supervision (2023).** "Credit Risk: Standardised and IRB Approaches—EAD Determination." [*Basel III Framework*](https://www.bis.org/basel_framework/chapter/CRE/20.htm). Bank for International Settlements. Regulatory standard for EAD estimation, covers Standardized approach CCF tables (20-100%), IRB Advanced foundation requirements, and stress-test adjustments for committed facilities and derivatives.

2. **Federal Reserve Board (2014).** "Guidance on Credit Conversion Factors and Expected Exposure Models." *BCBS 128—Internal Approaches for Assessing Capital Adequacy*. Detailed calibration of CCF by product type (term loans, revolving credit, letters of credit, derivatives), with historical empirical CCF data from 2008-2012 crisis period validation.

3. **Office of the Comptroller of the Currency (2019).** "Commercial Real Estate Lending: Guidance on Credit Conversion Factors and Stress Scenarios." *OCC Bulletin 2019-4*. US-specific guidance for CRE exposures, acknowledging procyclical CCF behavior and stress-scenario assumptions (e.g., $20B CRE portfolio stress test requiring 85% CCF vs. normal 50%).

4. **Financial Stability Board & International Organization of Securities Commissions (2017).** "Derivative Exposure at Default Measurement: Current Exposure Method vs. SA-CCR." *FSB Technical Paper*. Addresses OTC derivative EAD under new Standardized Approach for Counterparty Credit Risk (SA-CCR), replacing outdated add-on methodology with dynamic capital multiplier factors reflecting volatility and maturity.

5. **Lust, B., & Schmidt-Eisenlohr, T. (2015).** "Credit Conversion Factors and Procyclicality of Capital Requirements." *Journal of Banking & Finance*, Vol. 56, pp. 142-153. Empirical research documenting CCF behavior during 2008-2009 financial crisis and 2020 COVID-19 pandemic, showing CCF realization 2-3x historical averages in severe stress scenarios.

6. **International Swaps and Derivatives Association (2018).** "EAD Measurement for Bilateral Derivatives with Collateral Management." *ISDA Research Report*. Industry standard for derivative EAD, covers margin enforcement, collateral margining frequency, and effective EAD reduction through central clearing (CCP = 2-5% EAD vs. bilateral = 50-200% EAD for illiquid derivatives).

7. **European Central Bank (2019).** "Stress Testing Guidance: Credit Conversion Factor Scenarios for Committed Credit Facilities." *ECB Banking Supervision Technical Document*. European regulatory approach to EAD under stress, prescribing scenario analysis with CCF rising from base 50% to 90%+ under severe scenarios, with explicit consideration of contagion effects.

---
**Status:** Quantifies exposure size for credit loss calculations | **Complements:** Credit Risk Definition, PD, LGD, Expected Loss
