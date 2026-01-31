# Credit Risk Mitigation

## 1. Concept Skeleton
**Definition:** Techniques to reduce credit exposure; collateral, guarantees, netting, securitization, insurance; regulatory capital benefits; risk transfer mechanisms; contingent claims on default; operational and legal frameworks  
**Purpose:** Lower expected loss (EL = PD × LGD × EAD); reduce portfolio concentration; transfer tail risks; optimize regulatory capital allocation; manage systemic risk; cost-effective risk reduction vs. reduce lending  
**Prerequisites:** Credit risk fundamentals, probability of default, loss given default, exposure at default, credit derivatives, financial instruments, legal contracts, regulatory frameworks (Basel III)

## 2. Comparative Framing
| Mitigation Technique | Mechanism | Capital Benefit | Cost | Effectiveness | When to Use |
|-------------------|-----------|-----------------|------|----------------|------------|
| **Collateral** | Secure claim on assets; reduce LGD | Up to 60% RW reduction | Custody, valuation, liquidation risk | High for secured lending | Mortgages, auto loans, trade finance |
| **Guarantees** | Third-party absorbs loss if borrower defaults | PD × LGD reduction (depends on guarantor) | Guarantee fee (0.5-2% p.a.) | Depends on guarantor credit quality | Weak borrowers, structured deals |
| **Netting** | Offset positions with same counterparty | Lowers EAD for portfolios | Legal/operational setup | High (reduces EAD by 20-40%) | Derivatives, OTC trading |
| **Securitization** | Convert loans to tradeable securities; sell risk | Removes from balance sheet (capital relief) | Origination, underwriting, servicing (1-3%) | High if rated/distributed | Loan portfolios, mortgage banks |
| **Credit Insurance** | Protection seller pays if credit event | Reduces effective PD × LGD | Insurance premium (0.5-5% p.a.) | High but basis risk exists | Tail risks, concentrated exposures |
| **Diversification** | Reduce concentration; spread risk | Portfolio-level RW reduction (5-15%) | Operational complexity | Moderate (marginal benefit beyond ~50 assets) | Large portfolios, retail |
| **Term/Maturity Reduction** | Shorten exposure duration | Lower EAD (reduce time risk grows) | Opportunity cost if rates drop | Low-moderate | Non-performing, risky credits |

| Credit Event | Definition | Triggers | Recovery Mechanism |
|--------------|-----------|----------|------------------|
| **Default** | Payment missed beyond grace period | Missed coupon, principal, covenant breach | Bankruptcy, restructuring, collateral sale |
| **Distress** | Significant deterioration not yet default | Ratings downgrade, CDS spike, covenant issue | Forbearance, extension, loan modification |
| **Restructuring** | Terms modified to avoid default | Maturity extension, coupon reduction | Creditor agreement, stay proceedings |
| **Forbearance** | Temporary relief (payment holiday) | Economic hardship, temporary stress | Resume payments, structured recovery |
| **Acceleration** | Entire loan due immediately | Material default, change of control | Legal proceedings, collateral seizure |

## 3. Examples + Counterexamples

**Simple Example:**  
Bank lends $100M to real estate developer. Unsecured: PD=3%, LGD=60%, EAD=$100M → EL=$1.8M. Require 50% collateral (real estate): LGD drops to 30% → EL=$0.9M. Capital requirement: ~40% lower. Cost: Collateral monitoring 0.1% p.a. = $100k/year. Benefit/cost: 2x positive (net $800k capital relief).

**Perfect Fit:**  
Mortgage bank securitizes $1B portfolio. Rates mortgage-backed securities (MBS). Senior tranche rated AAA, junior tranche rated BBB. Sell senior (80% of deal) to insurance company. Junior retained (10%) as credit support. Bank keeps $100M servicing strip. Capital relief: $800M × RW=20% = $160M capital equivalent. Cost: Underwriting, rating agency fees 0.5% = $5M. NPV: Positive if bank would otherwise hold portfolio.

**Over-Collateralization:**  
Lend to startup on 150% collateral (excess of loan value). If asset value drops 40% → collateral now 90% of loan → loses mitigation benefit. Mark-to-market, rebalance periodically. Haircuts critical: Apply 30% to volatile assets.

**Guarantee Failure:**  
Bank A guarantees Bank B's exposure to Company X. Company X defaults. Bank B claims. Bank A refuses (alleges procedural error). Legal dispute → 18 months of delay. Guarantee only effective if guarantor solvent and operationally responsive. Guarantor credit quality paramount.

**Poor Netting Setup:**  
Bank X signs master agreement with hedge fund covering 100 derivatives. Hedge fund defaults; courts rule netting unenforceable (jurisdiction issue). Must liquidate positions individually → higher transaction costs, worse execution. Netting benefit lost. Legal setup critical.

**Concentration in Mitigation:**  
Portfolio: 500 small loans, each guaranteed by same parent company (holding company structure). Portfolio risk now concentrated on parent → higher systemic risk. Diversification benefit lost; mitigating one risk (borrower PD) creates new risk (guarantor PD).

## 4. Layer Breakdown
```
Credit Risk Mitigation Framework:

├─ Collateral-Based Mitigation:
│  ├─ Types of Collateral:
│  │   ├─ Real Property:
│  │   │   Land, buildings (mortgages)
│  │   │   Value stable, liquid (months to sell)
│  │   │   Haircut: 20-30% (for mortgages vs current market)
│  │   │   Time to liquidate: 3-12 months
│  │   ├─ Financial Collateral:
│  │   │   Stocks, bonds, cash
│  │   │   Highly liquid, price volatility
│  │   │   Haircut: 5-50% depending on asset (equity higher)
│  │   │   Time to liquidate: Days to hours
│  │   ├─ Trade Receivables:
│  │   │   Invoices, accounts receivable
│  │   │   Concentration risk (customer), default risk (debtor)
│  │   │   Haircut: 20-40%
│  │   │   Time to liquidate: 30-90 days
│  │   ├─ Inventory/Commodities:
│  │   │   Goods held, raw materials
│  │   │   Price volatile, storage costs, obsolescence
│  │   │   Haircut: 40-70%
│  │   │   Time to liquidate: 1-3 months
│  │   └─ Equipment/Vehicles:
│  │       Used asset values decline quickly
│  │       Haircut: 40-60%
│  │       Time to liquidate: 1-3 months
│  ├─ Collateral Valuation:
│  │   ├─ Initial Valuation:
│  │   │   Fair value at loan origination
│  │   │   Third-party appraisal typical
│  │   │   Cost: 0.1-1% of loan (depending on asset)
│  │   ├─ Mark-to-Market:
│  │   │   Daily or periodic revaluation
│  │   │   For securities: Market prices available
│  │   │   For property: Annual or event-driven
│  │   ├─ Haircuts:
│  │   │   │ Conservative discount to market value
│  │   │   │ Accounts for liquidation costs, market stress
│  │   │   │ Regulatory haircuts: Basel III specifies (5-75% by asset type)
│  │   │   │ Internal models may use lower (with approval)
│  │   │   └─ Example:
│  │   │       Stock market value $100M
│  │   │       Haircut 30%
│  │   │       Collateral value = $70M (for loan purposes)
│  │   ├─ Substitution:
│  │   │   Borrower may substitute collateral
│  │   │   Typical restrictions: Same or higher quality, liquid
│  │   │   Bank approval required
│  │   │   Reduces operational friction
│  │   └─ Concentration Risk:
│  │       Collateral concentrated in single asset/issuer
│  │       Reduces benefit (correlated default)
│  │       Limits: No single collateral > 10-25% of value
│  ├─ Loan-to-Value (LTV) Ratio:
│  │   ├─ Definition:
│  │   │   LTV = Loan Amount / Collateral Value
│  │   │   LTV > 100%: Under-collateralized
│  │   │   LTV = 100%: Fully collateralized
│  │   │   LTV < 100%: Over-collateralized
│  │   ├─ Risk Dynamics:
│  │   │   As collateral value falls → LTV rises → margin call
│  │   │   Borrower must add collateral or reduce loan
│  │   │   Procyclical: Rising rates/volatility → collateral falls → forced sales
│  │   ├─ Regulatory Limits:
│  │   │   Residential mortgages: LTV ≤ 80% typical
│  │   │   Commercial real estate: LTV ≤ 60-70%
│  │   │   Securities lending: LTV ≤ 50-100% (depends on collateral)
│  │   └─ Stress Testing:
│  │       Assume collateral value falls 20-50%
│  │       Calculate new LTV; assess need for intervention
│  │       Example: Real estate down 30% → LTV from 70% to 100%
│  └─ Legal and Operational:
│      ├─ Security Interest:
│      │   Perfected interest: Bank has priority claim
│      │   Filing requirements vary by jurisdiction/asset type
│      │   Continuous monitoring for lapses
│      ├─ Custody:
│      │   Physical possession (jewels, art) or account control
│      │   Segregated from bank's assets (reduce moral hazard)
│      │   Third-party custodian possible (expense)
│      └─ Enforcement:
│          Rights upon default: Seize, liquidate, absorb losses
│          Legal process delays; time-to-recover can be years
│          Proceeds applied: Secured creditors first
├─ Guarantees and Credit Enhancements:
│  ├─ Guarantee Structure:
│  │   ├─ Full Guarantee:
│  │   │   Guarantor liable for 100% of debt (principal + interest)
│  │   │   Bank can pursue guarantor if borrower defaults
│  │   │   Example: Parent company guarantees subsidiary loan
│  │   ├─ Partial Guarantee:
│  │   │   Guarantor liable for portion (e.g., 50%)
│  │   │   Splits risk between guarantor and bank
│  │   │   Example: Government guarantee on small business loan (80%)
│  │   ├─ Stand-by Guarantee:
│  │   │   Drawn only if primary source fails
│  │   │   Example: LC (letter of credit) backed by guarantee
│  │   │   Reduces frequency of draw (lower cost)
│  │   └─ Performance Guarantee:
│  │       Guarantees performance (not financial payment)
│  │       Example: Contractor guarantees project completion
│  ├─ Types of Guarantors:
│  │   ├─ Corporate Guarantor:
│  │   │   Parent company, affiliate
│  │   │   Usually highly rated (investment grade)
│  │   │   Strength: Large balance sheet, multiple income sources
│  │   ├─ Government Guarantor:
│  │   │   National, regional, or local government
│  │   │   Strength: Tax power, money supply, perceived low default
│  │   │   Risk: Regulatory change, political risk
│  │   ├─ Financial Institution:
│  │   │   Bank, insurance company
│  │   │   Usually rated A or higher
│  │   │   Strength: Capitalized, regulated, liquid
│  │   ├─ Multilateral Institution:
│  │   │   World Bank, regional development bank
│  │   │   Sovereign immunity (cannot sue easily)
│  │   │   Strength: Very low default probability
│  │   └─ Weaker Guarantors:
│  │       Individuals, small businesses
│  │       Lower credit quality
│  │       Limit: Usually not acceptable unless very wealthy individual
│  ├─ Guarantee Mechanics:
│  │   ├─ Guarantee Fee:
│  │   │   Charged to borrower (or guarantor)
│  │   │   Typically 0.5-5% per annum
│  │   │   Reflects guarantor PD × LGD + overhead
│  │   │   Example: Guarantor rated A (PD ≈ 0.3%) × LGD 50% = 0.15%, plus 0.35% spread = 0.5% fee
│  │   ├─ Recourse:
│  │   │   Bank can pursue guarantor for full amount upon default
│  │   │   Guarantor can pursue borrower (subrogation rights)
│  │   │   Recourse order matters (secured > unsecured)
│  │   ├─ Triggers:
│  │   │   Guarantee drawn when borrower defaults (typically >30 days)
│  │   │   Must verify default per guarantee terms
│  │   │   Guarantee issuer pays within timeframe (10-30 days typical)
│  │   └─ Cure Rights:
│  │       Guarantor may cure default (pay missed payment)
│  │       Prevents guarantee draw, keeps loan performing
│  │       Allows workout vs. liquidation
│  ├─ Capital Benefit:
│  │   ├─ Recognition:
│  │   │   Regulatory capital: Risk-weighted assets reduced
│  │   │   Internal models: PD × LGD reduced to guarantor levels (if highly-rated)
│  │   │   Example: Borrower PD=5%, LGD=50% → Guarantor PD=0.5% (rated A)
│  │   │   New EL = 0.5% × 50% = 0.25% (80% reduction)
│  │   ├─ Limits:
│  │   │   Must meet regulatory/accounting criteria
│  │   │   Guarantor must be regulated, highly-rated
│  │   │   Correlation with borrower reduces benefit
│  │   │   Example: Subsidiary guaranteed by parent (high correlation)
│  │   └─ Concentration:
│  │       Guarantor concentration: Same guarantor on many credits
│  │       Increases systemic risk; regulators limit
│  │       Concentration charge: Additional capital required
│  └─ Risk:
│      ├─ Guarantor Default:
│      │   Guarantor may not pay when due (moral hazard)
│      │   Weakening correlation: Borrower and guarantor distress related
│      │   Example: 2008: Corporate credit lines drawn when corporates weak
│      ├─ Enforceability:
│      │   Guarantee may be unenforceable (legal challenge)
│      │   Jurisdiction risk: Different laws, rulings
│      │   Operational risk: Guarantor procedure may be complex
│      └─ Substitution Risk:
│          Borrower changes to weaker guarantor
│          Must maintain approval process
├─ Netting:
│  ├─ Bilateral Netting:
│  │   ├─ Definition:
│  │   │   In event of counterparty default:
│  │   │   Net amounts owed in both directions
│  │   │   Only net amount exchanged
│  │   │   Example: Bank A owes Hedge Fund $10M (on equity swap)
│  │   │   Hedge Fund owes Bank A $15M (on interest rate swap)
│  │   │   Net: Hedge Fund pays Bank A $5M (not $15M + offsetting $10M)
│  │   ├─ Exposure Reduction:
│  │   │   Reduces EAD (replacement cost of contracts)
│  │   │   Example: 10 derivative trades with same counterparty
│  │   │   Before netting: Sum of mark-to-market positive values
│  │   │   After netting: Net positive value only
│  │   │   Reduction: 20-50% typical
│  │   ├─ Requirements:
│  │   │   Master agreement (ISDA standard)
│  │   │   Enforceable in counterparty's jurisdiction
│  │   │   Automatic upon default (no counterparty consent)
│  │   └─ Legal Risks:
│  │       Bankruptcy court may not honor netting
│  │       Specific jurisdictions block netting (e.g., some US states)
│  │       Operational: Multiple master agreements complicate
│  ├─ Collateral-Based Netting:
│  │   ├─ CSA (Credit Support Annex):
│  │   │   Collateral posted by counterparties
│  │   │   Reduces replacement cost (EAD)
│  │   │   Cash or securities accepted (typically)
│  │   │   Threshold: No collateral if exposure below threshold
│  │   ├─ Mechanics:
│  │   │   Counterparty A posts $10M when exposure exceeds threshold
│  │   │   Mark-to-market daily
│  │   │   If mark-to-market falls, collateral returned
│  │   │   If rises, additional collateral posted
│  │   ├─ Benefits:
│  │   │   EAD reduction: 50-80% typical
│  │   │   Procyclical: More collateral posted when volatility rises (when needed)
│  │   │   Cheaper than guarantee (no ongoing fee)
│  │   └─ Risks:
│  │       Collateral value may fall (same market stress as exposure)
│  │       Haircuts must account for correlation
│  │       Operational: Daily reconciliation required
│  ├─ Close-Out:
│  │   ├─ Definition:
│  │   │   Upon counterparty default: Unwind all positions
│  │   │   Close-out value = Mark-to-market as of default
│  │   │   Netting applied; net amount exchanged
│  │   ├─ Timing:
│  │   │   Should be prompt (hours to days)
│  │   │   Market conditions change; delay increases risk
│  │   │   Operational capability critical
│  │   └─ Valuation:
│  │       Multiple quotes; independent validation
│  │       Use bid prices (conservative: cost to replace)
│  └─ Central Clearing:
│      ├─ CCP (Central Counterparty):
│      │   Clearinghouse stands between counterparties
│      │   Standardized contracts (futures, cleared swaps)
│      │   Reduces bilateral counterparty risk
│      ├─ Mechanics:
│      │   Counterparty A trades with Counterparty B
│      │   CCP novates: A trades with CCP, CCP trades with B
│      │   A's credit risk: CCP only (not B)
│      │   B's credit risk: CCP only (not A)
│      ├─ Margin:
│      │   Initial margin: Posted upfront per CCP rules
│      │   Variation margin: Daily mark-to-market (paid both directions)
│      │   Guarantee fund: CCP fund covers CCP losses on member default
│      └─ Benefit:
│          Counterparty risk eliminated (CCP is creditworthy)
│          Netting across all participants (multilateral)
│          EAD reduction: 60-90% vs bilateral
├─ Securitization:
│  ├─ Structure:
│  │   ├─ Originator:
│  │   │   Bank/lender originates loans (mortgages, auto, credit card)
│  │   │   Typically retains servicing rights
│  │   ├─ SPV (Special Purpose Vehicle):
│  │   │   Bankruptcy-remote entity
│  │   │   Owns loan pool
│  │   │   Issues securities (MBS, ABS) backed by loans
│  │   ├─ Tranches:
│  │   │   Senior tranche: Paid first, lowest risk, lowest yield (e.g., AAA)
│  │   │   Mezzanine tranches: Paid after senior (e.g., A, BBB)
│  │   │   Subordinated/Equity tranche: Paid last, highest risk, highest yield
│  │   │   Waterfall: Losses absorbed in reverse order (equity first)
│  │   └─ Investors:
│  │       Buy securities in public/private market
│  │       Receive cash flows (principal + interest)
│      ├─ Capital Benefits:
│  │   ├─ Removal from Balance Sheet:
│  │   │   Loans sold to SPV (true sale)
│  │   │   Bank no longer holds credit risk
│  │   │   Reduces RWA (risk-weighted assets)
│  │   │   Capital relief: 10-50% (depends on securitization terms)
│  │   ├─ Reduction of Leverage:
│  │   │   Loan $ no longer on balance sheet → higher ROA
│  │   │   Allows new lending (capacity increase)
│  │   │   Cost: Securitization fees offset benefit
│  │   └─ Return on Capital:
│  │       Origination fee: 0.5-2% (typically 1%)
│  │       Yield difference: Investor buys at discount → bank realizes gain
│  │       Junior note: Bank may retain (and profit if portfolio performs)
│  ├─ Credit Support:
│  │   ├─ Over-Collateralization:
│  │   │   Loan pool value > Securitization value
│  │   │   Example: $1B loans securitized as $900M securities
│  │   │   $100M cushion (10%) covers losses
│  │   │   Protects senior tranches
│  │   ├─ Reserves:
│  │   │   Cash held for expected defaults
│  │   │   Released if actual defaults < expected
│  │   ├─ Subordination:
│  │   │   Subordinated tranches absorb losses first
│  │   │   Senior tranches protected
│  │   │   Tranche hierarchy crucial
│  │   └─ Guarantees:
│  │       Originator may guarantee portfolio performance
│  │       Reduces credit risk but increases originator risk
│  ├─ Risks:
│  │   ├─ Basis Risk:
│  │   │   Portfolio performance diverges from expected
│  │   │   Pool may have worse underwriting than average
│  │   │   Originator incentive mismatch (originate then sell)
│  │   ├─ Liquidity Risk:
│  │   │   Securities may not be tradeable in stress (2008)
│  │   │   Investors forced to hold to maturity
│  │   │   Loss of liquidity premium in yield
│  │   ├─ Model Risk:
│  │   │   Rating models may underestimate tail risk
│  │   │   Correlation assumptions fail in stress
│  │   │   Systematic default correlation (recession)
│  │   └─ Reputational:
│  │       Investor losses → negative PR for originator
│  │       Affects future funding ability
│  │       Regulatory scrutiny
│  └─ Accounting:
│      ├─ True Sale:
│      │   Loans transferred to SPV are no longer originator's asset
│      │   Off-balance-sheet treatment
│      │   Requires control transfer + recourse limits
│      ├─ Consolidation:
│      │   If SPV not truly independent → consolidation required
│      │   Assets/liabilities back on originator's balance sheet
│      │   Defeats capital relief purpose
│      └─ Impairment:
│          Retained junior tranches marked-to-market
│          Losses recorded if valuation falls
├─ Credit Insurance and Derivatives:
│  ├─ Credit Default Swap (CDS):
│  │   ├─ Structure:
│  │   │   Protection buyer pays periodic fee (premium)
│  │   │   Protection seller pays upon credit event
│  │   │   Notional amount: $10M typical
│  │   ├─ Premium:
│  │   │   Usually quoted in basis points (bps) p.a.
│  │   │   Example: 100 bps = 1% annually = $100k per $10M
│  │   │   Reflects PD × LGD + risk premium
│  │   ├─ Payout:
│  │   │   Triggered by: Bankruptcy, failure to pay, restructuring (per definition)
│  │   │   Cash settlement: Par - market value recovered
│  │   │   Example: $10M notional, par 100, market value 40 → payout $600k
│  │   │   Timing: 2-3 days after credit event (settlement lag)
│  │   ├─ Benefits:
│  │   │   Removes credit risk without selling loan/bond
│  │   │   Maintains client relationship (no visible sale)
│  │   │   Tax efficient (no realization event)
│  │   ├─ Risks:
│  │   │   Basis risk: Payoff may not exactly match exposure
│  │   │   Counterparty risk: Protection seller may default
│  │   │   Liquidity: CDS market may be illiquid (wide bid-ask)
│  │   └─ Accounting:
│  │       Fair value hedge: P&L offsets in same period
│  │       Economic hedge, not necessarily accounting hedge
│  ├─ Credit Insurance:
│  │   ├─ Monoline Insurers:
│  │   │   Specialize in credit insurance (bond insurance, CDS)
│  │   │   Rated AAA or AA typically
│  │   │   Provide credit wrapper to lower-rated securities
│  │   ├─ Premium:
│  │   │   Lower than CDS (perceived lower default by insurers)
│  │   │   0.25-2% p.a. typical (depends on risk)
│  │   ├─ Trigger:
│  │   │   Typically: Payment default (not restructuring/downgrade)
│  │   │   Narrower than CDS trigger
│  │   ├─ Benefits:
│  │   │   Lower cost than CDS
│  │   │   Insurance rating benefit (rating uplift)
│  │   │   May enable investment-grade rating
│  │   └─ Risks:
│  │       Insurance may not be honored (2008: AIG near default)
│  │       Counterparty risk high (monoline concentrated on mortgages)
│  │       Regulatory issues (insurance law varies by jurisdiction)
│  ├─ Payer Swaption / Receiver Swaption:
│  │   ├─ Definition:
│  │   │   Option to enter interest rate swap
│  │   │   Payer: Right to pay fixed, receive floating
│  │   │   Receiver: Right to receive fixed, pay floating
│  │   ├─ Uses:
│  │   │   Borrower fears rising rates → payer swaption hedge
│  │   │   If rates rise, exercise payer swap (lock in fixed payment)
│  │   │   If rates fall, let expire (benefit from lower rates)
│  │   ├─ Cost:
│  │   │   Option premium (upfront)
│  │   │   Reflects volatility + strike + duration
│  │   └─ Benefit:
│  │       Asymmetric: Upside uncapped, downside capped
│  └─ Equity Tranche / First Loss Position:
│      ├─ Concept:
│      │   Bank retains highest loss position (first loss)
│      │   Shows confidence in underwriting
│      │   Credit enhancement (absorbs initial defaults)
│      ├─ Example:
│      │   $1B loan pool: Bank retains $100M (10% equity)
│      │   First $100M of losses hit equity
│      │   Senior tranches ($900M) protected
│      ├─ Incentive:
│      │   Aligns originator (keeps best underwriting)
│      │   Reduces moral hazard vs. full securitization
│      ├─ Cost:
│      │   Bank capital tied up
│      │   Return needed to compensate
│      │   Leverage: $100M equity supports $1B credit
│      └─ Recovery:
│          If realized losses < expected → equity profits
│          If losses > expected → equity wiped out
├─ Diversification and Portfolio Effects:
│  ├─ Concentration Risk:
│  │   ├─ Definition:
│  │   │   Exposure to limited number of counterparties/sectors
│  │   │   Reduces benefit of diversification
│  │   │   Increases tail risk
│  │   ├─ Measures:
│  │   │   HHI (Herfindahl index): Σ w_i²
│  │   │   If N equal exposures: HHI = 1/N
│  │   │   HHI = 1: Perfect concentration; HHI = 1/N: Perfect diversification
│  │   ├─ Regulatory Limits:
│  │   │   Large exposure: >10% of capital
│  │   │   Aggregate large exposures: <800% of capital
│  │   │   Related parties (same group): Lower limits
│  │   └─ Portfolio Impact:
│  │       Concentration increases expected loss (portfolio level)
│      ├─ Granularity:
│  │   ├─ Definition:
│  │   │   Many small exposures vs. few large
│  │   │   Granular portfolio: Lower granularity adjustment
│  │   │   Concentrated portfolio: Higher adjustment needed
│  │   ├─ Loss Distribution:
│  │   │   Concentrated: Losses lumpy (few large defaults)
│  │   │   Granular: Losses smooth (many small defaults)
│  │   │   Granularity adjustment: Factor applied to risk weight
│  │   │   Example: 100 exposures of $1M each vs. 1 exposure $100M
│  │   │        Granular portfolio benefits from averaging
│  │   │        Concentrated portfolio needs buffer
│  │   └─ Regulatory Recognition:
│  │       IRB approaches allow granularity adjustments
│  │       Standardized approach assumes granular
│  │       Granularity factor depends on EAD concentration
│  ├─ Correlation & Systemic Risk:
│  │   ├─ Correlation Within Portfolio:
│  │   │   Positive correlation: Defaults cluster (recession)
│  │   │   Negative correlation: Offsetting defaults (rare)
│  │   │   Procyclical: Correlation rises in stress (diversification fails)
│  │   ├─ Systemic Risk:
│  │   │   Correlated defaults across portfolios
│  │   │   Financial crisis → all banks lose simultaneously
│  │   │   Regulatory concern: Reduce systemic tail risk
│  │   └─ Diversification Limits:
│  │       Benefit peaks around 50-100 exposures
│  │       Beyond: Diminishing returns (concentration on sectors/factors)
│  │       Cannot diversify away systematic risk (market/macro)
│  └─ Portfolio Stress Testing:
│      ├─ Concentration Scenarios:
│      │   Large customer default: Impacts 10-20% portfolio
│      │   Sector downturn: Impacts 30-50% portfolio
│      │   Macro recession: Impacts 60%+ portfolio
│      ├─ Combined Losses:
│      │   Large customer + sector downturn
│      │   Estimate potential losses
│      ├─ Capital Adequacy:
│      │   After stress losses: Remaining capital > regulatory minimum?
│      │   If not: Reduce portfolio, raise capital
│      └─ Limits:
│          Set position limits by counterparty
│          Sector limits to reduce concentration
│          Geographic limits (if applicable)
└─ Operational and Legal Framework:
   ├─ Documentation:
   │   ├─ Security Agreement:
   │   │   Legal document creating lien on collateral
   │   │   Recording/filing requirements (vary by jurisdiction)
   │   │   Perfection: Proper filing + notation
   │   ├─ Guarantee Agreement:
   │   │   Guarantor's unconditional promise
   │   │   Defines scope, limitations, waivers
   │   │   Executed by guarantor (authorized officer/individual)
   │   ├─ Master Agreement:
   │   │   ISDA for derivatives (netting, close-out)
   │   │   Creditor association agreements
   │   │   Standardized terms (facilitates enforcement)
   │   └─ Credit Support Annex:
   │       Collateral posting terms
   │       Mark-to-market, haircuts, thresholds
│       Alternative: Collateral Pledge Agreement
│   ├─ Counterparty Management:
│   │   ├─ Ongoing Monitoring:
│   │   │   Credit rating updates
│   │   │   Financial statements quarterly/annual
│   │   │   Market data (CDS spreads, stock price)
│   │   │   Covenant compliance (if applicable)
│   │   ├─ Triggers:
│   │   │   Downgrade below threshold → reduce limit
│   │   │   Negative news → reassess risk
│   │   │   Covenant breach → enforcement
│   │   └─ Escalation:
│   │       Limit breach: Reduce exposure
│   │       Rating downgrade: Review mitigation
│   │       Default risk evident: Prepare workout
│   ├─ Valuation and Haircuts:
│   │   ├─ Collateral Valuation:
│   │   │   Real estate: Appraisal (annually or trigger event)
│   │   │   Securities: Market prices (daily)
│   │   │   Receivables: Aging analysis (monthly)
│   │   ├─ Haircut Recalibration:
│   │   │   Quarterly or upon market stress
│   │   │   Increase haircuts if volatility rises
│   │   │   Example: Stock haircut 30% normal → 50% during market stress
│   │   └─ Model Risk:
│   │       Valuation models may not capture tail risk
│   │       Independent review; model governance
│   ├─ Enforcement:
│   │   ├─ Judicial:
│   │   │   Court proceedings (slow, expensive)
│   │   │   Judgment enforcement (lien on other assets)
│   │   ├─ Non-Judicial:
│   │   │   Self-help remedies: Setoff, liquidation
│   │   │   Faster but limited in scope
│   │   ├─ Settlement:
│   │   │   Negotiate with counterparty (cheaper, faster)
│   │   │   Loan modification, payment plans
│   │   └─ Cost:
│   │       Legal fees: 1-5% of claim
│   │       Enforcement delay: 6-24 months
│   │       Collection rate: 30-80% (varies greatly)
│   └─ Regulatory Framework:
│       ├─ Capital Relief:
│       │   Mitigation recognized in regulatory capital
│       │   Specific rules: Collateral haircuts, guarantee criteria
│       │   Standardized vs. IRB approaches differ
│       ├─ Concentration:
│       │   Large exposure limits (>10% capital)
│       │   Aggregate large exposures limits
│       │   Stress testing requirements
│       └─ Disclosure:
│           Public disclosure of concentration
│           Risk management disclosures (Pillar 3)
│           Quality of mitigation disclosed
```

**Interaction:** Assess credit exposure → Identify mitigation needs (concentration, tail risk) → Structure mitigation (collateral terms, guarantee, netting, securitization) → Negotiate terms → Implement (documentation, enforcement capability) → Monitor (collateral value, counterparty credit, covenant compliance) → Adjust (margin calls, limit breaches) → On default: Execute (liquidate collateral, claim on guarantor, netting) → Recover losses.

## 5. Mini-Project
Comprehensive credit risk mitigation analysis and optimization:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("CREDIT RISK MITIGATION ANALYSIS")
print("="*80)

class MitigationAnalyzer:
    """Analyze mitigation techniques and capital relief"""
    
    def __init__(self):
        pass
    
    def calculate_el(self, pd, lgd, ead):
        """Expected Loss = PD × LGD × EAD"""
        return pd * lgd * ead
    
    def collateral_mitigated_lgd(self, lgd_unsecured, collateral_value, ead, haircut=0.30):
        """
        Calculate LGD with collateral
        
        Mitigated LGD = max(0, (EAD - Collateral × (1-Haircut)) / EAD) × LGD_unsecured
        """
        collateral_effective = collateral_value * (1 - haircut)
        excess_loss = max(0, ead - collateral_effective)
        lgd_mitigated = (excess_loss / ead) * lgd_unsecured
        return lgd_mitigated
    
    def guarantee_mitigated_pd(self, pd_borrower, pd_guarantor, weight=0.8):
        """
        Blend PD: weighted average of borrower and guarantor
        weight: Importance of borrower PD vs guarantor
        """
        pd_blended = weight * pd_borrower + (1 - weight) * pd_guarantor
        return pd_blended
    
    def capital_requirement(self, pd, lgd, ead, rwa_percent=0.08):
        """
        Regulatory capital = RWA × capital ratio
        RWA varies by asset type (standardized) or formula (IRB)
        IRB formula approximation: RWA = N^(-1)(PD) + sqrt(R/(1-R)) × N^(-1)(Confidence%) + sqrt((1-R)/(R)) × N^(-1)(PD)
        Simplified: RWA_percent ≈ f(PD, LGD) varies from 5% to 150%
        """
        from scipy.stats import norm
        
        # Simplified IRB formula for RWA (annual minimum)
        correlation = 0.15  # Simplified assumption
        confidence = 0.999  # 99.9% confidence (regulatory standard)
        
        rwa_pct = (
            norm.ppf(pd) + np.sqrt(correlation / (1 - correlation)) * norm.ppf(confidence)
        ) / np.sqrt(1 - correlation)
        rwa_pct = norm.cdf(rwa_pct) * lgd * 12.5  # 12.5 = 1/0.08
        
        # Floor at standardized levels
        rwa_pct = max(rwa_pct, 0.20)  # Minimum 20% (investment grade bonds)
        rwa_pct = min(rwa_pct, 1.50)  # Maximum 150% (defaulted)
        
        rwa = ead * rwa_pct
        capital_required = rwa * 0.08  # 8% minimum
        
        return {
            'RWA': rwa,
            'RWA_percent': rwa_pct * 100,
            'Capital_Required': capital_required
        }
    
    def net_exposure_netting(self, positive_mtm, negative_mtm):
        """
        Calculate net exposure with bilateral netting
        Before: Sum of all positive values (replacement cost if default)
        After: Net positive value only
        """
        gross_positive = np.sum(positive_mtm)  # What we'd owe if they default
        net_exposure = max(0, gross_positive - np.sum(np.abs(negative_mtm)))
        
        eade_reduction = (1 - net_exposure / gross_positive) * 100 if gross_positive > 0 else 0
        
        return {
            'Gross_Positive_MTM': gross_positive,
            'Net_Exposure': net_exposure,
            'EAD_Reduction_Percent': eade_reduction
        }
    
    def collateral_haircuts_regulatory(self):
        """Basel III haircut schedule for collateral"""
        haircuts = {
            'Cash': 0.00,
            'Government Bonds (AAA)': 0.05,
            'Government Bonds (A)': 0.10,
            'Corporate Bonds (AAA)': 0.10,
            'Corporate Bonds (BBB)': 0.15,
            'Equities (Large Cap)': 0.30,
            'Equities (Small Cap)': 0.40,
            'Real Estate': 0.30,
            'Commodities': 0.50
        }
        return haircuts
    
    def ltv_analysis(self, loan_amount, collateral_value, haircut=0.30):
        """
        Loan-to-Value analysis
        LTV = Loan / (Collateral × (1 - Haircut))
        """
        collateral_effective = collateral_value * (1 - haircut)
        ltv = loan_amount / collateral_effective
        
        return {
            'Loan_Amount': loan_amount,
            'Collateral_Gross': collateral_value,
            'Collateral_Net': collateral_effective,
            'LTV_Percent': ltv * 100,
            'Comfortable': 'Yes' if ltv < 0.8 else 'Warning' if ltv < 1.0 else 'Risky'
        }
    
    def securitization_capital_relief(self, loan_pool_value, securitization_size, 
                                     senior_percent=0.80, capital_ratio_before=0.10):
        """
        Estimate capital relief from securitization
        """
        junior_retention = 1 - senior_percent  # Originator retains junior
        junior_size = securitization_size * junior_retention
        
        # RWA before: 100% of loan pool (rough)
        rwa_before = loan_pool_value * 1.0
        
        # RWA after: Only junior tranche retained (at higher RW)
        rwa_after = junior_size * 1.5  # Higher RW for riskier piece
        
        capital_before = rwa_before * capital_ratio_before
        capital_after = rwa_after * capital_ratio_before
        capital_relief = capital_before - capital_after
        
        return {
            'Loan_Pool_Value': loan_pool_value,
            'Securitization_Size': securitization_size,
            'Senior_Size': securitization_size * senior_percent,
            'Junior_Retained': junior_size,
            'RWA_Before': rwa_before,
            'RWA_After': rwa_after,
            'Capital_Before': capital_before,
            'Capital_After': capital_after,
            'Capital_Relief': capital_relief,
            'Relief_Percent': (capital_relief / capital_before * 100) if capital_before > 0 else 0
        }
    
    def cds_premium_analysis(self, pd, lgd, risk_premium=0.001):
        """
        Estimate CDS premium from PD and LGD
        Approximation: Premium ≈ PD × LGD + Risk Premium
        """
        base_premium = pd * lgd * 10000  # in basis points
        total_premium = base_premium + risk_premium * 10000
        
        return {
            'PD': pd,
            'LGD': lgd,
            'Expected_Loss': pd * lgd,
            'Base_Premium_bps': base_premium,
            'Risk_Premium_bps': risk_premium * 10000,
            'Total_Premium_bps': total_premium
        }

# Scenario 1: Collateral Mitigation
print("\n" + "="*80)
print("SCENARIO 1: COLLATERAL-BASED MITIGATION")
print("="*80)

analyzer = MitigationAnalyzer()

loan_amount = 10_000_000  # $10M
collateral_value = 8_000_000  # $8M real estate
pd = 0.03  # 3% probability of default
lgd_unsecured = 0.60  # 60% loss if defaults
haircut = 0.30  # 30% haircut on real estate

# Unsecured
el_unsecured = analyzer.calculate_el(pd, lgd_unsecured, loan_amount)
cap_unsecured = analyzer.capital_requirement(pd, lgd_unsecured, loan_amount)

# With collateral
lgd_secured = analyzer.collateral_mitigated_lgd(lgd_unsecured, collateral_value, loan_amount, haircut)
el_secured = analyzer.calculate_el(pd, lgd_secured, loan_amount)
cap_secured = analyzer.capital_requirement(pd, lgd_secured, loan_amount)

print(f"\nUnsecured Loan:")
print(f"  Expected Loss: ${el_unsecured:,.0f}")
print(f"  RWA: ${cap_unsecured['RWA']:,.0f} ({cap_unsecured['RWA_percent']:.1f}%)")
print(f"  Capital Required (8%): ${cap_unsecured['Capital_Required']:,.0f}")

print(f"\nWith Collateral:")
print(f"  Collateral Value: ${collateral_value:,.0f}")
print(f"  Haircut: {haircut*100:.0f}%")
print(f"  Net Collateral Value: ${collateral_value * (1 - haircut):,.0f}")
print(f"  Mitigated LGD: {lgd_secured*100:.1f}% (vs {lgd_unsecured*100:.0f}%)")
print(f"  Expected Loss: ${el_secured:,.0f}")
print(f"  RWA: ${cap_secured['RWA']:,.0f} ({cap_secured['RWA_percent']:.1f}%)")
print(f"  Capital Required: ${cap_secured['Capital_Required']:,.0f}")

print(f"\nCapital Relief:")
print(f"  Capital Savings: ${cap_unsecured['Capital_Required'] - cap_secured['Capital_Required']:,.0f}")
print(f"  Percent Reduction: {(1 - cap_secured['Capital_Required']/cap_unsecured['Capital_Required'])*100:.1f}%")

# LTV Analysis
ltv = analyzer.ltv_analysis(loan_amount, collateral_value, haircut)
print(f"\nLoan-to-Value Analysis:")
for key, val in ltv.items():
    if '_' in key:
        if 'Percent' in key:
            print(f"  {key}: {val:.1f}%")
        else:
            print(f"  {key}: ${val:,.0f}")
    else:
        print(f"  {key}: {val}")

# Scenario 2: Guarantee Mitigation
print("\n" + "="*80)
print("SCENARIO 2: GUARANTEE-BASED MITIGATION")
print("="*80)

pd_borrower = 0.05  # Weak borrower
pd_guarantor = 0.001  # Strong parent (rated A)
lgd = 0.50

# Unsecured
el_guarantee_before = analyzer.calculate_el(pd_borrower, lgd, loan_amount)
cap_guarantee_before = analyzer.capital_requirement(pd_borrower, lgd, loan_amount)

# With guarantee
pd_blended = analyzer.guarantee_mitigated_pd(pd_borrower, pd_guarantor, weight=0.5)  # Weight parent heavily
el_guarantee_after = analyzer.calculate_el(pd_blended, lgd, loan_amount)
cap_guarantee_after = analyzer.capital_requirement(pd_blended, lgd, loan_amount)

guarantee_fee = 0.005 * loan_amount  # 50 bps per annum

print(f"\nWithout Guarantee:")
print(f"  Borrower PD: {pd_borrower*100:.2f}%")
print(f"  Expected Loss: ${el_guarantee_before:,.0f}")
print(f"  Capital Required: ${cap_guarantee_before['Capital_Required']:,.0f}")

print(f"\nWith Parent Company Guarantee:")
print(f"  Guarantor PD: {pd_guarantor*100:.3f}%")
print(f"  Blended PD: {pd_blended*100:.3f}%")
print(f"  Expected Loss: ${el_guarantee_after:,.0f}")
print(f"  Capital Required: ${cap_guarantee_after['Capital_Required']:,.0f}")
print(f"  Guarantee Fee: ${guarantee_fee:,.0f} (50 bps p.a.)")

print(f"\nCapital Relief:")
print(f"  Capital Savings: ${cap_guarantee_before['Capital_Required'] - cap_guarantee_after['Capital_Required']:,.0f}")
print(f"  Percent Reduction: {(1 - cap_guarantee_after['Capital_Required']/cap_guarantee_before['Capital_Required'])*100:.1f}%")
print(f"  Cost (Fee): ${guarantee_fee:,.0f}")
print(f"  Net Benefit (Annual): ${(cap_guarantee_before['Capital_Required'] - cap_guarantee_after['Capital_Required'])*0.10 - guarantee_fee:,.0f}")

# Scenario 3: Netting
print("\n" + "="*80)
print("SCENARIO 3: BILATERAL NETTING AND CSA COLLATERAL")
print("="*80)

# Generate portfolio of derivatives with counterparty
np.random.seed(42)
n_trades = 10
positive_mtm = np.random.exponential(500_000, size=n_trades//2)  # Trades we'd receive if they default
negative_mtm = np.random.exponential(400_000, size=n_trades//2)  # Trades we'd pay if they default

netting_result = analyzer.net_exposure_netting(positive_mtm, negative_mtm)

print(f"\nDerivative Portfolio (10 trades with single counterparty):")
print(f"  Gross Positive MTM (we'd receive): ${netting_result['Gross_Positive_MTM']:,.0f}")
print(f"  Gross Negative MTM (we'd pay): ${np.sum(np.abs(negative_mtm)):,.0f}")
print(f"  Without Netting, EAD would be: ${netting_result['Gross_Positive_MTM']:,.0f}")
print(f"  With Netting, Net EAD is: ${netting_result['Net_Exposure']:,.0f}")
print(f"  EAD Reduction: {netting_result['EAD_Reduction_Percent']:.1f}%")

# CSA Collateral
csa_collateral = 2_000_000  # Counterparty posted $2M collateral
csa_haircut = 0.10  # 10% haircut on cash equivalent
csa_collateral_effective = csa_collateral * (1 - csa_haircut)
csa_protected_exposure = min(netting_result['Net_Exposure'], csa_collateral_effective)
csa_unprotected = max(0, netting_result['Net_Exposure'] - csa_collateral_effective)

print(f"\nCSA Collateral:")
print(f"  Collateral Posted: ${csa_collateral:,.0f}")
print(f"  Haircut: {csa_haircut*100:.0f}%")
print(f"  Effective Collateral: ${csa_collateral_effective:,.0f}")
print(f"  Protected Exposure: ${csa_protected_exposure:,.0f}")
print(f"  Unprotected Exposure: ${csa_unprotected:,.0f}")
print(f"  Total EAD with Netting + CSA: ${csa_unprotected:,.0f}")

# Scenario 4: Securitization
print("\n" + "="*80)
print("SCENARIO 4: SECURITIZATION AND CAPITAL RELIEF")
print("="*80)

loan_pool_value = 500_000_000  # $500M loan portfolio
securitization_size = 400_000_000  # Securitize $400M
senior_pct = 0.85  # 85% senior, 15% junior

securitization_result = analyzer.securitization_capital_relief(
    loan_pool_value, securitization_size, senior_percent=senior_pct
)

print(f"\nLoan Portfolio:")
print(f"  Total Value: ${securitization_result['Loan_Pool_Value']:,.0f}")
print(f"  Amount to Securitize: ${securitization_result['Securitization_Size']:,.0f}")

print(f"\nSecuritization Structure:")
print(f"  Senior Tranche (AAA): ${securitization_result['Senior_Size']:,.0f} ({senior_pct*100:.0f}%)")
print(f"  Junior Tranche (retained): ${securitization_result['Junior_Retained']:,.0f} ({(1-senior_pct)*100:.0f}%)")

print(f"\nCapital Impact:")
print(f"  RWA Before: ${securitization_result['RWA_Before']:,.0f}")
print(f"  RWA After: ${securitization_result['RWA_After']:,.0f}")
print(f"  Capital Before (8%): ${securitization_result['Capital_Before']:,.0f}")
print(f"  Capital After (8%): ${securitization_result['Capital_After']:,.0f}")
print(f"  Capital Relief: ${securitization_result['Capital_Relief']:,.0f}")
print(f"  Percent Reduction: {securitization_result['Relief_Percent']:.1f}%")

fees = securitization_result['Securitization_Size'] * 0.015  # 150 bps fee
print(f"\nSecuritization Costs:")
print(f"  Underwriting/Rating Fees (150 bps): ${fees:,.0f}")
print(f"  Net Benefit (after fees): ${securitization_result['Capital_Relief']*0.10 - fees:,.0f}")

# Scenario 5: CDS Premium Analysis
print("\n" + "="*80)
print("SCENARIO 5: CREDIT INSURANCE (CDS PRICING)")
print("="*80)

borrower_ratings = {
    'AAA': {'PD': 0.0001, 'LGD': 0.30},
    'A': {'PD': 0.0010, 'LGD': 0.40},
    'BBB': {'PD': 0.0050, 'LGD': 0.50},
    'BB': {'PD': 0.0300, 'LGD': 0.60},
    'B': {'PD': 0.1000, 'LGD': 0.70}
}

print(f"\nCDS Premium (5-year, representative pricing):")
print(f"{'Rating':<10} {'PD':<10} {'LGD':<10} {'Premium (bps)':<15}")
print("-" * 50)

cds_premiums = {}
for rating, params in borrower_ratings.items():
    cds_analysis = analyzer.cds_premium_analysis(params['PD'], params['LGD'])
    cds_premiums[rating] = cds_analysis['Total_Premium_bps']
    print(f"{rating:<10} {params['PD']*100:>8.2f}% {params['LGD']*100:>8.0f}% {cds_analysis['Total_Premium_bps']:>13.0f}")

# Scenario 6: Concentration and Diversification
print("\n" + "="*80)
print("SCENARIO 6: CONCENTRATION VS DIVERSIFICATION")
print("="*80)

# Concentrated portfolio
concentrated_exposures = np.array([100_000_000])  # One $100M exposure
concentrated_hhi = np.sum((concentrated_exposures / np.sum(concentrated_exposures))**2)

# Diversified portfolio
diversified_exposures = np.ones(100) * 1_000_000  # 100 $1M exposures
diversified_hhi = np.sum((diversified_exposures / np.sum(diversified_exposures))**2)

print(f"\nPortfolio Concentration Comparison:")
print(f"  Concentrated (1 × $100M):")
print(f"    Total Exposure: ${np.sum(concentrated_exposures):,.0f}")
print(f"    HHI: {concentrated_hhi:.4f}")
print(f"    Assessment: High concentration")

print(f"\n  Diversified (100 × $1M):")
print(f"    Total Exposure: ${np.sum(diversified_exposures):,.0f}")
print(f"    HHI: {diversified_hhi:.4f}")
print(f"    Assessment: Well diversified")

# Estimate loss distribution
single_default_concentrated = concentrated_exposures[0] * 0.50  # 50% LGD
single_default_diversified = diversified_exposures[0] * 0.50

print(f"\n  Single Default Loss Impact:")
print(f"    Concentrated: ${single_default_concentrated:,.0f} ({single_default_concentrated/np.sum(concentrated_exposures)*100:.1f}%)")
print(f"    Diversified: ${single_default_diversified:,.0f} ({single_default_diversified/np.sum(diversified_exposures)*100:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: LGD Mitigation (Collateral)
ax = axes[0, 0]
scenarios = ['Unsecured', 'With Collateral']
lgds = [lgd_unsecured * 100, lgd_secured * 100]
colors = ['red', 'green']
ax.bar(scenarios, lgds, color=colors, alpha=0.7)
ax.set_ylabel('LGD (%)')
ax.set_title('LGD Reduction via Collateral')
ax.set_ylim(0, 70)
for i, v in enumerate(lgds):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 2: Capital Reduction (Collateral)
ax = axes[0, 1]
scenarios = ['Unsecured', 'With Collateral']
capitals = [cap_unsecured['Capital_Required']/1000, cap_secured['Capital_Required']/1000]
ax.bar(scenarios, capitals, color=colors, alpha=0.7)
ax.set_ylabel('Capital Required ($k)')
ax.set_title('Capital Relief via Collateral')
for i, v in enumerate(capitals):
    ax.text(i, v + 10, f'${v:.0f}k', ha='center', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 3: PD Reduction (Guarantee)
ax = axes[0, 2]
scenarios = ['Borrower Only', 'With Guarantee']
pds = [pd_borrower * 100, pd_blended * 100]
ax.bar(scenarios, pds, color=['red', 'green'], alpha=0.7)
ax.set_ylabel('PD (%)')
ax.set_title('PD Reduction via Guarantee')
for i, v in enumerate(pds):
    ax.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Netting Benefit
ax = axes[1, 0]
scenarios = ['No Netting', 'With Netting', 'With Netting + CSA']
exposures = [
    netting_result['Gross_Positive_MTM']/1_000_000,
    netting_result['Net_Exposure']/1_000_000,
    csa_unprotected/1_000_000
]
ax.bar(scenarios, exposures, color=['red', 'orange', 'green'], alpha=0.7)
ax.set_ylabel('EAD ($M)')
ax.set_title('EAD Reduction via Netting and CSA')
for i, v in enumerate(exposures):
    ax.text(i, v + 0.1, f'${v:.1f}M', ha='center', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 5: CDS Premium Curve
ax = axes[1, 1]
ratings_list = list(borrower_ratings.keys())
premiums = [cds_premiums[r] for r in ratings_list]
colors_rating = ['green', 'blue', 'yellow', 'orange', 'red']
ax.plot(ratings_list, premiums, marker='o', linewidth=2, markersize=8, color='darkblue')
ax.fill_between(range(len(ratings_list)), premiums, alpha=0.3)
ax.set_ylabel('CDS Premium (basis points)')
ax.set_title('CDS Premium by Credit Rating')
ax.grid(alpha=0.3)
for i, (rating, premium) in enumerate(zip(ratings_list, premiums)):
    ax.text(i, premium + 10, f'{premium:.0f}', ha='center', fontsize=9)

# Plot 6: Concentration Impact (HHI)
ax = axes[1, 2]
portfolio_types = ['Concentrated\n(1 × $100M)', 'Diversified\n(100 × $1M)']
hhis = [concentrated_hhi, diversified_hhi]
colors_hhi = ['red', 'green']
ax.bar(portfolio_types, hhis, color=colors_hhi, alpha=0.7)
ax.set_ylabel('HHI (Concentration Index)')
ax.set_title('Portfolio Concentration Comparison')
ax.axhline(0.1, color='orange', linestyle='--', linewidth=2, label='Moderate threshold')
for i, v in enumerate(hhis):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Summary Table
print("\n" + "="*80)
print("MITIGATION SUMMARY TABLE")
print("="*80)

summary_data = {
    'Technique': [
        'Unsecured Baseline',
        'Collateral (30% haircut)',
        'Guarantee (PD 0.1%)',
        'Netting + CSA',
        'Securitization'
    ],
    'Capital_Before_k': [
        cap_unsecured['Capital_Required']/1000,
        cap_unsecured['Capital_Required']/1000,
        cap_guarantee_before['Capital_Required']/1000,
        cap_unsecured['Capital_Required']/1000,
        securitization_result['Capital_Before']/1000
    ],
    'Capital_After_k': [
        cap_unsecured['Capital_Required']/1000,
        cap_secured['Capital_Required']/1000,
        cap_guarantee_after['Capital_Required']/1000,
        (csa_unprotected*1.0*0.08*0.08)/1000,
        securitization_result['Capital_After']/1000
    ],
    'Relief_Percent': [
        0.0,
        (1 - cap_secured['Capital_Required']/cap_unsecured['Capital_Required'])*100,
        (1 - cap_guarantee_after['Capital_Required']/cap_guarantee_before['Capital_Required'])*100,
        (1 - (csa_unprotected*1.0*0.08*0.08)/(cap_unsecured['Capital_Required']))*100,
        securitization_result['Relief_Percent']
    ],
    'Annual_Cost_k': [
        0.0,
        (0.001 * loan_amount)/1000,  # 10 bps custody/monitoring
        guarantee_fee/1000,
        0.0,  # Netting no ongoing fee
        fees/1000
    ]
}

df_summary = pd.DataFrame(summary_data)
print("\n")
print(df_summary.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("1. Collateral reduces LGD significantly → lowers capital")
print("2. Guarantees from strong counterparties reduce PD → lower capital")
print("3. Netting is powerful (20-40% EAD reduction) and low-cost")
print("4. Securitization transfers risk off balance sheet → significant capital relief")
print("5. CDS pricing reflects credit quality (exponential with rating deterioration)")
print("6. Diversification is critical: 100 small exposures vs 1 large (concentration risk)")
print("7. Capital relief must offset mitigation costs to be economically valuable")
```

## 6. Challenge Round
1. **Optimal Mitigation Portfolio:** Given 100 credits (PDs, LGDs, sizes), determine optimal mitigation mix (collateral %, guarantees, securitization %) to minimize total cost (mitigation fees + capital charge) subject to regulatory constraints. Use linear/quadratic programming.

2. **Collateral Haircut Stress:** Model collateral value decline across asset classes (equities -50%, real estate -30%, bonds -10%) during market stress. Calculate LGD changes, margin calls, capital implications. When do mark-to-market haircuts dominate?

3. **Correlation in Guarantees:** Simulate portfolio where guarantor PD correlates with borrower PD (during recession both distressed). Estimate correlation impact on guarantee benefit. Compare standalone PD reduction vs. correlated scenario.

4. **Securitization Basis Risk:** Pool contains 500 mortgages with PDs and LGDs. Simulate defaults, losses. Compare expected losses to securitized tranches' expected payouts. Does mezzanine tranche premium cover tail risk adequately?

5. **CDS-Collateral Redundancy:** Hedge credit risk using both CDS and collateral. Optimize allocation to minimize joint costs. When is CDS preferred over collateral? (CDS liquid, collateral less costly if good assets available)

## 7. Key References
- [Basel Committee, "International Convergence of Capital Measurement and Capital Standards" (2006/2017, Basel III)](https://www.bis.org/bcbs/basel3.htm) - regulatory framework for credit risk mitigation recognition
- [Altman & Saunders, "Credit Risk Measurement: Developments Over the Last 20 Years" (2001)](https://www.jstor.org/stable/2673960) - evolution of credit risk models and mitigation techniques
- [Hull & White, "The Impact of Default Risk on the Prices of Options and Currency Swaps" (1995)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1003265) - credit risk in derivative pricing and mitigation

---
**Status:** Operational risk management and capital optimization | **Complements:** Credit Risk Fundamentals, Expected Loss Calculation, Credit Derivatives, Regulatory Capital, Portfolio Management
