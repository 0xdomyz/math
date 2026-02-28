# Credit Risk Mitigation

## 1. Concept Skeleton
**Definition:** Techniques to reduce credit exposure; collateral, guarantees, netting, securitization, insurance; regulatory capital benefits; risk transfer mechanisms; contingent claims on default; operational and legal frameworks  
**Purpose:** Lower expected loss (EL = PD Ã— LGD Ã— EAD); reduce portfolio concentration; transfer tail risks; optimize regulatory capital allocation; manage systemic risk; cost-effective risk reduction vs. reduce lending  
**Prerequisites:** Credit risk fundamentals, probability of default, loss given default, exposure at default, credit derivatives, financial instruments, legal contracts, regulatory frameworks (Basel III)

## 2. Comparative Framing
| Mitigation Technique | Mechanism | Capital Benefit | Cost | Effectiveness | When to Use |
|-------------------|-----------|-----------------|------|----------------|------------|
| **Collateral** | Secure claim on assets; reduce LGD | Up to 60% RW reduction | Custody, valuation, liquidation risk | High for secured lending | Mortgages, auto loans, trade finance |
| **Guarantees** | Third-party absorbs loss if borrower defaults | PD Ã— LGD reduction (depends on guarantor) | Guarantee fee (0.5-2% p.a.) | Depends on guarantor credit quality | Weak borrowers, structured deals |
| **Netting** | Offset positions with same counterparty | Lowers EAD for portfolios | Legal/operational setup | High (reduces EAD by 20-40%) | Derivatives, OTC trading |
| **Securitization** | Convert loans to tradeable securities; sell risk | Removes from balance sheet (capital relief) | Origination, underwriting, servicing (1-3%) | High if rated/distributed | Loan portfolios, mortgage banks |
| **Credit Insurance** | Protection seller pays if credit event | Reduces effective PD Ã— LGD | Insurance premium (0.5-5% p.a.) | High but basis risk exists | Tail risks, concentrated exposures |
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
Bank lends $100M to real estate developer. Unsecured: PD=3%, LGD=60%, EAD=$100M â†’ EL=$1.8M. Require 50% collateral (real estate): LGD drops to 30% â†’ EL=$0.9M. Capital requirement: ~40% lower. Cost: Collateral monitoring 0.1% p.a. = $100k/year. Benefit/cost: 2x positive (net $800k capital relief).

**Perfect Fit:**  
Mortgage bank securitizes $1B portfolio. Rates mortgage-backed securities (MBS). Senior tranche rated AAA, junior tranche rated BBB. Sell senior (80% of deal) to insurance company. Junior retained (10%) as credit support. Bank keeps $100M servicing strip. Capital relief: $800M Ã— RW=20% = $160M capital equivalent. Cost: Underwriting, rating agency fees 0.5% = $5M. NPV: Positive if bank would otherwise hold portfolio.

**Over-Collateralization:**  
Lend to startup on 150% collateral (excess of loan value). If asset value drops 40% â†’ collateral now 90% of loan â†’ loses mitigation benefit. Mark-to-market, rebalance periodically. Haircuts critical: Apply 30% to volatile assets.

**Guarantee Failure:**  
Bank A guarantees Bank B's exposure to Company X. Company X defaults. Bank B claims. Bank A refuses (alleges procedural error). Legal dispute â†’ 18 months of delay. Guarantee only effective if guarantor solvent and operationally responsive. Guarantor credit quality paramount.

**Poor Netting Setup:**  
Bank X signs master agreement with hedge fund covering 100 derivatives. Hedge fund defaults; courts rule netting unenforceable (jurisdiction issue). Must liquidate positions individually â†’ higher transaction costs, worse execution. Netting benefit lost. Legal setup critical.

**Concentration in Mitigation:**  
Portfolio: 500 small loans, each guaranteed by same parent company (holding company structure). Portfolio risk now concentrated on parent â†’ higher systemic risk. Diversification benefit lost; mitigating one risk (borrower PD) creates new risk (guarantor PD).

## 4. Layer Breakdown
```
Credit Risk Mitigation Framework:

â”œâ”€ Collateral-Based Mitigation:
â”‚  â”œâ”€ Types of Collateral:
â”‚  â”‚   â”œâ”€ Real Property:
â”‚  â”‚   â”‚   Land, buildings (mortgages)
â”‚  â”‚   â”‚   Value stable, liquid (months to sell)
â”‚  â”‚   â”‚   Haircut: 20-30% (for mortgages vs current market)
â”‚  â”‚   â”‚   Time to liquidate: 3-12 months
â”‚  â”‚   â”œâ”€ Financial Collateral:
â”‚  â”‚   â”‚   Stocks, bonds, cash
â”‚  â”‚   â”‚   Highly liquid, price volatility
â”‚  â”‚   â”‚   Haircut: 5-50% depending on asset (equity higher)
â”‚  â”‚   â”‚   Time to liquidate: Days to hours
â”‚  â”‚   â”œâ”€ Trade Receivables:
â”‚  â”‚   â”‚   Invoices, accounts receivable
â”‚  â”‚   â”‚   Concentration risk (customer), default risk (debtor)
â”‚  â”‚   â”‚   Haircut: 20-40%
â”‚  â”‚   â”‚   Time to liquidate: 30-90 days
â”‚  â”‚   â”œâ”€ Inventory/Commodities:
â”‚  â”‚   â”‚   Goods held, raw materials
â”‚  â”‚   â”‚   Price volatile, storage costs, obsolescence
â”‚  â”‚   â”‚   Haircut: 40-70%
â”‚  â”‚   â”‚   Time to liquidate: 1-3 months
â”‚  â”‚   â””â”€ Equipment/Vehicles:
â”‚  â”‚       Used asset values decline quickly
â”‚  â”‚       Haircut: 40-60%
â”‚  â”‚       Time to liquidate: 1-3 months
â”‚  â”œâ”€ Collateral Valuation:
â”‚  â”‚   â”œâ”€ Initial Valuation:
â”‚  â”‚   â”‚   Fair value at loan origination
â”‚  â”‚   â”‚   Third-party appraisal typical
â”‚  â”‚   â”‚   Cost: 0.1-1% of loan (depending on asset)
â”‚  â”‚   â”œâ”€ Mark-to-Market:
â”‚  â”‚   â”‚   Daily or periodic revaluation
â”‚  â”‚   â”‚   For securities: Market prices available
â”‚  â”‚   â”‚   For property: Annual or event-driven
â”‚  â”‚   â”œâ”€ Haircuts:
â”‚  â”‚   â”‚   â”‚ Conservative discount to market value
â”‚  â”‚   â”‚   â”‚ Accounts for liquidation costs, market stress
â”‚  â”‚   â”‚   â”‚ Regulatory haircuts: Basel III specifies (5-75% by asset type)
â”‚  â”‚   â”‚   â”‚ Internal models may use lower (with approval)
â”‚  â”‚   â”‚   â””â”€ Example:
â”‚  â”‚   â”‚       Stock market value $100M
â”‚  â”‚   â”‚       Haircut 30%
â”‚  â”‚   â”‚       Collateral value = $70M (for loan purposes)
â”‚  â”‚   â”œâ”€ Substitution:
â”‚  â”‚   â”‚   Borrower may substitute collateral
â”‚  â”‚   â”‚   Typical restrictions: Same or higher quality, liquid
â”‚  â”‚   â”‚   Bank approval required
â”‚  â”‚   â”‚   Reduces operational friction
â”‚  â”‚   â””â”€ Concentration Risk:
â”‚  â”‚       Collateral concentrated in single asset/issuer
â”‚  â”‚       Reduces benefit (correlated default)
â”‚  â”‚       Limits: No single collateral > 10-25% of value
â”‚  â”œâ”€ Loan-to-Value (LTV) Ratio:
â”‚  â”‚   â”œâ”€ Definition:
â”‚  â”‚   â”‚   LTV = Loan Amount / Collateral Value
â”‚  â”‚   â”‚   LTV > 100%: Under-collateralized
â”‚  â”‚   â”‚   LTV = 100%: Fully collateralized
â”‚  â”‚   â”‚   LTV < 100%: Over-collateralized
â”‚  â”‚   â”œâ”€ Risk Dynamics:
â”‚  â”‚   â”‚   As collateral value falls â†’ LTV rises â†’ margin call
â”‚  â”‚   â”‚   Borrower must add collateral or reduce loan
â”‚  â”‚   â”‚   Procyclical: Rising rates/volatility â†’ collateral falls â†’ forced sales
â”‚  â”‚   â”œâ”€ Regulatory Limits:
â”‚  â”‚   â”‚   Residential mortgages: LTV â‰¤ 80% typical
â”‚  â”‚   â”‚   Commercial real estate: LTV â‰¤ 60-70%
â”‚  â”‚   â”‚   Securities lending: LTV â‰¤ 50-100% (depends on collateral)
â”‚  â”‚   â””â”€ Stress Testing:
â”‚  â”‚       Assume collateral value falls 20-50%
â”‚  â”‚       Calculate new LTV; assess need for intervention
â”‚  â”‚       Example: Real estate down 30% â†’ LTV from 70% to 100%
â”‚  â””â”€ Legal and Operational:
â”‚      â”œâ”€ Security Interest:
â”‚      â”‚   Perfected interest: Bank has priority claim
â”‚      â”‚   Filing requirements vary by jurisdiction/asset type
â”‚      â”‚   Continuous monitoring for lapses
â”‚      â”œâ”€ Custody:
â”‚      â”‚   Physical possession (jewels, art) or account control
â”‚      â”‚   Segregated from bank's assets (reduce moral hazard)
â”‚      â”‚   Third-party custodian possible (expense)
â”‚      â””â”€ Enforcement:
â”‚          Rights upon default: Seize, liquidate, absorb losses
â”‚          Legal process delays; time-to-recover can be years
â”‚          Proceeds applied: Secured creditors first
â”œâ”€ Guarantees and Credit Enhancements:
â”‚  â”œâ”€ Guarantee Structure:
â”‚  â”‚   â”œâ”€ Full Guarantee:
â”‚  â”‚   â”‚   Guarantor liable for 100% of debt (principal + interest)
â”‚  â”‚   â”‚   Bank can pursue guarantor if borrower defaults
â”‚  â”‚   â”‚   Example: Parent company guarantees subsidiary loan
â”‚  â”‚   â”œâ”€ Partial Guarantee:
â”‚  â”‚   â”‚   Guarantor liable for portion (e.g., 50%)
â”‚  â”‚   â”‚   Splits risk between guarantor and bank
â”‚  â”‚   â”‚   Example: Government guarantee on small business loan (80%)
â”‚  â”‚   â”œâ”€ Stand-by Guarantee:
â”‚  â”‚   â”‚   Drawn only if primary source fails
â”‚  â”‚   â”‚   Example: LC (letter of credit) backed by guarantee
â”‚  â”‚   â”‚   Reduces frequency of draw (lower cost)
â”‚  â”‚   â””â”€ Performance Guarantee:
â”‚  â”‚       Guarantees performance (not financial payment)
â”‚  â”‚       Example: Contractor guarantees project completion
â”‚  â”œâ”€ Types of Guarantors:
â”‚  â”‚   â”œâ”€ Corporate Guarantor:
â”‚  â”‚   â”‚   Parent company, affiliate
â”‚  â”‚   â”‚   Usually highly rated (investment grade)
â”‚  â”‚   â”‚   Strength: Large balance sheet, multiple income sources
â”‚  â”‚   â”œâ”€ Government Guarantor:
â”‚  â”‚   â”‚   National, regional, or local government
â”‚  â”‚   â”‚   Strength: Tax power, money supply, perceived low default
â”‚  â”‚   â”‚   Risk: Regulatory change, political risk
â”‚  â”‚   â”œâ”€ Financial Institution:
â”‚  â”‚   â”‚   Bank, insurance company
â”‚  â”‚   â”‚   Usually rated A or higher
â”‚  â”‚   â”‚   Strength: Capitalized, regulated, liquid
â”‚  â”‚   â”œâ”€ Multilateral Institution:
â”‚  â”‚   â”‚   World Bank, regional development bank
â”‚  â”‚   â”‚   Sovereign immunity (cannot sue easily)
â”‚  â”‚   â”‚   Strength: Very low default probability
â”‚  â”‚   â””â”€ Weaker Guarantors:
â”‚  â”‚       Individuals, small businesses
â”‚  â”‚       Lower credit quality
â”‚  â”‚       Limit: Usually not acceptable unless very wealthy individual
â”‚  â”œâ”€ Guarantee Mechanics:
â”‚  â”‚   â”œâ”€ Guarantee Fee:
â”‚  â”‚   â”‚   Charged to borrower (or guarantor)
â”‚  â”‚   â”‚   Typically 0.5-5% per annum
â”‚  â”‚   â”‚   Reflects guarantor PD Ã— LGD + overhead
â”‚  â”‚   â”‚   Example: Guarantor rated A (PD â‰ˆ 0.3%) Ã— LGD 50% = 0.15%, plus 0.35% spread = 0.5% fee
â”‚  â”‚   â”œâ”€ Recourse:
â”‚  â”‚   â”‚   Bank can pursue guarantor for full amount upon default
â”‚  â”‚   â”‚   Guarantor can pursue borrower (subrogation rights)
â”‚  â”‚   â”‚   Recourse order matters (secured > unsecured)
â”‚  â”‚   â”œâ”€ Triggers:
â”‚  â”‚   â”‚   Guarantee drawn when borrower defaults (typically >30 days)
â”‚  â”‚   â”‚   Must verify default per guarantee terms
â”‚  â”‚   â”‚   Guarantee issuer pays within timeframe (10-30 days typical)
â”‚  â”‚   â””â”€ Cure Rights:
â”‚  â”‚       Guarantor may cure default (pay missed payment)
â”‚  â”‚       Prevents guarantee draw, keeps loan performing
â”‚  â”‚       Allows workout vs. liquidation
â”‚  â”œâ”€ Capital Benefit:
â”‚  â”‚   â”œâ”€ Recognition:
â”‚  â”‚   â”‚   Regulatory capital: Risk-weighted assets reduced
â”‚  â”‚   â”‚   Internal models: PD Ã— LGD reduced to guarantor levels (if highly-rated)
â”‚  â”‚   â”‚   Example: Borrower PD=5%, LGD=50% â†’ Guarantor PD=0.5% (rated A)
â”‚  â”‚   â”‚   New EL = 0.5% Ã— 50% = 0.25% (80% reduction)
â”‚  â”‚   â”œâ”€ Limits:
â”‚  â”‚   â”‚   Must meet regulatory/accounting criteria
â”‚  â”‚   â”‚   Guarantor must be regulated, highly-rated
â”‚  â”‚   â”‚   Correlation with borrower reduces benefit
â”‚  â”‚   â”‚   Example: Subsidiary guaranteed by parent (high correlation)
â”‚  â”‚   â””â”€ Concentration:
â”‚  â”‚       Guarantor concentration: Same guarantor on many credits
â”‚  â”‚       Increases systemic risk; regulators limit
â”‚  â”‚       Concentration charge: Additional capital required
â”‚  â””â”€ Risk:
â”‚      â”œâ”€ Guarantor Default:
â”‚      â”‚   Guarantor may not pay when due (moral hazard)
â”‚      â”‚   Weakening correlation: Borrower and guarantor distress related
â”‚      â”‚   Example: 2008: Corporate credit lines drawn when corporates weak
â”‚      â”œâ”€ Enforceability:
â”‚      â”‚   Guarantee may be unenforceable (legal challenge)
â”‚      â”‚   Jurisdiction risk: Different laws, rulings
â”‚      â”‚   Operational risk: Guarantor procedure may be complex
â”‚      â””â”€ Substitution Risk:
â”‚          Borrower changes to weaker guarantor
â”‚          Must maintain approval process
â”œâ”€ Netting:
â”‚  â”œâ”€ Bilateral Netting:
â”‚  â”‚   â”œâ”€ Definition:
â”‚  â”‚   â”‚   In event of counterparty default:
â”‚  â”‚   â”‚   Net amounts owed in both directions
â”‚  â”‚   â”‚   Only net amount exchanged
â”‚  â”‚   â”‚   Example: Bank A owes Hedge Fund $10M (on equity swap)
â”‚  â”‚   â”‚   Hedge Fund owes Bank A $15M (on interest rate swap)
â”‚  â”‚   â”‚   Net: Hedge Fund pays Bank A $5M (not $15M + offsetting $10M)
â”‚  â”‚   â”œâ”€ Exposure Reduction:
â”‚  â”‚   â”‚   Reduces EAD (replacement cost of contracts)
â”‚  â”‚   â”‚   Example: 10 derivative trades with same counterparty
â”‚  â”‚   â”‚   Before netting: Sum of mark-to-market positive values
â”‚  â”‚   â”‚   After netting: Net positive value only
â”‚  â”‚   â”‚   Reduction: 20-50% typical
â”‚  â”‚   â”œâ”€ Requirements:
â”‚  â”‚   â”‚   Master agreement (ISDA standard)
â”‚  â”‚   â”‚   Enforceable in counterparty's jurisdiction
â”‚  â”‚   â”‚   Automatic upon default (no counterparty consent)
â”‚  â”‚   â””â”€ Legal Risks:
â”‚  â”‚       Bankruptcy court may not honor netting
â”‚  â”‚       Specific jurisdictions block netting (e.g., some US states)
â”‚  â”‚       Operational: Multiple master agreements complicate
â”‚  â”œâ”€ Collateral-Based Netting:
â”‚  â”‚   â”œâ”€ CSA (Credit Support Annex):
â”‚  â”‚   â”‚   Collateral posted by counterparties
â”‚  â”‚   â”‚   Reduces replacement cost (EAD)
â”‚  â”‚   â”‚   Cash or securities accepted (typically)
â”‚  â”‚   â”‚   Threshold: No collateral if exposure below threshold
â”‚  â”‚   â”œâ”€ Mechanics:
â”‚  â”‚   â”‚   Counterparty A posts $10M when exposure exceeds threshold
â”‚  â”‚   â”‚   Mark-to-market daily
â”‚  â”‚   â”‚   If mark-to-market falls, collateral returned
â”‚  â”‚   â”‚   If rises, additional collateral posted
â”‚  â”‚   â”œâ”€ Benefits:
â”‚  â”‚   â”‚   EAD reduction: 50-80% typical
â”‚  â”‚   â”‚   Procyclical: More collateral posted when volatility rises (when needed)
â”‚  â”‚   â”‚   Cheaper than guarantee (no ongoing fee)
â”‚  â”‚   â””â”€ Risks:
â”‚  â”‚       Collateral value may fall (same market stress as exposure)
â”‚  â”‚       Haircuts must account for correlation
â”‚  â”‚       Operational: Daily reconciliation required
â”‚  â”œâ”€ Close-Out:
â”‚  â”‚   â”œâ”€ Definition:
â”‚  â”‚   â”‚   Upon counterparty default: Unwind all positions
â”‚  â”‚   â”‚   Close-out value = Mark-to-market as of default
â”‚  â”‚   â”‚   Netting applied; net amount exchanged
â”‚  â”‚   â”œâ”€ Timing:
â”‚  â”‚   â”‚   Should be prompt (hours to days)
â”‚  â”‚   â”‚   Market conditions change; delay increases risk
â”‚  â”‚   â”‚   Operational capability critical
â”‚  â”‚   â””â”€ Valuation:
â”‚  â”‚       Multiple quotes; independent validation
â”‚  â”‚       Use bid prices (conservative: cost to replace)
â”‚  â””â”€ Central Clearing:
â”‚      â”œâ”€ CCP (Central Counterparty):
â”‚      â”‚   Clearinghouse stands between counterparties
â”‚      â”‚   Standardized contracts (futures, cleared swaps)
â”‚      â”‚   Reduces bilateral counterparty risk
â”‚      â”œâ”€ Mechanics:
â”‚      â”‚   Counterparty A trades with Counterparty B
â”‚      â”‚   CCP novates: A trades with CCP, CCP trades with B
â”‚      â”‚   A's credit risk: CCP only (not B)
â”‚      â”‚   B's credit risk: CCP only (not A)
â”‚      â”œâ”€ Margin:
â”‚      â”‚   Initial margin: Posted upfront per CCP rules
â”‚      â”‚   Variation margin: Daily mark-to-market (paid both directions)
â”‚      â”‚   Guarantee fund: CCP fund covers CCP losses on member default
â”‚      â””â”€ Benefit:
â”‚          Counterparty risk eliminated (CCP is creditworthy)
â”‚          Netting across all participants (multilateral)
â”‚          EAD reduction: 60-90% vs bilateral
â”œâ”€ Securitization:
â”‚  â”œâ”€ Structure:
â”‚  â”‚   â”œâ”€ Originator:
â”‚  â”‚   â”‚   Bank/lender originates loans (mortgages, auto, credit card)
â”‚  â”‚   â”‚   Typically retains servicing rights
â”‚  â”‚   â”œâ”€ SPV (Special Purpose Vehicle):
â”‚  â”‚   â”‚   Bankruptcy-remote entity
â”‚  â”‚   â”‚   Owns loan pool
â”‚  â”‚   â”‚   Issues securities (MBS, ABS) backed by loans
â”‚  â”‚   â”œâ”€ Tranches:
â”‚  â”‚   â”‚   Senior tranche: Paid first, lowest risk, lowest yield (e.g., AAA)
â”‚  â”‚   â”‚   Mezzanine tranches: Paid after senior (e.g., A, BBB)
â”‚  â”‚   â”‚   Subordinated/Equity tranche: Paid last, highest risk, highest yield
â”‚  â”‚   â”‚   Waterfall: Losses absorbed in reverse order (equity first)
â”‚  â”‚   â””â”€ Investors:
â”‚  â”‚       Buy securities in public/private market
â”‚  â”‚       Receive cash flows (principal + interest)
â”‚      â”œâ”€ Capital Benefits:
â”‚  â”‚   â”œâ”€ Removal from Balance Sheet:
â”‚  â”‚   â”‚   Loans sold to SPV (true sale)
â”‚  â”‚   â”‚   Bank no longer holds credit risk
â”‚  â”‚   â”‚   Reduces RWA (risk-weighted assets)
â”‚  â”‚   â”‚   Capital relief: 10-50% (depends on securitization terms)
â”‚  â”‚   â”œâ”€ Reduction of Leverage:
â”‚  â”‚   â”‚   Loan $ no longer on balance sheet â†’ higher ROA
â”‚  â”‚   â”‚   Allows new lending (capacity increase)
â”‚  â”‚   â”‚   Cost: Securitization fees offset benefit
â”‚  â”‚   â””â”€ Return on Capital:
â”‚  â”‚       Origination fee: 0.5-2% (typically 1%)
â”‚  â”‚       Yield difference: Investor buys at discount â†’ bank realizes gain
â”‚  â”‚       Junior note: Bank may retain (and profit if portfolio performs)
â”‚  â”œâ”€ Credit Support:
â”‚  â”‚   â”œâ”€ Over-Collateralization:
â”‚  â”‚   â”‚   Loan pool value > Securitization value
â”‚  â”‚   â”‚   Example: $1B loans securitized as $900M securities
â”‚  â”‚   â”‚   $100M cushion (10%) covers losses
â”‚  â”‚   â”‚   Protects senior tranches
â”‚  â”‚   â”œâ”€ Reserves:
â”‚  â”‚   â”‚   Cash held for expected defaults
â”‚  â”‚   â”‚   Released if actual defaults < expected
â”‚  â”‚   â”œâ”€ Subordination:
â”‚  â”‚   â”‚   Subordinated tranches absorb losses first
â”‚  â”‚   â”‚   Senior tranches protected
â”‚  â”‚   â”‚   Tranche hierarchy crucial
â”‚  â”‚   â””â”€ Guarantees:
â”‚  â”‚       Originator may guarantee portfolio performance
â”‚  â”‚       Reduces credit risk but increases originator risk
â”‚  â”œâ”€ Risks:
â”‚  â”‚   â”œâ”€ Basis Risk:
â”‚  â”‚   â”‚   Portfolio performance diverges from expected
â”‚  â”‚   â”‚   Pool may have worse underwriting than average
â”‚  â”‚   â”‚   Originator incentive mismatch (originate then sell)
â”‚  â”‚   â”œâ”€ Liquidity Risk:
â”‚  â”‚   â”‚   Securities may not be tradeable in stress (2008)
â”‚  â”‚   â”‚   Investors forced to hold to maturity
â”‚  â”‚   â”‚   Loss of liquidity premium in yield
â”‚  â”‚   â”œâ”€ Model Risk:
â”‚  â”‚   â”‚   Rating models may underestimate tail risk
â”‚  â”‚   â”‚   Correlation assumptions fail in stress
â”‚  â”‚   â”‚   Systematic default correlation (recession)
â”‚  â”‚   â””â”€ Reputational:
â”‚  â”‚       Investor losses â†’ negative PR for originator
â”‚  â”‚       Affects future funding ability
â”‚  â”‚       Regulatory scrutiny
â”‚  â””â”€ Accounting:
â”‚      â”œâ”€ True Sale:
â”‚      â”‚   Loans transferred to SPV are no longer originator's asset
â”‚      â”‚   Off-balance-sheet treatment
â”‚      â”‚   Requires control transfer + recourse limits
â”‚      â”œâ”€ Consolidation:
â”‚      â”‚   If SPV not truly independent â†’ consolidation required
â”‚      â”‚   Assets/liabilities back on originator's balance sheet
â”‚      â”‚   Defeats capital relief purpose
â”‚      â””â”€ Impairment:
â”‚          Retained junior tranches marked-to-market
â”‚          Losses recorded if valuation falls
â”œâ”€ Credit Insurance and Derivatives:
â”‚  â”œâ”€ Credit Default Swap (CDS):
â”‚  â”‚   â”œâ”€ Structure:
â”‚  â”‚   â”‚   Protection buyer pays periodic fee (premium)
â”‚  â”‚   â”‚   Protection seller pays upon credit event
â”‚  â”‚   â”‚   Notional amount: $10M typical
â”‚  â”‚   â”œâ”€ Premium:
â”‚  â”‚   â”‚   Usually quoted in basis points (bps) p.a.
â”‚  â”‚   â”‚   Example: 100 bps = 1% annually = $100k per $10M
â”‚  â”‚   â”‚   Reflects PD Ã— LGD + risk premium
â”‚  â”‚   â”œâ”€ Payout:
â”‚  â”‚   â”‚   Triggered by: Bankruptcy, failure to pay, restructuring (per definition)
â”‚  â”‚   â”‚   Cash settlement: Par - market value recovered
â”‚  â”‚   â”‚   Example: $10M notional, par 100, market value 40 â†’ payout $600k
â”‚  â”‚   â”‚   Timing: 2-3 days after credit event (settlement lag)
â”‚  â”‚   â”œâ”€ Benefits:
â”‚  â”‚   â”‚   Removes credit risk without selling loan/bond
â”‚  â”‚   â”‚   Maintains client relationship (no visible sale)
â”‚  â”‚   â”‚   Tax efficient (no realization event)
â”‚  â”‚   â”œâ”€ Risks:
â”‚  â”‚   â”‚   Basis risk: Payoff may not exactly match exposure
â”‚  â”‚   â”‚   Counterparty risk: Protection seller may default
â”‚  â”‚   â”‚   Liquidity: CDS market may be illiquid (wide bid-ask)
â”‚  â”‚   â””â”€ Accounting:
â”‚  â”‚       Fair value hedge: P&L offsets in same period
â”‚  â”‚       Economic hedge, not necessarily accounting hedge
â”‚  â”œâ”€ Credit Insurance:
â”‚  â”‚   â”œâ”€ Monoline Insurers:
â”‚  â”‚   â”‚   Specialize in credit insurance (bond insurance, CDS)
â”‚  â”‚   â”‚   Rated AAA or AA typically
â”‚  â”‚   â”‚   Provide credit wrapper to lower-rated securities
â”‚  â”‚   â”œâ”€ Premium:
â”‚  â”‚   â”‚   Lower than CDS (perceived lower default by insurers)
â”‚  â”‚   â”‚   0.25-2% p.a. typical (depends on risk)
â”‚  â”‚   â”œâ”€ Trigger:
â”‚  â”‚   â”‚   Typically: Payment default (not restructuring/downgrade)
â”‚  â”‚   â”‚   Narrower than CDS trigger
â”‚  â”‚   â”œâ”€ Benefits:
â”‚  â”‚   â”‚   Lower cost than CDS
â”‚  â”‚   â”‚   Insurance rating benefit (rating uplift)
â”‚  â”‚   â”‚   May enable investment-grade rating
â”‚  â”‚   â””â”€ Risks:
â”‚  â”‚       Insurance may not be honored (2008: AIG near default)
â”‚  â”‚       Counterparty risk high (monoline concentrated on mortgages)
â”‚  â”‚       Regulatory issues (insurance law varies by jurisdiction)
â”‚  â”œâ”€ Payer Swaption / Receiver Swaption:
â”‚  â”‚   â”œâ”€ Definition:
â”‚  â”‚   â”‚   Option to enter interest rate swap
â”‚  â”‚   â”‚   Payer: Right to pay fixed, receive floating
â”‚  â”‚   â”‚   Receiver: Right to receive fixed, pay floating
â”‚  â”‚   â”œâ”€ Uses:
â”‚  â”‚   â”‚   Borrower fears rising rates â†’ payer swaption hedge
â”‚  â”‚   â”‚   If rates rise, exercise payer swap (lock in fixed payment)
â”‚  â”‚   â”‚   If rates fall, let expire (benefit from lower rates)
â”‚  â”‚   â”œâ”€ Cost:
â”‚  â”‚   â”‚   Option premium (upfront)
â”‚  â”‚   â”‚   Reflects volatility + strike + duration
â”‚  â”‚   â””â”€ Benefit:
â”‚  â”‚       Asymmetric: Upside uncapped, downside capped
â”‚  â””â”€ Equity Tranche / First Loss Position:
â”‚      â”œâ”€ Concept:
â”‚      â”‚   Bank retains highest loss position (first loss)
â”‚      â”‚   Shows confidence in underwriting
â”‚      â”‚   Credit enhancement (absorbs initial defaults)
â”‚      â”œâ”€ Example:
â”‚      â”‚   $1B loan pool: Bank retains $100M (10% equity)
â”‚      â”‚   First $100M of losses hit equity
â”‚      â”‚   Senior tranches ($900M) protected
â”‚      â”œâ”€ Incentive:
â”‚      â”‚   Aligns originator (keeps best underwriting)
â”‚      â”‚   Reduces moral hazard vs. full securitization
â”‚      â”œâ”€ Cost:
â”‚      â”‚   Bank capital tied up
â”‚      â”‚   Return needed to compensate
â”‚      â”‚   Leverage: $100M equity supports $1B credit
â”‚      â””â”€ Recovery:
â”‚          If realized losses < expected â†’ equity profits
â”‚          If losses > expected â†’ equity wiped out
â”œâ”€ Diversification and Portfolio Effects:
â”‚  â”œâ”€ Concentration Risk:
â”‚  â”‚   â”œâ”€ Definition:
â”‚  â”‚   â”‚   Exposure to limited number of counterparties/sectors
â”‚  â”‚   â”‚   Reduces benefit of diversification
â”‚  â”‚   â”‚   Increases tail risk
â”‚  â”‚   â”œâ”€ Measures:
â”‚  â”‚   â”‚   HHI (Herfindahl index): Î£ w_iÂ²
â”‚  â”‚   â”‚   If N equal exposures: HHI = 1/N
â”‚  â”‚   â”‚   HHI = 1: Perfect concentration; HHI = 1/N: Perfect diversification
â”‚  â”‚   â”œâ”€ Regulatory Limits:
â”‚  â”‚   â”‚   Large exposure: >10% of capital
â”‚  â”‚   â”‚   Aggregate large exposures: <800% of capital
â”‚  â”‚   â”‚   Related parties (same group): Lower limits
â”‚  â”‚   â””â”€ Portfolio Impact:
â”‚  â”‚       Concentration increases expected loss (portfolio level)
â”‚      â”œâ”€ Granularity:
â”‚  â”‚   â”œâ”€ Definition:
â”‚  â”‚   â”‚   Many small exposures vs. few large
â”‚  â”‚   â”‚   Granular portfolio: Lower granularity adjustment
â”‚  â”‚   â”‚   Concentrated portfolio: Higher adjustment needed
â”‚  â”‚   â”œâ”€ Loss Distribution:
â”‚  â”‚   â”‚   Concentrated: Losses lumpy (few large defaults)
â”‚  â”‚   â”‚   Granular: Losses smooth (many small defaults)
â”‚  â”‚   â”‚   Granularity adjustment: Factor applied to risk weight
â”‚  â”‚   â”‚   Example: 100 exposures of $1M each vs. 1 exposure $100M
â”‚  â”‚   â”‚        Granular portfolio benefits from averaging
â”‚  â”‚   â”‚        Concentrated portfolio needs buffer
â”‚  â”‚   â””â”€ Regulatory Recognition:
â”‚  â”‚       IRB approaches allow granularity adjustments
â”‚  â”‚       Standardized approach assumes granular
â”‚  â”‚       Granularity factor depends on EAD concentration
â”‚  â”œâ”€ Correlation & Systemic Risk:
â”‚  â”‚   â”œâ”€ Correlation Within Portfolio:
â”‚  â”‚   â”‚   Positive correlation: Defaults cluster (recession)
â”‚  â”‚   â”‚   Negative correlation: Offsetting defaults (rare)
â”‚  â”‚   â”‚   Procyclical: Correlation rises in stress (diversification fails)
â”‚  â”‚   â”œâ”€ Systemic Risk:
â”‚  â”‚   â”‚   Correlated defaults across portfolios
â”‚  â”‚   â”‚   Financial crisis â†’ all banks lose simultaneously
â”‚  â”‚   â”‚   Regulatory concern: Reduce systemic tail risk
â”‚  â”‚   â””â”€ Diversification Limits:
â”‚  â”‚       Benefit peaks around 50-100 exposures
â”‚  â”‚       Beyond: Diminishing returns (concentration on sectors/factors)
â”‚  â”‚       Cannot diversify away systematic risk (market/macro)
â”‚  â””â”€ Portfolio Stress Testing:
â”‚      â”œâ”€ Concentration Scenarios:
â”‚      â”‚   Large customer default: Impacts 10-20% portfolio
â”‚      â”‚   Sector downturn: Impacts 30-50% portfolio
â”‚      â”‚   Macro recession: Impacts 60%+ portfolio
â”‚      â”œâ”€ Combined Losses:
â”‚      â”‚   Large customer + sector downturn
â”‚      â”‚   Estimate potential losses
â”‚      â”œâ”€ Capital Adequacy:
â”‚      â”‚   After stress losses: Remaining capital > regulatory minimum?
â”‚      â”‚   If not: Reduce portfolio, raise capital
â”‚      â””â”€ Limits:
â”‚          Set position limits by counterparty
â”‚          Sector limits to reduce concentration
â”‚          Geographic limits (if applicable)
â””â”€ Operational and Legal Framework:
   â”œâ”€ Documentation:
   â”‚   â”œâ”€ Security Agreement:
   â”‚   â”‚   Legal document creating lien on collateral
   â”‚   â”‚   Recording/filing requirements (vary by jurisdiction)
   â”‚   â”‚   Perfection: Proper filing + notation
   â”‚   â”œâ”€ Guarantee Agreement:
   â”‚   â”‚   Guarantor's unconditional promise
   â”‚   â”‚   Defines scope, limitations, waivers
   â”‚   â”‚   Executed by guarantor (authorized officer/individual)
   â”‚   â”œâ”€ Master Agreement:
   â”‚   â”‚   ISDA for derivatives (netting, close-out)
   â”‚   â”‚   Creditor association agreements
   â”‚   â”‚   Standardized terms (facilitates enforcement)
   â”‚   â””â”€ Credit Support Annex:
   â”‚       Collateral posting terms
   â”‚       Mark-to-market, haircuts, thresholds
â”‚       Alternative: Collateral Pledge Agreement
â”‚   â”œâ”€ Counterparty Management:
â”‚   â”‚   â”œâ”€ Ongoing Monitoring:
â”‚   â”‚   â”‚   Credit rating updates
â”‚   â”‚   â”‚   Financial statements quarterly/annual
â”‚   â”‚   â”‚   Market data (CDS spreads, stock price)
â”‚   â”‚   â”‚   Covenant compliance (if applicable)
â”‚   â”‚   â”œâ”€ Triggers:
â”‚   â”‚   â”‚   Downgrade below threshold â†’ reduce limit
â”‚   â”‚   â”‚   Negative news â†’ reassess risk
â”‚   â”‚   â”‚   Covenant breach â†’ enforcement
â”‚   â”‚   â””â”€ Escalation:
â”‚   â”‚       Limit breach: Reduce exposure
â”‚   â”‚       Rating downgrade: Review mitigation
â”‚   â”‚       Default risk evident: Prepare workout
â”‚   â”œâ”€ Valuation and Haircuts:
â”‚   â”‚   â”œâ”€ Collateral Valuation:
â”‚   â”‚   â”‚   Real estate: Appraisal (annually or trigger event)
â”‚   â”‚   â”‚   Securities: Market prices (daily)
â”‚   â”‚   â”‚   Receivables: Aging analysis (monthly)
â”‚   â”‚   â”œâ”€ Haircut Recalibration:
â”‚   â”‚   â”‚   Quarterly or upon market stress
â”‚   â”‚   â”‚   Increase haircuts if volatility rises
â”‚   â”‚   â”‚   Example: Stock haircut 30% normal â†’ 50% during market stress
â”‚   â”‚   â””â”€ Model Risk:
â”‚   â”‚       Valuation models may not capture tail risk
â”‚   â”‚       Independent review; model governance
â”‚   â”œâ”€ Enforcement:
â”‚   â”‚   â”œâ”€ Judicial:
â”‚   â”‚   â”‚   Court proceedings (slow, expensive)
â”‚   â”‚   â”‚   Judgment enforcement (lien on other assets)
â”‚   â”‚   â”œâ”€ Non-Judicial:
â”‚   â”‚   â”‚   Self-help remedies: Setoff, liquidation
â”‚   â”‚   â”‚   Faster but limited in scope
â”‚   â”‚   â”œâ”€ Settlement:
â”‚   â”‚   â”‚   Negotiate with counterparty (cheaper, faster)
â”‚   â”‚   â”‚   Loan modification, payment plans
â”‚   â”‚   â””â”€ Cost:
â”‚   â”‚       Legal fees: 1-5% of claim
â”‚   â”‚       Enforcement delay: 6-24 months
â”‚   â”‚       Collection rate: 30-80% (varies greatly)
â”‚   â””â”€ Regulatory Framework:
â”‚       â”œâ”€ Capital Relief:
â”‚       â”‚   Mitigation recognized in regulatory capital
â”‚       â”‚   Specific rules: Collateral haircuts, guarantee criteria
â”‚       â”‚   Standardized vs. IRB approaches differ
â”‚       â”œâ”€ Concentration:
â”‚       â”‚   Large exposure limits (>10% capital)
â”‚       â”‚   Aggregate large exposures limits
â”‚       â”‚   Stress testing requirements
â”‚       â””â”€ Disclosure:
â”‚           Public disclosure of concentration
â”‚           Risk management disclosures (Pillar 3)
â”‚           Quality of mitigation disclosed
```

**Interaction:** Assess credit exposure â†’ Identify mitigation needs (concentration, tail risk) â†’ Structure mitigation (collateral terms, guarantee, netting, securitization) â†’ Negotiate terms â†’ Implement (documentation, enforcement capability) â†’ Monitor (collateral value, counterparty credit, covenant compliance) â†’ Adjust (margin calls, limit breaches) â†’ On default: Execute (liquidate collateral, claim on guarantor, netting) â†’ Recover losses.

## 5. Challenge Round
1. **Optimal Mitigation Portfolio:** Given 100 credits (PDs, LGDs, sizes), determine optimal mitigation mix (collateral %, guarantees, securitization %) to minimize total cost (mitigation fees + capital charge) subject to regulatory constraints. Use linear/quadratic programming.

2. **Collateral Haircut Stress:** Model collateral value decline across asset classes (equities -50%, real estate -30%, bonds -10%) during market stress. Calculate LGD changes, margin calls, capital implications. When do mark-to-market haircuts dominate?

3. **Correlation in Guarantees:** Simulate portfolio where guarantor PD correlates with borrower PD (during recession both distressed). Estimate correlation impact on guarantee benefit. Compare standalone PD reduction vs. correlated scenario.

4. **Securitization Basis Risk:** Pool contains 500 mortgages with PDs and LGDs. Simulate defaults, losses. Compare expected losses to securitized tranches' expected payouts. Does mezzanine tranche premium cover tail risk adequately?

5. **CDS-Collateral Redundancy:** Hedge credit risk using both CDS and collateral. Optimize allocation to minimize joint costs. When is CDS preferred over collateral? (CDS liquid, collateral less costly if good assets available)

## 6. Key References
- [Basel Committee, "International Convergence of Capital Measurement and Capital Standards" (2006/2017, Basel III)](https://www.bis.org/bcbs/basel3.htm) - regulatory framework for credit risk mitigation recognition
- [Altman & Saunders, "Credit Risk Measurement: Developments Over the Last 20 Years" (2001)](https://www.jstor.org/stable/2673960) - evolution of credit risk models and mitigation techniques
- [Hull & White, "The Impact of Default Risk on the Prices of Options and Currency Swaps" (1995)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1003265) - credit risk in derivative pricing and mitigation

---
**Status:** Operational risk management and capital optimization | **Complements:** Credit Risk Fundamentals, Expected Loss Calculation, Credit Derivatives, Regulatory Capital, Portfolio Management
