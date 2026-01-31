# Actuarial Statistics Topics Guide

**Complete reference of foundational and advanced actuarial statistics concepts with categories, brief descriptions, and sources.**

---

## I. Life Contingencies & Mortality

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Force of Mortality** | N/A | Instantaneous death rate μₓ; continuous hazard function | [SOA - Actuarial Mathematics](https://www.soa.org/) |
| **Survival Probability** | N/A | ₚₓ = P(lives to age x+p); age-dependent survival curves | [Wiki - Survival Function](https://en.wikipedia.org/wiki/Survival_function) |
| **Life Expectancy** | N/A | Expected remaining years at age x; measure of longevity | [WHO - Life Expectancy](https://www.who.int/) |
| **Mortality Tables** | N/A | ₚₓ and qₓ by age; empirical probabilities from population data | [CMI - Mortality Data](https://www.cmi.ac.uk/) |
| **Gompertz Law** | N/A | μₓ = A·B^x; exponential mortality increase with age | [Wiki - Gompertz](https://en.wikipedia.org/wiki/Gompertz%E2%80%93Makeham_law_of_mortality) |
| **Makeham Law** | N/A | μₓ = A + B·C^x; adds constant to Gompertz for accident rates | [Wiki - Makeham](https://en.wikipedia.org/wiki/Gompertz%E2%80%93Makeham_law_of_mortality) |

---

## II. Life Insurance Valuation

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Present Value of Benefits** | N/A | Aₓ = E[Z]; discounted expected death payouts | [SOA - Life Insurance Mathematics](https://www.soa.org/) |
| **Term Insurance (Aₓ:n̄|)** | N/A | Fixed benefit if death within n years; pure risk component | [Wiki - Term Life Insurance](https://en.wikipedia.org/wiki/Term_life_insurance) |
| **Whole Life Insurance (Āₓ)** | N/A | Benefit regardless of death age; perpetual contract | [Wiki - Whole Life Insurance](https://en.wikipedia.org/wiki/Whole_life_insurance) |
| **Endowment Insurance** | N/A | Benefit if death or reach age n; hybrid savings+insurance | [Wiki - Endowment Insurance](https://en.wikipedia.org/wiki/Endowment_insurance) |
| **Deferred Insurance (ₙ\|Āₓ)** | N/A | Benefit commences after n years; delayed coverage | [SOA - Actuarial Mathematics](https://www.soa.org/) |
| **Insurance with Varying Benefits** | N/A | Increasing/decreasing payments; scaled by time or age | [SOA - Actuarial Mathematics](https://www.soa.org/) |

---

## III. Annuities

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Immediate Annuity (aₓ)** | N/A | Payments at period end; deferred immediate annuity | [SOA - Annuity Mathematics](https://www.soa.org/) |
| **Annuity Due (äₓ)** | N/A | Payments at period start; one-period higher value | [Wiki - Annuity Due](https://en.wikipedia.org/wiki/Annuity_(finance)#Annuity_due) |
| **Life Annuity (aₓ̄)** | N/A | Payments until death; lifetime income stream | [Wiki - Life Annuity](https://en.wikipedia.org/wiki/Annuity_(finance)#Lifetime_income_annuity) |
| **Term Annuity (aₓ:n̄|)** | N/A | Payments for fixed period or life, whichever shorter | [SOA - Annuity Mathematics](https://www.soa.org/) |
| **Annuity with Guarantees** | N/A | Guaranteed period + life payments; minimum years assured | [Wiki - Guaranteed Period](https://en.wikipedia.org/wiki/Annuity_(finance)#Guaranteed_period) |
| **Deferred Annuity (ₙ|aₓ)** | N/A | Payments begin after n years; retirement-focused | [SOA - Deferred Annuities](https://www.soa.org/) |

---

## IV. Premium Calculation (Net & Gross)

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Net Premium** | N/A | Pure actuarial cost; present value of benefits = present value of premiums | [SOA - Premium Principles](https://www.soa.org/) |
| **Principle of Equivalence** | N/A | Net premium: E[Benefits] = E[Premiums]; zero expected profit | [Wiki - Equivalence Principle](https://en.wikipedia.org/wiki/Actuarial_present_value) |
| **Loading/Expense Margin** | N/A | Additional premium for expenses and profit; Gross Premium = Net + Loading | [SOA - Pricing Fundamentals](https://www.soa.org/) |
| **Premium Reserves** | N/A | Liability on balance sheet; accumulation of net premiums - benefits paid | [Wiki - Actuarial Reserve](https://en.wikipedia.org/wiki/Actuarial_reserve) |
| **Renewal Expenses** | N/A | Ongoing costs; commissions, administration, taxes | [SOA - Expense Analysis](https://www.soa.org/) |
| **Profit Testing** | N/A | Project cash flows, surplus, return on equity; strategic pricing | [Wiki - Profit Testing](https://en.wikipedia.org/wiki/Profit_testing) |

---

## V. Reserves & Liabilities

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Net Premium Reserve (V)** | N/A | Prospective reserve = PV(future benefits) - PV(future premiums) | [SOA - Reserve Calculation](https://www.soa.org/) |
| **Retrospective Reserve** | N/A | Accumulated premiums + interest - benefits paid; backward-looking | [SOA - Reserve Methods](https://www.soa.org/) |
| **Modified Reserve** | N/A | Reduced first-year reserve; recognizes high initial costs | [SOA - Reserve Methods](https://www.soa.org/) |
| **Inactive Reserves** | N/A | For policies with reduced benefits (surrender, lapse potential) | [SOA - Embedded Options](https://www.soa.org/) |
| **Deficiency Reserves** | N/A | Additional reserve if gross premium < net premium requirement | [SOA - Statutory Reserve](https://www.soa.org/) |
| **Statutory vs Actuarial Reserves** | N/A | Regulatory minimum vs optimal actuarial estimate; conservatism tradeoff | [NAIC - Valuation Manual](https://www.naic.org/) |

---

## VI. Interest Rate & Annuity Functions

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Effective Annual Rate (i)** | N/A | Compound interest; (1+i) = annual growth factor | [Wiki - Interest Rate](https://en.wikipedia.org/wiki/Interest_rate) |
| **Nominal Rate (i^(m))** | N/A | Convertible m times per year; i^(m)/m applied per period | [Wiki - Nominal Interest Rate](https://en.wikipedia.org/wiki/Nominal_interest_rate) |
| **Force of Interest (δ)** | N/A | Continuous compounding rate; ln(1+i) = δ | [Wiki - Force of Interest](https://en.wikipedia.org/wiki/Force_of_interest) |
| **Discount Factor (v)** | N/A | v = 1/(1+i); present value multiplier | [Wiki - Discount Factor](https://en.wikipedia.org/wiki/Discount_factor) |
| **Accumulation Factor** | N/A | (1+i)^n; future value multiplier for principal | [Wiki - Compound Interest](https://en.wikipedia.org/wiki/Compound_interest) |
| **Annuity-Certain Functions** | N/A | aₙ̄| (present), sₙ̄| (future); fixed-term payments | [SOA - Annuity Functions](https://www.soa.org/) |

---

## VII. Multiple Decrement Models

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Multiple Decrement Probabilities** | N/A | qₓ^(j) = probability of cause j during year; sum to total qₓ | [SOA - Multiple Decrements](https://www.soa.org/) |
| **Cause-Specific Mortality** | N/A | Separate rates for death, lapse, withdrawal; competing risks | [Wiki - Competing Risks](https://en.wikipedia.org/wiki/Competing_risks) |
| **Associated Single Decrement** | N/A | Hypothetical qₓ' if only one decrement acts; useful for analysis | [SOA - Decrement Models](https://www.soa.org/) |
| **Lapse Rates** | N/A | Policy termination probability; ψₓ; depends on price, service, market | [SOA - Lapse Assumptions](https://www.soa.org/) |
| **Surrender Assumptions** | N/A | Cash value withdrawal option; reflects policyholder behavior | [SOA - Surrender Analysis](https://www.soa.org/) |
| **Service Tables** | N/A | Combined active/inactive decrements; pension/group insurance focus | [SOA - Service Tables](https://www.soa.org/) |

---

## VIII. Pension Mathematics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Defined Benefit Plans** | N/A | Employer guarantees benefit formula (e.g., % of salary); actuarial liability | [Wiki - Defined Benefit](https://en.wikipedia.org/wiki/Defined_benefit_plan) |
| **Defined Contribution Plans** | N/A | Employer contributes fixed amount; member bears investment risk | [Wiki - Defined Contribution](https://en.wikipedia.org/wiki/Defined_contribution_plan) |
| **Accrued Benefit Obligation (ABO)** | N/A | Liability for benefits earned to date; frozen salary assumption | [FASB - ABO](https://www.fasb.org/) |
| **Projected Benefit Obligation (PBO)** | N/A | Liability including future salary growth; includes pay progression | [FASB - PBO](https://www.fasb.org/) |
| **Pension Funding** | N/A | Contribution strategy to meet obligations; actuarial cost methods | [SOA - Pension Funding](https://www.soa.org/) |
| **Actuarial Cost Methods** | N/A | Entry Age Normal, Unit Credit, Frozen Attained Age; cost allocation | [SOA - Cost Methods](https://www.soa.org/) |

---

## IX. Survival Analysis (Actuarial Application)

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Kaplan-Meier Estimator** | N/A | Non-parametric survival curve; empirical ₚₓ calculation | [Wiki - Kaplan-Meier](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) |
| **Cox Proportional Hazards** | N/A | Semi-parametric; hazard ratios for risk factors (age, health, smoking) | [Wiki - Cox Model](https://en.wikipedia.org/wiki/Proportional_hazards_model) |
| **Nelson-Aalen Estimator** | N/A | Cumulative hazard; alternative to Kaplan-Meier for grouped data | [Wiki - Nelson-Aalen](https://en.wikipedia.org/wiki/Nelson%E2%80%93Aalen_estimator) |
| **Graduation of Mortality Data** | N/A | Smooth ₚₓ from empirical; reduce random fluctuation; parametric fit | [SOA - Graduation Methods](https://www.soa.org/) |
| **Standardized Mortality Ratio (SMR)** | N/A | Observed/Expected deaths; compares to reference population | [Wiki - SMR](https://en.wikipedia.org/wiki/Standardized_mortality_ratio) |
| **Life Table Construction** | N/A | From raw deaths to age-specific rates; basis for pricing and reserving | [Wiki - Life Table](https://en.wikipedia.org/wiki/Life_table) |

---

## X. Risk Management & Solvency

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Longevity Risk** | N/A | Population lives longer than assumed; reserve/profitability impact | [SOA - Longevity Risk](https://www.soa.org/) |
| **Mortality Risk** | N/A | Actual deaths exceed expected; immediate loss on underwriting | [SOA - Mortality Risk](https://www.soa.org/) |
| **Lapse Risk** | N/A | Policyholders terminate unexpectedly; breaks up profit stream | [SOA - Lapse Risk](https://www.soa.org/) |
| **Interest Rate Risk** | N/A | Asset-liability mismatch; reinvestment and market risk | [Wiki - Interest Rate Risk](https://en.wikipedia.org/wiki/Interest_rate_risk) |
| **Stochastic Modeling** | N/A | Simulation of future scenarios; tail risks, VaR, expected shortfall | [SOA - Stochastic Methods](https://www.soa.org/) |
| **Solvency & Capital Requirements** | N/A | Regulatory capital (RBC); minimum reserves to withstand shocks | [NAIC - RBC Standards](https://www.naic.org/) |

---

## XI. Population Dynamics & Projections

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Population Growth Models** | N/A | Exponential, logistic, Leslie matrix; demographic projections | [Wiki - Population Model](https://en.wikipedia.org/wiki/Population_model) |
| **Fertility Rates** | N/A | Age-specific births (ASFR), Total Fertility Rate (TFR); natality | [Wiki - Fertility Rate](https://en.wikipedia.org/wiki/Fertility_rate) |
| **Migration** | N/A | In-migration, out-migration; affects population age structure | [Wiki - Migration](https://en.wikipedia.org/wiki/Human_migration) |
| **Age Structure** | N/A | Dependency ratio, median age; economic implications for pensions | [Wiki - Age Structure](https://en.wikipedia.org/wiki/Population_pyramid) |
| **Life Table Stationary Population** | N/A | Stable population assuming constant rates; theoretical benchmark | [Wiki - Stationary Population](https://en.wikipedia.org/wiki/Stationary_population) |
| **Projection Methodologies** | N/A | Component method, cohort-component; national or sub-group forecasts | [UN - Population Projections](https://population.un.org/) |

---

## XII. Disability & Health Insurance

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Disability Rates (dₓ)** | N/A | Probability of disability during year; varies by age, occupation | [SOA - Disability Rates](https://www.soa.org/) |
| **Recovery Rates** | N/A | Probability returning to work; depends on disability duration and type | [SOA - Recovery Assumptions](https://www.soa.org/) |
| **Disability Income Insurance** | N/A | Benefit replaces lost earnings; elimination period, benefit period | [Wiki - DI Insurance](https://en.wikipedia.org/wiki/Disability_insurance) |
| **Critical Illness Insurance** | N/A | Lump sum on diagnosis; cancer, heart attack, stroke, etc. | [Wiki - Critical Illness](https://en.wikipedia.org/wiki/Critical_illness_insurance) |
| **Long-Term Care** | N/A | Nursing, assisted living costs; duration and care level dependent | [SOA - LTC Costs](https://www.soa.org/) |
| **Morbidity & Claim Rates** | N/A | Medical event incidence; varies by population characteristics | [SOA - Morbidity Analysis](https://www.soa.org/) |

---

## XIII. Reinsurance & Risk Transfer

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Proportional Reinsurance** | N/A | Pro-rata loss sharing; quota share, surplus share | [Wiki - Proportional Reinsurance](https://en.wikipedia.org/wiki/Reinsurance) |
| **Non-Proportional Reinsurance** | N/A | Excess-of-loss; insurer retains layer, reinsurer above threshold | [Wiki - Excess of Loss](https://en.wikipedia.org/wiki/Reinsurance#Types) |
| **Retrocession** | N/A | Reinsurance of reinsurance; risk transfer chain | [Wiki - Retrocession](https://en.wikipedia.org/wiki/Retrocession) |
| **Commutation & Retroactive Covers** | N/A | Settlement of claims, coverage of past periods | [SOA - Reinsurance Terms](https://www.soa.org/) |
| **Catastrophe Bonds** | N/A | Securitization of extreme-tail risk; capital market instruments | [Wiki - Catastrophe Bond](https://en.wikipedia.org/wiki/Catastrophe_bond) |
| **Adverse Selection & Moral Hazard** | N/A | Information asymmetry, behavioral risks; underwriting controls | [Wiki - Adverse Selection](https://en.wikipedia.org/wiki/Adverse_selection) |

---

## XIV. Assumptions & Valuation Methods

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Mortality Assumptions** | N/A | ₚₓ from table, adjustments for population/year; assumption setting | [SOA - Mortality Assumptions](https://www.soa.org/) |
| **Interest Rate Assumptions** | N/A | Forward rates, yield curve; economic, market, long-term | [SOA - Interest Assumptions](https://www.soa.org/) |
| **Lapse & Surrender Assumptions** | N/A | Dynamic vs static; market value, pricing, regulatory basis | [SOA - Lapse Assumptions](https://www.soa.org/) |
| **Expense Assumptions** | N/A | Maintenance, commissions, acquisition; per-policy and percent-premium | [SOA - Expense Assumptions](https://www.soa.org/) |
| **Valuation Interest Rate** | N/A | Discount rate for liabilities; risk-free + spread or market consistent | [SOA - Valuation Rates](https://www.soa.org/) |
| **Profit Margin & Best Estimate** | N/A | Conservative vs best estimate; Solvency II, IFRS 17 frameworks | [EIOPA - IFRS 17](https://www.eiopa.europa.eu/) |

---

## XV. Regulatory & Accounting Standards

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Statutory Accounting** | N/A | NAIC SAP; insurance-specific rules, conservative reserving | [NAIC - SAP Handbook](https://www.naic.org/) |
| **GAAP (US Generally Accepted)** | N/A | ASC 944 (Insurance); earnings measurement, liability valuation | [FASB - ASC 944](https://www.fasb.org/) |
| **IFRS 17 (Insurance Contracts)** | N/A | Global standard; building blocks model, contract service margin | [IFRS - IFRS 17](https://www.ifrs.org/) |
| **Solvency II (EU)** | N/A | Quantitative requirements, risk-based capital, Own Risk Solvency Assessment | [EIOPA - Solvency II](https://www.eiopa.europa.eu/) |
| **Risk-Based Capital (RBC)** | N/A | NAIC formula; C1-C4 risk charges, minimum solvency ratios | [NAIC - RBC](https://www.naic.org/) |
| **Embedded Value & MCEV** | N/A | Intrinsic value of policies; shareholder value measure | [CEIOPS - EV Guidance](https://www.eiopa.europa.eu/) |

---

## XVI. Stochastic Methods & Modeling

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Stochastic Interest Rates** | N/A | Vasicek, CIR, Hull-White models; tree/Monte Carlo simulation | [Wiki - Interest Rate Models](https://en.wikipedia.org/wiki/Interest_rate_model) |
| **Equity Risk Models** | N/A | Lognormal, jump-diffusion; Black-Scholes framework | [Wiki - Black-Scholes](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) |
| **Longevity Risk Modeling** | N/A | Lee-Carter, Cairns-Blake-Dowd; stochastic mortality improvement | [SOA - Longevity Modeling](https://www.soa.org/) |
| **Correlation & Dependence** | N/A | Copulas for joint distributions; tail dependence | [Wiki - Copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)) |
| **Monte Carlo Simulation** | N/A | Path generation; confidence intervals for tail risks | [Wiki - Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) |
| **Scenario Analysis** | N/A | Deterministic shocks; stress testing for regulatory requirements | [SOA - Scenario Analysis](https://www.soa.org/) |

---

## XVII. Group Insurance & Employee Benefits

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Group Life Insurance** | N/A | Employer-sponsored; simplified underwriting, administrative pooling | [Wiki - Group Life](https://en.wikipedia.org/wiki/Group_life_insurance) |
| **Short-Term Disability (STD)** | N/A | Replacement during illness/injury; typically 13-26 weeks | [Wiki - Short-Term Disability](https://en.wikipedia.org/wiki/Disability_insurance) |
| **Long-Term Disability (LTD)** | N/A | Extended benefit; occupational definitions, offset provisions | [SOA - LTD Pricing](https://www.soa.org/) |
| **Medical/Health Insurance** | N/A | Fee-for-service, HMO, PPO; claim frequency, severity, trend | [Wiki - Health Insurance](https://en.wikipedia.org/wiki/Health_insurance) |
| **Workers' Compensation** | N/A | Mandated coverage; occupational injury/illness; indemnity + medical | [Wiki - Workers Compensation](https://en.wikipedia.org/wiki/Workers%27_compensation) |
| **Retirement Plans** | N/A | 401(k), pension, RRSP; accumulation and distribution phases | [Wiki - Retirement Plan](https://en.wikipedia.org/wiki/Retirement_savings_account) |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **Society of Actuaries (SOA)** | https://www.soa.org/ | Exams, research, standards for life, health, pensions |
| **Casualty Actuarial Society (CAS)** | https://www.casact.org/ | Property/casualty, reinsurance, ratemaking |
| **American Academy of Actuaries** | https://www.actuary.org/ | Standards of practice, guidance for US actuaries |
| **NAIC Valuation Manual** | https://www.naic.org/ | Statutory reserving, RBC, regulatory guidance |
| **International Actuarial Association** | https://www.actuaries.org/ | Global standards, professional guidance |
| **Bowers et al., Actuarial Mathematics** | Textbook | Comprehensive reference; life insurance, annuities, pensions |
| **Wikipedia - Actuarial Science** | https://en.wikipedia.org/wiki/Actuarial_science | Overview and cross-links to specific topics |

---

## Quick Stats

- **Total Topics Documented**: 90+
- **Main Categories**: 17
- **Sub-Topics**: Mortality, Insurance Valuation, Annuities, Pensions, Risk Management
- **Coverage**: Life Insurance → Pensions → Health/Disability → Reinsurance → Regulation
- **Focus Areas**: Actuarial Mathematics, Valuation, Risk, Regulatory Compliance

---

## Key Distinctions from General Statistics

1. **Survival-Centric**: Actuarial science focuses intensely on mortality/longevity probabilities (qₓ, ₚₓ)
2. **Time-Value of Money**: Present value and discount factors integral (not peripheral)
3. **Regulatory Compliance**: Statutory reserves, capital requirements, accounting standards (Solvency II, IFRS 17)
4. **Continuous Operations**: Long-duration contracts require sophisticated multi-year projections
5. **Risk Transfer**: Reinsurance, securitization, embedded options (surrender, lapse) critical
6. **Stochastic Necessity**: Economic uncertainty demands scenario analysis and Monte Carlo methods
7. **Professional Discipline**: Actuarial profession governed by standards of practice and ethics
