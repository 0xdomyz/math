# Group Life Insurance

## 1. Concept Skeleton
**Definition:** Employer-sponsored life insurance covering employees; single master policy with simplified individual underwriting; pooled administrative costs  
**Purpose:** Employee benefit; financial security for dependents; employer retention tool; lower premiums than individual policies  
**Prerequisites:** Group underwriting principles, mortality assumptions, employee turnover, administrative cost structures

## 2. Comparative Framing
| Feature | Group Life | Individual Life | Guaranteed Issue | Voluntary | 
|---------|-----------|-----------------|------------------|-----------|
| **Underwriting** | Minimal; employer only | Full medical exam | None; automatic | Limited |
| **Cost** | 30-50% lower per $1K | Higher; risk selection | Very high premium | Medium |
| **Coverage** | Standardized; multiple of salary | Customized | Fixed amount | Employee choice |
| **Portability** | Limited (COBRA 18-36 months) | Portable immediately | None after job loss | Portable by employee |
| **Employer Role** | Pays 50-80% typically | N/A; individual | N/A | Offers plan; employee pays |

## 3. Examples + Counterexamples

**Simple Example:**  
Company with 500 employees; 95% participation; average salary $50K; 1× salary coverage = $25M total benefit; annual cost $180/employee = $90K/year (~$3.6 per $1K benefit)

**Failure Case:**  
Employer discontinues group coverage without providing conversion option; 200 employees lose coverage immediately; 80 cannot qualify for individual policies due to health issues; lawsuits + regulatory penalties

**Edge Case:**  
Small business (12 employees); 1 high-wage executive ($300K); group rates based on average earnings ($40K); executive gets favorable rate due to pooling; cross-subsidy from lower-wage workers

## 4. Layer Breakdown
```
Group Life Insurance Structure:
├─ Master Policy Framework:
│   ├─ Employer Sponsor:
│   │   ├─ Master policy holder (party to contract with insurer)
│   │   ├─ Responsible for: Premium payment, enrollment administration
│   │   ├─ Can modify: Benefit levels, eligibility, plan design
│   │   ├─ Employee notification: Material changes require 30-day notice
│   │   └─ Fiduciary duty: ERISA requirement (private sector); ensure prudent administration
│   ├─ Eligible Employees:
│   │   ├─ Full-time definition: Typically ≥30 hours/week (PPACA standard)
│   │   ├─ Waiting period: Often 30-90 days (reduces adverse selection)
│   │   ├─ Eligibility classes:
│   │   │   ├─ Class 1: All full-time employees (most common)
│   │   │   ├─ Class 2: Union vs non-union (different benefits)
│   │   │   ├─ Class 3: By job classification (office vs field)
│   │   │   └─ Class 4: By salary band (higher salary → higher multiple)
│   │   └─ Participation requirement: Typically ≥75% participation for group rating
│   ├─ Participation Levels:
│   │   ├─ Fully insured: Employer contracts with insurer; insurer bears all risk
│   │   │   ├─ Premium fixed for 1-year period
│   │   │   ├─ Experience refund: If claims < premium, employer receives rebate
│   │   │   ├─ Stop-loss: Limits employer liability (e.g., if total claims > 105% premium)
│   │   │   └─ Transparency: Insurer keeps mortality risk; employer pays fixed
│   │   ├─ Self-funded (self-insured):
│   │   │   ├─ Employer retains risk; pays claims directly
│   │   │   ├─ Large employers only: Need size for predictable claims
│   │   │   ├─ Insurance purchased: Stop-loss coverage above threshold (e.g., $500K)
│   │   │   ├─ Savings: No insurer profit margin, administrative efficiency
│   │   │   └─ Risk: Large claim can exceed budget; financial volatility
│   │   └─ Minimum premium (hybrid):
│   │       ├─ Employer pays: Minimum fixed fee + % of claims
│   │       ├─ Insurer front-loads: Funds to cover expected claims
│   │       ├─ Balance: Shared risk model
│   │       └─ Use: Medium-sized employers (500-2,000 employees)
│   └─ Group Policy Terms:
│       ├─ Plan year: Typically calendar or fiscal year
│       ├─ Rate period: Usually 1 year (annual renewal)
│       ├─ Renewal options: Guaranteed renewable vs non-guaranteed
│       ├─ Rate guarantees: Fixed for 1-3 years (competitive advantage)
│       └─ Cancellation: Employer can terminate; insurer can non-renew for cause
├─ Benefit Design:
│   ├─ Coverage Amounts:
│   │   ├─ Standard structures:
│   │   │   ├─ 1× salary: Death benefit = employee's annual salary
│   │   │   ├─ 2× salary: Common in competitive markets
│   │   │   ├─ 3× salary: Generous plans (finance, tech companies)
│   │   │   ├─ Fixed amount: Flat $50K or $100K (simpler administration)
│   │   │   └─ Tiered: $100K (employees) + $50K (retirees) + $250K (executives)
│   │   ├─ Cap structures:
│   │   │   ├─ Annual salary cap: Max benefit $300K even if salary higher
│   │   │   ├─ Total cap: Max $1M per person (risk control)
│   │   │   └─ Offset reduction: Benefit reduced if <age 25 or >age 70
│   │   ├─ Underwriting classes (for premium rating):
│   │   │   ├─ Standard: Standard risk; uses industry mortality table (100% of table)
│   │   │   ├─ Preferred: Lower risk; better health/habits → 75-90% of table
│   │   │   ├─ Substandard: Higher risk; medical conditions → 125-150% of table
│   │   │   └─ Occupational rating: Hazardous jobs → 150-200% of table
│   │   └─ Example calculation:
│   │       ├─ Employee: Age 45, salary $60K, class standard
│   │       ├─ Benefit: 2× salary = $120K
│   │       ├─ Annual mortality rate (age 45, standard): 0.25% (from table)
│   │       ├─ APV per life: 0.0025 × $120K = $300 annually (ignoring discounts)
│   │       ├─ 500 employees, 95% participation: 475 covered
│   │       ├─ Total annual cost: $300 × 475 = $142,500
│   │       ├─ Plus admin: +$25K (~5% load) = $167,500 total premium
│   │       ├─ Per-employee monthly: $167,500 / (475 × 12) = $29.40/month
│   │       └─ Employer pays 75% = $22/month; employee pays 25% = $7/month
│   ├─ Eligibility & Enrollment:
│   │   ├─ Waiting period:
│   │   │   ├─ Purpose: Reduce adverse selection (healthy enrolls immediately, sick waits)
│   │   │   ├─ Typical length: 30, 60, or 90 days from hire
│   │   │   ├─ During waiting: No coverage; may accrue future coverage
│   │   │   └─ Example: Hired Jan 1; waiting 60 days; coverage begins Mar 1
│   │   ├─ Automatic enrollment (ACA requirement):
│   │   │   ├─ Default: Eligible employees automatically enrolled unless decline
│   │   │   ├─ Opt-out option: Employee can choose not to participate
│   │   │   ├─ Notice: Employer must explain benefit and waiver process
│   │   │   └─ Advantage: Higher participation; younger employees default-enroll
│   │   ├─ Life events triggering enrollment:
│   │   │   ├─ Hire date: Open enrollment (but waiting period may apply)
│   │   │   ├─ Marriage: Spouse coverage if plan allows
│   │   │   ├─ Birth/adoption: Child coverage triggered
│   │   │   ├─ Divorce: Coverage terminates for ex-spouse
│   │   │   ├─ Loss of other coverage: Allows mid-year enrollment
│   │   │   └─ COBRA qualifying event: Job loss → 18-36 months continuation
│   │   └─ Underwriting at enrollment:
│   │       ├─ Guaranteed issue: First $50K automatic (no medical questions)
│   │       ├─ Above guaranteed: Medical underwriting may apply
│   │       ├─ Medical history: Employer asks basic health questions
│   │       ├─ Flat extra: If pre-existing condition, add fixed charge per $1K
│   │       └─ Decline: Rarely done; group underwriting more permissive
│   ├─ Additional Riders & Options:
│   │   ├─ Accidental death benefit (AD&D):
│   │   │   ├─ Pays additional benefit if death is accidental (not illness/suicide)
│   │   │   ├─ Common: 100% additional (double indemnity) or 50% additional
│   │   │   ├─ Example: $120K base + $120K AD&D = $240K if accidental death
│   │   │   ├─ Cost: Minimal (5-10% of base premium)
│   │   │   └─ Common in high-risk industries
│   │   ├─ Dependent coverage:
│   │   │   ├─ Spouse coverage: Typically 50% of employee benefit (max $100K)
│   │   │   ├─ Child coverage: Fixed amounts; $5K-$25K per child
│   │   │   ├─ Cost: Usually employer pays
│   │   │   └─ Purpose: Financial security for family
│   │   ├─ Supplemental (voluntary) life:
│   │   │   ├─ Employee-paid additional coverage beyond employer plan
│   │   │   ├─ Common: 1-5× salary additional (up to $500K)
│   │   │   ├─ Underwriting: Limited (medical questions) or guaranteed issue
│   │   │   ├─ Guarantee to age: Typically age 70
│   │   │   └─ Portability: Often portable after job loss
│   │   ├─ Life event coverage:
│   │   │   ├─ Coverage at retirement: Plan ends at age 65-70 usually
│   │   │   ├─ Conversion option: Convert group to individual policy (no medical exam)
│   │   │   ├─ Individual rates: Higher than group (20-40% premium increase)
│   │   │   └─ Portability window: Typically 30-60 days post-termination
│   │   └─ Waiver of premium (disability):
│   │       ├─ If employee disabled; premiums continue without employee payment
│   │       ├─ Triggers: Typically LTD qualification or 6-month disability period
│   │       ├─ Waived: Employer covers waived employee premiums
│   │       └─ Cost: Minimal add-on to base premium
│   └─ Benefit Reductions & Exclusions:
│       ├─ Age-based reductions:
│       │   ├─ At age 65: Benefit often reduces to 50% (retirement assumption)
│       │   ├─ At age 70: Benefit may reduce to 25% (coverage wind-down)
│       │   ├─ Purpose: Reduce employer cost for older workers; reflect lower need
│       │   └─ Example: $120K at age 40 → $60K at age 65 → $30K at age 70
│       ├─ Exclusions (typical):
│       │   ├─ Suicide within first 2 years: No benefit (anti-selection)
│       │   ├─ Death while committing crime: Exclusion (moral hazard)
│       │   ├─ War/civil unrest: Often excluded (catastrophe control)
│       │   ├─ High-risk activity: Skydiving, mountaineering may exclude
│       │   └─ Beneficiary designation: Overrides legal heirs if named
│       └─ Coordination of benefits:
│           ├─ With other group coverage: Multiple employers (rare; one primary)
│           ├─ With Social Security: No offset typical; full benefit paid
│           ├─ With union plans: Coordination to avoid over-insurance
│           └─ Aggregate limit: May cap total benefits across employers
├─ Underwriting & Rating:
│   ├─ Group Underwriting (employer level):
│   │   ├─ Application process:
│   │   │   ├─ Employer completes detailed application
│   │   │   ├─ Insurer reviews: Industry, size, location, claims history
│   │   │   ├─ Field underwriting: Site visit for large groups (>500 employees)
│   │   │   ├─ Approval: 2-4 weeks typical
│   │   │   └─ Conditional: May require additional documentation
│   │   ├─ Group characteristics assessed:
│   │   │   ├─ Industry: Finance → low risk; construction → high risk
│   │   │   ├─ Size: Larger groups (>100) → better claims predictability
│   │   │   ├─ Turnover: High turnover → adverse selection (healthy stay, sick leave)
│   │   │   ├─ Geography: Urban centers → better mortality data; rural → less predictable
│   │   │   ├─ Coverage amount: High coverage → adverse selection (attract sick applicants)
│   │   │   └─ Waiting period: Longer waiting → lower adverse selection
│   │   └─ Risk factors:
│   │       ├─ Participation rate: <80% → potential adverse selection concern
│   │       ├─ Age distribution: Average age >50 → higher mortality → higher rates
│   │       ├─ Salary distribution: Wide spread → harder to predict claims
│   │       └─ Experience rating: Prior group claims (if available) used
│   ├─ Rating Methodology:
│   │   ├─ Community rating (default):
│   │   │   ├─ All employers same area/industry: Same rate
│   │   │   ├─ Insurer averages risk across book
│   │   │   ├─ Advantage: Predictable; new groups rated same as existing
│   │   │   ├─ Disadvantage: Large groups subsidize small; healthy subsidize sick
│   │   │   └─ Regulatory approval: Required in many states
│   │   ├─ Experience rating (common):
│   │   │   ├─ Employer's claims history used
│   │   │   ├─ Formula: Community rate × Experience factor
│   │   │   ├─ Experience factor calculation:
│   │   │   │   ├─ Factor = (Actual claims + credibility) / (Expected claims + credibility)
│   │   │   │   ├─ Credibility = n / (n + k), where n = group size, k = adjustment factor
│   │   │   │   ├─ Small groups (n<50): Low credibility → factor closer to community
│   │   │   │   ├─ Large groups (n>500): High credibility → factor reflects actual experience
│   │   │   │   └─ Example: Group size 200; 3-year actual claims $180K; expected $150K
│   │   │   │       ├─ Credibility ≈ 0.67 (moderate; blended)
│   │   │   │       ├─ Factor = (180K + 0.33×150K) / (150K + 0.33×150K) = 205K/200K = 1.025
│   │   │   │       ├─ Rate adjusted +2.5% for next year
│   │   │   │       └─ Good experience → factor <1.0, rate discount
│   │   │   └─ Renewal: Rates adjusted yearly (typically 1-3 year rate guarantee)
│   │   ├─ Age-adjusted rating:
│   │   │   ├─ If group average age shifts; rates may adjust
│   │   │   ├─ Age increase: Each additional year of average age → +2-3% rate
│   │   │   ├─ Population aging: As workforce matures, rates rise steadily
│   │   │   └─ Mitigation: Younger employees joining slow growth's impact
│   │   └─ Underwriting factors:
│   │       ├─ Standard rate: 100% of mortality table; community average
│   │       ├─ Preferred rate: 85-95% of table (better than average; low risk)
│   │       │   ├─ Trigger: Professional workers, higher education, healthcare
│   │       │   └─ Advantage: Attracts quality groups
│   │       ├─ Substandard rate: 110-125% of table (higher risk)
│   │       │   ├─ Trigger: Hazardous industry, poor past experience
│   │       │   └─ Management: Monitor closely; may impose waiting period
│   │       └─ Occupational adjustments:
│   │           ├─ Standard: Office, professional, retail → 100%
│   │           ├─ +15%: Light manufacturing, transport
│   │           ├─ +30%: Heavy manufacturing, mining, agriculture
│   │           └─ +50%: Hazardous work (construction, military)
│   └─ Individual Underwriting (enrollment):
│       ├─ Guaranteed issue (typical):
│       │   ├─ First $50K-$100K coverage: No medical questions
│       │   ├─ Assumption: Group underwriting provides sufficient gatekeeping
│       │   ├─ Lower amount: Reduces adverse selection incentive
│       │   └─ Timely enrollment: Within 30-60 days of hire
│       ├─ Limited underwriting (if >guaranteed):
│       │   ├─ Medical questions: Past health history, current conditions
│       │   ├─ Alcohol/tobacco: May affect rating if coverage high
│       │   ├─ Age: Rarely decline; may apply flat extra
│       │   └─ Decline: Rare in group context (negative morale impact)
│       ├─ Medical underwriting (rare):
│       │   ├─ Only for very high coverage ($500K+) or supplemental
│       │   ├─ Medical exam: Lab work, EKG if age >50 and coverage >$300K
│       │   └─ Decline: Possible if serious condition discovered
│       └─ Waiver or flat extra:
│           ├─ Condition discovered: May add fixed charge (e.g., +$50/year)
│           ├─ Not decline: Keep employee satisfied; maintain participation
│           └─ Aggregate impact: Usually small on total pool premium
├─ Claims Administration:
│   ├─ Claim Reporting:
│   │   ├─ Notification: Employer or family notifies insurer within 30 days
│   │   ├─ Documentation: Death certificate, beneficiary proof, claim form
│   │   ├─ Proof of loss: Required within 90 days (varies by state)
│   │   └─ Processing time: 10-30 days typical if straightforward
│   ├─ Beneficiary Issues:
│   │   ├─ Designation: Employee names beneficiary (spouse, children, parents)
│   │   ├─ Default: If no designation, usually goes to spouse/children per state law
│   │   ├─ Contested: If multiple claimants; insurer may interplead (court decides)
│   │   ├─ Slayer statute: If beneficiary murders insured; benefit forfeited
│   │   └─ Assignment: Employee can assign to creditor (collateral for loan)
│   ├─ Claim Payout:
│   │   ├─ Lump sum: Typical; beneficiary receives full benefit in one check
│   │   ├─ Installments: Optional; insurer may offer annuity settlement (lower present value)
│   │   ├─ Direct to estate: Or trust; depends on beneficiary designation
│   │   ├─ Taxes: Death benefit proceeds generally NOT subject to income tax (IRC §101)
│   │   └─ Timing: Payment within 15-30 days of approval; state law varies
│   └─ Contestability:
│       ├─ Period: Usually 2 years (insurable interest window)
│       ├─ After 2 years: Insurer can't deny based on misstatement in application
│       ├─ Exception: Fraud (always contestable; infinite period)
│       ├─ Example: Employee misrepresented smoking status; dies year 1; insurer can investigate
│       └─ Good faith: Modern courts interpret narrowly; favor beneficiary
├─ Costs & Pricing Factors:
│   ├─ Premium Components:
│   │   ├─ Mortality cost: 70-80% of premium (actual death claims)
│   │   ├─ Expense load: 15-20% (admin, commissions, claims processing)
│   │   ├─ Profit margin: 5-10% (insurer profit)
│   │   ├─ Risk adjustment: ±5-15% (group-specific adjustments)
│   │   └─ Example: $1.00/employee cost breakdown:
│   │       ├─ $0.70 = Mortality
│   │       ├─ $0.18 = Expense
│   │       ├─ $0.07 = Profit
│   │       ├─ $0.05 = Risk adjustment
│   │       └─ Total = $1.00/employee annually
│   ├─ Pricing Considerations:
│   │   ├─ Size: <50 employees → higher load; >1000 employees → lower load
│   │   ├─ Participation: >90% participation → lower load; <75% → higher load
│   │   ├─ Retention: Long-term clients → lower rates; new prospects → higher rates
│   │   ├─ Commissions: Broker paid 8-12% of premium (passed through in rates)
│   │   └─ Market: Competitive pressure → rates may be below cost for growth
│   └─ Trend Factors:
│       ├─ Mortality improvement: -1% to -2% annual (people living longer)
│       ├─ Medical inflation: +3% to +5% annual (cost of claims increasing)
│       ├─ Disability correlation: Economic downturn → more disability claims
│       ├─ Behavioral: Better wellness programs → lower claims
│       └─ Net trend: Typically +1% to +3% annual rate increase overall
├─ Administrative Roles:
│   ├─ Employer (Plan Sponsor):
│   │   ├─ Responsibilities: Collect premiums from employees, report to insurer
│   │   ├─ Enrollment: Maintain eligibility records, notify changes
│   │   ├─ Communication: Educate employees about benefits
│   │   ├─ ERISA compliance: Filing, plan documents, non-discrimination testing
│   │   ├─ Fiduciary duty: Prudently manage plan for employee benefit
│   │   └─ Liability: Generally limited (insurer primary); but fiduciary exposure
│   ├─ Insurance Company (Insurer):
│   │   ├─ Underwriting: Evaluate group, set rates, approve coverage
│   │   ├─ Policy administration: Maintain master policy, handle renewals
│   │   ├─ Claims: Adjudicate and pay claims
│   │   ├─ Compliance: Regulatory filings, consumer protections, solvency
│   │   └─ Customer service: Employer/employee support, enrollments
│   ├─ Third-Party Administrator (TPA):
│   │   ├─ Role: Intermediate between employer, insurer, employees
│   │   ├─ Services: Enrollment processing, eligibility records, claims support
│   │   ├─ Employer: Outsources to TPA to reduce internal burden
│   │   ├─ Insurer: May use TPA for claims handling (especially self-funded plans)
│   │   └─ Advantage: Economies of scale; specialized expertise
│   ├─ Broker/Consultant:
│   │   ├─ Employer advisor: Counsels on plan design, benefits strategy
│   │   ├─ Insurer liaison: Facilitates underwriting, renewal, claims
│   │   ├─ Commission: Typically 8-12% of premium (paid by insurer)
│   │   ├─ Advocacy: May push for lower rates on employer's behalf
│   │   └─ Impartiality: Broker represents employer's interests (not insurer's)
│   └─ Government:
│       ├─ Federal: DOL (ERISA), IRS (tax qualification), EEOC (non-discrimination)
│       ├─ State: Insurance commissioner (rate approval, insolvency), Labor dept (payroll deductions)
│       ├─ Oversight: Plan documents, disclosure requirements, consumer protections
│       └─ Mandates: ACA minimum coverage, disability accommodation
└─ Regulatory & Compliance Framework:
    ├─ ERISA (Employer Retirement Income Security Act):
    │   ├─ Coverage: Group plans with ≥2 employees (certain exceptions)
    │   ├─ Requirements: Plan document, SPD (Summary Plan Description), Form 5500 filing
    │   ├─ Fiduciary duty: Employers must act in participant interest
    │   ├─ Claims procedure: Fair and neutral process for disputes
    │   └─ Remedies: ERISA violations allow litigation; damages + attorney fees
    ├─ ACA (Affordable Care Act) Requirements:
    │   ├─ Essential benefits: Group health plans must cover certain preventive services
    │   ├─ Minimum participation: ≥50% of eligible employees must participate (safe harbor)
    │   ├─ Automatic enrollment: Large employers (≥100) must enroll eligible employees
    │   ├─ Notice requirements: Employees informed of benefits, costs, rights
    │   └─ Non-discrimination: Benefits can't discriminate based on health status
    ├─ Non-discrimination Testing (ERISA §510):
    │   ├─ Requirement: Plan can't discriminate against "protected groups"
    │   ├─ Protected: Employees based on age, sex, health status
    │   ├─ Testing: Employer must demonstrate no discrimination
    │   ├─ Highly compensated: Can't receive greater benefits
    │   └─ Nondiscrimination clause: Required in all group insurance policies
    ├─ Tax Treatment (IRC):
    │   ├─ Employer contributions: Tax-deductible business expense
    │   ├─ Employee premium: If paid via payroll deduction, usually tax-exempt
    │   ├─ Benefit proceeds: Generally NOT subject to income tax (IRC §101)
    │   ├─ Group-to-individual: Taxable income if employee buys individual with group rates
    │   └─ Tax qualification: Plan must meet requirements; audited annually
    ├─ Disclosure & Transparency:
    │   ├─ Plan document: Master policy and amendments available to employees
    │   ├─ SPD: Summary Plan Description in plain language; key terms explained
    │   ├─ Open enrollment materials: Benefits, coverage, rates explained
    │   ├─ Notice of material changes: 30-day notice required
    │   └─ Annual statements: Employees get itemized benefits, costs
    ├─ State Insurance Regulation:
    │   ├─ Policy requirements: Filing with insurance commissioner; rate approval
    │   ├─ Reserves: Insurer must hold adequate reserves for claims
    │   ├─ Insolvency fund: If insurer fails; member protection (varies by state)
    │   ├─ Guaranty association: Backstop for failed insurers; coverage limits
    │   └─ Consumer protections: Complaint handling, appeals process
    └─ COBRA (Continuation Health Coverage):
        ├─ Requirement: Employers with ≥20 employees must offer continuation
        ├─ Duration: 18 months after job loss (24 months if disability occurs)
        ├─ Premium: Employee pays full premium + 2% administrative fee
        ├─ Notice: Employer must notify employee of rights within 14 days
        ├─ Cost: Typically $800-2,000/month for family coverage (expensive)
        └─ Trade-off: Temporary coverage while seeking new employment
```

**Key Insight:** Group life insurance efficient due to pooling, simplified underwriting, and economies of scale; employer role critical to plan success

## 5. Mini-Project
[Would include: premium calculation by employee class, experience rating adjustment, participation analysis, cost allocation by department]

## 6. Challenge Round
When group life fails:
- **Participation cliff**: At 74% participation; group rates require ≥75%; below cliff triggers massive rate increase or coverage termination
- **Adverse selection spiral**: Young, healthy employees opt-out; older, sicker stay; claims jump; rates increase; more healthy exit; vicious cycle
- **Claim spike**: Unexpected deaths (bus accident, workplace tragedy); 5 claims in 1 month; insurer reviews; threatens non-renewal
- **Benefit overstatement**: Employee thinks benefit is salary amount; actually 1× salary; widow expects $100K, gets $50K; complaint + PR disaster
- **Retiree liability**: Group covers retirees; costs explode; insurer wasn't planning on retiree population; threatens coverage continuation
- **ERISA violation**: Employer fails to provide SPD; employees sue for lack of transparency; damages + attorney fees awarded

## 7. Key References
- [SOA - Group Life Insurance Fundamentals](https://www.soa.org/) - Underwriting, pricing, administration
- [ERISA Overview (Department of Labor)](https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/erisa) - Regulatory framework
- [Group Life Insurance Handbook (LOMA)](https://www.loma.org/) - Best practices, case studies

---
**Status:** Employer-sponsored coverage | **Complements:** Short-Term Disability, Retirement Plans, Medical Insurance
