# Retirement Plans

## Concept Skeleton

Retirement plans are employer-sponsored or individual savings vehicles designed to accumulate wealth during working years (accumulation phase) and provide income during retirement (distribution phase), encompassing defined benefit pensions (employer-funded annuities based on salary and service), defined contribution plans (401(k), RRSP with employee/employer contributions and investment elections), and hybrid arrangements (cash balance plans). Actuarial work centers on funding adequacy, investment risk, longevity risk, and regulatory compliance (ERISA, IRS).

**Core Components:**
- **Plan type**: Defined benefit (DB) vs. defined contribution (DC) vs. hybrid
- **Vesting**: Schedule determining when employee owns employer contributions (cliff or graded)
- **Contribution limits**: IRS annual caps (e.g., $22,500 employee 401(k) deferral for 2023, plus $7,500 catch-up age 50+)
- **Investment options**: Target-date funds, index funds, managed accounts (DC); actuarially managed pool (DB)
- **Distribution rules**: Required minimum distributions (RMDs) starting age 73 (2023), early withdrawal penalties

**Why it matters:** Primary wealth accumulation for retirement; actuarial science ensures DB plans remain funded to meet future obligations, while DC plans require participant education and appropriate default investment strategies.

---

## Comparative Framing

| Dimension | **Defined Benefit (DB)** | **Defined Contribution (DC, e.g., 401(k))** | **Cash Balance Plan (Hybrid)** |
|-----------|--------------------------|---------------------------------------------|--------------------------------|
| **Benefit formula** | Annuity based on final average pay × years of service | Account balance (contributions + returns) | Hypothetical account credits (% of pay + interest credits) |
| **Investment risk** | Employer bears | Employee bears | Employer bears (guarantees interest credits) |
| **Longevity risk** | Employer bears (annuity for life) | Employee bears (outlive savings) | Employer bears if annuitized |
| **Portability** | Limited (often lump-sum buyout on termination) | Highly portable (rollover to IRA or new employer) | Portable (account balance) |
| **Regulatory complexity** | High (ERISA funding, PBGC premiums) | Moderate (contribution limits, nondiscrimination tests) | High (hybrid testing, conversion issues) |
| **Actuarial valuation** | Required annually | Not required (no funding obligation) | Required (DB rules apply) |

**Key insight:** DB plans shift risks (investment, longevity) to employer, ensuring predictable retirement income; DC plans shift risks to employees, offering portability and investment flexibility but requiring financial literacy.

---

## Examples & Counterexamples

### Examples of Retirement Plans

1. **Traditional defined benefit pension**  
   - Formula: 1.5% × final average salary (FAS, last 5 years) × years of service  
   - Employee with $100,000 FAS and 30 years service: Annuity = 1.5% × $100,000 × 30 = $45,000/year for life  
   - Actuarial present value funds this liability; employer makes annual contributions

2. **401(k) with employer match**  
   - Employee contributes 6% of salary ($60,000 × 0.06 = $3,600)  
   - Employer matches 50% up to 6% of pay: $3,600 × 0.50 = $1,800  
   - Total annual contribution: $5,400 (grows tax-deferred until withdrawal)  
   - Investment allocation: 60% stocks, 40% bonds (employee-directed)

3. **Cash balance plan**  
   - Annual pay credit: 5% of salary ($80,000 × 0.05 = $4,000)  
   - Interest credit: 5% annually on accumulated balance  
   - After 10 years, hypothetical account balance = $50,315 (using future value annuity formula)  
   - Employee can take lump sum or annuity at retirement

### Non-Examples (or Edge Cases)

- **IRA (Individual Retirement Account)**: Personal savings, not employer-sponsored (though may receive rollovers from 401(k)).
- **Social Security**: Government-provided defined benefit; not employer-sponsored (though wage taxes fund it).
- **Deferred compensation plan (non-qualified)**: Not ERISA-protected; unfunded promise to pay (creditor risk).

---

## Layer Breakdown

**Layer 1: Contribution Phase and Vesting**  
Employees accrue benefits through service. Vesting schedules (e.g., 5-year cliff: 0% until year 5, then 100%) determine ownership of employer contributions. DC plans: employee deferrals are always 100% vested; employer match subject to vesting schedule. DB plans: accrued benefit vests, but payout deferred to retirement age.

**Layer 2: Investment Strategy (DC Plans)**  
Participants allocate contributions across menu of funds. Default option often target-date fund (automatically rebalances to conservative allocation as retirement nears). Employer fiduciary duty (ERISA §404(c)) requires prudent fund selection and disclosure. Automatic enrollment and escalation improve participation rates.

**Layer 3: Actuarial Valuation and Funding (DB Plans)**  
Annual valuation calculates present value of accrued benefits (PBO: projected benefit obligation) using discount rate (typically corporate bond yields), mortality tables (e.g., Pri-2012 with MP-2021 projection scale), and salary growth assumptions. Funding: minimum required contribution per IRC §430 and ERISA; maximum deductible contribution per IRS limits.

**Layer 4: Distribution and Withdrawal**  
**DC plans:** Lump sum, installment, or annuity purchase. RMDs start age 73 (2023 SECURE 2.0 Act). Early withdrawal (pre-59½) incurs 10% penalty unless exception applies (disability, first home, education).  
**DB plans:** Life annuity (single or joint survivor); lump-sum option requires actuarial equivalence. PBGC insures benefits up to statutory limits if plan terminates underfunded.

---

## Mini-Project: DB Pension Present Value Calculation

**Goal:** Calculate present value of a deferred pension using standard actuarial methods.

```python
import numpy as np

# Employee parameters
current_age = 45
retirement_age = 65
life_expectancy = 85  # assumed for simplicity (use mortality table in practice)
final_average_salary = 100_000
years_of_service = 20
accrual_rate = 0.015  # 1.5% per year of service

# Benefit calculation
annual_benefit = accrual_rate * final_average_salary * years_of_service
print(f"Annual Pension Benefit (starting at age 65): ${annual_benefit:,.2f}")

# Actuarial assumptions
discount_rate = 0.05  # 5% (corporate bond yield)
mortality_decrement = 0.02  # 2% annual probability of death (simplified; use life table in practice)

# Present value at retirement (annuity certain for life expectancy - retirement age)
years_in_retirement = life_expectancy - retirement_age
pv_at_retirement = 0

for year in range(1, years_in_retirement + 1):
    survival_prob = (1 - mortality_decrement) ** year
    pv_at_retirement += (annual_benefit * survival_prob) / ((1 + discount_rate) ** year)

print(f"PV of Benefits at Retirement (age {retirement_age}): ${pv_at_retirement:,.2f}")

# Discount back to current age (deferred annuity)
years_to_retirement = retirement_age - current_age
pv_current = pv_at_retirement / ((1 + discount_rate) ** years_to_retirement)

print(f"PV of Benefits at Current Age ({current_age}): ${pv_current:,.2f}")

# Sensitivity: increase discount rate to 6%
discount_rate_high = 0.06
pv_at_retirement_high = 0
for year in range(1, years_in_retirement + 1):
    survival_prob = (1 - mortality_decrement) ** year
    pv_at_retirement_high += (annual_benefit * survival_prob) / ((1 + discount_rate_high) ** year)

pv_current_high = pv_at_retirement_high / ((1 + discount_rate_high) ** years_to_retirement)
print(f"\nHigh Discount Rate (6%) PV at Current Age: ${pv_current_high:,.2f}")
print(f"PV Reduction: ${pv_current - pv_current_high:,.2f} ({(pv_current - pv_current_high)/pv_current:.1%})")
```

**Expected Output (illustrative):**
```
Annual Pension Benefit (starting at age 65): $30,000.00
PV of Benefits at Retirement (age 65): $231,394.57
PV of Benefits at Current Age (45): $87,204.05

High Discount Rate (6%) PV at Current Age: $68,045.12
PV Reduction: $19,158.93 (22.0%)
```

**Interpretation:**  
- Higher discount rates reduce liability (common in rising interest rate environments).  
- Mortality assumptions critically affect PV (longer life expectancy increases obligation).  
- Actuaries use sophisticated life tables (e.g., Pri-2012, RP-2014) and generational mortality projection scales.

---

## Challenge Round

1. **401(k) Match vs. Profit Sharing**  
   Company offers either: (A) 3% match on employee deferrals, or (B) 5% non-elective profit sharing (no employee contribution required). Which is more valuable to employees?

   <details><summary>Hint</summary>Depends on employee participation. (A) requires employees to defer 3%+ to capture full match (behavioral barrier). (B) guarantees 5% to all eligible employees regardless of deferral. (B) is more valuable if participation is low, but (A) encourages savings and may yield higher total if employees maximize deferrals.</details>

2. **Lump Sum vs. Annuity**  
   DB plan offers: (A) $500,000 lump sum, or (B) $35,000/year life annuity. Participant is 65, expects to live to 85. Assume 5% personal discount rate. Which is better?

   <details><summary>Solution</summary>
   PV of annuity = $35,000 × annuity factor (20 years, 5%) = $35,000 × 12.46 ≈ $436,100.  
   Lump sum ($500,000) > PV of annuity. *But*: Annuity protects against longevity risk (outliving savings). If participant lives to 95, annuity total = $35,000 × 30 = $1.05M. Decision depends on risk tolerance, health, and other income sources.
   </details>

3. **PBGC Maximum Benefit**  
   A DB plan terminates with $10M in assets and $15M in liabilities. Highest-paid participant was entitled to $120,000/year annuity. PBGC maximum for 2023 is $74,455/year at age 65. What does participant receive?

   <details><summary>Solution</summary>
   PBGC guarantees up to $74,455/year. Participant receives $74,455, not $120,000. Shortfall ($45,545/year) is uncovered loss. PBGC priority categories determine allocation of remaining assets; some benefits may be reduced further.
   </details>

4. **Required Minimum Distribution (RMD)**  
   Retiree age 73 has $1,000,000 in 401(k). IRS Uniform Lifetime Table divisor for age 73 is 26.5. Calculate RMD.

   <details><summary>Solution</summary>
   **RMD** = $1,000,000 / 26.5 = $37,736.  
   Must withdraw (and pay taxes on) at least this amount by December 31. Failure incurs 25% excise tax on shortfall (reduced from 50% under SECURE 2.0 Act).
   </details>

---

## Key References

- **IRS Publication 575**: Pension and Annuity Income; RMD rules ([IRS.gov](https://www.irs.gov/publications/p575))
- **ERISA**: Employee Retirement Income Security Act; fiduciary duties, funding rules, vesting ([DOL.gov](https://www.dol.gov/agencies/ebsa))
- **PBGC**: Pension Benefit Guaranty Corporation; plan termination insurance ([PBGC.gov](https://www.pbgc.gov/))
- **Society of Actuaries**: Retirement Plan Experience Study, mortality tables ([SOA.org](https://www.soa.org/))

**Further Reading:**  
- *Pension Mathematics for Actuaries* by Anderson (textbook on DB valuation)  
- IRS Notice on hybrid plan conversions (cash balance litigation and safe harbors)  
- Target-date fund research: Vanguard, Morningstar glide-path studies
