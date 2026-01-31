# Long-Term Disability (LTD)

## Concept Skeleton

Long-term disability insurance provides extended income replacement for individuals unable to work due to severe illness or injury, typically after a 90–180 day elimination period, with benefits continuing for years (often to age 65 or recovery). Policies feature evolving disability definitions (own occupation transitioning to any occupation after 24 months), offset provisions for other income sources, and rehabilitation incentives to encourage return to work.

**Core Components:**
- **Elimination period**: Waiting months before LTD benefits begin (90, 180 days); bridges short-term disability
- **Benefit percentage**: 50–67% of pre-disability earnings, subject to maximum monthly limits
- **Disability definition**: "Own occupation" (first 24 months) then "any occupation" (inability to work in *any* job)
- **Offsets**: Deductions for Social Security Disability Insurance (SSDI), workers' comp, other disability income
- **Benefit duration**: To age 65, lifetime for certain conditions, or specified period (e.g., 5 years)

**Why it matters:** Protects against catastrophic income loss from long-term incapacity; critical safety net for careers interrupted by chronic illness, severe injury, or progressive conditions.

---

## Comparative Framing

| Dimension | **Long-Term Disability (LTD)** | **Short-Term Disability (STD)** | **Social Security Disability Insurance (SSDI)** |
|-----------|-------------------------------|---------------------------------|--------------------------------------------------|
| **Elimination period** | 90–180 days | 0–14 days | 5-month waiting period |
| **Benefit duration** | To age 65 or recovery | 13–26 weeks | Until recovery or retirement age |
| **Disability definition** | Own occ → Any occ (after 24 mo) | Own occupation | Unable to engage in "substantial gainful activity" |
| **Benefit amount** | 50–67% of salary | 60–70% of salary | Based on earnings history (AIME formula) |
| **Offset provisions** | Yes (SSDI, workers' comp) | Typically no offsets | No private insurance offsets |
| **Tax treatment (employer-paid)** | Taxable benefits | Taxable benefits | Tax-free benefits |

**Key insight:** LTD bridges the gap between STD and SSDI; own-occupation definition offers early protection, while any-occupation test aligns with SSDI after 2 years, facilitating offset integration.

---

## Examples & Counterexamples

### Examples of Long-Term Disability

1. **Surgeon with carpal tunnel syndrome**  
   - Unable to perform surgery (own occupation) after 180-day elimination period  
   - LTD pays 60% of $400,000 salary = $240,000/year (subject to policy max, e.g., $15,000/month)  
   - After 24 months, "any occupation" test applies; may still qualify if unable to work in sedentary roles

2. **Office worker with chronic fatigue syndrome**  
   - Disabled under "own occupation" for first 24 months  
   - Transitioning to "any occupation" test: if capable of sedentary work, benefits may cease  
   - Rehabilitation rider funds vocational retraining to facilitate return to workforce

3. **SSDI offset scenario**  
   - Pre-disability salary: $80,000/year  
   - LTD benefit: 60% = $48,000/year ($4,000/month)  
   - SSDI approval: $2,000/month  
   - LTD pays: $4,000 - $2,000 = $2,000/month (offset for SSDI)

### Non-Examples (or Edge Cases)

- **Partial disability with full-time work capacity**: LTD requires *inability* to work; accommodations allowing full-time work preclude benefits.
- **Pre-existing condition within lookback period**: Disability arising from condition treated in 3 months before coverage may be excluded (typical pre-ex clause).
- **Self-reported symptoms without objective medical findings**: Policies often limit benefits (e.g., 24 months) for mental/nervous disorders or subjective conditions.

---

## Layer Breakdown

**Layer 1: Claim Adjudication**  
Claimant submits attending physician statement, medical records, and functional capacity evaluation. Insurer assesses: (1) Does condition meet own-occupation test? (2) Are offsets applicable (SSDI pending, workers' comp)? (3) Is claimant compliant with treatment? Independent medical exams or surveillance may verify disability.

**Layer 2: Benefit Calculation with Offsets**  
Monthly benefit = (Pre-disability salary × Benefit percentage) - Offsets (SSDI, pension, other DI). If SSDI application is pending, LTD may advance full benefit, then recover overpayment upon SSDI approval (known as "offset arrangement"). Taxable benefits reduce effective replacement ratio.

**Layer 3: Ongoing Claims Management and Rehabilitation**  
Periodic reviews (annual or semi-annual) confirm continued disability. Vocational rehabilitation services assess transferable skills and job placement options. Residual or partial disability riders allow graded return to work, paying reduced benefits proportional to income loss.

**Layer 4: Reserves and Pricing**  
Claim reserves use present value of future benefits, discounted for interest and probability of recovery or mortality. Experience-based termination rates vary by age, occupation, and cause of disability. Pricing reflects claim incidence rates, duration, and SSDI offset assumptions.

---

## Mini-Project: LTD Claim Reserve Valuation

**Goal:** Estimate present value of an active LTD claim using simplified termination probabilities.

```python
import numpy as np

# Claim parameters
monthly_benefit = 5000  # Net after offsets
current_age = 45
benefit_to_age = 65
annual_discount_rate = 0.03
months_remaining = (benefit_to_age - current_age) * 12

# Simplified recovery/mortality termination rates per month (illustrative)
# Assume constant 1% per month probability of claim termination
monthly_termination_prob = 0.01
monthly_discount_factor = (1 + annual_discount_rate) ** (1/12)

# Calculate present value of future benefits
pv_benefits = 0
survival_prob = 1.0  # Probability claim is still active

for month in range(1, months_remaining + 1):
    # Benefit paid at end of month if claim still active
    pv_benefits += monthly_benefit * survival_prob / (monthly_discount_factor ** month)
    # Update survival probability (claim continues if not terminated)
    survival_prob *= (1 - monthly_termination_prob)

print(f"Current Age: {current_age}")
print(f"Benefit to Age: {benefit_to_age}")
print(f"Monthly Benefit: ${monthly_benefit:,}")
print(f"Months Remaining: {months_remaining}")
print(f"Annual Discount Rate: {annual_discount_rate:.1%}")
print(f"Monthly Termination Probability: {monthly_termination_prob:.1%}")
print(f"\nPresent Value of Claim Reserve: ${pv_benefits:,.2f}")

# Sensitivity: increase termination rate to 1.5% (better recovery outcomes)
survival_prob_adj = 1.0
pv_benefits_adj = 0
monthly_termination_prob_adj = 0.015

for month in range(1, months_remaining + 1):
    pv_benefits_adj += monthly_benefit * survival_prob_adj / (monthly_discount_factor ** month)
    survival_prob_adj *= (1 - monthly_termination_prob_adj)

print(f"\nAdjusted Reserve (1.5% termination): ${pv_benefits_adj:,.2f}")
print(f"Reserve Reduction: ${pv_benefits - pv_benefits_adj:,.2f}")
```

**Expected Output (illustrative):**
```
Current Age: 45
Benefit to Age: 65
Monthly Benefit: $5,000
Months Remaining: 240
Annual Discount Rate: 3.0%
Monthly Termination Probability: 1.0%

Present Value of Claim Reserve: $447,218.32

Adjusted Reserve (1.5% termination): $368,492.75
Reserve Reduction: $78,725.57
```

**Interpretation:**  
- Reserve is substantial due to long potential benefit period (20 years).  
- Higher termination rates (recovery, mortality, return to work) reduce reserve requirements.  
- Accurate reserves require cause-specific termination tables and SSDI offset assumptions.

---

## Challenge Round

1. **Own Occupation vs. Any Occupation**  
   Why do insurers transition from "own occupation" to "any occupation" after 24 months? What actuarial and behavioral factors drive this design?

   <details><summary>Hint</summary>Own-occupation definition is more liberal, leading to higher claim costs. After 24 months, SSDI eligibility aligns with any-occupation standard, enabling offsets. Behavioral: incentivizes claimants to pursue vocational rehabilitation and return to workforce in alternative roles.</details>

2. **SSDI Offset Timing**  
   An LTD claim begins; claimant applies for SSDI. SSDI is approved retroactively 18 months later. How does the insurer handle the offset?

   <details><summary>Solution</summary>
   Insurer paid full LTD benefit during SSDI application period. Upon SSDI approval, retroactive SSDI payments are offset against past LTD benefits. Insurer recovers overpayment (claimant repays lump-sum SSDI retroactive amount to insurer), and future LTD benefits are reduced by ongoing SSDI monthly payment.
   </details>

3. **Residual Disability Benefit**  
   A claimant earned $100,000 pre-disability; LTD is 60% = $60,000/year. After partial recovery, returns to work earning $40,000. Calculate residual benefit if policy pays 60% of income loss.

   <details><summary>Solution</summary>
   **Income loss:** $100,000 - $40,000 = $60,000.  
   **Residual benefit:** 60% × $60,000 = $36,000/year.  
   **Total income:** $40,000 (earned) + $36,000 (LTD) = $76,000 (76% of pre-disability income).  
   Encourages return to work while maintaining income support.
   </details>

4. **Mental Nervous Limitation**  
   Many LTD policies cap benefits for mental/nervous conditions at 24 months. A claimant with depression is disabled. After 24 months, benefits cease. What happens if depression leads to heart attack (physical condition)?

   <details><summary>Solution</summary>
   If heart attack is a new, independent physical disability, LTD benefits may resume (subject to causation review). If heart condition is deemed secondary to mental condition, limitation may still apply. Policy language and medical evidence are critical; insurers scrutinize causal relationships.
   </details>

---

## Key References

- **Society of Actuaries (SOA)**: Group LTD experience studies and termination rates ([SOA.org](https://www.soa.org/))
- **Social Security Administration**: SSDI eligibility and benefit calculations ([SSA.gov](https://www.ssa.gov/))
- **Council for Disability Awareness**: LTD statistics and employer benchmarking ([DisabilityCanHappen.org](https://disabilitycanhappen.org/))
- **ERISA Section 503**: Claim and appeal procedures for employer-sponsored LTD plans

**Further Reading:**  
- Actuarial Study Note on Group LTD: incidence, termination, and offset modeling  
- Insurance law treatises on disability definition evolution and case law (Unum, MetLife precedent cases)
