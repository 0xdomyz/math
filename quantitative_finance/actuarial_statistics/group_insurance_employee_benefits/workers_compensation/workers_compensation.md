# Workers' Compensation

## Concept Skeleton

Workers' compensation is a state-mandated, no-fault insurance system providing medical care and wage replacement for employees injured or ill due to job-related causes, in exchange for employer immunity from tort liability. Benefits include medical treatment (unlimited in most states), temporary/permanent disability indemnity (percentage of wages), death benefits to dependents, and vocational rehabilitation, financed by employer-paid premiums based on payroll and occupational hazard classifications.

**Core Components:**
- **No-fault liability**: Employee receives benefits regardless of fault; employer gains lawsuit protection
- **Medical benefits**: Full coverage of necessary treatment, therapy, medications (no deductibles/copays)
- **Indemnity benefits**: Wage replacement during disability (typically 2/3 of average weekly wage, subject to state maximums)
- **Permanent disability ratings**: Scheduled awards (e.g., loss of hand) or unscheduled (whole-person impairment)
- **Occupational disease**: Coverage extends to illnesses arising from workplace exposure (asbestosis, carpal tunnel, etc.)

**Why it matters:** Protects workers from financial hardship after job injuries while shielding employers from unlimited litigation; actuarial pricing balances adequate reserves for long-tail claims with competitive rates across industries.

---

## Comparative Framing

| Dimension | **Workers' Compensation** | **Short-Term Disability (STD)** | **Long-Term Disability (LTD)** |
|-----------|---------------------------|---------------------------------|--------------------------------|
| **Cause of disability** | Job-related injury/illness only | Non-occupational illness/injury | Non-occupational illness/injury |
| **Benefit structure** | 2/3 wages (state max) + full medical | 60–70% salary for 13–26 weeks | 50–67% salary to age 65 |
| **Waiting period** | 3–7 days (varies by state) | 0–14 days | 90–180 days |
| **Coverage mandate** | Mandatory employer coverage | Voluntary employee benefit | Voluntary employee benefit |
| **Tax treatment** | Benefits tax-free | Taxable if employer-paid | Taxable if employer-paid |
| **Employer liability** | Exclusive remedy (no tort suits)* | N/A (no liability shield) | N/A (no liability shield) |

*Exceptions: intentional harm, gross negligence may allow tort claims in some states.

**Key insight:** Workers' comp is statutory and exclusive for job injuries; STD/LTD cover off-the-job disabilities. Integration is rare due to separate legal frameworks, but coordination may occur in return-to-work scenarios.

---

## Examples & Counterexamples

### Examples of Workers' Compensation

1. **Construction worker falls from scaffold**  
   - Fractures leg; requires surgery, 6 months recovery  
   - **Medical benefits**: Full coverage of ER, surgery, PT (no copays)  
   - **Temporary total disability (TTD)**: 2/3 × $1,200/week (AWW) = $800/week for 26 weeks  
   - **Permanent partial disability (PPD)**: 10% leg impairment = scheduled award (e.g., 52 weeks × 10% × $800 = $4,160)

2. **Office worker develops carpal tunnel from repetitive keyboard use**  
   - Occupational disease claim  
   - Medical treatment: splints, injections, potential surgery  
   - Modified duty assignment (ergonomic keyboard, reduced hours) with temporary partial disability if wages reduced

3. **Death benefit for workplace fatality**  
   - Employee killed in industrial accident  
   - Dependents receive weekly death benefits (e.g., 2/3 AWW until remarriage or children reach age 18)  
   - Funeral expenses (typically $5,000–$10,000 cap per state)

### Non-Examples (or Edge Cases)

- **Injury during lunch break off-site**: Typically not compensable (not "arising out of and in the course of employment").
- **Heart attack at work with no job-related stress/exertion**: May be denied unless causal link established (pre-existing condition).
- **Horseplay injury (self-inflicted)**: May be excluded if employee substantially deviated from job duties; employer may contest.

---

## Layer Breakdown

**Layer 1: Compensability and Claim Reporting**  
Employer must be notified promptly (state deadlines, e.g., 30 days). Claim filed with state workers' comp board or insurer. Compensability requires: (1) employment relationship, (2) injury/illness, (3) arising out of and in the course of employment. Disputed claims proceed to administrative hearings or board review.

**Layer 2: Medical Case Management**  
Injured worker receives treatment from approved provider panel (or personal physician in some states). Insurer may direct care to occupational health clinics, utilization review to control costs. Independent medical examinations (IMEs) assess maximum medical improvement (MMI) and impairment ratings.

**Layer 3: Indemnity Benefit Calculation**  
- **Temporary Total Disability (TTD)**: 2/3 AWW (average weekly wage, computed from prior 52 weeks) while unable to work.  
- **Temporary Partial Disability (TPD)**: If returns to light duty at reduced wage, benefit = 2/3 × (AWW - current wage).  
- **Permanent Partial Disability (PPD)**: Lump sum or weekly payments based on impairment rating (AMA Guides to Evaluation of Permanent Impairment).  
- **Permanent Total Disability (PTD)**: Lifetime benefits if unable to return to any employment.

**Layer 4: Pricing and Loss Reserving**  
Premium rates by class code (e.g., clerical 0.10 per $100 payroll; roofing 25.00 per $100). Experience modification factor adjusts rate based on employer's claim history. Reserves: case reserves for known claims, IBNR (incurred but not reported), and bulk reserves for future development of long-tail claims (e.g., cumulative trauma, occupational disease).

---

## Mini-Project: Premium Calculation by Class Code

**Goal:** Calculate workers' comp premium for a multi-class employer using manual rates and experience mod.

```python
import numpy as np

# Payroll and class codes
class_codes = [
    {"code": "8810", "description": "Clerical Office", "payroll": 1_000_000, "rate": 0.20},  # per $100
    {"code": "5403", "description": "Carpentry", "payroll": 500_000, "rate": 12.50},
    {"code": "5022", "description": "Masonry", "payroll": 300_000, "rate": 18.00},
]

# Experience modification factor (1.0 = average; <1.0 = better than average; >1.0 = worse)
experience_mod = 0.90  # 10% credit for good loss experience

# Calculate manual premium by class
manual_premiums = []
for cls in class_codes:
    premium = (cls["payroll"] / 100) * cls["rate"]
    manual_premiums.append(premium)
    print(f"{cls['description']:20s} | Payroll: ${cls['payroll']:>10,} | "
          f"Rate: ${cls['rate']:>6.2f} | Premium: ${premium:>10,.2f}")

total_manual_premium = np.sum(manual_premiums)
print(f"\n{'Total Manual Premium:':40s} ${total_manual_premium:>10,.2f}")

# Apply experience modification
final_premium = total_manual_premium * experience_mod
print(f"Experience Mod: {experience_mod:.2f}")
print(f"{'Final Premium (with Exp Mod):':40s} ${final_premium:>10,.2f}")

# Estimate expected losses (assume 65% loss ratio)
expected_losses = final_premium * 0.65
print(f"\n{'Expected Losses (65% loss ratio):':40s} ${expected_losses:>10,.2f}")
print(f"{'Overhead & Profit Margin:':40s} ${final_premium - expected_losses:>10,.2f}")
```

**Expected Output (illustrative):**
```
Clerical Office      | Payroll: $1,000,000 | Rate:   $0.20 | Premium:  $2,000.00
Carpentry            | Payroll:   $500,000 | Rate:  $12.50 | Premium: $62,500.00
Masonry              | Payroll:   $300,000 | Rate:  $18.00 | Premium: $54,000.00

Total Manual Premium:                    $118,500.00
Experience Mod: 0.90
Final Premium (with Exp Mod):             $106,650.00

Expected Losses (65% loss ratio):          $69,322.50
Overhead & Profit Margin:                  $37,327.50
```

**Interpretation:**  
- High-hazard classes (masonry, carpentry) dominate premium despite lower payroll.  
- Experience mod rewards employers with strong safety programs.  
- Loss ratio (losses / premium) is key metric; insurers target 60–70% for profitability.

---

## Challenge Round

1. **Exclusive Remedy Doctrine**  
   An employee injured by defective machinery sues employer for negligence. Employer cites workers' comp exclusive remedy. What is the likely outcome?

   <details><summary>Hint</summary>Employer is generally immune from tort suits; exclusive remedy doctrine bars negligence claims. *Exception*: Employee may sue third-party manufacturer for product liability (employer remains protected). Some states allow tort claims for intentional harm or gross negligence.</details>

2. **Subrogation Rights**  
   Employee injured in car accident caused by third party while driving for work. Workers' comp pays $100,000 in benefits. Employee recovers $150,000 in tort settlement. How does insurer recover?

   <details><summary>Solution</summary>
   Insurer has subrogation right to recover $100,000 from tort settlement. Employee retains $50,000 (difference). If settlement were only $80,000, insurer and employee share proportionally per state rules (e.g., insurer $80k, employee $0, or pro-rated).
   </details>

3. **Second Injury Fund**  
   Employee with pre-existing 20% back impairment suffers additional 30% impairment from work injury (total 50% combined). Employer liable for 30%; who pays remaining 20%?

   <details><summary>Solution</summary>
   State Second Injury Fund (or Special Disability Fund) reimburses employer for excess disability attributable to pre-existing condition. Encourages hiring workers with disabilities without penalizing employer for combined effects. (Note: Many states have phased out these funds due to funding challenges.)
   </details>

4. **Loss Development Factor**  
   An insurer sets case reserves of $500,000 for claims from policy year 2020. Historical data shows reserves develop to 1.4× by final settlement (average 5 years). Estimate ultimate losses.

   <details><summary>Solution</summary>
   **Ultimate losses** = $500,000 × 1.4 = $700,000.  
   Loss development triangles and chain-ladder methods refine these estimates; actuaries monitor paid vs. incurred ratios to adjust reserves annually.
   </details>

---

## Key References

- **National Council on Compensation Insurance (NCCI)**: Class codes, rate filings, loss cost data ([NCCI.com](https://www.ncci.com/))
- **State Workers' Compensation Boards**: Benefit schedules, dispute resolution, employer compliance (varies by state)
- **IAIABC (International Association of Industrial Accident Boards and Commissions)**: Research and best practices ([IAIABC.org](https://www.iaiabc.org/))
- **AMA Guides to the Evaluation of Permanent Impairment**: Standard reference for disability ratings

**Further Reading:**  
- *Workers' Compensation Insurance Pricing* (CAS study note): ratemaking, experience rating, retrospective rating plans  
- State-specific statutes: benefit tables, medical fee schedules, administrative procedures
