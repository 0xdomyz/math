# Medical/Health Insurance

## Concept Skeleton

Medical (health) insurance covers costs of medical care, including hospitalization, physician services, prescription drugs, and preventive care, under diverse delivery models (fee-for-service, HMO, PPO) with actuarial pricing driven by claim frequency, severity, and medical cost trend. Group health plans leverage employer sponsorship and regulatory frameworks (ACA, ERISA) to spread risk, manage adverse selection, and provide tax-advantaged coverage to employees.

**Core Components:**
- **Coverage tiers**: Individual, employee+spouse, employee+children, family
- **Cost-sharing**: Deductibles, coinsurance, copayments, out-of-pocket maximums
- **Network design**: HMO (gatekeeper model), PPO (preferred provider), EPO (exclusive provider), POS (point-of-service)
- **Essential health benefits (ACA)**: Minimum mandated coverage categories (hospitalization, maternity, mental health, etc.)
- **Medical trend**: Annual increase in per-capita healthcare costs (typically 5–8%) due to utilization and unit cost inflation

**Why it matters:** Largest component of employee benefits cost; actuarial pricing balances affordability with comprehensive coverage, managing financial risk for insurers and employers while ensuring access to care.

---

## Comparative Framing

| Dimension | **Medical/Health Insurance** | **Short-Term Disability** | **Workers' Compensation** |
|-----------|------------------------------|---------------------------|---------------------------|
| **Coverage scope** | Medical services, Rx, preventive care | Income replacement during illness | Medical + indemnity for job injuries |
| **Cost drivers** | Frequency × severity × trend | Claim incidence × duration | Injury rate × medical/indemnity costs |
| **Regulatory framework** | ACA, HIPAA, ERISA | State insurance regs, ERISA | State-mandated, statutory benefits |
| **Network/provider control** | HMO/PPO networks | N/A (income benefit) | Employer-directed care or state networks |
| **Pricing basis** | PMPM (per member per month) | % of salary | % of payroll by occupation class |
| **Claim variability** | High (infrequent large claims + frequent small) | Moderate (defined benefit periods) | Moderate to high (catastrophic claims) |

**Key insight:** Health insurance complexity arises from heterogeneous services (primary care to organ transplants), unpredictable utilization, and rapid medical cost inflation; disability and workers' comp have more bounded benefit structures.

---

## Examples & Counterexamples

### Examples of Medical/Health Insurance

1. **PPO plan with tiered cost-sharing**  
   - Annual deductible: $1,500 individual / $3,000 family  
   - In-network coinsurance: 80% (insurer) / 20% (member)  
   - Out-of-network: 60% / 40%  
   - Out-of-pocket max: $6,000 individual / $12,000 family  
   - Preventive care: $0 copay (ACA mandate)

2. **HMO with capitated primary care**  
   - Member selects PCP (primary care physician) as gatekeeper  
   - PCP receives fixed PMPM payment regardless of services rendered  
   - Specialist referrals required; no out-of-network coverage except emergencies  
   - Lower premiums, tighter cost control vs. PPO

3. **High-deductible health plan (HDHP) with HSA**  
   - $3,000 individual deductible (2024 minimum for HSA eligibility)  
   - After deductible, 100% coverage (no coinsurance)  
   - Paired with Health Savings Account (tax-advantaged contributions)  
   - Encourages consumer cost-consciousness

### Non-Examples (or Edge Cases)

- **Dental or vision insurance**: Separate product lines; not typically bundled with major medical (though ACA pediatric dental is essential benefit).
- **Critical illness or cancer insurance**: Supplemental indemnity products; pay lump sum on diagnosis, not reimbursement for services.
- **Medicaid or Medicare**: Government-funded programs; not employer-sponsored group insurance (though Medicare Advantage is insurer-administered).

---

## Layer Breakdown

**Layer 1: Benefit Design and Cost-Sharing**  
Actuaries model tradeoffs: lower deductibles increase premiums but reduce member cost barriers; higher coinsurance shifts risk to members. Out-of-pocket maximums cap member exposure per ACA. Tiered networks (in-network vs. out-of-network) steer utilization to contracted providers with negotiated discounts.

**Layer 2: Claim Cost Estimation (PMPM Framework)**  
Premium rates calculated as PMPM (per member per month):  
\[
\text{PMPM} = (\text{Frequency} \times \text{Severity}) + \text{Admin Load} + \text{Profit/Risk Margin}
\]
- **Frequency**: Claims per 1,000 members per year (e.g., 120 inpatient admits per 1,000)  
- **Severity**: Average cost per claim (e.g., $15,000 per admit)  
- **Trend**: Annual medical cost inflation (adjust prior year PMPM by trend factor)

**Layer 3: Experience Rating and Risk Adjustment**  
Large groups (>100 employees): insurer uses group's own claim history, adjusted for demographics and industry.  
Small groups (<100): community rating or adjusted community rating (ACA rules).  
Risk adjustment: High-cost claimants (e.g., cancer, transplants) may trigger reinsurance or risk corridors.

**Layer 4: Regulatory Compliance**  
- **ACA**: Essential health benefits, no pre-existing condition exclusions, 80/85% medical loss ratio (MLR) rebates  
- **HIPAA**: Privacy (PHI protection) and portability (no gaps in coverage for pre-ex)  
- **ERISA**: Fiduciary duties, claims appeal rights, summary plan descriptions

---

## Mini-Project: PMPM Premium Calculation with Trend

**Goal:** Calculate renewal premium for a 200-member group, adjusting for medical trend and demographic changes.

```python
import numpy as np

# Prior year group experience
prior_year_claims = 1_800_000  # Total paid claims
prior_year_members = 200
prior_year_pmpm = prior_year_claims / (prior_year_members * 12)

# Medical trend and demographic adjustments
medical_trend = 0.07  # 7% annual increase
demographic_factor = 1.02  # Aging workforce, 2% increase in expected costs
admin_load = 0.15  # 15% for admin and profit

# Calculate expected PMPM for renewal year
expected_claims_pmpm = prior_year_pmpm * (1 + medical_trend) * demographic_factor
premium_pmpm = expected_claims_pmpm / (1 - admin_load)

print(f"Prior Year Claims: ${prior_year_claims:,}")
print(f"Prior Year Members: {prior_year_members}")
print(f"Prior Year PMPM (claims only): ${prior_year_pmpm:,.2f}")
print(f"Medical Trend: {medical_trend:.1%}")
print(f"Demographic Factor: {demographic_factor:.2f}")
print(f"Expected Claims PMPM (renewal): ${expected_claims_pmpm:,.2f}")
print(f"Admin Load: {admin_load:.1%}")
print(f"Renewal Premium PMPM: ${premium_pmpm:,.2f}")
print(f"Annual Premium per Member: ${premium_pmpm * 12:,.2f}")
print(f"Total Group Annual Premium (200 members): ${premium_pmpm * 12 * 200:,.2f}")

# Sensitivity: reduce trend to 5% (e.g., improved utilization management)
expected_claims_pmpm_low = prior_year_pmpm * 1.05 * demographic_factor
premium_pmpm_low = expected_claims_pmpm_low / (1 - admin_load)
print(f"\nLow-Trend Scenario (5%):")
print(f"Renewal Premium PMPM: ${premium_pmpm_low:,.2f}")
print(f"Savings per member per year: ${(premium_pmpm - premium_pmpm_low) * 12:,.2f}")
```

**Expected Output (illustrative):**
```
Prior Year Claims: $1,800,000
Prior Year Members: 200
Prior Year PMPM (claims only): $750.00
Medical Trend: 7.0%
Demographic Factor: 1.02
Expected Claims PMPM (renewal): $818.25
Admin Load: 15.0%
Renewal Premium PMPM: $962.65
Annual Premium per Member: $11,551.76
Total Group Annual Premium (200 members): $2,310,352.94

Low-Trend Scenario (5%):
Renewal Premium PMPM: $916.18
Savings per member per year: $557.59
```

**Interpretation:**  
- Medical trend is dominant cost driver; 2% reduction in trend saves significant premium.  
- Demographic changes (aging, industry risk) compound trend effects.  
- Admin load covers insurer overhead, underwriting profit, risk charges.

---

## Challenge Round

1. **MLR Rebates (ACA)**  
   An insurer collects $10 million in premiums for a small-group plan, pays $7.5 million in claims. ACA requires 80% MLR for small groups. Calculate rebate owed.

   <details><summary>Solution</summary>
   **Required claims:** $10M × 0.80 = $8M.  
   **Actual claims:** $7.5M.  
   **Shortfall:** $8M - $7.5M = $0.5M.  
   **Rebate:** Insurer must refund $0.5M to policyholders (typically as premium credits).
   </details>

2. **Stop-Loss Reinsurance**  
   A self-insured employer purchases specific stop-loss at $150,000 per claimant. One employee incurs $500,000 in claims. How much does the employer pay?

   <details><summary>Solution</summary>
   **Employer pays:** $150,000 (up to stop-loss attachment point).  
   **Reinsurer pays:** $500,000 - $150,000 = $350,000.  
   Protects employer from catastrophic individual claims.
   </details>

3. **Claim Cost Distribution**  
   In a 1,000-member group, 10 members (1%) account for 50% of total claims. What does this imply for risk management?

   <details><summary>Hint</summary>Health insurance exhibits extreme skewness: small fraction of high-cost claimants drive aggregate costs. Insurers use reinsurance, large-claim pooling, and predictive modeling to manage volatility. Risk adjustment and premium stabilization mechanisms (e.g., ACA risk corridors) mitigate insurer exposure.</details>

4. **Preventive Care ROI**  
   Insurer spends $50 PMPM on preventive care programs (screenings, vaccinations, wellness). Assume 5% reduction in downstream hospitalization costs ($200 PMPM baseline). Calculate net savings.

   <details><summary>Solution</summary>
   **Hospitalization savings:** $200 × 0.05 = $10 PMPM.  
   **Preventive care cost:** $50 PMPM.  
   **Net cost:** $50 - $10 = $40 PMPM *increase*.  
   ROI is negative in short term; long-term benefits (chronic disease prevention) may justify investment but require multi-year horizon and population health perspective.
   </details>

---

## Key References

- **Society of Actuaries (SOA)**: Health care cost trend surveys and group health pricing ([SOA.org](https://www.soa.org/))
- **Affordable Care Act (ACA)**: Essential health benefits, MLR, rating rules ([HealthCare.gov](https://www.healthcare.gov/), [CMS.gov](https://www.cms.gov/))
- **NAIC**: Model laws for health insurance rating and consumer protections ([NAIC.org](https://www.naic.org/))
- **ERISA**: Plan administration and fiduciary duties for self-insured plans ([DOL.gov](https://www.dol.gov/agencies/ebsa))

**Further Reading:**  
- *Health Insurance* by Michael Morrisey (textbook on actuarial principles and policy)  
- CMS National Health Expenditure Accounts (data on spending trends)  
- HCCI (Health Care Cost Institute) reports on commercial insurance claim patterns
