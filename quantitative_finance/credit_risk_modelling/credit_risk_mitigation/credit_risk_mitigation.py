"""
Credit Risk Mitigation
Extracted from credit_risk_mitigation.md

Analysis of mitigation techniques and capital relief.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 80)
print("CREDIT RISK MITIGATION ANALYSIS")
print("=" * 80)


class MitigationAnalyzer:
    """Analyze mitigation techniques and capital relief"""

    def __init__(self):
        pass

    def calculate_el(self, pd, lgd, ead):
        """Expected Loss = PD × LGD × EAD"""
        return pd * lgd * ead

    def collateral_mitigated_lgd(
        self, lgd_unsecured, collateral_value, ead, haircut=0.30
    ):
        """Calculate LGD with collateral"""
        collateral_effective = collateral_value * (1 - haircut)
        excess_loss = max(0, ead - collateral_effective)
        lgd_mitigated = (excess_loss / ead) * lgd_unsecured
        return lgd_mitigated

    def guarantee_mitigated_pd(self, pd_borrower, pd_guarantor, weight=0.8):
        """Blend PD: weighted average of borrower and guarantor"""
        pd_blended = weight * pd_borrower + (1 - weight) * pd_guarantor
        return pd_blended

    def capital_requirement(self, pd, lgd, ead, rwa_percent=0.08):
        """Regulatory capital = RWA × capital ratio"""
        from scipy.stats import norm

        correlation = 0.15
        confidence = 0.999

        rwa_pct = (
            norm.ppf(pd)
            + np.sqrt(correlation / (1 - correlation)) * norm.ppf(confidence)
        ) / np.sqrt(1 - correlation)
        rwa_pct = norm.cdf(rwa_pct) * lgd * 12.5

        rwa_pct = max(rwa_pct, 0.20)
        rwa_pct = min(rwa_pct, 1.50)

        rwa = ead * rwa_pct
        capital_required = rwa * 0.08

        return {
            "RWA": rwa,
            "RWA_percent": rwa_pct * 100,
            "Capital_Required": capital_required,
        }

    def net_exposure_netting(self, positive_mtm, negative_mtm):
        """Calculate net exposure with bilateral netting"""
        gross_positive = np.sum(positive_mtm)
        net_exposure = max(0, gross_positive - np.sum(np.abs(negative_mtm)))

        eade_reduction = (
            (1 - net_exposure / gross_positive) * 100 if gross_positive > 0 else 0
        )

        return {
            "Gross_Positive_MTM": gross_positive,
            "Net_Exposure": net_exposure,
            "EAD_Reduction_Percent": eade_reduction,
        }

    def collateral_haircuts_regulatory(self):
        """Basel III haircut schedule for collateral"""
        haircuts = {
            "Cash": 0.00,
            "Government Bonds (AAA)": 0.05,
            "Government Bonds (A)": 0.10,
            "Corporate Bonds (AAA)": 0.10,
            "Corporate Bonds (BBB)": 0.15,
            "Equities (Large Cap)": 0.30,
            "Equities (Small Cap)": 0.40,
            "Real Estate": 0.30,
            "Commodities": 0.50,
        }
        return haircuts

    def ltv_analysis(self, loan_amount, collateral_value, haircut=0.30):
        """Loan-to-Value analysis"""
        collateral_effective = collateral_value * (1 - haircut)
        ltv = loan_amount / collateral_effective

        return {
            "Loan_Amount": loan_amount,
            "Collateral_Gross": collateral_value,
            "Collateral_Net": collateral_effective,
            "LTV_Percent": ltv * 100,
            "Comfortable": "Yes" if ltv < 0.8 else "Warning" if ltv < 1.0 else "Risky",
        }

    def securitization_capital_relief(
        self,
        loan_pool_value,
        securitization_size,
        senior_percent=0.80,
        capital_ratio_before=0.10,
    ):
        """Estimate capital relief from securitization"""
        junior_retention = 1 - senior_percent
        junior_size = securitization_size * junior_retention

        rwa_before = loan_pool_value * 1.0
        rwa_after = junior_size * 1.5

        capital_before = rwa_before * capital_ratio_before
        capital_after = rwa_after * capital_ratio_before
        capital_relief = capital_before - capital_after

        return {
            "Loan_Pool_Value": loan_pool_value,
            "Securitization_Size": securitization_size,
            "Senior_Size": securitization_size * senior_percent,
            "Junior_Retained": junior_size,
            "RWA_Before": rwa_before,
            "RWA_After": rwa_after,
            "Capital_Before": capital_before,
            "Capital_After": capital_after,
            "Capital_Relief": capital_relief,
            "Relief_Percent": (
                (capital_relief / capital_before * 100) if capital_before > 0 else 0
            ),
        }

    def cds_premium_analysis(self, pd, lgd, risk_premium=0.001):
        """Estimate CDS premium from PD and LGD"""
        base_premium = pd * lgd * 10000
        total_premium = base_premium + risk_premium * 10000

        return {
            "PD": pd,
            "LGD": lgd,
            "Expected_Loss": pd * lgd,
            "Base_Premium_bps": base_premium,
            "Risk_Premium_bps": risk_premium * 10000,
            "Total_Premium_bps": total_premium,
        }


# Scenario 1: Collateral Mitigation
print("\n" + "=" * 80)
print("SCENARIO 1: COLLATERAL-BASED MITIGATION")
print("=" * 80)

analyzer = MitigationAnalyzer()

loan_amount = 10_000_000
collateral_value = 8_000_000
pd = 0.03
lgd_unsecured = 0.60
haircut = 0.30

el_unsecured = analyzer.calculate_el(pd, lgd_unsecured, loan_amount)
cap_unsecured = analyzer.capital_requirement(pd, lgd_unsecured, loan_amount)

lgd_secured = analyzer.collateral_mitigated_lgd(
    lgd_unsecured, collateral_value, loan_amount, haircut
)
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
print(
    f"  Capital Savings: ${cap_unsecured['Capital_Required'] - cap_secured['Capital_Required']:,.0f}"
)
print(
    f"  Percent Reduction: {(1 - cap_secured['Capital_Required']/cap_unsecured['Capital_Required'])*100:.1f}%"
)

ltv = analyzer.ltv_analysis(loan_amount, collateral_value, haircut)
print(f"\nLoan-to-Value Analysis:")
for key, val in ltv.items():
    if "_" in key:
        if "Percent" in key:
            print(f"  {key}: {val:.1f}%")
        else:
            print(f"  {key}: ${val:,.0f}")
    else:
        print(f"  {key}: {val}")

# Scenario 2: Guarantee Mitigation
print("\n" + "=" * 80)
print("SCENARIO 2: GUARANTEE-BASED MITIGATION")
print("=" * 80)

pd_borrower = 0.05
pd_guarantor = 0.001
lgd = 0.50

el_guarantee_before = analyzer.calculate_el(pd_borrower, lgd, loan_amount)
cap_guarantee_before = analyzer.capital_requirement(pd_borrower, lgd, loan_amount)

pd_blended = analyzer.guarantee_mitigated_pd(pd_borrower, pd_guarantor, weight=0.5)
el_guarantee_after = analyzer.calculate_el(pd_blended, lgd, loan_amount)
cap_guarantee_after = analyzer.capital_requirement(pd_blended, lgd, loan_amount)

guarantee_fee = 0.005 * loan_amount

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
print(
    f"  Capital Savings: ${cap_guarantee_before['Capital_Required'] - cap_guarantee_after['Capital_Required']:,.0f}"
)
print(
    f"  Percent Reduction: {(1 - cap_guarantee_after['Capital_Required']/cap_guarantee_before['Capital_Required'])*100:.1f}%"
)

# Scenario 3: Netting
print("\n" + "=" * 80)
print("SCENARIO 3: BILATERAL NETTING AND CSA COLLATERAL")
print("=" * 80)

np.random.seed(42)
n_trades = 10
positive_mtm = np.random.exponential(500_000, size=n_trades // 2)
negative_mtm = np.random.exponential(400_000, size=n_trades // 2)

netting_result = analyzer.net_exposure_netting(positive_mtm, negative_mtm)

print(f"\nDerivative Portfolio (10 trades with single counterparty):")
print(
    f"  Gross Positive MTM (we'd receive): ${netting_result['Gross_Positive_MTM']:,.0f}"
)
print(f"  Gross Negative MTM (we'd pay): ${np.sum(np.abs(negative_mtm)):,.0f}")
print(f"  Without Netting, EAD would be: ${netting_result['Gross_Positive_MTM']:,.0f}")
print(f"  With Netting, Net EAD is: ${netting_result['Net_Exposure']:,.0f}")
print(f"  EAD Reduction: {netting_result['EAD_Reduction_Percent']:.1f}%")

csa_collateral = 2_000_000
csa_haircut = 0.10
csa_collateral_effective = csa_collateral * (1 - csa_haircut)
csa_protected_exposure = min(netting_result["Net_Exposure"], csa_collateral_effective)
csa_unprotected = max(0, netting_result["Net_Exposure"] - csa_collateral_effective)

print(f"\nCSA Collateral:")
print(f"  Collateral Posted: ${csa_collateral:,.0f}")
print(f"  Haircut: {csa_haircut*100:.0f}%")
print(f"  Effective Collateral: ${csa_collateral_effective:,.0f}")
print(f"  Protected Exposure: ${csa_protected_exposure:,.0f}")
print(f"  Unprotected Exposure: ${csa_unprotected:,.0f}")
print(f"  Total EAD with Netting + CSA: ${csa_unprotected:,.0f}")

# Scenario 4: Securitization
print("\n" + "=" * 80)
print("SCENARIO 4: SECURITIZATION AND CAPITAL RELIEF")
print("=" * 80)

loan_pool_value = 500_000_000
securitization_size = 400_000_000
senior_pct = 0.85

securitization_result = analyzer.securitization_capital_relief(
    loan_pool_value, securitization_size, senior_percent=senior_pct
)

print(f"\nLoan Portfolio:")
print(f"  Total Value: ${securitization_result['Loan_Pool_Value']:,.0f}")
print(f"  Amount to Securitize: ${securitization_result['Securitization_Size']:,.0f}")

print(f"\nSecuritization Structure:")
print(
    f"  Senior Tranche (AAA): ${securitization_result['Senior_Size']:,.0f} ({senior_pct*100:.0f}%)"
)
print(
    f"  Junior Tranche (retained): ${securitization_result['Junior_Retained']:,.0f} ({(1-senior_pct)*100:.0f}%)"
)

print(f"\nCapital Impact:")
print(f"  RWA Before: ${securitization_result['RWA_Before']:,.0f}")
print(f"  RWA After: ${securitization_result['RWA_After']:,.0f}")
print(f"  Capital Before: ${securitization_result['Capital_Before']:,.0f}")
print(f"  Capital After: ${securitization_result['Capital_After']:,.0f}")
print(
    f"  Capital Relief: ${securitization_result['Capital_Relief']:,.0f} ({securitization_result['Relief_Percent']:.1f}%)"
)

print("\n" + "=" * 80)
print("CREDIT RISK MITIGATION SUMMARY")
print("=" * 80)
print("Techniques analyzed: Collateral, Guarantees, Netting, Securitization")
print("All approaches reduce regulatory capital requirements")
print("Effectiveness depends on counterparty quality and market conditions")

if __name__ == "__main__":
    print("\nCredit risk mitigation analysis complete.")
