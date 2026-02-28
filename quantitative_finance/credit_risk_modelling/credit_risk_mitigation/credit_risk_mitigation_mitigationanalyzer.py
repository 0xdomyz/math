import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import warnings

# Auto-extracted from markdown file
# Source: credit_risk_mitigation.md

# --- Code Block 1 ---
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

