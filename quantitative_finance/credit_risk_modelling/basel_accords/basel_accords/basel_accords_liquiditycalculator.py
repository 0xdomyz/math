import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

class LiquidityCalculator:
    """Calculate LCR and NSFR"""
    
    def calculate_lcr(self, hqla_level1, hqla_level2a, hqla_level2b,
                      cash_outflows, cash_inflows):
        """
        Liquidity Coverage Ratio
        LCR = HQLA / Net Cash Outflows (30 days) ≥ 100%
        """
        # HQLA with weights
        hqla = hqla_level1 * 1.0 + hqla_level2a * 0.85 + hqla_level2b * 0.5
        
        # Cap level 2 at 40%
        level2 = hqla_level2a * 0.85 + hqla_level2b * 0.5
        if level2 > 0.4 * hqla:
            # Adjust
            adjustment = level2 - 0.4 * hqla
            hqla -= adjustment
        
        # Net cash outflows (cap inflows at 75% of outflows)
        capped_inflows = min(cash_inflows, 0.75 * cash_outflows)
        net_outflows = cash_outflows - capped_inflows
        
        lcr = hqla / net_outflows if net_outflows > 0 else np.inf
        
        return {
            'hqla': hqla,
            'net_outflows': net_outflows,
            'lcr': lcr,
            'compliant': lcr >= 1.0
        }
    
    def calculate_nsfr(self, asf_components, rsf_components):
        """
        Net Stable Funding Ratio
        NSFR = Available Stable Funding / Required Stable Funding ≥ 100%
        """
        # ASF: {amount: weight} dictionary
        asf = sum(amount * weight for amount, weight in asf_components)
        
        # RSF: {amount: weight} dictionary
        rsf = sum(amount * weight for amount, weight in rsf_components)
        
        nsfr = asf / rsf if rsf > 0 else np.inf
        
        return {
            'asf': asf,
            'rsf': rsf,
            'nsfr': nsfr,
            'compliant': nsfr >= 1.0
        }

# Scenario 1: Basel I vs Basel II Comparison
print("\n" + "="*70)
print("SCENARIO 1: Basel I vs Basel II Comparison")
print("="*70)

calc = BaselCapitalCalculator()

# Portfolio of loans
exposures = [
    {'amount': 100e6, 'rating': 'AAA', 'asset_class': 'corporate'},
    {'amount': 50e6, 'rating': 'BBB', 'asset_class': 'corporate'},
    {'amount': 200e6, 'rating': 'unrated', 'asset_class': 'corporate'},
    {'amount': 150e6, 'rating': None, 'asset_class': 'mortgage'},
    {'amount': 75e6, 'rating': None, 'asset_class': 'retail'},
]

print(f"\n{'Exposure':<15} {'Asset Class':<15} {'Rating':<10} {'Basel I RWA':<15} {'Basel II RWA':<15}")
print("-" * 70)

total_rwa_basel_i = 0
total_rwa_basel_ii = 0

for exp in exposures:
    # Basel I
    rwa_i = calc.basel_i_rwa(exp['amount'], exp['asset_class'])
    
    # Basel II Standardized
    rwa_ii = calc.basel_ii_standardized_rwa(
        exp['amount'], 
        exp.get('rating', 'unrated'), 
        exp['asset_class']
    )
    
    total_rwa_basel_i += rwa_i
    total_rwa_basel_ii += rwa_ii
    
    print(f"${exp['amount']/1e6:<14.0f}M {exp['asset_class']:<15} {str(exp.get('rating', 'N/A')):<10} "
          f"${rwa_i/1e6:<14.0f}M ${rwa_ii/1e6:<14.0f}M")

print(f"\n{'Total RWA:':<40} ${total_rwa_basel_i/1e6:.0f}M ${total_rwa_basel_ii/1e6:.0f}M")

capital_basel_i = total_rwa_basel_i * 0.08
capital_basel_ii = total_rwa_basel_ii * 0.08

print(f"{'Required Capital (8%):':<40} ${capital_basel_i/1e6:.1f}M ${capital_basel_ii/1e6:.1f}M")
print(f"Capital Reduction: {(capital_basel_i - capital_basel_ii)/capital_basel_i*100:.1f}%")

# Scenario 2: IRB vs Standardized
print("\n" + "="*70)
print("SCENARIO 2: Foundation IRB vs Standardized Approach")
print("="*70)

# Single exposure with different PDs
exposure_amount = 100e6
lgd = 0.45  # 45% LGD (F-IRB supervisor value)
maturity = 3.0

print(f"\nExposure: ${exposure_amount/1e6:.0f}M, LGD: {lgd*100:.0f}%, Maturity: {maturity} years")
print(f"\n{'PD':<10} {'Rating':<10} {'F-IRB RWA':<15} {'F-IRB RW%':<12} {'Std RWA':<15} {'Std RW%':<12}")
print("-" * 72)

pds = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
ratings = ['AAA', 'AA', 'A', 'BBB', 'BBB', 'BB', 'B']

for pd, rating in zip(pds, ratings):
    # F-IRB
    rwa_irb, rw_irb = calc.basel_ii_irb_rw(pd, lgd, exposure_amount, maturity)
    
    # Standardized
    rwa_std = calc.basel_ii_standardized_rwa(exposure_amount, rating, 'corporate')
    rw_std = rwa_std / exposure_amount
    
    print(f"{pd*100:<10.2f}% {rating:<10} ${rwa_irb/1e6:<14.1f}M {rw_irb*100:<11.1f}% "
          f"${rwa_std/1e6:<14.1f}M {rw_std*100:<11.1f}%")

print(f"\nF-IRB provides capital relief for low-PD exposures")

# Scenario 3: Basel III Capital Requirements
print("\n" + "="*70)
print("SCENARIO 3: Basel III Capital Requirements & Buffers")
print("="*70)

# Bank portfolio
total_rwa = 500e9  # $500B RWA

# Capital structure
cet1_capital = 45e9  # $45B common equity
at1_capital = 5e9    # $5B AT1 (CoCos)
tier2_capital = 10e9  # $10B Tier 2 debt

tier1_capital = cet1_capital + at1_capital
total_capital = tier1_capital + tier2_capital

ratios = calc.calculate_capital_ratios(total_rwa, cet1_capital, tier1_capital, total_capital)

print(f"\nBank Capital Structure:")
print(f"  CET1: ${cet1_capital/1e9:.1f}B")
print(f"  AT1: ${at1_capital/1e9:.1f}B")
print(f"  Tier 2: ${tier2_capital/1e9:.1f}B")
print(f"  Total Capital: ${total_capital/1e9:.1f}B")
print(f"  RWA: ${total_rwa/1e9:.1f}B")

print(f"\nCapital Ratios:")
print(f"  CET1 Ratio: {ratios['cet1_ratio']*100:.2f}%")
print(f"  Tier 1 Ratio: {ratios['tier1_ratio']*100:.2f}%")
print(f"  Total Capital Ratio: {ratios['total_ratio']*100:.2f}%")

# Check compliance scenarios
scenarios = [
    {'name': 'Non-Systemic Bank', 'ccyb': 0.0, 'gsib': 0.0},
    {'name': 'With Countercyclical Buffer', 'ccyb': 0.025, 'gsib': 0.0},
    {'name': 'G-SIB (Bucket 1)', 'ccyb': 0.01, 'gsib': 0.01},
    {'name': 'G-SIB (Bucket 3)', 'ccyb': 0.01, 'gsib': 0.02},
]

print(f"\n{'Scenario':<30} {'Required CET1':<15} {'Actual':<12} {'Excess':<12} {'Status':<10}")
print("-" * 79)

for scenario in scenarios:
    compliance = calc.check_compliance(
        ratios['cet1_ratio'], 
        scenario['ccyb'], 
        scenario['gsib']
    )
    
    status = "✓ Compliant" if compliance['compliant'] else "✗ BREACH"
    
    print(f"{scenario['name']:<30} {compliance['required_cet1']*100:<14.2f}% "
          f"{compliance['actual_cet1']*100:<11.2f}% {compliance['excess']*100:<11.2f}% {status:<10}")

# Scenario 4: Liquidity Coverage Ratio
print("\n" + "="*70)
print("SCENARIO 4: Liquidity Coverage Ratio (LCR)")
print("="*70)

liq_calc = LiquidityCalculator()

# HQLA
hqla_l1 = 50e9   # Cash, central bank reserves, sovereigns
hqla_l2a = 20e9  # Agency debt, covered bonds
hqla_l2b = 15e9  # Corporate bonds, equities

# Cash flows (30-day stress)
outflows = 100e9  # Deposit runoff, wholesale funding
inflows = 40e9    # Loan repayments

lcr_result = liq_calc.calculate_lcr(hqla_l1, hqla_l2a, hqla_l2b, outflows, inflows)

print(f"\nHigh-Quality Liquid Assets:")
print(f"  Level 1 (100% weight): ${hqla_l1/1e9:.1f}B")
print(f"  Level 2A (85% weight): ${hqla_l2a/1e9:.1f}B")
print(f"  Level 2B (50% weight): ${hqla_l2b/1e9:.1f}B")
print(f"  Total HQLA (weighted): ${lcr_result['hqla']/1e9:.1f}B")

print(f"\n30-Day Stressed Cash Flows:")
print(f"  Expected Outflows: ${outflows/1e9:.1f}B")
print(f"  Expected Inflows: ${inflows/1e9:.1f}B (capped at 75% of outflows)")
print(f"  Net Outflows: ${lcr_result['net_outflows']/1e9:.1f}B")

print(f"\nLCR: {lcr_result['lcr']*100:.1f}%")
print(f"Minimum Required: 100%")
print(f"Status: {'✓ Compliant' if lcr_result['compliant'] else '✗ BREACH'}")

# Scenario 5: Net Stable Funding Ratio
print("\n" + "="*70)
print("SCENARIO 5: Net Stable Funding Ratio (NSFR)")
print("="*70)

# Available Stable Funding (amount, weight)
asf_components = [
    (100e9, 1.0),   # Equity
    (50e9, 1.0),    # Long-term debt (>1 year)
    (200e9, 0.90),  # Stable retail deposits
    (100e9, 0.50),  # Less stable deposits
]

# Required Stable Funding (amount, weight)
rsf_components = [
    (300e9, 0.85),  # Corporate loans
    (100e9, 0.65),  # Retail loans (mortgages)
    (50e9, 0.15),   # HQLA Level 1
    (30e9, 0.50),   # Other liquid assets
]

nsfr_result = liq_calc.calculate_nsfr(asf_components, rsf_components)

print(f"\nAvailable Stable Funding: ${nsfr_result['asf']/1e9:.1f}B")
print(f"Required Stable Funding: ${nsfr_result['rsf']/1e9:.1f}B")
print(f"\nNSFR: {nsfr_result['nsfr']*100:.1f}%")
print(f"Minimum Required: 100%")
print(f"Status: {'✓ Compliant' if nsfr_result['compliant'] else '✗ BREACH'}")

# Scenario 6: IRB Parameter Sensitivity
print("\n" + "="*70)
print("SCENARIO 6: IRB Risk Weight Sensitivity Analysis")
print("="*70)

base_pd = 0.02
base_lgd = 0.45
base_ead = 100e6

print(f"\nBase Case: PD={base_pd*100:.0f}%, LGD={base_lgd*100:.0f}%, EAD=${base_ead/1e6:.0f}M")

# PD sensitivity
print(f"\n{'PD %':<10} {'RWA ($M)':<15} {'RW %':<10} {'Change vs Base':<15}")
print("-" * 50)

base_rwa, base_rw = calc.basel_ii_irb_rw(base_pd, base_lgd, base_ead)

for pd_mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
    pd = base_pd * pd_mult
    rwa, rw = calc.basel_ii_irb_rw(pd, base_lgd, base_ead)
    change = (rwa - base_rwa) / base_rwa * 100 if pd_mult != 1.0 else 0
    
    print(f"{pd*100:<10.1f} ${rwa/1e6:<14.1f} {rw*100:<9.1f} {change:+14.1f}%")

# LGD sensitivity
print(f"\n{'LGD %':<10} {'RWA ($M)':<15} {'RW %':<10} {'Change vs Base':<15}")
print("-" * 50)

for lgd in [0.20, 0.35, 0.45, 0.60, 0.75]:
    rwa, rw = calc.basel_ii_irb_rw(base_pd, lgd, base_ead)
    change = (rwa - base_rwa) / base_rwa * 100 if lgd != base_lgd else 0
    
    print(f"{lgd*100:<10.0f} ${rwa/1e6:<14.1f} {rw*100:<9.1f} {change:+14.1f}%")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Basel I vs II capital comparison
ax = axes[0, 0]
categories = ['Basel I', 'Basel II Std']
capitals = [capital_basel_i/1e6, capital_basel_ii/1e6]

bars = ax.bar(categories, capitals, color=['#d62728', '#2ca02c'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Required Capital ($M)')
ax.set_title('Basel I vs Basel II: Capital Requirements')
ax.grid(alpha=0.3, axis='y')

for bar, capital in zip(bars, capitals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${capital:.1f}M', ha='center', va='bottom')

# Plot 2: IRB vs Standardized by PD
ax = axes[0, 1]
pds_plot = np.linspace(0.001, 0.10, 50)
rws_irb = []
rws_std = []

for pd in pds_plot:
    _, rw = calc.basel_ii_irb_rw(pd, lgd, 100)
    rws_irb.append(rw * 100)
    
    # Approximate standardized mapping
    if pd < 0.01:
        rws_std.append(50)
    elif pd < 0.03:
        rws_std.append(100)
    else:
        rws_std.append(150)

ax.plot(pds_plot * 100, rws_irb, 'b-', linewidth=2.5, label='F-IRB')
ax.plot(pds_plot * 100, rws_std, 'r--', linewidth=2, label='Standardized')
ax.set_xlabel('PD (%)')
ax.set_ylabel('Risk Weight (%)')
ax.set_title('IRB vs Standardized: Risk Sensitivity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Basel III capital structure
ax = axes[0, 2]
components = ['CET1', 'AT1', 'Tier 2']
amounts = [cet1_capital/1e9, at1_capital/1e9, tier2_capital/1e9]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars = ax.bar(components, amounts, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=total_rwa * 0.045 / 1e9, color='r', linestyle='--', linewidth=2, label='Min CET1 (4.5%)')
ax.axhline(y=total_rwa * 0.07 / 1e9, color='orange', linestyle=':', linewidth=2, label='Min CET1 + CCB (7%)')
ax.set_ylabel('Capital ($B)')
ax.set_title('Basel III Capital Structure')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar, amount in zip(bars, amounts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${amount:.1f}B', ha='center', va='bottom')

# Plot 4: Capital requirements by bank type
ax = axes[1, 0]
bank_types = ['Regional', 'Large Bank', 'D-SIB', 'G-SIB (Low)', 'G-SIB (High)']
cet1_requirements = [7.0, 7.0, 8.0, 8.5, 10.0]  # Including buffers

bars = ax.barh(bank_types, cet1_requirements, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=4.5, color='r', linestyle='--', linewidth=1.5, label='Minimum (4.5%)')
ax.set_xlabel('CET1 Requirement (%)')
ax.set_title('Basel III: CET1 by Institution Type')
ax.legend()
ax.grid(alpha=0.3, axis='x')

for bar, req in zip(bars, cet1_requirements):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{req:.1f}%', ha='left', va='center', fontweight='bold')

# Plot 5: LCR components
ax = axes[1, 1]
components_lcr = ['Level 1\nHQLA', 'Level 2A\nHQLA', 'Level 2B\nHQLA', 'Net\nOutflows']
values_lcr = [hqla_l1/1e9, hqla_l2a/1e9 * 0.85, hqla_l2b/1e9 * 0.5, -lcr_result['net_outflows']/1e9]
colors_lcr = ['green', 'yellowgreen', 'gold', 'red']

bars = ax.bar(range(len(components_lcr)), values_lcr, color=colors_lcr, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
ax.set_xticks(range(len(components_lcr)))
ax.set_xticklabels(components_lcr)
ax.set_ylabel('Amount ($B, weighted)')
ax.set_title(f'LCR Components (LCR={lcr_result["lcr"]*100:.0f}%)')
ax.grid(alpha=0.3, axis='y')

# Plot 6: IRB RW sensitivity to PD and LGD
ax = axes[1, 2]
pds_grid = np.linspace(0.005, 0.10, 20)
lgds_grid = [0.20, 0.45, 0.75]

for lgd_val in lgds_grid:
    rws = []
    for pd in pds_grid:
        _, rw = calc.basel_ii_irb_rw(pd, lgd_val, 100)
        rws.append(rw * 100)
    
    ax.plot(pds_grid * 100, rws, linewidth=2.5, marker='o', markersize=4, label=f'LGD={lgd_val*100:.0f}%')

ax.set_xlabel('PD (%)')
ax.set_ylabel('Risk Weight (%)')
ax.set_title('IRB Risk Weight Sensitivity')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

