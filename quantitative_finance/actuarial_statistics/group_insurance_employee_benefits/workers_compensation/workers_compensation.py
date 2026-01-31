# Auto-extracted from markdown file
# Source: workers_compensation.md

# --- Code Block 1 ---
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

