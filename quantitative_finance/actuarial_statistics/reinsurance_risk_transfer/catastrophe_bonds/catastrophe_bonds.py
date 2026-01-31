# Auto-extracted from markdown file
# Source: catastrophe_bonds.md

# --- Code Block 1 ---
bond_size = 500000000
trigger_loss = 1000000000
expected_loss = 100000000
pricing_spread = 0.04

coupon = bond_size * (expected_loss / trigger_loss + pricing_spread)
print("Annual coupon:", coupon)

