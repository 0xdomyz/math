# Auto-extracted from markdown file
# Source: commutation_retroactive_covers.md

# --- Code Block 1 ---
reserves = 500000
discount_rate = 0.05
settlement_discount = 0.10

commutation_value = reserves * (1 - settlement_discount) / (1 + discount_rate)
print("Commutation value:", commutation_value)

