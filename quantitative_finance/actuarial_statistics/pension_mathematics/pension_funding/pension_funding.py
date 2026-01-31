# Auto-extracted from markdown file
# Source: pension_funding.md

# --- Code Block 1 ---
unfunded = 100000
years = 20
rate = 0.04
contribution = unfunded * rate / (1 - (1 + rate) ** (-years))
print("Annual contribution:", contribution)

