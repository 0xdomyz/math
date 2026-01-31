# Auto-extracted from markdown file
# Source: actuarial_cost_methods.md

# --- Code Block 1 ---
pbos = [50000, 60000, 70000]
salary = [40000, 50000, 60000]
costs = []
for pbo, sal in zip(pbos, salary):
    cost_pct = pbo / sal
    costs.append(cost_pct)
print(costs)

