
# Block 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define exotic types and characteristics
exotics_data = {
    'Option': [
        'Down-and-Out Call', 'Up-and-Out Put', 'Down-and-In Call',
        'Asian Arithmetic Call', 'Asian Geometric Call', 'Lookback Call',
        'Basket Call (2 assets)', 'Chooser Option', 'Cliquet Call',
        'Quanto Call', 'Swing Option', 'Binary One-Touch'
    ],
    'Path-Dependent': [
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
        'Weakly', 'No', 'Weakly', 'Weakly', 'Yes', 'Yes'
    ],
    'Pricing Difficulty': [
        'Medium', 'Medium', 'Medium', 'High', 'Medium', 'High',
        'High', 'Low', 'Low', 'Low', 'High', 'Medium'
    ],
    'Best Method': [
        'Lattice', 'Lattice', 'Lattice', 'MC/FD', 'Closed-form', 'MC',
        'MC', 'Closed-form', 'Closed-form', 'Closed-form', 'MC/FD', 'Lattice'
    ],
    'Primary Use': [
        'Cost reduction', 'Cost reduction', 'Leverage', 'Averaging hedge',
        'Averaging (math)', 'Best-price capture', 'Multi-asset exposure',
        'Optionality timing', 'Periodic coupons', 'Currency hedging',
        'Energy trading', 'Digital payoff'
    ]
}

df = pd.DataFrame(exotics_data)

print("\n" + "="*100)
print("EXOTIC OPTIONS: CLASSIFICATION & CHARACTERISTICS")
print("="*100)
print(df.to_string(index=False))

# Complexity matrix
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

pricing_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['Pricing_Score'] = df['Pricing Difficulty'].map(pricing_map)
path_dep_map = {'No': 0, 'Weakly': 1, 'Yes': 2}
df['Path_Score'] = df['Path-Dependent'].map(path_dep_map)

colors = {'Lattice': 'blue', 'MC': 'red', 'MC/FD': 'purple', 
          'Closed-form': 'green', 'Lattice/MC': 'orange', 'FD': 'brown'}
color_list = [colors.get(m, 'gray') for m in df['Best Method']]

scatter = ax.scatter(df['Path_Score'], df['Pricing_Score'], s=300, 
                     c=color_list, alpha=0.6, edgecolors='black', linewidth=1.5)

for idx, row in df.iterrows():
    ax.annotate(row['Option'], 
               (row['Path_Score'], row['Pricing_Score']),
               fontsize=8, ha='center', va='center', fontweight='bold')

ax.set_xlabel('Path Dependence (0=None, 1=Weak, 2=Strong)', fontsize=11, fontweight='bold')
ax.set_ylabel('Pricing Difficulty (1=Low, 2=Med, 3=High)', fontsize=11, fontweight='bold')
ax.set_title('Exotic Options: Complexity & Valuation Method', fontsize=13, fontweight='bold')
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(0.7, 3.3)
ax.grid(alpha=0.3)
ax.set_xticks([0, 1, 2])
ax.set_yticks([1, 2, 3])

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='black', label=method) 
                  for method, color in colors.items()]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*50)
print("COMPLEXITY DISTRIBUTION:")
print("="*50)
print("\nBy Pricing Difficulty:")
print(df['Pricing Difficulty'].value_counts().to_string())
print("\nBy Path Dependence:")
print(df['Path-Dependent'].value_counts().to_string())
print("\nBy Recommended Method:")
print(df['Best Method'].value_counts().to_string())