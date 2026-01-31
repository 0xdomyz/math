"""
Extracted from: casino_design_player_behavior.md
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_session_duration(baseline_mins=60, design_effects=None):
    """
    Simulate how design elements extend gambling sessions
    baseline_mins: Intended session length
    design_effects: Dict of design elements and their impact multipliers
    """
    if design_effects is None:
        design_effects = {
            'no_clocks': 1.3,
            'maze_layout': 1.2,
            'free_drinks': 1.4,
            'comfortable_seating': 1.15,
            'near_miss_sounds': 1.25,
        }
    
    # Compound effect of all design elements
    total_multiplier = 1.0
    for effect, multiplier in design_effects.items():
        total_multiplier *= multiplier
    
    actual_duration = baseline_mins * total_multiplier
    
    return actual_duration, total_multiplier

def simulate_spending_with_chips(intended_budget=100, num_sessions=1000):
    """
    Compare spending with cash vs chips
    """
    cash_spending = []
    chip_spending = []
    
    for _ in range(num_sessions):
        # Cash: More pain of payment, stop sooner
        cash_spent = intended_budget * np.random.uniform(0.8, 1.0)
        
        # Chips: Abstract value, spend more
        chip_spent = intended_budget * np.random.uniform(1.0, 1.5)
        
        cash_spending.append(cash_spent)
        chip_spending.append(chip_spent)
    
    return np.array(cash_spending), np.array(chip_spending)

def analyze_near_miss_effect(num_spins=10000):
    """
    Simulate near-miss effect on continued play
    Near-miss: Symbol one position away from win
    """
    outcomes = []
    
    for _ in range(num_spins):
        roll = np.random.random()
        
        if roll < 0.05:
            outcomes.append('win')
        elif roll < 0.20:  # 15% near-miss
            outcomes.append('near_miss')
        else:
            outcomes.append('loss')
    
    return outcomes

# Example 1: Session duration extension
print("=== Casino Design Impact on Session Duration ===\n")

baseline_intention = 60  # 1 hour intended

design_elements = {
    'no_clocks': 1.3,
    'maze_layout': 1.2,
    'free_drinks': 1.4,
    'comfortable_seating': 1.15,
    'near_miss_sounds': 1.25,
}

actual_duration, multiplier = simulate_session_duration(baseline_intention, design_elements)

print(f"Intended session: {baseline_intention} minutes")
print(f"\nDesign elements and their impact:")
for element, mult in design_elements.items():
    print(f"  {element.replace('_', ' ').title()}: +{(mult-1)*100:.0f}%")

print(f"\nCombined multiplier: {multiplier:.2f}x")
print(f"Actual session duration: {actual_duration:.0f} minutes ({actual_duration/60:.1f} hours)")
print(f"Extension: {actual_duration - baseline_intention:.0f} minutes longer than intended")

# Example 2: Chips vs cash spending
print("\n\n=== Chips vs Cash: Spending Comparison ===\n")

np.random.seed(42)

intended_budget = 100
cash, chips = simulate_spending_with_chips(intended_budget, num_sessions=1000)

print(f"Intended budget: ${intended_budget}\n")
print(f"Cash spending:")
print(f"  Average: ${np.mean(cash):.2f}")
print(f"  Median: ${np.median(cash):.2f}")
print(f"  Std Dev: ${np.std(cash):.2f}\n")

print(f"Chip spending:")
print(f"  Average: ${np.mean(chips):.2f}")
print(f"  Median: ${np.median(chips):.2f}")
print(f"  Std Dev: ${np.std(chips):.2f}\n")

overspending = np.mean(chips) - np.mean(cash)
print(f"Overspending with chips: ${overspending:.2f} ({overspending/np.mean(cash)*100:.1f}%)")

# Example 3: Near-miss effect
print("\n\n=== Near-Miss Effect Analysis ===\n")

np.random.seed(42)

outcomes = analyze_near_miss_effect(num_spins=10000)

from collections import Counter
outcome_counts = Counter(outcomes)

print(f"Outcome distribution (10,000 spins):")
print(f"  Wins: {outcome_counts['win']} ({outcome_counts['win']/len(outcomes)*100:.1f}%)")
print(f"  Near-misses: {outcome_counts['near_miss']} ({outcome_counts['near_miss']/len(outcomes)*100:.1f}%)")
print(f"  Losses: {outcome_counts['loss']} ({outcome_counts['loss']/len(outcomes)*100:.1f}%)\n")

print("Psychological impact:")
print("  Near-misses activate same brain regions as wins")
print("  Players perceive 'almost winning' as encouraging")
print("  Continue playing despite actual loss")

# Example 4: Sensory overload impact
print("\n\n=== Sensory Design Elements ===\n")

sensory_impacts = {
    'Winning sounds': {'frequency': 'High', 'impact': 'Reinforces persistence'},
    'Losing sounds': {'frequency': 'Low/None', 'impact': 'Minimizes negative emotion'},
    'Bright colors': {'frequency': 'Constant', 'impact': 'Stimulation, excitement'},
    'Comfortable seating': {'frequency': 'Constant', 'impact': 'Extended play duration'},
    'Free drinks': {'frequency': 'Frequent', 'impact': 'Impaired judgment (+40% risk)'},
    'Ambient scent': {'frequency': 'Constant', 'impact': 'Relaxation, increased spending'},
}

print(f"{'Element':<25} {'Frequency':<15} {'Impact':<40}")
print("-" * 80)

for element, details in sensory_impacts.items():
    print(f"{element:<25} {details['frequency']:<15} {details['impact']:<40}")

# Example 5: Loyalty program manipulation
print("\n\n=== Loyalty Program Psychology ===\n")

print("Tier system (example):")
print("  Bronze: $0-$1,000 spent → 1% cashback")
print("  Silver: $1,000-$5,000 → 2% cashback")
print("  Gold: $5,000-$20,000 → 3% cashback")
print("  Platinum: $20,000+ → 5% cashback\n")

spending_for_gold = 5000
expected_loss_rate = 0.10  # 10% house edge
expected_loss = spending_for_gold * expected_loss_rate
cashback_gold = spending_for_gold * 0.03

print(f"To reach Gold ($5,000 spent):")
print(f"  Expected loss (10% edge): -${expected_loss:.0f}")
print(f"  Cashback (3%): +${cashback_gold:.0f}")
print(f"  Net loss: -${expected_loss - cashback_gold:.0f}\n")

print("Psychological mechanisms:")
print("  Sunk cost fallacy: 'Already spent $4,800, must reach Gold'")
print("  Status seeking: Tier advancement feels like achievement")
print("  Reciprocity bias: Casino 'giving back', feel obligated")
print("  Loss aversion: Don't want to lose tier status")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Session duration extension
elements = list(design_elements.keys())
multipliers = [design_elements[e] for e in elements]
extensions = [(m - 1) * baseline_intention for m in multipliers]

colors_elements = plt.cm.Reds(np.linspace(0.4, 0.9, len(elements)))
axes[0, 0].barh(elements, extensions, color=colors_elements, alpha=0.7)
axes[0, 0].set_xlabel('Session Extension (minutes)')
axes[0, 0].set_title('Impact of Design Elements on Session Duration')
axes[0, 0].grid(alpha=0.3, axis='x')

# Format labels
labels = [e.replace('_', ' ').title() for e in elements]
axes[0, 0].set_yticks(range(len(elements)))
axes[0, 0].set_yticklabels(labels)

# Plot 2: Cash vs chips spending distribution
axes[0, 1].hist(cash, bins=30, alpha=0.6, label='Cash', color='green', density=True)
axes[0, 1].hist(chips, bins=30, alpha=0.6, label='Chips', color='red', density=True)
axes[0, 1].axvline(np.mean(cash), color='darkgreen', linestyle='--', linewidth=2, label=f'Cash mean: ${np.mean(cash):.0f}')
axes[0, 1].axvline(np.mean(chips), color='darkred', linestyle='--', linewidth=2, label=f'Chips mean: ${np.mean(chips):.0f}')
axes[0, 1].set_xlabel('Amount Spent ($)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Cash vs Chips Spending Distribution')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Near-miss vs actual win comparison
outcome_types = ['Wins\n(5%)', 'Near-Misses\n(15%)', 'Losses\n(80%)']
outcome_values = [outcome_counts['win'], outcome_counts['near_miss'], outcome_counts['loss']]
colors_outcomes = ['green', 'yellow', 'red']

axes[1, 0].bar(outcome_types, outcome_values, color=colors_outcomes, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Frequency (out of 10,000 spins)')
axes[1, 0].set_title('Near-Miss Manipulation (Feels Like Winning)')
axes[1, 0].grid(alpha=0.3, axis='y')

for i, val in enumerate(outcome_values):
    axes[1, 0].text(i, val + 100, f'{val:,}', ha='center', fontweight='bold')

# Plot 4: Compounded effect of design elements
cumulative_multiplier = [1.0]
for i, element in enumerate(elements):
    cumulative_multiplier.append(cumulative_multiplier[-1] * multipliers[i])

axes[1, 1].plot(range(len(cumulative_multiplier)), cumulative_multiplier, 'o-', 
               linewidth=2, markersize=8, color='darkred')
axes[1, 1].fill_between(range(len(cumulative_multiplier)), 1, cumulative_multiplier, alpha=0.2, color='red')
axes[1, 1].set_xticks(range(len(cumulative_multiplier)))
axes[1, 1].set_xticklabels(['None'] + labels, rotation=45, ha='right')
axes[1, 1].set_ylabel('Session Duration Multiplier')
axes[1, 1].set_title('Cumulative Impact of Design Elements')
axes[1, 1].axhline(1, color='black', linestyle='--', linewidth=1)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
