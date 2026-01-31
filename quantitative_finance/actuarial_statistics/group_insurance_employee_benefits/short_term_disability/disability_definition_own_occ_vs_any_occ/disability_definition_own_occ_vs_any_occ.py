# Auto-extracted from markdown file
# Source: disability_definition_own_occ_vs_any_occ.md

# --- Code Block 1 ---
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Simulated claim data: occupational factors predicting own-occ approval
np.random.seed(42)

n_claims = 200

data = {
    'Job_Complexity': np.random.randint(1, 10, n_claims),  # 1=simple, 10=complex
    'Physical_Demands': np.random.randint(1, 10, n_claims),  # 1=sedentary, 10=heavy labor
    'Functional_Limitation': np.random.uniform(0, 100, n_claims),  # % capacity loss
    'Transferable_Skills': np.random.randint(0, 100, n_claims),  # % skills applicable elsewhere
    'Age': np.random.randint(25, 65, n_claims),
    'Occupational_Specificity': np.random.randint(1, 10, n_claims),  # 1=generic, 10=highly specialized
}

df = pd.DataFrame(data)

# Outcome: Approved under own-occ definition?
# Rules (illustrative):
# - High job complexity + high limitation + low transferable skills → own-occ approval
# - Low job complexity + moderate limitation → any-occ only
# - High occupational specificity (pilot, surgeon) → own-occ easier

def determine_own_occ_approval(row):
    """Heuristic for own-occ approval"""
    complexity_factor = row['Job_Complexity'] / 10
    limitation_factor = row['Functional_Limitation'] / 100
    specificity_factor = row['Occupational_Specificity'] / 10
    transferability_factor = 1 - (row['Transferable_Skills'] / 100)
    
    score = (complexity_factor * 0.3 + 
             limitation_factor * 0.4 + 
             specificity_factor * 0.2 + 
             transferability_factor * 0.1)
    
    # Add noise for realism
    score += np.random.normal(0, 0.05)
    
    return 1 if score > 0.5 else 0

df['Own_Occ_Approved'] = df.apply(determine_own_occ_approval, axis=1)

print(f"Own-Occ Approval Rate: {df['Own_Occ_Approved'].mean():.1%}")
print(f"Any-Occ Only Rate: {(1 - df['Own_Occ_Approved']).mean():.1%}\n")

# Decision tree
X = df[['Job_Complexity', 'Physical_Demands', 'Functional_Limitation', 
        'Transferable_Skills', 'Occupational_Specificity']]
y = df['Own_Occ_Approved']

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X, y)

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tree.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance for Own-Occ Approval:")
print(importance_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Decision tree
plot_tree(tree, feature_names=X.columns, class_names=['Any-Occ', 'Own-Occ'],
          ax=axes[0, 0], fontsize=9, filled=True)
axes[0, 0].set_title('Decision Tree: Own-Occ Approval')

# Plot 2: Approval by Job Complexity
complexity_bins = pd.cut(df['Job_Complexity'], bins=5)
approval_by_complexity = df.groupby(complexity_bins)['Own_Occ_Approved'].mean()
axes[0, 1].bar(range(len(approval_by_complexity)), approval_by_complexity.values,
               color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Job Complexity (binned)')
axes[0, 1].set_ylabel('Own-Occ Approval Rate')
axes[0, 1].set_title('Approval Rate by Job Complexity')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Approval by Occupational Specificity
spec_bins = pd.cut(df['Occupational_Specificity'], bins=5)
approval_by_spec = df.groupby(spec_bins)['Own_Occ_Approved'].mean()
axes[1, 0].bar(range(len(approval_by_spec)), approval_by_spec.values,
               color='darkgreen', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Occupational Specificity (binned)')
axes[1, 0].set_ylabel('Own-Occ Approval Rate')
axes[1, 0].set_title('Approval Rate by Occupational Specificity')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Feature importance bar chart
axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'],
                color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Importance Score')
axes[1, 1].set_title('Feature Importance for Approval Decision')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity: Approval rate by limitation level
print("\n\nApproval Rate by Functional Limitation Level:")
limitation_bins = pd.cut(df['Functional_Limitation'], bins=[0, 25, 50, 75, 100])
print(df.groupby(limitation_bins)['Own_Occ_Approved'].agg(['count', 'sum', 'mean']))

