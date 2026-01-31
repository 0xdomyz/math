# Disability Definition: Own Occupation vs Any Occupation

## 1. Concept Skeleton
**Definition:** Criteria determining whether employee qualifies as "disabled" under STD policy; own occupation (can't do current job) vs any occupation (can't work any job)  
**Purpose:** Balance coverage adequacy with moral hazard; own occupation more generous, any occupation restrictive  
**Prerequisites:** Disability insurance fundamentals, functional capacity evaluation, occupational analysis

## 2. Comparative Framing
| Aspect | **Own Occupation (Own Occ)** | **Any Occupation** |
|--------|-------|--------|
| **Definition** | Unable to perform duties of regular job | Unable to work in *any* occupation |
| **Claimant burden** | Lower (prove can't do current role) | Higher (prove can't work anywhere) |
| **Insurer burden** | Higher (more claims approved) | Lower (stricter standard) |
| **Market use** | STD standard (first 2–3 years) | LTD common (after transition period) |
| **Premium impact** | Higher | Lower |
| **Return-to-work outcome** | Better (easier alternate work) | Worse (must be severely limited) |

## 3. Examples + Counterexamples

**Own Occupation Example:**  
Surgeon with severe arthritis: Cannot perform surgery (own occupation) → qualifies for STD benefits, despite ability to do office work or consulting

**Any Occupation Example:**  
Same surgeon: Under "any occupation" definition, if capable of sedentary office work → no benefits (fails test). Must prove unable to work in ANY capacity

**Transition Scenario:**  
STD covers first 24 months under own occupation. LTD begins; definition shifts to any occupation. Surgeon may lose LTD benefits if deemed capable of desk work (even lower-paid role)

**Edge Case:**  
Specialized high-income job (professional athlete, airline pilot): Own occupation protects lost earnings; any occupation test impossible to meet (too specific)

## 4. Layer Breakdown
```
Disability Definition Framework:
├─ Own Occupation (Own Occ) Definition:
│   ├─ Standard: "Unable to engage in the material duties of regular occupation"
│   ├─ Material Duties: Key job functions requiring significant time/effort
│   ├─ Reasonableness Test: Considering education, experience, training, mobility
│   ├─ Medical Evidence: Attending physician statement + functional capacity exam
│   └─ Approval threshold: Low (objective showing of occupational unfitness)
│
├─ Any Occupation (Any Occ) Definition:
│   ├─ Standard: "Unable to engage in any occupation for which reasonably suited"
│   ├─ Reasonableness Filter: Education, training, experience relevant
│   ├─ Market Test: Must consider available jobs in labor market
│   ├─ Earnings Test: Sometimes tied to % income loss (e.g., 20%+ loss triggers)
│   └─ Approval threshold: High (nearly total disability required)
│
├─ Hybrid Definitions:
│   ├─ "Own-Occ with Earnings Test": Own occ BUT reduced benefit if working
│   ├─ "Transitional Own-Occ": Own occ first 24 months, shifts to any occ
│   ├─ "Graded Any-Occ": Income-based; benefit decreases as earnings increase
│   └─ Goal: Prevent total loss of income while encouraging work
│
├─ Medical Evaluation Process:
│   ├─ Attending Physician Statement: Diagnosis, prognosis, treatment, functional limits
│   ├─ Functional Capacity Evaluation (FCE): Standardized testing of lifting, mobility, stamina
│   ├─ Occupational Analysis: Description of claimant's regular duties
│   ├─ Independent Medical Exam (IME): Insurer-retained physician review
│   └─ Integration: Compare medical findings to job demands
│
├─ Special Considerations:
│   ├─ Mental/Nervous Disorders: Often limited (24–36 months max benefit)
│   ├─ Subjective Conditions (pain): Higher burden of proof; may require objective imaging
│   ├─ Progressive Conditions: Own occ may shift over time as condition worsens
│   └─ Vocational Rehabilitation: Insurer provides training; impacts any-occ determination
│
└─ Burden of Proof:
    ├─ Claimant burden: Overcome benefit presumption with medical evidence
    ├─ Insurer burden: Prove claimant capable of occupation (any-occ test)
    └─ Standard: Preponderance of evidence (more likely than not)
```

## 5. Mini-Project: Own-Occ vs Any-Occ Decision Model

**Goal:** Build decision tree for occupation-based disability determinations.

```python
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
```

**Key Insights:**
- Own-occ definition: Focuses on occupational demands vs claimant's capabilities
- Any-occ definition: Requires near-total disability; rarely approved unless severe
- Occupational specificity crucial: Highly specialized jobs (pilots, surgeons) → own-occ approved more easily
- Transitional definitions common: 24-month own-occ, then shift to any-occ (aligns with SSDI)

## 6. Relationships & Dependencies
- **To Claim Adjudication:** Definition determines approval/denial; highest-impact factor
- **To Benefit Duration:** Own-occ → longer claims; any-occ → shorter (higher approval bar)
- **To Offsets:** SSDI "any occ" test aligns with LTD definition; simplifies offset coordination
- **To Rehabilitation:** Vocational rehab impacts any-occ determination (improved capacity → benefits cut)

## 7. Regulatory & Legal Context
- **Contract Language:** Definition codified in policy; amendments require full re-evaluation
- **State Regulation:** Some states limit any-occ use in STD (require own-occ for XX months)
- **Case Law:** Courts sometimes expand definition if claimant presents compelling evidence
- **SSDI Alignment:** LTD often shifts to SSDI's "any occ" after 24 months for seamless transition

## References
- [Council of Insurance Agents & Brokers (CIAB)](https://www.ciab.com) - Group DI best practices
- [American Council of Life Insurers (ACLI)](https://www.acli.com) - Disability insurance standards
- [Actuarial Standards Board: ASB #1](https://www.actuarialstandardsboard.org/) - Valuation and claim evaluation

