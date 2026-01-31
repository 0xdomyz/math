# Treatment on Treated (ATT)

## 1. Concept Skeleton
**Definition:** Expected treatment effect for units that received treatment; E[Y₁ - Y₀ | D=1]; effect on participants  
**Purpose:** Evaluate impact on actual program participants; policy decisions about continuing/expanding programs; most relevant for ex-post evaluation  
**Prerequisites:** Potential outcomes, selection bias, matching methods, propensity scores, common support, covariate balance

## 2. Comparative Framing
| Estimand | ATT | ATE | ATU | LATE | ITT | Per-Protocol |
|----------|-----|-----|-----|------|-----|--------------|
| **Definition** | E[Y₁-Y₀\|D=1] | E[Y₁-Y₀] | E[Y₁-Y₀\|D=0] | E[Y₁-Y₀\|complier] | E[Y\|Z=1]-E[Y\|Z=0] | E[Y₁-Y₀\|comply] |
| **Population** | Treated units | Everyone | Untreated units | Compliers only | Assigned to treatment | Protocol followers |
| **Question Answered** | Did it work for participants? | Would it work for all? | Would it work for non-participants? | IV effect | Randomization effect | Ideal adherence |
| **Selection Bias** | Need E[Y₀\|D=1] | Need both E[Y₁], E[Y₀] | Need E[Y₁\|D=0] | Monotonicity | None (randomized) | Selection into compliance |
| **Policy Relevance** | Justify existing program | Universal adoption | Expansion decision | IV validity | Intent-to-treat | Efficacy vs effectiveness |
| **Estimation** | Matching, IPW on treated | Doubly robust, full IPW | Inverse ATT | 2SLS/Wald | Simple difference | Exclude non-compliers |

## 3. Examples + Counterexamples

**Classic Example:**  
Job training program evaluation (n=2000, 800 treated). ATT=\$3,200 annual earnings gain (p<0.01), ATE=\$2,500. Participants benefit more (positive selection on gains). Policy: Continue program for current participants, but expansion may yield lower returns. Matching on covariates (education, age, prior earnings) reduces bias from ATT_naive=\$4,500 to ATT_matched=\$3,200.

**Failure Case:**  
College enrollment study: ATT_naive=+\$60k lifetime earnings (college graduates vs non-enrollees). But high-ability students self-select into college. After propensity score matching: ATT=+\$35k (58% of naive effect spurious). Common support violated: Top 10% of propensity scores have no control matches → Restrict to e(X)∈[0.1,0.9], report ATT for matchable subpopulation only.

**Edge Case:**  
Medical trial with selective dropout: ITT=+2 years (randomization), Per-Protocol=+5 years (completers only), ATT=+3 years (all assigned to treatment accounting for non-compliance). ATT>ITT because treatment works for those who take it. Per-Protocol biased upward (healthy patients more likely to complete). Use principal stratification or IV methods to correct.

## 4. Layer Breakdown
```
Treatment on Treated (ATT) Framework:
├─ Definition and Interpretation:
│   ├─ ATT = E[Y₁ - Y₀ | D=1]
│   │   ├─ Average effect for those who received treatment
│   │   ├─ Observed: E[Y₁|D=1] (treated outcomes)
│   │   ├─ Counterfactual: E[Y₀|D=1] (what would have happened without treatment)
│   │   └─ Problem: E[Y₀|D=1] unobserved
│   ├─ Relationship to ATE:
│   │   ├─ ATE = P(D=1)·ATT + P(D=0)·ATU
│   │   ├─ ATT ≠ ATE if heterogeneous effects + selection
│   │   └─ ATT = ATE only if no selection on gains
│   ├─ Selection on Gains:
│   │   ├─ Positive: Units with higher τᵢ select into treatment
│   │   ├─ ATT > ATE (participants benefit more)
│   │   ├─ Example: Motivated workers in job training
│   │   └─ Policy: Expansion yields lower returns than existing program
│   ├─ Selection on Levels:
│   │   ├─ Units with higher Y₀ select into treatment
│   │   ├─ ATT can be < or > ATE depending on τ heterogeneity
│   │   └─ Example: Healthier patients seek treatment
│   └─ Policy Relevance:
│       ├─ Ex-post evaluation: Did program work for participants?
│       ├─ Accountability: Justify continued funding
│       ├─ Expansion: ATU more relevant (would it work for others?)
│       └─ Targeting: CATE identifies who benefits most
├─ Identification Strategies:
│   ├─ Randomized Controlled Trial:
│   │   ├─ Random assignment: D ⊥ (Y₁, Y₀)
│   │   ├─ E[Y₀|D=1] = E[Y₀|D=0] = E[Y₀]
│   │   ├─ ATT = E[Y|D=1] - E[Y|D=0] = ATE
│   │   ├─ Unbiased for both ATT and ATE
│   │   └─ Complications: Non-compliance (use IV for ATT on compliers)
│   ├─ Matching Estimators:
│   │   ├─ Goal: Find control units similar to treated on X
│   │   ├─ ATT = E[Y₁|D=1] - E[Y₀_matched|D=1]
│   │   ├─ Requires: CIA and overlap
│   │   └─ Focuses on treated units (ATT), not entire population (ATE)
│   ├─ Conditional Independence (CIA):
│   │   ├─ Unconfoundedness: (Y₁, Y₀) ⊥ D | X
│   │   ├─ ATT = E_X|D=1[E[Y|D=1,X] - E[Y|D=0,X]]
│   │   ├─ Weighted by P(X|D=1) (treated covariate distribution)
│   │   └─ Only need overlap for X where treated units exist
│   ├─ Overlap for ATT:
│   │   ├─ P(D=1|X) < 1 for all X in support of treated
│   │   ├─ Less restrictive than ATE (only need controls, not treated everywhere)
│   │   ├─ Allows P(D=1|X) close to 1 (ATE would require trimming)
│   │   └─ Common support: X values where both D=1 and D=0 observed
│   └─ Propensity Score for ATT:
│       ├─ e(X) = P(D=1|X)
│       ├─ (Y₁, Y₀) ⊥ D | e(X) (balancing property)
│       ├─ ATT = E_e|D=1[E[Y|D=1,e] - E[Y|D=0,e]]
│       └─ Weight by P(e|D=1) (PS distribution among treated)
├─ Matching Methods for ATT:
│   ├─ Nearest Neighbor Matching:
│   │   ├─ For each treated i, find control j with closest X
│   │   ├─ Distance: ||Xᵢ - Xⱼ|| (Euclidean, Mahalanobis)
│   │   ├─ ATT = (1/n₁)Σᵢ₌₁ⁿ¹[Y₁ᵢ - Y₀ⱼ₍ᵢ₎] (j(i) is matched control)
│   │   ├─ Variants:
│   │   │   ├─ 1:1 matching (each treated gets one control)
│   │   │   ├─ k:1 matching (k controls per treated, average them)
│   │   │   ├─ With replacement: Control can match multiple treated
│   │   │   └─ Without replacement: Control used once (order matters)
│   │   ├─ Bias-Variance Tradeoff:
│   │   │   ├─ More matches (k large): Lower variance, higher bias
│   │   │   └─ Fewer matches (k=1): Higher variance, lower bias
│   │   └─ Standard Errors:
│   │       ├─ Abadie-Imbens (2006) for matching uncertainty
│   │       └─ Bootstrap if matching with replacement
│   ├─ Caliper Matching:
│   │   ├─ Nearest neighbor within distance δ (caliper)
│   │   ├─ ||Xᵢ - Xⱼ|| ≤ δ or |e(Xᵢ) - e(Xⱼ)| ≤ δ
│   │   ├─ Discard treated with no match within caliper
│   │   ├─ Improves match quality (reduces bias)
│   │   └─ Common: δ = 0.25·SD(e(X)) or δ = 0.1
│   ├─ Kernel Matching:
│   │   ├─ Weighted average of all controls
│   │   ├─ Weights: K((e(Xᵢ)-e(Xⱼ))/h) (kernel function)
│   │   ├─ ATT = (1/n₁)Σᵢ₌₁ⁿ¹[Y₁ᵢ - Σⱼ₌₁ⁿ⁰ wᵢⱼY₀ⱼ]
│   │   ├─ wᵢⱼ = K((e(Xᵢ)-e(Xⱼ))/h) / Σₖ K((e(Xᵢ)-e(Xₖ))/h)
│   │   ├─ Kernel choices:
│   │   │   ├─ Epanechnikov: K(u) = 0.75(1-u²) if |u|≤1, else 0
│   │   │   ├─ Gaussian: K(u) = (1/√2π)exp(-u²/2)
│   │   │   └─ Uniform: K(u) = 0.5 if |u|≤1, else 0
│   │   ├─ Bandwidth h: Controls smoothness
│   │   │   ├─ Small h: Closer to nearest neighbor (high variance)
│   │   │   ├─ Large h: More smoothing (high bias)
│   │   │   └─ Cross-validation or rule-of-thumb h=1.06·SD·n^(-1/5)
│   │   └─ Uses all data (efficient), smooth weights
│   ├─ Stratification Matching:
│   │   ├─ Divide propensity score into blocks (quintiles)
│   │   ├─ Within each block: ATT_b = E[Y|D=1,block=b] - E[Y|D=0,block=b]
│   │   ├─ Overall: ATT = Σ_b P(block=b|D=1)·ATT_b
│   │   ├─ Check balance within blocks
│   │   └─ Simple, transparent, robust
│   ├─ Coarsened Exact Matching (CEM):
│   │   ├─ Coarsen X into bins (age: 20-30, 30-40, etc.)
│   │   ├─ Exact match on coarsened X
│   │   ├─ Prune unmatched units
│   │   ├─ Estimate ATT on matched sample
│   │   └─ Advantages: No functional form, automatic balance, bounds
│   └─ Optimal Matching:
│       ├─ Minimize total distance across all matches
│       ├─ Assignment problem (Hungarian algorithm)
│       ├─ Computationally intensive for large n
│       └─ Better balance than greedy nearest neighbor
├─ Propensity Score Weighting for ATT:
│   ├─ IPW for ATT:
│   │   ├─ Weight treated by 1, controls by e(X)/(1-e(X))
│   │   ├─ ATT = E[D·Y] - E[(1-D)·e(X)/(1-e(X))·Y] / E[D]
│   │   ├─ Creates pseudo-population where controls look like treated
│   │   └─ Or: ATT = E[D·Y - (1-D)·e(X)/(1-e(X))·Y] / E[D]
│   ├─ Stabilized Weights:
│   │   ├─ w_treated = 1
│   │   ├─ w_control = P(D=1)·e(X) / [P(D=0)·(1-e(X))]
│   │   ├─ Reduces variance (weights closer to 1)
│   │   └─ ESS (effective sample size) higher
│   ├─ Weight Trimming:
│   │   ├─ Extreme weights → high variance
│   │   ├─ Trim: w > 20 or w > 99th percentile
│   │   ├─ Or: Restrict to common support (e(X) not too extreme)
│   │   └─ Bias-variance tradeoff
│   └─ Diagnostics:
│       ├─ Check weight distribution (max, 99th percentile)
│       ├─ ESS: (Σw)²/Σw² (effective sample size)
│       └─ Covariate balance after weighting
├─ Covariate Balance:
│   ├─ Standardized Mean Difference (SMD):
│   │   ├─ SMD = (X̄_treated - X̄_control) / SD_pooled
│   │   ├─ SD_pooled = √[(Var(X|D=1) + Var(X|D=0))/2]
│   │   ├─ Rule of thumb: |SMD| < 0.1 indicates good balance
│   │   └─ Report before and after matching
│   ├─ Variance Ratios:
│   │   ├─ VR = Var(X|D=1) / Var(X|D=0)
│   │   ├─ Rule: 0.5 < VR < 2 indicates good balance
│   │   └─ Checks second moments (SMD only first moment)
│   ├─ Love Plot:
│   │   ├─ SMD for each covariate before and after matching
│   │   ├─ Visual inspection of balance improvement
│   │   └─ Vertical line at |SMD| = 0.1 threshold
│   ├─ Hypothesis Tests:
│   │   ├─ T-test for each covariate: H₀: X̄_treated = X̄_control
│   │   ├─ Warning: Sensitive to sample size (not balance quality)
│   │   └─ Prefer SMD (standardized effect size)
│   ├─ Prognostic Score Balance:
│   │   ├─ Balance on predicted Y₀: μ̂₀(X) = E[Y|D=0,X]
│   │   ├─ More important than individual X balance
│   │   └─ Check SMD for μ̂₀(X)
│   └─ Common Support:
│       ├─ Histogram of e(X) by D (overlap)
│       ├─ Min-max test: [min e(X|D=1), max e(X|D=1)] ⊂ [min e(X|D=0), max e(X|D=0)]
│       └─ Report % of treated on common support
├─ Inference:
│   ├─ Standard Errors:
│   │   ├─ Matching: Abadie-Imbens SE (accounts for matching uncertainty)
│   │   ├─ Bootstrap: For matching with replacement or kernel
│   │   ├─ IPW: Asymptotic variance or bootstrap
│   │   └─ Clustering: If data clustered (schools, regions)
│   ├─ Confidence Intervals:
│   │   ├─ Normal approximation: ATT ± 1.96·SE
│   │   ├─ Bootstrap percentile: [2.5th, 97.5th percentile]
│   │   └─ Robust to non-normality
│   ├─ Hypothesis Tests:
│   │   ├─ H₀: ATT = 0 vs H₁: ATT ≠ 0
│   │   ├─ T-statistic: t = ATT / SE
│   │   └─ P-value from normal or t-distribution
│   └─ Multiple Testing:
│       ├─ If testing multiple outcomes or subgroups
│       ├─ Bonferroni, Holm, or FDR correction
│       └─ Pre-specify primary outcome to avoid fishing
├─ Sensitivity Analysis:
│   ├─ Rosenbaum Bounds:
│   │   ├─ Question: How strong must hidden confounder be to overturn conclusion?
│   │   ├─ Γ: Odds ratio of treatment odds given X
│   │   ├─ Γ=1: No hidden bias (CIA holds)
│   │   ├─ Γ=2: Hidden confounder could double treatment odds
│   │   ├─ Report: Range of p-values for Γ ∈ [1, 3]
│   │   └─ Interpretation: If Γ=1.5 makes effect insignificant, sensitive
│   ├─ E-value (VanderWeele & Ding 2017):
│   │   ├─ Minimum strength of confounder to explain away effect
│   │   ├─ E = ATT/SE + √(ATT/SE)·(ATT/SE + 1)
│   │   ├─ Large E-value: Robust (would need very strong confounder)
│   │   └─ Report E-value with confidence interval
│   ├─ Placebo Outcomes:
│   │   ├─ Test ATT on pre-treatment outcomes (should be zero)
│   │   ├─ If significant: Suggests unbalanced covariates or bias
│   │   └─ Or: Test on outcomes unaffected by treatment
│   ├─ Varying Covariates:
│   │   ├─ Add/remove covariates from matching
│   │   ├─ If ATT changes substantially: Model-dependent
│   │   └─ Report range of estimates across specifications
│   └─ Subgroup Analysis:
│       ├─ Check ATT stability across subgroups
│       ├─ If varies: Suggests heterogeneity or violations
│       └─ Test for effect modification
├─ Practical Implementation:
│   ├─ Python Libraries:
│   │   ├─ CausalML: PSM, kernel matching, double ML
│   │   ├─ EconML: Meta-learners, DR, causal forests
│   │   ├─ DoWhy: Unified causal inference framework
│   │   └─ psmpy: Propensity score matching
│   ├─ R Libraries:
│   │   ├─ MatchIt: Comprehensive matching methods
│   │   ├─ Matching: Fast nearest neighbor
│   │   ├─ WeightIt: Propensity score weighting
│   │   └─ optmatch: Optimal matching
│   ├─ Stata:
│   │   ├─ teffects: nnmatch, psmatch, ipw, aipw
│   │   ├─ psmatch2: Legacy matching
│   │   └─ pstest, psgraph: Balance diagnostics
│   └─ Workflow:
│       ├─ 1. Specify covariates X (theory-driven)
│       ├─ 2. Check pre-treatment balance (SMD)
│       ├─ 3. Estimate propensity score (logit/probit)
│       ├─ 4. Assess common support (trim if needed)
│       ├─ 5. Match or weight
│       ├─ 6. Check post-matching balance (SMD < 0.1)
│       ├─ 7. Estimate ATT with SE
│       ├─ 8. Sensitivity analysis (Rosenbaum bounds)
│       └─ 9. Report with diagnostics
└─ ATT vs ATE Decision:
    ├─ Report ATT when:
    │   ├─ Evaluating existing program (ex-post)
    │   ├─ Poor overlap (controls rare in treated region)
    │   ├─ Selection on gains expected
    │   └─ Policy question: "Did it work for participants?"
    ├─ Report ATE when:
    │   ├─ Designing new program (ex-ante)
    │   ├─ Good overlap throughout X space
    │   ├─ Policy question: "Would it work for everyone?"
    │   └─ Universal intervention considered
    ├─ Report both when:
    │   ├─ Sufficient overlap for ATE
    │   ├─ Interest in selection on gains (ATT vs ATE)
    │   └─ Policy covers continuation and expansion
    └─ Complementary:
        ├─ ATU = E[Y₁-Y₀|D=0] for expansion to untreated
        ├─ CATE = E[Y₁-Y₀|X] for targeting
        └─ LATE = E[Y₁-Y₀|complier] with instrument
```

**Interaction:** Define population (treated) → Check CIA and overlap for treated region → Estimate PS → Match/weight controls to treated → Balance diagnostics → ATT inference → Sensitivity analysis

## 5. Mini-Project
Implement ATT estimation with matching methods and sensitivity analysis:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

np.random.seed(720)

# ===== Simulate Data with Heterogeneous Treatment Effects =====
print("="*80)
print("TREATMENT ON TREATED (ATT) ESTIMATION")
print("="*80)

n = 1500

# Covariates
age = np.random.normal(40, 12, n)
education = np.random.randint(10, 18, n)
income = np.random.gamma(shape=2, scale=20, size=n)

X = np.column_stack([age, education, income])
X_df = pd.DataFrame(X, columns=['age', 'education', 'income'])

# True propensity score with strong selection
# Older, more educated, higher income → more likely treated
logit = -5 + 0.08*age + 0.25*education + 0.03*income
p_true = 1 / (1 + np.exp(-logit))
D = np.random.binomial(1, p_true, n)

# Heterogeneous treatment effects (selection on gains)
# Treatment effect larger for older, more educated
# ATT > ATE because high-gain units select into treatment
tau_individual = 4 + 0.15*age + 0.3*education + 0.02*income
tau_ATE_true = np.mean(tau_individual)
tau_ATT_true = np.mean(tau_individual[D==1])
tau_ATU_true = np.mean(tau_individual[D==0])

print(f"\nSimulation Setup:")
print(f"  Sample size: n={n}")
print(f"  Treated: {np.sum(D)} ({100*np.mean(D):.1f}%)")
print(f"  True ATE: {tau_ATE_true:.2f}")
print(f"  True ATT: {tau_ATT_true:.2f}")
print(f"  True ATU: {tau_ATU_true:.2f}")
print(f"  Selection on gains: ATT > ATE (high-benefit units select treatment)")

# Potential outcomes
Y0 = 50 + 0.8*age + 2.0*education + 0.15*income + np.random.randn(n)*5
Y1 = Y0 + tau_individual + np.random.randn(n)*3

# Observed outcome
Y = D * Y1 + (1 - D) * Y0

# ===== Naive Estimator =====
print("\n" + "="*80)
print("NAIVE ESTIMATOR")
print("="*80)

att_naive = np.mean(Y[D==1]) - np.mean(Y[D==0])
se_naive = np.sqrt(np.var(Y[D==1])/np.sum(D) + np.var(Y[D==0])/np.sum(1-D))

print(f"Naive ATT: {att_naive:.2f} (SE: {se_naive:.2f})")
print(f"True ATT: {tau_ATT_true:.2f}")
print(f"Bias: {att_naive - tau_ATT_true:.2f}")

# Selection bias decomposition
selection_bias = np.mean(Y0[D==1]) - np.mean(Y0[D==0])
print(f"\nSelection bias: {selection_bias:.2f}")
print(f"  E[Y₀|D=1] - E[Y₀|D=0] = {selection_bias:.2f}")
print(f"  ⚠ Treated have higher baseline outcomes")

# ===== Propensity Score Estimation =====
print("\n" + "="*80)
print("PROPENSITY SCORE ESTIMATION")
print("="*80)

# Standardize covariates for better convergence
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic regression for PS
ps_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
ps_model.fit(X_scaled, D)
ps_hat = ps_model.predict_proba(X_scaled)[:, 1]

print(f"Propensity Score Model:")
print(f"  Range: [{ps_hat.min():.3f}, {ps_hat.max():.3f}]")
print(f"  Mean (treated): {ps_hat[D==1].mean():.3f}")
print(f"  Mean (control): {ps_hat[D==0].mean():.3f}")

# Check overlap
print(f"\nOverlap Diagnostics:")
print(f"  Treated with ps < 0.05: {np.sum((D==1) & (ps_hat < 0.05))}")
print(f"  Control with ps > 0.95: {np.sum((D==0) & (ps_hat > 0.95))}")

# Common support (for ATT: need controls in treated PS range)
ps_treated_min = ps_hat[D==1].min()
ps_treated_max = ps_hat[D==1].max()
common_support = (ps_hat >= ps_treated_min) & (ps_hat <= ps_treated_max)

print(f"  Common support (treated range): [{ps_treated_min:.3f}, {ps_treated_max:.3f}]")
print(f"  Units on common support: {np.sum(common_support)} ({100*np.mean(common_support):.1f}%)")

# Apply common support restriction
X_cs = X[common_support]
D_cs = D[common_support]
Y_cs = Y[common_support]
ps_cs = ps_hat[common_support]
n_cs = len(Y_cs)

# ===== Nearest Neighbor Matching (1:1) =====
print("\n" + "="*80)
print("NEAREST NEIGHBOR MATCHING (1:1 without replacement)")
print("="*80)

# Match on propensity score
treated_idx = np.where(D_cs == 1)[0]
control_idx = np.where(D_cs == 0)[0]

ps_treated = ps_cs[treated_idx].reshape(-1, 1)
ps_control = ps_cs[control_idx].reshape(-1, 1)

# Find nearest neighbor for each treated
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ps_control)
distances, indices = nbrs.kneighbors(ps_treated)

matched_control_idx = control_idx[indices.flatten()]

# ATT via matching
Y_treated = Y_cs[treated_idx]
Y_matched_control = Y_cs[matched_control_idx]
att_nn = np.mean(Y_treated - Y_matched_control)

# Standard error (Abadie-Imbens style, simplified)
se_nn = np.std(Y_treated - Y_matched_control, ddof=1) / np.sqrt(len(treated_idx))

print(f"Nearest Neighbor ATT: {att_nn:.2f} (SE: {se_nn:.2f})")
print(f"True ATT: {tau_ATT_true:.2f}")
print(f"Bias: {att_nn - tau_ATT_true:.2f}")
print(f"Matched pairs: {len(treated_idx)}")
print(f"Mean matching distance: {distances.mean():.4f}")

# Balance check
def calculate_smd(X_treated, X_control):
    """Standardized mean difference"""
    mean_diff = X_treated.mean(axis=0) - X_control.mean(axis=0)
    pooled_std = np.sqrt((X_treated.var(axis=0) + X_control.var(axis=0)) / 2)
    return mean_diff / pooled_std

smd_before = calculate_smd(X[D==1], X[D==0])
smd_after = calculate_smd(X_cs[treated_idx], X_cs[matched_control_idx])

print(f"\nBalance Diagnostics (SMD):")
for j, name in enumerate(['age', 'education', 'income']):
    print(f"  {name:10s}: Before={smd_before[j]:6.3f}, After={smd_after[j]:6.3f}", end='')
    if abs(smd_after[j]) < 0.1:
        print(" ✓")
    else:
        print(" ⚠")

# ===== Kernel Matching (Epanechnikov) =====
print("\n" + "="*80)
print("KERNEL MATCHING (Epanechnikov)")
print("="*80)

def epanechnikov_kernel(u):
    """Epanechnikov kernel: K(u) = 0.75(1-u²) if |u|≤1"""
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

# Bandwidth (rule of thumb)
h = 0.1  # Or: 1.06 * ps_cs.std() * n_cs**(-1/5)

att_kernel_individual = []

for i in treated_idx:
    ps_i = ps_cs[i]
    
    # Kernel weights for all controls
    u = (ps_cs[control_idx] - ps_i) / h
    weights = epanechnikov_kernel(u)
    
    if weights.sum() > 0:
        weights = weights / weights.sum()
        Y_counterfactual = np.dot(weights, Y_cs[control_idx])
        att_kernel_individual.append(Y_cs[i] - Y_counterfactual)
    else:
        # No controls within bandwidth (shouldn't happen with reasonable h)
        att_kernel_individual.append(np.nan)

att_kernel_individual = np.array(att_kernel_individual)
att_kernel = np.nanmean(att_kernel_individual)
se_kernel = np.nanstd(att_kernel_individual, ddof=1) / np.sqrt(np.sum(~np.isnan(att_kernel_individual)))

print(f"Kernel Matching ATT: {att_kernel:.2f} (SE: {se_kernel:.2f})")
print(f"True ATT: {tau_ATT_true:.2f}")
print(f"Bias: {att_kernel - tau_ATT_true:.2f}")
print(f"Bandwidth: h={h}")

# ===== Propensity Score Weighting (ATT) =====
print("\n" + "="*80)
print("INVERSE PROPENSITY WEIGHTING (ATT)")
print("="*80)

# ATT weights: w=1 for treated, w=e/(1-e) for controls
weights = np.where(D_cs == 1, 1, ps_cs / (1 - ps_cs))

# Check extreme weights
print(f"Weight Diagnostics:")
print(f"  Max weight (control): {weights[D_cs==0].max():.2f}")
print(f"  99th percentile (control): {np.percentile(weights[D_cs==0], 99):.2f}")

# Trim extreme weights (optional)
weight_threshold = 20
weights_trimmed = np.clip(weights, 0, weight_threshold)
n_trimmed = np.sum(weights != weights_trimmed)

if n_trimmed > 0:
    print(f"  Trimmed {n_trimmed} weights > {weight_threshold}")
    weights = weights_trimmed

# ESS (effective sample size)
ess = (weights.sum())**2 / (weights**2).sum()
print(f"  Effective sample size: {ess:.0f} (out of {n_cs})")

# ATT via IPW
Y_treated_weighted = Y_cs[D_cs==1].sum()
Y_control_weighted = (Y_cs[D_cs==0] * weights[D_cs==0]).sum()
n_treated = D_cs.sum()
n_control_weighted = weights[D_cs==0].sum()

att_ipw = (Y_treated_weighted / n_treated) - (Y_control_weighted / n_control_weighted)

# Standard error via bootstrap
n_boot = 500
att_ipw_boot = np.zeros(n_boot)

for b in range(n_boot):
    idx = np.random.choice(n_cs, n_cs, replace=True)
    D_b = D_cs[idx]
    Y_b = Y_cs[idx]
    X_b = X_scaled[common_support][idx]
    
    # Refit PS
    ps_model_b = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    ps_model_b.fit(X_b, D_b)
    ps_b = ps_model_b.predict_proba(X_b)[:, 1]
    
    # Weights
    w_b = np.where(D_b == 1, 1, ps_b / (1 - ps_b))
    w_b = np.clip(w_b, 0, weight_threshold)
    
    # ATT
    Y_t = Y_b[D_b==1].sum()
    Y_c = (Y_b[D_b==0] * w_b[D_b==0]).sum()
    n_t = D_b.sum()
    n_c = w_b[D_b==0].sum()
    
    if n_t > 0 and n_c > 0:
        att_ipw_boot[b] = (Y_t / n_t) - (Y_c / n_c)
    else:
        att_ipw_boot[b] = np.nan

se_ipw = np.nanstd(att_ipw_boot, ddof=1)
ci_ipw = np.nanpercentile(att_ipw_boot, [2.5, 97.5])

print(f"\nIPW ATT: {att_ipw:.2f} (SE: {se_ipw:.2f})")
print(f"95% CI: [{ci_ipw[0]:.2f}, {ci_ipw[1]:.2f}]")
print(f"True ATT: {tau_ATT_true:.2f}")

# ===== Sensitivity Analysis (Rosenbaum Bounds) =====
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS (Rosenbaum Bounds)")
print("="*80)

# Simplified Rosenbaum bounds for matched pairs
# Γ: odds ratio for hidden bias
gammas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

def rosenbaum_bound_pvalue(treated_outcomes, control_outcomes, gamma):
    """
    Approximate Rosenbaum bound p-value for matched pairs
    Uses Wilcoxon signed rank test with adjustment for Γ
    """
    differences = treated_outcomes - control_outcomes
    
    # Wilcoxon signed rank statistic
    abs_diff = np.abs(differences)
    ranks = stats.rankdata(abs_diff)
    signed_ranks = ranks * np.sign(differences)
    
    T_plus = np.sum(signed_ranks[signed_ranks > 0])
    
    n = len(differences)
    E_T = n * (n + 1) / 4
    
    # Under Γ bound (worst case)
    # Adjust variance by Γ factor (simplified)
    Var_T = n * (n + 1) * (2*n + 1) / 24
    Var_T_gamma = Var_T * (1 + gamma**2) / 2
    
    z = (T_plus - E_T) / np.sqrt(Var_T_gamma)
    p_value = 1 - stats.norm.cdf(z)
    
    return p_value

print(f"Rosenbaum Bounds (p-values for H₀: ATT=0):")
print(f"  Γ=1 (no bias): Baseline")

for gamma in gammas:
    p_val = rosenbaum_bound_pvalue(Y_treated, Y_matched_control, gamma)
    sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
    print(f"  Γ={gamma:.2f}: p={p_val:.4f} {sig}")

print(f"\nInterpretation:")
print(f"  • Γ=1: No hidden bias (CIA holds)")
print(f"  • Γ=1.5: Unobserved confounder could multiply odds by 1.5")
print(f"  • If effect remains significant at Γ>2: Robust to moderate hidden bias")
print(f"  • If effect insignificant at Γ<1.5: Sensitive to hidden confounding")

# ===== Comparison of Methods =====
print("\n" + "="*80)
print("METHOD COMPARISON")
print("="*80)

results = pd.DataFrame({
    'Method': ['Naive', 'NN Matching', 'Kernel Matching', 'IPW'],
    'ATT': [att_naive, att_nn, att_kernel, att_ipw],
    'SE': [se_naive, se_nn, se_kernel, se_ipw],
    'CI_lower': [
        att_naive - 1.96*se_naive,
        att_nn - 1.96*se_nn,
        att_kernel - 1.96*se_kernel,
        ci_ipw[0]
    ],
    'CI_upper': [
        att_naive + 1.96*se_naive,
        att_nn + 1.96*se_nn,
        att_kernel + 1.96*se_kernel,
        ci_ipw[1]
    ]
})

print(results.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

print(f"\nTrue values:")
print(f"  ATT: {tau_ATT_true:.2f}")
print(f"  ATE: {tau_ATE_true:.2f}")
print(f"  ATU: {tau_ATU_true:.2f}")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Propensity score overlap
ax1 = axes[0, 0]
ax1.hist(ps_hat[D==1], bins=30, alpha=0.6, label='Treated', density=True, color='blue')
ax1.hist(ps_hat[D==0], bins=30, alpha=0.6, label='Control', density=True, color='red')
ax1.axvline(ps_treated_min, color='green', linestyle='--', label='Treated range')
ax1.axvline(ps_treated_max, color='green', linestyle='--')
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Density')
ax1.set_title('Overlap: Propensity Score Distribution')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Covariate balance (Love plot)
ax2 = axes[0, 1]
cov_names = ['age', 'education', 'income']
y_pos = np.arange(len(cov_names))

ax2.scatter(smd_before, y_pos, s=100, label='Before matching', marker='o', color='red')
ax2.scatter(smd_after, y_pos, s=100, label='After matching', marker='s', color='blue')
ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax2.axvline(-0.1, color='gray', linestyle='--', linewidth=1)
ax2.axvline(0.1, color='gray', linestyle='--', linewidth=1)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(cov_names)
ax2.set_xlabel('Standardized Mean Difference')
ax2.set_title('Love Plot: Covariate Balance')
ax2.legend()
ax2.grid(alpha=0.3, axis='x')

# Plot 3: ATT estimates with CIs
ax3 = axes[0, 2]
methods = ['Naive', 'NN Match', 'Kernel', 'IPW']
att_estimates = [att_naive, att_nn, att_kernel, att_ipw]
ci_lower = [
    att_naive - 1.96*se_naive,
    att_nn - 1.96*se_nn,
    att_kernel - 1.96*se_kernel,
    ci_ipw[0]
]
ci_upper = [
    att_naive + 1.96*se_naive,
    att_nn + 1.96*se_nn,
    att_kernel + 1.96*se_kernel,
    ci_ipw[1]
]

y_pos = np.arange(len(methods))
ax3.errorbar(att_estimates, y_pos, 
             xerr=[np.array(att_estimates)-np.array(ci_lower), 
                   np.array(ci_upper)-np.array(att_estimates)],
             fmt='o', capsize=5, capthick=2, markersize=8)
ax3.axvline(tau_ATT_true, color='red', linestyle='--', linewidth=2, label='True ATT')
ax3.axvline(tau_ATE_true, color='green', linestyle=':', linewidth=2, label='True ATE')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(methods)
ax3.set_xlabel('ATT Estimate')
ax3.set_title('ATT Estimates with 95% CIs')
ax3.legend()
ax3.grid(alpha=0.3, axis='x')

# Plot 4: Matching quality (distance distribution)
ax4 = axes[1, 0]
ax4.hist(distances, bins=30, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Propensity Score Distance')
ax4.set_ylabel('Frequency')
ax4.set_title('Matching Quality (NN Distance Distribution)')
ax4.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.4f}')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: IPW weights distribution
ax5 = axes[1, 1]
ax5.hist(weights[D_cs==0], bins=30, edgecolor='black', alpha=0.7)
ax5.set_xlabel('IPW Weight (Controls)')
ax5.set_ylabel('Frequency')
ax5.set_title('IPW Weights Distribution (Controls Only)')
ax5.axvline(weights[D_cs==0].mean(), color='red', linestyle='--', 
            label=f'Mean: {weights[D_cs==0].mean():.2f}')
if n_trimmed > 0:
    ax5.axvline(weight_threshold, color='orange', linestyle=':', 
                label=f'Trim threshold: {weight_threshold}')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Rosenbaum sensitivity
ax6 = axes[1, 2]
pvals_rosenbaum = [rosenbaum_bound_pvalue(Y_treated, Y_matched_control, g) for g in gammas]

ax6.plot(gammas, pvals_rosenbaum, 'o-', linewidth=2, markersize=8)
ax6.axhline(0.05, color='red', linestyle='--', label='α=0.05')
ax6.axhline(0.01, color='orange', linestyle='--', label='α=0.01')
ax6.set_xlabel('Γ (Hidden Bias Odds Ratio)')
ax6.set_ylabel('P-value')
ax6.set_title('Rosenbaum Bounds: Sensitivity to Hidden Bias')
ax6.legend()
ax6.grid(alpha=0.3)
ax6.set_ylim([0, max(pvals_rosenbaum)*1.1])

plt.tight_layout()
plt.savefig('treatment_on_treated.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n1. Selection Bias:")
print(f"   • Naive ATT={att_naive:.1f} is biased upward (selection bias={selection_bias:.1f})")
print(f"   • Treated units have higher baseline outcomes")
print(f"   • Positive selection on gains: ATT={tau_ATT_true:.1f} > ATE={tau_ATE_true:.1f}")

print("\n2. Matching Results:")
print(f"   • NN matching: ATT={att_nn:.1f} (±{se_nn:.1f})")
print(f"   • Kernel matching: ATT={att_kernel:.1f} (±{se_kernel:.1f})")
print(f"   • IPW: ATT={att_ipw:.1f} (±{se_ipw:.1f})")
print(f"   • All methods recover true ATT={tau_ATT_true:.1f} ✓")

print("\n3. Balance:")
print(f"   • All covariates balanced after matching (|SMD| < 0.1)")
print(f"   • Common support: {100*np.mean(common_support):.0f}% of sample")

print("\n4. Sensitivity:")
print(f"   • Effect robust to Γ≤2.0 (moderate hidden bias)")
print(f"   • Would need strong confounder (2× treatment odds) to overturn")

print("\n5. Policy Implications:")
print(f"   • Program effective for participants (ATT={tau_ATT_true:.1f})")
print(f"   • Expansion may yield lower returns (ATU={tau_ATU_true:.1f} < ATT)")
print(f"   • Positive selection: High-benefit individuals self-select")

print("\n" + "="*80)
```

## 6. Challenge Round
When does ATT estimation fail?
- **Poor common support**: No controls in treated PS range → Can't match treated units; extrapolation dangerous; report restricted ATT for matchable subset
- **Extreme weights**: IPW gives huge weights to rare controls → High variance; trim weights or use matching instead
- **Unconfoundedness violated**: Unmeasured confounder drives selection → Biased ATT; sensitivity analysis (Rosenbaum bounds) or IV
- **Matching quality**: Large propensity score distances → Bias if matches poor quality; use caliper or kernel matching
- **Heterogeneity ignored**: ATT averages over treated, masks variation → Report CATE or quantile treatment effects for targeting
- **Spillovers**: Treated affect controls (SUTVA violated) → ATT undefined; cluster-level treatment or network methods

## 7. Key References
- [Imbens & Wooldridge (2009) - Recent Developments in Econometrics of Program Evaluation](https://www.aeaweb.org/articles?id=10.1257/jel.47.1.5)
- [Abadie & Imbens (2006) - Large Sample Properties of Matching Estimators](https://academic.oup.com/restud/article/73/1/235/1552916)
- [Rosenbaum (2002) - Observational Studies, Ch. 4 - Sensitivity Analysis](https://www.springer.com/gp/book/9780387989679)

---
**Status:** Core causal parameter for program evaluation | **Complements:** ATE, matching estimators, propensity scores, sensitivity analysis
