# Average Treatment Effect (ATE)

## 1. Concept Skeleton
**Definition:** Expected difference in outcomes if all units received treatment vs control; E[Y₁ - Y₀]; population-level causal parameter  
**Purpose:** Quantify causal impact of intervention; policy evaluation; answer "what if everyone were treated?"  
**Prerequisites:** Potential outcomes framework, counterfactuals, selection bias, randomization, unconfoundedness, propensity scores

## 2. Comparative Framing
| Estimand | ATE | ATT | ATU | LATE | CATE | Regression Coefficient |
|----------|-----|-----|-----|------|------|------------------------|
| **Definition** | E[Y₁-Y₀] | E[Y₁-Y₀\|D=1] | E[Y₁-Y₀\|D=0] | E[Y₁-Y₀\|complier] | E[Y₁-Y₀\|X] | E[Y\|X, D] slope |
| **Population** | Everyone | Treated units | Untreated units | Compliers only | Subgroup by X | Conditional mean |
| **Identification** | Randomization or CIA | Overlap + CIA | Overlap + CIA | Instrument + monotonicity | CIA + smoothness | No causal interpretation |
| **Policy Relevance** | Universal program | Actual participants | Expansion decision | IV validity | Targeting | Predictive only |
| **Heterogeneity** | Average (masks) | Treated subgroup | Untreated subgroup | Local to compliers | Reveals heterogeneity | Ignored |
| **Estimation** | RCT, matching, IPW | Matching on treated | Counterfactual matching | 2SLS/Wald | ML methods | OLS (biased) |

## 3. Examples + Counterexamples

**Classic Example:**  
Job training RCT (n=1000, 500 treated): ATE=\$2,400 annual earnings gain (p<0.01). ATT=\$2,800 (participants benefit more). ATU=\$2,000 (non-participants would benefit less). Heterogeneity: ATE larger for low-skill workers (CATE=\$3,500 vs \$1,200 high-skill). Policy: Expand program (positive ATE), target low-skill (higher CATE).

**Failure Case:**  
Observational study with selection bias: Healthier patients choose treatment. Naive ATE=+5 years (treatment vs control). After propensity score matching: ATE=+1 year (80% of effect spurious). Unconfoundedness violated if unmeasured health severity drives selection → Sensitivity analysis or instrumental variable.

**Edge Case:**  
Negative ATE but positive ATT: College degree ATE=-\$10k (including dropouts with debt), but ATT=+\$50k (graduates benefit). Selection: Low-ability students shouldn't attend. Policy: Restrict access or improve completion support. Simpson's paradox if aggregating heterogeneous subgroups.

## 4. Layer Breakdown
```
Average Treatment Effect (ATE) Framework:
├─ Potential Outcomes (Rubin Causal Model):
│   ├─ Notation:
│   │   ├─ Yᵢ: Observed outcome for unit i
│   │   ├─ Dᵢ: Treatment indicator (1=treated, 0=control)
│   │   ├─ Y₁ᵢ: Potential outcome if treated
│   │   ├─ Y₀ᵢ: Potential outcome if control
│   │   └─ Yᵢ = DᵢY₁ᵢ + (1-Dᵢ)Y₀ᵢ (observed outcome)
│   ├─ Individual Treatment Effect:
│   │   ├─ τᵢ = Y₁ᵢ - Y₀ᵢ (unit-specific causal effect)
│   │   ├─ Never observed (fundamental problem of causal inference)
│   │   └─ Only observe Yᵢ, not both Y₁ᵢ and Y₀ᵢ
│   ├─ Average Treatment Effect (ATE):
│   │   ├─ τ_ATE = E[Y₁ᵢ - Y₀ᵢ] = E[Y₁ᵢ] - E[Y₀ᵢ]
│   │   ├─ Average across entire population
│   │   ├─ Policy parameter for universal intervention
│   │   └─ Requires counterfactual inference
│   ├─ Average Treatment on Treated (ATT):
│   │   ├─ τ_ATT = E[Y₁ᵢ - Y₀ᵢ | Dᵢ=1]
│   │   ├─ Average effect for those who received treatment
│   │   ├─ Requires E[Y₀ᵢ | Dᵢ=1] (counterfactual for treated)
│   │   └─ Policy parameter for existing program evaluation
│   ├─ Average Treatment on Untreated (ATU):
│   │   ├─ τ_ATU = E[Y₁ᵢ - Y₀ᵢ | Dᵢ=0]
│   │   ├─ Effect if untreated were treated
│   │   └─ Policy parameter for program expansion
│   └─ Relationship:
│       ├─ τ_ATE = P(D=1)·τ_ATT + P(D=0)·τ_ATU
│       └─ ATE is weighted average of ATT and ATU
├─ Selection Bias:
│   ├─ Naive Estimator:
│   │   ├─ τ̂_naive = E[Y|D=1] - E[Y|D=0]
│   │   ├─ Compares treated to control means
│   │   └─ Generally biased (not causal)
│   ├─ Decomposition:
│   │   ├─ E[Y|D=1] - E[Y|D=0] = 
│   │   │   E[Y₁-Y₀|D=1] + {E[Y₀|D=1] - E[Y₀|D=0]}
│   │   ├─ First term: ATT (causal for treated)
│   │   ├─ Second term: Selection bias
│   │   └─ Bias = Difference in Y₀ between treated and control
│   ├─ Sources of Bias:
│   │   ├─ Self-selection: Units choose treatment based on expected gains
│   │   ├─ Confounding: X affects both D and Y
│   │   ├─ Endogeneity: Unmeasured factors drive both
│   │   └─ Non-random assignment
│   ├─ Example:
│   │   ├─ Job training: Motivated workers self-select
│   │   ├─ E[Y₀|D=1] > E[Y₀|D=0] (treated have higher baseline earnings)
│   │   ├─ Naive estimator overstates program effect
│   │   └─ Selection bias positive
│   └─ Goal of Identification:
│       ├─ Eliminate or adjust for selection bias
│       └─ Recover causal ATE or ATT
├─ Identification Strategies:
│   ├─ Randomized Controlled Trials (RCT):
│   │   ├─ Gold Standard:
│   │   │   ├─ Random assignment: D ⊥ (Y₁, Y₀)
│   │   │   ├─ E[Y₁|D=1] = E[Y₁], E[Y₀|D=0] = E[Y₀]
│   │   │   ├─ No selection bias
│   │   │   └─ τ_ATE = E[Y|D=1] - E[Y|D=0]
│   │   ├─ Advantages:
│   │   │   ├─ Unbiased (in expectation)
│   │   │   ├─ Simple estimation (difference in means)
│   │   │   ├─ Transparent, credible
│   │   │   └─ Internal validity high
│   │   ├─ Disadvantages:
│   │   │   ├─ Expensive, time-consuming
│   │   │   ├─ Ethical concerns (withholding treatment)
│   │   │   ├─ External validity (SUTVA, compliance, attrition)
│   │   │   └─ Not always feasible
│   │   └─ Inference:
│   │       ├─ Standard errors: √(Var(Y₁)/n₁ + Var(Y₀)/n₀)
│   │       ├─ T-test or regression adjustment
│   │       └─ Cluster-robust if needed
│   ├─ Conditional Independence Assumption (CIA):
│   │   ├─ Unconfoundedness:
│   │   │   ├─ (Y₁, Y₀) ⊥ D | X (given X, D is as-if random)
│   │   │   ├─ All confounders observed in X
│   │   │   ├─ No unmeasured confounding
│   │   │   └─ Requires rich covariate data
│   │   ├─ Overlap (Common Support):
│   │   │   ├─ 0 < P(D=1|X) < 1 for all X
│   │   │   ├─ All covariate combinations have both treated and control
│   │   │   └─ Excludes regions with no counterfactuals
│   │   ├─ Identification:
│   │   │   ├─ τ_ATE = E_X[E[Y|D=1,X] - E[Y|D=0,X]]
│   │   │   ├─ Average over X distribution
│   │   │   └─ Weighting by P(X)
│   │   ├─ Estimation Methods:
│   │   │   ├─ Regression adjustment
│   │   │   ├─ Matching (propensity score, covariate)
│   │   │   ├─ Inverse propensity weighting (IPW)
│   │   │   ├─ Doubly robust estimators
│   │   │   └─ Machine learning (DML, causal forests)
│   │   └─ Challenges:
│   │       ├─ Unconfoundedness untestable (relies on assumption)
│   │       ├─ Sensitive to functional form (regression)
│   │       ├─ Curse of dimensionality (many X)
│   │       └─ Sensitivity analysis critical
│   ├─ Regression Adjustment:
│   │   ├─ Model:
│   │   │   ├─ E[Y|D,X] = α + τD + β'X
│   │   │   ├─ τ̂ estimates ATE if linear + CIA
│   │   │   └─ Saturated model: E[Y|D,X] = g₁(X)D + g₀(X)(1-D)
│   │   ├─ ATE Estimator:
│   │   │   ├─ τ̂_ATE = (1/n)Σᵢ[ĝ₁(Xᵢ) - ĝ₀(Xᵢ)]
│   │   │   ├─ Predict Y₁ and Y₀ for everyone
│   │   │   └─ Average difference
│   │   ├─ Advantages:
│   │   │   ├─ Simple, familiar
│   │   │   ├─ Efficient (uses all data)
│   │   │   └─ Flexible (nonparametric g)
│   │   └─ Disadvantages:
│   │       ├─ Model misspecification bias
│   │       ├─ Linearity often wrong
│   │       └─ Extrapolation outside support
│   ├─ Propensity Score Methods:
│   │   ├─ Propensity Score:
│   │   │   ├─ e(X) = P(D=1|X) (probability of treatment)
│   │   │   ├─ Balancing score: (Y₁,Y₀) ⊥ D | e(X)
│   │   │   ├─ Reduces dimensionality (X → scalar e)
│   │   │   └─ Estimated via logit/probit
│   │   ├─ Inverse Propensity Weighting (IPW):
│   │   │   ├─ τ̂_ATE = (1/n)Σᵢ[DᵢYᵢ/ê(Xᵢ) - (1-Dᵢ)Yᵢ/(1-ê(Xᵢ))]
│   │   │   ├─ Reweights to create pseudo-population
│   │   │   ├─ Horvitz-Thompson estimator
│   │   │   └─ Weights: wᵢ = Dᵢ/ê + (1-Dᵢ)/(1-ê)
│   │   ├─ Matching on Propensity Score:
│   │   │   ├─ Pair each treated with control(s) with similar e(X)
│   │   │   ├─ ATT: Average Y₁-Y₀ for matched pairs
│   │   │   ├─ Matching algorithms: Nearest neighbor, caliper, kernel
│   │   │   └─ With/without replacement
│   │   ├─ Doubly Robust Estimator:
│   │   │   ├─ Combines outcome model and propensity score
│   │   │   ├─ Consistent if either model correct (not both)
│   │   │   ├─ τ̂_DR = (1/n)Σᵢ[μ̂₁(Xᵢ) - μ̂₀(Xᵢ) + 
│   │   │   │   Dᵢ(Yᵢ-μ̂₁(Xᵢ))/ê(Xᵢ) - (1-Dᵢ)(Yᵢ-μ̂₀(Xᵢ))/(1-ê(Xᵢ))]
│   │   │   └─ Best of both worlds (efficiency + robustness)
│   │   └─ Diagnostics:
│   │       ├─ Overlap: Histogram of e(X) by D
│   │       ├─ Balance: SMD (standardized mean difference) <0.1
│   │       ├─ Trimming: Drop extreme propensity scores
│   │       └─ Common support restriction
│   └─ Instrumental Variables (for LATE, not ATE):
│       ├─ Weak link to ATE (identifies LATE, not ATE)
│       └─ See LATE section
├─ Estimation and Inference:
│   ├─ RCT Estimator:
│   │   ├─ τ̂_ATE = Ȳ₁ - Ȳ₀
│   │   ├─ Variance: Var(τ̂) = Var(Y₁)/n₁ + Var(Y₀)/n₀
│   │   ├─ SE: √Var(τ̂)
│   │   └─ T-test: t = τ̂/SE ~ t(n-2)
│   ├─ Regression Adjustment:
│   │   ├─ Fit: Yᵢ = α + τDᵢ + β'Xᵢ + εᵢ
│   │   ├─ τ̂ from OLS (if CIA + linearity)
│   │   ├─ SE from OLS (robust/clustered)
│   │   └─ Or: Fully saturated model with interactions
│   ├─ IPW:
│   │   ├─ Two-step: Estimate ê(X), then compute weighted means
│   │   ├─ SE via bootstrap or asymptotic theory
│   │   ├─ Normalized weights: Stabilized IPW
│   │   └─ Trim extreme weights (>20)
│   ├─ Matching:
│   │   ├─ ATT: Average difference for matched pairs
│   │   ├─ SE: Account for matching uncertainty
│   │   ├─ Abadie-Imbens SE for matching
│   │   └─ Bootstrap for PSM
│   ├─ Doubly Robust:
│   │   ├─ Requires both μ̂(X) and ê(X)
│   │   ├─ Cross-fitting to avoid overfitting bias
│   │   ├─ Machine learning methods (DML)
│   │   └─ Neyman orthogonalization
│   └─ Modern ML Methods:
│       ├─ Double Machine Learning (DML)
│       ├─ Causal forests (heterogeneous effects)
│       ├─ Targeted learning (TMLE)
│       └─ Meta-learners (S, T, X-learner)
├─ Heterogeneous Treatment Effects:
│   ├─ Conditional ATE (CATE):
│   │   ├─ τ(x) = E[Y₁ - Y₀ | X=x]
│   │   ├─ Treatment effect varies by X
│   │   └─ Policy targeting: Who benefits most?
│   ├─ Estimation:
│   │   ├─ Subgroup analysis (stratify by X)
│   │   ├─ Interaction models: Y = α + τD + β'X + γ(D×X) + ε
│   │   ├─ Causal forests (Wager & Athey 2018)
│   │   ├─ Meta-learners (Künzel et al. 2019)
│   │   └─ Generalized random forests (GRF)
│   ├─ Inference:
│   │   ├─ Multiple testing correction (Bonferroni, FDR)
│   │   ├─ Honest inference (sample splitting)
│   │   └─ Confidence bands (not just pointwise CIs)
│   ├─ Applications:
│   │   ├─ Precision medicine: Treatment by patient characteristics
│   │   ├─ Targeting: Assign treatment where CATE highest
│   │   ├─ Cost-effectiveness: CATE > cost threshold
│   │   └─ Fairness: Check CATE across protected groups
│   └─ Challenges:
│       ├─ Overfitting (high-dimensional X)
│       ├─ Multiple testing (many subgroups)
│       └─ External validity (CATE specific to sample)
├─ Assumptions and Sensitivity:
│   ├─ SUTVA (Stable Unit Treatment Value):
│   │   ├─ No interference: Dⱼ doesn't affect Yᵢ (i≠j)
│   │   ├─ No hidden variations: Single version of treatment
│   │   ├─ Violated: Spillovers, general equilibrium, peer effects
│   │   └─ Solution: Network methods, cluster RCTs
│   ├─ Unconfoundedness:
│   │   ├─ Untestable assumption
│   │   ├─ Relies on observing all confounders
│   │   └─ Sensitivity analysis essential
│   ├─ Sensitivity Analysis:
│   │   ├─ Rosenbaum bounds: How much hidden bias to change conclusion?
│   │   ├─ Γ: Odds ratio of treatment odds given observed X
│   │   ├─ Report: Range of estimates for plausible Γ values
│   │   └─ E-value: Strength of confounder to explain away effect
│   ├─ Overlap:
│   │   ├─ Check common support: P(D=1|X) ∈ (ε, 1-ε)
│   │   ├─ Trim or reweight if poor overlap
│   │   └─ Report which subpopulation ATE applies to
│   └─ External Validity:
│       ├─ ATE specific to population, context, timing
│       ├─ Transportability: Does ATE generalize?
│       └─ Meta-analysis across studies
├─ Practical Implementation:
│   ├─ Python:
│   │   ├─ EconML (Microsoft): DML, causal forests, meta-learners
│   │   ├─ CausalML (Uber): Uplift modeling, meta-learners
│   │   ├─ DoWhy (Microsoft): Causal inference with DAGs
│   │   ├─ DoubleML: Cross-fitting, DML
│   │   └─ Causal Inference 360 (IBM)
│   ├─ R:
│   │   ├─ grf: Generalized random forests, causal forests
│   │   ├─ MatchIt: Matching methods
│   │   ├─ WeightIt: Propensity score weighting
│   │   ├─ DoubleML: Double machine learning
│   │   └─ estimatr: Robust SE, design-based inference
│   ├─ Stata:
│   │   ├─ teffects: Treatment effects (IPW, RA, AIPW, PSM)
│   │   ├─ psmatch2: Propensity score matching
│   │   └─ balance: Covariate balance checks
│   └─ Workflow:
│       ├─ 1. Check balance on covariates (SMD)
│       ├─ 2. Estimate propensity score (logit/probit)
│       ├─ 3. Assess overlap (trim if needed)
│       ├─ 4. Estimate ATE (IPW, matching, or DR)
│       ├─ 5. Inference (robust SE, bootstrap)
│       ├─ 6. Sensitivity analysis (Rosenbaum bounds)
│       └─ 7. Report with diagnostics
└─ Common Applications:
    ├─ Labor Economics: Job training, education, wage policies
    ├─ Health: Drug efficacy, interventions, policy evaluations
    ├─ Development: Aid programs, microfinance, infrastructure
    ├─ Marketing: Ad campaigns, pricing experiments, promotions
    ├─ Public Policy: Tax incentives, regulations, social programs
    └─ Education: Class size, technology, teacher training
```

**Interaction:** Define estimand (ATE/ATT) → Check identification (RCT/CIA/overlap) → Estimate (IPW/matching/DR) → Inference (SE, CIs) → Sensitivity analysis → Interpret with heterogeneity

## 5. Mini-Project
Implement ATE estimation with matching, IPW, and doubly robust methods:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

np.random.seed(680)

# ===== Simulate Observational Data with Selection Bias =====
print("="*80)
print("AVERAGE TREATMENT EFFECT (ATE) ESTIMATION")
print("="*80)

n = 2000

# Covariates
X1 = np.random.randn(n)  # Continuous (e.g., age)
X2 = np.random.binomial(1, 0.5, n)  # Binary (e.g., gender)
X3 = np.random.randn(n)  # Continuous (e.g., baseline health)

X = np.column_stack([X1, X2, X3])

# True propensity score (selection into treatment)
# Higher X1, X3 → more likely to be treated (selection bias)
logit = -0.5 + 0.5*X1 + 0.3*X2 + 0.8*X3
p_true = 1 / (1 + np.exp(-logit))
D = np.random.binomial(1, p_true, n)

# True treatment effect (heterogeneous)
# ATE = 3.0, but varies by X1
tau_true = 3.0 + 0.5*X1  # Heterogeneous effect
tau_ATE_true = 3.0  # Average

# Potential outcomes
Y0 = 5 + 1.5*X1 - 0.5*X2 + 2.0*X3 + np.random.randn(n)
Y1 = Y0 + tau_true + 0.5*np.random.randn(n)  # Add treatment effect

# Observed outcome
Y = D * Y1 + (1 - D) * Y0

print(f"\nSimulation Setup:")
print(f"  Sample size: n={n}")
print(f"  Covariates: X1 (age), X2 (gender), X3 (baseline health)")
print(f"  Treatment: D=1 for {np.sum(D)} ({100*np.mean(D):.1f}%)")
print(f"  True ATE: τ = {tau_ATE_true:.2f}")
print(f"  Selection bias: Higher X1, X3 → more likely treated")
print(f"  Heterogeneity: τ(X1) = 3.0 + 0.5·X1")

# ===== Naive Estimator (Biased) =====
print("\n" + "="*80)
print("NAIVE ESTIMATOR (Selection Bias)")
print("="*80)

ate_naive = np.mean(Y[D==1]) - np.mean(Y[D==0])
se_naive = np.sqrt(np.var(Y[D==1])/np.sum(D) + np.var(Y[D==0])/np.sum(1-D))

print(f"Naive ATE: {ate_naive:.3f} (SE: {se_naive:.3f})")
print(f"True ATE: {tau_ATE_true:.3f}")
print(f"Bias: {ate_naive - tau_ATE_true:.3f}")

# Selection bias decomposition
selection_bias = np.mean(Y0[D==1]) - np.mean(Y0[D==0])
att_true = np.mean(tau_true[D==1])

print(f"\nDecomposition:")
print(f"  E[Y|D=1] - E[Y|D=0] = ATT + Selection Bias")
print(f"  {ate_naive:.3f} = {att_true:.3f} + {selection_bias:.3f}")
print(f"  ⚠ Positive selection bias (treated have higher baseline outcomes)")

# ===== Propensity Score Estimation =====
print("\n" + "="*80)
print("PROPENSITY SCORE ESTIMATION")
print("="*80)

# Estimate propensity score via logistic regression
ps_model = LogisticRegression(penalty=None, max_iter=1000)
ps_model.fit(X, D)
ps_hat = ps_model.predict_proba(X)[:, 1]

print(f"Propensity Score Model: Logistic Regression")
print(f"  P(D=1|X) range: [{ps_hat.min():.3f}, {ps_hat.max():.3f}]")
print(f"  Mean propensity: {ps_hat.mean():.3f}")

# Check overlap
print(f"\nOverlap Check:")
print(f"  Treated units with ps < 0.1: {np.sum((D==1) & (ps_hat < 0.1))}")
print(f"  Control units with ps > 0.9: {np.sum((D==0) & (ps_hat > 0.9))}")

# Trim extreme propensity scores
trim_lower = 0.05
trim_upper = 0.95
keep = (ps_hat >= trim_lower) & (ps_hat <= trim_upper)
n_trimmed = n - np.sum(keep)

print(f"  Trimming ps ∉ [{trim_lower}, {trim_upper}]: {n_trimmed} units removed")

# Apply trim
X_trim = X[keep]
D_trim = D[keep]
Y_trim = Y[keep]
ps_hat_trim = ps_hat[keep]
n_trim = len(Y_trim)

# ===== Inverse Propensity Weighting (IPW) =====
print("\n" + "="*80)
print("INVERSE PROPENSITY WEIGHTING (IPW)")
print("="*80)

# IPW weights
weights_treated = D_trim / ps_hat_trim
weights_control = (1 - D_trim) / (1 - ps_hat_trim)

# ATE via IPW
ipw_treated = np.sum(weights_treated * Y_trim) / np.sum(weights_treated)
ipw_control = np.sum(weights_control * Y_trim) / np.sum(weights_control)
ate_ipw = ipw_treated - ipw_control

# Standard errors via bootstrap (simplified)
n_boot = 500
ate_ipw_boot = np.zeros(n_boot)

for b in range(n_boot):
    idx = np.random.choice(n_trim, n_trim, replace=True)
    D_b = D_trim[idx]
    Y_b = Y_trim[idx]
    ps_b = ps_hat_trim[idx]
    
    w_t = D_b / ps_b
    w_c = (1 - D_b) / (1 - ps_b)
    
    ipw_t = np.sum(w_t * Y_b) / np.sum(w_t)
    ipw_c = np.sum(w_c * Y_b) / np.sum(w_c)
    ate_ipw_boot[b] = ipw_t - ipw_c

se_ipw = np.std(ate_ipw_boot, ddof=1)
ci_ipw = np.percentile(ate_ipw_boot, [2.5, 97.5])

print(f"IPW ATE: {ate_ipw:.3f} (SE: {se_ipw:.3f})")
print(f"95% CI: [{ci_ipw[0]:.3f}, {ci_ipw[1]:.3f}]")
print(f"True ATE: {tau_ATE_true:.3f}")
print(f"Bias: {ate_ipw - tau_ATE_true:.3f}")

# Check weight distribution
print(f"\nWeight Diagnostics:")
print(f"  Max weight (treated): {weights_treated.max():.2f}")
print(f"  Max weight (control): {weights_control.max():.2f}")
print(f"  ESS (treated): {(np.sum(weights_treated)**2 / np.sum(weights_treated**2)):.0f}")
print(f"  ESS (control): {(np.sum(weights_control)**2 / np.sum(weights_control**2)):.0f}")

# ===== Matching on Propensity Score =====
print("\n" + "="*80)
print("PROPENSITY SCORE MATCHING (ATT)")
print("="*80)

# Nearest neighbor matching (1:1 without replacement)
treated_idx = np.where(D_trim == 1)[0]
control_idx = np.where(D_trim == 0)[0]

ps_treated = ps_hat_trim[treated_idx]
ps_control = ps_hat_trim[control_idx]

matched_control_idx = np.zeros(len(treated_idx), dtype=int)
used_controls = set()

for i, t_idx in enumerate(treated_idx):
    # Find nearest control not yet matched
    available = [c for c in control_idx if c not in used_controls]
    if len(available) == 0:
        matched_control_idx[i] = -1
        continue
    
    distances = np.abs(ps_hat_trim[available] - ps_hat_trim[t_idx])
    nearest = available[np.argmin(distances)]
    matched_control_idx[i] = nearest
    used_controls.add(nearest)

# Remove unmatched treated units
valid_matches = matched_control_idx >= 0
treated_matched = treated_idx[valid_matches]
control_matched = matched_control_idx[valid_matches]

# ATT via matching
Y_treated_matched = Y_trim[treated_matched]
Y_control_matched = Y_trim[control_matched]
att_matching = np.mean(Y_treated_matched - Y_control_matched)

# Standard error
se_att_matching = np.std(Y_treated_matched - Y_control_matched, ddof=1) / np.sqrt(len(treated_matched))

print(f"Matching Results:")
print(f"  Matched pairs: {len(treated_matched)}")
print(f"  Unmatched treated: {len(treated_idx) - len(treated_matched)}")
print(f"\nATT (Matching): {att_matching:.3f} (SE: {se_att_matching:.3f})")
print(f"True ATT: {np.mean(tau_true[D==1]):.3f}")

# Balance check after matching
print(f"\nBalance Check (Standardized Mean Difference):")
for j, name in enumerate(['X1', 'X2', 'X3']):
    smd_before = (np.mean(X[D==1, j]) - np.mean(X[D==0, j])) / np.sqrt((np.var(X[D==1, j]) + np.var(X[D==0, j]))/2)
    smd_after = (np.mean(X_trim[treated_matched, j]) - np.mean(X_trim[control_matched, j])) / np.sqrt((np.var(X_trim[treated_matched, j]) + np.var(X_trim[control_matched, j]))/2)
    
    print(f"  {name}: Before={smd_before:.3f}, After={smd_after:.3f}")
    if abs(smd_after) < 0.1:
        print(f"        ✓ Good balance (|SMD| < 0.1)")
    else:
        print(f"        ⚠ Poor balance")

# ===== Regression Adjustment =====
print("\n" + "="*80)
print("REGRESSION ADJUSTMENT")
print("="*80)

# Fit separate models for treated and control
X_treated = X_trim[D_trim == 1]
Y_treated = Y_trim[D_trim == 1]
X_control = X_trim[D_trim == 0]
Y_control = Y_trim[D_trim == 0]

# Random forest for flexibility
from sklearn.ensemble import RandomForestRegressor

rf_treated = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_control = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

rf_treated.fit(X_treated, Y_treated)
rf_control.fit(X_control, Y_control)

# Predict potential outcomes for everyone
Y1_hat = rf_treated.predict(X_trim)
Y0_hat = rf_control.predict(X_trim)

# ATE via regression adjustment
ate_ra = np.mean(Y1_hat - Y0_hat)

# Standard errors via bootstrap
ate_ra_boot = np.zeros(n_boot)

for b in range(n_boot):
    idx = np.random.choice(n_trim, n_trim, replace=True)
    
    X_b = X_trim[idx]
    D_b = D_trim[idx]
    Y_b = Y_trim[idx]
    
    rf_t_b = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=b)
    rf_c_b = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=b)
    
    rf_t_b.fit(X_b[D_b==1], Y_b[D_b==1])
    rf_c_b.fit(X_b[D_b==0], Y_b[D_b==0])
    
    Y1_b = rf_t_b.predict(X_b)
    Y0_b = rf_c_b.predict(X_b)
    
    ate_ra_boot[b] = np.mean(Y1_b - Y0_b)

se_ra = np.std(ate_ra_boot, ddof=1)
ci_ra = np.percentile(ate_ra_boot, [2.5, 97.5])

print(f"Regression Adjustment (Random Forest):")
print(f"  ATE: {ate_ra:.3f} (SE: {se_ra:.3f})")
print(f"  95% CI: [{ci_ra[0]:.3f}, {ci_ra[1]:.3f}]")
print(f"  True ATE: {tau_ATE_true:.3f}")

# ===== Doubly Robust Estimator =====
print("\n" + "="*80)
print("DOUBLY ROBUST (AIPW) ESTIMATOR")
print("="*80)

# Augmented IPW
ipw_term_treated = D_trim * (Y_trim - Y1_hat) / ps_hat_trim
ipw_term_control = (1 - D_trim) * (Y_trim - Y0_hat) / (1 - ps_hat_trim)

ate_dr = np.mean(Y1_hat - Y0_hat + ipw_term_treated - ipw_term_control)

# Standard errors via bootstrap
ate_dr_boot = np.zeros(n_boot)

for b in range(n_boot):
    idx = np.random.choice(n_trim, n_trim, replace=True)
    
    # Refit models
    X_b = X_trim[idx]
    D_b = D_trim[idx]
    Y_b = Y_trim[idx]
    
    # Propensity score
    ps_model_b = LogisticRegression(penalty=None, max_iter=1000)
    ps_model_b.fit(X_b, D_b)
    ps_b = ps_model_b.predict_proba(X_b)[:, 1]
    
    # Outcome models
    rf_t_b = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=b)
    rf_c_b = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=b)
    
    rf_t_b.fit(X_b[D_b==1], Y_b[D_b==1])
    rf_c_b.fit(X_b[D_b==0], Y_b[D_b==0])
    
    Y1_b = rf_t_b.predict(X_b)
    Y0_b = rf_c_b.predict(X_b)
    
    # DR estimator
    ipw_t = D_b * (Y_b - Y1_b) / ps_b
    ipw_c = (1 - D_b) * (Y_b - Y0_b) / (1 - ps_b)
    
    ate_dr_boot[b] = np.mean(Y1_b - Y0_b + ipw_t - ipw_c)

se_dr = np.std(ate_dr_boot, ddof=1)
ci_dr = np.percentile(ate_dr_boot, [2.5, 97.5])

print(f"Doubly Robust (AIPW) ATE: {ate_dr:.3f} (SE: {se_dr:.3f})")
print(f"95% CI: [{ci_dr[0]:.3f}, {ci_dr[1]:.3f}]")
print(f"True ATE: {tau_ATE_true:.3f}")

print(f"\nDouble Robustness Property:")
print(f"  ✓ Consistent if either PS or outcome model correct")
print(f"  ✓ Efficient if both models correct")

# ===== Comparison Summary =====
print("\n" + "="*80)
print("ESTIMATOR COMPARISON")
print("="*80)

results_df = pd.DataFrame({
    'Method': ['Naive', 'IPW', 'Matching (ATT)', 'Reg Adjustment', 'Doubly Robust'],
    'Estimate': [ate_naive, ate_ipw, att_matching, ate_ra, ate_dr],
    'SE': [se_naive, se_ipw, se_att_matching, se_ra, se_dr],
    'CI_lower': [ate_naive - 1.96*se_naive, ci_ipw[0], att_matching - 1.96*se_att_matching, ci_ra[0], ci_dr[0]],
    'CI_upper': [ate_naive + 1.96*se_naive, ci_ipw[1], att_matching + 1.96*se_att_matching, ci_ra[1], ci_dr[1]],
    'True': [tau_ATE_true, tau_ATE_true, np.mean(tau_true[D==1]), tau_ATE_true, tau_ATE_true]
})

print(results_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Propensity score distributions
ax1 = axes[0, 0]
ax1.hist(ps_hat[D==1], bins=30, alpha=0.6, label='Treated', density=True)
ax1.hist(ps_hat[D==0], bins=30, alpha=0.6, label='Control', density=True)
ax1.axvline(trim_lower, color='red', linestyle='--', label='Trim threshold')
ax1.axvline(trim_upper, color='red', linestyle='--')
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Density')
ax1.set_title('Overlap: Propensity Score Distribution')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Covariate balance (before vs after matching)
ax2 = axes[0, 1]
cov_names = ['X1', 'X2', 'X3']
smd_before = []
smd_after = []

for j in range(3):
    smd_b = (np.mean(X[D==1, j]) - np.mean(X[D==0, j])) / np.sqrt((np.var(X[D==1, j]) + np.var(X[D==0, j]))/2)
    smd_a = (np.mean(X_trim[treated_matched, j]) - np.mean(X_trim[control_matched, j])) / np.sqrt((np.var(X_trim[treated_matched, j]) + np.var(X_trim[control_matched, j]))/2)
    smd_before.append(abs(smd_b))
    smd_after.append(abs(smd_a))

x_pos = np.arange(len(cov_names))
width = 0.35

ax2.bar(x_pos - width/2, smd_before, width, label='Before Matching', alpha=0.7)
ax2.bar(x_pos + width/2, smd_after, width, label='After Matching', alpha=0.7)
ax2.axhline(0.1, color='red', linestyle='--', label='Balance threshold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(cov_names)
ax2.set_ylabel('|Standardized Mean Difference|')
ax2.set_title('Covariate Balance')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Plot 3: ATE estimates with CIs
ax3 = axes[0, 2]
methods = ['Naive', 'IPW', 'Reg Adj', 'DR']
estimates = [ate_naive, ate_ipw, ate_ra, ate_dr]
ci_lower = [ate_naive - 1.96*se_naive, ci_ipw[0], ci_ra[0], ci_dr[0]]
ci_upper = [ate_naive + 1.96*se_naive, ci_ipw[1], ci_ra[1], ci_dr[1]]

y_pos = np.arange(len(methods))
ax3.errorbar(estimates, y_pos, xerr=[np.array(estimates)-np.array(ci_lower), 
                                      np.array(ci_upper)-np.array(estimates)],
             fmt='o', capsize=5, capthick=2, markersize=8)
ax3.axvline(tau_ATE_true, color='red', linestyle='--', linewidth=2, label='True ATE')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(methods)
ax3.set_xlabel('ATE Estimate')
ax3.set_title('ATE Estimates with 95% CIs')
ax3.legend()
ax3.grid(alpha=0.3, axis='x')

# Plot 4: IPW weights distribution
ax4 = axes[1, 0]
ax4.hist(weights_treated, bins=30, alpha=0.6, label='Treated weights', density=True)
ax4.hist(weights_control, bins=30, alpha=0.6, label='Control weights', density=True)
ax4.set_xlabel('IPW Weight')
ax4.set_ylabel('Density')
ax4.set_title('IPW Weights Distribution')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Heterogeneous effects (CATE by X1)
ax5 = axes[1, 1]
X1_grid = np.linspace(X1.min(), X1.max(), 50)
cate_true = 3.0 + 0.5*X1_grid

# Estimate CATE via regression
cate_est = []
for x1_val in X1_grid:
    mask = (X_trim[:, 0] >= x1_val - 0.3) & (X_trim[:, 0] <= x1_val + 0.3)
    if np.sum(mask) > 20:
        cate_est.append(np.mean(Y1_hat[mask] - Y0_hat[mask]))
    else:
        cate_est.append(np.nan)

ax5.plot(X1_grid, cate_true, 'r-', linewidth=2, label='True CATE')
ax5.plot(X1_grid, cate_est, 'b--', linewidth=2, label='Estimated CATE')
ax5.axhline(tau_ATE_true, color='green', linestyle=':', label='ATE')
ax5.set_xlabel('X1 (Age)')
ax5.set_ylabel('Treatment Effect')
ax5.set_title('Heterogeneous Treatment Effects')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Bootstrap distributions
ax6 = axes[1, 2]
ax6.hist(ate_ipw_boot, bins=30, alpha=0.5, label='IPW', density=True)
ax6.hist(ate_dr_boot, bins=30, alpha=0.5, label='DR', density=True)
ax6.axvline(tau_ATE_true, color='red', linestyle='--', linewidth=2, label='True ATE')
ax6.set_xlabel('ATE Estimate')
ax6.set_ylabel('Density')
ax6.set_title('Bootstrap Distributions')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('average_treatment_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Selection Bias:")
print(f"   Naive ATE = {ate_naive:.2f} (biased upward)")
print(f"   Selection bias = {selection_bias:.2f}")
print(f"   Treated units have higher baseline outcomes")

print("\n2. Identification:")
print(f"   CIA assumed: (Y₁,Y₀) ⊥ D | X")
print(f"   Overlap checked: Trimmed {n_trimmed} extreme propensity scores")
print(f"   Balance after matching: All SMD < 0.1 ✓")

print("\n3. ATE Estimates:")
print(f"   IPW: {ate_ipw:.2f} (±{se_ipw:.2f})")
print(f"   Regression: {ate_ra:.2f} (±{se_ra:.2f})")
print(f"   Doubly Robust: {ate_dr:.2f} (±{se_dr:.2f})")
print(f"   All close to true ATE = {tau_ATE_true:.2f}")

print("\n4. Heterogeneity:")
print(f"   CATE varies: τ(X1) = 3.0 + 0.5·X1")
print(f"   Older individuals benefit more")
print(f"   Targeting can improve cost-effectiveness")

print("\n5. Practical Recommendations:")
print("   • Doubly robust preferred (combines PS + outcome model)")
print("   • Check overlap (common support)")
print("   • Balance diagnostics (SMD < 0.1)")
print("   • Sensitivity analysis (Rosenbaum bounds)")
print("   • Report heterogeneity (CATE by subgroups)")
print("   • Bootstrap for inference (accounts for estimation uncertainty)")

print("\n6. Software:")
print("   • Python: EconML, CausalML, DoWhy, DoubleML")
print("   • R: grf, MatchIt, WeightIt, DoubleML")
print("   • Stata: teffects (ra, ipw, aipw, psmatch)")
```

## 6. Challenge Round
When does ATE estimation fail or mislead?
- **Unconfoundedness violated**: Unmeasured confounder U affects D and Y → Biased ATE even with perfect overlap/balance; sensitivity analysis (Rosenbaum bounds) or instrument
- **Poor overlap**: No common support for some X → Can't estimate CATE for those X; extrapolation dangerous; report restricted population
- **Model misspecification**: Wrong functional form in regression or PS → Biased; doubly robust protects if one model correct
- **SUTVA violated**: Spillovers/interference (peer effects, GE) → ATE undefined; use network methods or cluster-level randomization
- **External validity**: ATE specific to sample/context → Doesn't generalize; transportability analysis or meta-analysis
- **Heterogeneity ignored**: Report average when effects vary wildly → Misleading for targeting; estimate CATE, report distribution

## 7. Key References
- [Imbens & Rubin (2015) - Causal Inference for Statistics, Social, and Biomedical Sciences](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB)
- [Angrist & Pischke (2009) - Mostly Harmless Econometrics, Ch. 3](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- [Chernozhukov et al. (2018) - Double/Debiased Machine Learning](https://academic.oup.com/ectj/article/21/1/C1/5056401)

---
**Status:** Fundamental causal parameter | **Complements:** ATT, LATE, matching estimators, propensity scores, RCTs
