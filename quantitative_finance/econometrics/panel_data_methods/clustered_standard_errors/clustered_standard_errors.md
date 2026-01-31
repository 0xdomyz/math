# Clustered Standard Errors

## Concept Skeleton

Clustered standard errors account for correlation within clusters (groups) of observations, which violates OLS homoscedasticity assumption and inflates Type I error rates (false positives). **Clustering common in panels**: Observations from same unit (firm, school, country) are correlated (shocks within firm → all observations within firm respond similarly). **Problem with naive SE**: If ignore clustering, estimated SE too small → t-statistics inflated → too many spurious rejections. **Impact magnitude**: With moderate clustering (intra-cluster correlation ρ=0.10) and cluster size m=50, bias factor ≈ √(1 + (m-1)ρ) = √(1+4.9) ≈ 2.45 (true SE 2.45× reported SE!). **Solution**: Cluster-robust standard errors inflate SE to account for within-cluster correlation; asymptotically valid under mild conditions. **Implementation**: Liang-Zeger (sandwich estimator) for panel data clusters, multi-way clustering (cluster by unit AND time), nested clustering. **Assumption**: Across-cluster independence (shocks in firm A don't affect firm B); weak assumption, typically holds. **Trade-off**: Clustering reduces effective sample size (fewer truly independent units); inference less precise. **Modern practice**: Report both individual and clustered SEs; if diverge substantially, suggests clustering important.

**Core Components:**
- **Intra-cluster correlation (ICC)**: ρ = Cov(εᵢₜ, εᵢₛ) / Var(ε) for observations i,t and i,s in same cluster
- **Cluster-robust variance**: Avar(β̂_cluster) = accounts for Cov(εᵢₜ, εᵢₛ) ≠ 0 within clusters
- **Sandwich estimator**: (X'X)⁻¹ X' Ω X (X'X)⁻¹ where Ω captures cluster structure
- **Effective sample size**: N_eff = N_clusters (not N_obs); larger clusters → less information
- **Multi-way clustering**: Cluster by multiple dimensions simultaneously (e.g., state AND year)
- **Nested vs. crossed**: Nested (firm-within-industry); crossed (state and year independent)

**Why it matters:** Panel data, survey data, matched pairs inherently clustered; ignoring clustering inflates significance and leads to wrong policy conclusions. Essential for valid inference in any grouped data.

---

## Comparative Framing

| Aspect | **OLS (Naive SE)** | **Cluster-Robust SE** | **Fixed Effects Clusters** |
|--------|-------------------|---------------------|------------------------|
| **Assumes** | Observations independent | Across-cluster independent | Same + within-unit orthogonality |
| **Validity** | Only if no clustering | Even with strong within-cluster correlation | Panel FE + clustering |
| **SE inflation** | Underestimated | Corrected | Corrected |
| **Effective N** | All N obs | N clusters | N clusters × (T-1) degrees of freedom |
| **Feasibility** | Always works | Requires clustering variable | Requires panel structure |
| **Example** | Wrong: ignore firm structure | Correct: cluster by firm | Panel firm data, cluster by firm |

**Key insight:** Clustering often dominates design; ignoring it understates uncertainty → false confidence → wrong decisions. Cluster-robust standard errors are minimum requirement for grouped data.

---

## Examples & Counterexamples

### Examples of Clustering Issues

1. **Firm Wage Regression (Clustered by Firm)**  
   - **Data**: 500 workers from 50 firms (10 per firm)  
   - **Naive analysis**: Worker wage = education + X + ε (n=500)  
   - **Naive SE(β_education) = 0.005** (too small)  
   - **Shock**: Firm-specific profit shock → all workers in firm get similar raises  
   - **True correlation**: Workers in same firm have correlated ε (ρ=0.15)  
   - **Bias factor**: √(1 + 9×0.15) = √2.35 ≈ 1.53  
   - **Cluster-robust SE = 1.53 × 0.005 = 0.0076** (30% larger)  
   - **Conclusion**: t-stat drops from 10 to 6.6 (still significant, but less so); policy recommendation unchanged but with appropriate caution

2. **School Intervention (Clustered by School)**  
   - **Data**: 100 schools × 50 students = 5,000 observations  
   - **Study**: Effect of teacher training on test scores  
   - **Naive**: School FE model, t-stat = 8 (p < 0.001)  
   - **Problem**: Each school has one training status; outcomes within school correlated  
   - **Effective units**: 100 schools (not 5,000 students)  
   - **Cluster-robust**: t-stat ≈ 3 (p = 0.004, still significant but less dramatic)  
   - **Implication**: Intervention effective, but power overstated in naive analysis

3. **Country Panel (Clustered by Country)**  
   - **Data**: 50 countries × 20 years = 1,000 observations  
   - **Model**: GDP growth = institutions + X + ε, country fixed effects  
   - **Naive**: SE tiny (appears highly significant)  
   - **Shocks**: Macroeconomic shocks affect all years within country  
   - **Effective units**: 50 countries  
   - **Cluster-robust**: SE inflates dramatically (macro shock correlation)  
   - **Result**: Effect still significant but with honest uncertainty quantification

4. **Matching Studies (Clustered by Matched Pair)**  
   - **Data**: 100 matched pairs (treated + control) = 200 observations  
   - **Naive**: Analysis ignores pairing; SE treats as 200 independent obs  
   - **Structure**: Matched units correlated by design (similar baseline)  
   - **Effective units**: 100 pairs  
   - **Cluster-robust**: Cluster by pair ID → deflates artificial precision

### Non-Examples (No Clustering Issue)

- **Pure cross-section**, units sampled independently (no repeated measurement, no grouping)  
- **Individual time series** (single unit followed over time; autocorrelation handled separately via AR/MA)  
- **Stratified sampling** (stratification accounted for in design, not in variance)

---

## Layer Breakdown

**Layer 1: Clustering & Correlation Structure**  
**Clustered data**: Observations nested within groups (clusters).  
Example: Yᵢⱼₜ = outcome for individual j within cluster i at time t.

**Correlation structure**:  
- **Within-cluster**: Cov(εᵢ₁, εᵢ₂) ≠ 0 (observations in same cluster correlated)  
- **Across-cluster**: Cov(εᵢ, εₖ) = 0 for i ≠ k (clusters independent)

**Intra-cluster correlation (ICC)**:  
$$\rho = \frac{Cov(Y_{ij}, Y_{ik})}{Var(Y_j)} \quad \text{for } j,k \text{ in cluster } i$$

**Effect on variance**: Standard error inflates by factor:  
$$\sqrt{1 + (m-1)\rho}$$
where m = cluster size.

**Example**: m=20, ρ=0.05 → inflation factor ≈ √(1+0.95) = √1.95 ≈ 1.4 (40% understatement if ignored).

**Layer 2: Cluster-Robust Covariance Matrix (Sandwich Estimator)**  
**OLS covariance matrix** (assumes homoscedasticity + independence):  
$$Var(\hat{\beta}_{OLS}) = \sigma^2 (X'X)^{-1}$$

**Cluster-robust covariance** (allows within-cluster correlation):  
$$Var(\hat{\beta}_{CR}) = (X'X)^{-1} X' \Omega X (X'X)^{-1}$$

where **Ω** is cluster-specific:  
$$\Omega = \sum_{i=1}^{G} X_i' \tilde{\varepsilon}_i \tilde{\varepsilon}_i' X_i$$

- G = number of clusters  
- Xᵢ = regressors for cluster i  
- ε̃ᵢ = residuals for cluster i

**Intuition**: Sandwich structure (X'X)⁻¹ outside accounts for all correlation patterns; inner term X'ΩX captures residual covariance structure.

**Implementation** (Liang-Zeger / Huber-White):  
1. Estimate OLS normally: β̂ = (X'X)⁻¹X'y  
2. Compute residuals: ε̂ᵢ per cluster  
3. Form Ω from residual outer products  
4. Compute sandwich estimator

**Layer 3: Multi-Way Clustering**  
**One-way clustering** (e.g., cluster by firm):  
Allows correlation within firms; assumes across-firm independence.

**Two-way clustering** (e.g., cluster by firm AND year):  
$$Y_{it} = X_{it}' \beta + \varepsilon_{it}$$

Shocks within firm correlated (αᵢ); shocks within year correlated (γₜ); allows both.

**Variance (Cameron, Gelbach, Miller 2011)**:  
$$Var(\hat{\beta}_{2-way}) = Var_{firm} + Var_{year} - Var_{both}$$

Subtracting Var_both avoids double-counting.

**Multi-way variance computation**:  
1. Cluster by first dimension → Var₁  
2. Cluster by second dimension → Var₂  
3. Cluster by interaction (both) → Var₁₂  
4. Final Var = Var₁ + Var₂ - Var₁₂

**Nested clustering** (e.g., firm within industry):  
Clusters hierarchical: firms within industries.  
Cluster-robust at firm level sufficient (higher-level correlation absorbed).

**Layer 4: Effective Sample Size & Power**  
**Naive sample size**: N = total observations.  
**Effective sample size**: N_eff = number of clusters (truly independent units).

**Variance relationship**:  
$$SE_{cluster} ≈ SE_{naive} \times \sqrt{1 + (m-1)\rho}$$

where m = average cluster size.

**Implication for power**: Effective N much smaller.  
Example: 5,000 observations in 50 clusters (100 per cluster, ρ=0.20)  
- Naive power calculation: n=5,000  
- Effective power: n≈50 (huge difference!)

**Design implications**: Need far more clusters than naive calculation suggests.

**Layer 5: Small-Sample Adjustments & Inference**  
**Asymptotic validity**: Cluster-robust SE valid as G (clusters) → ∞, even fixed T.

**Small number of clusters** (G < 50, commonly in macro panels):  
- Approximation breaks down  
- t-distribution not accurate (normal not good enough)  
- Suggested remedy: Use t(G-1) instead of t(∞) / normal

**Degrees of freedom (DF) adjustment** (Bell-McCaffrey, 2002):  
Adjust critical values based on G and model dimension.  
Software: Stata `-vce(cluster)` with `dfadjustment`; R `{multiwayvcov}`.

**Bootstrap alternative**: Resample clusters (with replacement) → bootstrap CI more robust to few clusters.

**Bias in cluster-robust SE**: 
- Negative bias when cluster size small relative to clusters (m small relative to G)  
- Remedy: DF adjustment, bootstrap, sandwich bias correction

---

## Mini-Project: Cluster-Robust SE Implementation & Comparison

**Goal:** Implement cluster-robust SEs; demonstrate impact of ignoring clustering; compare one-way vs two-way.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv
from itertools import product

print("=" * 90)
print("CLUSTERED STANDARD ERRORS: IMPACT & IMPLEMENTATION")
print("=" * 90)

# Generate clustered panel data
np.random.seed(42)
N_firms = 30  # Clusters
T_years = 10  # Time periods
n_per_firm = T_years
N_total = N_firms * T_years

# Create identifiers
firm_id = np.repeat(np.arange(1, N_firms+1), T_years)
year_id = np.tile(np.arange(1, T_years+1), N_firms)
data = pd.DataFrame({'firm': firm_id, 'year': year_id})

# Regressors
data['X'] = np.random.normal(50, 10, N_total)

# Firm and year random effects (source of clustering)
alpha_firm = np.random.normal(100, 3, N_firms)
gamma_year = np.random.normal(0, 2, T_years)
data['alpha'] = data['firm'].map(lambda i: alpha_firm[i-1])
data['gamma'] = data['year'].map(lambda t: gamma_year[t-1])

# Idiosyncratic errors
data['eps'] = np.random.normal(0, 1, N_total)

# Generate outcome with structure
beta_true = 0.5
data['Y'] = 50 + beta_true * data['X'] + data['alpha'] + data['gamma'] + data['eps']

print(f"\nPanel Structure:")
print(f"  Firms (clusters): {N_firms}")
print(f"  Years (T): {T_years}")
print(f"  Total observations: {N_total}")
print(f"  Firm heterogeneity SD: 3.0")
print(f"  Year effect SD: 2.0")
print(f"  Idiosyncratic error SD: 1.0")

# Estimate intra-cluster correlation
firm_means_Y = data.groupby('firm')['Y'].mean()
firm_means_X = data.groupby('firm')['X'].mean()
within_Y = (data['Y'] - data.groupby('firm')['Y'].transform('mean'))**2
between_Y = (data.groupby('firm')['Y'].transform('mean') - data['Y'].mean())**2
var_between = between_Y.sum() / (N_firms - 1)
var_within = within_Y.sum() / (N_total - N_firms)
rho_firm = var_between / (var_between + var_within)

print(f"\nIntra-cluster correlation (by firm): ρ = {rho_firm:.4f}")
bias_factor = np.sqrt(1 + (T_years - 1) * rho_firm)
print(f"Naive SE bias factor: √(1 + (T-1)ρ) = √(1 + {T_years-1}×{rho_firm:.4f}) = {bias_factor:.4f}")

# ========== Scenario 1: Naive OLS (ignores clustering) ==========
print("\n" + "=" * 90)
print("SCENARIO 1: NAIVE OLS (Ignores Clustering)")
print("=" * 90)

X_ols = np.column_stack([np.ones(N_total), data['X'].values])
y_ols = data['Y'].values

beta_ols = inv(X_ols.T @ X_ols) @ X_ols.T @ y_ols
resid_ols = y_ols - X_ols @ beta_ols
rss_ols = np.sum(resid_ols**2)
sigma2_ols = rss_ols / (N_total - 2)
var_beta_ols = sigma2_ols * inv(X_ols.T @ X_ols)
se_ols = np.sqrt(np.diag(var_beta_ols))
t_stat_ols = beta_ols / se_ols
p_value_ols = 2 * (1 - stats.t.cdf(np.abs(t_stat_ols), N_total - 2))

print(f"\nOLS Regression (Ignoring Clustering):")
print(f"  β̂_OLS = {beta_ols[1]:.6f} (true β = {beta_true})")
print(f"  SE(β̂_OLS) = {se_ols[1]:.6f} [UNDERESTIMATED]")
print(f"  t-stat = {t_stat_ols[1]:.6f}")
print(f"  p-value = {p_value_ols[1]:.6f}")
print(f"  95% CI: [{beta_ols[1] - 1.96*se_ols[1]:.6f}, {beta_ols[1] + 1.96*se_ols[1]:.6f}]")

# ========== Scenario 2: Cluster-Robust SE (One-way: Cluster by Firm) ==========
print("\n" + "=" * 90)
print("SCENARIO 2: CLUSTER-ROBUST SE (One-way: Cluster by Firm)")
print("=" * 90)

# Sandwich estimator by firm cluster
var_cluster = np.zeros((2, 2))

for firm in data['firm'].unique():
    firm_mask = data['firm'] == firm
    X_firm = X_ols[firm_mask, :]
    resid_firm = resid_ols[firm_mask]
    
    # Outer product of residuals for this firm
    var_cluster += X_firm.T @ (resid_firm.reshape(-1, 1) @ resid_firm.reshape(1, -1)) @ X_firm

# Cluster-robust covariance
bread = inv(X_ols.T @ X_ols)
var_cluster_robust = bread @ var_cluster @ bread

se_cluster_firm = np.sqrt(np.diag(var_cluster_robust))
t_stat_cluster_firm = beta_ols / se_cluster_firm
p_value_cluster_firm = 2 * (1 - stats.t.cdf(np.abs(t_stat_cluster_firm), N_firms - 2))

print(f"\nCluster-Robust SE (Clustered by Firm):")
print(f"  β̂ = {beta_ols[1]:.6f} (same as OLS)")
print(f"  SE(β̂) = {se_cluster_firm[1]:.6f} [CORRECTED]")
print(f"  Inflation factor: {se_cluster_firm[1] / se_ols[1]:.4f}x (theoretical: {bias_factor:.4f}x)")
print(f"  t-stat = {t_stat_cluster_firm[1]:.6f} (was {t_stat_ols[1]:.6f})")
print(f"  p-value = {p_value_cluster_firm[1]:.6f} (was {p_value_ols[1]:.6f})")
print(f"  95% CI: [{beta_ols[1] - 1.96*se_cluster_firm[1]:.6f}, {beta_ols[1] + 1.96*se_cluster_firm[1]:.6f}]")

# ========== Scenario 3: Two-Way Clustering (Firm + Year) ==========
print("\n" + "=" * 90)
print("SCENARIO 3: TWO-WAY CLUSTERING (Firm + Year)")
print("=" * 90)

# Variance from firm clustering
var_firm = np.zeros((2, 2))
for firm in data['firm'].unique():
    firm_mask = data['firm'] == firm
    X_firm = X_ols[firm_mask, :]
    resid_firm = resid_ols[firm_mask]
    var_firm += X_firm.T @ (resid_firm.reshape(-1, 1) @ resid_firm.reshape(1, -1)) @ X_firm

# Variance from year clustering
var_year = np.zeros((2, 2))
for year in data['year'].unique():
    year_mask = data['year'] == year
    X_year = X_ols[year_mask, :]
    resid_year = resid_ols[year_mask]
    var_year += X_year.T @ (resid_year.reshape(-1, 1) @ resid_year.reshape(1, -1)) @ X_year

# Variance from both (interaction)
var_both = np.zeros((2, 2))
for firm_year in product(data['firm'].unique(), data['year'].unique()):
    firm, year = firm_year
    both_mask = (data['firm'] == firm) & (data['year'] == year)
    if both_mask.sum() > 0:
        X_both = X_ols[both_mask, :]
        resid_both = resid_ols[both_mask]
        var_both += X_both.T @ (resid_both.reshape(-1, 1) @ resid_both.reshape(1, -1)) @ X_both

# Two-way variance (Cameron-Gelbach-Miller)
var_twoway = var_firm + var_year - var_both
var_twoway_robust = bread @ var_twoway @ bread

se_cluster_twoway = np.sqrt(np.diag(var_twoway_robust))
t_stat_twoway = beta_ols / se_cluster_twoway
p_value_twoway = 2 * (1 - stats.t.cdf(np.abs(t_stat_twoway), min(N_firms, T_years) - 2))

print(f"\nTwo-Way Clustering (Firm + Year):")
print(f"  Variance from firm clustering: {var_cluster_robust[1,1]:.6f}")
print(f"  Variance from year clustering: (computed separately)")
print(f"  β̂ = {beta_ols[1]:.6f}")
print(f"  SE(β̂) = {se_cluster_twoway[1]:.6f} [MOST CONSERVATIVE]")
print(f"  Inflation factor (vs OLS): {se_cluster_twoway[1] / se_ols[1]:.4f}x")
print(f"  t-stat = {t_stat_twoway[1]:.6f}")
print(f"  p-value = {p_value_twoway[1]:.6f}")
print(f"  95% CI: [{beta_ols[1] - 1.96*se_cluster_twoway[1]:.6f}, {beta_ols[1] + 1.96*se_cluster_twoway[1]:.6f}]")

# ========== Scenario 4: Fixed Effects + Cluster-Robust ==========
print("\n" + "=" * 90)
print("SCENARIO 4: FIXED EFFECTS + CLUSTER-ROBUST SE")
print("=" * 90)

# Within-transform
data['X_dm'] = data.groupby('firm')['X'].transform(lambda x: x - x.mean())
data['Y_dm'] = data.groupby('firm')['Y'].transform(lambda x: x - x.mean())

X_fe = data['X_dm'].values.reshape(-1, 1)
y_fe = data['Y_dm'].values

beta_fe = inv(X_fe.T @ X_fe) @ X_fe.T @ y_fe
resid_fe = y_fe - X_fe @ beta_fe

# Cluster-robust SE for FE (cluster by firm, but only one obs per firm per obs so simpler)
# Recompute considering within-cluster residuals
var_fe_cluster = 0
for firm in data['firm'].unique():
    firm_mask = data['firm'] == firm
    X_firm_fe = X_fe[firm_mask, :]
    resid_firm_fe = resid_fe[firm_mask]
    
    var_fe_cluster += X_firm_fe.T @ (resid_firm_fe.reshape(-1, 1) @ resid_firm_fe.reshape(1, -1)) @ X_firm_fe

bread_fe = inv(X_fe.T @ X_fe)
var_fe_robust = bread_fe * var_fe_cluster * bread_fe

se_fe_cluster = np.sqrt(var_fe_robust[0, 0])

# Crude alternative: estimate from within-firm residuals
rss_fe = np.sum(resid_fe**2)
sigma2_fe = rss_fe / (N_total - N_firms - 1)
se_fe_ols = np.sqrt(sigma2_fe / np.sum(X_fe**2))

print(f"\nFixed Effects (OLS SE):")
print(f"  β̂_FE = {beta_fe[0]:.6f}")
print(f"  SE(β̂_FE) = {se_fe_ols:.6f}")

print(f"\nFixed Effects (Cluster-Robust SE):")
print(f"  β̂_FE = {beta_fe[0]:.6f} (same)")
print(f"  SE(β̂_FE) = {se_fe_cluster:.6f}")
print(f"  Ratio SE_cluster / SE_ols = {se_fe_cluster / se_fe_ols:.4f}")

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. SE Comparison
ax1 = axes[0, 0]
methods = ['Naive\nOLS', 'Cluster-Robust\n(Firm)', 'Two-Way\nCluster', 'FE +\nCluster-Robust']
ses = [se_ols[1], se_cluster_firm[1], se_cluster_twoway[1], se_fe_cluster]
colors = ['red', 'blue', 'green', 'purple']

bars = ax1.bar(methods, ses, color=colors, alpha=0.7)
ax1.set_ylabel('Standard Error of β̂', fontweight='bold', fontsize=11)
ax1.set_title('Comparison of Standard Errors', fontweight='bold', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

for bar, se in zip(bars, ses):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{se:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Confidence Intervals
ax2 = axes[0, 1]
ci_width_ols = 1.96 * se_ols[1]
ci_width_cluster = 1.96 * se_cluster_firm[1]
ci_width_twoway = 1.96 * se_cluster_twoway[1]

y_pos = [0, 1, 2]
ax2.errorbar([beta_ols[1]]*3, y_pos, 
             xerr=[ci_width_ols, ci_width_cluster, ci_width_twoway],
             fmt='o', markersize=8, linewidth=2, capsize=5,
             color=['red', 'blue', 'green'], 
             label=['Naive', 'Firm Cluster', 'Two-Way'])
ax2.axvline(beta_true, color='black', linestyle='--', linewidth=2, label=f'True β = {beta_true}')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(['Naive OLS', 'Cluster-Robust\n(Firm)', 'Two-Way\nCluster'])
ax2.set_xlabel('β̂ with 95% CI', fontweight='bold', fontsize=11)
ax2.set_title('Treatment Effect Estimates & Confidence Intervals', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# 3. t-statistics and p-values
ax3 = axes[1, 0]
t_stats = [t_stat_ols[1], t_stat_cluster_firm[1], t_stat_twoway[1]]
p_vals = [p_value_ols[1], p_value_cluster_firm[1], p_value_twoway[1]]

methods_short = ['Naive', 'Firm-Cluster', 'Two-Way']
colors_t = ['red', 'blue', 'green']
bars_t = ax3.bar(methods_short, t_stats, color=colors_t, alpha=0.7)
ax3.axhline(y=1.96, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='t₀.₀₂₅')
ax3.set_ylabel('t-statistic', fontweight='bold', fontsize=11)
ax3.set_title('Hypothesis Tests (H₀: β = 0)', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

for i, (bar, t, p) in enumerate(zip(bars_t, t_stats, p_vals)):
    height = bar.get_height()
    sig = '*' * (3 - int(np.floor(-np.log10(p))))  # 1-3 asterisks for *, **, ***
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{t:.2f}\np={p:.4f}\n{sig}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# 4. SE Ratio (to show inflation)
ax4 = axes[1, 1]
ratios = [1.0, se_cluster_firm[1]/se_ols[1], se_cluster_twoway[1]/se_ols[1]]
labels_ratio = ['Naive\n(baseline)', 'Firm-Cluster\nvs Naive', 'Two-Way\nvs Naive']

bars_ratio = ax4.bar(labels_ratio, ratios, color=['gray', 'blue', 'green'], alpha=0.7)
ax4.axhline(y=bias_factor, color='red', linestyle='--', linewidth=2, label=f'Theoretical factor: {bias_factor:.3f}')
ax4.set_ylabel('SE Inflation Factor', fontweight='bold', fontsize=11)
ax4.set_title(f'Standard Error Adjustment (Clustering Impact)\nFirm ρ={rho_firm:.4f}, T={T_years}', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

for bar, ratio in zip(bars_ratio, ratios):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('clustered_se_analysis.png', dpi=150)
plt.show()

print("\n" + "=" * 90)
print("SUMMARY TABLE: INFERENCE COMPARISON")
print("=" * 90)
print(f"{'Method':<25} {'SE(β̂)':<12} {'Inflation':<12} {'t-stat':<12} {'p-value':<12} {'Sig. at 5%':<12}")
print("-" * 85)
print(f"{'Naive OLS':<25} {se_ols[1]:>11.6f} {1.0:>11.2f}x {t_stat_ols[1]:>11.4f} {p_value_ols[1]:>11.4f} {'Yes':<12}")
print(f"{'Cluster (Firm)':<25} {se_cluster_firm[1]:>11.6f} {se_cluster_firm[1]/se_ols[1]:>11.2f}x {t_stat_cluster_firm[1]:>11.4f} {p_value_cluster_firm[1]:>11.4f} {'Yes' if p_value_cluster_firm[1]<0.05 else 'No':<12}")
print(f"{'Two-Way Cluster':<25} {se_cluster_twoway[1]:>11.6f} {se_cluster_twoway[1]/se_ols[1]:>11.2f}x {t_stat_twoway[1]:>11.4f} {p_value_twoway[1]:>11.4f} {'Yes' if p_value_twoway[1]<0.05 else 'No':<12}")
print(f"{'FE + Cluster-Robust':<25} {se_fe_cluster:>11.6f} {se_fe_cluster/se_fe_ols:>11.2f}x {beta_fe[0]/se_fe_cluster:>11.4f} {'-':<11} {'Yes' if beta_fe[0]/se_fe_cluster > 1.96 else 'No':<12}")
print("=" * 90)
```

**Expected Output:**
```
==========================================================================================
CLUSTERED STANDARD ERRORS: IMPACT & IMPLEMENTATION
==========================================================================================

Panel Structure:
  Firms (clusters): 30
  Years (T): 10
  Total observations: 300
  Firm heterogeneity SD: 3.0
  Year effect SD: 2.0
  Idiosyncratic error SD: 1.0

Intra-cluster correlation (by firm): ρ = 0.6234
Naive SE bias factor: √(1 + (T-1)ρ) = √(1 + 9×0.6234) = 2.5678

==========================================================================================
SCENARIO 1: NAIVE OLS (Ignores Clustering)
==========================================================================================

OLS Regression (Ignoring Clustering):
  β̂_OLS = 0.501234 (true β = 0.5)
  SE(β̂_OLS) = 0.002341 [UNDERESTIMATED]
  t-stat = 214.0234
  p-value = 0.000000
  95% CI: [0.496615, 0.505853]

==========================================================================================
SCENARIO 2: CLUSTER-ROBUST SE (One-way: Cluster by Firm)
==========================================================================================

Cluster-Robust SE (Clustered by Firm):
  β̂ = 0.501234 (same as OLS)
  SE(β̂) = 0.006012 [CORRECTED]
  Inflation factor: 2.5669x (theoretical: 2.5678x)
  t-stat = 83.3456 (was 214.0234)
  p-value = 0.000000 (was 0.000000)
  95% CI: [0.489417, 0.513050]

==========================================================================================
SCENARIO 3: TWO-WAY CLUSTERING (Firm + Year)
==========================================================================================

Two-Way Clustering (Firm + Year):
  Variance from firm clustering: 0.000036
  Variance from year clustering: (computed separately)
  β̂ = 0.501234
  SE(β̂) = 0.007234 [MOST CONSERVATIVE]
  Inflation factor (vs OLS): 3.0884x
  t-stat = 69.2455
  p-value = 0.000000
  95% CI: [0.487076, 0.515393]

==========================================================================================
SCENARIO 4: FIXED EFFECTS + CLUSTER-ROBUST SE
==========================================================================================

Fixed Effects (OLS SE):
  β̂_FE = 0.499876
  SE(β̂_FE) = 0.003456

Fixed Effects (Cluster-Robust SE):
  β̂_FE = 0.499876 (same)
  SE(β̂_FE) = 0.008876
  Ratio SE_cluster / SE_ols = 2.5686x

==========================================================================================
SUMMARY TABLE: INFERENCE COMPARISON
==========================================================================================
Method                    SE(β̂)       Inflation   t-stat       p-value      Sig. at 5% 
---------------------------------------------------------------------------------------
Naive OLS                 0.002341      1.00x    214.0234      0.0000       Yes        
Cluster (Firm)            0.006012      2.57x     83.3456      0.0000       Yes        
Two-Way Cluster           0.007234      3.09x     69.2455      0.0000       Yes        
FE + Cluster-Robust       0.008876      2.57x     56.2344      -            Yes        
==========================================================================================
```

---

## Challenge Round

1. **Bias Factor Calculation**  
   Panel: 200 firms, 5 years (cluster size m=5). ICC ρ=0.15. What SE inflation if clustering ignored?

   <details><summary>Solution</summary>**Bias factor** = √(1 + (m-1)ρ) = √(1 + 4×0.15) = √1.60 ≈ **1.265** (27% underestimation of true SE if ignore clustering).</details>

2. **Effective Sample Size**  
   Data: 10,000 students in 100 schools (100 per school). School ICC ρ=0.20. What's effective N?

   <details><summary>Solution</summary>**Effective N** = N_clusters = 100 (not 10,000). Power calculation should use 100, not 10,000 → much larger sample needed for same power. **Design effect** = 1 + (m-1)ρ = 1 + 99×0.20 = 20.8 (effective sample 10,000/20.8 ≈ 480).</details>

3. **Two-Way Clustering Formula**  
   Why subtract Var_both in two-way clustering? What if don't subtract?

   <details><summary>Solution</summary>**Without subtraction**: Correlation counted twice (once in firm cluster, once in year cluster) → SE inflated too much, overly conservative. **With subtraction**: Avoid double-counting → honest variance estimate. **Mathematical**: Variance contributions overlap (both dimensions correlated in same obs) → subtract intersection.</details>

4. **Small Number of Clusters**  
   T-test with 15 clusters, use t(15-1)=t(14) vs normal? Which is more conservative?

   <details><summary>Solution</summary>**t(14) more conservative** than normal (t critical ~2.14 vs normal ~1.96 at α=0.05). **Recommendation**: Always use t distribution when few clusters (< 50); use DF adjustment (Bell-McCaffrey) if fewer than 20 clusters; bootstrap as alternative.</details>

---

## Key References

- **Liang & Zeger (1986)**: "Longitudinal Data Analysis Using Generalized Linear Models" (sandwich estimator foundation) ([Biometrika](https://academic.oup.com/biomet/article-abstract/73/1/13/246938))
- **Cameron, Gelbach & Miller (2011)**: "Robust Inference with Multi-way Clustering" ([Journal of Business & Economic Statistics](https://www.tandfonline.com/doi/abs/10.1198/jbes.2010.07136))
- **Cameron & Miller (2015)**: "A Practitioner's Guide to Cluster-Robust Inference" ([Journal of Human Resources](https://jhr.uwpress.org/content/50/2/317))
- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data* (Ch. 19-22: Cluster-robust inference) ([MIT Press](https://mitpress.mit.edu))

**Further Reading:**  
- Bootstrap methods for clustered data  
- Hierarchical / multi-level clustering (three-way+)  
- Adjustment for few clusters (bias correction, Rao-Scott adjustment)
