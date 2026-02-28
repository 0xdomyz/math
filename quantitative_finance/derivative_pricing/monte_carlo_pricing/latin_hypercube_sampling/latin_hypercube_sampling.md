# Latin Hypercube Sampling

## Concept Skeleton
**Definition:** Quasi-random sampling technique dividing each dimension into k equiprobable strata and sampling one point per stratum per dimension  
**Purpose:** Achieve uniform space-filling coverage with O(log^d(N)) convergence in d dimensions vs. O(1/√N) standard MC; scale better to high dimensions  
**Prerequisites:** Stratified sampling, multidimensional distributions, Latin squares, quasi-random sequences

## Comparative Framing
| Technique | Latin Hypercube | Sobol/Halton | Stratified | Antithetic |
|-----------|-----------------|-------------|-----------|-----------|
| **Dimension Scaling** | Excellent (50+D) | Excellent (100+D) | Poor (curse of dimensionality) | Good (any D) |
| **Variance Reduction** | ~40-60% | ~50-70% | ~30-50% | ~50% |
| **Uniformity** | Guaranteed margins | Scrambled sequence | Per stratum | N/A |
| **Computational Cost** | Moderate (permutation) | Low | Low | Negligible |
| **Implementation** | Permutation matrices | Bit-reversal | Strata iteration | Negation |

## Examples + Counterexamples
**Simple Example:**  
Portfolio with 10 assets: LHS guarantees each decile of each asset marginal represented; standard MC may cluster in one region

**Success Case:**  
High-dimensional integral (20D): LHS achieves ~60% variance reduction; Sobol sequences similar; standard MC wasted

**Limitation Case:**  
Extremely rare events (tail 0.001%): LHS enforces uniform coverage → wastes samples in central region; importance sampling better

**Correlation Effect:**  
Correlated underlyings: LHS ensures each dimension covered, but correlations not explicitly preserved → may need Cholesky post-hoc

## Layer Breakdown
```
Latin Hypercube Sampling Process:
├─ Dimension & Stratum Setup:
│   ├─ Problem dimensionality: d (# of assets/risk factors)
│   ├─ Sample size: N
│   ├─ Strata per dimension: k = N (each dimension divided into N equiprobable bins)
│   └─ Goal: Exactly 1 sample per stratum per dimension
├─ Stratum Boundaries:
│   ├─ For dimension j ∈ [1, d]:
│   │   ├─ Divide [0,1] into N intervals: [0, 1/N], [1/N, 2/N], ..., [(N-1)/N, 1]
│   │   └─ Inverse CDF: Transform to actual distribution via Φ⁻¹ for normal, etc.
├─ Sample Generation:
│   ├─ For each dimension j:
│   │   ├─ Permutation π_j = random shuffle of {1, 2, ..., N}
│   │   ├─ Uniform in stratum i: u_ij ~ U(π_j(i)-1)/N, π_j(i)/N)
│   │   ├─ Transform to target distribution: X_ij = F_j⁻¹(u_ij)
│   │   └─ Result: Each dimension has exactly 1 sample per stratum
├─ Matrix Form:
│   ├─ Result: N × d matrix X where:
│   │   ├─ Rows: samples (paths)
│   │   ├─ Columns: dimensions (risk factors)
│   │   └─ Constraint: Each column is a permutation of strata assignments
├─ Multivariate Correlation (Optional):
│   ├─ If correlated: Apply Cholesky L to X
│   ├─ Y = L × X (preserves marginal coverage, induces correlations)
│   └─ Alternative: Use copulas to impose dependence
└─ Coverage Guarantee:
    ├─ Margin uniformity: For any dimension j, observed quantiles ≈ [0, 1/N, 2/N, ...]
    ├─ Space-filling: Scattered sample covers entire domain
    └─ Variance reduction: ~40-60% for smooth functions
```

**Interaction:** Stratify all dimensions → enforce permutation per dimension → sample uniformly in each stratum → margin-uniform coverage

## Challenge Round
When is Latin Hypercube suboptimal?
- Low dimensions (d ≤ 2): Antithetic or regular stratified simpler, same benefit
- Non-smooth functions: Discontinuities (barrier options) break uniformity advantage
- Extreme tail events: Enforced uniform coverage wastes samples in central region; importance sampling better
- Computational overhead: Generating permutations for high N slow; Sobol sequences faster
- Correlation preservation: Cholesky application adds complexity; not automatic from LHS

## Key References
- [Wikipedia - Latin Hypercube Sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)
- [McKay, Beckman, Conover - LHS Design of Experiments](https://www.jstor.org/stable/1268522)
- [Glasserman - Monte Carlo Methods (Chapter 4)](https://www.springer.com/gp/book/9780387004519)

---
**Status:** High-dimensional variance reduction | **Complements:** Sobol Sequences, Quasi-Random Numbers, Antithetic Variates
