# Monte Carlo vs Black-Scholes

## 1. Concept Skeleton
**Definition:** Comparative analysis of Monte Carlo simulation and Black-Scholes closed-form pricing for European options, evaluating accuracy, flexibility, computational cost, and practical tradeoffs.  
**Purpose:** Understand when MC provides value over closed-form, validate MC implementations against analytical benchmarks, manage convergence error vs runtime, choose optimal pricing engine  
**Prerequisites:** Black-Scholes formula, Monte Carlo fundamentals, European option payoffs, numerical convergence, variance reduction

## 2. Comparative Framing

| Aspect | Black-Scholes | Monte Carlo |
|--------|---------------|------------|
| **Closed-Form** | Yes, O(1) | No, O(n) |
| **European Call/Put** | Exact | Approximate (SE ∝ 1/√N) |
| **American Options** | No | Yes (via LSM) |
| **Exotic Payoffs** | No (limited) | Yes (very flexible) |
| **Path-Dependent** | No | Yes |
| **Multivariate** | Limited | Scales well |
| **Correlation** | Only via ρ | Full structure via Cholesky |
| **Stochastic Vol** | No | Yes (Heston, SABR) |
| **Calibration** | One σ | Full path distribution |
| **Market Reality** | Assumes perfect market | Can add friction, jumps |
| **Speed (1000 calls)** | < 1ms | 100ms-10s (depends on N) |
| **Cost of Greeks** | Analytical derivatives | Finite diff or AD (extra cost) |

## 3. Examples + Counterexamples

**Simple Example: ATM Call**  
S=100, K=100, T=1, σ=0.20, r=0.05, q=0:
- **BS:** C = $10.45 (exact, instantaneous)
- **MC (N=100K):** C ≈ $10.44 ± 0.05 (95% CI)
- **MC (N=1M):** C ≈ $10.451 ± 0.016 (95% CI)
- Verdict: MC converges to BS; error shrinks as O(1/√N)

**When MC Outshines BS: Asian Option**  
Payoff = max(average(S_t, t=0..T) - K, 0). BS doesn't apply (payoff path-dependent). MC simulation:
- Generate N paths, compute average price on each → payoff → discount
- Result: converges to true value regardless of path complexity
- BS alternative: approximate as lognormal with reduced effective volatility σ_eff (trick, suboptimal)

**Edge Case: Implied Vol Calibration**  
Market prices 100 options across strikes. To calibrate surface:
- **BS approach:** For each option price, invert BS → IV(K). Fast but assumes BS models each correctly; smile/skew not captured.
- **MC + calibration:** Simulate paths under Heston (stochastic vol), solve optimization: minimize |MC_price(params) - market_price| across all strikes. Slower but recovers true volatility surface shape.

## 4. Layer Breakdown

```
MC vs BS Comparison Framework:
├─ Speed & Computational Cost:
│   ├─ BS: O(1) time, independent of accuracy needs
│   ├─ MC: O(N × M) = paths × steps; N ∝ 1/SE²
│   ├─ For 4-digit accuracy: N ~ 10^8 paths required
│   └─ MC: useful when BS unavailable, not primarily speed
├─ Accuracy & Convergence:
│   ├─ BS: Exact under model assumptions (no approximation error)
│   ├─ MC: Statistical error SE = σ_payoff / √N
│   ├─ BS error: Model error if real world violates assumptions
│   ├─ MC + variance reduction: Can achieve BS-level accuracy faster
│   └─ Practical: MC accuracy sufficient for most derivatives
├─ Model Flexibility:
│   ├─ BS rigid: constant σ, no jumps, lognormal paths
│   ├─ MC flexible: any model (Heston, jump-diffusion, local vol)
│   ├─ Exotic payoffs: MC handles max/min/avg, BS very limited
│   ├─ Multivariate: MC scales, BS becomes multi-dimensional PDE
│   └─ Path dependence: MC natural, BS requires discretization trick
├─ Practical Integration:
│   ├─ Clearing, Risk Systems:
│   │   ├─ MC for portfolio risk (1000s of instruments)
│   │   ├─ BS for Greeks (hedge ratios, desk rebalancing)
│   │   ├─ Hybrid: MC for mark-to-market, BS deltas for hedging
│   │   └─ Time constraint: intraday updates need fast Greeks
│   ├─ Backtesting:
│   │   ├─ MC path-by-path, compute realized payoff
│   │   ├─ BS assumes volatility known → can't backtest smile
│   │   └─ MC reveals parameter sensitivity, calibration quality
│   └─ Regulatory:
│       ├─ VaR/Stressed VaR: MC required (multivariate risk)
│       ├─ CVA: MC for counterparty exposure across scenarios
│       └─ BS: component of model, not standalone for capital
├─ Greeks Computation:
│   ├─ BS: Analytical (Delta, Gamma, Vega exact)
│   ├─ MC: Bump method (Δ = [V(S+ε) - V(S-ε)] / 2ε, O(2N) cost)
│   ├─ MC + AD: Automatic differentiation (reverse mode, O(N) cost)
│   ├─ MC + Pathwise: Pathwise derivative (O(N), like AD)
│   └─ Practical: MC Greeks 2-3× slower than prices
└─ Validation & Cross-Checks:
    ├─ MC should reproduce BS for European options
    ├─ Convergence test: vary N, confirm SE ∝ 1/√N
    ├─ Greeks from BS, bump-check against MC finite diff
    ├─ Path count: typically 100K-1M per price (tradeoff)
    └─ Variance reduction: use antithetic/control to reduce N
```

**Interaction:** Choose MC when BS inapplicable (exotic, multivariate, stochastic vol); use BS to benchmark MC accuracy.

## 5. Mini-Project

Implement both BS and MC pricing for European options; compare convergence, speed, Greeks:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

class EuropeanCallPricers:
    """Wrapper comparing BS and MC pricing"""
    
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def black_scholes_price(self):
        """BS closed-form call price"""
        d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        call = self.S*np.exp(-self.q*self.T)*norm.cdf(d1) - \
               self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        return call
    
    def black_scholes_delta(self):
        """BS call delta"""
        d1 = (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / \
             (self.sigma*np.sqrt(self.T))
        return np.exp(-self.q*self.T)*norm.cdf(d1)
    
    def monte_carlo_price(self, n_paths=100000, n_steps=252, seed=42):
        """MC European call price and standard error"""
        np.random.seed(seed)
        dt = self.T / n_steps
        
        Z = np.random.randn(n_paths, n_steps)
        S_paths = np.zeros((n_paths, n_steps + 1))
        S_paths[:, 0] = self.S
        
        for t in range(n_steps):
            S_paths[:, t+1] = S_paths[:, t] * np.exp(
                (self.r - self.q - 0.5*self.sigma**2)*dt + 
                self.sigma*np.sqrt(dt)*Z[:, t]
            )
        
        payoff = np.maximum(S_paths[:, -1] - self.K, 0)
        call_price = np.exp(-self.r*self.T) * np.mean(payoff)
        se = np.exp(-self.r*self.T) * np.std(payoff) / np.sqrt(n_paths)
        
        return call_price, se
    
    def monte_carlo_delta_bump(self, n_paths=100000, bump=0.01):
        """MC delta via finite difference"""
        # Up bump
        pricer_up = EuropeanCallPricers(self.S + bump, self.K, self.T, self.r, self.sigma, self.q)
        price_up, _ = pricer_up.monte_carlo_price(n_paths=n_paths)
        
        # Down bump
        pricer_dn = EuropeanCallPricers(self.S - bump, self.K, self.T, self.r, self.sigma, self.q)
        price_dn, _ = pricer_dn.monte_carlo_price(n_paths=n_paths)
        
        delta = (price_up - price_dn) / (2*bump)
        return delta
    
    def monte_carlo_with_variance_reduction(self, n_paths=100000, n_steps=252, 
                                            use_antithetic=True, use_control=True):
        """MC with antithetic variates and control variate"""
        np.random.seed(42)
        dt = self.T / n_steps
        
        # Antithetic: generate paths and their anti-paths
        if use_antithetic:
            n_pairs = n_paths // 2
            Z = np.random.randn(n_pairs, n_steps)
            payoffs = []
            
            for sign in [1, -1]:
                S_paths = np.full((n_pairs, n_steps + 1), self.S)
                for t in range(n_steps):
                    S_paths[:, t+1] = S_paths[:, t] * np.exp(
                        (self.r - self.q - 0.5*self.sigma**2)*dt + 
                        sign*self.sigma*np.sqrt(dt)*Z[:, t]
                    )
                payoffs.append(np.maximum(S_paths[:, -1] - self.K, 0))
            
            all_payoffs = np.concatenate(payoffs)
        else:
            Z = np.random.randn(n_paths, n_steps)
            S_paths = np.full((n_paths, n_steps + 1), self.S)
            for t in range(n_steps):
                S_paths[:, t+1] = S_paths[:, t] * np.exp(
                    (self.r - self.q - 0.5*self.sigma**2)*dt + 
                    self.sigma*np.sqrt(dt)*Z[:, t]
                )
            all_payoffs = np.maximum(S_paths[:, -1] - self.K, 0)
        
        # Control variate: use BS call as control
        bs_price = self.black_scholes_price()
        bs_payoff_approx = bs_price * np.exp(self.r*self.T)  # Forward value
        
        if use_control:
            # Optimal coefficient: cov(payoff, control) / var(control)
            control_var = bs_payoff_approx
            coeff = np.cov(all_payoffs, control_var)[0, 1] / np.var(control_var)
            adjusted_payoffs = all_payoffs - coeff*(control_var - np.mean(all_payoffs))
            call_price = np.exp(-self.r*self.T) * (np.mean(adjusted_payoffs) + bs_price*np.exp(self.r*self.T))
        else:
            call_price = np.exp(-self.r*self.T) * np.mean(all_payoffs)
        
        se = np.exp(-self.r*self.T) * np.std(all_payoffs) / np.sqrt(len(all_payoffs))
        return call_price, se
    
    def convergence_study(self, n_range=None):
        """Study convergence: MC price vs N"""
        if n_range is None:
            n_range = np.logspace(2, 7, 30).astype(int)
        
        bs_price = self.black_scholes_price()
        mc_prices = []
        mc_errors = []
        mc_times = []
        
        for n in n_range:
            start = time.time()
            price, se = self.monte_carlo_price(n_paths=n, n_steps=100)
            elapsed = time.time() - start
            
            mc_prices.append(price)
            mc_errors.append(1.96*se)
            mc_times.append(elapsed)
        
        return n_range, mc_prices, mc_errors, mc_times, bs_price
    
    def greeks_comparison(self, n_paths=100000):
        """Compare BS and MC Greeks"""
        bs_delta = self.black_scholes_delta()
        mc_delta = self.monte_carlo_delta_bump(n_paths=n_paths, bump=0.5)
        
        print(f"{'='*70}")
        print(f"Greeks Comparison")
        print(f"{'='*70}")
        print(f"Delta (BS):   {bs_delta:.6f}")
        print(f"Delta (MC bump, h=0.5): {mc_delta:.6f}")
        print(f"Difference:  {abs(bs_delta - mc_delta):.6f}")

# Parameters
S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
pricer = EuropeanCallPricers(S, K, T, r, sigma, q)

# Get BS price
bs_price = pricer.black_scholes_price()
print(f"Black-Scholes Call Price: ${bs_price:.4f}\n")

# MC pricing benchmark
print("Monte Carlo Pricing Benchmark:")
for n_paths in [1000, 10000, 100000, 1000000]:
    mc_price, se = pricer.monte_carlo_price(n_paths=n_paths)
    ci = 1.96*se
    error = abs(mc_price - bs_price)
    print(f"  N={n_paths:7d}: ${mc_price:.4f} ± ${ci:.4f} (Error vs BS: ${error:.4f})")

# Variance reduction
print("\nVariance Reduction Study:")
mc_basic, se_basic = pricer.monte_carlo_price(n_paths=100000)
mc_antithetic, se_antithetic = pricer.monte_carlo_with_variance_reduction(
    n_paths=100000, use_antithetic=True, use_control=False
)
mc_control, se_control = pricer.monte_carlo_with_variance_reduction(
    n_paths=100000, use_antithetic=False, use_control=True
)
mc_both, se_both = pricer.monte_carlo_with_variance_reduction(
    n_paths=100000, use_antithetic=True, use_control=True
)

print(f"  Basic MC:        SE = ${se_basic:.4f}")
print(f"  Antithetic:      SE = ${se_antithetic:.4f} ({100*(1-se_antithetic/se_basic):.1f}% reduction)")
print(f"  Control Variate: SE = ${se_control:.4f} ({100*(1-se_control/se_basic):.1f}% reduction)")
print(f"  Both:            SE = ${se_both:.4f} ({100*(1-se_both/se_basic):.1f}% reduction)")

# Greeks comparison
pricer.greeks_comparison(n_paths=1000000)

# Convergence study
n_range, mc_prices, mc_errors, mc_times, bs_price = pricer.convergence_study()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: MC convergence
axes[0, 0].semilogx(n_range, mc_prices, 'o-', linewidth=2, markersize=6, label='MC Price')
axes[0, 0].axhline(bs_price, color='r', linestyle='--', linewidth=2, label='BS Price')
axes[0, 0].fill_between(n_range, 
                        np.array(mc_prices) - np.array(mc_errors),
                        np.array(mc_prices) + np.array(mc_errors),
                        alpha=0.2, label='95% CI')
axes[0, 0].set_xlabel('Number of Paths (log scale)')
axes[0, 0].set_ylabel('Call Price')
axes[0, 0].set_title('MC Convergence to BS')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Standard Error convergence (verify O(1/√N))
axes[0, 1].loglog(n_range, mc_errors, 'o-', linewidth=2, markersize=6)
# Theoretical: SE ∝ 1/√N
reference_se = mc_errors[0] * (n_range[0] / n_range) ** 0.5
axes[0, 1].loglog(n_range, reference_se, 'r--', linewidth=2, label='O(1/√N) reference')
axes[0, 1].set_xlabel('Number of Paths (log scale)')
axes[0, 1].set_ylabel('95% CI Half-Width (log scale)')
axes[0, 1].set_title('Standard Error Decay')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Computational time
axes[1, 0].loglog(n_range, np.array(mc_times)*1000, 'o-', linewidth=2, markersize=6)
# Linear scaling: T ∝ N
reference_time = np.array(mc_times[0]) * (n_range / n_range[0])
axes[1, 0].loglog(n_range, reference_time*1000, 'r--', linewidth=2, label='O(N) reference')
axes[1, 0].set_xlabel('Number of Paths (log scale)')
axes[1, 0].set_ylabel('Time (ms, log scale)')
axes[1, 0].set_title('Computational Time vs N')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Accuracy vs Time tradeoff
axes[1, 1].loglog(np.array(mc_times)*1000, mc_errors, 'o-', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Time (ms, log scale)')
axes[1, 1].set_ylabel('95% CI Half-Width (log scale)')
axes[1, 1].set_title('Accuracy-Speed Tradeoff')
axes[1, 1].grid(alpha=0.3)

# Annotation: target accuracy
target_accuracy = 0.01  # $0.01
target_idx = np.argmin(np.abs(np.array(mc_errors) - target_accuracy))
axes[1, 1].plot(mc_times[target_idx]*1000, mc_errors[target_idx], 'r*', 
               markersize=15, label=f'Target ±${target_accuracy:.2f}')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('mc_vs_bs_analysis.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: mc_vs_bs_analysis.png")
```

**Output Interpretation:**
- **Convergence:** MC prices cluster around BS; confidence interval shrinks as O(1/√N)
- **Time Trade-off:** Doubling N doubles time but halves standard error (costly improvement)
- **Greeks:** MC delta from bumps matches BS but noisier; 1M paths needed for 2-3 decimal places

## 6. Challenge Round

**Q1: For a European call, MC converges to BS as N → ∞. What does MC gain in practice?**  
A: MC shines when BS inapplicable: (1) Path-dependent payoffs (Asian, lookback) where BS has no closed form. (2) Multiple underlyings with correlation (basket options). (3) Stochastic volatility (Heston model); BS assumes σ constant. (4) Jump processes; BS assumes continuous paths. (5) Portfolio risk aggregation across 1000s of instruments. For vanilla European options, MC is slower than BS; use MC for validation, BS for speed.

**Q2: MC has statistical error SE = σ_payoff / √N. How do variance reduction techniques reduce SE?**  
A: Antithetic variates: use Z and -Z pairs → payoff variance approximately halves (negative correlation between pairs). Control variate: subtract BS price approximation before averaging → payoff closer to mean → lower variance. Combined: SE reduction 2-5×, meaning same accuracy with fewer paths. Cost: extra computation (worth it if overhead < time saved).

**Q3: A desk manager needs Greeks for 1000 options daily. Why not use MC for delta hedges?**  
A: MC Greeks via finite differences: bump S by ε, rerun N-path simulation, compute delta = ΔPrice/ε. Cost: 2N paths minimum per option (price up/down). For 1000 options: 2M path simulations. BS delta: instant analytical formula, no simulation needed. Solution: Hybrid approach. (1) MC for mark-to-market prices (accuracy > speed). (2) BS for daily deltas (speed > accuracy for small moves). (3) Update BS vol calibration hourly from MC market fits. Practical compromise for large portfolios.

**Q4: What is the "Curse of Dimensionality" in MC vs binomial trees?**  
A: Binomial trees scale exponentially with dimensions: d assets → 2^(n×d) nodes for n steps. Impractical beyond 3-4 assets. MC: N paths × M steps, independent of dimension → scales linear in d. For 100-asset basket: MC is the only feasible option. This is why MC dominates in portfolio risk, CVA, credit models.

**Q5: How do you choose N (number of paths) for MC? Is there a rule of thumb?**  
A: Standard practice: SE ≤ 1% of price. If price ≈ $10, need SE ≤ 0.10. For equity options, payoff std ≈ price magnitude. Then: N ≥ (σ_payoff / target_SE)². Example: σ = $10, target SE = $0.10 → N ≥ 10,000. Common targets: N = 100K (quick pricing, 1-2 cent accuracy), N = 1M (precise risk reporting). High-frequency: N = 10K (speed); regulatory: N = 10M (accuracy).

**Q6: MC prices have confidence intervals due to randomness. How do you interpret a 95% CI in a risk report?**  
A: 95% CI = [Price - 1.96×SE, Price + 1.96×SE]. Interpretation: if we ran the simulation 100 times (different random seeds), ~95 of them would have price in this interval. Risk: wide CI indicates high model uncertainty (high payoff variance or too few paths). Mitigation: (1) increase N, (2) apply variance reduction, (3) check if payoff volatility is genuinely high (legitimate risk indicator). In reports: quote price ± half-width (e.g., "$10.45 ± 0.05" for 95% CI).

## 7. Key References

- [Wikipedia: Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method) — Fundamentals, convergence, error analysis
- [Wikipedia: Black–Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) — Closed-form solution
- Glasserman: *Monte Carlo Methods in Financial Engineering* (Chapter 2) — MC vs analytical methods, convergence rates
- Hull: *Options, Futures & Derivatives* (Chapters 13-20) — Pricing methods, when to use each approach

**Status:** ✓ Standalone file. **Complements:** black_scholes_model.md, european_call_option.md, monte_carlo_fundamentals.md, variance_reduction_techniques.md
