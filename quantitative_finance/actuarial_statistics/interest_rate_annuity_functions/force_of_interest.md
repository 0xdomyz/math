# Force of Interest

## 1. Concept Skeleton
**Definition:** Continuous compounding rate δ = ln(1+i); instantaneous force of growth per unit time  
**Purpose:** Model continuous accumulation, stochastic interest rate models, elegant mathematical framework, theoretical foundation  
**Prerequisites:** Natural logarithm, exponential functions, continuous compounding, calculus-based finance

## 2. Comparative Framing
| Concept | Force of Interest δ | Effective Annual i | Nominal i^(m) |
|---------|-------------------|-------------------|----------------|
| **Formula** | δ = ln(1+i) | i = e^δ - 1 | i^(m) = m[(1+i)^{1/m} - 1] |
| **Accumulation** | e^{δt} (continuous) | (1+i)^n (annual) | (1+i^(m)/m)^{mn} (periodic) |
| **Domain** | ℝ; can be negative | (−1,∞); usually positive | i^(m) > -m (denominators >0) |
| **Usage** | Stochastic models | Insurance/actuarial | Bonds, mortgages |
| **Notation** | δ, often Greek delta | i, sometimes r | i^(m), j_m |

## 3. Examples + Counterexamples

**Simple Example:**  
5% effective annual (i = 0.05) ↔ δ = ln(1.05) ≈ 0.04879 (4.879% force); nearly identical for small rates

**Failure Case:**  
Claiming δ = i when large rates: 50% effective → δ = ln(1.5) ≈ 0.4055 (40.55%), not 50%; logarithm crucial

**Edge Case:**  
Negative force (deflation): δ = -0.02 (−2% continuous deflation) ↔ i = e^{−0.02} - 1 ≈ -1.98% annual; rare but mathematically valid

## 4. Layer Breakdown
```
Force of Interest Structure:
├─ Definition & Relationships:
│   ├─ δ = ln(1+i)  [convert from effective]
│   ├─ i = e^δ - 1  [convert to effective]
│   ├─ δ(t) = d[ln A(t)]/dt  [instantaneous rate]
│   ├─ A(t) = e^{∫₀^t δ(s) ds}  [continuous accumulation]
│   └─ Approximation: δ ≈ i for small rates (i < 0.1)
├─ Accumulation Functions:
│   ├─ Constant force: A(t) = e^{δt}  (simple exponential)
│   ├─ Variable force: A(t) = exp(∫₀^t δ(s) ds)  (path integral)
│   ├─ Discrete to continuous: Replace (1+i)^n with e^{δn}
│   └─ Differential form: dA/dt = δ·A  (fundamental equation)
├─ Present Value with Continuous Force:
│   ├─ PV of $1 at time t: v(t) = e^{−δt}
│   ├─ Annuity: a̅ₙ̄| = ∫₀ⁿ e^{−δt} dt = (1 − e^{−δn})/δ
│   ├─ Perpetuity: a̅_∞ = 1/δ
│   └─ Continuous payment: ∫₀ⁿ Rate(t)·e^{−δt} dt
├─ Comparison Across Rates:
│   ├─ For i = 0.05: δ ≈ 0.04879, i^(∞) ≈ 0.05129
│   ├─ Ordering: i^(∞) ≤ δ ≤ i (force sandwiched)
│   ├─ Gap narrows as i → 0 (asymptotically equal)
│   └─ Gap widens as i → ∞ (e.g., i=1: δ≈0.693, i^(∞)≈1)
└─ Applications:
    ├─ Vasicek model: dr(t) = a(b − r(t))dt + σ dW_t
    ├─ Stochastic mortality: μₓ(t) = μₓ₀ · exp(δ·t)
    ├─ Continuous cashflows: Price = ∫ cf(t)·e^{−δt} dt
    └─ Duration & convexity: D = −(1/P)dP/dδ
```

**Interaction:** Choose δ → Calculate e^{δt} → Discount continuous flows → Integrate → Solve for unknowns

## 5. Mini-Project
Calculate force of interest, analyze continuous accumulation, and model stochastic rates:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint
from scipy.optimize import fsolve

# 1. CONVERSION: EFFECTIVE ↔ FORCE OF INTEREST
print("=" * 70)
print("FORCE OF INTEREST: CONVERSIONS AND RELATIONSHIPS")
print("=" * 70)

effective_rates = [0.01, 0.03, 0.05, 0.10, 0.20, 0.50]

print("\nConversion table:")
print(f"{'Effective i':<15} {'Force δ':<15} {'δ as %':<15} {'Approx Error (i-δ)':<20}")
print("-" * 65)

conversions = []

for i in effective_rates:
    delta = np.log(1 + i)
    approx_error = i - delta
    
    print(f"{i*100:>6.2f}%        {delta*100:>6.4f}%        {delta*100:>6.4f}%        {approx_error*100:>6.4f} pp")
    
    conversions.append({
        'Effective i (%)': f"{i*100:.2f}",
        'Force δ (%)': f"{delta*100:.4f}",
        'Difference (pp)': f"{approx_error*1e4:.1f}",
        'Difference (%)': f"{(approx_error/i)*100:.2f}" if i > 0 else "N/A"
    })

print(f"\nNote: For small rates (i < 0.05), δ ≈ i (approximation valid)")
print(f"For large rates (i > 0.20), δ materially less than i")
print()

# 2. CONTINUOUS ACCUMULATION
print("=" * 70)
print("CONTINUOUS VS DISCRETE ACCUMULATION")
print("=" * 70)

principal = 1000
delta_rate = 0.05  # 5% force of interest
time_years = np.array([1, 5, 10, 20, 30])

# Corresponding effective rate
i_eff = np.exp(delta_rate) - 1

print(f"\nStarting principal: ${principal:,.2f}")
print(f"Force of interest: δ = {delta_rate*100:.2f}%")
print(f"Equivalent effective annual: i = {i_eff*100:.4f}%")
print()

accum_data = []

for t in time_years:
    # Continuous: A(t) = P * e^{δt}
    continuous = principal * np.exp(delta_rate * t)
    
    # Discrete annual: A(t) = P * (1+i)^t
    discrete = principal * (1 + i_eff) ** t
    
    # Monthly compounding (for comparison)
    i_monthly = delta_rate / 12
    monthly = principal * (1 + i_monthly) ** (12 * t)
    
    accum_data.append({
        'Years': t,
        'Continuous e^{δt}': continuous,
        'Discrete (1+i)^t': discrete,
        'Monthly (1+i^12/12)^{12t}': monthly,
        'Error (cont - disc)': continuous - discrete
    })

accum_df = pd.DataFrame(accum_data)
print(accum_df.to_string(index=False))
print()

# 3. CONTINUOUS ANNUITIES
print("=" * 70)
print("CONTINUOUS ANNUITIES (a̅ₙ̄|)")
print("=" * 70)

# Continuous annuity present value: a̅ₙ̄| = (1 - e^{-δn})/δ
n_periods = np.array([1, 5, 10, 20, 30, 50])
payment_rate = 1000  # $1,000/year payable continuously

delta = 0.05

print(f"\nPayment rate: ${payment_rate}/year (continuous)")
print(f"Force of interest: δ = {delta*100:.2f}%\n")

continuous_annuity_data = []

for n in n_periods:
    # Continuous annuity present value
    a_continuous = (1 - np.exp(-delta * n)) / delta
    
    # Annuity-due factor for comparison
    a_due_factor = (1 - (1 + i_eff)**(-n)) / i_eff * (1 + i_eff)
    
    # Perpetuity (n → ∞)
    a_perp = 1 / delta
    
    pv_continuous = payment_rate * a_continuous
    pv_perpetuity = payment_rate / delta
    
    continuous_annuity_data.append({
        'n (years)': n,
        'a̅ₙ̄|': f"{a_continuous:.6f}",
        'PV ($)': f"${pv_continuous:.2f}",
        '% of Perpetuity': f"{100 * a_continuous / a_perp:.2f}%"
    })

annuity_df = pd.DataFrame(continuous_annuity_data)
print(annuity_df.to_string(index=False))
print(f"\nContinuous Perpetuity (n→∞): PV = ${pv_perpetuity:,.2f}")
print()

# 4. PRESENT VALUE WITH VARIABLE FORCE
print("=" * 70)
print("PRESENT VALUE WITH VARIABLE FORCE OF INTEREST")
print("=" * 70)

# Example: Linear increase in force δ(t) = δ₀ + αt
delta_0 = 0.03  # Starting force
alpha = 0.001   # Annual increase
n_term = 10

def delta_func(t):
    """Variable force of interest δ(t) = δ₀ + αt"""
    return delta_0 + alpha * t

def accumulation_func(t):
    """A(t) = exp(∫₀^t δ(s) ds) for linear δ"""
    # ∫(δ₀ + αs)ds from 0 to t = δ₀*t + α*t²/2
    return np.exp(delta_0 * t + alpha * t**2 / 2)

# Present value of $1 at time t: e^{-∫δ(s)ds}
times = np.linspace(0, n_term, 100)
pv_factors = np.exp(-(delta_0 * times + alpha * times**2 / 2))

print(f"\nVariable force: δ(t) = {delta_0} + {alpha}t")
print(f"Time horizon: {n_term} years\n")

print(f"{'Year':<8} {'δ(t)':<12} {'∫δ ds':<12} {'PV Factor':<12} {'Accumulated $':<12}")
print("-" * 56)

for t in np.linspace(0, n_term, 11):
    delta_t = delta_func(t)
    integral = delta_0 * t + alpha * t**2 / 2
    pv_factor = np.exp(-integral)
    accum = np.exp(integral)
    print(f"{t:<8.1f} {delta_t*100:<11.3f}% {integral*100:<11.3f}% {pv_factor:<12.6f} ${accum:<11.2f}")

print()

# 5. STOCHASTIC FORCE (MONTE CARLO)
print("=" * 70)
print("STOCHASTIC FORCE: MONTE CARLO SIMULATION")
print("=" * 70)

# Vasicek model: dδ(t) = a(b - δ(t))dt + σ dW(t)
# Parameters (mean-reverting to b)
a = 0.05      # Mean reversion speed
b = 0.04      # Long-term mean force
sigma = 0.01  # Volatility
delta_0_vasicek = 0.05  # Initial force

n_sims = 10000
n_steps = 120  # 10 years, monthly steps
dt = 1/12
times_vasicek = np.arange(0, n_steps + 1) * dt

# Simulate paths
np.random.seed(42)
delta_paths = np.zeros((n_sims, n_steps + 1))
delta_paths[:, 0] = delta_0_vasicek

for i in range(n_steps):
    dW = np.random.normal(0, np.sqrt(dt), n_sims)
    delta_paths[:, i+1] = delta_paths[:, i] + a * (b - delta_paths[:, i]) * dt + sigma * dW

print(f"\nVasicek Model (mean-reverting):")
print(f"  Mean reversion speed (a): {a}")
print(f"  Long-term mean (b): {b*100:.2f}%")
print(f"  Volatility (σ): {sigma*100:.2f}%")
print(f"  Initial force δ(0): {delta_0_vasicek*100:.2f}%")
print(f"  Simulations: {n_sims}, Time horizon: {n_steps * dt:.1f} years\n")

# Statistics
final_deltas = delta_paths[:, -1]
print(f"Final force statistics (at t={n_steps*dt:.1f}y):")
print(f"  Mean: {np.mean(final_deltas)*100:.3f}%")
print(f"  Std Dev: {np.std(final_deltas)*100:.3f}%")
print(f"  Min: {np.min(final_deltas)*100:.3f}%")
print(f"  Max: {np.max(final_deltas)*100:.3f}%")
print(f"  Median: {np.median(final_deltas)*100:.3f}%")

# Percentiles
print(f"\nPercentiles:")
for pctl in [5, 25, 50, 75, 95]:
    val = np.percentile(final_deltas, pctl)
    print(f"  {pctl}th: {val*100:.3f}%")

print()

# 6. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Effective vs Force vs Nominal
ax = axes[0, 0]
i_range = np.linspace(0, 0.5, 100)
delta_range = np.log(1 + i_range)
nominal_range = i_range  # Annual compounded annually for this comparison

ax.plot(i_range * 100, i_range * 100, linewidth=2.5, label='Effective i', color='darkblue')
ax.plot(i_range * 100, delta_range * 100, linewidth=2.5, label='Force δ', color='red')
ax.fill_between(i_range * 100, delta_range * 100, i_range * 100, 
               alpha=0.2, color='gray', label='Spread (i - δ)')
ax.set_xlabel('Effective Annual Rate i (%)', fontsize=11)
ax.set_ylabel('Rate Value (%)', fontsize=11)
ax.set_title('Effective Rate vs Force of Interest', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Continuous accumulation
ax = axes[0, 1]
t_range = np.linspace(0, 20, 100)
accumulation_continuous = np.exp(0.05 * t_range)
accumulation_discrete = (1 + 0.05127) ** t_range

ax.plot(t_range, accumulation_continuous, linewidth=2.5, label='Continuous (e^{δt})', color='red')
ax.plot(t_range, accumulation_discrete, linewidth=2.5, label='Discrete ((1+i)^t)', color='blue')
ax.fill_between(t_range, accumulation_discrete, accumulation_continuous, 
               alpha=0.2, color='gray')
ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Accumulated $1', fontsize=11)
ax.set_title('Continuous vs Discrete Accumulation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Continuous annuity present value
ax = axes[1, 0]
n_annuity = np.linspace(0.1, 30, 100)
a_annuity = (1 - np.exp(-0.05 * n_annuity)) / 0.05

ax.plot(n_annuity, a_annuity, linewidth=2.5, color='darkgreen')
ax.axhline(1/0.05, color='r', linestyle='--', linewidth=2, label=f'Perpetuity = {1/0.05:.1f}')
ax.fill_between(n_annuity, 0, a_annuity, alpha=0.2, color='green')
ax.set_xlabel('Term (years)', fontsize=11)
ax.set_ylabel('a̅ₙ̄| (annuity factor)', fontsize=11)
ax.set_title('Continuous Annuity Present Value Factor', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 4: Stochastic paths (Vasicek)
ax = axes[1, 1]
# Plot sample paths
for i in range(0, n_sims, n_sims // 100):
    ax.plot(times_vasicek, delta_paths[i, :] * 100, alpha=0.05, color='blue')

# Plot mean and percentiles
mean_path = np.mean(delta_paths, axis=0)
p5_path = np.percentile(delta_paths, 5, axis=0)
p95_path = np.percentile(delta_paths, 95, axis=0)

ax.plot(times_vasicek, mean_path * 100, linewidth=2.5, color='red', label='Mean')
ax.plot(times_vasicek, p5_path * 100, linewidth=2, linestyle='--', color='orange', label='5th percentile')
ax.plot(times_vasicek, p95_path * 100, linewidth=2, linestyle='--', color='orange', label='95th percentile')
ax.axhline(b * 100, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Long-term mean = {b*100:.1f}%')
ax.fill_between(times_vasicek, p5_path * 100, p95_path * 100, alpha=0.2, color='orange')

ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Force of Interest δ(t) (%)', fontsize=11)
ax.set_title('Vasicek Model: Stochastic Force Simulation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('force_of_interest_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When force of interest misleads:
- **Non-constant δ(t)**: Real economies have variable rates; assuming constant δ breaks down mid-contract
- **Stochastic assumptions**: Vasicek can generate negative rates (realistic but complicates pricing); CIR avoids negative but less realistic
- **Discretization error**: Annual compounding (1+i)^n approximates continuous e^{δn}; error grows with rate magnitude
- **Inverse problem**: Given bond prices, solve for implied δ(t); non-unique for short-term, requires curve smoothing
- **Mortality stochasticity**: Assuming constant δ in mortality models ignores cohort effects; Lee-Carter needed for realistic projections

## 7. Key References
- [Force of Interest (Wikipedia)](https://en.wikipedia.org/wiki/Force_of_interest) - Mathematical definition
- [Vasicek Interest Rate Model](https://en.wikipedia.org/wiki/Vasicek_model) - Stochastic modeling framework
- [Bowers et al., Actuarial Mathematics (Chapter 3)](https://www.soa.org/) - Continuous payment formulas

---
**Status:** Theoretical foundation | **Complements:** Effective Annual Rate, Stochastic Models, Continuous Annuities
