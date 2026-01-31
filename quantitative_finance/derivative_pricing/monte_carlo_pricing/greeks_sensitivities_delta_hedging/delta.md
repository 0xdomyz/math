# Delta

## 1. Concept Skeleton
**Definition:** The rate of change of option price with respect to underlying asset price; mathematically Δ = ∂C/∂S, representing the hedge ratio for delta-neutral replication.  
**Purpose:** Quantify directional risk exposure, determine stock position needed to hedge option portfolio, replicate option value through continuous stock trading, manage market-neutral strategies  
**Prerequisites:** Option pricing, partial derivatives, hedging strategies, portfolio dynamics, risk-neutral valuation

## 2. Comparative Framing

| Greek | Delta (Δ) | Gamma (Γ) | Vega (ν) | Theta (θ) |
|-------|-----------|-----------|----------|-----------|
| **Definition** | ∂C/∂S (price change) | ∂²C/∂S² (delta change) | ∂C/∂σ (vol change) | ∂C/∂t (time decay) |
| **Units** | Dimensionless [-1, 1] | [1/stock price] | [dollars/1% vol] | [dollars/day] |
| **Call Value** | 0 to 1 (increases with S) | Always > 0 (convex) | > 0 (long vol) | Usually < 0 (theta decay) |
| **Put Value** | -1 to 0 (decreases with S) | Always > 0 (convex) | > 0 (long vol) | Often > 0 (time helps) |
| **Hedging Goal** | Delta-neutral: Δ_portfolio = 0 | Rehedge when Δ moves | Hedge vol exposure separately | Exploit time decay |
| **Rebalancing Frequency** | Daily to intraday | Weekly | Monthly | Continuous |

## 3. Examples + Counterexamples

**Simple Example: Delta Hedging a Short Call**  
Sell call: K=100, S₀=100, Δ=0.60. Obligated to hedge. Action: buy 60 shares (replicate delta exposure). If S → 101: call price rises $0.60 (approximately), short call loses $0.60, long 60 shares gain $60 = breaks even on delta (ignoring gamma). Repeat daily as delta changes.

**Failure Case: Ignoring Gamma**  
Delta hedge assumes linear relationship. Reality: S → 101, call Δ increases (gamma effect). Initial hedge of 60 shares insufficient. Stock up $60, call down $61 (more negative due to gamma). Net loss on rehedge: must buy more shares at higher price. Lesson: delta hedges decay and require frequent rebalancing; gamma loss accumulates.

**Edge Case: Deep ITM Call**  
S = 150, K = 100: Δ ≈ 1 (behaves like stock). Hedge: be short stock (delta-neutral). If S drops to 140: Δ → 0.95 (small change). Gamma very low (delta insensitive to moves). Rehedging needed rarely. Contrast: ATM (S=K): Δ=0.5, gamma peak, delta swings ±0.10 on 1% move.

## 4. Layer Breakdown

```
Delta Framework:
├─ Definition & Interpretation:
│   ├─ Delta Δ = ∂C/∂S = rate of price change w.r.t. spot
│   ├─ For small moves: ΔC ≈ Δ × ΔS (linear approximation)
│   ├─ Call Delta: ∈ [0, 1] (always non-negative)
│   │   ├─ ATM: Δ ≈ 0.5 (equally likely up/down)
│   │   ├─ ITM (S > K): Δ → 1 (acts like stock)
│   │   └─ OTM (S < K): Δ → 0 (expires worthless)
│   ├─ Put Delta: ∈ [-1, 0] (always non-positive)
│   │   ├─ ATM: Δ ≈ -0.5 (negative directional exposure)
│   │   ├─ ITM (S < K): Δ → -1 (like short stock)
│   │   └─ OTM (S > K): Δ → 0 (worthless)
│   └─ Relationship: Δ_call - Δ_put = e^{-qT} (always)
├─ Black-Scholes Formula:
│   ├─ Call Delta: Δ_C = e^{-qT} N(d1)
│   ├─ Put Delta: Δ_P = -e^{-qT} N(-d1) = e^{-qT} (N(d1) - 1)
│   ├─ Where d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
│   ├─ Properties:
│   │   ├─ As S → ∞: Δ_call → 1, Δ_put → 0
│   │   ├─ As S → 0: Δ_call → 0, Δ_put → -1
│   │   ├─ Δ increases monotonically with S (for calls)
│   │   └─ Maximum sensitivity at ATM (peak gamma region)
│   └─ Dividend Adjustment:
│       └─ Dividend yield q reduces Δ_call (forward price lower)
├─ Delta Hedging Strategy:
│   ├─ Goal: Create delta-neutral portfolio (Δ_total = 0)
│   ├─ Method: For each option sold, buy Δ shares
│   ├─ P&L Decomposition:
│   │   ├─ Delta P&L: ≈ 0 (hedged)
│   │   ├─ Gamma P&L: ≈ ½ Γ (ΔS)² (loss if short gamma)
│   │   ├─ Vega P&L: ≈ ν Δσ (loss if short vega, vol rises)
│   │   └─ Theta P&L: ≈ θ Δt (gain if holding positive theta)
│   ├─ Rehedging Dynamics:
│   │   ├─ Buy when delta increases (stock up → more hedging needed)
│   │   ├─ Sell when delta decreases (stock down → reduce hedge)
│   │   └─ Frequency: balances transaction costs vs residual risk
│   └─ Break-even Analysis:
│       ├─ Profit if realized volatility < implied vol (theta exceeds gamma loss)
│       ├─ Loss if realized vol > implied vol (gamma loss exceeds theta gain)
│       └─ Equilibrium: break-even when σ_realized = σ_implied
├─ Practical Implementation:
│   ├─ Continuous hedging (theory): rebalance infinitely often, zero P&L
│   ├─ Discrete hedging (practice): rehedge daily/hourly, realize P&L
│   ├─ Transaction costs: bid-ask spread + commissions reduce profitability
│   ├─ Model risk: delta wrong if market moves jump, vol regime changes
│   └─ Liquidity risk: cannot execute hedge if stock illiquid
└─ Portfolio Delta:
    ├─ Δ_portfolio = Σ Δᵢ × qᵢ (summed across all positions)
    ├─ Hedge: short Δ_portfolio shares to neutralize
    ├─ Monitoring: track daily as underlying moves, volatility changes
    └─ Reporting: key risk metric (VaR, stress testing)
```

**Interaction:** Delta changes with S (non-linearly due to gamma); requires rehedging. Frequency depends on gamma magnitude and transaction costs.

## 5. Mini-Project

Implement delta hedging a short call position; simulate P&L, gamma loss, rehedging frequency:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class DeltaHedgingSimulation:
    """Simulate delta hedging a short call over time"""
    
    def __init__(self, S0, K, T, r, sigma, q=0, rehedge_freq=1):
        """
        rehedge_freq: days between rehedges (1=daily, 7=weekly, etc.)
        """
        self.S0 = S0
        self.K = K
        self.T = T  # years
        self.r = r
        self.sigma = sigma
        self.q = q
        self.rehedge_freq = rehedge_freq
    
    def bs_call_price(self, S, t):
        """Call price at time t, spot S"""
        tau = self.T - t
        if tau <= 0:
            return max(S - self.K, 0)
        
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        return S*np.exp(-self.q*tau)*norm.cdf(d1) - \
               self.K*np.exp(-self.r*tau)*norm.cdf(d2)
    
    def bs_delta(self, S, t):
        """Call delta at time t, spot S"""
        tau = self.T - t
        if tau <= 0:
            return 1.0 if S > self.K else 0.0
        
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        return np.exp(-self.q*tau)*norm.cdf(d1)
    
    def bs_gamma(self, S, t):
        """Call gamma at time t, spot S"""
        tau = self.T - t
        if tau <= 0:
            return 0.0
        
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        return np.exp(-self.q*tau)*norm.pdf(d1) / (S*self.sigma*np.sqrt(tau))
    
    def simulate_hedging(self, n_paths=1000, n_days=252, seed=42):
        """
        Simulate delta hedging over time
        Returns: paths, portfolio values, delta values, P&L components
        """
        np.random.seed(seed)
        dt = self.T / n_days
        
        # Generate stock paths
        Z = np.random.randn(n_paths, n_days)
        S_paths = np.zeros((n_paths, n_days + 1))
        S_paths[:, 0] = self.S0
        
        for t in range(n_days):
            S_paths[:, t+1] = S_paths[:, t] * np.exp(
                (self.r - self.q - 0.5*self.sigma**2)*dt + 
                self.sigma*np.sqrt(dt)*Z[:, t]
            )
        
        # Track hedging dynamics
        option_values = np.zeros((n_paths, n_days + 1))
        deltas = np.zeros((n_paths, n_days + 1))
        hedge_shares = np.zeros((n_paths, n_days + 1))
        portfolio_values = np.zeros((n_paths, n_days + 1))
        pnl = np.zeros((n_paths, n_days + 1))
        gamma_pnl = np.zeros((n_paths, n_days + 1))
        theta_pnl = np.zeros((n_paths, n_days + 1))
        
        # Initial: sell call, buy delta shares
        for p in range(n_paths):
            option_values[p, 0] = self.bs_call_price(self.S0, 0)
            deltas[p, 0] = self.bs_delta(self.S0, 0)
            hedge_shares[p, 0] = deltas[p, 0]
            
            # Initial portfolio: short 1 call, long delta shares
            portfolio_values[p, 0] = hedge_shares[p, 0]*self.S0 - option_values[p, 0]
        
        # Simulate day-by-day
        rehedge_dates = np.arange(0, n_days, self.rehedge_freq, dtype=int)
        
        for day in range(1, n_days + 1):
            for p in range(n_paths):
                t = day * dt
                S = S_paths[p, day]
                
                # Option value at new spot
                option_values[p, day] = self.bs_call_price(S, t)
                deltas[p, day] = self.bs_delta(S, t)
                
                # Portfolio value before rehedge: hedge_shares * S - option_value
                portfolio_before = hedge_shares[p, day-1]*S - option_values[p, day]
                
                # Check if rehedging day
                if day in rehedge_dates:
                    # Rehedge: adjust shares to new delta
                    new_shares = deltas[p, day]
                    shares_to_trade = new_shares - hedge_shares[p, day-1]
                    
                    # Transaction costs (simplified: assume mid-price)
                    # In reality, would use bid-ask spread
                    hedge_shares[p, day] = new_shares
                else:
                    # No rehedging, maintain old hedge
                    hedge_shares[p, day] = hedge_shares[p, day-1]
                
                # Final portfolio value
                portfolio_values[p, day] = hedge_shares[p, day]*S - option_values[p, day]
                
                # Realized P&L
                pnl[p, day] = portfolio_values[p, day] - portfolio_values[p, 0]
                
                # Approximate P&L decomposition (using gamma/theta proxy)
                # Delta P&L ≈ 0 (hedged)
                # Gamma P&L ≈ ½ Γ (ΔS)²
                dS = S - S_paths[p, day-1]
                gamma = self.bs_gamma(S, t)
                gamma_pnl[p, day] = 0.5 * gamma * (dS**2)
                
                # Theta from time decay
                # Simple: theta * dt
                option_yesterday = self.bs_call_price(S_paths[p, day-1], t-dt)
                option_today_time_decay = self.bs_call_price(S_paths[p, day-1], t)
                theta_pnl[p, day] = option_yesterday - option_today_time_decay
        
        return {
            'paths': S_paths,
            'option_values': option_values,
            'deltas': deltas,
            'hedge_shares': hedge_shares,
            'portfolio_values': portfolio_values,
            'pnl': pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'final_payoff': np.maximum(S_paths[:, -1] - self.K, 0)
        }

# Parameters
S0, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02

# Simulate different rehedging frequencies
print("="*70)
print("DELTA HEDGING SIMULATION")
print("="*70)
print(f"\nParameters: S=${S0}, K=${K}, T={T} yrs, σ={sigma*100:.1f}%, r={r*100:.1f}%\n")

results_daily = DeltaHedgingSimulation(S0, K, T, r, sigma, q, rehedge_freq=1).simulate_hedging(n_paths=10000)
results_weekly = DeltaHedgingSimulation(S0, K, T, r, sigma, q, rehedge_freq=5).simulate_hedging(n_paths=10000)
results_biweekly = DeltaHedgingSimulation(S0, K, T, r, sigma, q, rehedge_freq=10).simulate_hedging(n_paths=10000)

print(f"{'Rehedging Frequency':^30} {'Mean P&L':^15} {'Std Dev':^15} {'Min P&L':^15} {'Max P&L':^15}")
print("-"*75)

for name, res in [("Daily", results_daily), ("Weekly", results_weekly), ("Bi-weekly", results_biweekly)]:
    final_pnl = res['pnl'][:, -1]
    print(f"{name:30} ${np.mean(final_pnl):^13.2f} ${np.std(final_pnl):^13.2f} ${np.min(final_pnl):^13.2f} ${np.max(final_pnl):^13.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Stock price paths
axes[0, 0].plot(results_daily['paths'][:100, :].T, alpha=0.3, linewidth=0.5, color='blue')
axes[0, 0].axhline(K, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Strike K')
axes[0, 0].set_title('100 Sample Stock Price Paths')
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Stock Price S')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Delta over time
sample_idx = 0
axes[0, 1].plot(results_daily['deltas'][sample_idx, :], linewidth=2, label='Daily Rehedge')
axes[0, 1].plot(results_weekly['deltas'][sample_idx, :], linewidth=2, label='Weekly Rehedge', alpha=0.7)
axes[0, 1].set_title('Delta Evolution (Sample Path)')
axes[0, 1].set_xlabel('Days')
axes[0, 1].set_ylabel('Delta')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Hedge shares over time
axes[0, 2].plot(results_daily['hedge_shares'][sample_idx, :], linewidth=2, label='Daily Rehedge', marker='o', markersize=3)
axes[0, 2].plot(results_weekly['hedge_shares'][sample_idx, :], linewidth=2, label='Weekly Rehedge', marker='s', markersize=3, alpha=0.7)
axes[0, 2].set_title('Hedge Position (Shares Held)')
axes[0, 2].set_xlabel('Days')
axes[0, 2].set_ylabel('Shares Held')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: P&L distribution for different rehedging frequencies
pnl_daily = results_daily['pnl'][:, -1]
pnl_weekly = results_weekly['pnl'][:, -1]
pnl_biweekly = results_biweekly['pnl'][:, -1]

axes[1, 0].hist(pnl_daily, bins=50, alpha=0.5, label='Daily', density=True)
axes[1, 0].hist(pnl_weekly, bins=50, alpha=0.5, label='Weekly', density=True)
axes[1, 0].hist(pnl_biweekly, bins=50, alpha=0.5, label='Bi-weekly', density=True)
axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Final P&L Distribution')
axes[1, 0].set_xlabel('P&L ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Cumulative P&L evolution (average)
axes[1, 1].plot(np.mean(results_daily['pnl'], axis=0), linewidth=2, label='Daily Rehedge')
axes[1, 1].plot(np.mean(results_weekly['pnl'], axis=0), linewidth=2, label='Weekly Rehedge', alpha=0.7)
axes[1, 1].plot(np.mean(results_biweekly['pnl'], axis=0), linewidth=2, label='Bi-weekly Rehedge', alpha=0.7)
axes[1, 1].fill_between(range(len(results_daily['pnl'][0, :])),
                         np.percentile(results_daily['pnl'], 5, axis=0),
                         np.percentile(results_daily['pnl'], 95, axis=0),
                         alpha=0.2, label='Daily 90% CI')
axes[1, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 1].set_title('Cumulative P&L Over Time (Mean ± 90% CI)')
axes[1, 1].set_xlabel('Days')
axes[1, 1].set_ylabel('P&L ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: P&L vs Realized Stock Move
final_stock_move = results_daily['paths'][:, -1] - S0
axes[1, 2].scatter(final_stock_move, pnl_daily, alpha=0.3, s=10)
axes[1, 2].plot(final_stock_move, 0.5*results_daily['gamma_pnl'][:, -1], 
               'r-', linewidth=2, label='Expected Gamma P&L', alpha=0.7)
axes[1, 2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 2].set_title('P&L vs Stock Move (Daily Rehedge)')
axes[1, 2].set_xlabel('Stock Price Move ($)')
axes[1, 2].set_ylabel('P&L ($)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('delta_hedging_simulation.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: delta_hedging_simulation.png")
```

**Output Interpretation:**
- **Delta Profile:** Increases from 0 (OTM) to 1 (ITM) as S increases
- **P&L Distribution:** Wider for less frequent rehedging (more gamma risk); mean P&L reflects theta vs gamma tradeoff
- **Gamma Loss:** Loss magnitude increases with stock moves; drives rehedging necessity

## 6. Challenge Round

**Q1: Why is delta always between 0 and 1 for calls, and between -1 and 0 for puts?**  
A: Delta measures hedge ratio needed to replicate the option. For calls: buying 1 call is like owning fraction Δ of stock (since call price moves Δ dollars per stock dollar). Δ ∈ [0, 1] because call value cannot exceed stock value (max payoff S_T) and cannot be negative. For puts: owning 1 put is like being short Δ shares (price moves inversely), hence Δ ∈ [-1, 0]. Boundary: deep ITM call Δ → 1 (acts like owning stock); deep OTM call Δ → 0 (expires worthless, no replication needed).

**Q2: A trader sells 1000 calls with delta 0.60. To delta-hedge, must the trader buy 600 shares? Is this enough for a profitable trade?**  
A: Yes, buy 600 shares to neutralize directional risk. But delta hedging is not a profitable trade by itself; it's risk management. P&L from short call + long 600 shares comes from: (1) Theta: time decay benefits short call (positive P&L daily if realizing lower vol). (2) Gamma loss: if stock moves significantly, delta changes → hedge becomes stale → rebalance at unfavorable prices (negative P&L). (3) Vega: if implied vol rises, short call loses (unhedged exposure). Profitability: realized vol < implied vol → theta gain > gamma loss → profit. Opposite → loss.

**Q3: Explain the relationship between delta and probability of exercise (ITM at expiry) for a European option.**  
A: Delta ≈ probability of ITM under risk-neutral measure. More precisely: for ATM call, Δ ≈ 0.50 ≈ P(S_T > K under Q-measure). But real probability is different (involves real drift μ, not risk-neutral r). Risk-neutral measure used for pricing (martingale property), while real probability for forecasting. Traders often conflate the two: "delta 0.60 = 60% chance ITM" is a mental shortcut, not rigorous. Rigorous: P(ITM under Q) ≈ N(d2) ≈ delta - (some correction term depending on gamma).

**Q4: Why must traders continuously rehedge delta exposure? Why not rehedge once per day?**  
A: Continuous rehedging (theory) eliminates all P&L except theta decay. In practice, discrete rehedging (daily/hourly) introduces gamma risk: between rehedges, delta changes; portfolio becomes unhedged. If stock moves large amount Δ S before next rehedge, portfolio loses ≈ 0.5 × gamma × (ΔS)² (realized gamma loss). Cost-benefit: more frequent rehedging (hourly) reduces gamma loss but increases transaction costs (bid-ask spread × trades). Optimal frequency balances gamma loss vs transaction cost; depends on vol, bid-ask, commission structure.

**Q5: How does dividend yield q affect delta? Why do deep ITM calls have delta < 1 if q > 0?**  
A: Forward price F = S e^{(r-q)T} includes dividend yield q. Higher q lowers forward (stock price drops ex-dividend dates), reducing expected S_T → lower call value & delta. BS delta includes e^{-qT} factor: Δ = e^{-qT} N(d1), which decreases with q. Intuition: dividend yield delays appreciation; better to own stock directly (get dividend) than own deep ITM call. Delta < 1 reflects this: even ITM call has reduced sensitivity to stock moves because dividend owner captures extra return.

**Q6: A delta-neutral portfolio is hedged. So it should have zero P&L, correct? If not, what explains any profit or loss?**  
A: Incorrect. Delta-neutral eliminates directional risk, but leaves residual Greeks unhedged. P&L sources: (1) **Gamma P&L**: ≈ 0.5 × Γ × (ΔS)². If short options (Γ < 0), large stock moves hurt. If long options (Γ > 0), large moves help. (2) **Vega P&L**: ≈ ν × Δσ. Vol changes (unhedged) affect portfolio. (3) **Theta P&L**: ≈ θ × Δt. Time decay (positive for short options, negative for long). (4) **Rho P&L**: ≈ ρ × Δr. Rate changes affect discount factors. Example: short call + delta hedge, assume vol constant & no rate change. P&L = θ × Δt - 0.5 × Γ × (realized_var - implied_var) × T. If realized vol < implied vol, theta gain exceeds gamma loss → profit.

## 7. Key References

- [Wikipedia: Greeks (Finance)](https://en.wikipedia.org/wiki/Greeks_(finance)) — All Greeks, definitions, sensitivities
- [Wikipedia: Delta (Finance)](https://en.wikipedia.org/wiki/Delta_(finance)) — Hedge ratio, probability interpretation
- Hull: *Options, Futures & Derivatives* (Chapter 19) — Greeks, hedging strategies, dynamic replication
- Paul Wilmott: *Introduces Quantitative Finance* — Intuitive Greeks, hedging P&L, gamma trading

**Status:** ✓ Standalone file. **Complements:** gamma.md, vega.md, theta.md, rho.md, delta_hedging_strategies.md, greeks_interactions.md
