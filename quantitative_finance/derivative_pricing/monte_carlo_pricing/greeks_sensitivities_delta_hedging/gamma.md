# Gamma

## 1. Concept Skeleton
**Definition:** The second derivative of option price with respect to asset price (Γ = ∂²C/∂S² = ∂Δ/∂S), measuring the rate of delta change and the convexity of the option value function.  
**Purpose:** Quantify curvature risk, identify rehedging frequency requirements, measure trading profitability from volatility, assess option buyer vs seller advantage in turbulent markets  
**Prerequisites:** Delta, option pricing, convexity, second-order Taylor expansion, volatility dynamics

## 2. Comparative Framing

| Aspect | Long Option (Buyer) | Short Option (Seller) | Long Straddle | Long Strangle |
|--------|---------------------|----------------------|---------------|---------------|
| **Gamma Sign** | Γ > 0 (positive) | Γ < 0 (negative) | Γ > 0 | Γ > 0 |
| **Delta Behavior** | Δ increases with S | Δ decreases with S | Δ ≈ 0 (calls & puts cancel) | Similar |
| **Stock Move Impact** | Profit from big moves | Loss from big moves | Profit if stock moves | Profit if stock moves |
| **Time Decay (θ)** | Typically negative | Typically positive | Negative (theta drag) | Less negative |
| **Profitability Depends On** | Realized > implied vol | Realized < implied vol | Realized > implied vol | Realized > implied vol |
| **Rehedging Losses** | Gain when rehedging | Lose when rehedging | Gain from rehedges | Gain from rehedges |
| **Peak Gamma** | ATM (strike nearest) | ATM | ATM | Between strikes |

## 3. Examples + Counterexamples

**Simple Example: Long Call Gamma**  
Long call: S=100, K=100, Γ=0.02. If S → 101 (move +1): delta changes from 0.50 to ~0.60 (gains +0.10 due to gamma). Profit from rehedge: buy low/sell high. If S → 99 (move -1): delta changes from 0.50 to ~0.40 (loses -0.10). But profit still realized on first move (if happens first). Over full period: realized volatility determines outcome.

**Failure Case: Short Straddle in Crisis**  
Sell ATM straddle (sell call + put), collect theta. Market calm: theta decay profits. Suddenly: market crashes 10% → S moves from 100 to 90. Gamma loss = 0.5 × Γ × (10)² ≈ significant. Long call & put both become ITM; huge losses. Gamma loss >> theta collected. Crisis amplifies: volatility spikes → vega loss. Lesson: selling gamma in calm markets is picking pennies in front of steamroller.

**Edge Case: Deep ITM/OTM Options**  
Deep ITM call (S=200, K=100): Γ ≈ 0 (delta ≈ 1, nearly constant). Stock move $5 → delta unchanged, gamma P&L ≈ 0. Deep OTM call (S=50, K=100): Γ ≈ 0 (delta ≈ 0, stays 0). Gamma peaks ATM: small move causes large delta swings → maximum rehedging benefit/cost.

## 4. Layer Breakdown

```
Gamma Framework:
├─ Definition & Interpretation:
│   ├─ Gamma Γ = ∂²C/∂S² = ∂Δ/∂S
│   ├─ Measures curvature of option price function
│   ├─ Always positive for both calls & puts (convex payoff)
│   ├─ Units: 1 / stock price (e.g., 0.01 per dollar)
│   ├─ Intuition:
│   │   ├─ High gamma: delta sensitive to spot moves (rehedge often)
│   │   ├─ Low gamma: delta stable (infrequent rehedging)
│   │   └─ Peak gamma: ATM (greatest uncertainty, most rehedging)
│   └─ Relationship to Volatility:
│       ├─ Gamma ∝ 1/σ (lower vol → higher gamma; need more frequent rehedges)
│       ├─ Gamma ∝ 1/√T (shorter time → higher gamma; near expiry, very sensitive)
│       └─ Implication: selling options when vol low + T short is dangerous (gamma blowup)
├─ Black-Scholes Formula:
│   ├─ Call/Put Gamma (same): Γ = e^{-qT} n(d1) / (S σ √T)
│   ├─ Where n(d1) = (1/√(2π)) exp(-d1²/2) (standard normal PDF)
│   ├─ d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
│   ├─ Properties:
│   │   ├─ Peaks when d1 ≈ 0 (ATM condition)
│   │   ├─ Decreases as T → ∞ (long-dated options less sensitive)
│   │   ├─ Increases as T → 0 (approaches singularity)
│   │   └─ Monotonically decreases away from ATM
│   └─ ATM Gamma Approximation:
│       └─ Γ_ATM ≈ 1 / (S × σ × √(2πT))
├─ Gamma P&L:
│   ├─ Taylor Expansion: ΔC ≈ Δ × ΔS + ½ Γ × (ΔS)²
│   ├─ Second term = Gamma P&L: ½ Γ × (ΔS)²
│   ├─ Realized Gamma P&L (path-dependent):
│   │   ├─ Long gamma profits: gains regardless of direction (both up/down moves)
│   │   ├─ Short gamma losses: loses on both large moves
│   │   └─ Formula: cumulative ½ Σ Γᵢ × (ΔSᵢ)²
│   ├─ Continuous Formula:
│   │   ├─ Gamma P&L ≈ ½ Γ × [realized variance] × T
│   │   ├─ Where realized variance = (1/T) Σ (ln(Sᵢ₊₁/Sᵢ))²
│   │   └─ Interpretation: profit = long gamma × realized vol²
│   └─ Hedging Dynamics:
│       ├─ Buy low, sell high (from rehedging) when Γ > 0
│       ├─ Sell low, buy high (forced) when Γ < 0
│       └─ Net: gamma P&L accumulates over path, not just final move
├─ Practical Implications:
│   ├─ Portfolio Gamma:
│   │   ├─ Γ_portfolio = Σ Γᵢ × qᵢ (sum across all positions)
│   │   ├─ Positive: long options, profit from vol
│   │   └─ Negative: short options, lose from vol
│   ├─ Rehedging Frequency:
│   │   ├─ High gamma → daily rehedging necessary
│   │   ├─ Low gamma → weekly/monthly acceptable
│   │   └─ Optimization: balance gamma loss vs transaction costs
│   ├─ Risk Management:
│   │   ├─ VaR/stress: gamma amplifies losses in tail (non-linear risk)
│   │   ├─ Gamma explosion: near expiry ATM, tiny moves cause huge P&L swings
│   │   └─ Monitoring: track gamma daily; cap per desk/trader
│   └─ Trading Strategies:
│       ├─ Long gamma trades: buy straddle/strangle, profit from volatility
│       ├─ Short gamma trades: sell spreads, profit from time decay (require vol < realized)
│       └─ Gamma scalping: long gamma position, rehedge frequently to lock in vol spreads
└─ Gamma Convexity:
    ├─ Option value is convex in spot price
    ├─ Buyer (long option) benefits from uncertainty (convexity advantage)
    ├─ Seller (short option) loses from uncertainty (forced to rehedge at losses)
    └─ This is why optionality has value beyond expected payoff
```

**Interaction:** Gamma drives rehedging P&L; theta (time decay) partially offset by gamma losses in delta hedge; net P&L depends on realized vol vs implied vol.

## 5. Mini-Project

Measure gamma P&L; compare long straddle vs short straddle under different volatility scenarios:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GammaAnalysis:
    """Analyze gamma effects on hedged positions"""
    
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def bs_call_price(self, S, t):
        tau = self.T - t
        if tau <= 0:
            return max(S - self.K, 0)
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        return S*np.exp(-self.q*tau)*norm.cdf(d1) - \
               self.K*np.exp(-self.r*tau)*norm.cdf(d2)
    
    def bs_gamma(self, S, t):
        tau = self.T - t
        if tau <= 0:
            return 0.0
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        return np.exp(-self.q*tau)*norm.pdf(d1) / (S*self.sigma*np.sqrt(tau))
    
    def bs_theta(self, S, t):
        tau = self.T - t
        if tau <= 0:
            return 0.0
        d1 = (np.log(S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*tau) / \
             (self.sigma*np.sqrt(tau))
        d2 = d1 - self.sigma*np.sqrt(tau)
        theta_annual = -S*np.exp(-self.q*tau)*norm.pdf(d1)*self.sigma/(2*np.sqrt(tau)) + \
                      self.q*S*np.exp(-self.q*tau)*norm.cdf(d1) - \
                      self.r*self.K*np.exp(-self.r*tau)*norm.cdf(d2)
        return theta_annual / 365
    
    def gamma_pnl_path(self, S_path, dt):
        """
        Compute gamma P&L along a price path
        S_path: array of prices [S_0, S_1, ..., S_n]
        dt: time step (years)
        """
        n = len(S_path) - 1
        gamma_pnl = np.zeros(n)
        times = np.linspace(0, self.T, n+1)
        
        for i in range(n):
            t = times[i]
            S_current = S_path[i]
            S_next = S_path[i+1]
            dS = S_next - S_current
            
            gamma = self.bs_gamma(S_current, t)
            gamma_pnl[i] = 0.5 * gamma * (dS**2)
        
        return np.cumsum(gamma_pnl)
    
    def theta_pnl_path(self, S_path, dt):
        """
        Compute theta P&L along a path
        """
        n = len(S_path) - 1
        theta_pnl = np.zeros(n)
        times = np.linspace(0, self.T, n+1)
        
        for i in range(n):
            t = times[i]
            S = S_path[i]
            theta = self.bs_theta(S, t)
            theta_pnl[i] = theta * 1  # Per day
        
        return np.cumsum(theta_pnl)
    
    def simulate_pnl(self, n_paths=1000, n_days=252, realized_vol=None):
        """
        Simulate P&L for long & short straddles
        realized_vol: volatility to use for path generation (if None, use self.sigma)
        """
        if realized_vol is None:
            realized_vol = self.sigma
        
        np.random.seed(42)
        dt = self.T / n_days
        
        # Generate paths
        Z = np.random.randn(n_paths, n_days)
        S_paths = np.zeros((n_paths, n_days + 1))
        S_paths[:, 0] = self.S
        
        for t in range(n_days):
            S_paths[:, t+1] = S_paths[:, t] * np.exp(
                (self.r - self.q - 0.5*realized_vol**2)*dt + 
                realized_vol*np.sqrt(dt)*Z[:, t]
            )
        
        # Compute P&L components
        long_straddle_pnl = []
        short_straddle_pnl = []
        gamma_pnl_list = []
        theta_pnl_list = []
        
        for p in range(n_paths):
            gamma_pnl = self.gamma_pnl_path(S_paths[p, :], dt)
            theta_pnl = self.theta_pnl_path(S_paths[p, :], dt)
            
            # Realized payoff at expiry
            final_S = S_paths[p, -1]
            call_payoff = max(final_S - self.K, 0)
            put_payoff = max(self.K - final_S, 0)
            
            # Initial straddle cost
            call_price = self.bs_call_price(self.S, 0)
            put_price = self.bs_call_price(self.S, 0) - self.S*np.exp(-self.q*self.T) + \
                       self.K*np.exp(-self.r*self.T)
            # (Simplified put price; normally would use BS directly)
            
            straddle_cost = call_price + put_price
            
            # Long straddle: pay premium, receive payoff, profit from gamma + theta
            long_pnl = call_payoff + put_payoff - straddle_cost + gamma_pnl[-1] + theta_pnl[-1]
            
            # Short straddle: receive premium, pay payoff, lose from gamma - theta
            short_pnl = straddle_cost - (call_payoff + put_payoff) - gamma_pnl[-1] - theta_pnl[-1]
            
            long_straddle_pnl.append(long_pnl)
            short_straddle_pnl.append(short_pnl)
            gamma_pnl_list.append(gamma_pnl[-1])
            theta_pnl_list.append(theta_pnl[-1])
        
        return {
            'long_straddle_pnl': np.array(long_straddle_pnl),
            'short_straddle_pnl': np.array(short_straddle_pnl),
            'gamma_pnl': np.array(gamma_pnl_list),
            'theta_pnl': np.array(theta_pnl_list),
            'final_stock_move': S_paths[:, -1] - self.S
        }

# Parameters
S, K, T, r, sigma, q = 100, 100, 0.25, 0.05, 0.20, 0.02
analyzer = GammaAnalysis(S, K, T, r, sigma, q)

print("="*70)
print("GAMMA ANALYSIS")
print("="*70)

# Current gamma at ATM
gamma_atm = analyzer.bs_gamma(S, 0)
print(f"\nCurrent Gamma (ATM): {gamma_atm:.6f}")
print(f"Interpretation: Delta changes by {gamma_atm:.4f} per $1 stock move")

# Simulate under different realized volatility scenarios
print(f"\n{'Realized Vol':^20} {'Long Straddle Mean P&L':^25} {'Short Straddle Mean P&L':^25}")
print("-"*70)

for realized_vol in [0.10, 0.15, 0.20, 0.30, 0.40]:
    res = analyzer.simulate_pnl(n_paths=5000, n_days=252, realized_vol=realized_vol)
    long_mean = np.mean(res['long_straddle_pnl'])
    short_mean = np.mean(res['short_straddle_pnl'])
    print(f"{realized_vol*100:>6.1f}%     ${long_mean:>20.2f}    ${short_mean:>20.2f}")

# Detailed simulation with implied vol = 20%
res_details = analyzer.simulate_pnl(n_paths=10000)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Gamma profile across spot prices
spot_range = np.linspace(80, 120, 100)
gammas = [analyzer.bs_gamma(s, 0) for s in spot_range]

axes[0, 0].plot(spot_range, gammas, 'b-', linewidth=2)
axes[0, 0].axvline(S, color='r', linestyle=':', alpha=0.7, label='ATM')
axes[0, 0].fill_between(spot_range, 0, gammas, alpha=0.2)
axes[0, 0].set_title('Gamma Profile (Across Spot Prices)')
axes[0, 0].set_xlabel('Spot Price S')
axes[0, 0].set_ylabel('Gamma')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Gamma vs Time to Maturity
times = np.linspace(T, 0.01, 50)
gammas_time = [analyzer.bs_gamma(S, T-t) for t in times]

axes[0, 1].plot(times*365, gammas_time, 'b-', linewidth=2)
axes[0, 1].set_title('Gamma vs Time to Maturity (ATM)')
axes[0, 1].set_xlabel('Days to Expiry')
axes[0, 1].set_ylabel('Gamma')
axes[0, 1].invert_xaxis()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Gamma P&L vs Stock Move
axes[0, 2].scatter(res_details['final_stock_move'], res_details['gamma_pnl'], 
                  alpha=0.3, s=10)
# Theoretical curve: 0.5 * gamma * (move)^2
move_range = np.linspace(-20, 20, 100)
theoretical_pnl = 0.5 * gamma_atm * (move_range**2)
axes[0, 2].plot(move_range, theoretical_pnl, 'r-', linewidth=2, label='Theoretical')
axes[0, 2].set_title('Gamma P&L vs Stock Move')
axes[0, 2].set_xlabel('Stock Price Move ($)')
axes[0, 2].set_ylabel('Gamma P&L ($)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Long Straddle P&L Distribution
axes[1, 0].hist(res_details['long_straddle_pnl'], bins=50, alpha=0.7, edgecolor='blue')
axes[1, 0].axvline(np.mean(res_details['long_straddle_pnl']), color='r', 
                   linestyle='--', linewidth=2, label=f"Mean: ${np.mean(res_details['long_straddle_pnl']):.2f}")
axes[1, 0].axvline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 0].set_title('Long Straddle P&L Distribution')
axes[1, 0].set_xlabel('P&L ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Short Straddle P&L Distribution
axes[1, 1].hist(res_details['short_straddle_pnl'], bins=50, alpha=0.7, edgecolor='red')
axes[1, 1].axvline(np.mean(res_details['short_straddle_pnl']), color='r', 
                   linestyle='--', linewidth=2, label=f"Mean: ${np.mean(res_details['short_straddle_pnl']):.2f}")
axes[1, 1].axvline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 1].set_title('Short Straddle P&L Distribution')
axes[1, 1].set_xlabel('P&L ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Gamma P&L vs Theta P&L (decomposition)
axes[1, 2].scatter(res_details['gamma_pnl'], res_details['theta_pnl'], 
                  alpha=0.3, s=10, c=res_details['final_stock_move'], cmap='RdYlGn')
axes[1, 2].set_title('Gamma P&L vs Theta P&L')
axes[1, 2].set_xlabel('Gamma P&L ($)')
axes[1, 2].set_ylabel('Theta P&L ($)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gamma_analysis.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: gamma_analysis.png")
```

**Output Interpretation:**
- **Gamma Profile:** Peaks ATM; near zero OTM/ITM
- **P&L Distribution:** Long straddle profits on volatility; short straddle loses on big moves
- **Gamma vs Theta:** Tradeoff for short options; theta gain offset by gamma loss in volatile markets

## 6. Challenge Round

**Q1: Gamma is always positive for both calls and puts. Why? What does this tell us about option value convexity?**  
A: Option payoff is convex in spot price: call max(S-K, 0) has slope 0 then 1 (convex kink), put max(K-S, 0) has slope -1 then 0 (also convex). Since Γ = ∂²C/∂S² and convex functions have positive second derivatives, Γ > 0 always. Implication: option holders have "convexity advantage"— they profit from uncertainty (large moves in either direction) whereas linear instruments (stock, forward) are indifferent. This is why buying options in calm markets (low premium) can be profitable; realized vol might exceed implied.

**Q2: Why does gamma increase as time to expiry decreases (especially near ATM)? Why do traders fear "gamma explosion" on expiration day?**  
A: Gamma ∝ 1/(σ√T). As T → 0, gamma → ∞. Intuitively: near expiry, small spot moves can flip option from OTM (worth $0) to ITM (worth ΔS). Delta becomes discontinuous at strike at expiry. Traders fear gamma explosion because: (1) tiny stock moves cause huge delta swings, (2) rehedging becomes impossible/costly (bid-ask spread too wide), (3) in a crash, can't rehedge fast enough (gap risk). Example: short 1000 ATM calls on expiry day, delta jumps from 0.50 to 0.90 in seconds → forced to buy 40,000 shares in illiquid market → massive loss.

**Q3: A portfolio is "long gamma" (Γ > 0) and delta-hedged (Δ ≈ 0). What is the realized P&L if stock volatility doubles?**  
A: Realized P&L = ½ Γ × [realized variance]. If realized variance doubles, P&L ≈ ½ Γ × 2×V ≈ doubles (roughly). But variance compounds, not just vol; if vol doubles, daily moves double → variance quadruples → realized gamma P&L ≈ quadruples. Practical example: long straddle with Γ=0.02, realized vol increases from 20% → 30%: daily realized variance increases from (0.20/√252)² ≈ 0.0000001 to (0.30/√252)² ≈ 0.00000225 (2.25× larger). Over 1 year: cumulative P&L ≈ ½ × 0.02 × (excess variance realized) = significant profit.

**Q4: Explain the P&L decomposition: net P&L = Theta P&L - Gamma Loss + Vega P&L + Rho P&L for a delta-hedged short call.**  
A: Delta-hedged: Δ ≈ 0, directional risk eliminated. Remaining sources: (1) **Theta (+)**: short option benefits from time decay; collect daily theta. (2) **Gamma (--)**: gamma loss = 0.5 × Γ × (realized variance). Short call Γ < 0, so gamma loss is negative (money out). (3) **Vega (-)**: if IV rises, short call value increases (negative P&L for seller). (4) **Rho (-)**: if rates rise, call value typically increases (slight negative). Net: P&L = θ × Δt - 0.5|Γ|×var_realized - ν×ΔIV - ρ×Δr. Profitable if: theta gain > gamma loss + vol change loss + rate change effect.

**Q5: A trader has Γ=0.01 per dollar and holds delta-neutral position. What stock move causes $1,000 gamma P&L?**  
A: Gamma P&L = 0.5 × Γ × (ΔS)². Setting this equal to 1000: 0.5 × 0.01 × (ΔS)² = 1000 → (ΔS)² = 200,000 → ΔS ≈ $447. So 47% move causes $1k gamma P&L. This illustrates gamma leverage: small portfolio gamma can generate large P&L from large market moves. For daily monitoring: if S=$100, 1% daily move → ΔS = $1 → gamma P&L = 0.5×0.01×1² = $0.005 (negligible). But 20% annual move (sqrt(252) = 16× daily) → cumulative gamma P&L ≈ $0.005 × 252 ≈ $1.26 (plus compounding). Reason trader monitors gamma: large moves compound gamma P&L over time or in single event.

**Q6: A bank sells 100,000 ATM calls expiring in 7 days. Why is this position extraordinarily risky despite being delta-hedged?**  
A: 7 days to expiry: gamma is extremely high (Γ ∝ 1/√T → huge). ATM: Γ at peak. Small stock move → delta swings dramatically → hedge stale immediately → forced to rebalance at market prices with widened spreads (end-of-week liquidity low). Notional huge (100k calls). Scenario: stock gaps up 2% overnight → delta jumps 0.50→0.90 → must buy 40k shares, each bid-ask widens on EOD → slippage ≈ 0.10/share → loss = $4M (just on rehedge, before underlying move). Classic: short-dated, ATM, large notional = maximum gamma risk. Banks manage this with strict gamma limits by desk/trader; gamma risk committee monitors daily.

## 7. Key References

- [Wikipedia: Gamma (Finance)](https://en.wikipedia.org/wiki/Gamma_(finance)) — Definition, convexity, ATM peak
- [Wikipedia: Greeks (Finance)](https://en.wikipedia.org/wiki/Greeks_(finance)) — Gamma in context of other Greeks
- Hull: *Options, Futures & Derivatives* (Chapter 19) — Gamma dynamics, hedging P&L decomposition
- Paul Wilmott: *Introduces Quantitative Finance* — Gamma trading, volatility speculation, rehedging mechanics

**Status:** ✓ Standalone file. **Complements:** delta.md, vega.md, theta.md, greeks_interactions.md, delta_hedging_strategies.md
