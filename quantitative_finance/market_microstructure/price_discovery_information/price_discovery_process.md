# Price Discovery Process

## 1. Concept Skeleton
**Definition:** Mechanism by which markets incorporate new information into asset prices through trading activity  
**Purpose:** Aggregates dispersed private information, establishes consensus valuation, facilitates efficient capital allocation  
**Prerequisites:** Market efficiency, information asymmetry, order flow, trading mechanisms

## 2. Comparative Framing
| Venue | Continuous Trading | Call Auction | Dealer Market | Dark Pool |
|-------|-------------------|--------------|---------------|-----------|
| **Transparency** | Full order book | Batch reveal | Quote-driven | Pre-trade opaque |
| **Price Discovery** | Incremental | Concentrated | Dealer quotes | Minimal contribution |
| **Information** | Sequential | Aggregated | Spread-encoded | Hidden |
| **Efficiency** | High for liquid | High at open/close | Moderate | Low (piggybacks) |

## 3. Examples + Counterexamples

**Efficient Discovery:**  
Apple earnings released after-hours → futures trade immediately → gap-up at open fully reflects news. Information incorporated within minutes

**Slow Discovery:**  
Small-cap biotech research published in journal → weeks to reflect in price. Low analyst coverage, infrequent trading → slow incorporation

**Failed Discovery:**  
Dark pool executes 1M shares at stale midpoint while public market moved 2% → no price impact, no information revelation

## 4. Layer Breakdown
```
Price Discovery Framework:
├─ Information Flow:
│   ├─ Public Information:
│   │   - Earnings announcements, economic data
│   │   - Instantaneous dissemination (newswires)
│   │   - Efficient markets hypothesis: immediate reflection
│   │   - Empirical: 90% incorporated within 5 minutes
│   ├─ Private Information:
│   │   - Analyst research, insider knowledge
│   │   - Revealed through trading (Kyle model)
│   │   - Gradual incorporation via order flow
│   │   - Detection: PIN (probability of informed trading)
│   └─ Order Flow Information:
│       - Aggregate supply/demand signals
│       - Permanent vs temporary price impact
│       - Market microstructure noise vs fundamental
├─ Hasbrouck Information Share:
│   ├─ Contribution to Price Discovery:
│   │   - IS = variance of efficient price innovation from venue
│   │   - Sum across venues = 100%
│   │   - NYSE vs NASDAQ contribution for interlisted stocks
│   ├─ Methodology:
│   │   - Vector Error Correction Model (VECM)
│   │   - Common factor (efficient price) + transitory noise
│   │   - Cholesky decomposition of innovations
│   ├─ Applications:
│   │   - ETF vs underlying stocks: Who leads?
│   │   - Futures vs spot: Information leadership
│   │   - Lit vs dark venues: Contribution to discovery
│   └─ Findings:
│       - Transparent venues dominate (70-90% IS)
│       - Dark pools: 5-15% contribution
│       - Futures often lead spot (1-5 minutes)
├─ Price Efficiency Measures:
│   ├─ Variance Ratio Test:
│   │   - VR(k) = Var(k-period return) / (k × Var(1-period))
│   │   - Random walk: VR = 1
│   │   - VR < 1: Mean reversion (overreaction)
│   │   - VR > 1: Momentum (slow incorporation)
│   ├─ Autocorrelation:
│   │   - Efficient: returns uncorrelated
│   │   - Negative autocorr: Bid-ask bounce
│   │   - Positive autocorr: Herding, slow discovery
│   ├─ Event Study:
│   │   - CAR (cumulative abnormal returns)
│   │   - Speed of adjustment post-announcement
│   │   - Overreaction vs underreaction
│   └─ Price Impact Decay:
│       - Permanent component: information
│       - Temporary component: liquidity
│       - Ratio indicates discovery quality
├─ Trading Mechanisms:
│   ├─ Continuous Limit Order Book:
│   │   - Quote updates reveal valuation changes
│   │   - Depth adjustments signal confidence
│   │   - Cancellations indicate information events
│   │   - High-frequency price discovery (milliseconds)
│   ├─ Call Auctions:
│   │   - Batch orders at fixed times (open/close)
│   │   - Maximize executable volume at single price
│   │   - Aggregates diverse information
│   │   - Lower volatility vs continuous (empirical)
│   ├─ Dealer Markets:
│   │   - Quotes reflect dealer's information + inventory
│   │   - Spread width encodes uncertainty
│   │   - Slower discovery vs order-driven
│   └─ Dark Pools:
│       - Minimal price discovery (passive pricing)
│       - Piggybacks on lit market midpoint
│       - Reduces information leakage for blocks
├─ Cross-Market Discovery:
│   ├─ Lead-Lag Relationships:
│   │   - Which venue moves first?
│   │   - Granger causality tests
│   │   - Futures → Spot: 1-5 minute lead
│   │   - Large-cap → Small-cap: ~30 minutes
│   ├─ ETF Arbitrage:
│   │   - ETF premium/discount signals basket value
│   │   - Authorized participants arbitrage
│   │   - Information flow: ETF ↔ Constituents
│   ├─ Options Market:
│   │   - Implied volatility leads realized
│   │   - Put-call parity violations signal mispricing
│   │   - Informed traders use options (leverage)
│   └─ International Linkages:
│       - US → Asia spillovers overnight
│       - ADRs vs home market co-movement
│       - FX markets bridge time zones
└─ Impediments to Discovery:
    ├─ Fragmentation: Information split across venues
    ├─ Latency: Stale quotes, slow aggregation
    ├─ Opacity: Dark pools, internalization
    ├─ Manipulation: Spoofing distorts order flow
    └─ Asynchronous Trading: Non-overlapping hours
```

**Interaction:** Information event → informed trading → order flow imbalance → price adjustment → market efficiency

## 5. Mini-Project
Simulate price discovery via order flow:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Price Discovery Simulation
class PriceDiscoveryMarket:
    def __init__(self, fundamental_volatility=0.01, noise_volatility=0.005):
        self.fundamental = 100.0  # True value
        self.price = 100.0  # Market price
        self.fundamental_vol = fundamental_volatility
        self.noise_vol = noise_volatility
        self.history = []
        
    def update_fundamental(self):
        """Fundamental value follows random walk"""
        shock = np.random.normal(0, self.fundamental_vol)
        self.fundamental += shock
        return shock
    
    def receive_informed_order(self, informed_fraction=0.3):
        """Informed traders know fundamental, trade toward it"""
        mispricing = self.fundamental - self.price
        
        # Informed trade size proportional to mispricing
        informed_order = informed_fraction * mispricing
        
        return informed_order
    
    def receive_noise_order(self):
        """Noise traders trade randomly"""
        return np.random.normal(0, self.noise_vol * 100)
    
    def update_price(self, order_flow, lambda_param=0.1):
        """Kyle's lambda: price impact of order flow"""
        price_impact = lambda_param * order_flow
        self.price += price_impact
        
        return price_impact
    
    def record_state(self, t, order_flow, price_impact, fundamental_shock):
        self.history.append({
            'time': t,
            'fundamental': self.fundamental,
            'price': self.price,
            'mispricing': self.fundamental - self.price,
            'order_flow': order_flow,
            'price_impact': price_impact,
            'fundamental_shock': fundamental_shock
        })

# Simulation parameters
n_periods = 1000
informed_fraction = 0.3  # 30% of mispricing traded per period
lambda_param = 0.05  # Price impact coefficient

# Initialize market
market = PriceDiscoveryMarket(fundamental_volatility=0.02, noise_volatility=0.01)

# Simulate
for t in range(n_periods):
    # Fundamental shock
    fundamental_shock = market.update_fundamental()
    
    # Informed and noise trading
    informed_order = market.receive_informed_order(informed_fraction)
    noise_order = market.receive_noise_order()
    total_order_flow = informed_order + noise_order
    
    # Price adjusts
    price_impact = market.update_price(total_order_flow, lambda_param)
    
    # Record
    market.record_state(t, total_order_flow, price_impact, fundamental_shock)

# Extract data
history = market.history
time = [h['time'] for h in history]
fundamental = np.array([h['fundamental'] for h in history])
price = np.array([h['price'] for h in history])
mispricing = np.array([h['mispricing'] for h in history])
order_flow = np.array([h['order_flow'] for h in history])

# Analysis
print("Price Discovery Simulation Results")
print("=" * 70)
print(f"\nMarket Parameters:")
print(f"Informed Fraction: {informed_fraction*100:.0f}% of mispricing")
print(f"Price Impact (λ): {lambda_param}")
print(f"Fundamental Volatility: {market.fundamental_vol*100:.1f}%")
print(f"Noise Volatility: {market.noise_vol*100:.1f}%")

# Price efficiency metrics
print(f"\nPrice Efficiency Metrics:")
print(f"Mean Mispricing: ${np.abs(mispricing).mean():.4f}")
print(f"Std Dev Mispricing: ${mispricing.std():.4f}")
print(f"Max Mispricing: ${np.abs(mispricing).max():.4f}")

# Correlation: order flow vs fundamental changes
fundamental_changes = np.diff(fundamental)
order_flow_truncated = order_flow[1:]
if len(fundamental_changes) > 0:
    corr_flow_fundamental = np.corrcoef(order_flow_truncated, fundamental_changes)[0, 1]
    print(f"Corr(Order Flow, Fundamental Change): {corr_flow_fundamental:.3f}")

# Price vs fundamental correlation
corr_price_fundamental = np.corrcoef(price, fundamental)[0, 1]
print(f"Corr(Price, Fundamental): {corr_price_fundamental:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price vs Fundamental
axes[0, 0].plot(time, fundamental, label='Fundamental Value', linewidth=2, alpha=0.8)
axes[0, 0].plot(time, price, label='Market Price', linewidth=2, alpha=0.8)
axes[0, 0].fill_between(time, fundamental, price, alpha=0.2, color='red', 
                        label='Mispricing')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Value ($)')
axes[0, 0].set_title('Price Discovery: Market Price vs Fundamental Value')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Mispricing over time
axes[0, 1].plot(time, mispricing, linewidth=1, alpha=0.7)
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(time, 0, mispricing, alpha=0.3)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Mispricing ($)')
axes[0, 1].set_title('Price Discovery Errors')
axes[0, 1].grid(alpha=0.3)

# Half-life of mispricing (mean reversion speed)
# Estimate AR(1): mispricing[t] = phi * mispricing[t-1] + error
if len(mispricing) > 1:
    mispricing_lag = mispricing[:-1]
    mispricing_current = mispricing[1:]
    
    if mispricing_lag.std() > 0:
        phi = np.corrcoef(mispricing_lag, mispricing_current)[0, 1]
        if phi > 0 and phi < 1:
            half_life = -np.log(2) / np.log(phi)
            print(f"\nMean Reversion:")
            print(f"AR(1) coefficient: {phi:.3f}")
            print(f"Half-life of mispricing: {half_life:.1f} periods")

# Plot 3: Order flow and price impact
axes[1, 0].scatter(order_flow, np.gradient(price), alpha=0.3, s=10)
axes[1, 0].set_xlabel('Order Flow')
axes[1, 0].set_ylabel('Price Change ($)')
axes[1, 0].set_title('Price Impact of Order Flow')
axes[1, 0].grid(alpha=0.3)

# Linear regression
if order_flow.std() > 0:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        order_flow, np.gradient(price)
    )
    x_fit = np.linspace(order_flow.min(), order_flow.max(), 100)
    y_fit = slope * x_fit + intercept
    axes[1, 0].plot(x_fit, y_fit, 'r-', linewidth=2, 
                   label=f'λ={slope:.4f}, R²={r_value**2:.3f}')
    axes[1, 0].legend()
    
    print(f"\nEstimated Price Impact:")
    print(f"Lambda (λ): {slope:.4f}")
    print(f"R²: {r_value**2:.3f}")
    print(f"True λ: {lambda_param}")

# Plot 4: Information share (variance decomposition)
price_changes = np.diff(price)
fundamental_changes = np.diff(fundamental)

# Variance attribution
var_fundamental = np.var(fundamental_changes)
var_price = np.var(price_changes)
var_mispricing_change = np.var(np.diff(mispricing))

# Information share = var(fundamental changes) / var(price changes)
if var_price > 0:
    information_share = var_fundamental / var_price
    noise_share = var_mispricing_change / var_price
    
    print(f"\nVariance Decomposition:")
    print(f"Var(Fundamental Changes): {var_fundamental:.6f}")
    print(f"Var(Price Changes): {var_price:.6f}")
    print(f"Var(Mispricing Changes): {var_mispricing_change:.6f}")
    print(f"Information Share: {information_share*100:.1f}%")
    print(f"Noise Share: {noise_share*100:.1f}%")
else:
    information_share = 0
    noise_share = 0

categories = ['Information\n(Fundamental)', 'Noise\n(Microstructure)']
shares = [information_share * 100, noise_share * 100]
colors = ['green', 'red']

axes[1, 1].bar(categories, shares, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Contribution to Price Variance (%)')
axes[1, 1].set_title('Information vs Noise in Price Discovery')
axes[1, 1].set_ylim([0, max(105, max(shares) * 1.1)])
axes[1, 1].grid(alpha=0.3, axis='y')

for i, (cat, share) in enumerate(zip(categories, shares)):
    axes[1, 1].text(i, share + 2, f'{share:.1f}%', ha='center', fontsize=10, 
                   fontweight='bold')

plt.tight_layout()
plt.show()

# Efficiency test: Variance Ratio
def variance_ratio_test(returns, k):
    """
    Variance ratio test for random walk hypothesis
    k: holding period
    """
    n = len(returns)
    
    # k-period returns (overlapping)
    k_returns = np.array([np.sum(returns[i:i+k]) for i in range(n-k+1)])
    
    # Variance ratio
    var_k = np.var(k_returns, ddof=1)
    var_1 = np.var(returns, ddof=1)
    
    if var_1 > 0:
        vr = var_k / (k * var_1)
    else:
        vr = np.nan
    
    return vr

price_returns = np.diff(price) / price[:-1]

print(f"\nVariance Ratio Tests (Random Walk = 1.0):")
for k in [2, 5, 10]:
    vr = variance_ratio_test(price_returns, k)
    print(f"VR({k}): {vr:.3f}", end="")
    if not np.isnan(vr):
        if abs(vr - 1.0) < 0.1:
            print(" [Efficient]")
        elif vr < 1.0:
            print(" [Mean Reversion]")
        else:
            print(" [Momentum/Slow Discovery]")
    else:
        print()

# Price discovery speed
# How many periods to incorporate 90% of fundamental shock?
def measure_discovery_speed(mispricing, threshold=0.1):
    """Measure how quickly mispricing decays"""
    speeds = []
    
    # Find large shocks
    for i in range(1, len(mispricing) - 50):
        shock_size = abs(mispricing[i] - mispricing[i-1])
        
        if shock_size > 0.5:  # Significant shock
            # Track decay
            initial_mispricing = abs(mispricing[i])
            
            for j in range(i+1, min(i+50, len(mispricing))):
                current_mispricing = abs(mispricing[j])
                
                if current_mispricing < threshold * initial_mispricing:
                    speeds.append(j - i)
                    break
    
    return speeds

discovery_speeds = measure_discovery_speed(mispricing, threshold=0.1)
if len(discovery_speeds) > 0:
    print(f"\nPrice Discovery Speed:")
    print(f"Mean time to 90% incorporation: {np.mean(discovery_speeds):.1f} periods")
    print(f"Median: {np.median(discovery_speeds):.1f} periods")
    print(f"Number of shocks analyzed: {len(discovery_speeds)}")
```

## 6. Challenge Round
Why do futures often lead spot markets in price discovery despite representing same asset?
- **Lower transaction costs**: Futures margins ~5-10% vs 100% cash, higher leverage attracts informed traders (Kyle model: informed prefer low-cost venues)
- **Centralized liquidity**: Single futures exchange vs fragmented spot (50+ venues) → easier to execute size without information leakage
- **Professional dominance**: Institutional/HFT in futures vs retail in spot → faster information incorporation (sophisticated traders)
- **Microstructure advantages**: Electronic matching, transparent order book, no short-sale restrictions → efficient price discovery
- **Empirical**: ES futures lead SPY ETF by 1-5 minutes; information share 60-70% futures vs 30-40% spot (Hasbrouck 2003)

## 7. Key References
- [Hasbrouck (1995) - One Security, Many Markets](https://www.jstor.org/stable/2329348)
- [Kyle (1985) - Continuous Auctions and Insider Trading](https://www.jstor.org/stable/1913210)
- [Barclay & Warner (1993) - Stealth Trading and Volatility](https://www.sciencedirect.com/science/article/abs/pii/0304405X9390029B)
- [Madhavan (2000) - Market Microstructure: A Survey](https://www.sciencedirect.com/science/article/abs/pii/S1386418100000070)

---
**Status:** Price discovery mechanisms | **Complements:** Information Asymmetry, Kyle Model, Market Efficiency
