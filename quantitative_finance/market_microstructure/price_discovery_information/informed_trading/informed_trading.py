import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

np.random.seed(42)

# PIN Model Simulation and Estimation
def simulate_pin_model(n_days=250, alpha=0.3, mu=300, epsilon_buy=200, epsilon_sell=200):
    """
    Simulate Easley-O'Hara PIN model
    
    Parameters:
    - n_days: number of trading days
    - alpha: probability of information event
    - mu: arrival rate of informed traders (per day)
    - epsilon_buy, epsilon_sell: arrival rates of uninformed buyers/sellers
    """
    results = []
    
    for day in range(n_days):
        # Is there an information event?
        info_event = np.random.random() < alpha
        
        if info_event:
            # Good or bad news (50/50)
            good_news = np.random.random() < 0.5
        else:
            good_news = None
        
        # Uninformed trading
        n_uninformed_buy = np.random.poisson(epsilon_buy)
        n_uninformed_sell = np.random.poisson(epsilon_sell)
        
        # Informed trading (only if info event)
        if info_event:
            n_informed = np.random.poisson(mu)
            
            if good_news:
                n_buy = n_uninformed_buy + n_informed
                n_sell = n_uninformed_sell
            else:
                n_buy = n_uninformed_buy
                n_sell = n_uninformed_sell + n_informed
        else:
            n_buy = n_uninformed_buy
            n_sell = n_uninformed_sell
        
        results.append({
            'day': day,
            'info_event': info_event,
            'good_news': good_news,
            'n_buy': n_buy,
            'n_sell': n_sell,
            'imbalance': n_buy - n_sell
        })
    
    return results

def pin_likelihood(params, buy_counts, sell_counts):
    """
    Log-likelihood function for PIN model
    
    params: [alpha, mu, epsilon_buy, epsilon_sell]
    """
    alpha, mu, eps_b, eps_s = params
    
    # Constrain parameters
    alpha = max(0.001, min(0.999, alpha))
    mu = max(0.1, mu)
    eps_b = max(0.1, eps_b)
    eps_s = max(0.1, eps_s)
    
    n_days = len(buy_counts)
    log_lik = 0
    
    for i in range(n_days):
        B = buy_counts[i]
        S = sell_counts[i]
        
        # Probability of observing (B, S) on this day
        # Case 1: No info event (prob = 1-alpha)
        prob_no_info = (1 - alpha) * \
                      stats.poisson.pmf(B, eps_b) * \
                      stats.poisson.pmf(S, eps_s)
        
        # Case 2: Good news (prob = alpha/2)
        prob_good = (alpha / 2) * \
                   stats.poisson.pmf(B, eps_b + mu) * \
                   stats.poisson.pmf(S, eps_s)
        
        # Case 3: Bad news (prob = alpha/2)
        prob_bad = (alpha / 2) * \
                  stats.poisson.pmf(B, eps_b) * \
                  stats.poisson.pmf(S, eps_s + mu)
        
        # Total probability
        prob_total = prob_no_info + prob_good + prob_bad
        
        if prob_total > 0:
            log_lik += np.log(prob_total)
        else:
            log_lik += -1e10  # Penalize invalid parameters
    
    return -log_lik  # Negative for minimization

# Simulation
print("Informed Trading and PIN Estimation")
print("=" * 70)

# True parameters
alpha_true = 0.30  # 30% chance of info event per day
mu_true = 300  # Informed trades per day
eps_b_true = 200  # Uninformed buys
eps_s_true = 200  # Uninformed sells

PIN_true = alpha_true * mu_true / (alpha_true * mu_true + eps_b_true + eps_s_true)

print(f"\nTrue Parameters:")
print(f"α (info event probability): {alpha_true*100:.0f}%")
print(f"μ (informed rate): {mu_true}")
print(f"ε_buy (uninformed buy rate): {eps_b_true}")
print(f"ε_sell (uninformed sell rate): {eps_s_true}")
print(f"True PIN: {PIN_true:.3f}")

# Simulate data
np.random.seed(42)
n_days = 250
data = simulate_pin_model(n_days, alpha_true, mu_true, eps_b_true, eps_s_true)

buy_counts = np.array([d['n_buy'] for d in data])
sell_counts = np.array([d['n_sell'] for d in data])
imbalances = np.array([d['imbalance'] for d in data])

print(f"\nSimulated Data ({n_days} days):")
print(f"Mean Buy Trades per Day: {buy_counts.mean():.1f}")
print(f"Mean Sell Trades per Day: {sell_counts.mean():.1f}")
print(f"Mean Imbalance: {imbalances.mean():.1f}")
print(f"Std Dev Imbalance: {imbalances.std():.1f}")

# Days with info events
info_days = [d for d in data if d['info_event']]
print(f"Actual Info Event Days: {len(info_days)} ({len(info_days)/n_days*100:.1f}%)")

# Estimate PIN
print(f"\nEstimating PIN using Maximum Likelihood...")

# Initial guess
x0 = [0.25, 250, 180, 180]

# Bounds
bounds = [(0.001, 0.999), (1, 1000), (1, 1000), (1, 1000)]

# Optimize
result = minimize(pin_likelihood, x0, args=(buy_counts, sell_counts),
                 method='L-BFGS-B', bounds=bounds)

alpha_est, mu_est, eps_b_est, eps_s_est = result.x
PIN_est = alpha_est * mu_est / (alpha_est * mu_est + eps_b_est + eps_s_est)

print(f"\nEstimated Parameters:")
print(f"α (info event probability): {alpha_est*100:.1f}% (true: {alpha_true*100:.0f}%)")
print(f"μ (informed rate): {mu_est:.1f} (true: {mu_true})")
print(f"ε_buy (uninformed buy rate): {eps_b_est:.1f} (true: {eps_b_true})")
print(f"ε_sell (uninformed sell rate): {eps_s_est:.1f} (true: {eps_s_true})")
print(f"Estimated PIN: {PIN_est:.3f} (true: {PIN_true:.3f})")
print(f"Estimation Error: {abs(PIN_est - PIN_true):.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Daily buy vs sell
axes[0, 0].scatter(buy_counts, sell_counts, alpha=0.5, s=30)

# Mark info event days
info_indices = [i for i, d in enumerate(data) if d['info_event']]
axes[0, 0].scatter(buy_counts[info_indices], sell_counts[info_indices],
                  c='red', s=50, alpha=0.7, label='Info Event Days')

axes[0, 0].plot([0, max(buy_counts.max(), sell_counts.max())],
               [0, max(buy_counts.max(), sell_counts.max())],
               'k--', linewidth=1, alpha=0.5, label='Buy = Sell')
axes[0, 0].set_xlabel('Buy Trades')
axes[0, 0].set_ylabel('Sell Trades')
axes[0, 0].set_title('Daily Buy vs Sell (Red = Info Event)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Order imbalance time series
days = range(n_days)
axes[0, 1].bar(days, imbalances, alpha=0.6, edgecolor='black', linewidth=0.5)

# Highlight info days
for idx in info_indices:
    axes[0, 1].axvline(idx, color='red', alpha=0.3, linewidth=0.5)

axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Order Imbalance (Buy - Sell)')
axes[0, 1].set_title('Daily Order Flow Imbalance (Red lines = Info Events)')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Distribution of buy/sell counts
axes[1, 0].hist(buy_counts, bins=30, alpha=0.6, label='Buy Trades', color='green',
               edgecolor='black')
axes[1, 0].hist(sell_counts, bins=30, alpha=0.6, label='Sell Trades', color='red',
               edgecolor='black')
axes[1, 0].axvline(buy_counts.mean(), color='green', linestyle='--', linewidth=2,
                  label=f'Mean Buy: {buy_counts.mean():.0f}')
axes[1, 0].axvline(sell_counts.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean Sell: {sell_counts.mean():.0f}')
axes[1, 0].set_xlabel('Number of Trades')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Daily Trade Counts')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 4: PIN comparison
categories = ['True PIN', 'Estimated PIN']
pin_values = [PIN_true, PIN_est]
colors = ['blue', 'orange']

bars = axes[1, 1].bar(categories, pin_values, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('PIN Value')
axes[1, 1].set_title('Probability of Informed Trading (PIN)')
axes[1, 1].set_ylim([0, max(pin_values) * 1.2])
axes[1, 1].grid(alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, pin_values)):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                   f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Statistical tests
print(f"\nStatistical Analysis:")

# Test: Do info days have higher imbalance?
info_imbalances = [d['imbalance'] for d in data if d['info_event']]
no_info_imbalances = [d['imbalance'] for d in data if not d['info_event']]

t_stat, p_value = stats.ttest_ind(np.abs(info_imbalances), np.abs(no_info_imbalances))
print(f"Info Days Abs Imbalance: {np.mean(np.abs(info_imbalances)):.1f}")
print(f"No Info Days Abs Imbalance: {np.mean(np.abs(no_info_imbalances)):.1f}")
print(f"t-test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Info days have significantly higher imbalance")

# Correlation: Imbalance autocorrelation
autocorr = np.corrcoef(imbalances[:-1], imbalances[1:])[0, 1]
print(f"\nImbalance Autocorrelation (lag 1): {autocorr:.3f}")
print(f"Interpretation: {'Persistent' if autocorr > 0.1 else 'Random walk'}")

# Distribution tests
print(f"\nDistribution Normality Tests:")
_, p_buy = stats.shapiro(buy_counts)
_, p_sell = stats.shapiro(sell_counts)
print(f"Buy Counts Normal? p={p_buy:.4f} ({'Yes' if p_buy > 0.05 else 'No, Poisson expected'})")
print(f"Sell Counts Normal? p={p_sell:.4f} ({'Yes' if p_sell > 0.05 else 'No, Poisson expected'})")
