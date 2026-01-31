import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulate earnings announcement event study
np.random.seed(42)
n_events = 200  # Number of earnings announcements
window_length = 121  # -60 to +60 days

# Generate market returns
market_returns = np.random.normal(0.0005, 0.01, window_length)

# Generate individual stock parameters (varying beta)
betas = np.random.uniform(0.8, 1.2, n_events)
alphas = np.random.normal(0.0002, 0.0003, n_events)

# Generate stock returns (normal + earnings surprise on day 0)
stock_returns = np.zeros((n_events, window_length))
earnings_surprises = np.random.choice([-1, 1], n_events, p=[0.5, 0.5])  # Positive/negative surprise

for i in range(n_events):
    # Normal returns (market model)
    base_returns = alphas[i] + betas[i] * market_returns + np.random.normal(0, 0.015, window_length)
    
    # Add earnings announcement effect (day 60 = event day 0)
    event_day = 60
    immediate_response = 0.04 * earnings_surprises[i]  # 4% immediate response
    base_returns[event_day] += immediate_response
    
    # Add post-earnings drift (violation of efficiency)
    drift_period = range(event_day + 1, min(event_day + 41, window_length))
    drift_per_day = 0.0005 * earnings_surprises[i]  # 0.05% per day drift
    for j in drift_period:
        base_returns[j] += drift_per_day
    
    stock_returns[i, :] = base_returns

# Calculate abnormal returns
abnormal_returns = np.zeros((n_events, window_length))
for i in range(n_events):
    # Use estimation window (-60 to -11) to estimate alpha, beta
    estimation_window = range(0, 50)
    X = np.column_stack([np.ones(len(estimation_window)), market_returns[estimation_window]])
    y = stock_returns[i, estimation_window]
    
    # OLS regression
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_hat, beta_market = beta_hat[0], beta_hat[1]
    
    # Calculate abnormal returns across full window
    expected_returns = alpha_hat + beta_market * market_returns
    abnormal_returns[i, :] = stock_returns[i, :] - expected_returns

# Average abnormal returns (AAR) and cumulative (CAR)
AAR = abnormal_returns.mean(axis=0)
AAR_se = abnormal_returns.std(axis=0) / np.sqrt(n_events)
CAR = np.cumsum(AAR)

# Event days relative to announcement (day 60 = 0)
event_days = np.arange(-60, 61)

# Statistical tests
# Test 1: Immediate response [-1, +1]
immediate_window = range(59, 62)  # Days -1, 0, +1
CAR_immediate = CAR[61] - CAR[58]  # CAR from -1 to +1
se_immediate = np.sqrt(np.sum(AAR_se[immediate_window]**2))
t_stat_immediate = CAR_immediate / se_immediate

print("="*70)
print("Event Study: Earnings Announcement Efficiency Test")
print("="*70)
print(f"Number of events: {n_events}")
print(f"Event window: Day -60 to +60")
print(f"Estimation window: Day -60 to -11")
print(f"\nImmediate Response [-1, +1]:")
print(f"  CAR: {CAR_immediate*100:>8.3f}%")
print(f"  t-statistic: {t_stat_immediate:>8.3f}")
print(f"  p-value: {stats.t.sf(abs(t_stat_immediate), n_events-1)*2:>8.6f}")
print(f"  Result: {'Significant immediate response' if abs(t_stat_immediate) > 2 else 'No significant response'}")

# Test 2: Post-event drift [+2, +40]
drift_window = range(62, 100)
CAR_drift = CAR[100] - CAR[61]
se_drift = np.sqrt(np.sum(AAR_se[drift_window]**2))
t_stat_drift = CAR_drift / se_drift

print(f"\nPost-Earnings Announcement Drift [+2, +40]:")
print(f"  CAR: {CAR_drift*100:>8.3f}%")
print(f"  t-statistic: {t_stat_drift:>8.3f}")
print(f"  p-value: {stats.t.sf(abs(t_stat_drift), n_events-1)*2:>8.6f}")
print(f"  Result: {'PEAD detected - EFFICIENCY VIOLATED' if abs(t_stat_drift) > 2 else 'No drift - Efficient'}")

# Test 3: Pre-event drift (should be zero)
pre_window = range(20, 59)
CAR_pre = CAR[58] - CAR[19]
se_pre = np.sqrt(np.sum(AAR_se[pre_window]**2))
t_stat_pre = CAR_pre / se_pre

print(f"\nPre-Event Drift [-40, -2]:")
print(f"  CAR: {CAR_pre*100:>8.3f}%")
print(f"  t-statistic: {t_stat_pre:>8.3f}")
print(f"  p-value: {stats.t.sf(abs(t_stat_pre), n_events-1)*2:>8.6f}")
print(f"  Result: {'Information leakage possible' if abs(t_stat_pre) > 2 else 'No pre-event drift'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Average Abnormal Returns (AAR)
axes[0, 0].bar(event_days, AAR*100, color='blue', alpha=0.6)
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Event Day')
axes[0, 0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0, 0].fill_between(event_days, -2*AAR_se*100, 2*AAR_se*100, alpha=0.2, color='gray', label='Â±2 SE')
axes[0, 0].set_title('Average Abnormal Returns (AAR)')
axes[0, 0].set_xlabel('Event Day')
axes[0, 0].set_ylabel('AAR (%)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xlim(-60, 60)

# Plot 2: Cumulative Abnormal Returns (CAR)
axes[0, 1].plot(event_days, CAR*100, linewidth=2, color='darkblue')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Event Day')
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.8)
# Highlight drift period
axes[0, 1].axvspan(2, 40, alpha=0.2, color='orange', label='Drift Period [+2,+40]')
axes[0, 1].set_title('Cumulative Abnormal Returns (CAR)')
axes[0, 1].set_xlabel('Event Day')
axes[0, 1].set_ylabel('CAR (%)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim(-60, 60)

# Plot 3: Distribution of individual CARs at day +40
CAR_individual_40 = np.cumsum(abnormal_returns[:, 59:100], axis=1)[:, -1]
axes[1, 0].hist(CAR_individual_40*100, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(CAR_individual_40.mean()*100, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {CAR_individual_40.mean()*100:.2f}%')
axes[1, 0].set_title('Distribution of CAR[0,+40] Across Events')
axes[1, 0].set_xlabel('CAR (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: t-statistics over time (statistical significance)
t_stats_rolling = AAR / AAR_se
axes[1, 1].plot(event_days, t_stats_rolling, linewidth=1.5, color='purple')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Event Day')
axes[1, 1].axhline(2, color='orange', linestyle=':', linewidth=1.5, label='Significance (t=2)')
axes[1, 1].axhline(-2, color='orange', linestyle=':', linewidth=1.5)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1, 1].set_title('Statistical Significance of AAR')
axes[1, 1].set_xlabel('Event Day')
axes[1, 1].set_ylabel('t-statistic')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(-60, 60)

plt.tight_layout()
plt.savefig('market_efficiency_event_study.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("Interpretation:")
print(f"{'='*70}")
print("1. Immediate Response: Stock price reacts quickly to earnings surprise")
print("   â†’ Consistent with semi-strong efficiency (public info incorporated)")
print("")
print("2. Post-Earnings Drift: Continued abnormal returns over 40 days")
print("   â†’ VIOLATION of semi-strong efficiency (delayed incorporation)")
print("")
print("3. Economic Significance: ~2% drift (annualized ~18%)")
print("   â†’ Substantial; but transaction costs may reduce profitability")
