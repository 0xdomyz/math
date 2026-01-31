"""
Extracted from: return_to_player.md
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

np.random.seed(42)

print("="*70)
print("RETURN TO PLAYER: TRACKING ACTUAL VS THEORETICAL")
print("="*70)

# ============================================================================
# 1. RTP BY GAME (THEORETICAL)
# ============================================================================

print("\n" + "="*70)
print("1. THEORETICAL RTP BY GAME")
print("="*70)

games_rtp_theoretical = {
    'Blackjack (basic strategy)': 0.995,
    'Roulette (European)': 0.973,
    'Craps (Pass line)': 0.9859,
    'Baccarat (Banker)': 0.9894,
    'Slots (typical)': 0.95,
    'Video Poker (good table)': 0.99,
    'Keno': 0.70,
}

df_games = pd.DataFrame({
    'Game': list(games_rtp_theoretical.keys()),
    'Theoretical RTP': list(games_rtp_theoretical.values()),
})

df_games['Theoretical HE %'] = (1 - df_games['Theoretical RTP']) * 100

print("\nTheoretical RTP and House Edge:")
print(df_games[['Game', 'Theoretical RTP', 'Theoretical HE %']].to_string(index=False))

# ============================================================================
# 2. SIMULATE OBSERVED RTP (SHORT-TERM)
# ============================================================================

print("\n" + "="*70)
print("2. OBSERVED RTP: SIMULATING GAMBLING SESSIONS")
print("="*70)

def simulate_game_session(rtp, n_bets=100, bet_amount=1.0):
    """
    Simulate a gambling session.
    Assume each bet returns either ~0 or ~1 based on RTP.
    """
    house_edge = 1 - rtp
    expected_payout = rtp * bet_amount
    
    # Simplified: each bet returns exactly payout (no variance)
    # For realism, add variance
    outcomes = []
    
    for _ in range(n_bets):
        # Outcome: either win (return bet_amount) or lose (return 0)
        # Weighted by payout probability
        if np.random.random() < rtp:
            outcome = bet_amount  # Win (get bet back)
        else:
            outcome = 0  # Lose (get nothing)
        outcomes.append(outcome)
    
    total_wagered = n_bets * bet_amount
    total_returned = np.sum(outcomes)
    observed_rtp = total_returned / total_wagered
    
    return observed_rtp, total_returned, total_wagered

# Simulate multiple sessions for blackjack
print("\nBlackjack (99.5% theoretical RTP): Multiple session simulations")
print(f"{'Session':>10} {'Bets':>10} {'Observed RTP':>15} {'Deviation':>15}")
print("-" * 50)

theoretical_rtp_bj = 0.995
for session_bets in [100, 500, 1000, 5000]:
    obs_rtps = []
    for _ in range(10):
        obs_rtp, _, _ = simulate_game_session(theoretical_rtp_bj, session_bets)
        obs_rtps.append(obs_rtp)
    
    mean_obs = np.mean(obs_rtps)
    deviation = (mean_obs - theoretical_rtp_bj) * 100
    print(f"{'Avg':>10} {session_bets:>10} {mean_obs:>15.4f} {deviation:>14.2f}%")

# ============================================================================
# 3. CONVERGENCE: HOW MANY BETS NEEDED FOR ACCURATE RTP?
# ============================================================================

print("\n" + "="*70)
print("3. CONVERGENCE: SAMPLE SIZE FOR ACCURATE RTP")
print("="*70)

def estimate_convergence(theoretical_rtp, confidence=0.95):
    """
    Estimate bets needed for accurate RTP estimation.
    Using normal approximation: SE = √(p(1-p)/n)
    For 95% CI: n ≈ 1.96² × p(1-p) / ε²
    where ε is desired error (e.g., 0.01 = ±1%)
    """
    p = theoretical_rtp
    z = stats.norm.ppf((1 + confidence) / 2)
    
    errors = [0.01, 0.02, 0.05, 0.10]  # ±1%, ±2%, ±5%, ±10% error
    results = {}
    
    for error in errors:
        n = (z**2 * p * (1-p)) / (error**2)
        results[error] = int(np.ceil(n))
    
    return results

print("\nSample size needed for accurate RTP estimation (95% confidence):")
print(f"Game: Roulette (97.3% theoretical RTP)")
print(f"{'Target Accuracy':>20} {'Bets Required':>20} {'Approx Hours':>20}")
print("-" * 60)

convergence = estimate_convergence(0.973)
hands_per_hour = 30  # Roulette spins/hour

for error, n_bets in convergence.items():
    error_pct = error * 100
    hours = n_bets / hands_per_hour
    print(f"±{error_pct:>18.1f}% {n_bets:>20,} {hours:>19.1f} hours")

# ============================================================================
# 4. STATISTICAL TEST: OBSERVED VS THEORETICAL
# ============================================================================

print("\n" + "="*70)
print("4. HYPOTHESIS TEST: IS MACHINE FAIR?")
print("="*70)

def statistical_test_rtp(observed_rtp, theoretical_rtp, n_bets):
    """
    Chi-square test to see if observed RTP significantly different from theoretical.
    H0: Observed RTP = Theoretical RTP (machine fair)
    H1: Observed RTP ≠ Theoretical RTP (machine biased)
    """
    # For binomial: expected wins/losses based on theoretical
    expected_wins = n_bets * theoretical_rtp
    expected_losses = n_bets * (1 - theoretical_rtp)
    
    observed_wins = n_bets * observed_rtp
    observed_losses = n_bets * (1 - observed_rtp)
    
    # Chi-square statistic
    chi2 = ((observed_wins - expected_wins)**2 / expected_wins +
           (observed_losses - expected_losses)**2 / expected_losses)
    
    # P-value (df=1 for binomial)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value

print("\nTesting if observed RTP differs significantly from theoretical:")
print(f"Machine: Slots (95% theoretical RTP)")
print(f"Test significance: α = 0.05")
print()

theoretical = 0.95
n_bets_test = 1000

# Simulate observed RTPs (some fair, some rigged)
scenarios = [
    ('Fair machine', 0.95),
    ('Slightly loose (96%)', 0.96),
    ('Rigged tight (93%)', 0.93),
]

print(f"{'Scenario':<25} {'Observed RTP':>15} {'Chi-square':>15} {'p-value':>15} {'Fair?':>10}")
print("=" * 80)

for scenario_name, observed_rtp in scenarios:
    chi2, p_value = statistical_test_rtp(observed_rtp, theoretical, n_bets_test)
    fair = 'Yes' if p_value > 0.05 else 'No'
    print(f"{scenario_name:<25} {observed_rtp:>15.4f} {chi2:>15.3f} {p_value:>15.4f} {fair:>10}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Theoretical RTP by game
ax1 = axes[0, 0]
colors_rtp = ['green' if rtp > 0.97 else 'yellow' if rtp > 0.93 else 'red' 
             for rtp in df_games['Theoretical RTP']]
ax1.barh(df_games['Game'], df_games['Theoretical RTP'], color=colors_rtp, alpha=0.7, edgecolor='black')
ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Theoretical RTP')
ax1.set_title('Theoretical RTP by Game')
ax1.set_xlim(0.65, 1.01)
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Convergence (blackjack)
ax2 = axes[0, 1]
n_range = np.logspace(1, 5, 50)  # 10 to 100,000 bets
se_range = []

for n in n_range:
    p = 0.995  # Blackjack RTP
    se = np.sqrt(p * (1-p) / n)
    se_range.append(se)

ax2.loglog(n_range, np.array(se_range)*100, linewidth=2, label='95% CI half-width')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='±1% error')
ax2.set_xlabel('Number of Bets')
ax2.set_ylabel('Error Range (%)')
ax2.set_title('Convergence: How Many Bets for Accurate RTP?')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Distribution of observed RTP (10,000 simulations)
ax3 = axes[1, 0]
observed_rtps_sim = []
for _ in range(10000):
    obs_rtp, _, _ = simulate_game_session(0.95, n_bets=100)
    observed_rtps_sim.append(obs_rtp)

ax3.hist(observed_rtps_sim, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax3.axvline(x=0.95, color='red', linestyle='--', linewidth=2, label='Theoretical (95%)')
ax3.axvline(x=np.mean(observed_rtps_sim), color='green', linestyle='--', linewidth=2, label='Mean observed')
ax3.set_xlabel('Observed RTP')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Observed RTP (100 bets, 10k simulations)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Sample size vs detection power
ax4 = axes[1, 1]
true_rtps = np.array([0.93, 0.94, 0.95, 0.96, 0.97])
theoretical_rtp = 0.95
n_samples = 100

detection_power = []
for true_rtp in true_rtps:
    # Power: probability of detecting difference if it exists
    # Simpler: just show effect size grows with deviation
    effect_size = (true_rtp - theoretical_rtp) * 100
    detection_power.append(abs(effect_size))

colors_detect = ['red' if rtp < 0.95 else 'green' for rtp in true_rtps]
ax4.bar(true_rtps * 100, detection_power, color=colors_detect, alpha=0.7, edgecolor='black', width=0.5)
ax4.axvline(x=95, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Theoretical (95%)')
ax4.set_xlabel('True RTP (%)')
ax4.set_ylabel('Effect Size (%)')
ax4.set_title('Detection: Bias from Theoretical RTP')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('return_to_player_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: return_to_player_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ RTP = inverse of house edge; determines long-term profitability")
print("✓ Theoretical RTP accurate only over many bets (1000+)")
print("✓ Short-term: Variance allows deviation from RTP")
print("✓ Game selection: Highest RTP = best player value")
