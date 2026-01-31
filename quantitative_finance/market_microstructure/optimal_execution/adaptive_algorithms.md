# Adaptive Execution Algorithms: Machine Learning for Optimal Trading

## 1. Concept Skeleton
**Definition:** Dynamic execution strategies that adjust parameters in real-time using market state observations and predictive models  
**Purpose:** Outperform static schedules by exploiting short-term predictability; minimize costs via learned market patterns  
**Prerequisites:** Reinforcement learning, time series prediction, online optimization, market microstructure

## 2. Comparative Framing
| Strategy | Adaptive/ML-Based | POV | TWAP | Almgren-Chriss |
|----------|------------------|-----|------|----------------|
| **Flexibility** | Real-time learning | Volume-responsive | Fixed schedule | Pre-determined |
| **Information Use** | Order book + history | Volume only | Time only | Risk aversion |
| **Complexity** | Very high (ML models) | Medium | Low | High (optimization) |
| **Training Needed** | Extensive (historical) | None | None | Calibration |
| **Out-of-Sample** | Can adapt to regime shifts | Reactive only | Static | Assumes stationarity |

## 3. Examples + Counterexamples

**Simple Example:**  
RL agent learns: "When spread widens + volume drops → pause execution" → Saves 3 bps by avoiding illiquidity

**Failure Case:**  
Model trained on normal volatility → Flash crash arrives → Agent freezes (out-of-distribution) → Fails deadline

**Edge Case:**  
Agent discovers exploiting venue routing (send IOC to exchange A) → Adversarial counter-strategy emerges → Arms race

## 4. Layer Breakdown
```
Adaptive Execution Framework:
├─ State Representation (Market Features):
│   ├─ Order Book: Bid-ask spread, depth at multiple levels
│   ├─ Volume: Recent volume, volume imbalance (buy vs sell)
│   ├─ Price Dynamics: Volatility, momentum, microstructure noise
│   ├─ Execution State: Remaining quantity, time left, current slippage
│   └─ External: VIX, sector momentum, news sentiment
├─ Model Architecture:
│   ├─ Reinforcement Learning:
│   │   ├─ State: s_t = (order_book, volume, time, remaining_qty)
│   │   ├─ Action: a_t = (slice_size, order_type, venue)
│   │   ├─ Reward: r_t = -cost_t (minimize execution cost)
│   │   └─ Policy: π(a|s) learned via Q-learning/PPO/A3C
│   ├─ Supervised Learning:
│   │   ├─ Predict: short-term price movement, volume, spread
│   │   ├─ Features: Limit order book shape, trade flow
│   │   └─ Model: LSTM, Transformer, Gradient Boosting
│   └─ Online Optimization:
│       ├─ Update cost model: f(slice_size, market_state)
│       ├─ Resolve optimization: min_{schedule} E[cost | current_state]
│       └─ Recalibrate: Every N trades or regime shift
├─ Action Space:
│   ├─ Timing: Execute now vs wait
│   ├─ Aggressiveness: Market order vs limit (various levels)
│   ├─ Size: Slice quantity (continuous or discrete bins)
│   └─ Venue: Dark pool, lit exchange, smart router
├─ Training Process:
│   ├─ Historical Simulation: Replay order book data
│   ├─ Reward Shaping: Balance cost vs completion risk
│   ├─ Exploration: ε-greedy, entropy bonus for diverse actions
│   └─ Validation: Out-of-sample testing, regime robustness
└─ Risk Management:
    ├─ Action Constraints: Max slice size, min spread
    ├─ Completion Guarantee: Accelerate if deadline approaches
    ├─ Model Confidence: Revert to TWAP if uncertainty high
    └─ Circuit Breakers: Halt if market conditions anomalous
```

**Interaction:** Observe state → Model predicts optimal action → Execute → Observe reward → Update model → Repeat

## 5. Mini-Project
Implement Q-learning agent for adaptive execution with market simulation:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

np.random.seed(42)
random.seed(42)

# ============================================================================
# MARKET ENVIRONMENT SIMULATOR
# ============================================================================

class MarketEnvironment:
    """Simulated market with realistic dynamics for execution."""
    
    def __init__(self, initial_price=100.0, volatility=0.002, n_steps=50):
        self.initial_price = initial_price
        self.volatility = volatility
        self.n_steps = n_steps
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.price = self.initial_price
        
        # Generate price path
        returns = np.random.normal(0, self.volatility, self.n_steps)
        self.prices = self.initial_price * np.exp(np.cumsum(returns))
        
        # Generate spread dynamics (mean-reverting)
        spread_shocks = np.random.normal(0, 0.0002, self.n_steps)
        self.spreads = 0.001 + np.cumsum(spread_shocks)
        self.spreads = np.abs(self.spreads)  # Keep positive
        
        # Generate volume profile (U-shaped intraday)
        time_factor = np.linspace(0, 2*np.pi, self.n_steps)
        self.volumes = 1000 * (1 + 0.5*np.cos(time_factor - np.pi))
        self.volumes += np.random.normal(0, 200, self.n_steps)
        self.volumes = np.maximum(self.volumes, 100)
        
        return self.get_state()
    
    def get_state(self):
        """Return current market state features."""
        if self.current_step >= self.n_steps:
            return None
        
        # State: [spread, volume, price_momentum, time_remaining]
        spread = self.spreads[self.current_step]
        volume = self.volumes[self.current_step]
        
        # Recent price momentum
        lookback = min(5, self.current_step)
        if lookback > 0:
            momentum = (self.prices[self.current_step] - self.prices[self.current_step-lookback]) / self.prices[self.current_step-lookback]
        else:
            momentum = 0
        
        time_remaining = (self.n_steps - self.current_step) / self.n_steps
        
        return np.array([spread, volume/1000, momentum*100, time_remaining])
    
    def step(self, slice_size):
        """
        Execute trade and return cost.
        slice_size: Number of shares to trade (normalized 0-1)
        """
        if self.current_step >= self.n_steps:
            return None, 0, True
        
        price = self.prices[self.current_step]
        spread = self.spreads[self.current_step]
        volume = self.volumes[self.current_step]
        
        # Market impact: proportional to (slice_size / volume)
        market_impact = 0.01 * (slice_size / (volume/1000))
        
        # Total execution cost: spread + market impact
        execution_cost = price * (spread/2 + market_impact)
        
        self.current_step += 1
        next_state = self.get_state()
        done = (self.current_step >= self.n_steps)
        
        return next_state, execution_cost, done

# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """Q-learning agent for adaptive execution."""
    
    def __init__(self, state_dim=4, n_actions=5, learning_rate=0.01, 
                 discount=0.95, epsilon=0.2):
        self.state_dim = state_dim
        self.n_actions = n_actions  # Discrete action space (slice sizes)
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        
        # Q-table approximation (discretize states)
        self.q_table = {}
    
    def discretize_state(self, state):
        """Convert continuous state to discrete for Q-table."""
        # Discretize each dimension
        spread_bin = int(np.clip(state[0] * 1000, 0, 10))  # spread [0, 0.01]
        volume_bin = int(np.clip(state[1], 0, 5))  # volume [0, 5000]
        momentum_bin = int(np.clip((state[2] + 5) * 2, 0, 20))  # momentum [-5%, 5%]
        time_bin = int(state[3] * 10)  # time [0, 1]
        
        return (spread_bin, volume_bin, momentum_bin, time_bin)
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key][action]
    
    def choose_action(self, state, remaining_qty, explore=True):
        """
        Epsilon-greedy action selection.
        Actions: Different slice sizes from 0% to 100% of remaining.
        """
        if explore and random.random() < self.epsilon:
            # Exploration: random action
            action = random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: best action
            state_key = self.discretize_state(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)
            action = np.argmax(self.q_table[state_key])
        
        # Map action to slice size (0%, 25%, 50%, 75%, 100% of remaining)
        slice_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        slice_size = slice_fractions[action] * remaining_qty
        
        return action, slice_size
    
    def update(self, state, action, reward, next_state):
        """Q-learning update rule."""
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        
        # Current Q-value
        q_current = self.q_table[state_key][action]
        
        # Max Q-value for next state
        if next_state is not None:
            next_state_key = self.discretize_state(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.n_actions)
            q_next_max = np.max(self.q_table[next_state_key])
        else:
            q_next_max = 0
        
        # Q-learning update
        q_target = reward + self.gamma * q_next_max
        self.q_table[state_key][action] += self.lr * (q_target - q_current)

# ============================================================================
# TRAINING
# ============================================================================

print("="*70)
print("ADAPTIVE EXECUTION: Q-LEARNING AGENT TRAINING")
print("="*70)

env = MarketEnvironment(n_steps=50)
agent = QLearningAgent(state_dim=4, n_actions=5, learning_rate=0.1, epsilon=0.3)

total_quantity = 10000  # shares to execute
n_episodes = 500

episode_costs = []
episode_completions = []

for episode in range(n_episodes):
    state = env.reset()
    remaining_qty = total_quantity
    total_cost = 0
    
    while remaining_qty > 0:
        # Choose action
        action, slice_size = agent.choose_action(state, remaining_qty, explore=True)
        
        # Execute in environment
        next_state, cost, done = env.step(slice_size)
        
        # Negative reward (minimize cost)
        reward = -cost
        
        # Penalize if failed to complete
        if done and remaining_qty > 0:
            reward -= 100  # Large penalty
        
        # Update agent
        agent.update(state, action, reward, next_state)
        
        # Update state
        state = next_state
        remaining_qty -= slice_size
        total_cost += cost
        
        if done:
            break
    
    episode_costs.append(total_cost)
    completion_rate = max(0, (total_quantity - remaining_qty) / total_quantity)
    episode_completions.append(completion_rate)
    
    if (episode + 1) % 100 == 0:
        avg_cost = np.mean(episode_costs[-100:])
        avg_completion = np.mean(episode_completions[-100:])
        print(f"Episode {episode+1}: Avg Cost = {avg_cost:.2f}, "
              f"Completion = {avg_completion*100:.1f}%")

print("\n✓ Training complete")

# ============================================================================
# TESTING: RL vs BASELINES
# ============================================================================

print("\n" + "="*70)
print("PERFORMANCE COMPARISON: RL vs BASELINES")
print("="*70)

n_test_episodes = 50
rl_costs = []
twap_costs = []
pov_costs = []

for episode in range(n_test_episodes):
    env_test = MarketEnvironment(n_steps=50)
    
    # RL Agent (no exploration)
    state = env_test.reset()
    remaining_qty = total_quantity
    rl_cost = 0
    
    while remaining_qty > 0:
        action, slice_size = agent.choose_action(state, remaining_qty, explore=False)
        next_state, cost, done = env_test.step(slice_size)
        rl_cost += cost
        remaining_qty -= slice_size
        state = next_state
        if done:
            break
    
    rl_costs.append(rl_cost)
    
    # TWAP Baseline
    env_test.reset()
    twap_slice = total_quantity / 50
    twap_cost = 0
    for _ in range(50):
        _, cost, _ = env_test.step(twap_slice)
        twap_cost += cost
    twap_costs.append(twap_cost)
    
    # POV 10% Baseline
    env_test.reset()
    pov_remaining = total_quantity
    pov_cost = 0
    for step in range(50):
        volume = env_test.volumes[step]
        pov_slice = min(0.1 * volume, pov_remaining)
        _, cost, _ = env_test.step(pov_slice)
        pov_cost += cost
        pov_remaining -= pov_slice
        if pov_remaining <= 0:
            break
    pov_costs.append(pov_cost)

rl_avg = np.mean(rl_costs)
twap_avg = np.mean(twap_costs)
pov_avg = np.mean(pov_costs)

print(f"\nRL Agent:       Avg Cost = ${rl_avg:.2f} ± ${np.std(rl_costs):.2f}")
print(f"TWAP Baseline:  Avg Cost = ${twap_avg:.2f} ± ${np.std(twap_costs):.2f}")
print(f"POV 10%:        Avg Cost = ${pov_avg:.2f} ± ${np.std(pov_costs):.2f}")

improvement_vs_twap = (twap_avg - rl_avg) / twap_avg * 100
improvement_vs_pov = (pov_avg - rl_avg) / pov_avg * 100

print(f"\nRL Improvement vs TWAP: {improvement_vs_twap:+.1f}%")
print(f"RL Improvement vs POV:  {improvement_vs_pov:+.1f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Learning curve (cost over episodes)
ax1 = axes[0, 0]
window = 20
smoothed_costs = np.convolve(episode_costs, np.ones(window)/window, mode='valid')
ax1.plot(smoothed_costs, linewidth=2, color='blue')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Execution Cost ($)')
ax1.set_title('Q-Learning: Cost Reduction Over Training')
ax1.grid(True, alpha=0.3)

# Plot 2: Completion rate over training
ax2 = axes[0, 1]
smoothed_completion = np.convolve(episode_completions, np.ones(window)/window, mode='valid')
ax2.plot(smoothed_completion * 100, linewidth=2, color='green')
ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Target: 100%')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Completion Rate (%)')
ax2.set_title('Q-Learning: Completion Rate Improvement')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Strategy comparison (box plot)
ax3 = axes[0, 2]
data_to_plot = [rl_costs, twap_costs, pov_costs]
bp = ax3.boxplot(data_to_plot, labels=['RL Agent', 'TWAP', 'POV 10%'],
                patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_ylabel('Total Execution Cost ($)')
ax3.set_title('Cost Distribution: RL vs Baselines')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: RL agent behavior (sample execution)
ax4 = axes[1, 0]
env_demo = MarketEnvironment(n_steps=50)
state = env_demo.reset()
remaining = total_quantity
execution_times = []
slice_sizes = []

while remaining > 0 and env_demo.current_step < env_demo.n_steps:
    action, slice_size = agent.choose_action(state, remaining, explore=False)
    execution_times.append(env_demo.current_step)
    slice_sizes.append(slice_size)
    next_state, _, done = env_demo.step(slice_size)
    remaining -= slice_size
    state = next_state
    if done:
        break

ax4.bar(execution_times, slice_sizes, alpha=0.7, color='purple', edgecolor='black')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Slice Size (shares)')
ax4.set_title('RL Agent: Adaptive Slice Sizing')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Market conditions during demo execution
ax5 = axes[1, 1]
ax5_spread = ax5.twinx()
ax5.plot(env_demo.volumes, 'b-', linewidth=2, label='Volume')
ax5_spread.plot(env_demo.spreads * 10000, 'r-', linewidth=2, label='Spread (bps)')
ax5.scatter(execution_times, [env_demo.volumes[t] for t in execution_times],
           s=100, c='blue', alpha=0.5, zorder=5)
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Volume', color='blue')
ax5_spread.set_ylabel('Spread (bps)', color='red')
ax5.set_title('RL Agent: Market Condition Awareness')
ax5.tick_params(axis='y', labelcolor='blue')
ax5_spread.tick_params(axis='y', labelcolor='red')
ax5.grid(True, alpha=0.3)

# Plot 6: Q-value heatmap (sample state slice)
ax6 = axes[1, 2]
# Sample Q-values for different spread and time combinations
spreads_sample = np.linspace(0.001, 0.01, 10)
times_sample = np.linspace(0, 1, 10)
q_matrix = np.zeros((10, 10))

for i, spread in enumerate(spreads_sample):
    for j, time_rem in enumerate(times_sample):
        sample_state = np.array([spread, 1.0, 0.0, time_rem])
        state_key = agent.discretize_state(sample_state)
        if state_key in agent.q_table:
            q_matrix[i, j] = np.max(agent.q_table[state_key])

im = ax6.imshow(q_matrix, aspect='auto', cmap='viridis', origin='lower')
ax6.set_xlabel('Time Remaining (normalized)')
ax6.set_ylabel('Spread (bps)')
ax6.set_title('Learned Q-Values: Spread vs Time')
ax6.set_xticks(np.arange(0, 10, 2))
ax6.set_xticklabels([f'{t:.1f}' for t in times_sample[::2]])
ax6.set_yticks(np.arange(0, 10, 2))
ax6.set_yticklabels([f'{s*10000:.0f}' for s in spreads_sample[::2]])
plt.colorbar(im, ax=ax6, label='Max Q-Value')

plt.tight_layout()
plt.savefig('adaptive_execution_rl.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: adaptive_execution_rl.png")
plt.show()
```

## 6. Challenge Round
Why do ML-based execution algorithms fail?
- Overfitting: Model memorizes training patterns → Breaks in new regimes (COVID, meme stocks)
- Adversarial gaming: Market makers detect algo patterns → Widen spreads when algo trades
- Reward hacking: RL agent discovers unintended exploit (e.g., cancel-replace spam) → Regulatory violation
- Data snooping: Backtest on same data used for training → Illusory alpha disappears live
- Non-stationarity: Market microstructure evolves (reg changes, HFT competition) → Stale models degrade

## 7. Key References
- [Cartea, Jaimungal, Penalva: Algorithmic & HFT (2015)](https://www.cambridge.org/core/books/algorithmic-and-highfrequency-trading/802D4E1B2C7E8E0B3B8F6C4F2C0B6D1F)
- [Nevmyvaka et al., "Reinforcement Learning for Optimal Execution" (2006)](https://www.icml.cc/Conferences/2006/proceedings/papers/415_Reinforcement_Lear.pdf)
- [Hendricks & Wilcox, "RL for Market Making" (2014)](https://arxiv.org/abs/1406.1507)

---
**Status:** Research frontier with production adoption | **Complements:** TWAP, POV, Optimal Control Models