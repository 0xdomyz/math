# Randomness & Fairness

## 1. Concept Skeleton
**Definition:** Processes where outcomes are unpredictable and all possibilities have equal (or stated) probability; fairness ensures stated probabilities match actual implementation  
**Purpose:** Guarantee game integrity, prevent bias/cheating, build trust, meet regulatory requirements  
**Prerequisites:** Probability distributions, independence, statistical testing

## 2. Comparative Framing
| Property | True Randomness | Pseudo-Random | Biased/Unfair | Predictable |
|----------|-----------------|---------------|---------------|------------|
| **Source** | Physical entropy | Deterministic algorithm + seed | Systematic deviation | Exploitable pattern |
| **Unpredictability** | Quantum-based | Computationally hard | Unknown bias exists | Known pattern |
| **Reproducibility** | None | Yes (same seed) | Yes | Yes |
| **Example** | Radioactive decay | MT19937 RNG | Weighted die | Card counting |

## 3. Examples + Counterexamples

**Simple Example:**  
Fair coin: P(heads) = P(tails) = 50%. Over 1,000,000 flips, observe 499,950 heads → extremely close to 50%.

**Failure Case:**  
Assuming historical frequency = true probability after few trials. After 10 flips, 7 heads doesn't prove coin is biased (binomial variance).

**Edge Case:**  
Extremely long sequences appear biased. 1,000,000 fair coin flips guaranteed to have long runs of heads/tails; this is expected, not bias.

## 4. Layer Breakdown
```
Randomness & Fairness Framework:
├─ True Randomness Properties:
│   ├─ Independence: Outcome doesn't depend on prior outcomes
│   ├─ Unpredictability: Cannot forecast future results
│   ├─ Uniform distribution: Each outcome equally likely
│   ├─ Entropy: Measure of information/disorder
│   └─ No patterns: No autocorrelation or patterns exploitable
├─ Random Number Generation (RNG):
│   ├─ Hardware RNG: Physical entropy (radioactive decay, thermal noise)
│   │   Pros: True randomness; Cons: Slow, expensive
│   ├─ Pseudo-RNG (PRNG): Deterministic algorithm
│   │   Mersenne Twister, xorshift, PCG
│   │   Pros: Fast, reproducible; Cons: Deterministic if seed known
│   ├─ Cryptographic RNG: PRNG with unpredictable seed
│   │   /dev/urandom, ChaCha20
│   │   Pros: Unpredictable; Cons: Slower than standard PRNG
│   └─ Seeding: Entropy source must be unpredictable
├─ Testing for Fairness:
│   ├─ Chi-square test: Observed vs expected frequencies
│   ├─ Kolmogorov-Smirnov: Distribution conformity
│   ├─ Runs test: Detect patterns/streaks
│   ├─ Spectral test: Check dimension distribution
│   ├─ ENT: Entropy analysis
│   └─ NIST Test Suite: Comprehensive randomness battery
├─ Sources of Bias:
│   ├─ Hardware defects: Physical asymmetry (die, cards, wheel)
│   ├─ Wear patterns: Roulette wheel tilt, wheel speed variation
│   ├─ Dealer skills: Controlled shuffles, peeking
│   ├─ Software bugs: RNG implementation errors
│   ├─ Predictable seed: Time-based seed from system clock
│   └─ Insufficient entropy: Multiple outcomes depend on same seed
├─ Regulatory Requirements:
│   ├─ Certification: Independent labs (GLI, eCOGRA) test RNG
│   ├─ Audit trails: Log all transactions, reproducibility
│   ├─ Frequency testing: Regular statistical verification
│   ├─ Source code review: Third-party code inspection
│   ├─ Physical inspection: Equipment checked for tampering
│   └─ Probabilities disclosed: Player informed of actual odds
└─ Fairness in Practice:
    ├─ Casinos: Math-certified, audited, sealed machines
    ├─ Online: SSL encryption, audited software, payout verification
    ├─ Peer-to-peer: Decentralized verification (blockchain)
    └─ Self-managed: User verifies with cryptographic hashes
```

## 5. Mini-Project
Test randomness and detect bias:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Example 1: Test for fairness (die)
print("=== Fairness Test: Six-Sided Die ===\n")

# Fair die (expected 1000 of each face)
fair_rolls = np.random.randint(1, 7, 6000)
fair_counts = np.bincount(fair_rolls, minlength=7)[1:]

# Biased die (favor 6)
biased_rolls = np.random.choice([1, 2, 3, 4, 5, 6], 6000, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
biased_counts = np.bincount(biased_rolls, minlength=7)[1:]

# Expected frequencies
expected = np.array([1000] * 6)

# Chi-square test
chi2_fair, p_fair = stats.chisquare(fair_counts, expected)
chi2_biased, p_biased = stats.chisquare(biased_counts, expected)

print(f"Fair die:")
print(f"  Observed: {fair_counts}")
print(f"  Chi-square: {chi2_fair:.4f}, p-value: {p_fair:.4f}")
print(f"  Conclusion: {'Fair' if p_fair > 0.05 else 'Biased'} (α=0.05)")

print(f"\nBiased die (favor 6):")
print(f"  Observed: {biased_counts}")
print(f"  Chi-square: {chi2_biased:.4f}, p-value: {p_biased:.4f}")
print(f"  Conclusion: {'Fair' if p_biased > 0.05 else 'Biased'}")

# Example 2: Runs test (detect patterns)
print("\n\n=== Runs Test: Detect Patterns ===\n")

# Fair coin flips
fair_flips = np.random.randint(0, 2, 1000)

# Flips with pattern (alternating tendency)
pattern_flips = np.array([i % 2 for i in range(1000)])

def runs_test(sequence):
    """Count runs (consecutive same values)"""
    runs = 1
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            runs += 1
    return runs

fair_runs = runs_test(fair_flips)
pattern_runs = runs_test(pattern_flips)

print(f"Fair flips (1000 values):")
print(f"  Observed runs: {fair_runs}")
print(f"  Expected runs: ~500")

print(f"\nPattern flips (alternating):")
print(f"  Observed runs: {pattern_runs}")
print(f"  Expected runs: ~500")
print(f"  Conclusion: Pattern flips have TOO MANY runs (predictable)")

# Example 3: Autocorrelation analysis
print("\n\n=== Autocorrelation Analysis ===\n")

# Fair sequence
fair_seq = np.random.randn(1000)

# Autocorrelated sequence (each value depends on previous)
autocorr_seq = np.zeros(1000)
autocorr_seq[0] = np.random.randn()
for i in range(1, 1000):
    autocorr_seq[i] = 0.8 * autocorr_seq[i-1] + 0.2 * np.random.randn()

# Calculate autocorrelation
acf_fair = np.correlate(fair_seq - fair_seq.mean(), fair_seq - fair_seq.mean(), mode='full')
acf_autocorr = np.correlate(autocorr_seq - autocorr_seq.mean(), 
                            autocorr_seq - autocorr_seq.mean(), mode='full')

print(f"Fair sequence autocorrelation at lag 1: {acf_fair[1000]/acf_fair[1000]:.4f}")
print(f"Autocorrelated sequence at lag 1: {acf_autocorr[1001]/acf_autocorr[1000]:.4f}")

# Example 4: Entropy calculation
print("\n\n=== Entropy Analysis ===\n")

def shannon_entropy(sequence):
    """Calculate Shannon entropy (0 = totally predictable, log₂(n) = fully random)"""
    unique, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / len(sequence)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

# Generate sequences
fair_seq = np.random.randint(0, 2, 10000)  # Fair coins
biased_seq = np.random.choice([0, 1], 10000, p=[0.9, 0.1])  # 90% zeros
uniform_seq = np.random.randint(0, 256, 10000)  # Uniform 0-255

ent_fair = shannon_entropy(fair_seq)
ent_biased = shannon_entropy(biased_seq)
ent_uniform = shannon_entropy(uniform_seq)

print(f"Fair coin flips (1000 values): H = {ent_fair:.4f} bits")
print(f"  Expected for fair coin: 1.0 bits")
print(f"  (Higher = more random)")

print(f"\nBiased coin (90% zeros): H = {ent_biased:.4f} bits")
print(f"  (Lower entropy = more predictable)")

print(f"\nUniform 0-255: H = {ent_uniform:.4f} bits")
print(f"  Expected for 256 values: 8.0 bits")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Die fairness
axes[0, 0].bar(range(1, 7), fair_counts, alpha=0.5, label='Fair', color='green')
axes[0, 0].bar(range(1, 7), biased_counts, alpha=0.5, label='Biased', color='red')
axes[0, 0].axhline(1000, color='black', linestyle='--', linewidth=2, label='Expected (1000)')
axes[0, 0].set_xlabel('Face Value')
axes[0, 0].set_ylabel('Frequency (6000 rolls)')
axes[0, 0].set_title('Chi-Square Test: Fair vs Biased Die')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Runs test
sequences = ['Fair\nFlips', 'Pattern\nFlips', 'Expected']
run_counts = [fair_runs, pattern_runs, 500]
colors_runs = ['green', 'red', 'gray']

axes[0, 1].bar(sequences, run_counts, color=colors_runs, alpha=0.7)
axes[0, 1].set_ylabel('Number of Runs')
axes[0, 1].set_title('Runs Test: Detect Alternating Patterns')
axes[0, 1].set_ylim([0, 1000])
for i, count in enumerate(run_counts):
    axes[0, 1].text(i, count + 20, f'{count}', ha='center', fontweight='bold')

# Plot 3: Autocorrelation
lags = np.arange(-50, 51)
acf_fair_norm = np.correlate(fair_seq - fair_seq.mean(), fair_seq - fair_seq.mean(), 
                             mode='full')[950:1051] / np.correlate(fair_seq - fair_seq.mean(), 
                                                                    fair_seq - fair_seq.mean(), 
                                                                    mode='full')[1000]
acf_autocorr_norm = np.correlate(autocorr_seq - autocorr_seq.mean(), 
                                 autocorr_seq - autocorr_seq.mean(), 
                                 mode='full')[950:1051] / np.correlate(autocorr_seq - autocorr_seq.mean(), 
                                                                        autocorr_seq - autocorr_seq.mean(), 
                                                                        mode='full')[1000]

axes[1, 0].plot(lags, acf_fair_norm, linewidth=2, label='Fair sequence', color='green')
axes[1, 0].plot(lags, acf_autocorr_norm, linewidth=2, label='Autocorrelated', color='red')
axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].fill_between(lags, -1.96/np.sqrt(1000), 1.96/np.sqrt(1000), alpha=0.2, color='gray')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')
axes[1, 0].set_title('Autocorrelation Function: Detect Dependencies')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Entropy
sequences_ent = ['Fair\nCoin', 'Biased\n(90% zeros)', 'Uniform\n0-255']
entropies = [ent_fair, ent_biased, ent_uniform]
max_entropies = [1, 1, 8]
colors_ent = ['green', 'red', 'blue']

x_pos = np.arange(len(sequences_ent))
width = 0.35

axes[1, 1].bar(x_pos - width/2, entropies, width, label='Observed', alpha=0.7, color=colors_ent)
axes[1, 1].bar(x_pos + width/2, max_entropies, width, label='Maximum', alpha=0.3, color=colors_ent)
axes[1, 1].set_ylabel('Entropy (bits)')
axes[1, 1].set_title('Shannon Entropy: Randomness Measure')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(sequences_ent)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is randomness testing insufficient?
- Long-term patterns beyond statistical detection window
- Correlated outcomes (multiple wheels in casino drift together)
- Conditional bias (bias only under certain conditions)
- Hardware deterioration (wheel becomes biased over time)
- Social engineering (insider knowledge bypasses tests)

## 7. Key References
- [Wikipedia: Randomness](https://en.wikipedia.org/wiki/Randomness)
- [NIST Randomness Test Suite](https://csrc.nist.gov/publications/detail/sp/800-22/final)
- [Cryptographically Secure Random Number Generation](https://en.wikipedia.org/wiki/Cryptographically_secure_pseudorandom_number_generator)
- [Casino Gaming Standards](https://glivegas.com)

---
**Status:** Trust and integrity foundation | **Complements:** Basic Probability, House Edge, Fairness Verification
