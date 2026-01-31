# Sampling Bias & Representative Sampling

## 5.1 Concept Skeleton
**Definition:** Bias when sample differs systematically from population; Representative when sample reflects population  
**Purpose:** Ensure inferences from sample generalize to population  
**Prerequisites:** Population concepts, sampling methods, probability

## 5.2 Comparative Framing
| Sampling Method | Representativeness | Bias Risk | Use Case |
|-----------------|-------------------|-----------|----------|
| **Random** | High if n large | None (theor.) | Gold standard |
| **Stratified** | High if strata defined well | Minimal | Ensure groups included |
| **Convenience** | Low (volunteers) | Selection bias | Pilot/exploratory |
| **Cluster** | Depends on cluster homogeneity | Geographic bias | Large populations |
| **Systematic** | If population random order | Periodic pattern bias | Structured lists |

## 5.3 Examples + Counterexamples

**Classic Bias Example:**  
1936 election poll: Surveyed car owners + phone owners (wealthy) → Wrong prediction. Poor people couldn't afford these.

**Good Sampling:**  
Exit polls (random sample of voters leaving polls) → usually accurate

**Edge Case:**  
Nonresponse bias: People who respond to survey differ from non-responders (depression prevalence underestimated if depressed less likely to respond)

## 5.4 Layer Breakdown
```
Sources of Bias:
├─ Selection Bias: Who gets sampled (excluded groups)
├─ Nonresponse Bias: Who responds (self-selection)
├─ Measurement Bias: How we measure (question wording)
├─ Survivorship Bias: Only observe survivors (survivorship)
└─ Healthy User Bias: People making effort differ systematically
```

## 5.5 Mini-Project
Detect selection bias in data:
```python
# Survey: "How much do you value exercise?"
# Bias: Surveyed at gym (selection) vs online (self-selection)
# Result: Mean rating at gym > online (biased upward at gym)

import numpy as np
import matplotlib.pyplot as plt

gym_ratings = np.random.normal(8, 1, 100)  # Biased high
online_ratings = np.random.normal(6, 1, 100)  # Lower (non-exercisers included)

print(f"Gym mean: {gym_ratings.mean():.2f}")
print(f"Online mean: {online_ratings.mean():.2f}")

# Solution: Weight responses inversely to inclusion probability
```

## 5.6 Challenge Round
When is perfect representativeness impossible?
- Studying homeless populations (no sampling frame)
- Rare diseases (can't find enough cases)
- Studying past events (survivors only exist)

When can you proceed despite bias?
- Bias direction known (can adjust estimates)
- Internal comparisons (comparing subgroups within sample)
- Machine learning (predictive accuracy matters more than rep.)

## 5.7 Key References
- [Sampling Bias Examples](https://en.wikipedia.org/wiki/Sampling_bias)
- [Literary Digest Fiasco (1936)](https://en.wikipedia.org/wiki/1936_Literary_Digest_prediction_error)
- [Selection Bias in Clinical Trials](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3879437/)

---
**Status:** Foundational concept | **Complements:** Experimental Design, Data Collection
