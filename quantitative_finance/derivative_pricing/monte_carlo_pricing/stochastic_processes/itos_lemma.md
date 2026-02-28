# Ito's Lemma

## Concept Skeleton
**Definition:** Stochastic calculus chain rule for functions of Ito processes  
**Purpose:** Derive SDEs for transformed variables, e.g., $\ln S$ in GBM  
**Prerequisites:** Brownian motion, partial derivatives, SDEs

## Comparative Framing
| Rule | Ordinary Chain Rule | Ito's Lemma |
|---|---|---|
| **Extra Term** | No | Yes ($\frac12 \sigma^2 f_{xx}$) |
| **Noise** | Deterministic | Stochastic |
| **Use** | ODEs | SDEs |

## Examples + Counterexamples
**Simple Example:**  
For GBM, Ito gives $d\ln S = (\mu-\tfrac12\sigma^2)dt + \sigma dW$.

**Failure Case:**  
Using ordinary chain rule misses the $\tfrac12\sigma^2$ term; produces biased drift.

**Edge Case:**  
If $\sigma=0$, Ito reduces to standard chain rule.

## Layer Breakdown
```
Ito's Lemma:
├─ Process: dX = a(X,t)dt + b(X,t)dW
├─ Function: Y = f(X,t)
├─ Result:
│   └─ dY = (f_t + a f_x + ½ b^2 f_{xx}) dt + b f_x dW
└─ Use:
    ├─ Solve SDEs
    └─ Derive distributions
```

**Interaction:** Apply Ito → transform SDE → simulate or solve

## Challenge Round
**Q1:** Why does Ito add the second derivative term?  
**A1:** Brownian motion has non-zero quadratic variation; $dW^2=dt$.

**Q2:** What is quadratic variation?  
**A2:** The sum of squared increments converges to time: $\sum (\Delta W)^2 \to T$.

**Q3:** When is Ito not applicable?  
**A3:** For processes with jumps; use Ito–Lévy formula.

**Q4:** Why is Ito essential for GBM?  
**A4:** It yields the correct lognormal distribution by adjusting drift.

## Key References
- [Itô's lemma](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma)

---
**Status:** Core stochastic calculus tool | **Complements:** GBM, jump diffusion
