# Ito's Lemma

## 1. Concept Skeleton
**Definition:** Stochastic calculus chain rule for functions of Ito processes  
**Purpose:** Derive SDEs for transformed variables, e.g., $\ln S$ in GBM  
**Prerequisites:** Brownian motion, partial derivatives, SDEs

## 2. Comparative Framing
| Rule | Ordinary Chain Rule | Ito's Lemma |
|---|---|---|
| **Extra Term** | No | Yes ($\frac12 \sigma^2 f_{xx}$) |
| **Noise** | Deterministic | Stochastic |
| **Use** | ODEs | SDEs |

## 3. Examples + Counterexamples

**Simple Example:**  
For GBM, Ito gives $d\ln S = (\mu-\tfrac12\sigma^2)dt + \sigma dW$.

**Failure Case:**  
Using ordinary chain rule misses the $\tfrac12\sigma^2$ term; produces biased drift.

**Edge Case:**  
If $\sigma=0$, Ito reduces to standard chain rule.

## 4. Layer Breakdown
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

## 5. Mini-Project
Verify Ito correction for GBM log transform:
```python
import sympy as sp

S, mu, sigma, t = sp.symbols('S mu sigma t', positive=True)

f = sp.log(S)
fx = sp.diff(f, S)
fxx = sp.diff(fx, S)

# Ito terms
ito_drift = mu*S*fx + sp.Rational(1,2)*sigma**2*S**2*fxx
ito_diff = sigma*S*fx

print("Ito drift term:", sp.simplify(ito_drift))
print("Ito diffusion term:", sp.simplify(ito_diff))
```

## 6. Challenge Round

**Q1:** Why does Ito add the second derivative term?  
**A1:** Brownian motion has non-zero quadratic variation; $dW^2=dt$.

**Q2:** What is quadratic variation?  
**A2:** The sum of squared increments converges to time: $\sum (\Delta W)^2 \to T$.

**Q3:** When is Ito not applicable?  
**A3:** For processes with jumps; use Ito–Lévy formula.

**Q4:** Why is Ito essential for GBM?  
**A4:** It yields the correct lognormal distribution by adjusting drift.

## 7. Key References
- [Itô's lemma](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma)

---
**Status:** Core stochastic calculus tool | **Complements:** GBM, jump diffusion
