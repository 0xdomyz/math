# Credit Risk Modelling Topics Guide

**Reference of foundational and advanced credit risk modelling concepts with categories, brief descriptions, and sources.**

---

## I. Credit Risk Fundamentals
| Topic | Description | Source |
|-------|-------------|--------|
| **Credit Risk Definition** | Risk of loss from borrower default or credit deterioration | [Investopedia](https://www.investopedia.com/terms/c/creditrisk.asp) |
| **Default Probability (PD)** | Likelihood borrower fails to meet obligations | [BIS](https://www.bis.org/basel_framework/chapter/CRE/20.htm) |
| **Loss Given Default (LGD)** | Proportion of exposure lost if default occurs | [BIS](https://www.bis.org/basel_framework/chapter/CRE/20.htm) |
| **Exposure at Default (EAD)** | Total value exposed at time of default | [BIS](https://www.bis.org/basel_framework/chapter/CRE/20.htm) |
| **Expected Loss (EL)** | EL = PD × LGD × EAD | [BIS](https://www.bis.org/basel_framework/chapter/CRE/20.htm) |

---

## II. Credit Risk Modelling Approaches
| Topic | Description | Source |
|-------|-------------|--------|
| **Scorecard Models** | Logistic regression for default prediction | [Moody's Analytics](https://www.moodysanalytics.com/risk-perspectives-magazine/managing-credit-risk/scorecard-models) |
| **Structural Models** | Default as firm value falls below threshold (Merton model) | [Wikipedia](https://en.wikipedia.org/wiki/Merton_model) |
| **Reduced-Form Models** | Default as random process (intensity-based) | [Wikipedia](https://en.wikipedia.org/wiki/Reduced_form_model) |
| **Transition Matrices** | Credit rating migration probabilities | [BIS](https://www.bis.org/publ/bcbs_wp16.pdf) |
| **Machine Learning Models** | Tree-based, SVM, neural networks for credit scoring | [Nature ML Credit Risk](https://www.nature.com/articles/s41599-019-0308-7) |

---

## III. Portfolio Credit Risk
| Topic | Description | Source |
|-------|-------------|--------|
| **Credit VaR** | Value-at-risk for credit portfolios | [BIS](https://www.bis.org/publ/bcbs118.pdf) |
| **Credit Correlation** | Default dependence between obligors | [Wikipedia](https://en.wikipedia.org/wiki/Credit_risk#Credit_risk_modelling) |
| **Concentration Risk** | Risk from large exposures to single obligor/sector | [BIS](https://www.bis.org/publ/bcbs189.pdf) |
| **Granularity Adjustment** | Correction for finite portfolio size | [BIS](https://www.bis.org/publ/bcbs118.pdf) |
| **Stress Testing** | Scenario analysis for adverse conditions | [BIS](https://www.bis.org/basel_framework/chapter/SCO/30.htm) |

---

## IV. Regulatory & Practical Aspects
| Topic | Description | Source |
|-------|-------------|--------|
| **Basel Accords (II/III)** | International standards for credit risk capital | [BIS Basel III](https://www.bis.org/bcbs/basel3.htm) |
| **Internal Ratings-Based (IRB)** | Bank-specific risk parameter estimation | [BIS](https://www.bis.org/basel_framework/chapter/CRE/20.htm) |
| **Credit Risk Mitigation** | Collateral, guarantees, netting to reduce risk | [BIS](https://www.bis.org/basel_framework/chapter/CRE/22.htm) |
| **Model Validation** | Backtesting, benchmarking, regulatory review | [BIS](https://www.bis.org/basel_framework/chapter/CRE/41.htm) |
| **IFRS 9/CECL** | Accounting standards for expected credit loss | [IFRS 9](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) |

---

## Reference Sources
| Source | URL | Coverage |
|--------|-----|----------|
| **BIS Basel Framework** | https://www.bis.org/basel_framework/ | Regulatory standards, definitions, methodologies |
| **Investopedia Credit Risk** | https://www.investopedia.com/terms/c/creditrisk.asp | General concepts, examples |
| **Moody's Analytics** | https://www.moodysanalytics.com | Industry practice, scorecard models |
| **Wikipedia Credit Risk** | https://en.wikipedia.org/wiki/Credit_risk | Overview, model types |

---

## Quick Stats
- **Total Topics Documented**: 20+
- **Categories**: 4
- **Coverage**: Fundamentals → Modelling → Portfolio → Regulation
