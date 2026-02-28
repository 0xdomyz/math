# Loss Given Default (LGD)

## 1. Concept Skeleton
**Definition:** Proportion of exposure lost when borrower defaults; LGD = (EAD - Recoveries) / EAD; 1 - recovery rate; varies by seniority, collateral type, and economic conditions; key driver of loss severity in credit portfolios; pro-cyclical (increases in downturns when collateral values fall simultaneously with defaults).

**Purpose:**
- Quantify severity of default impact per unit exposure; translates default probability into actual loss amount
- Determine regulatory capital required (Basel III capital charge = K(PD, LGD, EAD); capital ∝ LGD)
- Price credit spreads and interest rates; LGD determines risk premium and profit margins
- Model recovery processes and collateral dynamics; distinguish secured vs unsecured risks
- Differentiate capital requirements: secured mortgages (LGD 10-25%) vs unsecured cards (LGD 70-100%)
- Stress-test portfolio under downturn LGD scenarios (correlation of high defaults with asset price declines)

**Prerequisites:**
- Collateral valuation methods (appraisal, market-based, income approach, liquidation value basics)
- Recovery processes and bankruptcy law (seniority priority, creditor hierarchy, reorganization vs liquidation)
- Portfolio data: loan structure, collateral types, seniority levels, subordination provisions
- Economic cycle dynamics: asset price volatility correlated with defaults (pro-cyclical risk)
- Workout duration and recovery timing; present value discounting of multi-year recoveries
- Industry LGD benchmarks (mortgages 10-25%, auto loans 20-35%, unsecured 70-100%)

## 2. Comparative Framing
| Loan Type | Typical LGD | Collateral | Recovery Rate | Key Driver |
|-----------|------------|-----------|---------------|------------|
| **Secured Mortgage** | 10%-25% | Real estate | 75%-90% | Property value decline |
| **Auto Loan** | 20%-35% | Vehicle | 65%-80% | Repossession/sale delays |
| **Unsecured Credit Card** | 70%-100% | None | 0%-30% | Bankruptcy treatment |
| **Corporate Bonds** | 30%-50% | Firm assets | 50%-70% | Capital structure, bankruptcy code |
| **Trade Finance** | 5%-20% | Inventory/goods | 80%-95% | Collateral liquidity |

## 3. Examples + Counterexamples

**Simple Example:**  
$100K mortgage, property worth $120K at default. After sale costs (5%), net recovery $114K. LGD = (100-114)/100 = -14% (gain!) â†’ Use LGD=0

**Failure Case:**  
Assuming constant LGD across economic cycle. 2008: Real estate LGD doubled as property prices fell 50%. Fixed models missed this

**Edge Case:**  
Unsecured loan during pandemic; borrower has no recovery value but later returns to work. Time-dependent LGD; recovery may take years

## 4. Layer Breakdown
```
Loss Given Default Framework:
â”œâ”€ LGD Components:
â”‚   â”œâ”€ Collateral value at default: Market price at time of loss
â”‚   â”œâ”€ Recovery amount: Proceeds from liquidation
â”‚   â”œâ”€ Recovery costs: Legal, administrative, sales friction
â”‚   â”œâ”€ Recovery timing: When received (present value discount)
â”‚   â””â”€ Seniority: Priority in bankruptcy (affects recovery rank)
â”œâ”€ Types of Recovery:
â”‚   â”œâ”€ Collateral sales: Secured assets liquidated
â”‚   â”œâ”€ Debt restructuring: Waive/extend obligations
â”‚   â”œâ”€ Guarantees: Third-party payment
â”‚   â””â”€ Bankruptcy proceeds: Distribution from estate
â”œâ”€ LGD Dynamics:
â”‚   â”œâ”€ Pro-cyclical: LGD rises in downturns (collateral values fall)
â”‚   â”œâ”€ Correlation with PD: High default rates + low recoveries compound losses
â”‚   â””â”€ Workout duration: Short-term vs long-term recovery timelines
â”œâ”€ LGD Levels by Collateral:
â”‚   â”œâ”€ High recovery (LGD 5-20%): Liquid collateral (cash, securities)
â”‚   â”œâ”€ Medium recovery (LGD 20-50%): Real estate, inventory
â”‚   â”œâ”€ Low recovery (LGD 50-100%): Unsecured, subordinated debt
â”‚   â””â”€ Recovery hierarchy: Senior secured â†’ unsecured â†’ subordinated
â””â”€ Valuation Methods:
    â”œâ”€ Appraisal: Professional assessment
    â”œâ”€ Market-based: Comparable sales
    â”œâ”€ Income approach: Cash flow valuation
    â””â”€ Liquidation value: Fire-sale price
```

## 5. Challenge Round
When is LGD estimation problematic?
- **Collateral value volatility**: Real estate, equity collateral highly pro-cyclical; price crashes during defaults
- **Recovery correlation**: PD and LGD often positively correlated (defaults cluster with falling collateral values)
- **Workout uncertainty**: Recovery timelines variable; months to years affect present value significantly
- **Seniority complexity**: Multiple creditors, subordination structures; recovery depends on bankruptcy code
- **Fraud/deterioration**: Collateral may be hidden or rapidly deteriorate post-default (reputational damage)

## 6. Key References
- [Basel III LGD Standards](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory LGD definitions; floor values (0-35% for secured, 0% for guarantees); downturn LGD concept; IRB A-IRB estimation requirements and supervisor approval process.

- Altman, E. Z., Resti, A., & Sironi, A. (2004). "Analyzing and Explaining Default Recovery Rates." ISDA Research Report. Empirical study of 1500+ defaults 1970-2003; LGD varies 20-35% by seniority class; documents pro-cyclicality (LGD doubles in downturns).

- [Collateral Valuation Methods](https://en.wikipedia.org/wiki/Collateral_(finance)) - Comprehensive overview of appraisal approaches (cost, market-based, income), haircut calculations, and mark-to-market procedures; includes regulatory frameworks.

- [Bankruptcy Code Recovery Hierarchy](https://en.wikipedia.org/wiki/Priority_of_claims) - Detailed treatment of seniority structure; secured vs unsecured creditors; subordination provisions; international bankruptcy code variations (US, UK, EU).

- Carey, M., & Gordy, M. (2003). "Recovery Risk in Credit Cards: A Study of Default and Recoveries from Charge-Offs." Federal Reserve Board Finance and Economics Discussion Series 2003-38. Empirical LGD estimates for consumer credit; recovery time lags 12-36 months post-default; present value adjustments critical.

- Bellotti, T., & Crook, J. (2012). "Loss Given Default Models for Retail Credit Portfolios: A Comparative Study and Implementation Issues." European Journal of Operational Research, 218(2), 412-422. Comparative analysis of LGD estimation methods; regression vs machine learning; collateral correlation impact on downturn LGD.

---
**Status:** Key severity parameter for credit losses | **Complements:** Credit Risk Definition, PD, EAD, Expected Loss
