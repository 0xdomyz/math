# Econometrics Topics Guide

**Complete reference of econometric theory and applied methods with categories, brief descriptions, and sources.**

---

## I. Classical Linear Regression Model

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Simple Linear Regression** | N/A | Y = β₀ + β₁X + ε; bivariate relationship estimation | [Greene - Econometric Analysis](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899) |
| **Multiple Regression** | N/A | Y = β₀ + β₁X₁ + ... + βₖXₖ + ε; multivariate relationships | [Wooldridge - Introductory Econometrics](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge) |
| **Ordinary Least Squares (OLS)** | N/A | Minimize Σ(yᵢ - ŷᵢ)²; BLUE under Gauss-Markov assumptions | [Wiki - OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares) |
| **Gauss-Markov Theorem** | N/A | OLS = Best Linear Unbiased Estimator under classical assumptions | [Wiki - Gauss-Markov](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) |
| **Classical Assumptions** | N/A | Linearity, exogeneity, homoscedasticity, no autocorrelation, no multicollinearity | [Wooldridge Ch. 3-5](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge) |
| **R-squared & Adjusted R²** | N/A | Goodness-of-fit; proportion of variance explained; adjusted penalizes complexity | [Wiki - R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination) |
| **Residual Analysis** | N/A | Diagnostic plots; normality tests, patterns indicate violations | [Greene Ch. 5](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899) |

---

## II. Violations of Classical Assumptions

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Heteroscedasticity** | N/A | Non-constant error variance; biased SE; White/robust standard errors | [Wiki - Heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity) |
| **Autocorrelation** | N/A | Correlated errors across time/space; Durbin-Watson test; Newey-West SE | [Wiki - Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) |
| **Multicollinearity** | N/A | High correlation among regressors; inflates SE; VIF detection | [Wiki - Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity) |
| **Endogeneity** | N/A | Corr(X, ε) ≠ 0; omitted variables, measurement error, simultaneity | [Wiki - Endogeneity](https://en.wikipedia.org/wiki/Endogeneity_(econometrics)) |
| **Omitted Variable Bias** | N/A | Missing relevant X causes biased β̂; direction depends on corr(X₁, X₂) | [Wooldridge Ch. 3.3](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge) |
| **Measurement Error** | N/A | Errors in X or Y attenuate coefficients; classical vs non-classical | [Greene Ch. 8.3](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899) |

---

## III. Instrumental Variables & Causal Inference

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Instrumental Variables (IV)** | N/A | Address endogeneity; Z correlated with X but not ε; 2SLS estimation | [Wiki - IV](https://en.wikipedia.org/wiki/Instrumental_variables_estimation) |
| **Two-Stage Least Squares (2SLS)** | N/A | First stage: regress X on Z; second stage: Y on X̂ | [Wooldridge Ch. 15](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge) |
| **Generalized Method of Moments (GMM)** | N/A | Flexible estimation via moment conditions; overidentification tests | [Wiki - GMM](https://en.wikipedia.org/wiki/Generalized_method_of_moments) |
| **Difference-in-Differences (DiD)** | N/A | Treatment vs control pre/post intervention; parallel trends assumption | [Wiki - DiD](https://en.wikipedia.org/wiki/Difference_in_differences) |
| **Regression Discontinuity Design** | N/A | Causal inference at threshold; sharp vs fuzzy RDD | [Wiki - RDD](https://en.wikipedia.org/wiki/Regression_discontinuity_design) |
| **Propensity Score Matching** | N/A | Match treated and control by P(treatment|X); observational studies | [Wiki - PSM](https://en.wikipedia.org/wiki/Propensity_score_matching) |
| **Randomized Controlled Trials (RCT)** | N/A | Gold standard for causality; random assignment eliminates selection bias | [Wiki - RCT](https://en.wikipedia.org/wiki/Randomized_controlled_trial) |

---

## IV. Time Series Econometrics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Stationarity** | N/A | Constant mean/variance over time; unit root tests (ADF, KPSS) | [Wiki - Stationary Process](https://en.wikipedia.org/wiki/Stationary_process) |
| **Unit Root Tests** | N/A | Dickey-Fuller, Phillips-Perron; test for non-stationarity | [Wiki - Unit Root](https://en.wikipedia.org/wiki/Unit_root) |
| **ARIMA Models** | N/A | AutoRegressive Integrated Moving Average; Box-Jenkins methodology | [Wiki - ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) |
| **Vector Autoregression (VAR)** | N/A | System of equations; each variable function of own/others' lags | [Wiki - VAR](https://en.wikipedia.org/wiki/Vector_autoregression) |
| **Cointegration** | N/A | Long-run equilibrium relationship; Engle-Granger, Johansen tests | [Wiki - Cointegration](https://en.wikipedia.org/wiki/Cointegration) |
| **Error Correction Model (ECM)** | N/A | Short-run dynamics adjust to long-run equilibrium | [Wiki - ECM](https://en.wikipedia.org/wiki/Error_correction_model) |
| **ARCH/GARCH Models** | N/A | Model time-varying volatility; conditional heteroscedasticity | [Wiki - GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) |
| **Granger Causality** | N/A | X Granger-causes Y if X lags improve Y forecasts | [Wiki - Granger Causality](https://en.wikipedia.org/wiki/Granger_causality) |
| **Impulse Response Functions** | N/A | Dynamic effect of shock on system variables over time | [Wiki - IRF](https://en.wikipedia.org/wiki/Impulse_response) |

---

## V. Panel Data & Longitudinal Methods

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Panel Data Structure** | N/A | Cross-sectional units observed over time; N individuals, T periods | [Wooldridge Ch. 13-14](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge) |
| **Fixed Effects (FE)** | N/A | Control time-invariant unobserved heterogeneity; within estimator | [Wiki - Fixed Effects](https://en.wikipedia.org/wiki/Fixed_effects_model) |
| **Random Effects (RE)** | N/A | Treat individual effects as random; GLS estimation; efficiency gain | [Wiki - Random Effects](https://en.wikipedia.org/wiki/Random_effects_model) |
| **Hausman Test** | N/A | Test FE vs RE; H₀: random effects consistent | [Wiki - Hausman Test](https://en.wikipedia.org/wiki/Durbin%E2%80%93Wu%E2%80%93Hausman_test) |
| **Dynamic Panel Data** | N/A | Lagged dependent variable; Arellano-Bond GMM estimator | [Wiki - Dynamic Panel](https://en.wikipedia.org/wiki/Dynamic_panel_data) |
| **Clustered Standard Errors** | N/A | Account for within-cluster correlation; robust inference | [Cameron & Miller (2015)](http://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf) |

---

## VI. Limited Dependent Variables

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Binary Choice Models** | N/A | Y ∈ {0,1}; logit (logistic) vs probit (normal CDF) | [Wiki - Binary Regression](https://en.wikipedia.org/wiki/Binary_regression) |
| **Logistic Regression** | N/A | Log-odds linear; odds ratio interpretation; maximum likelihood | [Wiki - Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) |
| **Probit Model** | N/A | Latent variable framework; normal CDF link function | [Wiki - Probit Model](https://en.wikipedia.org/wiki/Probit_model) |
| **Multinomial Logit/Probit** | N/A | Multiple unordered outcomes; IIA assumption in multinomial logit | [Wiki - Multinomial Logit](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) |
| **Ordered Logit/Probit** | N/A | Ordered categorical outcomes; threshold model | [Wiki - Ordered Probit](https://en.wikipedia.org/wiki/Ordered_probit) |
| **Tobit Model** | N/A | Censored/truncated data; censored regression at threshold | [Wiki - Tobit](https://en.wikipedia.org/wiki/Tobit_model) |
| **Sample Selection** | N/A | Heckman correction; non-random sample causes bias | [Wiki - Heckman Correction](https://en.wikipedia.org/wiki/Heckman_correction) |
| **Count Data Models** | N/A | Poisson regression, Negative Binomial; Y = non-negative integers | [Wiki - Poisson Regression](https://en.wikipedia.org/wiki/Poisson_regression) |

---

## VII. Maximum Likelihood & Estimation Theory

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Maximum Likelihood Estimation** | N/A | Choose θ maximizing L(θ|data); asymptotically efficient | [Wiki - MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) |
| **Likelihood Ratio Test** | N/A | Compare nested models; -2log(L₀/L₁) ~ χ²(df) | [Wiki - LRT](https://en.wikipedia.org/wiki/Likelihood-ratio_test) |
| **Wald Test** | N/A | Test restrictions; (R̂β - r)'[RV̂R']⁻¹(R̂β - r) ~ χ²(q) | [Wiki - Wald Test](https://en.wikipedia.org/wiki/Wald_test) |
| **Lagrange Multiplier Test** | N/A | Score test; estimate under H₀ only; computationally efficient | [Wiki - Score Test](https://en.wikipedia.org/wiki/Score_test) |
| **Asymptotic Properties** | N/A | Consistency, asymptotic normality, efficiency; large sample theory | [Greene Ch. 4](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899) |
| **Delta Method** | N/A | Asymptotic distribution of function of estimator; Taylor expansion | [Wiki - Delta Method](https://en.wikipedia.org/wiki/Delta_method) |

---

## VIII. Bayesian Econometrics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Bayesian Inference** | N/A | Posterior ∝ Prior × Likelihood; incorporate prior beliefs | [Wiki - Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_inference) |
| **Prior Distributions** | N/A | Informative, non-informative, conjugate priors | [Wiki - Prior](https://en.wikipedia.org/wiki/Prior_probability) |
| **Markov Chain Monte Carlo (MCMC)** | N/A | Gibbs sampling, Metropolis-Hastings; sample from posterior | [Wiki - MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) |
| **Bayesian Model Comparison** | N/A | Bayes factors, DIC, WAIC; posterior predictive checks | [Wiki - Bayes Factor](https://en.wikipedia.org/wiki/Bayes_factor) |
| **Hierarchical Models** | N/A | Multi-level structure; pooling information across groups | [Wiki - Hierarchical Bayes](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling) |

---

## IX. Model Selection & Validation

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Information Criteria** | N/A | AIC, BIC, HQ; penalize complexity; lower is better | [Wiki - AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) |
| **Cross-Validation** | N/A | K-fold, leave-one-out; out-of-sample prediction accuracy | [Wiki - Cross-Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) |
| **Specification Tests** | N/A | RESET test, linktest; detect functional form misspecification | [Wiki - RESET](https://en.wikipedia.org/wiki/Ramsey_RESET_test) |
| **Goodness of Fit** | N/A | R², pseudo-R² (McFadden, Cox-Snell); likelihood-based measures | [Wiki - Pseudo-R²](https://en.wikipedia.org/wiki/Pseudo-R-squared) |
| **Outliers & Influential Points** | N/A | Cook's D, leverage, DFBETAS; robust regression methods | [Wiki - Cook's Distance](https://en.wikipedia.org/wiki/Cook%27s_distance) |

---

## X. Regularization & Machine Learning Methods

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Ridge Regression (L2)** | N/A | Penalize Σβ²; shrinks coefficients; reduces multicollinearity | [Wiki - Ridge](https://en.wikipedia.org/wiki/Ridge_regression) |
| **LASSO (L1)** | N/A | Penalize Σ|β|; feature selection via sparsity | [Wiki - LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) |
| **Elastic Net** | N/A | Combined L1 + L2 penalties; handles grouped variables | [Wiki - Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization) |
| **Tree-Based Methods** | N/A | Decision trees, random forests, gradient boosting for econometrics | [Wiki - Random Forest](https://en.wikipedia.org/wiki/Random_forest) |
| **Neural Networks in Econometrics** | N/A | Non-linear function approximation; deep learning applications | [Wiki - Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network) |
| **High-Dimensional Econometrics** | N/A | p > n problems; post-selection inference, double machine learning | [Belloni et al. (2014)](https://arxiv.org/abs/1201.0220) |

---

## XI. Spatial Econometrics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Spatial Autocorrelation** | N/A | Correlation across spatial units; Moran's I, Geary's C | [Wiki - Spatial Autocorrelation](https://en.wikipedia.org/wiki/Spatial_analysis#Spatial_autocorrelation) |
| **Spatial Lag Model** | N/A | Y depends on neighbors' Y; ρWY term | [Wiki - Spatial Econometrics](https://en.wikipedia.org/wiki/Spatial_econometrics) |
| **Spatial Error Model** | N/A | Errors spatially correlated; λWε term | [LeSage & Pace (2009)](https://www.taylorfrancis.com/books/mono/10.1201/9781420064254/introduction-spatial-econometrics-james-lesage-robert-pace) |
| **Weight Matrices** | N/A | Define spatial relationships; contiguity, distance-based, k-nearest | [Wiki - Spatial Weight Matrix](https://en.wikipedia.org/wiki/Spatial_analysis) |

---

## XII. Treatment Effects & Program Evaluation

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Average Treatment Effect (ATE)** | N/A | E[Y₁ - Y₀]; population average causal effect | [Wiki - ATE](https://en.wikipedia.org/wiki/Average_treatment_effect) |
| **Treatment on Treated (ATT)** | N/A | E[Y₁ - Y₀|D=1]; effect for those who received treatment | [Imbens & Rubin (2015)](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB) |
| **Local Average Treatment Effect** | N/A | LATE; effect for compliers in IV framework | [Wiki - LATE](https://en.wikipedia.org/wiki/Local_average_treatment_effect) |
| **Synthetic Control Methods** | N/A | Weighted combination of controls mimics treated unit pre-intervention | [Wiki - Synthetic Control](https://en.wikipedia.org/wiki/Synthetic_control_method) |
| **Matching Estimators** | N/A | Nearest neighbor, kernel matching; balance covariates | [Stuart (2010)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2943670/) |

---

## XIII. Nonparametric & Semiparametric Methods

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Kernel Regression** | N/A | Local polynomial smoothing; bandwidth selection critical | [Wiki - Kernel Regression](https://en.wikipedia.org/wiki/Kernel_regression) |
| **Splines** | N/A | Piecewise polynomials; regression splines, smoothing splines | [Wiki - Spline](https://en.wikipedia.org/wiki/Spline_(mathematics)) |
| **Partially Linear Models** | N/A | Y = Xβ + g(Z) + ε; parametric + nonparametric components | [Robinson (1988)](https://www.jstor.org/stable/1912705) |
| **Quantile Regression** | N/A | Estimate conditional quantiles; robust to outliers | [Wiki - Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression) |
| **Nonparametric Density Estimation** | N/A | Kernel density estimation; histogram smoothing | [Wiki - KDE](https://en.wikipedia.org/wiki/Kernel_density_estimation) |

---

## XIV. Forecasting

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Point Forecasts** | N/A | Conditional mean prediction; minimize MSE | [Hyndman & Athanasopoulos](https://otexts.com/fpp3/) |
| **Interval Forecasts** | N/A | Prediction intervals; quantify forecast uncertainty | [Hyndman & Athanasopoulos](https://otexts.com/fpp3/) |
| **Forecast Evaluation** | N/A | RMSE, MAE, MAPE; Diebold-Mariano test for comparison | [Wiki - Forecast Error](https://en.wikipedia.org/wiki/Forecast_error) |
| **Combining Forecasts** | N/A | Weighted average of multiple models; often outperforms individual | [Timmermann (2006)](https://doi.org/10.1016/S1574-0706(05)01019-3) |
| **Forecast Encompassing** | N/A | Test if one forecast contains all info of another | [Harvey et al. (1998)](https://doi.org/10.1016/S0169-2070(98)00015-9) |

---

## XV. Applied Microeconometrics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Labor Economics Applications** | N/A | Wage equations, returns to education, discrimination | [Wooldridge Ch. 7](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge) |
| **Health Economics Models** | N/A | Insurance effects, treatment decisions, cost-effectiveness | [Jones (2000)](https://doi.org/10.1016/S0167-6296(00)00049-4) |
| **Demand Estimation** | N/A | Consumer choice, price elasticity, discrete choice models | [Berry et al. (1995)](https://www.jstor.org/stable/2171802) |
| **Production Functions** | N/A | Cobb-Douglas, CES, TFP estimation; simultaneity issues | [Ackerberg et al. (2015)](https://doi.org/10.3982/ECTA13408) |

---

## XVI. Applied Macroeconometrics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Business Cycle Analysis** | N/A | HP filter, band-pass filters; dating turning points | [Wiki - Business Cycle](https://en.wikipedia.org/wiki/Business_cycle) |
| **Structural VAR** | N/A | Identify shocks via restrictions; Cholesky, long-run, sign restrictions | [Wiki - SVAR](https://en.wikipedia.org/wiki/Vector_autoregression#Structural_VAR) |
| **DSGE Models** | N/A | Dynamic Stochastic General Equilibrium; microfounded macro | [Wiki - DSGE](https://en.wikipedia.org/wiki/Dynamic_stochastic_general_equilibrium) |
| **Factor Models** | N/A | Extract common factors; diffusion indexes; large datasets | [Stock & Watson (2002)](https://www.nber.org/papers/w9042) |

---

## XVII. Financial Econometrics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Asset Return Properties** | N/A | Fat tails, volatility clustering, leverage effects | [Campbell et al. (1997)](https://press.princeton.edu/books/hardcover/9780691043012/the-econometrics-of-financial-markets) |
| **Market Efficiency Tests** | N/A | Weak, semi-strong, strong form; event studies | [Wiki - Efficient Market](https://en.wikipedia.org/wiki/Efficient-market_hypothesis) |
| **Portfolio Optimization** | N/A | Mean-variance; estimation error impact on weights | [Markowitz (1952)](https://doi.org/10.2307/2975974) |
| **Risk Management Models** | N/A | VaR, CVaR; backtesting risk models | [Jorion (2006)](https://www.mheducation.com/highered/product/value-risk-3rd-edition-jorion/M9780071464956.html) |
| **High-Frequency Econometrics** | N/A | Microstructure noise, realized volatility, tick time vs calendar time | [Aït-Sahalia & Jacod (2014)](https://press.princeton.edu/books/hardcover/9780691161433/high-frequency-financial-econometrics) |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **Wooldridge - Introductory Econometrics** | https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge | Standard graduate text; OLS, time series, panel data, IV |
| **Greene - Econometric Analysis** | https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899 | Comprehensive reference; estimation theory, limited dependent variables |
| **Wikipedia - Econometrics Portal** | https://en.wikipedia.org/wiki/Portal:Econometrics | 200+ topics organized by methodology |
| **Stock & Watson - Introduction to Econometrics** | https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000005522 | Undergraduate focus; applied emphasis |
| **Angrist & Pischke - Mostly Harmless** | https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics | Causal inference; IV, DiD, RDD applications |

---

## Quick Stats

- **Total Topics Documented**: 130+
- **Categories**: 17
- **Coverage**: Classical Regression → Time Series → Causal Inference → ML Integration
- **Applications**: Micro, Macro, Finance, Spatial, High-Dimensional

