# Machine Learning Models for Credit Risk

## 1. Concept Skeleton
**Definition:** Black-box algorithms (random forest, gradient boosting, neural networks) learning default patterns from high-dimensional data; can capture non-linearities and interactions  
**Purpose:** Improve predictive accuracy over traditional scorecard, handle feature engineering automatically, exploit large datasets  
**Prerequisites:** Supervised learning, feature engineering, cross-validation, model evaluation, computational resources

## 2. Comparative Framing
| Model | Interpretability | Data Needs | Feature Engineering | Regulatory Approval | Speed |
|-------|-----------------|-----------|-------------------|-------------------|-------|
| **Logistic Scorecard** | High (coefficients) | Moderate | Manual | Easy | Real-time |
| **Random Forest** | Medium (importance) | Large | Automatic | Growing | Fast |
| **Gradient Boosting** | Medium (importance) | Large | Automatic | Growing | Fast |
| **Neural Network** | Very Low (black box) | Very Large | Automatic | Difficult | Variable |
| **Ensemble** | Medium (weighted) | Large | Mixed | Moderate | Medium |

## 3. Examples + Counterexamples

**Simple Example:**  
Random forest with 500 trees on 200 features predicts 93% accuracy vs. scorecard's 85%. Deployed for real-time approvals

**Failure Case:**  
Model trained on 2015-2019 data, deployed in 2020 pandemic; loan-income relationship breaks (many unemployed). Model drift without retraining

**Edge Case:**  
Neural network achieves 95% accuracy but fails fairness audit (gender bias hidden in latent layers); unfolds to discover proxy variables

### 3B. Technical Counterexample: Model Overfitting and Out-of-Sample Degradation

**Common Misconception:** "My XGBoost model achieves 92% AUC on training data and 90% AUC on 20% holdout test set. The model is well-calibrated and ready for production. Out-of-sample degradation is minimal (2%), indicating good generalization."

**Why This Fails:** In-time sample testing uses recent historical data (train 2018-2019, test 2019-2020) with similar macro conditions. True out-of-sample test requires testing on completely different regime/population. 2% degradation in normal times masks 10-15% degradation in crisis conditions.

**Quantitative Example:**

**Model Development (Normal Times):**
- Training data: 2015-2019 (low unemployment 4.5-3.5%, moderate defaults 1.5-2%)
- Testing data: 2019-2020 (unemployment stable 3.5-4%, defaults 1.8%)
- XGBoost model with 5,000 trees, depth=6, 150 features
- Features: Income, debt-to-income (DTI), loan-to-value (LTV), credit score, etc.
- Train AUC: 0.920
- Test AUC: 0.905 (2.2% degradation)
- Estimated default precision: 88% (of borrowers scored as high-risk, 88% actually default)

**Production Deployment 2020 (Regime Shift):**
- COVID pandemic shock: Unemployment spikes from 4% to 14% (March-April 2020)
- Income volatility increases dramatically (income drops for service/hospitality workers)
- Loan-to-income relationship breaks: unemployed borrowers previously had good income history
- Model still produces AUC ~0.78 (13% degradation from test), but meaningful:
  - Previous precision 88%, now only 72% (default prediction much noisier)
  - False positive rate: borrowers marked "high-risk" but don't default; credit denied unnecessarily
  - False negative rate: borrowers marked safe but default at 5%+ rates

**Why Model Failed in Crisis:**
1. **Feature Dependency:** Income feature trained on stable economy. In crisis, income becomes highly volatile; model can't capture rapid decline.
2. **Non-linear Relationships:** DTI threshold of 40% seemed safe historically (low defaults). In recession, defaults occur at 35-38% DTI due to income uncertainty.
3. **Correlation Breakdown:** Feature independence assumed (training set). In crisis, job loss (income: -100%) correlates with sector/geography; model assumes independence violated.
4. **Macro Drift:** Model trained in macro environment (low unemployment, stable rates, 2% default base rate). Crisis: 12% unemployment, volatile rates, 8% default base rate. High-dimensional models sensitive to base rate shifts.

**Evidence from 2008 Financial Crisis:**
- Banks deployed models trained on 2003-2007 data (pre-crisis)
- 2008-2009 crisis: AUC degraded from ~0.88 to ~0.72
- Defaults among borrowers scored low-risk (bottom 20%) jumped from 1% to 8%
- Model rejections in crisis: safe borrowers rejected (false positives); uncreditworthy approved (false negatives)

**Comparison: Alternative Approaches**
- **Logistic Scorecard:** Simple coefficients (Income: +0.10, DTI: -0.08, etc.). In crisis? Relationships shift but coefficients stable; easier to retrain monthly.
  - Pre-crisis: AUC 0.85, Crisis AUC: 0.78 (7% degradation, vs ML 13%)
- **Tree Ensemble (simpler):** Shallow trees (depth 3), fewer features (20). Less prone to overfitting.
  - Pre-crisis: AUC 0.87, Crisis AUC: 0.81 (6% degradation)
- **Ensemble (multiple models):** Combine logistic, tree, and ML model with weights. Robust to individual model drift.
  - Pre-crisis: AUC 0.88, Crisis AUC: 0.83 (5% degradation)

**Regulatory Response:** 
- Federal Reserve SR 11-7 requires model validation on out-of-sample data representative of stress scenarios
- Supervisory stress testing explicitly tests models on crisis conditions (unemployment 10%+, unemployment 7% drop, etc.)
- Banks must document model performance degradation in stress scenarios
- Capital requirements adjusted if model shows >10% degradation in stress

**Correct Approach:**
1. **Validation on Stressed Data:** Test on 2007-2009 financial crisis period; evaluate degradation under high-unemployment regime
2. **Sensitivity Analysis:** Show how model predictions change with income -20%, unemployment +5pp, rates +200bps, etc.
3. **Backtesting:** Compare monthly model predictions to realized defaults every quarter; detect drift early
4. **Retraining Triggers:** Trigger model retraining if AUC drops >5%, PSI (population stability) > 0.25, or documented macro change
5. **Ensemble Methods:** Combine multiple models (logistic, tree, neural) to reduce single-model risk
6. **Feature Stability:** Monitor feature importance over time; if income importance drops 50% suddenly, investigate distribution shift

## 4. Layer Breakdown
```
Machine Learning Framework for Credit Risk:
â”œâ”€ Data Preparation:
â”‚   â”œâ”€ Features: Raw + engineered variables
â”‚   â”œâ”€ Targets: Binary (default/non-default) or multi-class
â”‚   â”œâ”€ Imbalance handling: SMOTE, class weights, or stratified sampling
â”‚   â””â”€ Scaling: Normalization for distance-based algorithms
â”œâ”€ Model Classes:
â”‚   â”œâ”€ Tree-based:
â”‚   â”‚   â”œâ”€ Random Forest: Ensemble of bootstrap samples
â”‚   â”‚   â”œâ”€ Gradient Boosting: Sequential error correction
â”‚   â”‚   â”œâ”€ XGBoost: Regularized boosting
â”‚   â”‚   â””â”€ LightGBM: Fast large-scale boosting
â”‚   â”œâ”€ Linear/Logistic:
â”‚   â”‚   â”œâ”€ Logistic Regression: Baseline interpretable
â”‚   â”‚   â”œâ”€ Ridge/Lasso: Regularized regression
â”‚   â”‚   â””â”€ Elastic Net: Combined L1/L2 penalties
â”‚   â”œâ”€ Distance-based:
â”‚   â”‚   â”œâ”€ KNN: k-nearest neighbors
â”‚   â”‚   â””â”€ SVM: Support vector machines
â”‚   â”œâ”€ Neural Networks:
â”‚   â”‚   â”œâ”€ Feedforward: Dense layers
â”‚   â”‚   â”œâ”€ CNN: Convolutional (image data)
â”‚   â”‚   â””â”€ RNN: Sequential (time series)
â”‚   â””â”€ Ensemble: Weighted combination of models
â”œâ”€ Hyperparameter Tuning:
â”‚   â”œâ”€ Grid search: Exhaustive parameter sweep
â”‚   â”œâ”€ Random search: Probabilistic sampling
â”‚   â”œâ”€ Bayesian optimization: Smart parameter selection
â”‚   â””â”€ Early stopping: Prevent overfitting during training
â”œâ”€ Validation Strategy:
â”‚   â”œâ”€ Train/val/test split: Temporal or random
â”‚   â”œâ”€ Cross-validation: K-fold for stable estimates
â”‚   â”œâ”€ Walk-forward: Time-series appropriate
â”‚   â””â”€ Stratified: Preserve class imbalance in folds
â”œâ”€ Evaluation Metrics:
â”‚   â”œâ”€ Discrimination: AUC-ROC, Gini, K-S statistic
â”‚   â”œâ”€ Calibration: Actual vs predicted default rate
â”‚   â”œâ”€ Stability: Robustness across time periods
â”‚   â”œâ”€ Fairness: Equal treatment across demographics
â”‚   â””â”€ Explainability: SHAP values, feature importance
â”œâ”€ Monitoring in Production:
â”‚   â”œâ”€ Model drift: Shift in predicted vs actual PD
â”‚   â”œâ”€ Population drift: Input distribution changes
â”‚   â”œâ”€ Performance degradation: Declining AUC
â”‚   â””â”€ Retraining triggers: Monthly/quarterly refresh
â””â”€ Challenges:
    â”œâ”€ Black-box risk: Difficulty explaining decisions
    â”œâ”€ Data quality: Garbage in â†’ garbage out
    â”œâ”€ Overfitting: High train accuracy, low test
    â”œâ”€ Regulatory concerns: Fair lending, bias
    â””â”€ Concept drift: Relationships change over time
```

## 5. Challenge Round
When are ML models problematic?
- **Black box risk**: Can't explain why loan denied; regulatory violation
- **Overfitting**: High test AUC but fails on new population; poor generalization
- **Data quality**: Models amplify underlying data biases (redlining embedded)
- **Concept drift**: Relationships change (2019 scorecard fails in 2020 pandemic)
- **Regulatory resistance**: Banks slow to adopt; require explainability + governance
- **Fairness issues**: Equal treatment across demographics; model may violate fair lending

## 6. Key References
- [Scikit-Learn Credit Scoring Guide](https://scikit-learn.org/stable/modules/ensemble.html) - Detailed implementation of Random Forest, Gradient Boosting, feature importance; cross-validation strategies for credit data; imbalanced class handling.

- [XGBoost Library Documentation](https://xgboost.readthedocs.io/en/stable/) - Production-grade gradient boosting implementation; regularization (L1/L2), early stopping, custom loss functions for imbalanced credit data; benchmarking vs logistic regression.

- Mitchell, T. M. (1997). "Machine Learning." McGraw-Hill. Foundational ML textbook; chapters on overfitting, cross-validation, feature selection; practical guidance on avoiding model drift in production.

- [ML Fairness in Finance - Arxiv 1908.04913](https://arxiv.org/abs/1908.04913) - Comprehensive treatment of fairness in credit ML models; bias detection methods (SHAP, permutation importance); regulatory fair lending constraints.

- Friedman, J. H., Hastie, T., & Tibshirani, R. (2009). "The Elements of Statistical Learning" (2nd ed.). Springer. Advanced ML theory; chapters on boosting, regularization, and model stability; includes financial prediction applications.

- Khandani, A. E., Kim, A. J., & Andrew, W. (2010). "The Adaptive Use of Machine Learning for Prediction of Loan Default Risk." SSRN. Empirical study comparing logistic regression, neural networks, SVM on 1M loan dataset; documents model drift post-2008; proposes adaptive retraining frameworks.

---
**Status:** Modern alternative to traditional scorecards | **Complements:** Explainability, fairness testing, model governance
