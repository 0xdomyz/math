# Discounted Cash Flow (DCF) Valuation

## 1. Concept Skeleton
**Definition:** Intrinsic value estimation by discounting expected future free cash flows to present value using weighted average cost of capital (WACC)  
**Purpose:** Determine fair value of equity independent of market price; assess investment opportunities; identify over/undervalued securities  
**Prerequisites:** Time value of money, financial statements (income statement, balance sheet, cash flow statement), cost of capital (CAPM, WACC), terminal value concepts, growth rate estimation

## 2. Comparative Framing
| Valuation Method | DCF | Comparable Companies | Precedent Transactions | DDM (Dividend Discount) |
|------------------|-----|---------------------|----------------------|------------------------|
| **Basis** | Future free cash flows | Trading multiples (P/E, EV/EBITDA) | Acquisition multiples | Future dividends |
| **Independence** | Market-independent (intrinsic) | Market-dependent | Transaction-dependent | Dividend policy dependent |
| **Assumptions** | Growth, WACC, terminal value | Comparability assumption | Control premium included | Payout ratio stability |
| **Complexity** | High (many inputs) | Low-moderate | Moderate | Low-moderate |
| **Best For** | Stable, predictable cash flows | Public companies with peers | M&A scenarios | Dividend-paying mature firms |
| **Sensitivity** | High to discount rate, growth | Less sensitive to assumptions | Reflects deal environment | Ignores retained earnings value |

| DCF Model Type | FCFF (Firm) | FCFE (Equity) | Adjusted Present Value (APV) |
|----------------|-------------|---------------|------------------------------|
| **Cash Flow** | Free Cash Flow to Firm | Free Cash Flow to Equity | Unlevered FCF + Tax shields |
| **Discount Rate** | WACC (blended) | Cost of Equity (Ke) | Cost of unlevered equity |
| **Value Output** | Enterprise Value | Equity Value directly | Enterprise Value + financing |
| **Leverage** | Fixed capital structure | Variable leverage OK | Changing leverage scenarios |
| **Best For** | Standard valuation | Leverage changes expected | LBOs, restructuring |

## 3. Examples + Counterexamples

**Simple Example:**  
Tech company: FCFF = $100M (year 1), grows 10% for 5 years, then 3% perpetual. WACC = 12%.  
PV(Years 1-5) = $100M/(1.12) + $110M/(1.12)² + ... ≈ $361M  
Terminal Value = $146M/(0.12-0.03) = $1,622M → PV = $920M  
Enterprise Value = $361M + $920M = $1,281M  
Less: Net Debt $200M → Equity Value = $1,081M

**Perfect Fit:**  
Mature utility company with stable regulated cash flows, predictable CapEx, 2% growth matching GDP. DCF provides reliable baseline vs volatile market pricing during interest rate cycles.

**Failure Case:**  
Early-stage biotech with negative cash flows, binary FDA approval outcome. DCF assumes smooth growth → misleading. Use scenario analysis or real options valuation instead.

**Edge Case:**  
Cyclical manufacturing during recession: Current FCF negative, normalized average = $50M. Use mid-cycle normalized cash flow, not current depressed level. Otherwise DCF undervalues significantly.

**Common Mistake:**  
Using nominal growth rate with real discount rate (or vice versa). Must match: Real rate with real growth, or nominal with nominal. Mismatch causes significant valuation error.

**Counterexample:**  
High-growth startup: Projecting 50% growth perpetually is unrealistic (implies eventual GDP dominance). Must taper to sustainable long-term rate (2-5%) in terminal value calculation.

## 4. Layer Breakdown
```
DCF Valuation Process:

├─ Free Cash Flow Calculation:
│   ├─ FCFF (Free Cash Flow to Firm):
│   │   │ Start: EBIT (Earnings Before Interest & Tax)
│   │   │ Adjust: × (1 - Tax Rate) = NOPAT (Net Operating Profit After Tax)
│   │   │ Add back: Depreciation & Amortization (non-cash expense)
│   │   │ Subtract: Capital Expenditures (CapEx) = cash spent on PP&E
│   │   │ Subtract: Increase in Net Working Capital (NWC)
│   │   │         NWC = (Accounts Receivable + Inventory) - Accounts Payable
│   │   │         Growing business needs more working capital (cash tied up)
│   │   └─ Formula: FCFF = NOPAT + D&A - CapEx - ΔNWC
│   │       Alternative: FCFF = CFO (Cash from Operations) + Interest(1-T) - CapEx
│   ├─ FCFE (Free Cash Flow to Equity):
│   │   │ Start: Net Income (after interest, taxes)
│   │   │ Add back: Depreciation & Amortization
│   │   │ Subtract: CapEx
│   │   │ Subtract: ΔNWC
│   │   │ Add: Net Borrowing (new debt - debt repayment)
│   │   └─ Formula: FCFE = Net Income + D&A - CapEx - ΔNWC + Net Borrowing
│   │       Direct: FCFE = FCFF - Interest(1-T) + Net Borrowing
│   └─ Projection Period:
│       ├─ Explicit Forecast (5-10 years): Detailed projections year-by-year
│       │   Revenue growth assumptions (market size, share, pricing)
│       │   Operating margins (EBITDA margin, EBIT margin trends)
│       │   CapEx intensity (% of revenue, maintenance vs growth)
│       │   Working capital needs (cash conversion cycle)
│       ├─ Growth Drivers:
│       │   Industry growth rate (GDP correlation, TAM expansion)
│       │   Market share gains/losses
│       │   Pricing power (inflation pass-through)
│       │   Operating leverage (margins improve with scale)
│       └─ Reasonableness Checks:
│           Implied market share (can't exceed 100% of TAM)
│           ROIC (Return on Invested Capital) sustainability
│           Competitive dynamics (high margins attract entry)
│
├─ Discount Rate (Cost of Capital):
│   ├─ WACC (Weighted Average Cost of Capital):
│   │   │ Formula: WACC = (E/V)×Ke + (D/V)×Kd×(1-T)
│   │   │ E = Market value of equity (shares outstanding × price)
│   │   │ D = Market value of debt (or book value if illiquid)
│   │   │ V = E + D (total enterprise value)
│   │   │ Ke = Cost of equity (expected return to shareholders)
│   │   │ Kd = Cost of debt (yield to maturity on bonds, or interest rate)
│   │   │ T = Corporate tax rate (debt creates tax shield: interest deductible)
│   │   └─ Intuition: Blended rate reflecting both debt (cheaper, tax-advantaged)
│   │       and equity (more expensive, residual claimant)
│   ├─ Cost of Equity (Ke):
│   │   ├─ CAPM (Capital Asset Pricing Model):
│   │   │   │ Formula: Ke = Rf + β × (Rm - Rf)
│   │   │   │ Rf = Risk-free rate (10-year Treasury yield ≈ 2-5%)
│   │   │   │ β = Beta (systematic risk, regression of stock vs market)
│   │   │   │     β > 1: More volatile than market (tech, cyclicals)
│   │   │   │     β < 1: Less volatile (utilities, consumer staples)
│   │   │   │     β = 0: Risk-free asset
│   │   │   │ Rm = Expected market return (historical S&P 500 ≈ 10-12%)
│   │   │   │ (Rm - Rf) = Equity Risk Premium (ERP) ≈ 5-7%
│   │   │   └─ Example: Rf=4%, β=1.2, ERP=6% → Ke = 4% + 1.2×6% = 11.2%
│   │   ├─ Build-Up Method (for private companies):
│   │   │   Ke = Rf + Equity Risk Premium + Size Premium + Company-Specific Risk
│   │   │   Size premium: Small companies riskier (illiquidity, access to capital)
│   │   │   Company-specific: Key person risk, concentration, financial leverage
│   │   └─ Fama-French Three-Factor Model (alternative):
│   │       Ke = Rf + β_market×MRP + β_size×SMB + β_value×HML
│   │       SMB = Small Minus Big (size factor)
│   │       HML = High Minus Low (value factor)
│   ├─ Cost of Debt (Kd):
│   │   ├─ Direct Observation:
│   │   │   Yield to maturity on existing bonds
│   │   │   If no bonds: Credit rating → typical spread over Treasury
│   │   │       AAA: +50 bps, A: +150 bps, BBB: +250 bps, BB: +400 bps
│   │   ├─ Synthetic Rating:
│   │   │   Interest Coverage Ratio = EBIT / Interest Expense
│   │   │   >8.5x → AAA, 6-8.5x → AA, 4.5-6x → A, 3-4.5x → BBB, <3x → Junk
│   │   └─ After-Tax: Kd×(1-T) reflects tax deductibility of interest
│   └─ Adjustments:
│       ├─ Country Risk Premium (for emerging markets):
│       │   Add spread reflecting sovereign default risk
│       │   Example: Developed Ke=11%, EM add 3% → Ke=14%
│       ├─ Unlevering/Relevering Beta:
│       │   β_unlevered = β_levered / [1 + (1-T) × (D/E)]
│       │   Use to compare companies with different leverage
│       │   Relever for target company's capital structure
│       └─ Iterative WACC:
│           WACC calculation uses V (enterprise value)
│           But V is output of DCF (circular)
│           Solution: Iterate until convergence, or assume target D/E ratio
│
├─ Terminal Value (Continuing Value):
│   │ Represents value beyond explicit forecast period
│   │ Typically 60-80% of total enterprise value
│   ├─ Perpetuity Growth Method (Gordon Growth Model):
│   │   │ Formula: TV = FCF_T+1 / (WACC - g)
│   │   │ FCF_T+1 = Free cash flow in first year after forecast (Year T+1)
│   │   │ g = Perpetual growth rate (typically 2-3%, ≈ GDP growth)
│   │   │ WACC = Discount rate
│   │   │ Assumption: Company grows at constant rate forever
│   │   ├─ Constraints:
│   │   │   g < WACC (otherwise TV undefined/infinite)
│   │   │   g ≤ GDP growth long-term (can't outgrow economy forever)
│   │   │   g < Risk-free rate often (conservatism)
│   │   ├─ Reasonableness:
│   │   │   If g = 3%, implies doubling every 24 years perpetually
│   │   │   High-growth companies must taper to g in explicit period
│   │   └─ Sensitivity:
│   │       TV highly sensitive to g
│   │       g=2% vs g=3% can change valuation by 20-40%
│   ├─ Exit Multiple Method:
│   │   │ Formula: TV = EBITDA_T × Exit Multiple
│   │   │ Exit Multiple = EV/EBITDA from comparable companies
│   │   │ Typical: 8-12x for mature industrials, 15-25x for growth tech
│   │   ├─ Advantages:
│   │   │   Market-based (reflects investor sentiment)
│   │   │   Avoids perpetuity growth assumption
│   │   ├─ Disadvantages:
│   │   │   Introduces market dependency (defeats DCF purpose)
│   │   │   Multiple may be artificially high/low at valuation date
│   │   └─ Cross-Check:
│   │       Compare to perpetuity growth method
│   │       If vastly different → revisit assumptions
│   └─ Present Value of Terminal Value:
│       TV occurs at end of Year T
│       PV(TV) = TV / (1 + WACC)^T
│       Discount back T years to present
│
├─ Enterprise Value (EV) to Equity Value Bridge:
│   │ DCF produces Enterprise Value (value to all capital providers)
│   │ Must adjust to get Equity Value (value to shareholders only)
│   ├─ Formula:
│   │   Equity Value = Enterprise Value
│   │                  - Net Debt
│   │                  + Non-Operating Assets
│   │                  - Minority Interests
│   │                  + Associate/JV Investments
│   ├─ Net Debt:
│   │   = Total Debt (short-term + long-term)
│   │     - Cash and Cash Equivalents
│   │     - Marketable Securities
│   │   Intuition: Equity holders inherit debt obligation, benefit from excess cash
│   │   If Net Debt negative (more cash than debt) → adds to equity value
│   ├─ Non-Operating Assets:
│   │   Investments, real estate held for sale, discontinued operations
│   │   Not included in operating FCF projections
│   │   Add back at fair value (or carrying value if reasonable)
│   ├─ Minority Interests:
│   │   Subsidiaries not 100% owned
│   │   Consolidated financials include 100% of cash flows
│   │   But equity holders only own parent's share
│   │   Subtract minority portion
│   └─ Associates/JVs:
│       20-50% ownership: Equity method accounting
│       Only proportional earnings included in income
│       But owns valuable stake → add market value
│
├─ Sensitivity Analysis:
│   ├─ One-Way Sensitivity (Tornado Diagram):
│   │   Vary one input at a time (e.g., WACC ±1%, growth ±0.5%)
│   │   Calculate resulting equity value range
│   │   Rank variables by impact magnitude
│   │   Typical: Terminal growth and WACC most impactful
│   ├─ Two-Way Sensitivity Table:
│   │   Matrix varying two inputs simultaneously
│   │   Example: WACC (rows) vs Terminal Growth (columns)
│   │   Creates grid of equity values → assess plausible ranges
│   └─ Scenario Analysis:
│       Define 3-5 scenarios: Base, Bull, Bear, Stress
│       Bull: High growth, margin expansion, low WACC
│       Bear: Low growth, margin compression, high WACC
│       Stress: Recession scenario
│       Probability-weight scenarios for expected value
│
└─ Sanity Checks and Validation:
    ├─ Implied Multiples:
    │   Calculate EV/EBITDA, P/E from DCF output
    │   Compare to comparable companies
    │   If DCF implies EV/EBITDA = 30x while peers at 12x → revisit assumptions
    ├─ ROIC vs WACC:
    │   Return on Invested Capital = NOPAT / Invested Capital
    │   If ROIC > WACC: Value-creating (growth adds value)
    │   If ROIC < WACC: Value-destroying (shrink or fix operations)
    │   Terminal value assumes ROIC ≈ WACC long-term (competitive equilibrium)
    ├─ Terminal Value % of Total:
    │   Healthy: 50-75% of enterprise value
    │   Too high (>80%): Explicit period too short or unrealistic
    │   Too low (<40%): Terminal growth too conservative or WACC too high
    └─ Market Comparison:
        Compare DCF equity value per share to current market price
        >20% premium: Potential overvaluation or overly optimistic assumptions
        >20% discount: Potential undervaluation or market sees risks
```

**Interaction:** Project future cash flows (revenue → EBIT → FCFF/FCFE) → Discount using WACC/Ke → Calculate terminal value → Present value all cash flows → Sum to enterprise value → Adjust for net debt and non-operating items → Equity value per share → Compare to market price → Sensitivity analysis → Iterate assumptions

## 5. Mini-Project
Comprehensive DCF valuation with sensitivity analysis and visualizations:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DISCOUNTED CASH FLOW (DCF) VALUATION MODEL")
print("="*80)

class DCFModel:
    """Full DCF valuation with FCFF, terminal value, and sensitivity analysis"""
    
    def __init__(self, company_name="Example Corp"):
        self.company_name = company_name
        self.financials = {}
        self.projections = pd.DataFrame()
        self.wacc = None
        self.terminal_value = None
        self.enterprise_value = None
        self.equity_value = None
        
    def set_base_year_financials(self, revenue, ebit_margin, tax_rate, 
                                 nwc_pct_revenue, capex_pct_revenue, 
                                 da_pct_revenue):
        """Set base year financial metrics"""
        self.financials = {
            'revenue': revenue,
            'ebit_margin': ebit_margin,
            'tax_rate': tax_rate,
            'nwc_pct_revenue': nwc_pct_revenue,
            'capex_pct_revenue': capex_pct_revenue,
            'da_pct_revenue': da_pct_revenue
        }
        
    def project_cash_flows(self, years=5, revenue_growth_rates=None,
                          terminal_growth_rate=0.025):
        """Project free cash flows for explicit forecast period"""
        if revenue_growth_rates is None:
            # Default: Declining growth over forecast period
            revenue_growth_rates = [0.10, 0.08, 0.06, 0.05, 0.04]
        
        # Initialize projection DataFrame
        periods = list(range(1, years + 1))
        self.projections = pd.DataFrame(index=periods)
        
        # Revenue projections
        revenue = self.financials['revenue']
        revenues = []
        for g in revenue_growth_rates:
            revenue = revenue * (1 + g)
            revenues.append(revenue)
        self.projections['Revenue'] = revenues
        self.projections['Revenue_Growth'] = revenue_growth_rates
        
        # EBIT and NOPAT
        self.projections['EBIT'] = (self.projections['Revenue'] * 
                                     self.financials['ebit_margin'])
        self.projections['Taxes'] = (self.projections['EBIT'] * 
                                      self.financials['tax_rate'])
        self.projections['NOPAT'] = (self.projections['EBIT'] - 
                                      self.projections['Taxes'])
        
        # Add back D&A (non-cash)
        self.projections['D&A'] = (self.projections['Revenue'] * 
                                    self.financials['da_pct_revenue'])
        
        # CapEx (cash outflow for growth/maintenance)
        self.projections['CapEx'] = (self.projections['Revenue'] * 
                                      self.financials['capex_pct_revenue'])
        
        # Change in Net Working Capital
        self.projections['NWC'] = (self.projections['Revenue'] * 
                                    self.financials['nwc_pct_revenue'])
        self.projections['Change_NWC'] = self.projections['NWC'].diff().fillna(
            self.projections['NWC'].iloc[0] * 0.5  # Assume initial NWC increase
        )
        
        # Free Cash Flow to Firm (FCFF)
        self.projections['FCFF'] = (
            self.projections['NOPAT'] +
            self.projections['D&A'] -
            self.projections['CapEx'] -
            self.projections['Change_NWC']
        )
        
        self.terminal_growth_rate = terminal_growth_rate
        
        return self.projections
    
    def calculate_wacc(self, risk_free_rate, equity_risk_premium, beta,
                      market_value_equity, market_value_debt, cost_of_debt,
                      tax_rate):
        """Calculate Weighted Average Cost of Capital"""
        # Cost of Equity (CAPM)
        cost_of_equity = risk_free_rate + beta * equity_risk_premium
        
        # Total value
        total_value = market_value_equity + market_value_debt
        
        # Weights
        weight_equity = market_value_equity / total_value
        weight_debt = market_value_debt / total_value
        
        # WACC
        self.wacc = (weight_equity * cost_of_equity + 
                    weight_debt * cost_of_debt * (1 - tax_rate))
        
        self.wacc_components = {
            'Cost_of_Equity': cost_of_equity,
            'Cost_of_Debt_After_Tax': cost_of_debt * (1 - tax_rate),
            'Weight_Equity': weight_equity,
            'Weight_Debt': weight_debt,
            'WACC': self.wacc
        }
        
        return self.wacc
    
    def calculate_terminal_value(self, method='perpetuity'):
        """Calculate terminal value using perpetuity growth or exit multiple"""
        if self.projections.empty:
            raise ValueError("Must project cash flows first")
        
        last_fcff = self.projections['FCFF'].iloc[-1]
        
        if method == 'perpetuity':
            # Terminal FCF (grow last year's FCF by terminal growth rate)
            terminal_fcf = last_fcff * (1 + self.terminal_growth_rate)
            
            # Gordon Growth Model: TV = FCF / (WACC - g)
            if self.wacc <= self.terminal_growth_rate:
                raise ValueError(f"WACC ({self.wacc:.2%}) must exceed terminal "
                               f"growth ({self.terminal_growth_rate:.2%})")
            
            self.terminal_value = terminal_fcf / (self.wacc - self.terminal_growth_rate)
            
        elif method == 'exit_multiple':
            # Use EV/EBITDA multiple (assume 10x for example)
            exit_multiple = 10.0
            last_ebitda = self.projections['EBIT'].iloc[-1] + self.projections['D&A'].iloc[-1]
            self.terminal_value = last_ebitda * exit_multiple
        
        return self.terminal_value
    
    def calculate_enterprise_value(self):
        """Calculate present value of projected FCF + terminal value"""
        if self.wacc is None:
            raise ValueError("Must calculate WACC first")
        if self.terminal_value is None:
            raise ValueError("Must calculate terminal value first")
        
        # Present value of explicit forecast period
        discount_factors = [(1 + self.wacc) ** t for t in range(1, len(self.projections) + 1)]
        self.projections['Discount_Factor'] = discount_factors
        self.projections['PV_FCFF'] = (self.projections['FCFF'] / 
                                        self.projections['Discount_Factor'])
        
        pv_forecast_period = self.projections['PV_FCFF'].sum()
        
        # Present value of terminal value
        terminal_discount_factor = (1 + self.wacc) ** len(self.projections)
        pv_terminal_value = self.terminal_value / terminal_discount_factor
        
        # Enterprise Value
        self.enterprise_value = pv_forecast_period + pv_terminal_value
        
        self.valuation_summary = {
            'PV_Forecast_Period': pv_forecast_period,
            'PV_Terminal_Value': pv_terminal_value,
            'Terminal_Value_Percent': pv_terminal_value / self.enterprise_value * 100,
            'Enterprise_Value': self.enterprise_value
        }
        
        return self.enterprise_value
    
    def calculate_equity_value(self, total_debt, cash, non_operating_assets=0,
                              minority_interest=0):
        """Bridge from enterprise value to equity value"""
        net_debt = total_debt - cash
        
        self.equity_value = (self.enterprise_value - 
                            net_debt + 
                            non_operating_assets - 
                            minority_interest)
        
        self.bridge = {
            'Enterprise_Value': self.enterprise_value,
            'Less_Total_Debt': -total_debt,
            'Add_Cash': cash,
            'Add_Non_Operating_Assets': non_operating_assets,
            'Less_Minority_Interest': -minority_interest,
            'Equity_Value': self.equity_value
        }
        
        return self.equity_value
    
    def value_per_share(self, shares_outstanding):
        """Calculate equity value per share"""
        if self.equity_value is None:
            raise ValueError("Must calculate equity value first")
        
        return self.equity_value / shares_outstanding
    
    def sensitivity_analysis_1d(self, parameter, values, shares_outstanding):
        """One-way sensitivity analysis"""
        base_equity_value = self.equity_value
        results = []
        
        for val in values:
            # Create temporary copy to avoid modifying base case
            temp_model = DCFModel(self.company_name)
            temp_model.financials = self.financials.copy()
            temp_model.projections = self.projections.copy()
            temp_model.wacc = self.wacc
            temp_model.terminal_growth_rate = self.terminal_growth_rate
            
            if parameter == 'wacc':
                temp_model.wacc = val
            elif parameter == 'terminal_growth':
                temp_model.terminal_growth_rate = val
            elif parameter == 'revenue_growth':
                # Adjust all revenue growth rates proportionally
                growth_adjustment = val
                temp_model.projections['Revenue_Growth'] *= growth_adjustment
            
            # Recalculate if needed
            if parameter in ['wacc', 'terminal_growth']:
                temp_model.terminal_value = temp_model.calculate_terminal_value('perpetuity')
                temp_model.enterprise_value = temp_model.calculate_enterprise_value()
                temp_equity = temp_model.equity_value = (
                    temp_model.enterprise_value - 
                    (self.bridge['Less_Total_Debt'] * -1 - self.bridge['Add_Cash'])
                )
            else:
                temp_equity = base_equity_value
            
            value_per_share = temp_equity / shares_outstanding
            results.append({
                'Parameter_Value': val,
                'Equity_Value': temp_equity,
                'Value_Per_Share': value_per_share
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis_2d(self, param1_name, param1_values,
                               param2_name, param2_values, shares_outstanding):
        """Two-way sensitivity table"""
        results = np.zeros((len(param1_values), len(param2_values)))
        
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                temp_model = DCFModel(self.company_name)
                temp_model.financials = self.financials.copy()
                temp_model.projections = self.projections.copy()
                temp_model.wacc = self.wacc
                temp_model.terminal_growth_rate = self.terminal_growth_rate
                
                if param1_name == 'wacc':
                    temp_model.wacc = val1
                elif param1_name == 'terminal_growth':
                    temp_model.terminal_growth_rate = val1
                
                if param2_name == 'wacc':
                    temp_model.wacc = val2
                elif param2_name == 'terminal_growth':
                    temp_model.terminal_growth_rate = val2
                
                temp_model.terminal_value = temp_model.calculate_terminal_value('perpetuity')
                temp_model.enterprise_value = temp_model.calculate_enterprise_value()
                temp_equity = (temp_model.enterprise_value - 
                              (self.bridge['Less_Total_Debt'] * -1 - self.bridge['Add_Cash']))
                
                results[i, j] = temp_equity / shares_outstanding
        
        return pd.DataFrame(results, 
                          index=[f"{val1:.1%}" for val1 in param1_values],
                          columns=[f"{val2:.1%}" for val2 in param2_values])

# Example Company Valuation
print("\n" + "="*80)
print("BASE CASE SCENARIO")
print("="*80)

dcf = DCFModel("TechGrowth Inc.")

# Base year financials (all in millions except percentages)
dcf.set_base_year_financials(
    revenue=1000,           # $1,000M revenue
    ebit_margin=0.20,       # 20% EBIT margin
    tax_rate=0.25,          # 25% tax rate
    nwc_pct_revenue=0.15,   # 15% of revenue tied up in working capital
    capex_pct_revenue=0.08, # 8% of revenue for CapEx
    da_pct_revenue=0.05     # 5% D&A
)

# Project 5 years with declining growth
revenue_growth = [0.12, 0.10, 0.08, 0.06, 0.05]  # Tapering from 12% to 5%
projections = dcf.project_cash_flows(
    years=5, 
    revenue_growth_rates=revenue_growth,
    terminal_growth_rate=0.025  # 2.5% perpetual growth
)

print("\nProjected Cash Flows ($ millions):")
print(projections.round(1).to_string())

# Calculate WACC
risk_free = 0.04      # 4% Treasury yield
erp = 0.06            # 6% equity risk premium
beta = 1.3            # Tech company, volatile
mv_equity = 5000      # $5B market cap
mv_debt = 1000        # $1B debt
cost_debt = 0.05      # 5% cost of debt

wacc = dcf.calculate_wacc(
    risk_free_rate=risk_free,
    equity_risk_premium=erp,
    beta=beta,
    market_value_equity=mv_equity,
    market_value_debt=mv_debt,
    cost_of_debt=cost_debt,
    tax_rate=0.25
)

print(f"\n" + "="*80)
print("COST OF CAPITAL CALCULATION")
print("="*80)
for key, val in dcf.wacc_components.items():
    if 'Weight' in key or 'Cost' in key or key == 'WACC':
        print(f"{key}: {val:.2%}")

# Terminal Value
tv = dcf.calculate_terminal_value(method='perpetuity')
print(f"\nTerminal Value (Perpetuity Growth Method): ${tv:,.1f}M")
print(f"Terminal Growth Rate: {dcf.terminal_growth_rate:.2%}")

# Enterprise Value
ev = dcf.calculate_enterprise_value()
print(f"\n" + "="*80)
print("ENTERPRISE VALUE")
print("="*80)
for key, val in dcf.valuation_summary.items():
    if 'Percent' in key:
        print(f"{key}: {val:.1f}%")
    else:
        print(f"{key}: ${val:,.1f}M")

# Equity Value
total_debt = 1000     # $1B debt
cash = 200            # $200M cash
equity_val = dcf.calculate_equity_value(
    total_debt=total_debt,
    cash=cash,
    non_operating_assets=50,  # $50M investments
    minority_interest=0
)

print(f"\n" + "="*80)
print("EQUITY VALUE BRIDGE")
print("="*80)
for key, val in dcf.bridge.items():
    print(f"{key}: ${val:,.1f}M")

shares_outstanding = 100  # 100M shares
value_per_share = dcf.value_per_share(shares_outstanding)
print(f"\nShares Outstanding: {shares_outstanding:.0f}M")
print(f"Implied Value Per Share: ${value_per_share:.2f}")

# Market comparison
current_market_price = 45.00  # Assume trading at $45
upside = (value_per_share / current_market_price - 1) * 100
print(f"\nCurrent Market Price: ${current_market_price:.2f}")
print(f"Implied Upside/(Downside): {upside:+.1f}%")

# Implied multiples sanity check
last_ebitda = projections['EBIT'].iloc[-1] + projections['D&A'].iloc[-1]
implied_ev_ebitda = ev / last_ebitda
print(f"\nImplied EV/EBITDA (Year 5): {implied_ev_ebitda:.1f}x")

# ROIC calculation
last_nopat = projections['NOPAT'].iloc[-1]
last_invested_capital = (projections['NWC'].iloc[-1] + 
                         projections['CapEx'].iloc[-1] * 5)  # Rough approximation
roic = last_nopat / last_invested_capital
print(f"Implied ROIC: {roic:.1%}")
print(f"ROIC vs WACC: {roic - wacc:+.1%} (Value {'creating' if roic > wacc else 'destroying'})")

# Sensitivity Analysis
print(f"\n" + "="*80)
print("ONE-WAY SENSITIVITY ANALYSIS")
print("="*80)

# WACC sensitivity
wacc_range = np.linspace(0.08, 0.14, 7)
wacc_sens = dcf.sensitivity_analysis_1d('wacc', wacc_range, shares_outstanding)
print("\nWACC Sensitivity:")
print(wacc_sens.to_string(index=False))

# Terminal growth sensitivity
term_growth_range = np.linspace(0.015, 0.035, 5)
growth_sens = dcf.sensitivity_analysis_1d('terminal_growth', term_growth_range, 
                                         shares_outstanding)
print("\nTerminal Growth Sensitivity:")
print(growth_sens.to_string(index=False))

# Two-way sensitivity table
print(f"\n" + "="*80)
print("TWO-WAY SENSITIVITY: WACC vs TERMINAL GROWTH")
print("="*80)
print("(Value Per Share)")

wacc_vals = np.linspace(0.09, 0.13, 5)
growth_vals = np.linspace(0.02, 0.03, 6)
sens_table = dcf.sensitivity_analysis_2d('wacc', wacc_vals, 'terminal_growth', 
                                        growth_vals, shares_outstanding)
print("\n" + sens_table.to_string())

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Revenue and FCFF projections
ax = axes[0, 0]
x_years = projections.index
ax.bar(x_years, projections['Revenue'], alpha=0.6, label='Revenue', color='skyblue')
ax.plot(x_years, projections['FCFF'], 'o-', linewidth=2, markersize=8, 
        label='Free Cash Flow', color='green')
ax.set_xlabel('Year')
ax.set_ylabel('$ Millions')
ax.set_title('Revenue and Free Cash Flow Projections')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Cash flow waterfall (Year 1)
ax = axes[0, 1]
year1 = projections.iloc[0]
components = ['NOPAT', 'D&A', 'CapEx', 'ΔNWC', 'FCFF']
values = [year1['NOPAT'], year1['D&A'], -year1['CapEx'], 
          -year1['Change_NWC'], year1['FCFF']]
colors = ['green', 'lightgreen', 'red', 'orange', 'darkgreen']
cumulative = np.cumsum([0] + values[:-1])
ax.bar(components, values, bottom=cumulative[:len(values)], color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('$ Millions')
ax.set_title('FCFF Waterfall (Year 1)')
ax.grid(alpha=0.3, axis='y')

# Plot 3: WACC components
ax = axes[0, 2]
components_wacc = ['Cost of\nEquity', 'Cost of Debt\n(After-Tax)', 'WACC']
values_wacc = [dcf.wacc_components['Cost_of_Equity'] * 100,
               dcf.wacc_components['Cost_of_Debt_After_Tax'] * 100,
               dcf.wacc * 100]
colors_wacc = ['red', 'blue', 'purple']
bars = ax.bar(components_wacc, values_wacc, color=colors_wacc, alpha=0.7)
ax.set_ylabel('Rate (%)')
ax.set_title('Cost of Capital Components')
ax.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, values_wacc):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}%',
            ha='center', fontweight='bold')

# Plot 4: Enterprise value breakdown
ax = axes[1, 0]
components_ev = ['PV Forecast\nPeriod', 'PV Terminal\nValue']
values_ev = [dcf.valuation_summary['PV_Forecast_Period'],
             dcf.valuation_summary['PV_Terminal_Value']]
colors_ev = ['steelblue', 'coral']
bars = ax.bar(components_ev, values_ev, color=colors_ev, alpha=0.7)
ax.set_ylabel('$ Millions')
ax.set_title('Enterprise Value Components')
for bar, val in zip(bars, values_ev):
    ax.text(bar.get_x() + bar.get_width()/2, val + 100, f'${val:.0f}M',
            ha='center', fontweight='bold')
ax.text(0.5, dcf.enterprise_value + 300, 
        f'Total EV: ${dcf.enterprise_value:,.0f}M',
        ha='center', fontsize=12, fontweight='bold', 
        transform=ax.transData)

# Plot 5: WACC sensitivity
ax = axes[1, 1]
ax.plot(wacc_sens['Parameter_Value'] * 100, wacc_sens['Value_Per_Share'], 
        'o-', linewidth=2, markersize=8, color='darkred')
ax.axhline(value_per_share, color='green', linestyle='--', 
           label=f'Base: ${value_per_share:.2f}')
ax.axhline(current_market_price, color='blue', linestyle='--',
           label=f'Market: ${current_market_price:.2f}')
ax.fill_between(wacc_sens['Parameter_Value'] * 100,
                wacc_sens['Value_Per_Share'],
                current_market_price, alpha=0.2)
ax.set_xlabel('WACC (%)')
ax.set_ylabel('Value Per Share ($)')
ax.set_title('Sensitivity to WACC')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Terminal growth sensitivity
ax = axes[1, 2]
ax.plot(growth_sens['Parameter_Value'] * 100, growth_sens['Value_Per_Share'],
        'o-', linewidth=2, markersize=8, color='darkgreen')
ax.axhline(value_per_share, color='green', linestyle='--',
           label=f'Base: ${value_per_share:.2f}')
ax.axhline(current_market_price, color='blue', linestyle='--',
           label=f'Market: ${current_market_price:.2f}')
ax.fill_between(growth_sens['Parameter_Value'] * 100,
                growth_sens['Value_Per_Share'],
                current_market_price, alpha=0.2, color='green')
ax.set_xlabel('Terminal Growth Rate (%)')
ax.set_ylabel('Value Per Share ($)')
ax.set_title('Sensitivity to Terminal Growth')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Heat map for 2D sensitivity
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sens_table.astype(float), annot=True, fmt='.1f', cmap='RdYlGn',
            center=current_market_price, cbar_kws={'label': 'Value Per Share ($)'},
            ax=ax)
ax.set_xlabel('Terminal Growth Rate')
ax.set_ylabel('WACC')
ax.set_title('Two-Way Sensitivity: Value Per Share Heatmap')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. Base case DCF suggests {upside:+.1f}% {'upside' if upside > 0 else 'downside'} vs market")
print(f"2. Terminal value represents {dcf.valuation_summary['Terminal_Value_Percent']:.0f}% of EV")
print(f"3. WACC range 9-13% → Value range ${wacc_sens['Value_Per_Share'].min():.2f} - ${wacc_sens['Value_Per_Share'].max():.2f}")
print(f"4. Terminal growth +50 bps → +${growth_sens['Value_Per_Share'].iloc[-1] - value_per_share:.2f} per share")
print(f"5. Company {'creates' if roic > wacc else 'destroys'} value (ROIC {roic:.1%} vs WACC {wacc:.1%})")
print(f"6. Implied EV/EBITDA {implied_ev_ebitda:.1f}x (compare to sector peers)")
```

## 6. Challenge Round
Advanced DCF concepts and extensions:

1. **Circular Reference Problem:** WACC calculation requires enterprise value (for D/E ratio), but EV is the DCF output. Implement iterative solver to converge on consistent WACC and EV simultaneously. When does iteration fail to converge?

2. **Multi-Stage Growth Models:** Company has 3 phases: High growth (20%, 3 years) → Transition (declining from 20% to 3%, 7 years) → Mature (3% perpetual). Project cash flows with smooth transition. How sensitive is valuation to transition period length?

3. **Adjusted Present Value (APV):** Value company with changing leverage (e.g., LBO scenario). APV = Unlevered firm value + PV(Tax shields) - PV(Financial distress costs). When is APV preferred over WACC-based DCF?

4. **Real Options Embedded in DCF:** Pharmaceutical company with drug pipeline. Each Phase I/II/III trial is an option to continue or abandon. Modify DCF to incorporate option value (using Black-Scholes or binomial trees). Compare to standard DCF.

5. **Scenario Modeling with Correlations:** Define Bull/Base/Bear scenarios for revenue growth, margins, WACC. But these are correlated (recession → low growth AND high WACC). Use copulas or correlation matrix to sample joint scenarios. Calculate probability-weighted fair value.

6. **Mid-Year Convention:** Standard DCF assumes cash flows at year-end. More realistic: Cash flows throughout year (mid-year convention). Adjust discount factors: Instead of (1+WACC)^t, use (1+WACC)^(t-0.5). Quantify impact on valuation.

7. **Non-Constant Growth Path:** Instead of smooth declining growth, model realistic lumpy growth (new product launch in Year 3 → spike, then fade). How does this affect terminal value calculation? Should TV be based on normalized FCF?

## 7. Key References
- [Damodaran, "Investment Valuation" (3rd Edition, 2012)](http://pages.stern.nyu.edu/~adamodar/) - comprehensive DCF methodology with sector-specific approaches
- [McKinsey, "Valuation: Measuring and Managing the Value of Companies" (7th Edition, 2020)](https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/valuation) - practitioner guide with case studies
- [CFA Institute, "Equity Asset Valuation" (3rd Edition, 2015)](https://www.cfainstitute.org/) - rigorous treatment of DCF models, sensitivity analysis, and valuation multiples

---
**Status:** Core intrinsic valuation method | **Complements:** Financial Statement Analysis, WACC/Cost of Capital, Terminal Value Methods, Comparable Company Analysis, Precedent Transactions, Dividend Discount Models
