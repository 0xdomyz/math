# Valuation Sensitivity & Scenario Analysis

## 1. Concept Skeleton
**Definition:** Quantification of how valuation changes with variations in key assumptions; identification of value drivers; risk assessment  
**Purpose:** Understand valuation robustness; identify critical assumptions; stress-test investment thesis; communicate uncertainty range  
**Prerequisites:** Base case valuation model, key drivers (WACC, growth, margins), sensitivity tables, scenario weighting

## 2. Comparative Framing
| Approach | One-Way Sensitivity | Two-Way Sensitivity Table | Scenario Analysis | Tornado Diagram |
|----------|-------------------|------------------------|------------------|-----------------|
| **Definition** | Change one variable; observe impact | Grid of two key drivers | Discrete scenarios (bull/base/bear) | Ranked impact bars |
| **Complexity** | Simple; quick | Moderate; good visual | Moderate; narrative | Simple; communication |
| **Key insight** | Which variable most sensitive? | Interaction between drivers | Holistic outcome; probability-weighted | Ranked driver importance |
| **Best use** | Initial exploration; identify drivers | Presentation to stakeholders | Decision-making; multiple paths | Executive summary |
| **Resolution** | Continuous (1% increments) | Discrete grid (±20% typical) | Lumpy (3-5 scenarios) | Ranked; prioritized |
| **Communication** | Good for analysts | Excellent for board/investors | Best for storytelling | Excellent for focus |

## 3. Examples + Counterexamples

**Simple Example:**  
DCF valuation $60/share (base case). WACC 8%; terminal growth 3%. Sensitivity: WACC 7% → $75/share; 9% → $48/share. Terminal growth 2% → $50/share; 4% → $75/share. Range $48-75 reasonable; guides buy/sell decisions around current market price.

**Failure Case:**  
Valuation $60/share appears robust (table shows narrow range). But overlooked: If competitor enters market (1 in 5 chance), revenue growth falls to 1% (vs. base 5%). New valuation: $30/share. Scenario analysis would have captured tail risk; simple sensitivity obscured.

**Edge Case:**  
Sensitivity shows valuation insensitive to WACC (8% range → only 5% valuation change). But WACC actually highly uncertain (±2%). Discovery: Valuation insensitivity to uncertain variable suggests model overconstrained elsewhere; possibly terminal value assumption unreasonably high.

## 4. Layer Breakdown
```
Sensitivity & Scenario Analysis Structure:
├─ One-Way Sensitivity Analysis:
│   ├─ Mechanics:
│   │   ├─ Hold all variables constant (base case)
│   │   ├─ Change one variable in increments (±1%, ±5%, ±10%)
│   │   ├─ Recalculate valuation
│   │   ├─ Tabulate results
│   │   └─ Format: Variable values (rows) vs. Valuation output (column)
│   ├─ Example: WACC Sensitivity
│   │   ├─ Base case valuation: $60/share (WACC 8%, terminal growth 3%, revenue growth 5%)
│   │   ├─ WACC scenario range: 6% to 10% (in 0.5% increments)
│   │   ├─ Calculation method:
│   │   │   ├─ Year 1-5 FCF unchanged (deterministic given growth 5%)
│   │   │   │   ├─ Year 1: $120M
│   │   │   │   ├─ Year 2: $126M
│   │   │   │   ├─ Year 3: $132M
│   │   │   │   ├─ Year 4: $139M
│   │   │   │   ├─ Year 5: $146M
│   │   │   ├─ Terminal value recalculated at each WACC:
│   │   │   │   ├─ TV = FCF(Y5) × (1 + g) / (WACC - g)
│   │   │   │   ├─ WACC 6%: TV = $146M × 1.03 / (0.06 - 0.03) = $5,023M
│   │   │   │   ├─ WACC 7%: TV = $146M × 1.03 / (0.07 - 0.03) = $3,765M
│   │   │   │   ├─ WACC 8%: TV = $146M × 1.03 / (0.08 - 0.03) = $3,005M (base)
│   │   │   │   ├─ WACC 9%: TV = $146M × 1.03 / (0.09 - 0.03) = $2,502M
│   │   │   │   ├─ WACC 10%: TV = $146M × 1.03 / (0.10 - 0.03) = $2,147M
│   │   │   ├─ Present value of cash flows (unchanged across sensitivity):
│   │   │   │   ├─ Year 1-5 PV ≈ $550M (discounted at respective WACC)
│   │   │   │   └─ Note: This also changes with WACC; simplified here
│   │   │   ├─ Simplified assumption: Treat PV(Y1-5) as roughly constant
│   │   │   │   ├─ (More precisely: Earlier years less sensitive to WACC than terminal)
│   │   │   │   └─ Total EV ≈ PV(Y1-5) + PV(TV)
│   │   │   └─ Total valuation at each WACC:
│   │   │       ├─ WACC 6%: EV ≈ $550M + $3,400M ≈ $3,950M (simplified); ~$7.90/share
│   │   │       ├─ WACC 7%: EV ≈ $3,200M; ~$6.40/share
│   │   │       ├─ WACC 8%: EV ≈ $3,050M; ~$6.10/share (actual ~$60/share after adjustments)
│   │   │       ├─ WACC 9%: EV ≈ $2,900M; ~$5.80/share
│   │   │       └─ WACC 10%: EV ≈ $2,750M; ~$5.50/share
│   │   ├─ Interpretation:
│   │   │   ├─ Each 1% WACC change → ~10-15% valuation change (large impact)
│   │   │   ├─ WACC 8%±1% → Valuation $5.50-7.90/share (range $2.40; ~40% swing)
│   │   │   ├─ Implication: Valuation highly sensitive to cost of capital
│   │   │   └─ Investment insight: If uncertain about WACC, valuation range wide (risky)
│   │   └─ Alternative drivers (similar one-way sensitivity):
│   │       ├─ Terminal growth (2%-4%) → Valuation $45-80/share (range $35; highly sensitive)
│   │       ├─ Revenue growth Y1-5 (2%-8%) → Valuation $52-68/share (less sensitive than WACC/terminal)
│   │       ├─ Operating margin (12%-18%) → Valuation $55-65/share (moderate sensitivity)
│   │       └─ Capex as % of revenue (3%-5%) → Valuation $58-62/share (low sensitivity)
│   ├─ Sensitivity Table Format:
│   │   ├─ Header: Variable name; values tested
│   │   ├─ Column: Valuation outcome (per share, enterprise value, or equity value)
│   │   ├─ Example:
│   │   │   ├─ WACC | Valuation/share
│   │   │   ├─ 6.0% | $7.90
│   │   │   ├─ 6.5% | $7.25
│   │   │   ├─ 7.0% | $6.40
│   │   │   ├─ 7.5% | $6.15
│   │   │   ├─ 8.0% | $6.10 (base)
│   │   │   ├─ 8.5% | $5.90
│   │   │   ├─ 9.0% | $5.80
│   │   │   ├─ 9.5% | $5.60
│   │   │   └─ 10.0% | $5.50
│   │   ├─ Visual: Can plot line (sensitivity curve; shows relationship)
│   │   └─ Interpretation: Steeper curve = more sensitive variable
│   └─ Limitations:
│       ├─ Assumes independence: Change WACC; ignore that higher rates may reduce growth
│       ├─ Reality: Variables correlated (recession → higher WACC, lower growth simultaneously)
│       ├─ Recommendation: Use two-way sensitivity for key correlated pairs
│       └─ Note: One-way useful for initial screening; two-way for deeper analysis
├─ Two-Way Sensitivity Tables:
│   ├─ Mechanics:
│   │   ├─ Vary two key drivers simultaneously
│   │   ├─ Format: Rows = Variable 1; Columns = Variable 2; Cell entries = Valuation
│   │   ├─ Matrix visualization: Color-coded (green = high valuation; red = low)
│   │   └─ Insight: See interaction between drivers
│   ├─ Example: WACC vs. Terminal Growth Two-Way Table
│   │   ├─ Row headers: WACC (6%, 7%, 8%, 9%, 10%)
│   │   ├─ Column headers: Terminal Growth (2%, 2.5%, 3%, 3.5%, 4%)
│   │   ├─ Cell formula: Valuation = f(WACC, Terminal Growth) [from DCF model]
│   │   ├─ Example calculation for WACC 8%, Terminal Growth 3% (base):
│   │   │   ├─ TV = $146M × 1.03 / (0.08 - 0.03) = $3,005M
│   │   │   ├─ EV ≈ $3,050M → ~$60/share (as before)
│   │   ├─ Table entries:
│   │   │   ├─        | 2.0% | 2.5% | 3.0% | 3.5% | 4.0%
│   │   │   ├─ 6.0%  | $9.50 | $10.40 | $11.50 | $13.00 | $15.20
│   │   │   ├─ 7.0%  | $6.80 | $7.30 | $8.00 | $8.90 | $10.20
│   │   │   ├─ 8.0%  | $5.00 | $5.50 | $6.10 | $6.80 | $7.80 ← base $6.10
│   │   │   ├─ 9.0%  | $4.00 | $4.40 | $4.80 | $5.30 | $5.90
│   │   │   └─ 10.0% | $3.40 | $3.70 | $4.00 | $4.35 | $4.80
│   │   ├─ Interpretation:
│   │   │   ├─ Base case (8%, 3%): $6.10/share
│   │   │   ├─ Bull case (6%, 3.5%): $13.00/share (lower WACC, higher growth)
│   │   │   ├─ Bear case (10%, 2%): $3.40/share (higher WACC, lower growth)
│   │   │   ├─ Range: $3.40-$15.20 (77% swing; huge uncertainty)
│   │   │   └─ Implication: Valuation critically dependent on these two assumptions
│   │   ├─ Visual insight:
│   │   │   ├─ Table shows non-linear relationship
│   │   │   ├─ As WACC approaches Terminal Growth, valuation → ∞ (denominator → 0)
│   │   │   ├─ For example: WACC 3.1%, Terminal Growth 3.0% → extreme sensitivity
│   │   │   └─ Risk: Narrow margin of safety between assumptions
│   │   └─ Interaction insight:
│   │       ├─ Higher WACC + Lower growth = Compounding negative (worst case $3.40)
│   │       ├─ Lower WACC + Higher growth = Compounding positive (best case $15.20)
│   │       ├─ Mixed (8%, 2%) = $5.00 (below base)
│   │       └─ Message: Multiple drivers move together in scenarios
│   ├─ Alternative two-way combinations:
│   │   ├─ Revenue Growth vs. EBIT Margin (operational drivers)
│   │   ├─ Terminal Growth vs. EBIT Margin (long-term sustainability)
│   │   ├─ WACC vs. Revenue Growth (cost of capital vs. upside)
│   │   └─ Pick pairs with: (1) High correlation, (2) High uncertainty, (3) High impact
│   └─ Presentation:
│       ├─ Color coding (heatmap):
│       │   ├─ Green: High valuation ($10+/share)
│       │   ├─ Yellow: Mid valuation ($5-8/share)
│       │   ├─ Red: Low valuation (<$4/share)
│       │   └─ Visual immediately communicates downside/upside range
│       ├─ Highlight base case cell (centered; shows positioning)
│       └─ Add current market price overlay (if available; shows if trading fair value)
├─ Scenario Analysis:
│   ├─ Definition:
│   │   ├─ Discrete scenarios (not continuous sensitivity)
│   │   ├─ Each scenario = coherent set of assumptions (multiple drivers bundled)
│   │   ├─ Examples: Bull case, Base case, Bear case (typically 3 scenarios; up to 5 if nuance needed)
│   │   └─ Each scenario assigned probability
│   ├─ Bull Case (Optimistic):
│   │   ├─ Assumptions:
│   │   │   ├─ Revenue growth: 8% annually (vs. 5% base)
│   │   │   │   ├─ Driver: Company gains market share; new products succeed; pricing power
│   │   │   │   └─ Probability: 25% (1 in 4 chance of strong execution)
│   │   │   ├─ EBIT margin: 17% (vs. 15% base)
│   │   │   │   ├─ Driver: Operating leverage; scale benefits; cost control
│   │   │   │   └─ Assumption: Margin expansion sustainable (high confidence)
│   │   │   ├─ WACC: 7% (vs. 8% base)
│   │   │   │   ├─ Driver: Lower risk premium if company outperforms; reduces leverage
│   │   │   │   └─ Rationale: Success reduces financial risk
│   │   │   ├─ Terminal growth: 3.5% (vs. 3% base)
│   │   │   │   ├─ Driver: Company becomes larger player; grows faster than GDP long-term
│   │   │   │   └─ Ceiling: Not >3.5% (still limited by market maturity)
│   │   │   └─ Terminal FCF: $180M (higher due to growth + margin assumptions)
│   │   ├─ Valuation impact:
│   │   │   ├─ Higher FCF streams (8% growth) → larger explicit period value
│   │   │   ├─ Higher terminal FCF ($180M) + higher terminal growth (3.5%) → larger terminal value
│   │   │   ├─ Lower WACC (7%) → lower discount rate → higher present value
│   │   │   ├─ Combined effect: Valuation $80-90/share (30-50% above base $60)
│   │   │   └─ Example: Bull case $85/share
│   │   ├─ Narrative:
│   │   │   ├─ Company executes strategy flawlessly
│   │   │   ├─ New market opens; scale achieved; margins expand
│   │   │   ├─ Risk profile improves; valuation re-rates higher
│   │   │   └─ Catalyst: Product launch success; market share gains
│   │   └─ Probability assessment:
│   │       ├─ Management track record: Strong (25% bull case warranted)
│   │       ├─ Market conditions: Favorable (tailwind; increases bull odds)
│   │       ├─ Competitive position: Defensible (moat present; reduces downside risk from competition)
│   │       └─ Typical probability: 20-30% for bull case (not most likely, but plausible)
│   ├─ Base Case (Expected):
│   │   ├─ Assumptions:
│   │   │   ├─ Revenue growth: 5% (consensus; balanced)
│   │   │   ├─ EBIT margin: 15% (current; stable forward)
│   │   │   ├─ WACC: 8% (neutral cost of capital)
│   │   │   ├─ Terminal growth: 3% (GDP growth aligned)
│   │   │   └─ Terminal FCF: $146M
│   │   ├─ Valuation: $60/share (our base case)
│   │   ├─ Narrative:
│   │   │   ├─ Company executes moderately well
│   │   │   ├─ Revenue grows with market; margins stable
│   │   │   ├─ No major improvements or deterioration
│   │   │   └─ Catalyst: Normal business progress
│   │   └─ Probability: 50% (most likely scenario)
│   ├─ Bear Case (Pessimistic):
│   │   ├─ Assumptions:
│   │   │   ├─ Revenue growth: 2% annually (vs. 5% base)
│   │   │   │   ├─ Driver: Market saturation; competition; product obsolescence
│   │   │   │   └─ Probability: 25% (if thesis wrong or market shifts)
│   │   │   ├─ EBIT margin: 12% (vs. 15% base)
│   │   │   │   ├─ Driver: Margin pressure; price competition; cost inflation
│   │   │   │   └─ Assumption: Company defensive but not value-accretive
│   │   │   ├─ WACC: 9.5% (vs. 8% base)
│   │   │   │   ├─ Driver: Higher risk perception if execution falters; leverage risk rises
│   │   │   │   └─ Rationale: Market demands higher return for distressed business
│   │   │   ├─ Terminal growth: 2% (vs. 3% base)
│   │   │   │   ├─ Driver: Long-term growth constrained; market shift adversity
│   │   │   │   └─ Ceiling: Still positive (not bankruptcy scenario)
│   │   │   └─ Terminal FCF: $110M (lower due to all headwinds)
│   │   ├─ Valuation impact:
│   │   │   ├─ Lower FCF streams (2% growth) → smaller explicit period value
│   │   │   ├─ Lower terminal FCF ($110M) + lower terminal growth (2%) → smaller terminal value
│   │   │   ├─ Higher WACC (9.5%) → higher discount rate → lower present value
│   │   │   ├─ Combined effect: Valuation $35-45/share (25-40% below base $60)
│   │   │   └─ Example: Bear case $40/share
│   │   ├─ Narrative:
│   │   │   ├─ Company faces headwinds; execution struggles
│   │   │   ├─ Market competitive; margins squeeze; growth disappoints
│   │   │   ├─ Risk profile worsens; valuation multiple contracts
│   │   │   └─ Catalyst: Competitive loss; margin erosion; guidance cut
│   │   └─ Probability: 25% (significant downside risk; reflects execution or market risks)
│   ├─ Probability-Weighted Valuation:
│   │   ├─ Formula: Expected Value = P(Bull) × Value(Bull) + P(Base) × Value(Base) + P(Bear) × Value(Bear)
│   │   ├─ Example:
│   │   │   ├─ Bull: 25% × $85 = $21.25
│   │   │   ├─ Base: 50% × $60 = $30.00
│   │   │   ├─ Bear: 25% × $40 = $10.00
│   │   │   ├─ Expected Value: $21.25 + $30.00 + $10.00 = $61.25/share
│   │   │   └─ Interpretation: Risk-adjusted fair value accounting for scenarios
│   │   ├─ Comparison to base case:
│   │   │   ├─ Base case only: $60/share
│   │   │   ├─ Expected (probability-weighted): $61.25/share
│   │   │   ├─ Difference: +$1.25 (about 2%; small difference suggests fairly balanced risk)
│   │   │   └─ Implication: Bull and bear risks roughly balanced
│   │   ├─ Probability adjustment:
│   │   │   ├─ If probabilities shifted: Bull 35%, Base 40%, Bear 25%
│   │   │   ├─ New expected: 0.35×$85 + 0.40×$60 + 0.25×$40 = $63.75/share (higher)
│   │   │   ├─ Rationale: If bull case more likely, valuation rises
│   │   │   └─ Exercise: What probabilities justify each valuation?
│   │   └─ Sensitivity to probabilities:
│   │       ├─ If Bull 50%, Base 30%, Bear 20% → Expected $68/share (bullish)
│   │       ├─ If Bull 20%, Base 50%, Bear 30% → Expected $57/share (bearish)
│   │       └─ Message: Probability estimates matter as much as scenario valuations
│   ├─ Additional Scenarios (if needed):
│   │   ├─ Upside Case (between Bull and Base)
│   │   │   ├─ Valuation: $72/share
│   │   │   ├─ Probability: 15%
│   │   │   └─ Use: Better granularity; captures nuance
│   │   ├─ Downside Case (between Base and Bear)
│   │   │   ├─ Valuation: $48/share
│   │   │   ├─ Probability: 15%
│   │   │   └─ Use: More detail on risk profile
│   │   └─ Tail Risk Case (extreme)
│   │       ├─ Example: Bankruptcy/restructuring → valuation $10-20/share
│   │       ├─ Probability: 5-10% (captured unlikely bad outcome)
│   │       └─ Message: Communicates downside cap if thesis breaks
│   └─ Narrative Integration:
│       ├─ Bull case: "If new product succeeds and market share expands"
│       ├─ Base case: "Most likely path given current trajectory"
│       ├─ Bear case: "If competitive pressures materialize and margins compress"
│       ├─ Combined: "Expected value $61, with range $40-85 depending on execution"
│       └─ Communication: Scenarios as stories; easier to understand than formulas
├─ Monte Carlo Simulation (Advanced):
│   ├─ Definition:
│   │   ├─ Continuous probability distributions for each variable
│   │   ├─ Thousands of simulations; randomly draw from distributions
│   │   ├─ Outcome: Distribution of valuations (not just discrete scenarios)
│   │   └─ Use: Capture full uncertainty range; tail risk quantification
│   ├─ Setup:
│   │   ├─ Define probability distribution for each key variable:
│   │   │   ├─ Revenue growth: Normal distribution, mean 5%, std dev 2%
│   │   │   │   ├─ Interpretation: 68% of outcomes between 3%-7%
│   │   │   │   └─ Can capture: Occasional extreme outcomes (0% or 10%)
│   │   │   ├─ EBIT margin: Beta distribution, mode 15%, range 10%-20%
│   │   │   ├─ WACC: Normal distribution, mean 8%, std dev 1%
│   │   │   ├─ Terminal growth: Normal distribution, mean 3%, std dev 0.5%
│   │   │   └─ Note: Can assume correlation between variables (e.g., higher growth→lower WACC)
│   │   ├─ Run simulation:
│   │   │   ├─ Random draw from each distribution (e.g., growth 4.2%, margin 14.8%, WACC 7.9%)
│   │   │   ├─ Calculate valuation from this set
│   │   │   ├─ Repeat 10,000 times
│   │   │   └─ Result: 10,000 valuation outcomes
│   │   └─ Output:
│   │       ├─ Mean valuation: $58/share
│   │       ├─ Std dev: $12/share
│   │       ├─ 5th percentile: $38/share (downside tail)
│   │       ├─ 95th percentile: $82/share (upside tail)
│   │       ├─ Distribution histogram: Show frequency of each valuation level
│   │       └─ Interpretation: Wide range reflects high uncertainty
│   ├─ Advantages over scenarios:
│   │   ├─ Continuous vs. discrete: Captures full spectrum, not just 3-5 points
│   │   ├─ Correlation: Can model that high growth → lower risk → lower WACC (realistic)
│   │   ├─ Tail risk: Explicitly quantifies probability of extreme outcomes
│   │   ├─ Sensitivity: Identifies which variables most impact distribution spread
│   │   └─ Decision: Can set confidence levels (e.g., 80% confident valuation > $45)
│   └─ Limitations:
│       ├─ Complex: Requires programming; harder to communicate
│       ├─ Black box: Many assumptions; less transparent than scenarios
│       ├─ Distribution assumption: If wrong distribution chosen, results misleading
│       ├─ Correlation specification: Difficult to estimate; errors cascade
│       └─ Recommendation: Use scenarios for main analysis; Monte Carlo as supplement for tail risk
├─ Tornado Diagram (Impact Ranking):
│   ├─ Definition:
│   │   ├─ Bars represent each variable's impact on valuation
│   │   ├─ Longer bar = larger impact
│   │   ├─ Sorted by length (largest at top)
│   │   ├─ Resembles tornado shape (large at top, narrow at bottom)
│   │   └─ Use: Communicate which variables matter most
│   ├─ Construction:
│   │   ├─ For each variable:
│   │   │   ├─ Calculate high-case valuation (variable at +1 std dev)
│   │   │   ├─ Calculate low-case valuation (variable at -1 std dev)
│   │   │   ├─ Bar range = High - Low
│   │   │   └─ Rank by bar length
│   │   ├─ Example:
│   │   │   ├─ Terminal growth: ±0.5% → Valuation range $50-70 → Bar width $20 (largest)
│   │   │   ├─ WACC: ±1% → Valuation range $48-72 → Bar width $24 (largest!)
│   │   │   ├─ Revenue growth: ±2% → Valuation range $55-65 → Bar width $10
│   │   │   ├─ EBIT margin: ±2% → Valuation range $58-62 → Bar width $4
│   │   │   └─ Terminal FCF: ±5% → Valuation range $57-63 → Bar width $6
│   │   ├─ Sorted by impact:
│   │   │   ├─ WACC: $24 width (top; most impactful)
│   │   │   ├─ Terminal growth: $20 width
│   │   │   ├─ Revenue growth: $10 width
│   │   │   ├─ Terminal FCF: $6 width
│   │   │   └─ EBIT margin: $4 width (bottom; least impactful)
│   │   └─ Visual: Bars extend left/right from center line (base case $60)
│   ├─ Interpretation:
│   │   ├─ WACC and Terminal growth drive most valuation uncertainty
│   │   ├─ EBIT margin has minimal impact (less critical to get right)
│   │   ├─ Investment focus: Reduce WACC/terminal growth uncertainty (most valuable)
│   │   └─ Due diligence priority: Investigate WACC drivers; terminal sustainability
│   ├─ Application:
│   │   ├─ Identify where to spend analysis time (biggest bars)
│   │   ├─ Communicate with stakeholders (clear visual of key risks)
│   │   ├─ Hedging priority: If concerned about value, hedge variables with largest bars
│   │   └─ Sensitivity investment: If valuation highly sensitive to one variable, confidence lower
│   └─ Distinction from two-way table:
│       ├─ Two-way: Shows interaction between two variables
│       ├─ Tornado: Shows independent impact of each variable
│       ├─ Complementary: Use both for complete picture
│       └─ Tornado simpler; two-way more insightful
├─ Validation & Application:
│   ├─ Sanity checks:
│   │   ├─ If sensitivity tables show valuation insensitive to uncertain variable (e.g., 5% change in WACC → 0.5% valuation change)
│   │   │   ├─ Possible explanations: (1) Terminal value dominates (large %); (2) Model error; (3) Offsetting effects
│   │   │   ├─ Investigation: Recalculate; verify formulas
│   │   │   └─ Implication: If true, valuation robust to cost-of-capital risk
│   │   ├─ If scenario range very wide ($30-100/share)
│   │   │   ├─ Interpretation: High uncertainty; hard to value
│   │   │   ├─ Decision: Requires either high margin of safety or strong conviction
│   │   │   └─ Action: Narrow range via better assumptions or skip investment
│   │   ├─ If two-way table asymmetric (upper-left quadrant extreme; lower-right benign)
│   │   │   ├─ Example: WACC 6%+Growth 4% = $50 vs. WACC 10%+Growth 2% = $2
│   │   │   ├─ Interpretation: Valuation model unstable; sensitive to parameter ranges
│   │   │   └─ Implication: Model may be overparameterized; simplify or add constraints
│   │   └─ If expected value (probability-weighted) materially different from base case
│   │       ├─ Example: Base $60 vs. Expected $51 (17% difference)
│   │       ├─ Interpretation: Downside risk exceeds upside potential (skewed distribution)
│   │       ├─ Investment decision: Use expected value, not base case, for sizing
│   │       └─ Message: Risk-adjusted valuation often lower than base case
│   ├─ Investment application:
│   │   ├─ If current price below expected value: Buy candidate (risk-reward favorable)
│   │   ├─ If current price above valuation range: Avoid (overvalued in all scenarios)
│   │   ├─ If current price in range but near bear case: Requires margin of safety (>20% discount to bear)
│   │   └─ If current price in bull case → bear case range: Fair value; no edge
│   ├─ Portfolio sizing:
│   │   ├─ High sensitivity (wide range) → Smaller position (more uncertainty)
│   │   ├─ Low sensitivity (narrow range) → Larger position (higher confidence)
│   │   ├─ Formula: Position size ∝ 1 / Valuation range width
│   │   └─ Example: Range $50-70 (20-unit width) → smaller than Range $55-65 (10-unit width)
│   └─ Communication:
│       ├─ Stakeholder audience (board/investors): Use scenario narratives + expected value
│       ├─ Analyst audience (technical): Show two-way tables + tornado diagrams
│       ├─ Executive summary: Lead with valuation range (bear-base-bull) + key drivers
│       └─ Detail: Appendix contains sensitivity tables + tornado diagram
```

**Key Insight:** Sensitivity identifies what matters; scenarios bundle assumptions; probability weighting yields expected value; use all three for robust valuation view

## 5. Mini-Project
[Code would include: sensitivity calculations, two-way table generation, scenario modeling, Monte Carlo simulation, tornado diagram visualization]

## 6. Challenge Round
When sensitivity analysis deceives:
- **False precision**: Sensitivity table shows valuation $57-63 (narrow range); investor feels confident; actual key assumptions (terminal growth, WACC) highly correlated; true range $45-80
- **Parameter instability**: Changes WACC assumption 1%; valuation changes 15%; seems sensitive; but actually model overlevered on terminal value; fixing valuation formula → sensitivity becomes 5% (true picture)
- **Probability trap**: Assigning 25% to bear case; but historical frequency of similar situations → 40% bear outcome; underweighting downside; recommendation too optimistic
- **Scenario cherry-picking**: Bull case assumes "perfect execution"; but company has 10% past success rate; assigning 25% probability to bull unrealistic (should be 5-10%)
- **Terminal value dominance hidden**: Explicit period (Y1-5) sensitivities look reasonable; but terminal value = 75% of total value; terminal growth changes 2.5%→3.0% → 20% valuation change (sensitivity lurking)
- **Tornado bar misinterpretation**: WACC bar longest; thought to be most important; but in base case, other variables already highly certain; WACC uncertainty matters less in context

## 7. Key References
- [CFA Institute - Sensitivity Analysis Framework](https://www.cfainstitute.org/) - Best practices, scenario development
- [Aswath Damodaran - Valuation Scenarios](https://pages.stern.nyu.edu/~adamodar/) - Case studies, probability weighting
- [McKinsey - Monte Carlo Valuation](https://www.mckinsey.com/) - Simulation, tail risk quantification

---
**Status:** Risk quantification | **Complements:** DCF Modeling, Scenario Planning, Decision-Making, Uncertainty Management
