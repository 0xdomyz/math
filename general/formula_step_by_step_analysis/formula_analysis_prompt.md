# Copilot Instructions: Formula Deep Dive Analysis

## Task
Create an interactive Python analysis script using `#%%` cell markers for VS Code interactive mode that performs a comprehensive mathematical deep dive of a given formula.

## Format Requirements
- Use `#%%` to separate each cell (enables interactive execution with Shift+Enter)
- Keep analysis concise - as short as possible while being complete
- Explain each part separately, one cell per component
- DO NOT create a class or function-based script
- Layout code chunk by chunk for slow, interactive exploration

## Structure Template

### 1. Setup Cell
```python
# %% Setup
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

### 2. Formula Decomposition Granularity

**IMPORTANT: Break down formulas to the fundamental operation level.**

**Decomposition Rules:**
- Each part should represent ONE fundamental mathematical operation or transformation
- Basic arithmetic (±, ×, ÷) between simple terms can be combined
- Complex operations like log, exp, reciprocal, square root are SEPARATE parts
- Multi-step expressions must be broken into layers

**What is ONE part:**
- ✓ `S/K` (simple ratio)
- ✓ `ln(S/K)` (logarithm of ratio)
- ✓ `σ²/2` (square and divide)
- ✓ `1 + x` (simple addition)
- ✓ `e^(-rT)` (exponential)

**What is TOO COMPLEX for one part:**
- ✗ `d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)` - This combines: log, addition, multiplication, square root, division
- ✗ `(bp - q) / b` - Should split into: (bp - q) as numerator, then division by b
- ✗ `S₀·N(d₁) - K·e^(-rT)·N(d₂)` - Should show e^(-rT), then each term, then subtraction

**Example Breakdown:** For `d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)`:
```
Part 0: S/K (moneyness ratio)
Part 1: ln(S/K) (log moneyness)
Part 2: σ²/2 (variance adjustment)
Part 3: (r + σ²/2)T (drift term)
Part 4: ln(S/K) + (r + σ²/2)T (numerator: drift-adjusted log moneyness)
Part 5: σ√T (volatility scaling)
Part 6: d₁ = [numerator] / [denominator] (complete standardization)
```

Break the formula into constituent parts (Part 0, Part 1, Part 2, etc.)
Each part gets its own cell with:

```python
# %% [markdown]
# ### PART N: [Component Name]
#
# **Formula:** [equation or description]
#
# **Domain:** [...] | **Range:** [...] | **Shape:** [...]

# %%
[define variables and compute values]

# %% [markdown]
# **Properties:** [2-3 line explanation of behavior]
# - [Key value/interpretation 1]
# - [Key value/interpretation 2]
# - [Key value/interpretation 3]

# %%
[create single focused plot showing this component]
plt.figure(figsize=(10, 6))
[plotting code]
plt.show()

# %% [markdown]
# **Motivation:** [1-2 sentences explaining WHY this step exists]
```

### 3. Required Elements for Each Part
- **Domain and Range**: Mathematical bounds
- **Shape description**: Linear, hyperbolic, sigmoid, etc.
- **Key properties**: Monotonicity, continuity, special values
- **Visual plot**: Single clear visualization showing behavior
- **Interpretation**: What the math means practically
- **Motivation**: Why this transformation is used

### 4. Additional Analysis Cells
After individual parts, include:

```python
# %% [markdown]
# ### Overlay: All Transformations Together
# Show all parts on same plot to see composition

# %%
[code for overlay visualization]

# %% [markdown]
# ### Sensitivity Analysis
# Plot derivative or rate of change

# %%
[code for sensitivity analysis]

# %% [markdown]
# ### 3D Visualization
# Surface plot showing formula behavior across input space

# %%
# %%
[code for 3D visualization]

# %% [markdown]
# ### Related Formulas (Summary)
# Dictionary/table of similar formulas with brief comparisons

# %%
[code with dictionary/print statements]

# %% [markdown]
# ### Key Properties Summary

# %%
[print statements summarizing all key mathematical properties]

# %% [markdown]
# ### Design Intuition

# %%
[print statements explaining overall design philosophy]
```

## Content Guidelines

### For Each Component Part:
1. **Markdown header**: Use `# %% [markdown]` followed by `# ### PART N: Title`
2. **Formula and metadata**: Domain, range, shape in markdown with bold labels
3. **Code cell**: Use `# %%` marker before code blocks
4. **Properties in markdown**: Use `# %% [markdown]` with bold **Properties:** and bullet lists with `-`
5. **Single plot**: Clean visualization with grid, labels, legends, reference lines
6. **Motivation in markdown**: Use `# %% [markdown]` with bold **Motivation:**

### Plot Requirements:
- Use clear colors (blue, green, red, purple, orange)
- Include reference lines (axhline/axvline) for key values
- Add grid with `alpha=0.3`
- Proper labels and titles (fontsize=10-13)
- Use `linewidth=2-3` for main curves
- Show special points or regions (e.g., oversold/overbought zones)

##Use markdown cells (`# %% [markdown]`) for explanatory text, formulas, properties, and motivations
- Use code cells (`# %%`) for executable Python code
- Keep inline comments minimal (only brief clarifications like `# Current stock price`)
- Use descriptive variable names (`rsi`, `reciprocal`, `normalized`)
- Keep calculations simple and clear
- No functions or classes - direct execution
- Values for visualization: typically `np.linspace(0.01, 10, 500)`
- Format bullets in markdown with `-` not `•
- Values for visualization: typically `np.linspace(0.01, 10, 500)`

## Example Flow for RSI Formula

```
Setup → 
[markdown] Part 0: Input Parameters →
[markdown] Part 1a: Average Gain (AG) → [code] → [markdown] Properties → [code] Plot → [markdown] Motivation →
[markdown] Part 1b: Average Loss (AL) → [code] → [markdown] Properties → [code] Plot → [markdown] Motivation →
[markdown] Part 1c: RS = AG/AL (Relative Strength) → [code] → [markdown] Properties → [code] Plot → [markdown] Motivation →
[markdown] Part 2a: 1 + RS (Normalization Shift) → [code] → [markdown] Properties → [code] Plot → [markdown] Motivation →
[markdown] Part 2b: 1/(1+RS) (Reciprocal) → [code] → [markdown] Properties → [code] Plot → [markdown] Motivation →
[markdown] Part 2c: 100/(1+RS) (Scale to 0-100) → [code] → [markdown] Properties → [code] Plot → [markdown] Motivation →
[markdown] Part 3: Complete RSI Formula → [code] → [markdown] Properties → [code] Plot → [markdown] Motivation →
[markdown] Overlay → [code] → 
[markdown] Sensitivity → [code] → 
[markdown] 3D plots → [code] → 
[markdown] Related formulas → [code] → 
[markdown] Summary → [code]
```

**Example: Breaking down Black-Scholes d₁:**

Instead of showing `d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)` as one part, break it down:

```
Part 1a: S/K (moneyness ratio) - visualize the ratio
Part 1b: ln(S/K) (log transformation) - show logarithmic scale
Part 1c: σ²/2 (variance adjustment) - plot the constant
Part 1d: (r + σ²/2)T (drift term) - combine and visualize
Part 1e: ln(S/K) + (r + σ²/2)T (numerator) - add components
Part 1f: σ√T (denominator) - volatility scaling
Part 1g: Complete d₁ (final division) - show numerator/denominator
```

## Key Principles

1. **Granularity**: Break down to fundamental operations (one operation per part)
2. **Conciseness**: Maximum information, minimum words
3. **Completeness**: All mathematical properties covered
4. **Visual**: Every transformation visualized
5. **Intuition**: Explain WHY, not just WHAT
6. **Interactive**: Each cell runnable independently
7. **Layered**: Build complexity gradually through small steps

## Output Style

Use markdown cells for all descriptive text
- Keep markdown content concise (1-2 lines max per point)
- Use bullet points (`-`) for lists of properties in markdown
- Print statements for summaries, not plots
- Title format: "Part N: [Name]" in plot titles
- Separate markdown and code with proper cell marker
- Keep comments concise (1-2 lines max)
- Use bullet points for lists of properties
- Print statements for summaries, not plots
- Title format: "Part N: [Name]" in plot titles
- End with design philosophy and key thinking summary

## Formula Input Format

When given a formula like:
```
Formula: [equation]
Where: [variable definitions]
```

Immediately decompose it into mathematical steps and create one cell per step, following the template above.

## Success Criteria

The generated script should:
- ✓ Run cell-by-cell interactively in VS Code
- ✓ Use markdown cells (`# %% [markdown]`) for all explanatory content
- ✓ Explain each transformation completely but concisely  
- ✓ Visualize every part with clean, informative plots
- ✓ Provide mathematical properties (domain, range, shape, etc.) in markdown format
- ✓ Include intuition/motivation for each step in markdown cells
- ✓ Show related formulas in summary form
- ✓ Be as short as possible while remaining complete
- ✓ Enable slow, exploratory learning through interactive execution
- ✓ Render nicely in VS Code with formatted markdown cells
