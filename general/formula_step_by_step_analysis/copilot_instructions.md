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

### 2. Formula Decomposition
Break the formula into constituent parts (Part 0, Part 1, Part 2, etc.)
Each part gets its own cell with:

```python
# %% PART N: [Component Name]
# Brief description (1 line)
# Domain: [...], Range: [...], Shape: [...]

[define variables and compute values]

# Properties: [2-3 line explanation of behavior]
# [List key values/interpretations with bullets]

[create single focused plot showing this component]
plt.figure(figsize=(10, 6))
[plotting code]
plt.show()

# Motivation: [1-2 sentences explaining WHY this step exists]
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
# %% Overlay: All Transformations Together
# Show all parts on same plot to see composition

# %% Sensitivity Analysis
# Plot derivative or rate of change

# %% 3D Visualization
# Surface plot showing formula behavior across input space

# %% Related Formulas (Summary)
# Dictionary/table of similar formulas with brief comparisons

# %% Key Properties Summary
# Print statement summarizing all key mathematical properties

# %% Design Intuition
# Print statement explaining overall design philosophy
```

## Content Guidelines

### For Each Component Part:
1. **One-line description**: What this part does mathematically
2. **Properties comment**: Domain, range, shape (1 line)
3. **Behavior explanation**: 2-3 lines with inline comments explaining key values
4. **Single plot**: Clean visualization with grid, labels, legends, reference lines
5. **Motivation**: 1 sentence explaining the purpose

### Plot Requirements:
- Use clear colors (blue, green, red, purple, orange)
- Include reference lines (axhline/axvline) for key values
- Add grid with `alpha=0.3`
- Proper labels and titles (fontsize=10-13)
- Use `linewidth=2-3` for main curves
- Show special points or regions (e.g., oversold/overbought zones)

### Code Style:
- Comments above code blocks, not inline (except brief clarifications)
- Use descriptive variable names (`rsi`, `reciprocal`, `normalized`)
- Keep calculations simple and clear
- No functions or classes - direct execution
- Values for visualization: typically `np.linspace(0.01, 10, 500)`

## Example Flow for RSI Formula

```
Setup → 
Part 0: RS (identity) → 
Part 1: 1+RS (normalization) → 
Part 2: 100/(1+RS) (reciprocal) → 
Part 3: Complete formula → 
Overlay → 
Sensitivity → 
3D plots → 
Related formulas → 
Summary
```

## Key Principles

1. **Conciseness**: Maximum information, minimum words
2. **Completeness**: All mathematical properties covered
3. **Visual**: Every transformation visualized
4. **Intuition**: Explain WHY, not just WHAT
5. **Interactive**: Each cell runnable independently
6. **Progressive**: Build understanding part by part

## Output Style

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
- ✓ Explain each transformation completely but concisely  
- ✓ Visualize every part with clean, informative plots
- ✓ Provide mathematical properties (domain, range, shape, etc.)
- ✓ Include intuition/motivation for each step
- ✓ Show related formulas in summary form
- ✓ Be as short as possible while remaining complete
- ✓ Enable slow, exploratory learning through interactive execution
