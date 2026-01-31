# Data Visualization

## 6.1 Concept Skeleton
**Definition:** Graphical representation of data revealing patterns, distributions, relationships, anomalies  
**Purpose:** Communicate insights, guide analysis, detect errors, support decision-making  
**Prerequisites:** Data types (categorical, continuous), basic graph literacy

## 6.2 Comparative Framing
| Plot Type | Histogram | Scatter Plot | Box Plot | Bar Chart |
|-----------|-----------|--------------|----------|-----------|
| **Data Type** | 1 continuous | 2 continuous | 1 continuous + groups | 1 categorical |
| **Shows** | Distribution shape | Relationship pattern | Summary + outliers | Category frequencies |
| **Best For** | Normality check | Correlation | Compare groups | Categorical comparisons |

## 6.3 Examples + Counterexamples

**Simple Example:**  
Histogram reveals right-skew in income data, guiding median vs mean choice

**Failure Case:**  
Pie chart with 15 categories → unreadable; use bar chart instead

**Edge Case:**  
Simpson's Paradox: Aggregate scatter shows negative trend, but within-group trends positive

## 6.4 Layer Breakdown
```
Visualization Types:
├─ Univariate (Single Variable):
│   ├─ Histogram:
│   │   ├─ Shows: Frequency distribution
│   │   ├─ Use: Assess normality, skew, modality
│   │   └─ Caution: Bin width affects appearance
│   ├─ Box Plot:
│   │   ├─ Shows: Median, quartiles, outliers
│   │   ├─ Use: Compare groups, detect outliers
│   │   └─ Limitation: Hides distribution shape
│   ├─ Dot Plot:
│   │   ├─ Shows: Individual data points
│   │   └─ Use: Small datasets, exact values
│   └─ Density Plot:
│       ├─ Shows: Smoothed distribution
│       └─ Use: Continuous probability estimate
├─ Bivariate (Two Variables):
│   ├─ Scatter Plot:
│   │   ├─ Shows: Relationship between x and y
│   │   ├─ Use: Correlation, trend detection
│   │   └─ Add: Regression line, confidence bands
│   ├─ Line Plot:
│   │   ├─ Shows: Trends over time/sequence
│   │   └─ Use: Time series, sequential data
│   └─ Heatmap:
│       ├─ Shows: 2D frequency or correlation
│       └─ Use: Matrix visualization
├─ Categorical:
│   ├─ Bar Chart:
│   │   ├─ Shows: Category frequencies
│   │   └─ Orientation: Vertical or horizontal
│   └─ Pie Chart:
│       ├─ Shows: Part-to-whole percentages
│       └─ Caution: Hard to compare angles (avoid if >5 categories)
└─ Design Principles:
    ├─ Clear labels and titles
    ├─ Appropriate scales (avoid distortion)
    ├─ Color for meaning, not decoration
    └─ Minimize chart junk
```

## 6.5 Mini-Project
Create comprehensive visualizations:
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)

# Generate sample data
heights = np.random.normal(170, 10, 200)
weights = 50 + 0.6 * heights + np.random.normal(0, 5, 200)
categories = np.random.choice(['A', 'B', 'C', 'D'], 200)
time_series = np.cumsum(np.random.randn(100))

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# 1. Histogram
ax1 = plt.subplot(3, 3, 1)
ax1.hist(heights, bins=30, alpha=0.7, edgecolor='black')
ax1.set_title('Histogram: Height Distribution')
ax1.set_xlabel('Height (cm)')
ax1.set_ylabel('Frequency')

# 2. Box Plot
ax2 = plt.subplot(3, 3, 2)
data_by_cat = [heights[categories == cat] for cat in ['A', 'B', 'C', 'D']]
ax2.boxplot(data_by_cat, labels=['A', 'B', 'C', 'D'])
ax2.set_title('Box Plot: Heights by Category')
ax2.set_ylabel('Height (cm)')

# 3. Bar Chart
ax3 = plt.subplot(3, 3, 3)
cat_counts = [np.sum(categories == cat) for cat in ['A', 'B', 'C', 'D']]
ax3.bar(['A', 'B', 'C', 'D'], cat_counts, alpha=0.7, edgecolor='black')
ax3.set_title('Bar Chart: Category Frequencies')
ax3.set_ylabel('Count')

# 4. Scatter Plot
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(heights, weights, alpha=0.5)
# Add regression line
z = np.polyfit(heights, weights, 1)
p = np.poly1d(z)
ax4.plot(heights, p(heights), "r--", linewidth=2)
ax4.set_title('Scatter Plot: Height vs Weight')
ax4.set_xlabel('Height (cm)')
ax4.set_ylabel('Weight (kg)')

# 5. Density Plot
ax5 = plt.subplot(3, 3, 5)
density = stats.gaussian_kde(heights)
x_range = np.linspace(heights.min(), heights.max(), 100)
ax5.plot(x_range, density(x_range))
ax5.fill_between(x_range, density(x_range), alpha=0.3)
ax5.set_title('Density Plot: Height')
ax5.set_xlabel('Height (cm)')
ax5.set_ylabel('Density')

# 6. Q-Q Plot
ax6 = plt.subplot(3, 3, 6)
stats.probplot(heights, dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot: Normality Check')

# 7. Line Plot (Time Series)
ax7 = plt.subplot(3, 3, 7)
ax7.plot(time_series)
ax7.set_title('Line Plot: Time Series')
ax7.set_xlabel('Time')
ax7.set_ylabel('Value')
ax7.grid(True, alpha=0.3)

# 8. Violin Plot
ax8 = plt.subplot(3, 3, 8)
parts = ax8.violinplot(data_by_cat, positions=[1, 2, 3, 4], showmeans=True)
ax8.set_xticks([1, 2, 3, 4])
ax8.set_xticklabels(['A', 'B', 'C', 'D'])
ax8.set_title('Violin Plot: Distribution by Category')
ax8.set_ylabel('Height (cm)')

# 9. Heatmap (Correlation)
ax9 = plt.subplot(3, 3, 9)
data_matrix = np.column_stack([heights, weights, np.random.randn(200)])
corr_matrix = np.corrcoef(data_matrix.T)
im = ax9.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax9.set_xticks([0, 1, 2])
ax9.set_yticks([0, 1, 2])
ax9.set_xticklabels(['Height', 'Weight', 'Var3'])
ax9.set_yticklabels(['Height', 'Weight', 'Var3'])
ax9.set_title('Heatmap: Correlation Matrix')
plt.colorbar(im, ax=ax9)

# Add correlation values
for i in range(3):
    for j in range(3):
        ax9.text(j, i, f'{corr_matrix[i, j]:.2f}',
                ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics:")
print(f"  Heights: Mean={np.mean(heights):.2f}, SD={np.std(heights, ddof=1):.2f}")
print(f"  Weights: Mean={np.mean(weights):.2f}, SD={np.std(weights, ddof=1):.2f}")
print(f"  Correlation: {np.corrcoef(heights, weights)[0,1]:.3f}")
```

## 6.6 Challenge Round
When is visualization the wrong choice?
- **Exact values needed**: Tables better for precision (e.g., financial reports)
- **Too much data**: Overplotting obscures patterns (use sampling or aggregation)
- **3D plots**: Often misleading; prefer multiple 2D views
- **Dual y-axes**: Can mislead comparisons; use separate plots
- **Pie charts**: Almost always inferior to bar charts for comparison

## 6.7 Key References
- [Edward Tufte - Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Wikipedia - Statistical Graphics](https://en.wikipedia.org/wiki/Statistical_graphics)
- Thinking: Choose plot based on data type and question; Always check for misleading scales or design

---
**Status:** Essential exploratory tool | **Complements:** All descriptive statistics, EDA, Communication
