"""
Deep Dive Analysis: RSI (Relative Strength Index) Formula
==========================================================

Formula: RSI = 100 - (100 / (1 + RS))
Where RS = Average Gain / Average Loss

This module provides comprehensive mathematical analysis including:
- Formula decomposition and component analysis
- Mathematical properties (domain, range, behavior)
- Visualization (2D and 3D plots)
- Intuition and motivation
- Related formulas
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class RSIFormulaAnalysis:
    """Comprehensive analysis of the RSI formula."""

    def __init__(self):
        self.formula_name = "Relative Strength Index (RSI)"
        self.formula = "RSI = 100 - (100 / (1 + RS))"

    def print_header(self, title, level=1):
        """Print formatted section headers."""
        if level == 1:
            print("\n" + "=" * 80)
            print(f"{title:^80}")
            print("=" * 80)
        elif level == 2:
            print("\n" + "-" * 80)
            print(f"{title}")
            print("-" * 80)
        else:
            print(f"\n{title}")
            print("~" * len(title))

    # =========================================================================
    # PART 1: FORMULA BREAKDOWN
    # =========================================================================

    def decompose_formula(self):
        """Break down the RSI formula into its constituent parts."""
        self.print_header("PART 1: FORMULA DECOMPOSITION", level=1)

        parts = {
            "Part 0": {
                "name": "RS (Relative Strength)",
                "formula": "RS = Average Gain / Average Loss",
                "description": "The fundamental ratio of gains to losses",
                "role": "Core input representing market momentum",
            },
            "Part 1": {
                "name": "Normalization Factor (1 + RS)",
                "formula": "f₁(RS) = 1 + RS",
                "description": "Shifts RS by 1 to avoid division by zero",
                "role": "Ensures mathematical stability and proper scaling",
            },
            "Part 2": {
                "name": "Reciprocal Transformation",
                "formula": "f₂(RS) = 100 / (1 + RS)",
                "description": "Inverts the normalized RS and scales to 0-100",
                "role": "Converts to percentage-like scale, inverting the relationship",
            },
            "Part 3": {
                "name": "Complementary Transformation",
                "formula": "RSI = 100 - f₂(RS) = 100 - (100 / (1 + RS))",
                "description": "Final subtraction from 100",
                "role": "Flips the scale so high RS → high RSI (intuitive direction)",
            },
        }

        print("\nThe RSI formula can be decomposed into the following parts:\n")
        for key, part in parts.items():
            print(f"{key}: {part['name']}")
            print(f"  Formula: {part['formula']}")
            print(f"  Description: {part['description']}")
            print(f"  Role: {part['role']}\n")

        return parts

    # =========================================================================
    # PART 2: MATHEMATICAL PROPERTIES ANALYSIS
    # =========================================================================

    def analyze_properties(self):
        """Analyze mathematical properties of each part and the whole formula."""
        self.print_header("PART 2: MATHEMATICAL PROPERTIES", level=1)

        properties = {}

        # Part 0: RS
        self.print_header("Part 0: RS (Relative Strength)", level=2)
        properties["RS"] = {
            "domain": "[0, +∞)",
            "range": "[0, +∞)",
            "support": "Non-negative real numbers",
            "continuity": "Continuous everywhere in domain",
            "monotonicity": "Identity function (strictly increasing)",
            "asymptotic_behavior": "RS → +∞ as gains dominate",
            "special_values": {
                "RS = 0": "No gains (all losses)",
                "RS = 1": "Equal gains and losses",
                "RS → ∞": "No losses (all gains)",
            },
            "shape": "Linear (y = x)",
        }
        self._print_properties(properties["RS"])

        # Part 1: 1 + RS
        self.print_header("Part 1: Normalization (1 + RS)", level=2)
        properties["1+RS"] = {
            "domain": "RS ∈ [0, +∞)",
            "range": "[1, +∞)",
            "support": "Real numbers ≥ 1",
            "continuity": "Continuous everywhere",
            "monotonicity": "Strictly increasing",
            "asymptotic_behavior": "Approaches +∞ as RS → +∞",
            "special_values": {
                "RS = 0": "f₁ = 1 (minimum value)",
                "RS = 1": "f₁ = 2 (balanced point)",
                "RS → ∞": "f₁ → +∞",
            },
            "shape": "Linear with y-intercept at 1",
        }
        self._print_properties(properties["1+RS"])

        # Part 2: 100 / (1 + RS)
        self.print_header("Part 2: Reciprocal (100 / (1 + RS))", level=2)
        properties["reciprocal"] = {
            "domain": "RS ∈ [0, +∞)",
            "range": "(0, 100]",
            "support": "Positive real numbers up to 100",
            "continuity": "Continuous everywhere",
            "monotonicity": "Strictly decreasing",
            "asymptotic_behavior": "Approaches 0 as RS → +∞, approaches 100 as RS → 0",
            "special_values": {
                "RS = 0": "f₂ = 100 (maximum)",
                "RS = 1": "f₂ = 50 (midpoint)",
                "RS = 9": "f₂ = 10",
                "RS → ∞": "f₂ → 0 (approaches but never reaches)",
            },
            "shape": "Rectangular hyperbola (inverse relationship)",
            "concavity": "Convex (second derivative > 0)",
            "derivative": "f₂'(RS) = -100 / (1 + RS)²",
        }
        self._print_properties(properties["reciprocal"])

        # Part 3: RSI = 100 - (100 / (1 + RS))
        self.print_header("Part 3: RSI (Complete Formula)", level=2)
        properties["RSI"] = {
            "domain": "RS ∈ [0, +∞)",
            "range": "[0, 100)",
            "support": "Real numbers in [0, 100)",
            "continuity": "Continuous everywhere",
            "monotonicity": "Strictly increasing",
            "asymptotic_behavior": "Approaches 100 as RS → +∞, equals 0 when RS = 0",
            "special_values": {
                "RS = 0": "RSI = 0 (extreme oversold)",
                "RS = 1": "RSI = 50 (neutral)",
                "RS = 9": "RSI = 90 (extreme overbought)",
                "RS → ∞": "RSI → 100 (approaches but never reaches)",
            },
            "shape": "Sigmoid-like (S-shaped curve)",
            "concavity": "Concave (second derivative < 0)",
            "derivative": "RSI'(RS) = 100 / (1 + RS)²",
            "inflection_point": "None (monotonically increasing, always concave)",
            "bounds": "Always between 0 and 100",
            "interpretation": {
                "RSI < 30": "Oversold condition",
                "RSI = 50": "Neutral (balanced gains/losses)",
                "RSI > 70": "Overbought condition",
            },
        }
        self._print_properties(properties["RSI"])

        return properties

    def _print_properties(self, props):
        """Helper to print properties dictionary."""
        for key, value in props.items():
            if isinstance(value, dict):
                print(f"\n  {key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    print(f"    • {k}: {v}")
            else:
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        print()

    # =========================================================================
    # PART 3: VISUALIZATION
    # =========================================================================

    def plot_all_parts(self):
        """Create comprehensive visualizations for all formula parts."""
        self.print_header("PART 3: VISUALIZATION", level=1)

        # Generate RS values
        rs_values = np.linspace(0, 10, 500)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("RSI Formula: Component Analysis", fontsize=16, fontweight="bold")

        # Plot 1: RS (identity function)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(rs_values, rs_values, "b-", linewidth=2, label="RS = x")
        ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="RS = 1")
        ax1.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("RS Input", fontsize=10)
        ax1.set_ylabel("RS Output", fontsize=10)
        ax1.set_title("Part 0: RS (Identity)", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.set_xlim([0, 10])

        # Plot 2: 1 + RS
        ax2 = fig.add_subplot(2, 3, 2)
        normalized = 1 + rs_values
        ax2.plot(rs_values, normalized, "g-", linewidth=2, label="1 + RS")
        ax2.axhline(y=2, color="gray", linestyle="--", alpha=0.5, label="RS = 1 → 2")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("RS", fontsize=10)
        ax2.set_ylabel("1 + RS", fontsize=10)
        ax2.set_title("Part 1: Normalization (1 + RS)", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.set_xlim([0, 10])

        # Plot 3: 100 / (1 + RS)
        ax3 = fig.add_subplot(2, 3, 3)
        reciprocal = 100 / (1 + rs_values)
        ax3.plot(rs_values, reciprocal, "r-", linewidth=2, label="100 / (1 + RS)")
        ax3.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="y = 50")
        ax3.axvline(x=1, color="gray", linestyle="--", alpha=0.5, label="RS = 1")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("RS", fontsize=10)
        ax3.set_ylabel("100 / (1 + RS)", fontsize=10)
        ax3.set_title(
            "Part 2: Reciprocal Transformation", fontsize=12, fontweight="bold"
        )
        ax3.legend()
        ax3.set_xlim([0, 10])
        ax3.set_ylim([0, 105])

        # Plot 4: RSI (Complete Formula)
        ax4 = fig.add_subplot(2, 3, 4)
        rsi = 100 - (100 / (1 + rs_values))
        ax4.plot(rs_values, rsi, "purple", linewidth=3, label="RSI")
        ax4.axhline(y=30, color="red", linestyle="--", alpha=0.5, label="Oversold (30)")
        ax4.axhline(y=50, color="gray", linestyle="-", alpha=0.5, label="Neutral (50)")
        ax4.axhline(
            y=70, color="green", linestyle="--", alpha=0.5, label="Overbought (70)"
        )
        ax4.fill_between(rs_values, 0, 30, alpha=0.1, color="red")
        ax4.fill_between(rs_values, 70, 100, alpha=0.1, color="green")
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel("RS (Relative Strength)", fontsize=10)
        ax4.set_ylabel("RSI Value", fontsize=10)
        ax4.set_title("Part 3: Complete RSI Formula", fontsize=12, fontweight="bold")
        ax4.legend(loc="lower right")
        ax4.set_xlim([0, 10])
        ax4.set_ylim([0, 100])

        # Plot 5: Derivative (Rate of Change)
        ax5 = fig.add_subplot(2, 3, 5)
        derivative = 100 / ((1 + rs_values) ** 2)
        ax5.plot(
            rs_values, derivative, "orange", linewidth=2, label="RSI'(RS) = 100/(1+RS)²"
        )
        ax5.grid(True, alpha=0.3)
        ax5.set_xlabel("RS", fontsize=10)
        ax5.set_ylabel("Rate of Change", fontsize=10)
        ax5.set_title("RSI Derivative (Sensitivity)", fontsize=12, fontweight="bold")
        ax5.legend()
        ax5.set_xlim([0, 10])
        ax5.text(
            5,
            ax5.get_ylim()[1] * 0.8,
            "Sensitivity decreases\nas RS increases",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 6: Comparison of all transformations
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(
            rs_values,
            rs_values / 10,
            "b-",
            alpha=0.6,
            label="RS (scaled)",
            linewidth=1.5,
        )
        ax6.plot(
            rs_values,
            (1 + rs_values) / 11 * 100,
            "g-",
            alpha=0.6,
            label="1+RS (scaled)",
            linewidth=1.5,
        )
        ax6.plot(
            rs_values, reciprocal, "r-", alpha=0.6, label="100/(1+RS)", linewidth=1.5
        )
        ax6.plot(rs_values, rsi, "purple", alpha=0.9, label="RSI", linewidth=2.5)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel("RS", fontsize=10)
        ax6.set_ylabel("Output Value", fontsize=10)
        ax6.set_title("Overlay: All Transformations", fontsize=12, fontweight="bold")
        ax6.legend(loc="right")
        ax6.set_xlim([0, 10])
        ax6.set_ylim([0, 100])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig("rsi_formula_analysis_2d.png", dpi=300, bbox_inches="tight")
        print("\n✓ 2D plots saved as 'rsi_formula_analysis_2d.png'")
        plt.show()

    def plot_3d_analysis(self):
        """Create 3D visualizations showing RSI behavior."""
        self.print_header("3D Visualization: RSI Surface", level=2)

        # Create figure with two 3D subplots
        fig = plt.figure(figsize=(16, 7))
        fig.suptitle("RSI Formula: 3D Analysis", fontsize=16, fontweight="bold")

        # Subplot 1: RSI as function of Average Gain and Average Loss
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")

        avg_gain = np.linspace(0.1, 10, 100)
        avg_loss = np.linspace(0.1, 10, 100)
        AG, AL = np.meshgrid(avg_gain, avg_loss)
        RS_grid = AG / AL
        RSI_grid = 100 - (100 / (1 + RS_grid))

        surf1 = ax1.plot_surface(
            AG, AL, RSI_grid, cmap=cm.viridis, alpha=0.8, antialiased=True
        )

        # Add reference plane at RSI = 50
        ax1.plot_surface(AG, AL, np.ones_like(RSI_grid) * 50, alpha=0.2, color="gray")

        ax1.set_xlabel("Average Gain", fontsize=10)
        ax1.set_ylabel("Average Loss", fontsize=10)
        ax1.set_zlabel("RSI Value", fontsize=10)
        ax1.set_title(
            "RSI as Function of Gains and Losses", fontsize=12, fontweight="bold"
        )
        ax1.view_init(elev=25, azim=45)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

        # Subplot 2: RSI contour with different RS ratios
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        rs_range = np.linspace(0.01, 15, 100)
        time_steps = np.linspace(0, 100, 100)
        RS_mesh, T_mesh = np.meshgrid(rs_range, time_steps)
        RSI_mesh = 100 - (100 / (1 + RS_mesh))

        surf2 = ax2.plot_surface(
            T_mesh, RS_mesh, RSI_mesh, cmap=cm.coolwarm, alpha=0.8, antialiased=True
        )

        # Add reference planes
        ax2.plot_surface(
            T_mesh,
            RS_mesh,
            np.ones_like(RSI_mesh) * 30,
            alpha=0.15,
            color="red",
            label="Oversold",
        )
        ax2.plot_surface(
            T_mesh,
            RS_mesh,
            np.ones_like(RSI_mesh) * 70,
            alpha=0.15,
            color="green",
            label="Overbought",
        )

        ax2.set_xlabel("Time/Observation", fontsize=10)
        ax2.set_ylabel("RS (Relative Strength)", fontsize=10)
        ax2.set_zlabel("RSI Value", fontsize=10)
        ax2.set_title("RSI Evolution with Varying RS", fontsize=12, fontweight="bold")
        ax2.view_init(elev=20, azim=135)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig("rsi_formula_analysis_3d.png", dpi=300, bbox_inches="tight")
        print("✓ 3D plots saved as 'rsi_formula_analysis_3d.png'")
        plt.show()

    # =========================================================================
    # PART 4: INTUITION AND MOTIVATION
    # =========================================================================

    def explain_intuition(self):
        """Provide intuition and motivation for each part and the whole formula."""
        self.print_header("PART 4: INTUITION AND MOTIVATION", level=1)

        intuitions = {
            "Part 0: RS (Relative Strength)": {
                "intuition": """
                RS represents the fundamental concept of momentum - how strong are the 
                gains relative to the losses? It's a pure ratio that directly captures
                market strength.
                
                - RS > 1: Bulls are winning (more/bigger gains than losses)
                - RS = 1: Perfect balance between bulls and bears
                - RS < 1: Bears are winning (more/bigger losses than gains)
                
                This is the raw signal we want to measure.""",
                "motivation": """
                Using a ratio (rather than difference) makes RS scale-invariant. Whether
                you're trading a $1 stock or a $1000 stock, the same RS value indicates
                the same relative momentum. This normalization is crucial for comparing
                different assets.""",
            },
            "Part 1: Normalization (1 + RS)": {
                "intuition": """
                Adding 1 to RS serves multiple purposes:
                
                1. MATHEMATICAL: Prevents division by zero when RS = 0
                2. CENTERING: Creates a natural midpoint at RS = 1 → (1 + 1) = 2
                3. SYMMETRY: Makes the reciprocal transformation more balanced
                
                Without this, the formula would be undefined for RS = 0 (no gains).""",
                "motivation": """
                This simple shift is mathematically elegant. It ensures the function is
                well-defined everywhere while preserving the ordering relationship.
                The "+1" creates a reference point that makes the subsequent transformation
                produce the intuitive 50% midpoint.""",
            },
            "Part 2: Reciprocal (100 / (1 + RS))": {
                "intuition": """
                The reciprocal inverts the relationship and compresses infinite RS values
                into a bounded [0, 100] range. This transformation is KEY to RSI's behavior:
                
                - As RS grows large (strong gains), the output approaches 0
                - As RS approaches 0 (strong losses), the output approaches 100
                
                The "100" numerator scales this to a percentage-like metric.""",
                "motivation": """
                Reciprocal functions have a special property: they're extremely sensitive
                to small changes when RS is small, but become less sensitive as RS grows.
                
                This creates the characteristic "logarithmic-like" spacing that prevents
                extreme values from dominating. A move from RS = 0.1 to 0.2 matters more
                than a move from 10 to 20, which matches trader intuition about momentum
                changes.""",
            },
            "Part 3: RSI (Complete Formula)": {
                "intuition": """
                The final "100 - x" flips everything so that:
                
                HIGH RS → HIGH RSI (overbought)
                LOW RS → LOW RSI (oversold)
                
                This makes RSI intuitive: higher values = stronger buying pressure,
                matching natural trader expectations.
                
                The result is a bounded oscillator [0, 100] that:
                - Never exceeds 100 (impossible to be "more than all gains")
                - Never goes below 0 (impossible to be "more than all losses")
                - Centers around 50 (balanced market)
                - Shows diminishing returns at extremes (hard to reach 0 or 100)""",
                "motivation": """
                J. Welles Wilder designed RSI in 1978 to solve specific problems:
                
                1. BOUNDED: Unlike price, RSI has clear boundaries for comparison
                2. NORMALIZED: Works across different assets and timeframes
                3. MOMENTUM CAPTURE: Measures velocity of price changes, not just price
                4. MEAN REVERTING: Extreme values suggest potential reversals
                5. VISUAL: Easy to identify overbought/oversold on a fixed scale
                
                The mathematical structure creates a sigmoid-like curve that:
                - Is most sensitive near RS = 1 (neutral zone)
                - Becomes less sensitive at extremes (requires stronger moves)
                - Naturally identifies statistical outliers (>70 or <30)
                
                This makes RSI self-normalizing: in any market, extreme readings are
                rare and potentially significant.""",
            },
            "Overall Design Philosophy": {
                "intuition": """
                The entire formula is a composition of simple functions that together
                create sophisticated behavior:
                
                Linear → Reciprocal → Complement
                
                Each transformation serves a purpose:
                1. Ratio: Normalizes by scale
                2. Shift: Stabilizes mathematically
                3. Invert: Bounds and compresses
                4. Flip: Makes directionally intuitive
                
                The genius is in the composition - each simple step creates complex
                behavior when combined.""",
                "motivation": """
                The formula elegantly solves the challenge of converting unbounded,
                noisy price momentum into a statistically meaningful, visually
                interpretable, and actionable signal.
                
                It's an early example of using nonlinear transformations to extract
                signal from noise - a principle now common in machine learning but
                revolutionary in 1978.""",
            },
        }

        for part, content in intuitions.items():
            self.print_header(part, level=2)
            print("INTUITION:")
            print(content["intuition"])
            print("\nMOTIVATION:")
            print(content["motivation"])
            print()

    # =========================================================================
    # PART 5: RELATED FORMULAS
    # =========================================================================

    def show_related_formulas(self):
        """Show related formulas and provide summary comparisons."""
        self.print_header("PART 5: RELATED FORMULAS", level=1)

        related = {
            "Stochastic Oscillator": {
                "formula": "%K = 100 × (Close - Low_n) / (High_n - Low_n)",
                "similarity": "Bounded [0,100] oscillator, momentum indicator",
                "difference": "Uses price position within range, not gain/loss ratio",
                "use_case": "Better for identifying price reversal points",
            },
            "Williams %R": {
                "formula": "%R = -100 × (High_n - Close) / (High_n - Low_n)",
                "similarity": "Bounded oscillator, inverted scale [-100, 0]",
                "difference": "Measures position from high, not momentum ratio",
                "use_case": "Emphasizes resistance levels and tops",
            },
            "Money Flow Index (MFI)": {
                "formula": "MFI = 100 - (100 / (1 + Money Flow Ratio))",
                "similarity": "IDENTICAL mathematical structure to RSI!",
                "difference": "Uses volume-weighted price (money flow) instead of price",
                "use_case": "RSI but with volume confirmation",
            },
            "Commodity Channel Index (CCI)": {
                "formula": "CCI = (Price - SMA) / (0.015 × Mean Deviation)",
                "similarity": "Momentum oscillator measuring deviation from average",
                "difference": "Unbounded, uses standard deviation, centers at 0",
                "use_case": "Identifies cyclical trends and extremes",
            },
            "MACD": {
                "formula": "MACD = EMA_12 - EMA_26, Signal = EMA_9(MACD)",
                "similarity": "Momentum indicator using moving averages",
                "difference": "Unbounded, shows absolute momentum differences",
                "use_case": "Trend following and momentum confirmation",
            },
            "Rate of Change (ROC)": {
                "formula": "ROC = 100 × (Price_t - Price_{t-n}) / Price_{t-n}",
                "similarity": "Measures momentum as percentage change",
                "difference": "Unbounded, direct price comparison, not smoothed",
                "use_case": "Raw momentum, more volatile than RSI",
            },
            "True Strength Index (TSI)": {
                "formula": "TSI = 100 × EMA(EMA(momentum)) / EMA(EMA(|momentum|))",
                "similarity": "Double-smoothed momentum oscillator",
                "difference": "More complex smoothing, different scaling",
                "use_case": "Smoother than RSI, fewer false signals",
            },
            "Logistic Function": {
                "formula": "f(x) = L / (1 + e^(-k(x-x₀)))",
                "similarity": "S-shaped sigmoid curve, bounded output",
                "difference": "Uses exponential, RSI uses reciprocal",
                "use_case": "General sigmoid transformation (ML, statistics)",
                "note": "RSI has similar shape but different math",
            },
            "Normalized Price": {
                "formula": "Norm = (Price - Min) / (Max - Min)",
                "similarity": "Scales to [0, 1] range",
                "difference": "Linear scaling, doesn't measure momentum",
                "use_case": "Feature scaling in ML, not momentum",
            },
            "Alternative RSI Formula": {
                "formula": "RSI = 100 × RS / (1 + RS)",
                "similarity": "MATHEMATICALLY EQUIVALENT to standard RSI!",
                "difference": "Different form, same result",
                "derivation": """
                Start with: RSI = 100 - (100 / (1 + RS))
                           = 100 - 100/(1 + RS)
                           = [100(1 + RS) - 100] / (1 + RS)
                           = [100 + 100·RS - 100] / (1 + RS)
                           = 100·RS / (1 + RS)
                
                This form shows RSI as a direct transformation of RS!""",
                "use_case": "Sometimes computationally more efficient",
            },
        }

        print("\nRelated Formulas and Comparisons:\n")
        for name, info in related.items():
            print(f"{'─' * 80}")
            print(f"{name}")
            print(f"{'─' * 80}")
            print(f"Formula:     {info['formula']}")
            print(f"Similarity:  {info['similarity']}")
            print(f"Difference:  {info['difference']}")
            print(f"Use Case:    {info['use_case']}")
            if "note" in info:
                print(f"Note:        {info['note']}")
            if "derivation" in info:
                print(f"\nDerivation:\n{info['derivation']}")
            print()

        # Summary comparison table
        self.print_header("Summary Comparison: Key Momentum Indicators", level=2)
        print(
            """
┌─────────────┬──────────┬───────────┬──────────────┬─────────────────────┐
│ Indicator   │  Range   │   Type    │  Input Data  │   Key Feature       │
├─────────────┼──────────┼───────────┼──────────────┼─────────────────────┤
│ RSI         │ [0, 100] │ Momentum  │ Price change │ Gain/loss ratio     │
│ MFI         │ [0, 100] │ Momentum  │ Price+volume │ RSI with volume     │
│ Stochastic  │ [0, 100] │ Momentum  │ Price range  │ Position in range   │
│ Williams %R │[-100, 0] │ Momentum  │ Price range  │ Distance from high  │
│ CCI         │ Unbounded│ Momentum  │ Price vs MA  │ Deviation measure   │
│ MACD        │ Unbounded│ Trend     │ EMAs         │ Trend + momentum    │
│ ROC         │ Unbounded│ Momentum  │ Price change │ Raw % change        │
│ TSI         │[-100,100]│ Momentum  │ Smoothed chg │ Double smoothing    │
└─────────────┴──────────┴───────────┴──────────────┴─────────────────────┘

Key Insight: RSI and MFI share identical mathematical structure, differing
only in their input (price momentum vs. money flow). This makes MFI essentially
"volume-weighted RSI".
        """
        )

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def run_complete_analysis(self):
        """Run the complete deep dive analysis."""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print(
            "█" + "  RSI FORMULA: COMPREHENSIVE MATHEMATICAL DEEP DIVE".center(78) + "█"
        )
        print("█" + " " * 78 + "█")
        print("█" * 80)

        # Run all analysis components
        self.decompose_formula()
        self.analyze_properties()
        self.explain_intuition()
        self.show_related_formulas()

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_all_parts()
        self.plot_3d_analysis()

        self.print_header("ANALYSIS COMPLETE", level=1)
        print(
            """
The complete RSI formula analysis has been generated, including:

✓ Formula decomposition into 4 distinct parts
✓ Mathematical properties for each component
✓ Visual behavior analysis (2D and 3D plots)
✓ Intuition and motivation for each transformation
✓ Related formulas and comparisons

Files generated:
  • rsi_formula_analysis_2d.png - 2D visualization of all components
  • rsi_formula_analysis_3d.png - 3D surface plots

Key Takeaways:
───────────────
1. RSI transforms unbounded momentum (RS) into a bounded [0,100] oscillator
2. The reciprocal function creates logarithmic-like sensitivity
3. The formula is most sensitive around RS = 1 (neutral point)
4. Mathematical structure ensures extreme values are rare but meaningful
5. Design elegantly balances simplicity with sophisticated behavior

The RSI formula remains one of the most elegant applications of nonlinear
transformation in technical analysis, converting noisy price data into a
statistically meaningful and visually interpretable signal.
        """
        )


# =============================================================================
# EXECUTE ANALYSIS
# =============================================================================

if __name__ == "__main__":
    analyzer = RSIFormulaAnalysis()
    analyzer.run_complete_analysis()
