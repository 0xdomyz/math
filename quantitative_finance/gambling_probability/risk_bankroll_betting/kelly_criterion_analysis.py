"""
Kelly Criterion: execution and visualizations.
Extracted from: kelly_criterion.py
"""

import matplotlib.pyplot as plt
import numpy as np
from kelly_criterion_core import kelly_fraction


def run():
    # Example 1: Simple sports betting scenarios
    print("=== Kelly Criterion in Sports Betting ===\n")

    scenarios = [
        {"name": "Weak edge (51% vs 50%)", "prob": 0.51, "odds": 1.0},
        {"name": "Moderate edge (55% vs 50%)", "prob": 0.55, "odds": 1.0},
        {"name": "Strong edge (60% vs 50%)", "prob": 0.60, "odds": 1.0},
        {"name": "Underdog (65% vs 33% odds)", "prob": 0.65, "odds": 2.0},
    ]

    print(
        f"{'Scenario':<30} {'Prob':<10} {'Kelly %':<12} {'1/2 Kelly':<12} {'1/4 Kelly':<12}"
    )
    print("-" * 80)

    for scenario in scenarios:
        f_kelly = kelly_fraction(scenario["prob"], scenario["odds"])
        print(
            f"{scenario['name']:<30} {scenario['prob']:<10.1%} {f_kelly*100:<11.2f}% {f_kelly/2*100:<11.2f}% {f_kelly/4*100:<11.2f}%"
        )

    # Example 2: Bet size vs edge
    print("\n\n=== Kelly Size Sensitivity to Edge ===\n")

    base_prob = 0.5
    edges = np.array([0.01, 0.02, 0.05, 0.10, 0.15])

    print(f"{'Edge %':<10} {'Win Prob':<12} {'Kelly %':<12} {'Growth per bet':<15}")
    print("-" * 50)

    for edge in edges:
        prob = base_prob + edge / 2
        f = kelly_fraction(prob, 1.0)
        growth = prob * np.log(1 + f) + (1 - prob) * np.log(1 - f)
        print(f"{edge*100:<9.2f}% {prob:<11.1%} {f*100:<11.2f}% {growth*100:<14.3f}%")

    # Example 3: Comparing betting strategies
    print("\n\n=== Strategy Comparison: Bankroll Growth ===\n")

    def simulate_betting(
        initial_bankroll, prob_win, num_bets, strategy_name, strategy_func
    ):
        """Simulate betting strategy and return final bankroll"""
        bankroll = initial_bankroll
        wealth_history = [bankroll]

        for bet in range(num_bets):
            if bankroll <= 0:
                wealth_history.append(0)
                continue

            bet_size = strategy_func(bankroll)

            if np.random.random() < prob_win:
                bankroll += bet_size
            else:
                bankroll -= bet_size

            wealth_history.append(bankroll)

        return np.array(wealth_history)

    initial_bank = 1000
    prob = 0.55
    num_bets_sim = 100
    num_simulations = 1000

    # Different strategies
    def kelly_strategy(bankroll):
        f = kelly_fraction(prob, 1.0)
        return bankroll * f

    def half_kelly_strategy(bankroll):
        f = kelly_fraction(prob, 1.0) / 2
        return bankroll * f

    def quarter_kelly_strategy(bankroll):
        f = kelly_fraction(prob, 1.0) / 4
        return bankroll * f

    def fixed_unit_strategy(bankroll):
        return 50  # Fixed $50 bets

    # Run simulations
    strategies = [
        ("Kelly", kelly_strategy),
        ("1/2 Kelly", half_kelly_strategy),
        ("1/4 Kelly", quarter_kelly_strategy),
        ("Fixed $50", fixed_unit_strategy),
    ]

    results = {}
    for name, strat in strategies:
        final_values = []
        for sim in range(num_simulations):
            wealth = simulate_betting(initial_bank, prob, num_bets_sim, name, strat)
            final_values.append(wealth[-1])

        results[name] = {
            "final_values": final_values,
            "mean": np.mean(final_values),
            "median": np.median(final_values),
            "std": np.std(final_values),
            "ruin_rate": np.sum(np.array(final_values) <= 0) / num_simulations,
        }

    print(f"After {num_bets_sim} bets, {num_simulations} simulations:\n")
    print(
        f"{'Strategy':<15} {'Mean Final':<15} {'Median':<15} {'Std Dev':<15} {'Ruin Rate':<12}"
    )
    print("-" * 70)

    for name in [s[0] for s in strategies]:
        mean_val = results[name]["mean"]
        median_val = results[name]["median"]
        std_val = results[name]["std"]
        ruin_rate = results[name]["ruin_rate"]
        print(
            f"{name:<15} ${mean_val:<14,.0f} ${median_val:<14,.0f} ${std_val:<14,.0f} {ruin_rate:<11.1%}"
        )

    # Example 4: Kelly growth model
    print("\n\n=== Expected Bankroll Growth ===\n")

    initial = 10000
    prob_growth = 0.55
    edge = 0.05

    f_kelly_sim = kelly_fraction(prob_growth, 1.0)
    expected_growth = prob_growth * np.log(1 + f_kelly_sim) + (
        1 - prob_growth
    ) * np.log(1 - f_kelly_sim)

    time_to_double = np.log(2) / expected_growth
    time_to_10x = np.log(10) / expected_growth

    print(f"Starting bankroll: ${initial:,}")
    print(f"Win probability: {prob_growth:.1%}")
    print(f"Kelly fraction: {f_kelly_sim*100:.2f}%")
    print(f"Expected growth per bet: {expected_growth*100:.3f}%")
    print(f"Expected time to double: {time_to_double:.0f} bets")
    print(f"Expected time to 10x: {time_to_10x:.0f} bets")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Kelly vs Edge
    edges_plot = np.linspace(0, 0.20, 100)
    kelly_sizes = []

    for edge_p in edges_plot:
        p_kelly = 0.5 + edge_p / 2
        f = kelly_fraction(p_kelly, 1.0)
        kelly_sizes.append(f)

    axes[0, 0].plot(
        edges_plot * 100, np.array(kelly_sizes) * 100, linewidth=2, color="darkgreen"
    )
    axes[0, 0].fill_between(
        edges_plot * 100, np.array(kelly_sizes) * 100, alpha=0.3, color="green"
    )
    axes[0, 0].set_xlabel("Edge (%)")
    axes[0, 0].set_ylabel("Kelly Fraction (%)")
    axes[0, 0].set_title("Optimal Bet Size vs Edge")
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Distribution of final values
    colors_strat = ["darkgreen", "green", "lightgreen", "gray"]
    labels_strat = [s[0] for s in strategies]

    for label, color in zip(labels_strat, colors_strat):
        axes[0, 1].hist(
            results[label]["final_values"], bins=30, alpha=0.5, label=label, color=color
        )

    axes[0, 1].set_xlabel("Final Bankroll ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Final Wealth Distribution\n({num_simulations} simulations)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis="y")

    # Plot 3: Growth curves
    num_bets_curve = np.arange(0, 300, 1)
    for f_kelly_val, label_frac in [
        (kelly_fraction(prob, 1.0), "Kelly"),
        (kelly_fraction(prob, 1.0) / 2, "1/2 Kelly"),
        (kelly_fraction(prob, 1.0) / 4, "1/4 Kelly"),
    ]:
        expected_wealth = initial_bank * np.exp(
            num_bets_curve
            * (prob * np.log(1 + f_kelly_val) + (1 - prob) * np.log(1 - f_kelly_val))
        )
        axes[1, 0].plot(num_bets_curve, expected_wealth, linewidth=2, label=label_frac)

    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Number of Bets")
    axes[1, 0].set_ylabel("Expected Bankroll (log scale)")
    axes[1, 0].set_title("Expected Growth: Kelly vs Fractional Kelly")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Risk vs Growth (Frontier)
    kelly_fractions_range = np.linspace(0, kelly_fraction(prob, 1.0) * 2, 50)
    growth_rates = []
    ruin_risks = []

    for f in kelly_fractions_range:
        if 0 <= f < 1:
            g = prob * np.log(1 + f) + (1 - prob) * np.log(1 - f)
            growth_rates.append(g * 100)
            # Approximate risk of ruin
            if f > 0:
                ruin_risk = ((1 - prob) / prob) ** 100  # Rough estimate
            else:
                ruin_risk = 0
            ruin_risks.append(ruin_risk)

    axes[1, 1].plot(ruin_risks, growth_rates, "o-", linewidth=2, markersize=4)
    kelly_idx = np.argmax(growth_rates)
    axes[1, 1].scatter(
        ruin_risks[kelly_idx],
        growth_rates[kelly_idx],
        color="red",
        s=200,
        marker="*",
        label="Full Kelly",
        zorder=5,
    )
    axes[1, 1].set_xlabel("Risk of Ruin (approximate)")
    axes[1, 1].set_ylabel("Expected Growth (%)")
    axes[1, 1].set_title("Growth-Risk Frontier")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("kelly_criterion.png", dpi=100, bbox_inches="tight")
    print("\n✓ Visualization saved: kelly_criterion.png")
    plt.show()


if __name__ == "__main__":
    run()
