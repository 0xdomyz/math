"""
Expected Value (EV) Analysis: execution and visualizations.
Extracted from: expected_value.py
"""

import matplotlib.pyplot as plt
import numpy as np
from expected_value_math import calculate_ev


def run():
    # Example 1: Common gambling scenarios
    print("=== Expected Value in Common Bets ===\n")

    scenarios = [
        {"name": "Fair coin (50-50)", "prob": 0.50, "payout": 2.0, "bet": 1.0},
        {"name": "Roulette red (18/38)", "prob": 18 / 38, "payout": 2.0, "bet": 1.0},
        {"name": "Blackjack basic strategy", "prob": 0.51, "payout": 1.0, "bet": 1.0},
        {
            "name": "Sports: True 60%, offered 1.95",
            "prob": 0.60,
            "payout": 1.95,
            "bet": 1.0,
        },
        {
            "name": "Underdog: True 30%, offered 3.5",
            "prob": 0.30,
            "payout": 3.5,
            "bet": 1.0,
        },
        {"name": "Keno typical", "prob": 0.25, "payout": 1.50, "bet": 1.0},
    ]

    print(f"{'Bet Type':<35} {'Prob':<10} {'Payout':<10} {'EV ($)':<12} {'EV %':<10}")
    print("-" * 80)

    for scenario in scenarios:
        ev, ev_pct = calculate_ev(scenario["prob"], scenario["payout"], scenario["bet"])
        status = "✓ Positive" if ev > 0 else "✗ Negative"
        print(
            f"{scenario['name']:<35} {scenario['prob']:<10.4f} {scenario['payout']:<10.2f} {ev:+7.4f} {ev_pct:+7.2f}% {status}"
        )

    # Example 2: Finding value in sports betting
    print("\n\n=== Value Betting: Your Edge vs Market ===\n")

    # Your assessment vs market
    bets = [
        {"name": "Team A", "your_prob": 0.58, "market_prob": 0.55},
        {"name": "Team B", "your_prob": 0.45, "market_prob": 0.48},
        {"name": "Team C", "your_prob": 0.50, "market_prob": 0.52},
    ]

    print(
        f"{'Team':<15} {'Your %':<12} {'Market %':<12} {'Difference':<12} {'Value?':<15}"
    )
    print("-" * 65)

    for bet in bets:
        diff = (bet["your_prob"] - bet["market_prob"]) * 100
        value = "VALUE" if bet["your_prob"] > bet["market_prob"] else "AVOID"
        print(
            f"{bet['name']:<15} {bet['your_prob']:<12.1%} {bet['market_prob']:<12.1%} {diff:+7.2f}% {value:<15}"
        )

    # Example 3: EV vs number of bets
    print("\n\n=== Cumulative EV: Diminishing Variance ===\n")

    prob_true = 0.55
    payout_true = 2.0  # Even odds
    bet_unit = 100

    ev_per_bet, ev_pct_per_bet = calculate_ev(prob_true, payout_true, bet_unit)

    print(f"Single bet EV: ${ev_per_bet:.2f} ({ev_pct_per_bet:.2f}%)\n")
    print(f"{'Num Bets':<12} {'Total EV':<15} {'Std Dev':<15} {'EV/SD Ratio':<15}")
    print("-" * 55)

    for n_bets in [1, 10, 100, 1000, 10000]:
        total_ev = ev_per_bet * n_bets
        variance = (
            prob_true * (payout_true - bet_unit) ** 2
            + (1 - prob_true) * (-bet_unit) ** 2
            - ev_per_bet**2
        ) * n_bets
        std_dev = np.sqrt(variance)
        ratio = total_ev / std_dev if std_dev > 0 else 0
        print(f"{n_bets:<12} ${total_ev:<14,.0f} ${std_dev:<14,.0f} {ratio:<15.2f}")

    # Example 4: Break-even analysis
    print("\n\n=== Break-Even: How Many Bets Needed? ===\n")

    # Compare positive and negative EV
    scenarios_breakeven = [
        {"name": "+5% EV", "ev_pct": 0.05},
        {"name": "+2% EV", "ev_pct": 0.02},
        {"name": "+1% EV", "ev_pct": 0.01},
        {"name": "-5% EV (casino)", "ev_pct": -0.05},
    ]

    scenarios_summary = []

    for scenario in scenarios_breakeven:
        ev_pct = scenario["ev_pct"]
        # Kelly approx: bankroll doubling in ln(2) / ev_pct bets
        if ev_pct > 0:
            double_time = np.log(2) / ev_pct
            scenarios_summary.append(
                {
                    "name": scenario["name"],
                    "ev": ev_pct,
                    "double": double_time,
                    "bust": "Never (positive EV)",
                }
            )
        else:
            bust_time = np.log(0.05) / ev_pct  # Time to lose 95%
            scenarios_summary.append(
                {
                    "name": scenario["name"],
                    "ev": ev_pct,
                    "double": "Never",
                    "bust": f"{abs(bust_time):.0f} bets",
                }
            )

    print(f"{'Strategy':<20} {'EV %':<12} {'Time to 2x':<20} {'Time to Bust':<20}")
    print("-" * 72)

    for s in scenarios_summary:
        print(
            f"{s['name']:<20} {s['ev']*100:+6.2f}% {str(s['double']):<20} {str(s['bust']):<20}"
        )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: EV vs Probability
    probs = np.linspace(0.3, 0.7, 100)
    evs_fair = [calculate_ev(p, 2.0, 1.0)[0] for p in probs]
    evs_biased = [calculate_ev(p, 1.95, 1.0)[0] for p in probs]

    axes[0, 0].plot(
        probs * 100, evs_fair, linewidth=2, label="Fair (2.0 payout)", color="green"
    )
    axes[0, 0].plot(
        probs * 100, evs_biased, linewidth=2, label="Biased (1.95 payout)", color="red"
    )
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[0, 0].fill_between(
        probs * 100,
        evs_fair,
        0,
        where=(np.array(evs_fair) > 0),
        alpha=0.2,
        color="green",
    )
    axes[0, 0].fill_between(
        probs * 100,
        evs_fair,
        0,
        where=(np.array(evs_fair) <= 0),
        alpha=0.2,
        color="red",
    )
    axes[0, 0].set_xlabel("Probability of Winning (%)")
    axes[0, 0].set_ylabel("Expected Value ($)")
    axes[0, 0].set_title("EV vs Win Probability")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Casino games comparison
    games = [
        "Fair Coin",
        "Roulette",
        "Blackjack",
        "60% True\n(1.95)",
        "30% True\n(3.5)",
    ]
    ev_values = [
        0,
        18 / 38 * 2 - 1,
        0.51 * 1 - 0.49,
        0.6 * 1.95 - 1,
        0.3 * 3.5 - 1,
    ]
    colors_games = [
        "gray" if e == 0 else "green" if e > 0 else "red" for e in ev_values
    ]

    axes[0, 1].bar(games, ev_values, color=colors_games, alpha=0.7)
    axes[0, 1].axhline(0, color="black", linestyle="-", linewidth=1)
    axes[0, 1].set_ylabel("Expected Value ($)")
    axes[0, 1].set_title("EV Comparison: Common Bets")
    axes[0, 1].grid(alpha=0.3, axis="y")

    # Plot 3: Cumulative EV vs variance
    num_bets_range = np.logspace(0, 4, 50)
    variance = (
        prob_true * (payout_true - bet_unit) ** 2
        + (1 - prob_true) * (-bet_unit) ** 2
        - ev_per_bet**2
    )
    ev_cumulative = ev_per_bet * num_bets_range
    variance_range = variance * num_bets_range
    std_dev_range = np.sqrt(variance_range)

    axes[1, 0].plot(
        num_bets_range,
        ev_cumulative,
        linewidth=2,
        label="Expected Value",
        color="green",
    )
    axes[1, 0].fill_between(
        num_bets_range,
        ev_cumulative - 2 * std_dev_range,
        ev_cumulative + 2 * std_dev_range,
        alpha=0.2,
        color="blue",
        label="±2 SD (95% CI)",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylabel("Cumulative Outcome ($)")
    axes[1, 0].set_xlabel("Number of Bets (log scale)")
    axes[1, 0].set_title("EV Convergence: Law of Large Numbers")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: EV % for different scenarios
    scenarios_plot = [
        "Negative\n(Casino)",
        "Weak\nPositive",
        "Moderate\nPositive",
        "Strong\nPositive",
    ]
    ev_pcts_plot = [-0.05, 0.01, 0.05, 0.10]
    times_to_double = [
        np.log(2) / abs(e) if e != 0 else float("inf") for e in ev_pcts_plot
    ]

    colors_ev = ["red", "yellow", "lightgreen", "darkgreen"]
    axes[1, 1].bar(scenarios_plot, times_to_double, color=colors_ev, alpha=0.7)
    axes[1, 1].set_ylabel("Bets to Double Bankroll")
    axes[1, 1].set_title("Impact of EV % on Bankroll Growth")
    axes[1, 1].set_yscale("log")
    for i, (label, val) in enumerate(zip(scenarios_plot, times_to_double)):
        if val != float("inf"):
            axes[1, 1].text(i, val * 2, f"{int(val):,}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("expected_value_analysis.png", dpi=100, bbox_inches="tight")
    print("\n✓ Visualization saved: expected_value_analysis.png")
    plt.show()


if __name__ == "__main__":
    run()
