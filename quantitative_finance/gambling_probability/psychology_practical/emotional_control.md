# Emotional Control

## 1. Concept Skeleton
**Definition:** Managing tilt, fear, and overconfidence during gambling sessions  
**Purpose:** Prevent impulsive bets, preserve bankroll, maintain strategy discipline  
**Prerequisites:** Self-awareness, variance understanding

## 2. Comparative Framing
| Emotion | Typical Behavior | Outcome | Countermeasure |
|---|---|---|---|
| Tilt | Chasing losses | Negative EV | Stop-loss + breaks |
| Fear | Under-betting | Missed EV | Pre-set sizing |
| Overconfidence | Over-betting | Blow-ups | Kelly fraction |

## 3. Examples + Counterexamples
**Simple Example:** After two losses, player takes a break and resets.  
**Failure Case:** Doubling bets to recover losses triggers ruin.  
**Edge Case:** Winning streaks can cause risk-seeking distortions.

## 4. Layer Breakdown
```
Emotional Control:
├─ Identify triggers
├─ Pre-commit to rules
├─ Use cooldown breaks
├─ Track emotions vs outcomes
└─ Review sessions objectively
```

## 5. Mini-Project
Track session mood vs results:
```python
sessions = ["calm", "tilt", "calm", "overconfident"]
profits = [50, -120, 30, -200]
for s, p in zip(sessions, profits):
    print(s, p)
```

## 6. Challenge Round
- Emotional bias is hard to detect in real time.  
- High variance masks poor decision quality.  
- Fatigue amplifies tilt risk.

## 7. Key References
- [Tilt (Poker)](https://en.wikipedia.org/wiki/Tilt_(poker))
- [Loss Aversion](https://en.wikipedia.org/wiki/Loss_aversion)
- [Problem Gambling Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4541962/)
