# Commission in Baccarat

## 1. Concept Skeleton
**Definition:** Commission is the 5% fee on banker wins to offset banker advantage  
**Purpose:** Calculate true house edge and compare banker vs player bets  
**Prerequisites:** Expected value, probability, payout structure

## 2. Comparative Framing
| Bet | Win Prob | Payout | House Edge |
|---|---:|---:|---:|
| Banker | 50.68% | 0.95:1 | ~1.06% |
| Player | 49.32% | 1:1 | ~1.24% |
| Tie | 9.52% | 8:1 | ~14.36% |

## 3. Examples + Counterexamples
**Simple Example:** Banker wins pays 0.95x; reduces EV to near player bet.  
**Failure Case:** Ignoring commission makes banker look too good.  
**Edge Case:** Commission-free variants alter optimal bet choice.

## 4. Layer Breakdown
```
Commission Analysis:
├─ Determine base win probabilities
├─ Apply commission to banker payouts
├─ Compute EV per bet
├─ Compare banker vs player
└─ Avoid tie due to large edge
```

## 5. Mini-Project
Compute EV per $1 bet:
```python
p_banker = 0.5068
p_player = 0.4932
p_tie = 0.0952

banker_ev = p_banker * 0.95 - (1 - p_banker - p_tie)
player_ev = p_player * 1.0 - (1 - p_player - p_tie)

print("Banker EV:", round(banker_ev, 4))
print("Player EV:", round(player_ev, 4))
```

## 6. Challenge Round
- Side bets often carry large house edges.  
- Commission-free tables may hide worse payout rules.  
- Short-term streaks mislead bet selection.

## 7. Key References
- [Baccarat](https://en.wikipedia.org/wiki/Baccarat)
- [Wizard of Odds Baccarat](https://www.wizardofodds.com/games/baccarat/)
- [Casino Payout Structures](https://www.casinopedia.org/)
