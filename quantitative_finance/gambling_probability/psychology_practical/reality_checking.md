# Reality Checking

## 1. Concept Skeleton
**Definition:** Mandatory prompts showing time and money spent during play  
**Purpose:** Interrupt dissociation and reduce impulsive losses  
**Prerequisites:** Responsible gambling principles

## 2. Comparative Framing
| Feature | Example | Effect | Limitation |
|---|---|---|---|
| Time alerts | 60-min popups | Awareness | Can be ignored |
| Spend alerts | $100 loss | Loss awareness | May trigger chase |
| Forced breaks | 10-min lockout | Interrupts streaks | Frustration |

## 3. Examples + Counterexamples
**Simple Example:** A 1-hour alert prompts a player to leave.  
**Failure Case:** Alerts ignored by highly aroused players.  
**Edge Case:** Small alerts can backfire by triggering loss chasing.

## 4. Layer Breakdown
```
Reality Checking:
├─ Display time and losses
├─ Require acknowledgement
├─ Offer break or exit
├─ Log user responses
└─ Reinforce limits
```

## 5. Mini-Project
Compute alert frequency:
```python
session_minutes = 180
alert_interval = 30
alerts = session_minutes // alert_interval
print("Alerts per session:", alerts)
```

## 6. Challenge Round
- Users can dismiss alerts without behavior change.  
- Poorly timed alerts increase frustration.  
- Effectiveness depends on enforcement strength.

## 7. Key References
- [Responsible Gambling Features](https://www.gamblingcommission.gov.uk)
- [NCPG](https://www.ncpg.org/)
- [Problem Gambling Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4541962/)
