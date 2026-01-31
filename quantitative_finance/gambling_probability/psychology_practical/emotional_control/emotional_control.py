"""
Extracted from: emotional_control.md
"""

sessions = ["calm", "tilt", "calm", "overconfident"]
profits = [50, -120, 30, -200]
for s, p in zip(sessions, profits):
    print(s, p)
