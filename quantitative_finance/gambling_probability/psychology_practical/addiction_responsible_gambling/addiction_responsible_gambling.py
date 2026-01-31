"""
Extracted from: addiction_responsible_gambling.md
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class GamblingTracker:
    """Track gambling behavior and identify warning signs"""
    
    def __init__(self):
        self.sessions = []
    
    def add_session(self, date, duration_mins, amount_wagered, amount_won, emotions):
        """
        Record gambling session
        emotions: list like ['stressed', 'excited', 'desperate']
        """
        net_result = amount_won - amount_wagered
        self.sessions.append({
            'date': date,
            'duration': duration_mins,
            'wagered': amount_wagered,
            'won': amount_won,
            'net': net_result,
            'emotions': emotions
        })
    
    def get_summary(self):
        """Calculate summary statistics"""
        if not self.sessions:
            return None
        
        total_wagered = sum(s['wagered'] for s in self.sessions)
        total_won = sum(s['won'] for s in self.sessions)
        net_result = total_won - total_wagered
        avg_duration = np.mean([s['duration'] for s in self.sessions])
        num_sessions = len(self.sessions)
        
        return {
            'total_wagered': total_wagered,
            'total_won': total_won,
            'net_result': net_result,
            'avg_duration': avg_duration,
            'num_sessions': num_sessions,
            'avg_loss_per_session': -net_result / num_sessions if net_result < 0 else 0
        }
    
    def assess_risk(self):
        """
        Assess addiction risk based on DSM-5-like criteria
        Returns risk level: Low, Moderate, High
        """
        risk_score = 0
        
        summary = self.get_summary()
        if not summary:
            return "Unknown", 0
        
        # Frequency (sessions per week)
        if summary['num_sessions'] > 4:
            risk_score += 1  # Daily/frequent gambling
        
        # Duration (avg session length)
        if summary['avg_duration'] > 180:  # 3+ hours
            risk_score += 1
        
        # Financial impact (net loss)
        if summary['net_result'] < -500:
            risk_score += 1
        if summary['net_result'] < -2000:
            risk_score += 1
        
        # Emotional patterns
        negative_emotions = ['stressed', 'desperate', 'anxious', 'depressed']
        emotion_count = sum(1 for s in self.sessions for e in s['emotions'] if e in negative_emotions)
        if emotion_count / len(self.sessions) > 0.5:
            risk_score += 1  # Gambling to escape negative emotions
        
        # Chasing losses (sessions after big loss)
        for i in range(1, len(self.sessions)):
            if self.sessions[i-1]['net'] < -100 and self.sessions[i]['wagered'] > self.sessions[i-1]['wagered'] * 1.5:
                risk_score += 1
                break
        
        # Risk level
        if risk_score >= 4:
            return "High Risk", risk_score
        elif risk_score >= 2:
            return "Moderate Risk", risk_score
        else:
            return "Low Risk", risk_score

def dsm5_screening():
    """
    DSM-5 gambling disorder screening questions
    """
    questions = [
        "Do you need to gamble with increasing amounts to achieve desired excitement?",
        "Are you restless or irritable when attempting to cut down gambling?",
        "Have you made repeated unsuccessful efforts to control/stop gambling?",
        "Are you often preoccupied with gambling (reliving past experiences, planning next venture)?",
        "Do you gamble when feeling distressed (helpless, guilty, anxious, depressed)?",
        "After losing money, do you often return to 'get even' (chasing losses)?",
        "Do you lie to conceal the extent of your gambling?",
        "Have you jeopardized a relationship, job, or educational opportunity due to gambling?",
        "Do you rely on others for money to relieve desperate financial situations caused by gambling?",
    ]
    
    return questions

# Example 1: Self-assessment questionnaire
print("=== DSM-5 Gambling Disorder Screening ===\n")

questions = dsm5_screening()

print("Answer YES to each question that applies to you in the past 12 months:\n")
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

print("\n\nScoring:")
print("0-1 YES: No problem")
print("2-3 YES: Mild problem (seek self-help resources)")
print("4-5 YES: Moderate problem (consider counseling)")
print("6+ YES: Severe problem (seek professional help immediately)")

# Example 2: Behavioral tracking
print("\n\n=== Gambling Behavior Tracking ===\n")

tracker = GamblingTracker()

# Simulate 2 weeks of gambling data
np.random.seed(42)

# Week 1: Moderate, recreational
for day in range(7):
    if np.random.random() < 0.3:  # Gamble 2-3 times per week
        tracker.add_session(
            date=datetime.now() - timedelta(days=14-day),
            duration_mins=np.random.randint(60, 120),
            amount_wagered=np.random.randint(50, 150),
            amount_won=np.random.randint(0, 200),
            emotions=['excited', 'entertained']
        )

# Week 2: Escalating pattern
for day in range(7):
    if np.random.random() < 0.6:  # Increased frequency
        tracker.add_session(
            date=datetime.now() - timedelta(days=7-day),
            duration_mins=np.random.randint(120, 240),  # Longer sessions
            amount_wagered=np.random.randint(200, 500),  # Higher stakes
            amount_won=np.random.randint(0, 300),  # Still losing
            emotions=['stressed', 'desperate', 'anxious']
        )

summary = tracker.get_summary()
risk_level, risk_score = tracker.assess_risk()

print(f"Total sessions: {summary['num_sessions']}")
print(f"Total wagered: ${summary['total_wagered']:,.0f}")
print(f"Total won: ${summary['total_won']:,.0f}")
print(f"Net result: ${summary['net_result']:+,.0f}")
print(f"Average session duration: {summary['avg_duration']:.0f} minutes")
print(f"Average loss per session: ${summary['avg_loss_per_session']:.0f}")
print(f"\nRisk Level: {risk_level} (Score: {risk_score}/6)")

# Example 3: Responsible gambling limits
print("\n\n=== Responsible Gambling Limits ===\n")

monthly_income = 4000
disposable_income = monthly_income * 0.20  # 20% after expenses

print(f"Monthly income: ${monthly_income:,.0f}")
print(f"Disposable income (20%): ${disposable_income:,.0f}\n")

print("Recommended gambling budget:")
print(f"  Maximum monthly: ${disposable_income * 0.05:,.0f} (5% of disposable)")
print(f"  Maximum weekly: ${disposable_income * 0.05 / 4:,.0f}")
print(f"  Maximum per session: ${disposable_income * 0.05 / 8:,.0f}\n")

print("Session limits:")
print("  Time limit: 1-2 hours maximum")
print("  Break frequency: 15 minutes every hour")
print("  Alcohol: Avoid while gambling")
print("  Emotional state: Don't gamble when stressed/depressed")

# Example 4: Loss chasing simulation
print("\n\n=== Danger of Chasing Losses ===\n")

def simulate_chasing(initial_loss=100, chase_attempts=5):
    """Simulate outcome of chasing losses"""
    bankroll = 1000
    total_lost = initial_loss
    bankroll -= initial_loss
    
    print(f"Initial loss: ${initial_loss}")
    print(f"Remaining bankroll: ${bankroll}\n")
    
    for attempt in range(1, chase_attempts + 1):
        bet_amount = total_lost  # Bet to recover all losses
        
        if bankroll < bet_amount:
            print(f"Attempt {attempt}: Insufficient funds. BANKRUPT.")
            break
        
        # 45% chance to win (typical house edge)
        if np.random.random() < 0.45:
            bankroll += bet_amount
            total_lost = 0
            print(f"Attempt {attempt}: WIN! Recovered ${bet_amount}. Bankroll: ${bankroll}")
            break
        else:
            bankroll -= bet_amount
            total_lost += bet_amount
            print(f"Attempt {attempt}: LOSS. Bet: ${bet_amount}, Total lost: ${total_lost}, Bankroll: ${bankroll}")
    
    print(f"\nFinal bankroll: ${bankroll}")
    print(f"Total loss: ${1000 - bankroll}")
    return bankroll

np.random.seed(42)
final = simulate_chasing(initial_loss=100, chase_attempts=5)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Session frequency over time
session_dates = [s['date'] for s in tracker.sessions]
sessions_per_week = [len([s for s in tracker.sessions if s['date'] >= datetime.now() - timedelta(days=7*i+7) 
                          and s['date'] < datetime.now() - timedelta(days=7*i)]) for i in range(2)][::-1]

weeks = ['Week 1', 'Week 2']
axes[0, 0].bar(weeks, sessions_per_week, color=['green', 'red'], alpha=0.7)
axes[0, 0].axhline(3, color='orange', linestyle='--', linewidth=2, label='Warning threshold')
axes[0, 0].set_ylabel('Number of Sessions')
axes[0, 0].set_title('Gambling Frequency (Warning Sign: Increasing)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Cumulative losses
cumulative = np.cumsum([s['net'] for s in tracker.sessions])
session_nums = np.arange(1, len(cumulative) + 1)

axes[0, 1].plot(session_nums, cumulative, 'o-', linewidth=2, color='darkred')
axes[0, 1].fill_between(session_nums, cumulative, 0, where=(cumulative < 0), alpha=0.2, color='red')
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Session Number')
axes[0, 1].set_ylabel('Cumulative Net Result ($)')
axes[0, 1].set_title('Cumulative Losses (Downward Trend = Problem)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Emotional patterns
all_emotions = []
for s in tracker.sessions:
    all_emotions.extend(s['emotions'])

emotion_counts = {}
for emotion in set(all_emotions):
    emotion_counts[emotion] = all_emotions.count(emotion)

emotions_sorted = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
emotions_labels = [e[0] for e in emotions_sorted]
emotions_values = [e[1] for e in emotions_sorted]

colors_emotion = ['red' if e in ['stressed', 'desperate', 'anxious', 'depressed'] else 'green' for e in emotions_labels]
axes[1, 0].barh(emotions_labels, emotions_values, color=colors_emotion, alpha=0.7)
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_title('Emotional States (Red = Warning Signs)')
axes[1, 0].grid(alpha=0.3, axis='x')

# Plot 4: Risk score radar
risk_categories = ['Frequency', 'Duration', 'Financial', 'Emotional', 'Chasing']
risk_values = [
    min(summary['num_sessions'] / 7, 1) * 10,  # Frequency
    min(summary['avg_duration'] / 240, 1) * 10,  # Duration
    min(abs(summary['net_result']) / 2000, 1) * 10,  # Financial
    len([e for e in all_emotions if e in ['stressed', 'desperate', 'anxious']]) / len(all_emotions) * 10,  # Emotional
    0.8 * 10  # Chasing (from assessment)
]

angles = np.linspace(0, 2 * np.pi, len(risk_categories), endpoint=False).tolist()
risk_values += risk_values[:1]
angles += angles[:1]

ax_polar = plt.subplot(224, projection='polar')
ax_polar.plot(angles, risk_values, 'o-', linewidth=2, color='red')
ax_polar.fill(angles, risk_values, alpha=0.25, color='red')
ax_polar.set_xticks(angles[:-1])
ax_polar.set_xticklabels(risk_categories)
ax_polar.set_ylim(0, 10)
ax_polar.set_title('Addiction Risk Profile (Higher = More Risk)', y=1.08)
ax_polar.grid(True)

plt.tight_layout()
plt.show()
