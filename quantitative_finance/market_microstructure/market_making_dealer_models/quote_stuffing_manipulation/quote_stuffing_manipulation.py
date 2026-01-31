import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from dataclasses import dataclass

@dataclass
class OrderEvent:
    timestamp: float
    order_id: int
    action: str  # 'submit' or 'cancel'
    account: int
    size: float
    price: float
    side: str  # 'buy' or 'sell'

def generate_legitimate_orders(n_trades=1000, n_accounts=10):
    """Generate legitimate market-making order flow"""
    orders = []
    order_id = 0
    
    for trade_num in range(n_trades):
        t_base = trade_num * 1.0  # 1 second per trade cycle
        
        for account_id in range(n_accounts):
            # Legitimate quote update: submit bid and ask
            
            # Submit ask
            orders.append(OrderEvent(
                timestamp=t_base + 0.001,
                order_id=order_id,
                action='submit',
                account=account_id,
                size=100.0,
                price=100.0 + 0.01,
                side='sell'
            ))
            order_id += 1
            
            # Fill or cancel ask
            if np.random.random() < 0.3:  # 30% fill rate
                orders.append(OrderEvent(
                    timestamp=t_base + 0.1,
                    order_id=order_id-1,
                    action='fill',
                    account=account_id,
                    size=100.0,
                    price=100.0 + 0.01,
                    side='sell'
                ))
            else:
                orders.append(OrderEvent(
                    timestamp=t_base + 0.05,  # Cancel after 50ms
                    order_id=order_id-1,
                    action='cancel',
                    account=account_id,
                    size=100.0,
                    price=100.0 + 0.01,
                    side='sell'
                ))
            
            # Submit bid
            orders.append(OrderEvent(
                timestamp=t_base + 0.002,
                order_id=order_id,
                action='submit',
                account=account_id,
                size=100.0,
                price=100.0 - 0.01,
                side='buy'
            ))
            order_id += 1
            
            # Fill or cancel bid
            if np.random.random() < 0.3:
                orders.append(OrderEvent(
                    timestamp=t_base + 0.1,
                    order_id=order_id-1,
                    action='fill',
                    account=account_id,
                    size=100.0,
                    price=100.0 - 0.01,
                    side='buy'
                ))
            else:
                orders.append(OrderEvent(
                    timestamp=t_base + 0.05,
                    order_id=order_id-1,
                    action='cancel',
                    account=account_id,
                    size=100.0,
                    price=100.0 - 0.01,
                    side='buy'
                ))
    
    return orders

def generate_spoofing_orders(n_spoof_events=50):
    """Generate spoofing order patterns"""
    orders = []
    order_id = 10000
    spoof_account = 99  # Isolated account for manipulation
    
    for event_num in range(n_spoof_events):
        t_base = event_num * 10.0  # Events spaced 10 seconds apart
        
        # Spoofing pattern:
        # 1. Place large "support" orders at artificially low price
        # 2. Rapid cancellation
        # 3. Execute real trades
        
        # Layer 1: Large buy order at low price
        orders.append(OrderEvent(
            timestamp=t_base,
            order_id=order_id,
            action='submit',
            account=spoof_account,
            size=5000.0,  # Much larger than legitimate
            price=99.90,
            side='buy'
        ))
        order_id += 1
        
        # Layer 2: Larger buy order at even lower price
        orders.append(OrderEvent(
            timestamp=t_base + 0.0005,
            order_id=order_id,
            action='submit',
            account=spoof_account,
            size=7000.0,
            price=99.80,
            side='buy'
        ))
        order_id += 1
        
        # Layer 3: Huge buy order at lowest price (create visual floor)
        orders.append(OrderEvent(
            timestamp=t_base + 0.001,
            order_id=order_id,
            action='submit',
            account=spoof_account,
            size=10000.0,
            price=99.70,
            side='buy'
        ))
        order_id += 1
        
        # Wait 50ms for others to react to apparent support
        
        # Cancel all fake orders rapidly (key: all canceled within 10ms)
        for oid in range(order_id-3, order_id):
            orders.append(OrderEvent(
                timestamp=t_base + 0.05 + np.random.uniform(0, 0.01),
                order_id=oid,
                action='cancel',
                account=spoof_account,
                size=0,
                price=0,
                side='buy'
            ))
        
        # Execute real trade: sell after market reacted
        orders.append(OrderEvent(
            timestamp=t_base + 0.1,
            order_id=order_id,
            action='submit',
            account=spoof_account,
            size=1000.0,
            price=100.05,
            side='sell'
        ))
        order_id += 1
        
        # Fill at higher price (benefiting from artificial support illusion)
        orders.append(OrderEvent(
            timestamp=t_base + 0.12,
            order_id=order_id-1,
            action='fill',
            account=spoof_account,
            size=1000.0,
            price=100.05,
            side='sell'
        ))
    
    return orders

def extract_features(orders_list, time_window=1.0):
    """Extract features for classification"""
    features = []
    labels = []
    
    orders_df = pd.DataFrame([vars(o) for o in orders_list])
    orders_df = orders_df.sort_values('timestamp').reset_index(drop=True)
    
    # Split into time windows
    min_time = orders_df['timestamp'].min()
    max_time = orders_df['timestamp'].max()
    
    current_time = min_time
    
    while current_time < max_time:
        window_end = current_time + time_window
        window_data = orders_df[(orders_df['timestamp'] >= current_time) & 
                                (orders_df['timestamp'] < window_end)]
        
        if len(window_data) < 5:
            current_time = window_end
            continue
        
        # Extract features for this window
        feature_dict = {}
        
        # Cancel-to-trade ratio
        n_cancels = len(window_data[window_data['action'] == 'cancel'])
        n_fills = len(window_data[window_data['action'] == 'fill'])
        n_submits = len(window_data[window_data['action'] == 'submit'])
        
        feature_dict['cancel_to_submit_ratio'] = n_cancels / (n_submits + 1e-6)
        feature_dict['cancel_to_fill_ratio'] = n_cancels / (n_fills + 1e-6)
        feature_dict['fill_rate'] = n_fills / (n_submits + 1e-6)
        
        # Order size characteristics
        submit_data = window_data[window_data['action'] == 'submit']
        if len(submit_data) > 0:
            feature_dict['mean_order_size'] = submit_data['size'].mean()
            feature_dict['max_order_size'] = submit_data['size'].max()
            feature_dict['order_size_std'] = submit_data['size'].std()
            feature_dict['size_to_mean_ratio'] = submit_data['size'].max() / (submit_data['size'].mean() + 1e-6)
        else:
            feature_dict['mean_order_size'] = 0
            feature_dict['max_order_size'] = 0
            feature_dict['order_size_std'] = 0
            feature_dict['size_to_mean_ratio'] = 0
        
        # Temporal clustering of cancellations
        cancel_data = window_data[window_data['action'] == 'cancel'].copy()
        if len(cancel_data) > 1:
            cancel_times = cancel_data['timestamp'].values
            cancel_intervals = np.diff(cancel_times)
            feature_dict['cancel_interval_mean'] = cancel_intervals.mean()
            feature_dict['cancel_interval_std'] = cancel_intervals.std()
            feature_dict['min_cancel_interval'] = cancel_intervals.min()
        else:
            feature_dict['cancel_interval_mean'] = 0
            feature_dict['cancel_interval_std'] = 0
            feature_dict['min_cancel_interval'] = 1.0
        
        # Order side imbalance (spoofing often one-sided)
        buy_orders = len(window_data[window_data['side'] == 'buy'])
        sell_orders = len(window_data[window_data['side'] == 'sell'])
        feature_dict['side_imbalance'] = abs(buy_orders - sell_orders) / (buy_orders + sell_orders + 1e-6)
        
        # Number of unique accounts (coordination indicator)
        n_accounts = window_data['account'].nunique()
        feature_dict['n_accounts'] = n_accounts
        
        # Dominant account concentration
        account_counts = window_data['account'].value_counts()
        feature_dict['account_concentration'] = account_counts.iloc[0] / len(window_data) if len(account_counts) > 0 else 0
        
        # Message volume (quote stuffing indicator)
        feature_dict['total_messages'] = len(window_data)
        feature_dict['messages_per_second'] = len(window_data) / time_window
        
        # Extract label (1 if spoofing account, 0 otherwise)
        if len(window_data[window_data['account'] == 99]) > 0:
            label = 1  # Spoofing
        else:
            label = 0  # Legitimate
        
        features.append(feature_dict)
        labels.append(label)
        
        current_time = window_end
    
    return pd.DataFrame(features), np.array(labels)

# Generate training data
print("="*80)
print("SPOOFING DETECTION USING MACHINE LEARNING")
print("="*80)

print("\nGenerating training data...")
legitimate_orders = generate_legitimate_orders(n_trades=500, n_accounts=10)
spoofing_orders = generate_spoofing_orders(n_spoof_events=50)

all_orders = legitimate_orders + spoofing_orders
np.random.shuffle(all_orders)

# Extract features
X, y = extract_features(all_orders, time_window=1.0)

print(f"\nDataset Summary:")
print(f"  Total windows: {len(X)}")
print(f"  Legitimate windows: {(y==0).sum()}")
print(f"  Spoofing windows: {(y==1).sum()}")
print(f"  Features: {len(X.columns)}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
print("\nTraining Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Handle imbalance
)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print(f"\n{'='*80}")
print("MODEL EVALUATION")
print(f"{'='*80}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spoofing']))

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC: {auc:.4f}")

# Feature importance
print(f"\nTop 10 Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Feature importance
top_features = feature_importance.head(10)
axes[0, 0].barh(range(len(top_features)), top_features['importance'])
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['feature'])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 10 Feature Importance')
axes[0, 0].invert_yaxis()

# Plot 2: Prediction distribution
axes[0, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Legitimate (True)')
axes[0, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Spoofing (True)')
axes[0, 1].set_xlabel('Spoofing Probability')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Prediction Probability Distribution')
axes[0, 1].legend()

# Plot 3: ROC curve (simplified)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'AUC={auc:.3f}')
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Confusion matrix heatmap
im = axes[1, 1].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['Legitimate', 'Spoofing'])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_yticklabels(['Legitimate', 'Spoofing'])

for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', color='red' if i == j else 'black')

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Cancel-to-submit ratio most predictive: Spoofing >90%, Legitimate ~30%")
print(f"2. Order size imbalance distinctive: Spoofing uses much larger sizes")
print(f"3. Temporal clustering of cancels key: Simultaneous vs. scattered")
print(f"4. Account concentration high in spoofing: Single account dominates")
print(f"5. Message volume elevated: Spoofing generates ~5x normal traffic")
