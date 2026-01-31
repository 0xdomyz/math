import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats

@dataclass
class ExecutionRecord:
    """Record of single order execution"""
    order_id: str
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    size: int
    client_type: str  # 'retail' or 'institutional'
    venue: str
    executed_price: float
    nbbo_bid: float
    nbbo_ask: float
    nbbo_mid: float
    vwap_benchmark: float
    commission: float
    rebate: float  # negative = payment received
    total_cost_bps: float

class BestExecutionAnalyzer:
    """Analyze order executions for best execution compliance"""
    
    def __init__(self):
        self.executions: list = []
        self.quarterly_reviews = {}
    
    def add_execution(self, record: ExecutionRecord):
        """Add execution record"""
        self.executions.append(record)
    
    def compute_effective_spread(self, exec_record):
        """
        Effective spread: how much inside/outside NBBO was execution?
        ES = |executed price - mid| / mid × 10,000 (bps)
        """
        if exec_record.side == 'buy':
            distance = exec_record.executed_price - exec_record.nbbo_mid
        else:  # sell
            distance = exec_record.nbbo_mid - exec_record.executed_price
        
        effective_spread_bps = (distance / exec_record.nbbo_mid) * 10000
        return effective_spread_bps
    
    def compute_implementation_shortfall(self, exec_record):
        """
        Implementation shortfall: deviation from VWAP benchmark
        IS = (executed price - benchmark) × size × 10,000 (bps)
        
        Positive IS = worse than benchmark (cost)
        Negative IS = better than benchmark (gain)
        """
        if exec_record.side == 'buy':
            price_impact = exec_record.executed_price - exec_record.vwap_benchmark
        else:  # sell
            price_impact = exec_record.vwap_benchmark - exec_record.executed_price
        
        is_bps = (price_impact / exec_record.vwap_benchmark) * 10000
        return is_bps
    
    def compute_all_in_cost(self, exec_record):
        """
        Total cost: spread + commission - rebate (if any)
        """
        # Direct costs
        bid_ask_spread = (exec_record.nbbo_ask - exec_record.nbbo_bid) / exec_record.nbbo_mid * 10000
        commission_bps = (exec_record.commission / exec_record.nbbo_mid) * 10000
        rebate_bps = (abs(exec_record.rebate) / exec_record.nbbo_mid) * 10000  # negative rebate = payment
        
        all_in_cost = exec_record.total_cost_bps  # Simplified: already calculated
        return all_in_cost
    
    def analyze_executions(self):
        """Generate comprehensive execution analysis"""
        if not self.executions:
            print("No executions to analyze")
            return None
        
        results = []
        for record in self.executions:
            es = self.compute_effective_spread(record)
            is_val = self.compute_implementation_shortfall(record)
            all_in = self.compute_all_in_cost(record)
            
            results.append({
                'order_id': record.order_id,
                'timestamp': record.timestamp,
                'side': record.side,
                'size': record.size,
                'client_type': record.client_type,
                'venue': record.venue,
                'executed_price': record.executed_price,
                'nbbo_mid': record.nbbo_mid,
                'vwap': record.vwap_benchmark,
                'effective_spread_bps': es,
                'implementation_shortfall_bps': is_val,
                'all_in_cost_bps': all_in,
                'commission_bps': (record.commission / record.nbbo_mid) * 10000,
                'rebate_bps': (abs(record.rebate) / record.nbbo_mid) * 10000
            })
        
        return pd.DataFrame(results)
    
    def quarterly_review(self, quarter_start, quarter_end):
        """
        FINRA Rule 5310-04: Quarterly Best Execution Review
        Compare executions vs benchmarks, identify outliers
        """
        df = self.analyze_executions()
        
        # Filter for quarter
        mask = (df['timestamp'] >= quarter_start) & (df['timestamp'] <= quarter_end)
        quarter_df = df[mask].copy()
        
        review = {
            'period': f"{quarter_start.date()} to {quarter_end.date()}",
            'total_orders': len(quarter_df),
            'total_volume': quarter_df['size'].sum(),
            
            # Effective spread analysis (Rule 611 compliance)
            'es_mean_bps': quarter_df['effective_spread_bps'].mean(),
            'es_median_bps': quarter_df['effective_spread_bps'].median(),
            'es_95th_bps': quarter_df['effective_spread_bps'].quantile(0.95),
            'es_outliers': len(quarter_df[quarter_df['effective_spread_bps'] > 5.0]),
            
            # Implementation shortfall (benchmark adherence)
            'is_mean_bps': quarter_df['implementation_shortfall_bps'].mean(),
            'is_std_bps': quarter_df['implementation_shortfall_bps'].std(),
            'is_outliers': len(quarter_df[quarter_df['implementation_shortfall_bps'] > 10.0]),
            
            # All-in cost analysis
            'cost_mean_bps': quarter_df['all_in_cost_bps'].mean(),
            'cost_median_bps': quarter_df['all_in_cost_bps'].median(),
            'cost_95th_bps': quarter_df['all_in_cost_bps'].quantile(0.95),
            
            # Venue analysis
            'venues_used': quarter_df['venue'].unique(),
            'venue_volume_dist': quarter_df.groupby('venue')['size'].sum().to_dict(),
            
            # Client type analysis
            'retail_orders': len(quarter_df[quarter_df['client_type'] == 'retail']),
            'institutional_orders': len(quarter_df[quarter_df['client_type'] == 'institutional']),
            'retail_cost_mean_bps': quarter_df[quarter_df['client_type'] == 'retail']['all_in_cost_bps'].mean(),
            'institutional_cost_mean_bps': quarter_df[quarter_df['client_type'] == 'institutional']['all_in_cost_bps'].mean(),
            
            # Rebate analysis
            'rebate_revenue': quarter_df['rebate_bps'].sum() * quarter_df['size'].sum() / 10000,
            'rebate_routing_orders': len(quarter_df[quarter_df['rebate_bps'] > 0]),
            
            # Outlier analysis (potential violations)
            'outlier_orders': quarter_df[quarter_df['effective_spread_bps'] > 5.0][['order_id', 'effective_spread_bps', 'venue']].to_dict('records')
        }
        
        return review, quarter_df
    
    def compliance_assessment(self, df):
        """
        Assess compliance with best execution standards
        
        Thresholds:
        - Effective Spread: < 2 bps good, 2-5 bps acceptable, > 5 bps violation
        - Implementation Shortfall: < 5 bps good, 5-10 bps acceptable, > 10 bps violation
        - All-in cost: < 3 bps good, 3-7 bps acceptable, > 7 bps violation
        """
        compliance = {
            'es_compliant': len(df[df['effective_spread_bps'] <= 5.0]) / len(df),
            'is_compliant': len(df[df['implementation_shortfall_bps'] <= 10.0]) / len(df),
            'cost_compliant': len(df[df['all_in_cost_bps'] <= 7.0]) / len(df),
            
            'es_violations': len(df[df['effective_spread_bps'] > 5.0]),
            'is_violations': len(df[df['implementation_shortfall_bps'] > 10.0]),
            'cost_violations': len(df[df['all_in_cost_bps'] > 7.0]),
            
            'overall_compliance_pct': min(
                len(df[df['effective_spread_bps'] <= 5.0]),
                len(df[df['implementation_shortfall_bps'] <= 10.0]),
                len(df[df['all_in_cost_bps'] <= 7.0])
            ) / len(df) * 100
        }
        
        return compliance

# Simulate executions
print("="*80)
print("BEST EXECUTION COMPLIANCE ANALYZER")
print("="*80)

analyzer = BestExecutionAnalyzer()

# Simulate 1000 orders over a quarter
np.random.seed(42)
base_price = 100.0

for i in range(1000):
    timestamp = datetime(2024, 1, 1) + timedelta(hours=i/50)  # Spread over ~2 weeks
    
    # NBBO: mid + random noise
    nbbo_mid = base_price + np.random.normal(0, 0.05)
    nbbo_bid = nbbo_mid - np.random.uniform(0.005, 0.015)
    nbbo_ask = nbbo_mid + np.random.uniform(0.005, 0.015)
    
    # VWAP benchmark (for large orders)
    vwap = nbbo_mid + np.random.normal(0, 0.02)
    
    # Order details
    side = np.random.choice(['buy', 'sell'])
    size = np.random.choice([100, 500, 1000, 5000])
    client_type = np.random.choice(['retail', 'institutional'], p=[0.6, 0.4])
    
    # Execution: mostly good, some violations
    if np.random.random() < 0.95:
        # Good execution (95% of orders)
        if side == 'buy':
            executed_price = nbbo_ask + np.random.normal(0, 0.01)
        else:
            executed_price = nbbo_bid - np.random.normal(0, 0.01)
    else:
        # Violations (5% of orders - rebate-driven worse routing)
        if side == 'buy':
            executed_price = nbbo_ask + np.random.uniform(0.01, 0.05)
        else:
            executed_price = nbbo_bid - np.random.uniform(0.01, 0.05)
    
    # Venue routing
    venue = np.random.choice(['NYSE', 'NASDAQ', 'CBOE', 'Dark Pool'], p=[0.3, 0.35, 0.25, 0.1])
    
    # Costs
    commission = 0.001 * size  # $0.01 per 100 shares
    rebate = np.random.choice([0.0003, 0.0002, 0.0001, 0], p=[0.2, 0.3, 0.3, 0.2]) * size  # Maker-taker
    
    bid_ask_spread_amt = nbbo_ask - nbbo_bid
    total_cost_bps = (bid_ask_spread_amt / nbbo_mid + commission / nbbo_mid - rebate / nbbo_mid) * 10000
    
    record = ExecutionRecord(
        order_id=f"ORDER_{i+1:05d}",
        timestamp=timestamp,
        side=side,
        size=size,
        client_type=client_type,
        venue=venue,
        executed_price=executed_price,
        nbbo_bid=nbbo_bid,
        nbbo_ask=nbbo_ask,
        nbbo_mid=nbbo_mid,
        vwap_benchmark=vwap,
        commission=commission,
        rebate=rebate,
        total_cost_bps=total_cost_bps
    )
    
    analyzer.add_execution(record)
    
    # Update base price
    base_price = (nbbo_bid + nbbo_ask) / 2

# Analyze
df = analyzer.analyze_executions()
print(f"\nTotal executions analyzed: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nExecution Quality Summary (all orders):")
print(f"  Effective Spread: {df['effective_spread_bps'].mean():.2f} bps (mean)")
print(f"  Implementation Shortfall: {df['implementation_shortfall_bps'].mean():.2f} bps (mean)")
print(f"  All-In Cost: {df['all_in_cost_bps'].mean():.2f} bps (mean)")

# Compliance assessment
compliance = analyzer.compliance_assessment(df)
print(f"\nCompliance Assessment:")
print(f"  Overall compliance rate: {compliance['overall_compliance_pct']:.1f}%")
print(f"  ES violations (>5 bps): {compliance['es_violations']} orders")
print(f"  IS violations (>10 bps): {compliance['is_violations']} orders")
print(f"  Cost violations (>7 bps): {compliance['cost_violations']} orders")

# Quarterly review
q1_start = datetime(2024, 1, 1)
q1_end = datetime(2024, 1, 15)
review, q1_df = analyzer.quarterly_review(q1_start, q1_end)

print(f"\nQuarterly Review (Q1 2024):")
print(f"  Period: {review['period']}")
print(f"  Total orders: {review['total_orders']}")
print(f"  Total volume: {review['total_volume']:,} shares")
print(f"  Effective Spread (mean): {review['es_mean_bps']:.2f} bps")
print(f"  Implementation Shortfall (mean): {review['is_mean_bps']:.2f} bps")
print(f"  All-In Cost (mean): {review['cost_mean_bps']:.2f} bps")
print(f"  Outlier orders (>5 bps ES): {review['es_outliers']}")

print(f"\nClient Type Analysis:")
print(f"  Retail orders: {review['retail_orders']} (cost: {review['retail_cost_mean_bps']:.2f} bps mean)")
print(f"  Institutional orders: {review['institutional_orders']} (cost: {review['institutional_cost_mean_bps']:.2f} bps mean)")

print(f"\nVenue Distribution:")
for venue, volume in sorted(review['venue_volume_dist'].items(), key=lambda x: x[1], reverse=True):
    pct = volume / review['total_volume'] * 100
    print(f"  {venue}: {volume:,} shares ({pct:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Effective Spread Distribution
axes[0, 0].hist(df['effective_spread_bps'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(5.0, color='red', linestyle='--', label='Violation threshold (5 bps)')
axes[0, 0].set_title('Distribution of Effective Spreads')
axes[0, 0].set_xlabel('Effective Spread (bps)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Implementation Shortfall
axes[0, 1].hist(df['implementation_shortfall_bps'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(10.0, color='red', linestyle='--', label='Violation threshold (10 bps)')
axes[0, 1].set_title('Distribution of Implementation Shortfalls')
axes[0, 1].set_xlabel('IS (bps)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: All-In Cost
axes[0, 2].hist(df['all_in_cost_bps'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 2].axvline(7.0, color='red', linestyle='--', label='Violation threshold (7 bps)')
axes[0, 2].set_title('Distribution of All-In Costs')
axes[0, 2].set_xlabel('Cost (bps)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3, axis='y')

# Plot 4: Retail vs Institutional Costs
retail_costs = df[df['client_type'] == 'retail']['all_in_cost_bps']
inst_costs = df[df['client_type'] == 'institutional']['all_in_cost_bps']
bp = axes[1, 0].boxplot([retail_costs, inst_costs], labels=['Retail', 'Institutional'])
axes[1, 0].set_title('Execution Costs: Retail vs Institutional')
axes[1, 0].set_ylabel('Cost (bps)')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 5: Cost by Venue
venue_costs = df.groupby('venue')['all_in_cost_bps'].agg(['mean', 'std'])
axes[1, 1].bar(venue_costs.index, venue_costs['mean'], yerr=venue_costs['std'], capsize=5, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Execution Cost by Venue')
axes[1, 1].set_ylabel('Mean Cost (bps)')
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: ES vs IS Scatter
scatter = axes[1, 2].scatter(df['effective_spread_bps'], df['implementation_shortfall_bps'], 
                             c=df['all_in_cost_bps'], cmap='RdYlGn_r', alpha=0.6, s=30)
axes[1, 2].set_title('ES vs IS (colored by cost)')
axes[1, 2].set_xlabel('Effective Spread (bps)')
axes[1, 2].set_ylabel('Implementation Shortfall (bps)')
plt.colorbar(scatter, ax=axes[1, 2], label='Cost (bps)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Effective Spread: Rule 611 (trade-through) compliance metric")
print(f"2. Implementation Shortfall: Measures execution vs benchmark (VWAP)")
print(f"3. All-In Cost: Sum of direct costs (spread + commissions - rebates)")
print(f"4. Client segmentation: Retail often pays more (wider spreads)")
print(f"5. Venue analysis: Rebate structures drive routing decisions")
print(f"6. Outliers: Potential best execution violations requiring investigation")
print(f"7. Quarterly reviews: FINRA Rule 5310-04 mandate for compliance")
