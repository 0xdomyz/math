import numpy as np
import pandas as pd

# Venue setup
venues = {
    'Exchange_A': {'type': 'lit', 'bid': 50.00, 'ask': 50.02, 'bid_size': 5000, 'ask_size': 5000, 
                   'take_fee': 0.0030, 'make_rebate': 0.0020},
    'Exchange_B': {'type': 'lit', 'bid': 49.99, 'ask': 50.01, 'bid_size': 2000, 'ask_size': 2000, 
                   'take_fee': 0.0015, 'make_rebate': 0.0010},
    'DarkPool_X': {'type': 'dark', 'midpoint': 50.01, 'fill_prob': 0.30, 'max_fill': 3000, 
                   'fee': 0.0005},
    'DarkPool_Y': {'type': 'dark', 'midpoint': 50.01, 'fill_prob': 0.20, 'max_fill': 2000, 
                   'fee': 0.0003},
}

# Order to execute
order_side = 'buy'
order_size = 10000  # shares
urgency = 'moderate'  # Options: 'low' (patient), 'moderate', 'high' (urgent)

# Smart Order Router logic
fills = []
remaining = order_size

print("=" * 70)
print("SMART ORDER ROUTER EXECUTION")
print("=" * 70)
print(f"Order: {order_side.upper()} {order_size:,} shares")
print(f"Urgency Level: {urgency}")
print()

# Step 1: Try dark pools first (if patient or moderate urgency)
if urgency in ['low', 'moderate']:
    print("Step 1: Route to Dark Pools")
    for venue_name, venue in venues.items():
        if venue['type'] == 'dark' and remaining > 0:
            # Simulate fill based on probability
            attempted = min(remaining, venue['max_fill'])
            filled = int(attempted * venue['fill_prob'])
            if filled > 0:
                fill_price = venue['midpoint'] + venue['fee']
                fills.append({
                    'venue': venue_name,
                    'shares': filled,
                    'price': fill_price,
                    'fee': venue['fee'] * filled
                })
                remaining -= filled
                print(f"  {venue_name:15s}: Filled {filled:>6,} shares @ ${fill_price:.4f} (fee ${venue['fee']*filled:.2f})")
    print()

# Step 2: Post passive orders on lit exchanges (if low or moderate urgency)
if urgency in ['low', 'moderate'] and remaining > 0:
    print("Step 2: Post Limit Orders (Maker Rebate Strategy)")
    # Post at best bid+1 tick (slightly aggressive, improve fill probability)
    best_bid = max([v['bid'] for v in venues.values() if v['type'] == 'lit'])
    limit_price = best_bid + 0.01  # 1 cent above bid (likely to fill on uptick)
    
    # Choose venue with best maker rebate
    best_rebate_venue = max(
        [(name, v) for name, v in venues.items() if v['type'] == 'lit'],
        key=lambda x: x[1]['make_rebate']
    )
    venue_name, venue = best_rebate_venue
    
    # Simulate partial fill (assume 40% fill rate for limit orders)
    posted = remaining
    filled = int(posted * 0.40)
    fill_price = limit_price - venue['make_rebate']  # Net price after rebate
    
    fills.append({
        'venue': venue_name + ' (Limit)',
        'shares': filled,
        'price': fill_price,
        'fee': -venue['make_rebate'] * filled  # Negative = rebate earned
    })
    remaining -= filled
    print(f"  {venue_name:15s}: Posted {posted:>6,}, Filled {filled:>6,} @ ${limit_price:.4f} (rebate ${venue['make_rebate']*filled:.2f})")
    print(f"                  Net effective price: ${fill_price:.4f}")
    print()

# Step 3: Aggressively take liquidity if remaining shares (high urgency or fallback)
if remaining > 0:
    print("Step 3: Take Liquidity on Lit Exchanges")
    # Find venue with best ask price and lowest take fee
    lit_venues = [(name, v) for name, v in venues.items() if v['type'] == 'lit']
    best_ask_venue = min(lit_venues, key=lambda x: x[1]['ask'] + x[1]['take_fee'])
    venue_name, venue = best_ask_venue
    
    filled = min(remaining, venue['ask_size'])
    fill_price = venue['ask'] + venue['take_fee']
    
    fills.append({
        'venue': venue_name + ' (Market)',
        'shares': filled,
        'price': fill_price,
        'fee': venue['take_fee'] * filled
    })
    remaining -= filled
    print(f"  {venue_name:15s}: Filled {filled:>6,} @ ${venue['ask']:.4f} (fee ${venue['take_fee']*filled:.2f})")
    print(f"                  Net effective price: ${fill_price:.4f}")
    print()

# Summary
fills_df = pd.DataFrame(fills)
total_filled = fills_df['shares'].sum()
avg_price = (fills_df['shares'] * fills_df['price']).sum() / total_filled
total_fees = fills_df['fee'].sum()
total_cost = (fills_df['shares'] * fills_df['price']).sum()

print("=" * 70)
print("EXECUTION SUMMARY")
print("=" * 70)
print(f"Total Filled:                {total_filled:>12,} shares ({total_filled/order_size:.1%})")
print(f"Average Fill Price:          ${avg_price:>11.4f}")
print(f"Total Fees/Rebates:          ${total_fees:>11.2f}")
print(f"Total Execution Cost:        ${total_cost:>11,.2f}")
print()

# Compare to naive execution (all market orders on single exchange)
naive_venue = venues['Exchange_A']
naive_price = naive_venue['ask'] + naive_venue['take_fee']
naive_cost = order_size * naive_price
cost_savings = naive_cost - total_cost
savings_bps = (cost_savings / (order_size * 50)) * 10000

print(f"Benchmark (Naive Execution):")
print(f"  Single Venue Market Order:   ${naive_cost:>11,.2f}  (${naive_price:.4f}/share)")
print()
print(f"SOR Savings:                   ${cost_savings:>11,.2f}  ({savings_bps:.1f} bps)")
print("=" * 70)