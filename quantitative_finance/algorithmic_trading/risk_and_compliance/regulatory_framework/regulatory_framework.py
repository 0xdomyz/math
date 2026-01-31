import pandas as pd

# Mock market data: NBBO across multiple exchanges
market_data = pd.DataFrame({
    'exchange': ['NYSE', 'NASDAQ', 'BATS', 'IEX'],
    'bid': [50.00, 49.99, 50.01, 50.00],
    'ask': [50.02, 50.01, 50.03, 50.02],
    'bid_size': [5000, 3000, 2000, 1000],
    'ask_size': [5000, 4000, 2000, 1500],
    'access_fee': [0.0030, 0.0025, 0.0020, 0.0000],  # $/share
    'maker_rebate': [0.0020, 0.0015, 0.0015, 0.0000]
})

print("=" * 80)
print("REGULATION NMS COMPLIANCE CHECKER")
print("=" * 80)
print("\n Market Data:")
print(market_data.to_string(index=False))
print()

# Calculate NBBO (National Best Bid and Offer)
nbbo_bid = market_data['bid'].max()
nbbo_ask = market_data['ask'].min()
nbbo_bid_exchange = market_data.loc[market_data['bid'].idxmax(), 'exchange']
nbbo_ask_exchange = market_data.loc[market_data['ask'].idxmin(), 'exchange']

print(f"NBBO: Best Bid ${nbbo_bid:.2f} ({nbbo_bid_exchange}), Best Ask ${nbbo_ask:.2f} ({nbbo_ask_exchange})")
print(f"NBBO Spread: ${nbbo_ask - nbbo_bid:.4f} ({(nbbo_ask - nbbo_bid) / nbbo_bid * 10000:.1f} bps)")
print()

# Scenario 1: Buy Order (must get best ask or better)
order_side = 'buy'
order_size = 3000
preferred_venue = 'NYSE'  # Payment for order flow destination

print("=" * 80)
print(f"SCENARIO 1: {order_side.upper()} {order_size:,} shares")
print("=" * 80)

# Check if preferred venue offers best price
preferred_ask = market_data.loc[market_data['exchange'] == preferred_venue, 'ask'].values[0]
print(f"Preferred Venue ({preferred_venue}): Ask ${preferred_ask:.2f}")
print(f"NBBO Ask (Best):                   ${nbbo_ask:.2f} ({nbbo_ask_exchange})")

if preferred_ask > nbbo_ask:
    trade_through_amount = (preferred_ask - nbbo_ask) * order_size
    print(f"\n⚠ TRADE-THROUGH VIOLATION (Reg NMS Rule 611)")
    print(f"   Routing to {preferred_venue} at ${preferred_ask:.2f} when {nbbo_ask_exchange} offers ${nbbo_ask:.2f}")
    print(f"   Customer harm: ${trade_through_amount:.2f} ({(preferred_ask - nbbo_ask) / nbbo_ask * 10000:.1f} bps)")
    print(f"\n✓ COMPLIANT ROUTING: Route to {nbbo_ask_exchange} at ${nbbo_ask:.2f}")
else:
    print(f"\n✓ COMPLIANT: {preferred_venue} offers best price (${preferred_ask:.2f} = NBBO ${nbbo_ask:.2f})")

print()

# Scenario 2: Sell Order (must get best bid or better)
order_side = 'sell'
order_size = 4000
preferred_venue = 'BATS'

print("=" * 80)
print(f"SCENARIO 2: {order_side.upper()} {order_size:,} shares")
print("=" * 80)

preferred_bid = market_data.loc[market_data['exchange'] == preferred_venue, 'bid'].values[0]
print(f"Preferred Venue ({preferred_venue}): Bid ${preferred_bid:.2f}")
print(f"NBBO Bid (Best):                   ${nbbo_bid:.2f} ({nbbo_bid_exchange})")

if preferred_bid < nbbo_bid:
    trade_through_amount = (nbbo_bid - preferred_bid) * order_size
    print(f"\n⚠ TRADE-THROUGH VIOLATION (Reg NMS Rule 611)")
    print(f"   Routing to {preferred_venue} at ${preferred_bid:.2f} when {nbbo_bid_exchange} offers ${nbbo_bid:.2f}")
    print(f"   Customer harm: ${trade_through_amount:.2f} ({(nbbo_bid - preferred_bid) / nbbo_bid * 10000:.1f} bps)")
    print(f"\n✓ COMPLIANT ROUTING: Route to {nbbo_bid_exchange} at ${nbbo_bid:.2f}")
else:
    print(f"\n✓ COMPLIANT: {preferred_venue} offers best price (${preferred_bid:.2f} = NBBO ${nbbo_bid:.2f})")

print()

# Scenario 3: Access Fee Analysis (Rule 610)
print("=" * 80)
print("SCENARIO 3: ACCESS FEE COMPLIANCE (Reg NMS Rule 610)")
print("=" * 80)

max_access_fee = 0.0030
violations = market_data[market_data['access_fee'] > max_access_fee]
if not violations.empty:
    print(f"⚠ ACCESS FEE VIOLATIONS (Max allowed: ${max_access_fee:.4f}/share):")
    print(violations[['exchange', 'access_fee']].to_string(index=False))
else:
    print(f"✓ ALL VENUES COMPLIANT: Access fees ≤ ${max_access_fee:.4f}/share")

print()

# Scenario 4: Sub-Penny Quoting (Rule 612)
print("=" * 80)
print("SCENARIO 4: SUB-PENNY QUOTING (Reg NMS Rule 612)")
print("=" * 80)

# Attempt to place limit order at sub-penny price
stock_price = 50.00
limit_order_price_valid = 50.01  # Valid (penny increment)
limit_order_price_invalid = 50.015  # Invalid (sub-penny for stock ≥$1.00)

print(f"Stock Price: ${stock_price:.2f}")
print(f"\nLimit Order 1: ${limit_order_price_valid:.4f}")
if limit_order_price_valid % 0.01 == 0:
    print("  ✓ VALID: Penny increment")
else:
    print("  ⚠ INVALID: Sub-penny increment (rejected)")

print(f"\nLimit Order 2: ${limit_order_price_invalid:.4f}")
if limit_order_price_invalid % 0.01 == 0:
    print("  ✓ VALID: Penny increment")
else:
    print("  ⚠ INVALID: Sub-penny increment (rejected)")

print("\n" + "=" * 80)