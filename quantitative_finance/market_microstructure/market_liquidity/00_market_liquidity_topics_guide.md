# Market Liquidity Topics Guide

**Complete reference of market liquidity concepts, measurement frameworks, and dimensions with categories, brief descriptions, and sources.**

---

## I. Liquidity Definitions & Fundamentals

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Liquidity Concept** | ✓ liquidity_definitions.md | Ability to trade assets quickly without significant price impact | [Wiki - Market Liquidity](https://en.wikipedia.org/wiki/Market_liquidity) |
| **Three Dimensions of Liquidity** | ✓ liquidity_definitions.md | Tightness (cost), Depth (volume), Resiliency (mean reversion speed) | [Kyle (1985)](https://www.jstor.org/stable/1913210) |
| **Tightness** | ✓ liquidity_definitions.md | Immediate transaction costs; spread between bid and ask prices | [Wiki - Bid-Ask Spread](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread) |
| **Depth** | ✓ liquidity_definitions.md | Volume available at various price levels; ability to trade size | [Wiki - Order Book](https://en.wikipedia.org/wiki/Order_book) |
| **Resiliency** | ✓ liquidity_definitions.md | Speed of mean reversion after order flow shock; price recovery | [Kyle (1985)](https://www.jstor.org/stable/1913210) |
| **Immediacy** | ✓ liquidity_definitions.md | Ability to execute trades quickly without delay | [Wiki - Market Liquidity](https://en.wikipedia.org/wiki/Market_liquidity) |

---

## II. Spread-Based Liquidity Measures

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Quoted Spread** | ✓ liquidity_measures.md | Bid-Ask spread = Ask - Bid; directly observable from order book | [Wiki - Bid-Ask Spread](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread) |
| **Effective Spread** | ✓ liquidity_measures.md | 2×\|Transaction Price - Midpoint\|; actual execution cost | [Huang & Stoll (1997)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=62392) |
| **Realized Spread** | ✓ liquidity_measures.md | Spread accounting for inventory reversion; asymmetric information cost | [Huang & Stoll (1997)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=62392) |
| **Roll Spread Estimator** | ✓ liquidity_measures.md | 2√(-Cov(Δp_t, Δp_{t-1})); estimates spread from return autocorrelation | [Roll (1984)](https://www.jstor.org/stable/2490519) |
| **Relative Spread** | ✓ liquidity_measures.md | (Bid-Ask Spread / Midpoint) × 10,000; normalized in basis points | [Wiki - Basis Points](https://en.wikipedia.org/wiki/Basis_point) |
| **High-Low Spread** | N/A | Daily high - low price; coarse proxy for intraday liquidity | [Khan Academy](https://www.khanacademy.org/economics-finance-domain) |

---

## III. Price Impact & Depth Measures

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Amihud Illiquidity Ratio** | ✓ liquidity_measures.md | ([\|r_t\| / Volume_t]) averaged; price impact per dollar traded | [Amihud (2002)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=297126) |
| **Kyle's Lambda (λ)** | ✓ liquidity_measures.md | Regression: Δp_t = λ × Q_t + ε; price impact per share traded | [Kyle (1985)](https://www.jstor.org/stable/1913210) |
| **Price Impact** | ✓ liquidity_measures.md | Permanent component: inventory cost; temporary component: adverse selection | [Huang & Stoll (1997)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=62392) |
| **Market Depth** | ✓ market_depth_imbalance.md | Volume available at best bid/ask; larger depth = better liquidity | [Wiki - Order Book](https://en.wikipedia.org/wiki/Order_book) |
| **Order Book Depth** | ✓ order_book_depth.md | Shape of volume distribution at multiple price levels | [Wiki - Order Book](https://en.wikipedia.org/wiki/Order_book) |
| **Market Impact Function** | ✓ liquidity_measures.md | Nonlinear impact: larger orders face steeper curves | [Almgren & Chriss (2001)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=208282) |

---

## IV. Order Book Analysis

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Order Book Structure** | ✓ order_book_depth.md | Bid/ask levels with volumes; central to modern market microstructure | [Wiki - Order Book](https://en.wikipedia.org/wiki/Order_book) |
| **Bid-Ask Imbalance** | ✓ market_depth_imbalance.md | (Ask Depth - Bid Depth) / (Ask Depth + Bid Depth); directional pressure | [Chordia & Subrahmanyam (2004)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=571342) |
| **Order Book Imbalance** | ✓ market_depth_imbalance.md | Excess volume on one side; predicts price movements | [Bouchaud et al. (2018)](https://arxiv.org/abs/1011.3725) |
| **Volume-Weighted Average Price (VWAP)** | N/A | Execution benchmark; price weighted by volume throughout day | [Wiki - VWAP](https://en.wikipedia.org/wiki/Volume-weighted_average_price) |
| **Time-Weighted Average Price (TWAP)** | N/A | Simple average price over time periods; simpler than VWAP | [Wiki - TWAP](https://en.wikipedia.org/wiki/Time-weighted_average_price) |
| **Limit Order Book Dynamics** | ✓ order_book_depth.md | Orders added/cancelled; affects available liquidity profile | [Wiki - Order Book](https://en.wikipedia.org/wiki/Order_book) |

---

## V. Trading Dynamics & Execution

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Order Flow** | ✓ liquidity_measures.md | Signed volume (buy = +, sell = -); drives price discovery | [Kyle (1985)](https://www.jstor.org/stable/1913210) |
| **Execution Slippage** | ✓ liquidity_measures.md | Difference between expected price and actual execution price | [Almgren & Chriss (2001)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=208282) |
| **Market Orders vs Limit Orders** | ✓ order_book_depth.md | Market orders: immediate execution; Limit orders: patient, potential no-fill | [Wiki - Order Book](https://en.wikipedia.org/wiki/Order_book) |
| **Partial Fill Rates** | N/A | Percentage of limit order volume filled; liquidity provider risk | [Rosu (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1349300) |
| **Price Reversal** | ✓ liquidity_definitions.md | Return to midpoint after shock; inverse of temporary spread component | [Roll (1984)](https://www.jstor.org/stable/2490519) |

---

## VI. Market Microstructure Theory

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Inventory Theory** | ✓ liquidity_definitions.md | Dealers adjust spreads to manage inventory; tighter spread → higher imbalance | [Ho & Stoll (1981)](https://www.jstor.org/stable/2490519) |
| **Adverse Selection** | ✓ liquidity_definitions.md | Asymmetric information; uninformed traders widen spreads to protect against informed trading | [Akerlof (1970)](https://www.jstor.org/stable/1802996) |
| **Information Asymmetry** | ✓ liquidity_definitions.md | Insiders have better information; affects bid-ask spread and order processing | [Kyle (1985)](https://www.jstor.org/stable/1913210) |
| **Order Processing Cost** | ✓ liquidity_definitions.md | Fees/commissions; component of effective spread | [Wiki - Transaction Cost](https://en.wikipedia.org/wiki/Transaction_cost) |
| **Market Maker Role** | N/A | Provides liquidity; profits from spread; manages inventory risk | [Wiki - Market Maker](https://en.wikipedia.org/wiki/Market_maker) |
| **Dealer Behavior** | ✓ liquidity_definitions.md | Spreads vary with volatility, inventory position, competition | [Ho & Stoll (1981)](https://www.jstor.org/stable/2490519) |

---

## VII. Liquidity Commonality & Systemic Factors

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Liquidity Commonality** | ✓ commonality_in_liquidity.md | Liquidity across assets co-moves; systemic shocks affect all | [Chordia & Subrahmanyam (2004)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=571342) |
| **Time-Varying Liquidity** | ✓ commonality_in_liquidity.md | Liquidity varies with market conditions, volatility, and correlation | [Chordia & Subrahmanyam (2004)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=571342) |
| **Volatility-Liquidity Relationship** | ✓ commonality_in_liquidity.md | Higher volatility typically → worse liquidity; inversely related | [Chordia & Subrahmanyam (2004)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=571342) |
| **Correlation & Liquidity** | ✓ commonality_in_liquidity.md | Increased correlation across assets worsens overall liquidity | [Brunnermeier & Pedersen (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1371320) |
| **Funding Liquidity** | N/A | Ability to obtain funding; constraints reduce asset liquidity provision | [Brunnermeier & Pedersen (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1371320) |
| **Systemic Liquidity Risk** | ✓ commonality_in_liquidity.md | Joint failure of liquidity provision; market-wide drying-up of liquidity | [Brunnermeier & Pedersen (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1371320) |

---

## VIII. Advanced Liquidity Topics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **High-Frequency Trading (HFT) Impact** | N/A | Speed-of-light execution; debate: improves liquidity vs increases fragility | [Brogaard et al. (2018)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1377091) |
| **Market Fragmentation** | N/A | Multiple venues; liquidity split across exchanges; reduces effective depth | [Wiki - Market Fragmentation](https://en.wikipedia.org/wiki/Market_fragmentation) |
| **Liquidity Provision Models** | N/A | Dealer markets vs electronic limit order books vs hybrid | [Wiki - Market Structure](https://en.wikipedia.org/wiki/Market_structure) |
| **Liquidity Crises** | N/A | Sudden liquidity drying-up; correlation breakdown, funding shock | [Brunnermeier & Pedersen (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1371320) |
| **Flash Crash Dynamics** | N/A | Extreme intraday volatility event; liquidity evaporation amplifies moves | [SEC Report 2010](https://www.sec.gov/news/studies/2010/marketevents-20100518.pdf) |

---

## IX. Asset-Specific Liquidity Variations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Equity Liquidity Characteristics** | N/A | Large-cap > mid-cap > small-cap; cross-listing effects | [Wiki - Market Cap](https://en.wikipedia.org/wiki/Market_capitalization) |
| **Bond Market Liquidity** | N/A | Heterogeneous dealer networks; less transparent than equities | [SIFMA](https://www.sifma.org) |
| **Currency Market Liquidity** | N/A | 24-hour trading; major pairs highly liquid; exotics illiquid | [Wiki - Foreign Exchange](https://en.wikipedia.org/wiki/Foreign_exchange_market) |
| **Cryptocurrency Liquidity** | N/A | Highly variable; fragmented across DEXs/CEXs; manipulation risks | [CoinMarketCap](https://coinmarketcap.com/) |
| **Commodity Liquidity** | N/A | Futures > spot; concentrated in standardized contracts | [Wiki - Commodity Market](https://en.wikipedia.org/wiki/Commodity_market) |

---

## X. Regulatory & Market Quality Considerations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Market Quality Metrics** | N/A | Liquidity, volatility, efficiency; regulatory framework goals | [SEC Rules](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company_type=only-reg&match=&filenum=&State=&SIC=&myHID=&search_text=&CIK=&count=100&FromDate=&ToDate=) |
| **Market Impact Regulation** | N/A | Position limits, circuit breakers to prevent dislocations | [Wiki - Circuit Breaker](https://en.wikipedia.org/wiki/Circuit_breaker_(finance)) |
| **Best Execution** | N/A | Regulatory requirement to achieve best available prices/liquidity | [SEC Regulation SHO](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company_type=only-reg&match=&filenum=&State=&SIC=&myHID=&search_text=&CIK=&count=100&FromDate=&ToDate=) |
| **Tick Size & Minimum Spread Requirements** | N/A | Regulatory grid; impacts liquidity provision economics | [Wiki - Tick Size](https://en.wikipedia.org/wiki/Tick_size) |
| **Transparency Requirements** | N/A | Pre/post-trade data; affects information asymmetry and market-making | [MiFID II](https://ec.europa.eu/finance/securities/mifid/) |

---

## XI. Cross-Asset Liquidity Connections

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Spillover Effects** | ✓ commonality_in_liquidity.md | Liquidity shocks in one asset affect others; sector/market correlations | [Chordia & Subrahmanyam (2004)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=571342) |
| **Funding Constraints & Deleveraging** | N/A | Margin calls force asset sales; liquidity cascades across portfolio | [Brunnermeier & Pedersen (2009)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1371320) |
| **Carry Trade Unwind** | N/A | Risk-off sentiment; simultaneous exit from correlated positions | [Wiki - Carry Trade](https://en.wikipedia.org/wiki/Carry_trade) |
| **Index Rebalancing Liquidity Impact** | N/A | Passive fund flows affect constituents; temporal predictability | [Edmans et al. (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1550014) |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **Wiki - Market Liquidity** | https://en.wikipedia.org/wiki/Market_liquidity | Comprehensive overview; definitions, measures, frameworks |
| **Kyle (1985) "Continuous Auctions"** | https://www.jstor.org/stable/1913210 | Foundational model; depth, tightness, resiliency framework |
| **Chordia & Subrahmanyam (2004)** | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=571342 | Liquidity commonality; cross-asset co-movement empirics |
| **Brunnermeier & Pedersen (2009)** | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1371320 | Market liquidity illiquidity model; systemic risk |
| **Huang & Stoll (1997)** | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=62392 | Effective spread decomposition; information cost separation |
| **Almgren & Chriss (2001)** | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=208282 | Market impact models; execution algorithms foundations |

---

## Quick Stats

- **Total Topics Documented**: 60+
- **Workspace Files Created**: 5
- **Categories**: 11
- **Spread-Based Measures**: 6
- **Price Impact Measures**: 4
- **Coverage**: Definitions → Measurement → Microstructure Theory → Advanced Topics

