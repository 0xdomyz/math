# Market Microstructure Topics Guide

**Complete reference of market microstructure theory and practice with categories, brief descriptions, and sources.**

---

## I. Market Structure & Organization

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Order-Driven Markets** | N/A | Electronic order book matching; price discovery via limit orders | [O'Hara: Market Microstructure Theory](https://www.amazon.com/Market-Microstructure-Theory-Maureen-OHara/dp/0631207619) |
| **Quote-Driven Markets** | N/A | Dealer/market maker posts bid-ask spreads; OTC structure | [Harris: Trading and Exchanges](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708) |
| **Hybrid Markets** | N/A | Combines electronic and floor trading; designated market makers | [Wikipedia - Market Microstructure](https://en.wikipedia.org/wiki/Market_microstructure) |
| **Dark Pools** | N/A | Off-exchange venues; minimal pre-trade transparency; block trading | [Rosenblatt Securities](https://www.rosenblatt.com/) |
| **Exchanges & Trading Venues** | N/A | NYSE, NASDAQ, CME; Reg NMS fragmentation | [SEC Market Structure](https://www.sec.gov/marketstructure) |
| **Latency & Co-location** | N/A | Speed advantages; data center proximity; HFT infrastructure | [JPM Market Structure Report](https://www.jpmorgan.com/) |

---

## II. Order Types & Execution

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Market Orders** | N/A | Immediate execution at best available price; price uncertainty | [Harris: Trading and Exchanges](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708) |
| **Limit Orders** | N/A | Price-limited; provides liquidity; execution uncertainty | [Foucault et al: Market Liquidity](https://www.amazon.com/Market-Liquidity-Trading-Regulation-Thierry/dp/0190844469) |
| **Stop Orders** | N/A | Triggered at threshold; becomes market order | [Investopedia - Order Types](https://www.investopedia.com/terms/o/order.asp) |
| **Iceberg Orders** | N/A | Hidden quantity; displays only tip; reduces market impact | [SEC Trading Analysis](https://www.sec.gov/marketstructure) |
| **Time-in-Force** | N/A | IOC (immediate or cancel), FOK (fill or kill), GTC, Day orders | [CME Order Types](https://www.cmegroup.com/) |
| **Pegged Orders** | N/A | Dynamic pricing tied to reference (mid-quote, primary peg) | [FINRA Order Types](https://www.finra.org/) |
| **Smart Order Routing (SOR)** | N/A | Algorithms split orders across venues; minimize cost | [SEC Concept Release](https://www.sec.gov/rules/concept.shtml) |

---

## III. Bid-Ask Spread & Transaction Costs

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Bid-Ask Spread** | N/A | Cost of immediacy; difference between best bid and ask | [Roll (1984)](https://www.jstor.org/stable/2327617) |
| **Effective Spread** | N/A | 2 × |trade price - midpoint|; realized transaction cost | [Bessembinder & Kaufman (1997)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb02744.x) |
| **Realized Spread** | N/A | Captures price reversion; dealer profit component | [Huang & Stoll (1996)](https://www.jstor.org/stable/2329367) |
| **Adverse Selection Costs** | N/A | Trading with informed counterparties; permanent price impact | [Glosten & Milgrom (1985)](https://www.sciencedirect.com/science/article/abs/pii/0304405X85900443) |
| **Inventory Costs** | N/A | Dealer risk from holding positions; rebalancing costs | [Stoll (1978)](https://www.jstor.org/stable/2327007) |
| **Order Processing Costs** | N/A | Fixed overhead; clearing, settlement, systems | [Stoll (1989)](https://www.jstor.org/stable/2352946) |
| **Tick Size** | N/A | Minimum price increment; impacts spread, price discovery | [SEC Tick Size Pilot](https://www.sec.gov/rules/other/2015/34-74892.pdf) |

---

## IV. Market Liquidity

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Liquidity Definitions** | N/A | Tightness (spread), depth (quantity), resilience (recovery) | [Kyle (1985)](https://www.jstor.org/stable/1913210) |
| **Order Book Depth** | N/A | Volume available at price levels; absorbs market orders | [Parlour & Seppi (2008)](https://www.annualreviews.org/doi/abs/10.1146/annurev.financial.8.082507.104702) |
| **Market Depth Imbalance** | N/A | Asymmetric depth signals directional pressure | [Cao et al (2009)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2009.01469.x) |
| **Liquidity Measures** | N/A | Amihud illiquidity, Roll spread estimator, effective spread | [Amihud (2002)](https://www.sciencedirect.com/science/article/abs/pii/S0304405X01000726) |
| **Commonality in Liquidity** | N/A | Correlated liquidity shocks across assets; systematic risk | [Chordia et al (2000)](https://www.jstor.org/stable/222555) |
| **Liquidity Provision** | N/A | Market makers, HFT; profit from bid-ask spread | [Menkveld (2013)](https://www.jstor.org/stable/43303831) |
| **Liquidity Crises** | N/A | Flash crashes; order book depletion; circuit breakers | [Kirilenko et al (2017)](https://www.jstor.org/stable/26652722) |

---

## V. Price Discovery & Information

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Price Discovery Process** | N/A | Incorporation of information into prices; efficient markets | [Hasbrouck (1995)](https://www.jstor.org/stable/2329348) |
| **Information Asymmetry** | N/A | Informed vs uninformed traders; adverse selection | [Glosten & Milgrom (1985)](https://www.sciencedirect.com/science/article/abs/pii/0304405X85900443) |
| **Kyle Model** | N/A | Strategic informed trading; price impact; market depth | [Kyle (1985)](https://www.jstor.org/stable/1913210) |
| **Glosten-Milgrom Model** | N/A | Sequential trade; spread widens with adverse selection | [Glosten & Milgrom (1985)](https://www.sciencedirect.com/science/article/abs/pii/0304405X85900443) |
| **Informed Trading** | N/A | Private information advantage; PIN (probability of informed trading) | [Easley et al (1996)](https://www.jstor.org/stable/2329394) |
| **Order Flow Toxicity** | N/A | VPIN (volume-synchronized PIN); high-frequency measure | [Easley et al (2012)](https://www.jstor.org/stable/41349501) |
| **Market Efficiency** | N/A | Weak, semi-strong, strong forms; speed of adjustment | [Fama (1970)](https://www.jstor.org/stable/2325486) |

---

## VI. Market Impact & Price Dynamics

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Permanent Impact** | N/A | Information revelation; price change persists | [Hasbrouck (1991)](https://www.jstor.org/stable/2328955) |
| **Temporary Impact** | N/A | Inventory/liquidity effect; mean-reverts | [Madhavan et al (1997)](https://www.jstor.org/stable/2329541) |
| **Price Impact Models** | N/A | Square-root law; Almgren-Chriss optimal execution | [Almgren & Chriss (2000)](https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf) |
| **Transaction Cost Analysis (TCA)** | N/A | VWAP, implementation shortfall, arrival price benchmarks | [Kissell & Glantz (2003)](https://www.amazon.com/Optimal-Trading-Strategies-Quantitative-Approaches/dp/0814407242) |
| **Market Impact Decay** | N/A | Relaxation time; transient vs permanent components | [Bouchaud et al (2004)](https://arxiv.org/abs/cond-mat/0406224) |
| **Slippage** | N/A | Difference between expected and executed price | [Investopedia - Slippage](https://www.investopedia.com/terms/s/slippage.asp) |
| **Limit Order Book Dynamics** | N/A | Queue position, order arrival/cancellation rates | [Cont et al (2010)](https://arxiv.org/abs/1003.3796) |

---

## VII. High-Frequency Trading (HFT)

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **HFT Strategies** | N/A | Market making, arbitrage, latency arbitrage, momentum ignition | [Brogaard et al (2014)](https://academic.oup.com/rfs/article-abstract/27/8/2267/1582754) |
| **Latency Arbitrage** | N/A | Speed advantage; stale quotes exploitation | [Budish et al (2015)](https://academic.oup.com/qje/article/130/4/1547/1916146) |
| **Colocation Services** | N/A | Exchange proximity; microsecond advantages | [SEC Market Structure](https://www.sec.gov/marketstructure) |
| **Queue Priority** | N/A | Price-time priority; FIFO matching rules | [Foucault et al (2005)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2005.00796.x) |
| **Maker-Taker Fees** | N/A | Rebates for liquidity provision; fee structures | [Colliard & Foucault (2012)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2012.01769.x) |
| **Spoofing & Layering** | N/A | Illegal order manipulation; phantom liquidity | [Dodd-Frank Act](https://www.cftc.gov/LawRegulation/DoddFrankAct/index.htm) |
| **Flash Crashes** | N/A | Rapid price collapse/recovery; automated trading feedback | [CFTC-SEC Flash Crash Report](https://www.sec.gov/news/studies/2010/marketevents-report.pdf) |

---

## VIII. Optimal Execution & Trading Algorithms

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Optimal Execution Theory** | N/A | Trade-off: market impact vs timing risk | [Almgren & Chriss (2000)](https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf) |
| **VWAP Algorithms** | N/A | Track volume-weighted average price; slice over time | [Kissell & Glantz (2003)](https://www.amazon.com/Optimal-Trading-Strategies-Quantitative-Approaches/dp/0814407242) |
| **TWAP Algorithms** | N/A | Time-weighted; uniform slicing strategy | [Kissell & Glantz (2003)](https://www.amazon.com/Optimal-Trading-Strategies-Quantitative-Approaches/dp/0814407242) |
| **Implementation Shortfall** | N/A | Difference between paper portfolio and executed; opportunity cost | [Perold (1988)](https://www.jstor.org/stable/4479223) |
| **Participation Rate Strategies** | N/A | Target percentage of volume; Percent-of-Volume (POV) | [Almgren (2003)](https://www.math.nyu.edu/~almgren/) |
| **Adaptive Algorithms** | N/A | Dynamic adjustment to market conditions; ML-based | [Cartea et al (2015)](https://www.amazon.com/Algorithmic-High-Frequency-Trading-Mathematics-Finance/dp/1107091144) |
| **Trade Scheduling** | N/A | Intraday volume curves; U-shaped volume patterns | [Admati & Pfleiderer (1988)](https://www.jstor.org/stable/2962302) |

---

## IX. Microstructure Noise & Data

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Microstructure Noise** | N/A | Bid-ask bounce, discretization errors; high-frequency artifacts | [Aït-Sahalia et al (2005)](https://www.jstor.org/stable/1392757) |
| **Bid-Ask Bounce** | N/A | Artificial volatility from spread crossing | [Roll (1984)](https://www.jstor.org/stable/2327617) |
| **Tick Data** | N/A | Trade-by-trade, quote updates; nanosecond timestamps | [TAQ Database](https://www.nyse.com/market-data/historical) |
| **Time & Sales Data** | N/A | Executed trades only; time, price, volume | [NASDAQ TotalView](https://www.nasdaq.com/solutions/nasdaq-totalview) |
| **Level 2 & Level 3 Data** | N/A | Order book depth (L2), full order flow (L3) | [CME Market Data](https://www.cmegroup.com/market-data.html) |
| **Realized Volatility** | N/A | Sum of squared returns; consistent under noise | [Andersen et al (2001)](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00418) |
| **Signature Plot** | N/A | Realized variance vs sampling frequency; optimal sampling | [Zhang et al (2005)](https://www.jstor.org/stable/3647587) |

---

## X. Market Making & Dealer Models

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Market Maker Inventory** | N/A | Risk management; mean-reversion strategies | [Stoll (1978)](https://www.jstor.org/stable/2327007) |
| **Dealer Spread Models** | N/A | Stoll decomposition; adverse selection vs inventory | [Stoll (1989)](https://www.jstor.org/stable/2352946) |
| **Avellaneda-Stoikov Model** | N/A | Stochastic control; optimal bid-ask quotes | [Avellaneda & Stoikov (2008)](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf) |
| **Reservation Price** | N/A | Indifference pricing; inventory-dependent quotes | [Cartea et al (2015)](https://www.amazon.com/Algorithmic-High-Frequency-Trading-Mathematics-Finance/dp/1107091144) |
| **Penny Jumping** | N/A | Queue jumping by one tick; latency race | [Foucault et al (2005)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2005.00796.x) |
| **Quote Stuffing** | N/A | Rapid order submission/cancellation; system overload | [SEC Risk Alert](https://www.sec.gov/marketstructure) |
| **Designated Market Makers** | N/A | NYSE specialists; obligations and privileges | [NYSE DMM](https://www.nyse.com/market-model) |

---

## XI. Fragmentation & Regulation

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Regulation NMS** | N/A | Order protection, access, sub-penny rules; US equity markets | [SEC Reg NMS](https://www.sec.gov/rules/final/34-51808.pdf) |
| **Best Execution** | N/A | Broker duty to achieve favorable terms; NBBO routing | [FINRA Best Execution](https://www.finra.org/rules-guidance/key-topics/best-execution) |
| **National Best Bid & Offer (NBBO)** | N/A | Consolidated best prices across exchanges | [SEC Market Data](https://www.sec.gov/marketstructure/market-data) |
| **Trade-Through Rule** | N/A | Cannot trade at inferior price if protected quote exists | [Reg NMS Rule 611](https://www.sec.gov/rules/final/34-51808.pdf) |
| **Market Access Rule** | N/A | Risk controls; pre-trade checks; SEC Rule 15c3-5 | [SEC Market Access Rule](https://www.sec.gov/rules/final/2010/34-63241.pdf) |
| **Dodd-Frank Act** | N/A | Post-2008 reforms; swaps trading, transparency | [CFTC Dodd-Frank](https://www.cftc.gov/LawRegulation/DoddFrankAct/index.htm) |
| **MiFID II** | N/A | European markets; transparency, unbundling, best execution | [ESMA MiFID II](https://www.esma.europa.eu/policy-rules/mifid-ii-and-mifir) |

---

## XII. Special Topics & Advanced Concepts

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Circuit Breakers** | N/A | Trading halts; limit-up/limit-down mechanisms | [SEC LULD](https://www.sec.gov/rules/sro/nyse/2011/34-64547.pdf) |
| **Volatility Auctions** | N/A | Call auctions at open/close; price discovery mechanisms | [Comerton-Forde et al (2010)](https://www.sciencedirect.com/science/article/abs/pii/S1042443110000131) |
| **Block Trading** | N/A | Large orders; upstairs market; negotiated trades | [Keim & Madhavan (1996)](https://www.jstor.org/stable/2329535) |
| **Cross-Asset Arbitrage** | N/A | ETF arbitrage, futures-spot, index rebalancing | [Hasbrouck (2003)](https://www.jstor.org/stable/1262685) |
| **Payment for Order Flow (PFOF)** | N/A | Retail brokers route to market makers; conflicts of interest | [SEC PFOF Study](https://www.sec.gov/marketstructure) |
| **Intraday Patterns** | N/A | U-shaped volume/volatility; opening/closing effects | [Admati & Pfleiderer (1988)](https://www.jstor.org/stable/2962302) |
| **Pre-Open & After-Hours Trading** | N/A | Extended hours; reduced liquidity; wider spreads | [Barclay & Hendershott (2003)](https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1540-6261.2003.00609.x) |
| **Minimum Variance Hedging** | N/A | Reduce execution risk; delta hedging during trades | [Gatheral & Schied (2013)](https://arxiv.org/abs/1011.5882) |
| **Herding & Crowding** | N/A | Correlated strategies; liquidity evaporation | [Bikhchandani & Sharma (2001)](https://www.imf.org/external/pubs/ft/staffp/2000/00-00/pdf/bikhchan.pdf) |

---

## Key Textbooks & Resources

1. **Maureen O'Hara** - *Market Microstructure Theory* (Blackwell, 1995)
2. **Larry Harris** - *Trading and Exchanges* (Oxford, 2003)
3. **Thierry Foucault et al** - *Market Liquidity: Theory, Evidence, and Policy* (Oxford, 2013)
4. **Álvaro Cartea et al** - *Algorithmic and High-Frequency Trading* (Cambridge, 2015)
5. **Joel Hasbrouck** - *Empirical Market Microstructure* (Oxford, 2007)

---

## Data Sources

- **NYSE TAQ** (Trade and Quote database)
- **NASDAQ TotalView** (Full order book)
- **CME Globex Level 2/3** (Futures order flow)
- **Thomson Reuters Tick History** (Global equities/FX)
- **Bloomberg Terminal** (Real-time & historical microstructure data)
- **Lobster** (Academic limit order book data)

---

**Last Updated:** January 31, 2026
