# Data review

## Data source

Kaggle: https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data

## General info

### Context
The dataset was created from raw order book messaged acquired by subscribing to the websocket of coinbase.com. The order books where then aggregated to construct snapshots of the limit order book at frequencies of 1 second, 1 minute and 5 minutes.

### Content
The dataset contains roughly 12 days of limit order book data for Bitcoin (BTC), Ethereum (ETH) and Cardano (ADA).

The data contains information for the 15 best bid / ask price levels in the order book.

### Features 

midpoint = the midpoint between the best bid and the best ask
spread = the difference between the best bid and the best ask

**{x} from 0 to 14 (number of levels: 15)**

bids_distance_x = the distance of bid level x from the midprice in %
asks_distance_x = the distance of ask level x from the midprice in %

bids_market_notional_x = volume of market orders at bid level x
bids_limit_notional_x = volume of limit orders at bid level x
bids_cancel_notional_x = volume of canceled orders at bid level x

asks_market_notional_x = volume of market orders at ask level x
asks_limit_notional_x = volume of limit orders at ask level x
asks_cancel_notional_x = volume of canceled orders at ask level x

### EDA notes

