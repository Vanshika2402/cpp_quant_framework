# Japan Derivatives Pricing and Threshold Framework — C++17

This is a C++17 conversion of the Python notebook framework. It preserves the main workflow:

1. Load `config/config.yaml` and `data/option_chain_japan.csv`.
2. Fetch Yahoo Finance historical/quote data when `curl` is available, or fall back to local historical CSVs.
3. Calculate historical volatility, annualized return, skewness, and kurtosis.
4. Price equity options with Black–Scholes and futures options with Black-76.
5. Calculate implied volatility, binomial-tree price, Monte Carlo price/error, and finite-difference Greeks.
6. Generate mispricing trading signals.
7. Run threshold optimization from 1% to 10%.
8. Write CSV outputs, Markdown report, scenario-analysis CSVs, and SVG plots.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Run

```bash
./build/japan_derivatives --config config/config.yaml
```

For offline operation, provide historical CSVs named like:

```text
data/SoftBank_Group_Corp_historical_prices.csv
```

Each historical CSV must contain at least:

```csv
date,close
2024-01-04,6000.0
2024-01-05,6025.0
```

The C++ implementation intentionally keeps market-data access isolated. For production low-latency trading, replace the Yahoo/curl adapter with your exchange, broker, or internal normalized market-data feed.
