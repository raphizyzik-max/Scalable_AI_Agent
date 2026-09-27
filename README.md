# Scalabe_AI_Agent
An AI agent that utilizes the scalable capital CLI to manage a portfolio itself.

Requires Python 3.11+, an installed and authenticated `sc` CLI, and an
`OPENAI_API_KEY` environment variable (or a `.env` file beside `agent.py`).

```sh
python3 -m pip install -r requirements.txt
python3 agent.py AAPL MSFT --analysis-only
python3 agent.py AAPL MSFT
python3 agent.py AAPL MSFT --execute
```

The default mode creates trade previews. `--analysis-only` skips previews;
`--execute` displays each full disclosure and requires a separate `YES` before
submission. Python enforces sizing and score thresholds and refreshes portfolio
limits before confirming. The model only produces research and scores.

Review `agent_config.toml` before use. Its starting policy buys at scores of 86 or
higher, sells holdings below 40, targets 3%/5%/10% positions by score band, caps
individual positions at 10%, and skips orders below EUR 10. The model setting
retains the previous script's value; it must be available to your API account.
Use `--config PATH` to load another configuration. Omitting tickers uses screeners.

Run offline tests with `python3 -m unittest discover -s tests -v`. Tests mock broker,
market-data and model calls; they do not submit trades.
