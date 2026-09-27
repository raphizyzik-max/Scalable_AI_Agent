"""Scalable Capital portfolio analysis and policy-enforced trade planning CLI.

The model produces research and scores only. Python validates portfolio arithmetic,
score bands, position caps, cash limits, sell eligibility and trade confirmations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:  # Keep calculations and tests importable before dependencies are installed.
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import yfinance as yf
except ImportError:
    yf = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "agent_config.toml"
CENT = Decimal("0.01")
PERCENT = Decimal("100")
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
TICKER_RE = re.compile(r"^[A-Z0-9.^=\-]{1,24}$")
EQUITY_TYPES = {"EQUITY", "SHARE", "STOCK"}


class AgentError(RuntimeError):
    """Base class for user-facing failures."""


class ConfigurationError(AgentError):
    """The TOML configuration is invalid."""


class ScalableCLIError(AgentError):
    """A Scalable CLI command failed or returned an invalid payload."""


class AnalysisError(AgentError):
    """Market-data or OpenAI analysis failed validation."""


@dataclass(frozen=True)
class ScoreBand:
    min_score: int
    max_score: int
    target_position_pct: Decimal


@dataclass(frozen=True)
class Settings:
    model: str
    reasoning_effort: str
    system_prompt: str
    openai_timeout_seconds: int
    sc_command_timeout_seconds: int
    candidate_limit: int
    sell_below_score: int
    buy_min_score: int
    max_single_position_pct: Decimal
    min_order_amount_eur: Decimal
    score_bands: tuple[ScoreBand, ...]


@dataclass
class PortfolioSnapshot:
    cash_balance: Decimal
    available_cash: Decimal
    holdings_value: Decimal
    total_value: Decimal
    holdings: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)

    def model_context(self) -> dict[str, Any]:
        return {
            "cash_balance_eur": money_text(self.cash_balance),
            "available_cash_eur": money_text(self.available_cash),
            "holdings_value_eur": money_text(self.holdings_value),
            "total_portfolio_value_eur": money_text(self.total_value),
        }


@dataclass
class TradePlanItem:
    side: str
    instrument_id: str
    isin: str
    score: int
    amount: Decimal | None = None
    shares: Decimal | None = None
    target_position_pct: Decimal | None = None
    current_position_value: Decimal = Decimal("0")


@dataclass
class TradePlan:
    items: list[TradePlanItem]
    warnings: list[str]


ANALYSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "portfolio_equity_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "portfolio_summary": {"type": "string"},
            "evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "instrument_id": {"type": "string"},
                        "score": {"type": "integer"},
                        "forward_metrics_summary": {"type": "string"},
                        "bull_case": {"type": "string"},
                        "bear_case": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["BUY", "HOLD", "SELL", "PASS"]},
                        "confidence_notes": {"type": "string"},
                    },
                    "required": [
                        "instrument_id", "score", "forward_metrics_summary",
                        "bull_case", "bear_case", "verdict", "confidence_notes",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["portfolio_summary", "evaluations"],
        "additionalProperties": False,
    },
}


def _required_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing or invalid [{key}] configuration section")
    return value


def _config_int(mapping: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{key} must be an integer >= {minimum}")
    return value


def to_decimal(value: Any, field_name: str, *, allow_none: bool = False) -> Decimal | None:
    if value is None and allow_none:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise AgentError(f"{field_name} is not a valid decimal: {value!r}") from exc
    if not number.is_finite():
        raise AgentError(f"{field_name} must be finite")
    return number


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    try:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Cannot load {path}: {exc}") from exc
    agent = _required_mapping(raw, "agent")
    prompts = _required_mapping(agent, "prompts")
    risk = _required_mapping(raw, "risk")
    bands_raw = risk.get("score_bands")
    if not isinstance(bands_raw, list) or not bands_raw:
        raise ConfigurationError("risk.score_bands must contain at least one band")
    bands: list[ScoreBand] = []
    covered_scores: set[int] = set()
    for index, item in enumerate(bands_raw):
        if not isinstance(item, dict):
            raise ConfigurationError(f"risk.score_bands[{index}] must be a table")
        minimum = _config_int(item, "min_score")
        maximum = _config_int(item, "max_score")
        target = to_decimal(item.get("target_position_pct"), "target_position_pct")
        if minimum > maximum or maximum > 100 or target <= 0:
            raise ConfigurationError(f"Invalid score band at index {index}")
        scores = set(range(minimum, maximum + 1))
        if covered_scores.intersection(scores):
            raise ConfigurationError("risk.score_bands cannot overlap")
        covered_scores.update(scores)
        bands.append(ScoreBand(minimum, maximum, target))
    buy_minimum = _config_int(risk, "buy_min_score")
    if covered_scores != set(range(buy_minimum, 101)):
        raise ConfigurationError("risk.score_bands must cover every score from buy_min_score through 100")
    max_position = to_decimal(risk.get("max_single_position_pct"), "max_single_position_pct")
    min_order = to_decimal(risk.get("min_order_amount_eur"), "min_order_amount_eur")
    if not (Decimal("0") < max_position <= PERCENT):
        raise ConfigurationError("max_single_position_pct must be in (0, 100]")
    if min_order <= 0:
        raise ConfigurationError("min_order_amount_eur must be positive")
    if any(band.target_position_pct > max_position for band in bands):
        raise ConfigurationError("A score-band target exceeds max_single_position_pct")
    prompt = prompts.get("system_prompt")
    model = agent.get("model")
    effort = agent.get("reasoning_effort")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigurationError("agent.prompts.system_prompt must be non-empty")
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError("agent.model must be non-empty")
    if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise ConfigurationError("agent.reasoning_effort is invalid")
    sell_below = _config_int(risk, "sell_below_score")
    if not (0 <= sell_below <= buy_minimum <= 100):
        raise ConfigurationError("Score thresholds must satisfy 0 <= sell <= buy <= 100")
    return Settings(
        model=model, reasoning_effort=effort, system_prompt=prompt.strip(),
        openai_timeout_seconds=_config_int(agent, "openai_timeout_seconds", 1),
        sc_command_timeout_seconds=_config_int(agent, "sc_command_timeout_seconds", 1),
        candidate_limit=_config_int(agent, "candidate_limit", 1),
        sell_below_score=sell_below, buy_min_score=buy_minimum,
        max_single_position_pct=max_position, min_order_amount_eur=min_order,
        score_bands=tuple(sorted(bands, key=lambda band: band.min_score)),
    )


def decimal_text(value: Decimal | str | int | float) -> str:
    return format(to_decimal(value, "decimal value"), "f")


def money_text(value: Decimal) -> str:
    return format(value.quantize(CENT, rounding=ROUND_DOWN), ".2f")


def _clean_number(value: Any, *, percentage: bool = False) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if percentage:
        number *= 100
    return round(number, 2) if math.isfinite(number) else None


def _news_title(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = content.get("title")
    return str(title).strip() if title else None


def get_stock_evaluation_data(ticker: str) -> dict[str, Any]:
    """Fetches FORWARD-LOOKING valuation metrics, risk metrics, and context."""
    if yf is None:
        raise AnalysisError("yfinance is not installed; run `pip install -r requirements.txt`")
    symbol = normalize_ticker(ticker)
    result: dict[str, Any] = {"ticker": symbol, "data_complete": False, "errors": []}
    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
        if not isinstance(info, dict):
            info = {}
        current_price = _clean_number(info.get("currentPrice") or info.get("regularMarketPrice"))
        target_price = _clean_number(info.get("targetMeanPrice"))

        upside_pct = None
        if current_price and current_price > 0 and target_price:
            upside_pct = round(((target_price - current_price) / current_price) * 100, 1)

        # Calculate distance from 52-week High (detects parabolic rallies)
        fifty_two_high = _clean_number(info.get("fiftyTwoWeekHigh"))
        pct_from_52w_high = None
        if fifty_two_high and current_price:
            pct_from_52w_high = round(((current_price - fifty_two_high) / fifty_two_high) * 100, 1)

        # Calculate 1-year return history
        stock_1yr_return = None
        try:
            stock_hist = stock.history(period="1y", auto_adjust=False)
            if not stock_hist.empty and len(stock_hist.index) >= 2:
                first = float(stock_hist['Close'].iloc[0])
                last = float(stock_hist['Close'].iloc[-1])
                if first > 0:
                    stock_1yr_return = _clean_number((last - first) / first * 100)
        except Exception as exc:
            result["errors"].append(f"one-year history unavailable: {exc}")

        news = get_stock_news(symbol)
        if "error" in news:
            result["errors"].append(news["error"])
        resolved_isin = None
        try:
            candidate_isin = stock.get_isin()
            if isinstance(candidate_isin, str) and is_valid_isin(candidate_isin):
                resolved_isin = candidate_isin.strip().upper()
        except Exception as exc:
            result["errors"].append(f"ISIN lookup unavailable: {exc}")

        result.update({
            "ticker": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "isin": resolved_isin,
            "quote_type": info.get("quoteType"),
            "currency": info.get("currency"),
            "market_cap": _clean_number(info.get("marketCap")),
            "current_price": current_price,
            "pct_from_52_week_high": pct_from_52w_high,
            "analyst_target_upside_pct": upside_pct,
            "analyst_rating": info.get("recommendationKey"),
            "forward_pe": _clean_number(info.get("forwardPE")),
            "trailing_pe": _clean_number(info.get("trailingPE")),
            "peg_ratio": _clean_number(info.get("pegRatio")),
            "quarterly_revenue_growth_pct": _clean_number(info.get("revenueGrowth"), percentage=True),
            "quarterly_earnings_growth_pct": _clean_number(info.get("earningsGrowth"), percentage=True),
            "profit_margin_pct": _clean_number(info.get("profitMargins"), percentage=True),
            "debt_to_equity": _clean_number(info.get("debtToEquity")),
            "free_cash_flow": _clean_number(info.get("freeCashflow")),
            "beta": _clean_number(info.get("beta")),
            "past_one_year_return_pct": stock_1yr_return,
            "recent_news_titles": [item["title"] for item in news.get("news", [])],
        })
        result["data_complete"] = current_price is not None and current_price > 0
        if not result["data_complete"]:
            result["errors"].append("a positive current price is unavailable")
    except Exception as exc:
        result["errors"].append(f"market data fetch failed: {exc}")
    return result


# ==========================================
# 1. Scalable CLI Execution Engine
# ==========================================
def run_sc_command(cmd_list: list[str], timeout_seconds: int = 60) -> dict:
    """Executes a Scalable CLI command, appends --json, and parses output."""
    full_cmd = ["sc"] + cmd_list
    if "--json" not in full_cmd:
        full_cmd.append("--json")
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, check=False, timeout=timeout_seconds
        )
    except FileNotFoundError as exc:
        raise ScalableCLIError("The `sc` executable is not installed or is not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ScalableCLIError(f"Scalable CLI command timed out after {timeout_seconds}s: {' '.join(full_cmd)}") from exc
    raw_output = result.stdout.strip()
    try:
        envelope = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        detail = result.stderr.strip() or raw_output or "no output"
        raise ScalableCLIError(f"Scalable CLI returned non-JSON output: {detail}") from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("ok"), bool):
        raise ScalableCLIError("Scalable CLI returned an invalid machine envelope")
    if not envelope["ok"] or result.returncode != 0:
        error = envelope.get("error")
        if isinstance(error, dict):
            hints = envelope.get("hints") or []
            raise ScalableCLIError(
                f"{error.get('code', 'unknown_error')}: {error.get('message', 'Scalable CLI command failed')}. "
                + (f"Hints: {'; '.join(map(str, hints))}" if hints else "")
            )
        raise ScalableCLIError(result.stderr.strip() or "Scalable CLI command failed")
    data = envelope.get("data")
    if not isinstance(data, dict):
        raise ScalableCLIError("Scalable CLI success response is missing object field `data`")
    return data


def _broker_result(data: dict[str, Any], command: str) -> dict[str, Any]:
    result = data.get("result")
    if not isinstance(result, dict):
        raise ScalableCLIError(f"{command} response is missing object field data.result")
    return result


# ==========================================
# 2. Tool Implementations (Python <-> CLI)
# ==========================================
def get_portfolio_overview(timeout_seconds: int = 60) -> dict:
    """Fetches account summary, cash balance, and general context."""
    return _broker_result(
        run_sc_command(["broker", "overview"], timeout_seconds), "broker overview"
    )


def get_portfolio_holdings(timeout_seconds: int = 60) -> dict:
    """Fetches current owned stock/ETF positions and share quantities."""
    return _broker_result(
        run_sc_command(["broker", "holdings"], timeout_seconds), "broker holdings"
    )


def search_stock(query: str, timeout_seconds: int = 60) -> dict:
    """Searches for stocks or ETFs by name to retrieve their ISIN."""
    return _broker_result(
        run_sc_command(["broker", "search", query], timeout_seconds), "broker search"
    )


def preview_buy_order(isin: str, amount: Decimal | str, timeout_seconds: int = 60) -> dict:
    """Phase 1: Previews a buy order and gets a confirmation ID."""
    return run_sc_command(
        [
            "broker",
            "trade",
            "buy",
            "--isin",
            isin,
            "--amount",
            decimal_text(amount),
            "--order-type",
            "market",
        ],
        timeout_seconds,
    )


def confirm_buy_order(
    isin: str, amount: Decimal | str, confirmation_id: str,
    *, accept_unsuitable: bool = False, timeout_seconds: int = 60,
) -> dict:
    """Phase 2: Confirms and submits the previewed buy trade."""
    return run_sc_command(
        [
            "broker",
            "trade",
            "buy",
            "--isin",
            isin,
            "--amount",
            decimal_text(amount),
            "--order-type",
            "market",
            "--confirm",
            confirmation_id,
        ] + (["--accept-unsuitable"] if accept_unsuitable else []),
        timeout_seconds,
    )


def preview_sell_order(isin: str, shares: Decimal | str, timeout_seconds: int = 60) -> dict:
    """Phase 1: Previews a sell order by share quantity and gets a confirmation ID."""
    return run_sc_command(
        [
            "broker",
            "trade",
            "sell",
            "--isin",
            isin,
            "--shares",
            decimal_text(shares),
            "--order-type",
            "market",
        ],
        timeout_seconds,
    )


def confirm_sell_order(
    isin: str, shares: Decimal | str, confirmation_id: str,
    *, timeout_seconds: int = 60,
) -> dict:
    """Phase 2: Confirms and submits the previewed sell trade."""
    return run_sc_command(
        [
            "broker",
            "trade",
            "sell",
            "--isin",
            isin,
            "--shares",
            decimal_text(shares),
            "--order-type",
            "market",
            "--confirm",
            confirmation_id,
        ],
        timeout_seconds,
    )


def get_portfolio_cash_breakdown(timeout_seconds: int = 60) -> dict[str, Any]:
    return _broker_result(
        run_sc_command(["broker", "cash-breakdown"], timeout_seconds), "broker cash-breakdown"
    )


def get_stock_news(ticker: str) -> dict:
    """Fetches recent news headlines for a stock via Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        news_items = (stock.news or [])[:5]  # Limit to top 5 articles to save tokens
        articles = [{"title": title} for item in news_items if (title := _news_title(item))]
        return {"ticker": ticker, "news": articles}
    except Exception as e:
        return {"error": f"Failed to fetch news for {ticker}: {str(e)}"}


def check_cli_capabilities(timeout_seconds: int = 60) -> None:
    capabilities = run_sc_command(["capabilities"], timeout_seconds)
    commands = capabilities.get("commands")
    required = {
        "broker.overview", "broker.cash-breakdown", "broker.holdings",
        "broker.search", "broker.trade.buy", "broker.trade.sell",
    }
    if not isinstance(commands, list):
        raise ScalableCLIError("sc capabilities response is missing the commands list")
    missing = sorted(required.difference(map(str, commands)))
    if missing:
        raise ScalableCLIError("Installed Scalable CLI lacks required commands: " + ", ".join(missing))


# ==========================================
# 3. Portfolio Data and Instrument Resolution
# ==========================================
def _optional_decimal(value: Any, field_name: str) -> Decimal | None:
    if value is None or value == "":
        return None
    return to_decimal(value, field_name)


def build_portfolio_snapshot(
    overview: dict[str, Any], cash_breakdown: dict[str, Any], holdings_payload: dict[str, Any]
) -> PortfolioSnapshot:
    raw_items = holdings_payload.get("items")
    if not isinstance(raw_items, list):
        raise ScalableCLIError("broker holdings result is missing list field `items`")
    normalized_holdings = []
    holdings_sum = Decimal("0")
    seen_isins: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ScalableCLIError(f"broker holdings item {index} is not an object")
        isin = str(raw.get("isin") or "").strip().upper()
        if not is_valid_isin(isin):
            raise ScalableCLIError(f"broker holdings item {index} has an invalid ISIN")
        if isin in seen_isins:
            raise ScalableCLIError(f"broker holdings contains duplicate ISIN {isin}")
        seen_isins.add(isin)
        quantity = _optional_decimal(raw.get("quantity"), f"holdings[{index}].quantity")
        pending = _optional_decimal(raw.get("pending_quantity"), f"holdings[{index}].pending_quantity")
        quote = _optional_decimal(raw.get("quote_mid_price"), f"holdings[{index}].quote_mid_price")
        reported_value = _optional_decimal(raw.get("valuation"), f"holdings[{index}].valuation")
        computed_value = max(quantity or Decimal("0"), Decimal("0")) * quote if quote is not None else Decimal("0")
        current_value = max(reported_value or Decimal("0"), computed_value)
        pending_value = max(pending or Decimal("0"), Decimal("0")) * quote if quote is not None else Decimal("0")
        holdings_sum += max(reported_value or computed_value, Decimal("0"))
        normalized_holdings.append({
            **raw,
            "isin": isin,
            "quantity_decimal": quantity or Decimal("0"),
            "pending_quantity_decimal": pending or Decimal("0"),
            "current_position_value_decimal": current_value + pending_value,
        })
    valuation = overview.get("valuation")
    if not isinstance(valuation, dict):
        raise ScalableCLIError("broker overview result is missing object field `valuation`")
    holdings_value = _optional_decimal(valuation.get("total"), "valuation.total")
    warnings: list[str] = []
    if holdings_value is None:
        holdings_value = holdings_sum
        warnings.append("Overview total valuation was missing; summed holding valuations instead.")
    if holdings_value < 0:
        raise ScalableCLIError("Portfolio holdings valuation cannot be negative")
    cash_balance = _optional_decimal(cash_breakdown.get("cash_balance"), "cash_balance")
    buying_power = _optional_decimal(cash_breakdown.get("buying_power_without_credit"), "buying_power_without_credit")
    if cash_balance is None or buying_power is None:
        raise ScalableCLIError("broker cash-breakdown must provide cash_balance and buying_power_without_credit")
    available_cash = max(min(cash_balance, buying_power), Decimal("0"))
    total_value = holdings_value + cash_balance
    if total_value <= 0:
        raise ScalableCLIError("Total portfolio value must be positive")
    if holdings_sum > 0 and abs(holdings_value - holdings_sum) > max(Decimal("1"), holdings_value * Decimal("0.02")):
        warnings.append(
            "Overview valuation differs materially from summed security holdings; the overview total "
            "(which may include other asset classes) is used as denominator."
        )
    return PortfolioSnapshot(cash_balance, available_cash, holdings_value, total_value, normalized_holdings, warnings)


def fetch_portfolio_snapshot(settings: Settings) -> PortfolioSnapshot:
    timeout = settings.sc_command_timeout_seconds
    return build_portfolio_snapshot(
        get_portfolio_overview(timeout), get_portfolio_cash_breakdown(timeout), get_portfolio_holdings(timeout)
    )


def is_valid_isin(value: str) -> bool:
    isin = value.strip().upper()
    if not ISIN_RE.fullmatch(isin):
        return False
    expanded = "".join(char if char.isdigit() else str(ord(char) - 55) for char in isin)
    total = 0
    for index, char in enumerate(reversed(expanded)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
        total += digit // 10 + digit % 10
    return total % 10 == 0


def normalize_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper()
    if not TICKER_RE.fullmatch(symbol):
        raise AnalysisError(f"Invalid ticker syntax: {ticker!r}")
    return symbol


def _resolve_holding_ticker(isin: str) -> tuple[str, dict[str, Any], bool]:
    if yf is None:
        raise AnalysisError("yfinance is not installed; run `pip install -r requirements.txt`")
    try:
        search = yf.Search(isin, max_results=1, news_count=0)
        quotes = search.quotes or []
    except Exception as exc:
        return "", {"data_complete": False, "errors": [f"ticker lookup failed: {exc}"]}, False
    for quote in quotes[:1]:
        if not isinstance(quote, dict) or not quote.get("symbol"):
            continue
        try:
            symbol = normalize_ticker(str(quote["symbol"]))
        except AnalysisError:
            continue
        market_data = get_stock_evaluation_data(symbol)
        expected = market_data.get("isin")
        verified = not expected or expected == isin
        return symbol, market_data, verified
    return "", {"data_complete": False, "errors": ["No Yahoo ticker matched the ISIN"]}, False


def _resolve_candidate_isin(
    ticker: str, market_data: dict[str, Any], settings: Settings
) -> tuple[str, dict[str, Any], bool, str]:
    search_result = search_stock(ticker, settings.sc_command_timeout_seconds)
    items = search_result.get("items")
    if not isinstance(items, list):
        raise ScalableCLIError("broker search result is missing list field `items`")
    valid_items = [item for item in items if isinstance(item, dict) and is_valid_isin(str(item.get("isin") or ""))]
    expected = market_data.get("isin")
    if isinstance(expected, str) and is_valid_isin(expected):
        for item in valid_items:
            if str(item["isin"]).strip().upper() == expected.strip().upper():
                return expected.strip().upper(), item, True, "Yahoo ISIN matched Scalable search"
        return "", {}, False, "Yahoo ISIN did not match Scalable search"
    if len(valid_items) == 1:
        item = valid_items[0]
        return str(item["isin"]).strip().upper(), item, True, "single Scalable search result"
    return "", {}, False, "Scalable search was empty or ambiguous"


def _is_equity(broker_security_type: Any, market_data: dict[str, Any]) -> bool:
    types = {str(broker_security_type or "").upper(), str(market_data.get("quote_type") or "").upper()}
    return bool(types.intersection(EQUITY_TYPES))


def build_instruments(
    snapshot: PortfolioSnapshot, candidate_tickers: Iterable[str], settings: Settings
) -> list[dict[str, Any]]:
    instruments: list[dict[str, Any]] = []
    by_isin: dict[str, dict[str, Any]] = {}
    market_cache: dict[str, dict[str, Any]] = {}
    for holding in snapshot.holdings:
        isin = holding["isin"]
        ticker, market_data, ticker_verified = _resolve_holding_ticker(isin)
        if ticker:
            market_cache[ticker] = market_data
        equity = _is_equity(holding.get("security_type"), market_data)
        instrument = {
            "instrument_id": isin, "isin": isin, "ticker": ticker,
            "name": holding.get("name") or market_data.get("name") or isin,
            "security_type": holding.get("security_type"),
            "is_holding": True, "is_candidate": False,
            "quantity": decimal_text(holding["quantity_decimal"]),
            "pending_quantity": decimal_text(holding["pending_quantity_decimal"]),
            "current_position_value_eur": money_text(holding["current_position_value_decimal"]),
            "broker_quote_mid_price": holding.get("quote_mid_price"),
            "broker_quote_currency": holding.get("quote_currency"),
            "broker_quote_is_outdated": holding.get("quote_is_outdated"),
            "market_data": market_data, "is_equity": equity,
            "isin_verified": True, "ticker_verified": ticker_verified,
            "trade_data_complete": bool(equity and ticker_verified and market_data.get("data_complete")),
            "data_notes": [] if ticker_verified else ["Yahoo ticker-to-ISIN match could not be verified; trades are blocked."],
        }
        instruments.append(instrument)
        by_isin[isin] = instrument
    for ticker in dict.fromkeys(normalize_ticker(value) for value in candidate_tickers):
        market_data = market_cache.get(ticker)
        if market_data is None:
            market_data = get_stock_evaluation_data(ticker)
            market_cache[ticker] = market_data
        isin, scalable_item, isin_verified, resolution_note = _resolve_candidate_isin(ticker, market_data, settings)
        if isin and isin in by_isin:
            existing = by_isin[isin]
            existing["is_candidate"] = True
            existing["ticker"] = ticker
            existing["market_data"] = market_data
            existing["ticker_verified"] = True
            existing["trade_data_complete"] = bool(existing["is_equity"] and market_data.get("data_complete"))
            existing["data_notes"].append(resolution_note)
            continue
        equity = _is_equity(scalable_item.get("security_type"), market_data)
        instrument = {
            "instrument_id": isin or f"ticker:{ticker}", "isin": isin, "ticker": ticker,
            "name": scalable_item.get("name") or market_data.get("name") or ticker,
            "security_type": scalable_item.get("security_type") or market_data.get("quote_type"),
            "is_holding": False, "is_candidate": True,
            "quantity": "0", "pending_quantity": "0", "current_position_value_eur": "0.00",
            "market_data": market_data, "is_equity": equity,
            "isin_verified": isin_verified, "ticker_verified": True,
            "trade_data_complete": bool(equity and isin_verified and market_data.get("data_complete")),
            "data_notes": [resolution_note],
        }
        instruments.append(instrument)
        if isin:
            by_isin[isin] = instrument
    return instruments


# 4. Trade Planning & Human Safety Check

def analyze_instruments(
    instruments: list[dict[str, Any]], snapshot: PortfolioSnapshot, settings: Settings
) -> dict[str, Any]:
    if OpenAI is None:
        raise AnalysisError("openai is not installed; run `pip install -r requirements.txt`")
    if not os.environ.get("OPENAI_API_KEY"):
        raise AnalysisError("OPENAI_API_KEY is not set")
    model_input = {
        "task": "Analyze every instrument independently and return the required structured result.",
        "portfolio": snapshot.model_context(), "instruments": instruments,
    }
    try:
        client = OpenAI(timeout=settings.openai_timeout_seconds, max_retries=2)
        response = client.responses.create(
            model=settings.model,
            instructions=settings.system_prompt,
            input=json.dumps(model_input, ensure_ascii=False, default=str, allow_nan=False),
            reasoning={"effort": settings.reasoning_effort},
            text={"format": ANALYSIS_JSON_SCHEMA},
            store=False,
        )
    except Exception as exc:
        raise AnalysisError(f"OpenAI analysis request failed: {exc}") from exc
    output_text = getattr(response, "output_text", "")
    if not output_text:
        raise AnalysisError("OpenAI returned no structured analysis text")
    try:
        analysis = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise AnalysisError("OpenAI returned invalid JSON despite the structured schema") from exc
    return validate_analysis(analysis, instruments, settings)


def canonical_verdict(instrument: dict[str, Any], score: int, settings: Settings) -> str:
    if instrument["is_holding"] and score < settings.sell_below_score:
        return "SELL"
    if score >= settings.buy_min_score:
        return "BUY"
    return "HOLD" if instrument["is_holding"] else "PASS"


def validate_analysis(analysis: Any, instruments: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    if not isinstance(analysis, dict) or not isinstance(analysis.get("evaluations"), list):
        raise AnalysisError("Analysis must contain an evaluations list")
    if not isinstance(analysis.get("portfolio_summary"), str):
        raise AnalysisError("Analysis must contain a portfolio_summary string")
    expected = {item["instrument_id"]: item for item in instruments}
    seen: set[str] = set()
    for evaluation in analysis["evaluations"]:
        if not isinstance(evaluation, dict):
            raise AnalysisError("Every evaluation must be an object")
        identifier = evaluation.get("instrument_id")
        if not isinstance(identifier, str) or identifier not in expected or identifier in seen:
            raise AnalysisError(f"Unexpected or duplicate instrument_id: {identifier!r}")
        seen.add(identifier)
        score = evaluation.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not (0 <= score <= 100):
            raise AnalysisError(f"Invalid score for {identifier}: {score!r}")
        for key in ("forward_metrics_summary", "bull_case", "bear_case", "verdict", "confidence_notes"):
            if not isinstance(evaluation.get(key), str) or not evaluation[key].strip():
                raise AnalysisError(f"Evaluation {identifier} has invalid {key}")
        model_verdict = evaluation["verdict"]
        if model_verdict not in {"BUY", "HOLD", "SELL", "PASS"}:
            raise AnalysisError(f"Evaluation {identifier} has invalid verdict")
        enforced_verdict = canonical_verdict(expected[identifier], score, settings)
        evaluation["model_verdict"] = model_verdict
        evaluation["verdict"] = enforced_verdict
        evaluation["verdict_was_normalized"] = model_verdict != enforced_verdict
    missing = sorted(set(expected).difference(seen))
    if missing:
        raise AnalysisError("OpenAI omitted instruments: " + ", ".join(missing))
    return analysis


def target_pct_for_score(score: int, settings: Settings) -> Decimal | None:
    for band in settings.score_bands:
        if band.min_score <= score <= band.max_score:
            return band.target_position_pct
    return None


def calculate_buy_amount(
    *, score: int, total_portfolio_value: Decimal, current_position_value: Decimal,
    available_cash: Decimal, settings: Settings,
) -> tuple[Decimal, Decimal | None]:
    target_pct = target_pct_for_score(score, settings)
    if target_pct is None or total_portfolio_value <= 0 or available_cash <= 0:
        return Decimal("0.00"), target_pct
    cap_value = total_portfolio_value * settings.max_single_position_pct / PERCENT
    target_value = total_portfolio_value * target_pct / PERCENT
    remaining_position_capacity = min(cap_value, target_value) - max(current_position_value, Decimal("0"))
    amount = min(remaining_position_capacity, available_cash)
    if amount <= 0:
        return Decimal("0.00"), target_pct
    amount = amount.quantize(CENT, rounding=ROUND_DOWN)
    if amount < settings.min_order_amount_eur:
        return Decimal("0.00"), target_pct
    return amount, target_pct


def build_trade_plan(
    analysis: dict[str, Any], instruments: list[dict[str, Any]],
    snapshot: PortfolioSnapshot, settings: Settings,
) -> TradePlan:
    evaluations = {item["instrument_id"]: item for item in analysis["evaluations"]}
    warnings = list(snapshot.warnings)
    items: list[TradePlanItem] = []
    for instrument in instruments:
        evaluation = evaluations[instrument["instrument_id"]]
        if not instrument["is_holding"] or evaluation["score"] >= settings.sell_below_score:
            continue
        shares = to_decimal(instrument["quantity"], "holding quantity")
        if not instrument["trade_data_complete"]:
            warnings.append(f"SELL preview blocked for {instrument['instrument_id']}: market/ticker data or equity classification was not fully verified.")
        elif shares <= 0:
            warnings.append(f"SELL preview skipped for {instrument['instrument_id']}: no shares.")
        else:
            items.append(TradePlanItem(
                side="sell", instrument_id=instrument["instrument_id"], isin=instrument["isin"],
                score=evaluation["score"], shares=shares,
                current_position_value=to_decimal(instrument["current_position_value_eur"], "current position value"),
            ))
    remaining_cash = snapshot.available_cash
    buy_candidates = sorted(
        (instrument for instrument in instruments if evaluations[instrument["instrument_id"]]["score"] >= settings.buy_min_score),
        key=lambda item: (-evaluations[item["instrument_id"]]["score"], item["instrument_id"]),
    )
    for instrument in buy_candidates:
        evaluation = evaluations[instrument["instrument_id"]]
        if not instrument["trade_data_complete"] or not instrument["isin"]:
            warnings.append(f"BUY preview blocked for {instrument['instrument_id']}: ISIN, market data, or equity classification was not fully verified.")
            continue
        current_value = to_decimal(instrument["current_position_value_eur"], "current position value")
        amount, target_pct = calculate_buy_amount(
            score=evaluation["score"], total_portfolio_value=snapshot.total_value,
            current_position_value=current_value, available_cash=remaining_cash, settings=settings,
        )
        if amount <= 0:
            warnings.append(f"BUY preview skipped for {instrument['instrument_id']}: already at/above target, insufficient cash, or amount below minimum.")
            continue
        items.append(TradePlanItem(
            side="buy", instrument_id=instrument["instrument_id"], isin=instrument["isin"],
            score=evaluation["score"], amount=amount, target_position_pct=target_pct,
            current_position_value=current_value,
        ))
        remaining_cash -= amount
    return TradePlan(items=items, warnings=warnings)


def _preview_confirmation(preview: dict[str, Any]) -> tuple[str, bool]:
    confirmation = preview.get("confirmation")
    if not isinstance(confirmation, dict) or not isinstance(confirmation.get("id"), str) or not confirmation["id"].strip():
        raise ScalableCLIError("Trade preview is missing confirmation.id")
    return confirmation["id"], bool(confirmation.get("requires_accept_unsuitable"))


def validate_trade_preview(preview: dict[str, Any], item: TradePlanItem) -> None:
    result = preview.get("result")
    intent = result.get("intent") if isinstance(result, dict) else None
    if not isinstance(intent, dict):
        raise ScalableCLIError("Trade preview is missing result.intent")
    if str(intent.get("side") or "").upper() != item.side.upper():
        raise ScalableCLIError("Trade preview side does not match the planned order")
    if str(intent.get("isin") or "").upper() != item.isin.upper():
        raise ScalableCLIError("Trade preview ISIN does not match the planned order")
    expected = item.amount if item.side == "buy" else item.shares
    actual = intent.get("amount") if item.side == "buy" else intent.get("shares")
    if expected is None or to_decimal(actual, "preview order size") != expected:
        raise ScalableCLIError("Trade preview size does not match the planned order")
    if result.get("pre_trade_checks_passed") is not True:
        raise ScalableCLIError("Trade preview did not pass every pre-trade check")
    submission = result.get("order_submission")
    if not isinstance(submission, dict) or submission.get("submitted") is not False:
        raise ScalableCLIError("Phase-1 response is not an unsubmitted trade preview")
    _preview_confirmation(preview)
    if not isinstance(preview.get("presentation"), dict):
        raise ScalableCLIError("Trade preview is missing the required human presentation")
    compliance = preview.get("compliance")
    required_flags = (
        "must_present_all_information", "requires_explicit_user_confirmation_between_phases",
        "forbid_automatic_phase_2_execution", "confirmation_must_be_separate_step",
    )
    if not isinstance(compliance, dict) or any(compliance.get(flag) is not True for flag in required_flags):
        raise ScalableCLIError("Trade preview is missing required compliance flags")


def _display_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def render_trade_presentation(preview: dict[str, Any]) -> None:
    presentation = preview.get("presentation")
    if not isinstance(presentation, dict):
        raise ScalableCLIError("Trade preview has no presentation object")
    order = presentation.get("section_order")
    sections = presentation.get("sections")
    if not isinstance(order, list) or not isinstance(sections, dict):
        raise ScalableCLIError("Trade preview presentation is malformed")
    required_paths = presentation.get("required_leaf_paths")
    if not isinstance(required_paths, list) or not all(isinstance(path, str) for path in required_paths):
        raise ScalableCLIError("Trade preview is missing required disclosure paths")
    rendered_paths: set[str] = set()
    print("\n--- Scalable pre-trade disclosure ---")
    for section_key in order:
        if not isinstance(section_key, str):
            raise ScalableCLIError("Trade presentation contains an invalid section key")
        section = sections.get(section_key)
        if not isinstance(section, dict):
            raise ScalableCLIError(f"Trade presentation is missing section {section_key!r}")
        print(f"\n{section.get('title', section_key)}")
        fields = section.get("fields")
        if not isinstance(fields, list):
            raise ScalableCLIError(f"Trade presentation section {section_key!r} has no fields")
        for field_value in fields:
            if not isinstance(field_value, dict):
                raise ScalableCLIError("Trade presentation contains an invalid field")
            path = field_value.get("path")
            if not isinstance(path, str):
                raise ScalableCLIError("Trade presentation field has no JSON path")
            rendered_paths.add(path)
            print(f"- {field_value.get('label')}: {_display_value(field_value.get('value'))}")
    missing_paths = sorted(set(required_paths).difference(rendered_paths))
    if missing_paths:
        raise ScalableCLIError("Trade presentation omitted required disclosure paths: " + ", ".join(missing_paths))
    confirmation_id, _ = _preview_confirmation(preview)
    print(f"\nConfirmation ID: {confirmation_id}")
    compliance = preview.get("compliance")
    if isinstance(compliance, dict) and compliance.get("instruction"):
        print(f"Compliance instruction: {compliance['instruction']}")


def _latest_position(snapshot: PortfolioSnapshot, isin: str) -> dict[str, Any] | None:
    return next((holding for holding in snapshot.holdings if holding["isin"] == isin), None)


def revalidate_before_confirmation(item: TradePlanItem, settings: Settings) -> None:
    latest = fetch_portfolio_snapshot(settings)
    holding = _latest_position(latest, item.isin)
    if item.side == "sell":
        latest_shares = holding["quantity_decimal"] if holding else Decimal("0")
        if item.shares is None or latest_shares < item.shares:
            raise ScalableCLIError("Sell confirmation blocked because the latest share balance is below the previewed size")
        return
    current_value = holding["current_position_value_decimal"] if holding else Decimal("0")
    maximum_now, _ = calculate_buy_amount(
        score=item.score, total_portfolio_value=latest.total_value,
        current_position_value=current_value, available_cash=latest.available_cash, settings=settings,
    )
    if item.amount is None or item.amount > maximum_now:
        previewed_amount = money_text(item.amount or Decimal("0"))
        raise ScalableCLIError(
            "Buy confirmation blocked because refreshed cash/position limits allow "
            f"only €{money_text(maximum_now)} versus the previewed €{previewed_amount}"
        )


def execute_trade_plan(plan: TradePlan, settings: Settings, execute: bool) -> None:
    if not plan.items:
        print("\nNo trade previews are required by the enforced score and risk rules.")
        return
    for item in plan.items:
        size = f"€{money_text(item.amount)}" if item.side == "buy" and item.amount is not None else f"{decimal_text(item.shares or Decimal('0'))} shares"
        print(f"\nPreparing {item.side.upper()} preview: {item.instrument_id}, score {item.score}, {size}")
        try:
            if item.side == "buy":
                preview = preview_buy_order(item.isin, item.amount, settings.sc_command_timeout_seconds)
            else:
                preview = preview_sell_order(item.isin, item.shares, settings.sc_command_timeout_seconds)
            validate_trade_preview(preview, item)
            render_trade_presentation(preview)
        except AgentError as exc:
            print(f"Preview failed: {exc}", file=sys.stderr)
            continue
        if not execute:
            print("Preview only: no order was submitted. Use --execute for human-confirmed phase 2.")
            continue
        try:
            answer = input("\nType YES to submit this exact order, or anything else to skip: ").strip()
        except EOFError:
            print("\nNo confirmation input available; stopping without further orders.")
            return
        if answer != "YES":
            print("Order skipped.")
            continue
        try:
            # Refresh after the human decision so stale state cannot bypass the limits.
            revalidate_before_confirmation(item, settings)
            confirmation_id, requires_unsuitable = _preview_confirmation(preview)
            if item.side == "buy":
                submitted = confirm_buy_order(
                    item.isin, item.amount, confirmation_id,
                    accept_unsuitable=requires_unsuitable, timeout_seconds=settings.sc_command_timeout_seconds,
                )
            else:
                submitted = confirm_sell_order(
                    item.isin, item.shares, confirmation_id, timeout_seconds=settings.sc_command_timeout_seconds,
                )
            print("Order submission response received.")
            order_result = submitted.get("result")
            if isinstance(order_result, dict):
                submission = order_result.get("order_submission")
                if isinstance(submission, dict):
                    print(f"Submitted: {submission.get('submitted')} | Order ID: {submission.get('order_id')}")
        except AgentError as exc:
            print(f"Order confirmation failed; check broker status before retrying: {exc}", file=sys.stderr)


def get_trending_tickers(limit=4) -> list[str]:
    """
    Dynamically fetches high-quality, large-cap growth candidates
    strictly filtering out mutual funds and ETFs.
    """
    if yf is None:
        raise AnalysisError("yfinance is not installed; run `pip install -r requirements.txt`")
    print("\nScanning live market for tech and growth equities...")

    screener_pool = [
        "growth_technology_stocks",
        "undervalued_growth",
        "most_actives",
    ]
    selected_tickers = []
    for screener_key in screener_pool:
        try:
            print(f"Live querying yfinance screener: '{screener_key}'...")
            res = yf.screen(screener_key)
            quotes = res.get("quotes", [])
            for q in quotes:
                sym = q.get("symbol", "").upper()
                mcap = q.get("marketCap") or 0
                quote_type = q.get("quoteType", "")
                if quote_type == "EQUITY" and sym and "." not in sym and "-" not in sym:
                    if mcap >= 10_000_000_000 and sym not in selected_tickers:
                        selected_tickers.append(sym)
                        if len(selected_tickers) >= limit:
                            return selected_tickers
        except Exception as e:
            print(f"Warning: Screener '{screener_key}' failed: {e}", file=sys.stderr)
    final_tickers = selected_tickers[:limit]
    if not final_tickers:
        print("All live screeners failed or returned no eligible candidates. Using tech fallback.")
        final_tickers = ["NVDA", "AAPL", "MSFT", "TSM"][:limit]
    print(f"Selected live candidates for evaluation: {final_tickers}")
    return final_tickers


# 5. Agent Loop

def print_analysis(analysis: dict[str, Any], instruments: list[dict[str, Any]], plan: TradePlan) -> None:
    instrument_index = {item["instrument_id"]: item for item in instruments}
    print(f"\nPortfolio analysis\n{analysis['portfolio_summary']}")
    for evaluation in analysis["evaluations"]:
        instrument = instrument_index[evaluation["instrument_id"]]
        label = instrument.get("ticker") or instrument.get("name") or evaluation["instrument_id"]
        print(f"\n{label} ({evaluation['instrument_id']}) — {evaluation['score']}/100")
        print(f"Forward metrics: {evaluation['forward_metrics_summary']}")
        print(f"Bull case: {evaluation['bull_case']}")
        print(f"Bear case: {evaluation['bear_case']}")
        print(f"Verdict: {evaluation['verdict']}")
        if evaluation["verdict_was_normalized"]:
            print(f"Policy note: model verdict {evaluation['model_verdict']} was normalized to match the configured score thresholds.")
        print(f"Confidence/data: {evaluation['confidence_notes']}")
    if plan.warnings:
        print("\nSafety notes")
        for warning in plan.warnings:
            print(f"- {warning}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Scalable Capital holdings and preview policy-sized equity trades.")
    parser.add_argument("tickers", nargs="*", help="Candidate Yahoo ticker symbols. If omitted, large-cap screeners are used.")
    parser.add_argument(
        "--execute", action="store_true",
        help="After rendering each full Scalable phase-1 disclosure, ask for a separate explicit YES before submitting that exact order.",
    )
    parser.add_argument("--analysis-only", action="store_true", help="Run analysis and sizing but do not create Scalable trade previews.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to agent TOML configuration.")
    args = parser.parse_args(argv)
    if args.execute and args.analysis_only:
        parser.error("--execute and --analysis-only cannot be combined")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if load_dotenv is None:
            raise ConfigurationError("python-dotenv is not installed; run `pip install -r requirements.txt`")
        load_dotenv(BASE_DIR / ".env")
        settings = load_settings(args.config)
        check_cli_capabilities(settings.sc_command_timeout_seconds)
        tickers = (
            [normalize_ticker(ticker) for ticker in args.tickers]
            if args.tickers else get_trending_tickers(settings.candidate_limit)
        )
        print(f"Candidates: {', '.join(tickers)}")
        snapshot = fetch_portfolio_snapshot(settings)
        print(
            f"Account: cash €{money_text(snapshot.cash_balance)}, "
            f"available cash €{money_text(snapshot.available_cash)}, "
            f"holdings €{money_text(snapshot.holdings_value)}, total €{money_text(snapshot.total_value)}"
        )
        instruments = build_instruments(snapshot, tickers, settings)
        analysis = analyze_instruments(instruments, snapshot, settings)
        plan = build_trade_plan(analysis, instruments, snapshot, settings)
        print_analysis(analysis, instruments, plan)
        if args.analysis_only:
            print("\nAnalysis-only mode: no trade preview or submission was requested.")
        else:
            execute_trade_plan(plan, settings, args.execute)
        return 0
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled; no further action was taken.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
