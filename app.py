# ================================================================
# COMMODITY PRO TRADER ASSISTANT
# MCX COMMODITY OPTIONS ONLY
#
# Upstox + Streamlit
#
# IMPORTANT:
# This application DOES NOT resolve or trade MCX futures.
# It finds MCX CE/PE option contracts directly.
#
# Required Streamlit Secret:
# UPSTOX_ACCESS_TOKEN = "your_upstox_access_token"
#
# Run:
# streamlit run app.py
# ================================================================

import time
import threading
import gzip
import json
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ================================================================
# PAGE
# ================================================================

st.set_page_config(
    page_title="Commodity PRO Trader Assistant",
    page_icon="🛢️",
    layout="wide",
)


# ================================================================
# CONSTANTS
# ================================================================

API_BASE = "https://api.upstox.com"
IST = ZoneInfo("Asia/Kolkata")

API_LOCK = threading.Lock()
LAST_API_CALL = 0.0
MIN_API_GAP = 0.45


# ================================================================
# COMMODITY ALIASES
# ================================================================

ALIASES = {
    "GOLD": "GOLD",
    "GOLDM": "GOLDM",
    "GOLD MINI": "GOLDM",
    "GOLDMINI": "GOLDM",

    "SILVER": "SILVER",
    "SILVERM": "SILVERM",
    "SILVER MINI": "SILVERM",
    "SILVERMINI": "SILVERM",

    "CRUDE": "CRUDEOIL",
    "CRUDE OIL": "CRUDEOIL",
    "CRUDEOIL": "CRUDEOIL",

    "CRUDE OIL MINI": "CRUDEOILMINI",
    "CRUDEOILMINI": "CRUDEOILMINI",

    "NATURAL GAS": "NATURALGAS",
    "NATURAL GAS": "NATURALGAS",
    "NAT GAS": "NATURALGAS",
    "NATGAS": "NATURALGAS",

    "COPPER": "COPPER",
    "ZINC": "ZINC",
    "ALUMINIUM": "ALUMINIUM",
    "ALUMINUM": "ALUMINIUM",
    "LEAD": "LEAD",
    "NICKEL": "NICKEL",
}


# ================================================================
# HELPERS
# ================================================================

def now_ist():
    return datetime.now(IST)


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        x = float(value)

        if np.isfinite(x):
            return x

        return default

    except Exception:
        return default


def fmt_number(value, decimals=2):
    x = safe_float(value)

    if not np.isfinite(x):
        return "—"

    return f"{x:,.{decimals}f}"


def fmt_money(value, decimals=2):
    x = safe_float(value)

    if not np.isfinite(x):
        return "—"

    return f"₹{x:,.{decimals}f}"


def normalize_symbol(value):
    raw = " ".join(
        str(value or "").strip().upper().split()
    )

    return ALIASES.get(
        raw,
        raw.replace(" ", "")
    )


def normalize_text(value):
    return "".join(
        ch for ch in str(value or "").upper()
        if ch.isalnum()
    )


def get_token():
    try:
        token = st.secrets.get(
            "UPSTOX_ACCESS_TOKEN",
            ""
        )
    except Exception:
        token = ""

    token = str(token).strip()

    if not token:
        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN is missing. "
            "Add your Upstox access token in Streamlit Secrets."
        )

    return token


# ================================================================
# UPSTOX API
# ================================================================

def api_get(
    path,
    params=None,
    timeout=30
):
    global LAST_API_CALL

    with API_LOCK:

        gap = time.time() - LAST_API_CALL

        if gap < MIN_API_GAP:
            time.sleep(
                MIN_API_GAP - gap
            )

        try:
            response = requests.get(
                API_BASE + path,
                params=params or {},
                headers={
                    "Accept": "application/json",
                    "Authorization":
                        f"Bearer {get_token()}",
                },
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Unable to connect to Upstox: {exc}"
            ) from exc

        LAST_API_CALL = time.time()

    if response.status_code == 401:
        raise RuntimeError(
            "Upstox access token is invalid or expired. "
            "Generate a fresh token and update "
            "UPSTOX_ACCESS_TOKEN."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "Upstox API rate limit reached. "
            "Please wait a few seconds and try again."
        )

    if not response.ok:

        try:
            detail = response.json()
        except Exception:
            detail = response.text[:500]

        raise RuntimeError(
            f"Upstox API error {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            "Upstox returned an invalid JSON response."
        ) from exc


@st.cache_data(ttl=900, show_spinner=False)
def load_mcx_instrument_master():
    """Load Upstox official MCX BOD instrument master.

    This is used as the primary discovery source because MCX commodity
    option availability can be inconsistent when discovered through the
    free-text Instrument Search API. The master contains the actual live
    MCX_FO CE/PE contracts and their underlying_key values.

    IMPORTANT: only CE/PE rows are ever returned by the filtering layer;
    futures are never selected or used for trading.
    """
    url = "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz"

    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        raw = response.content

        try:
            raw = gzip.decompress(raw)
        except (OSError, gzip.BadGzipFile):
            # Some environments/proxies transparently decompress the file.
            pass

        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("Upstox MCX instrument master returned an unexpected format.")

        return payload
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to download the Upstox MCX instrument master: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Upstox MCX instrument master could not be decoded.") from exc


def _row_commodity_matches(row, symbol):
    if not isinstance(row, dict):
        return False

    target = normalize_text(symbol)

    fields = [
        normalize_text(row.get("underlying_symbol", "")),
        normalize_text(row.get("name", "")),
        normalize_text(row.get("short_name", "")),
    ]

    # Exact underlying/name matches first. This prevents SILVER from
    # accidentally selecting SILVERM contracts.
    if target and any(value == target for value in fields if value):
        return True

    trading = normalize_text(row.get("trading_symbol", ""))
    return bool(target and trading.startswith(target + " "))


def _master_mcx_options(symbol):
    """Return active MCX CE/PE rows for one commodity from the BOD master."""
    symbol = normalize_symbol(symbol)
    today = date.today().isoformat()
    rows = []

    for row in load_mcx_instrument_master():
        if not isinstance(row, dict):
            continue

        if str(row.get("exchange", "")).upper() != "MCX":
            continue
        if str(row.get("segment", "")).upper() != "MCX_FO":
            continue
        if str(row.get("instrument_type", "")).upper() not in {"CE", "PE"}:
            continue
        if not _row_commodity_matches(row, symbol):
            continue

        key = str(row.get("instrument_key", "")).strip()
        underlying_key = str(row.get("underlying_key", "")).strip()
        expiry = expiry_string(row.get("expiry"))
        strike = safe_float(row.get("strike_price"))

        if not key or not underlying_key or not expiry or expiry < today:
            continue
        if not np.isfinite(strike):
            continue

        rows.append(row)

    return rows


@st.cache_data(ttl=120, show_spinner=False)
def search_mcx_options(symbol, option_type, expiry_keyword=None):
    """Discover MCX CE/PE contracts without resolving any future.

    Primary source: official Upstox MCX instrument master.
    Secondary source: Upstox Instrument Search API.

    The master-first approach fixes cases such as SILVER where the
    free-text search may not return the active MCX option series.
    """
    symbol = normalize_symbol(symbol)
    option_type = str(option_type or "").upper().strip()

    if option_type not in ("CE", "PE"):
        return []

    # ------------------------------------------------------------
    # PRIMARY: official MCX instrument master
    # ------------------------------------------------------------
    try:
        master_rows = [
            row for row in _master_mcx_options(symbol)
            if str(row.get("instrument_type", "")).upper() == option_type
        ]

        if master_rows:
            if expiry_keyword:
                # Instrument-master rows have real expiry dates, so apply
                # relative expiry keywords locally instead of relying on
                # the search API's interpretation for MCX.
                expiries = sorted({expiry_string(r.get("expiry")) for r in master_rows})
                selected_expiry = _select_expiry_for_keyword(expiries, expiry_keyword)
                if selected_expiry:
                    filtered = [r for r in master_rows if expiry_string(r.get("expiry")) == selected_expiry]
                    if filtered:
                        return filtered
            return master_rows
    except Exception:
        # Fall through to the supported search API.
        pass

    # ------------------------------------------------------------
    # SECONDARY: Instrument Search API
    # ------------------------------------------------------------
    variants = {
        "GOLD": ["GOLD"],
        "GOLDM": ["GOLDM", "GOLD MINI", "GOLD"],
        "SILVER": ["SILVER"],
        "SILVERM": ["SILVERM", "SILVER MINI", "SILVER"],
        "CRUDEOIL": ["CRUDEOIL", "CRUDE OIL", "CRUDE"],
        "CRUDEOILMINI": ["CRUDEOILMINI", "CRUDE OIL MINI", "CRUDE MINI", "CRUDE"],
        "NATURALGAS": ["NATURALGAS", "NATURAL GAS", "NAT GAS", "NATGAS"],
        "COPPER": ["COPPER"],
        "ZINC": ["ZINC"],
        "ALUMINIUM": ["ALUMINIUM", "ALUMINUM"],
        "LEAD": ["LEAD"],
        "NICKEL": ["NICKEL"],
    }.get(symbol, [symbol])

    rows = []
    for query in variants:
        try:
            rows.extend(_search_instrument_rows(query, instrument_type=option_type, expiry=expiry_keyword))
        except Exception:
            continue

    if not rows:
        for query in variants:
            try:
                rows.extend(_search_instrument_rows(query, instrument_type=option_type, expiry=None))
            except Exception:
                continue

    cleaned = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("exchange", "")).upper() != "MCX":
            continue
        if str(row.get("segment", "")).upper() != "MCX_FO":
            continue
        if str(row.get("instrument_type", "")).upper() != option_type:
            continue
        if not _row_commodity_matches(row, symbol):
            continue
        key = str(row.get("instrument_key", "")).strip()
        if not key or key in seen:
            continue
        if not str(row.get("underlying_key", "")).strip():
            continue
        expiry = expiry_string(row.get("expiry"))
        if not expiry or expiry < date.today().isoformat():
            continue
        seen.add(key)
        cleaned.append(row)

    return cleaned


def _select_expiry_for_keyword(expiries, keyword):
    """Map Upstox-style relative expiry keywords to actual MCX dates."""
    valid = sorted(str(x) for x in expiries if x)
    if not valid:
        return ""

    today = date.today()
    future = []
    for value in valid:
        try:
            d = date.fromisoformat(value)
        except ValueError:
            continue
        if d >= today:
            future.append(d)

    if not future:
        return ""

    key = str(keyword or "").lower().strip()
    if key in {"current_month", "this_month", "near_month", "monthly"}:
        same_month = [d for d in future if d.year == today.year and d.month == today.month]
        return min(same_month).isoformat() if same_month else min(future).isoformat()
    if key in {"next_month", "far_month"}:
        next_month = [d for d in future if (d.year, d.month) > (today.year, today.month)]
        return min(next_month).isoformat() if next_month else min(future).isoformat()
    if key in {"current_week", "this_week", "near_week", "weekly", "next_week", "far_week"}:
        return min(future).isoformat()

    # Specific date is also accepted.
    try:
        requested = date.fromisoformat(key)
        candidates = [d for d in future if d == requested]
        return requested.isoformat() if candidates else ""
    except ValueError:
        return ""

def option_matches_commodity(
    row,
    symbol,
    option_type
):

    if not isinstance(row, dict):
        return False

    exchange = str(
        row.get("exchange", "")
    ).upper()

    segment = str(
        row.get("segment", "")
    ).upper()

    instrument_type = str(
        row.get("instrument_type", "")
    ).upper()

    if exchange != "MCX":
        return False

    if segment != "MCX_FO":
        return False

    if instrument_type != option_type:
        return False

    target = normalize_text(symbol)

    underlying_symbol = normalize_text(
        row.get("underlying_symbol", "")
    )

    trading_symbol = normalize_text(
        row.get("trading_symbol", "")
    )

    name = normalize_text(
        row.get("name", "")
    )

    short_name = normalize_text(
        row.get("short_name", "")
    )

    if underlying_symbol == target:
        return True

    if target and target in trading_symbol:
        return True

    if target and target in name:
        return True

    if target and target in short_name:
        return True

    return False


def expiry_string(value):
    if value is None or value == "":
        return ""

    # Current Instrument Search responses use YYYY-MM-DD strings, while
    # the downloadable BOD instrument master may contain epoch milliseconds.
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        try:
            ts = float(value)
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=IST).date().isoformat()
        except Exception:
            pass

    text = str(value).strip()
    if not text:
        return ""
    return text[:10]


def clean_options(
    rows,
    symbol,
    option_type
):

    today = date.today().isoformat()

    result = []

    for row in rows:

        if not option_matches_commodity(
            row,
            symbol,
            option_type
        ):
            continue

        instrument_key = str(
            row.get("instrument_key", "")
        ).strip()

        if not instrument_key:
            continue

        expiry = expiry_string(
            row.get("expiry")
        )

        if not expiry:
            continue

        if expiry < today:
            continue

        strike = safe_float(
            row.get("strike_price")
        )

        if not np.isfinite(strike):
            continue

        result.append(row)

    return result


def deduplicate_contracts(rows):

    output = {}

    for row in rows:

        key = str(
            row.get("instrument_key", "")
        ).strip()

        if key:
            output[key] = row

    return list(output.values())


# ================================================================
# RESOLVE MCX OPTIONS DIRECTLY
#
# No futures.
# ================================================================

@st.cache_data(
    ttl=90,
    show_spinner=False
)
@st.cache_data(ttl=90, show_spinner=False)
def resolve_mcx_options(symbol):
    """
    Resolve the nearest active MCX CE/PE expiry without resolving
    or trading an MCX future.

    We discover CE and PE directly, then choose the earliest expiry
    for which both sides exist.
    """
    symbol = normalize_symbol(symbol)

    if not symbol:
        raise RuntimeError("Please enter a commodity.")

    # Do not rely on one expiry keyword. MCX commodity option
    # availability differs by contract and expiry.
    expiry_attempts = [
        "current_month",
        "this_month",
        "near_month",
        "next_month",
        None,
    ]

    last_error = None

    for expiry_keyword in expiry_attempts:
        all_rows = []

        for option_type in ("CE", "PE"):
            try:
                rows = search_mcx_options(
                    symbol,
                    option_type,
                    expiry_keyword,
                )

                cleaned = clean_options(
                    rows,
                    symbol,
                    option_type,
                )

                all_rows.extend(cleaned)

            except Exception as exc:
                last_error = exc

        all_rows = deduplicate_contracts(all_rows)

        if not all_rows:
            continue

        # Only future/non-expired dates are allowed.
        expiries = sorted(
            {
                expiry_string(row.get("expiry"))
                for row in all_rows
                if expiry_string(row.get("expiry"))
                and expiry_string(row.get("expiry")) >= date.today().isoformat()
            }
        )

        for selected_expiry in expiries:
            selected = [
                row for row in all_rows
                if expiry_string(row.get("expiry")) == selected_expiry
            ]

            ce_count = sum(
                1 for row in selected
                if str(row.get("instrument_type", "")).upper() == "CE"
            )
            pe_count = sum(
                1 for row in selected
                if str(row.get("instrument_type", "")).upper() == "PE"
            )

            if ce_count == 0 or pe_count == 0:
                continue

            underlying_key = ""
            for row in selected:
                key = str(row.get("underlying_key", "")).strip()
                if key:
                    underlying_key = key
                    break

            if not underlying_key:
                continue

            return {
                "symbol": symbol,
                "expiry": selected_expiry,
                "underlying_key": underlying_key,
                "contracts": selected,
            }

    detail = ""
    if last_error:
        detail = f" Upstox detail: {last_error}"

    raise RuntimeError(
        f"Could not find active MCX CE/PE options for '{symbol}'. "
        f"The app searched MCX_FO directly and excluded futures."
        f"{detail}"
    )


# ================================================================
# FULL MARKET QUOTE V3
# ================================================================

@st.cache_data(
    ttl=20,
    show_spinner=False
)
def get_quotes(instrument_keys):

    keys = [
        str(x).strip()
        for x in instrument_keys
        if str(x).strip()
    ]

    keys = list(dict.fromkeys(keys))

    if not keys:
        return {}

    result = {}

    for start in range(
        0,
        len(keys),
        500
    ):

        chunk = keys[
            start:start + 500
        ]

        payload = api_get(
            "/v3/market-quote/quotes",
            params={
                "instrument_key":
                    ",".join(chunk)
            },
            timeout=30,
        )

        data = payload.get(
            "data",
            {}
        )

        if isinstance(data, dict):

            for key, value in data.items():

                if isinstance(value, dict):
                    result[
                        str(key)
                    ] = value

    return result


# ================================================================
# OPTION GREEKS
# ================================================================

@st.cache_data(
    ttl=20,
    show_spinner=False
)
def get_option_greeks(
    instrument_keys
):

    keys = [
        str(x).strip()
        for x in instrument_keys
        if str(x).strip()
    ]

    keys = list(dict.fromkeys(keys))

    if not keys:
        return {}

    result = {}

    for start in range(
        0,
        len(keys),
        50
    ):

        chunk = keys[
            start:start + 50
        ]

        payload = api_get(
            "/v3/market-quote/option-greek",
            params={
                "instrument_key":
                    ",".join(chunk)
            },
            timeout=30,
        )

        data = payload.get(
            "data",
            {}
        )

        if isinstance(data, dict):

            for key, value in data.items():

                if isinstance(value, dict):
                    result[
                        str(key)
                    ] = value

    return result


# ================================================================
# QUOTE PARSER
# ================================================================

def parse_quote(value):

    if not isinstance(value, dict):
        value = {}

    ohlc = value.get(
        "ohlc",
        {}
    ) or {}

    depth = value.get(
        "depth",
        {}
    ) or {}

    buy = depth.get(
        "buy",
        []
    ) or []

    sell = depth.get(
        "sell",
        []
    ) or []

    bid = np.nan
    ask = np.nan

    if buy and isinstance(
        buy[0],
        dict
    ):
        bid = safe_float(
            buy[0].get("price")
        )

    if sell and isinstance(
        sell[0],
        dict
    ):
        ask = safe_float(
            sell[0].get("price")
        )

    return {
        "ltp": safe_float(
            value.get("last_price")
        ),

        "open": safe_float(
            ohlc.get("open")
        ),

        "high": safe_float(
            ohlc.get("high")
        ),

        "low": safe_float(
            ohlc.get("low")
        ),

        "close": safe_float(
            ohlc.get("close")
        ),

        "previous_close": safe_float(
            ohlc.get("close")
        ),

        "volume": safe_float(
            value.get(
                "volume",
                ohlc.get(
                    "volume"
                )
            ),
            0
        ),

        "oi": safe_float(
            value.get("oi")
        ),

        "previous_oi": safe_float(
            value.get(
                "previous_oi",
                value.get("prev_oi")
            )
        ),

        "bid": bid,
        "ask": ask,
    }


def parse_greek(value):

    if not isinstance(value, dict):
        value = {}

    return {
        "delta": safe_float(
            value.get("delta")
        ),

        "gamma": safe_float(
            value.get("gamma")
        ),

        "theta": safe_float(
            value.get("theta")
        ),

        "vega": safe_float(
            value.get("vega")
        ),

        "iv": safe_float(
            value.get("iv")
        ),

        "pop": safe_float(
            value.get("pop")
        ),

        "volume": safe_float(
            value.get("volume"),
            0
        ),

        "oi": safe_float(
            value.get("oi")
        ),
    }


# ================================================================
# BUILD OPTION DATAFRAME
# ================================================================

def build_option_dataframe(
    resolved
):

    contracts = resolved.get(
        "contracts",
        []
    )

    keys = [
        str(
            row.get("instrument_key")
        )
        for row in contracts
        if row.get("instrument_key")
    ]

    quotes = get_quotes(keys)

    greeks = get_option_greeks(keys)

    rows = []

    for contract in contracts:

        key = str(
            contract.get(
                "instrument_key",
                ""
            )
        )

        quote = parse_quote(
            quotes.get(
                key,
                {}
            )
        )

        greek = parse_greek(
            greeks.get(
                key,
                {}
            )
        )

        oi = quote["oi"]

        if not np.isfinite(oi):
            oi = greek["oi"]

        volume = quote["volume"]

        if not np.isfinite(volume):
            volume = greek["volume"]

        lot_size = safe_float(
            contract.get(
                "lot_size",
                contract.get(
                    "minimum_lot",
                    1
                )
            ),
            1
        )

        rows.append({
            "instrument_key": key,

            "trading_symbol":
                contract.get(
                    "trading_symbol",
                    ""
                ),

            "option_type":
                str(
                    contract.get(
                        "instrument_type",
                        ""
                    )
                ).upper(),

            "strike":
                safe_float(
                    contract.get(
                        "strike_price"
                    )
                ),

            "expiry":
                expiry_string(
                    contract.get(
                        "expiry"
                    )
                ),

            "lot_size": lot_size,

            "tick_size":
                safe_float(
                    contract.get(
                        "tick_size"
                    )
                ),

            "underlying_key":
                str(
                    contract.get(
                        "underlying_key",
                        ""
                    )
                ).strip(),

            "ltp":
                quote["ltp"],

            "open":
                quote["open"],

            "high":
                quote["high"],

            "low":
                quote["low"],

            "close":
                quote["close"],

            "volume":
                volume,

            "oi":
                oi,

            "previous_oi":
                quote["previous_oi"],

            "bid":
                quote["bid"],

            "ask":
                quote["ask"],

            "delta":
                greek["delta"],

            "gamma":
                greek["gamma"],

            "theta":
                greek["theta"],

            "vega":
                greek["vega"],

            "iv":
                greek["iv"],

            "pop":
                greek["pop"],
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Upstox returned no usable MCX option contracts."
        )

    return (
        df
        .sort_values(
            ["strike", "option_type"]
        )
        .reset_index(drop=True)
    )


# ================================================================
# CANDLE DATA
# ================================================================

def parse_candles(payload):

    data = payload.get(
        "data",
        {}
    ) or {}

    candles = data.get(
        "candles",
        []
    ) or []

    rows = []

    for candle in candles:

        if len(candle) < 6:
            continue

        timestamp = pd.to_datetime(
            candle[0],
            errors="coerce"
        )

        rows.append({
            "timestamp": timestamp,

            "open":
                safe_float(candle[1]),

            "high":
                safe_float(candle[2]),

            "low":
                safe_float(candle[3]),

            "close":
                safe_float(candle[4]),

            "volume":
                safe_float(
                    candle[5],
                    0
                ),

            "oi":
                safe_float(
                    candle[6]
                )
                if len(candle) > 6
                else np.nan,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return (
        df
        .dropna(
            subset=[
                "timestamp",
                "close"
            ]
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )


@st.cache_data(
    ttl=60,
    show_spinner=False
)
def get_intraday(
    instrument_key,
    interval
):

    encoded = requests.utils.quote(
        str(instrument_key),
        safe=""
    )

    payload = api_get(
        f"/v3/historical-candle/"
        f"intraday/"
        f"{encoded}/"
        f"minutes/"
        f"{interval}",
        timeout=30,
    )

    return parse_candles(
        payload
    )


@st.cache_data(
    ttl=600,
    show_spinner=False
)
def get_daily(
    instrument_key
):

    end_date = date.today()

    start_date = (
        end_date
        - timedelta(days=370)
    )

    encoded = requests.utils.quote(
        str(instrument_key),
        safe=""
    )

    payload = api_get(
        f"/v3/historical-candle/"
        f"{encoded}/"
        f"days/1/"
        f"{end_date.isoformat()}/"
        f"{start_date.isoformat()}",
        timeout=30,
    )

    return parse_candles(
        payload
    )


# ================================================================
# TECHNICAL INDICATORS
# ================================================================

def add_indicators(df):

    x = df.copy()

    if x.empty:
        return x

    close = x["close"]

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    x["rsi"] = (
        100
        -
        100 / (1 + rs)
    )

    x["ema20"] = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    x["ema50"] = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            x["high"] - x["low"],

            (
                x["high"]
                - previous_close
            ).abs(),

            (
                x["low"]
                - previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    x["atr14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    typical_price = (
        x["high"]
        + x["low"]
        + x["close"]
    ) / 3

    volume = (
        x["volume"]
        .fillna(0)
    )

    cumulative_volume = (
        volume
        .cumsum()
        .replace(
            0,
            np.nan
        )
    )

    x["vwap"] = (
        typical_price
        * volume
    ).cumsum() / cumulative_volume

    # ADX

    up_move = x["high"].diff()

    down_move = -x["low"].diff()

    plus_dm = np.where(
        (
            (up_move > down_move)
            &
            (up_move > 0)
        ),
        up_move,
        0
    )

    minus_dm = np.where(
        (
            (down_move > up_move)
            &
            (down_move > 0)
        ),
        down_move,
        0
    )

    atr = x["atr14"].replace(
        0,
        np.nan
    )

    plus_di = (
        100
        *
        pd.Series(
            plus_dm,
            index=x.index
        )
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
        /
        atr
    )

    minus_di = (
        100
        *
        pd.Series(
            minus_dm,
            index=x.index
        )
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
        /
        atr
    )

    dx = (
        100
        *
        (
            plus_di
            - minus_di
        ).abs()
        /
        (
            plus_di
            + minus_di
        ).replace(
            0,
            np.nan
        )
    )

    x["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    x["volume_ma20"] = (
        x["volume"]
        .rolling(20)
        .mean()
    )

    return x


def timeframe_analysis(df):

    if df.empty or len(df) < 25:

        return {
            "trend": "UNKNOWN",
            "score": 0,
            "rsi": np.nan,
            "adx": np.nan,
            "atr": np.nan,
            "ema20": np.nan,
            "ema50": np.nan,
            "vwap": np.nan,
            "close": np.nan,
            "volume_confirmed": False,
        }

    x = add_indicators(df)

    row = x.iloc[-1]

    close = safe_float(
        row["close"]
    )

    ema20 = safe_float(
        row["ema20"]
    )

    ema50 = safe_float(
        row["ema50"]
    )

    rsi = safe_float(
        row["rsi"]
    )

    adx = safe_float(
        row["adx"]
    )

    vwap = safe_float(
        row["vwap"]
    )

    volume = safe_float(
        row["volume"],
        0
    )

    volume_ma = safe_float(
        row["volume_ma20"],
        0
    )

    bullish = (
        np.isfinite(close)
        and np.isfinite(ema20)
        and np.isfinite(ema50)
        and np.isfinite(rsi)
        and close > ema20
        and ema20 >= ema50
        and rsi >= 52
    )

    bearish = (
        np.isfinite(close)
        and np.isfinite(ema20)
        and np.isfinite(ema50)
        and np.isfinite(rsi)
        and close < ema20
        and ema20 <= ema50
        and rsi <= 48
    )

    if bullish:
        trend = "BULLISH"
    elif bearish:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    score = 0

    if (
        np.isfinite(close)
        and np.isfinite(ema20)
    ):
        if close > ema20:
            score += 20

    if (
        np.isfinite(ema20)
        and np.isfinite(ema50)
    ):
        if ema20 > ema50:
            score += 20

    if np.isfinite(rsi):

        if rsi >= 52:
            score += 15

        elif rsi <= 48:
            score += 15

    if np.isfinite(adx):
        if adx >= 20:
            score += 15

    if (
        np.isfinite(close)
        and np.isfinite(vwap)
    ):
        if close >= vwap:
            score += 10

    volume_confirmed = (
        volume_ma > 0
        and volume >= volume_ma
    )

    if volume_confirmed:
        score += 5

    return {
        "trend": trend,
        "score": min(
            int(score),
            100
        ),
        "rsi": rsi,
        "adx": adx,
        "atr": safe_float(
            row["atr14"]
        ),
        "ema20": ema20,
        "ema50": ema50,
        "vwap": vwap,
        "close": close,
        "volume_confirmed":
            volume_confirmed,
    }


# ================================================================
# OPTION STRUCTURE
# ================================================================

def calculate_option_structure(
    option_df,
    underlying_price
):

    result = {
        "atm": np.nan,
        "pcr": np.nan,
        "support": np.nan,
        "resistance": np.nan,
        "put_wall": np.nan,
        "call_wall": np.nan,
    }

    if option_df.empty:
        return result

    strikes = (
        pd.to_numeric(
            option_df["strike"],
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    if len(strikes) == 0:
        return result

    if np.isfinite(
        underlying_price
    ):

        atm = min(
            strikes,
            key=lambda x:
                abs(
                    x
                    - underlying_price
                )
        )

    else:

        atm = float(
            np.median(
                strikes
            )
        )

    result["atm"] = float(atm)

    calls = option_df[
        option_df["option_type"] == "CE"
    ].copy()

    puts = option_df[
        option_df["option_type"] == "PE"
    ].copy()

    call_oi = (
        calls["oi"]
        .fillna(0)
        .clip(lower=0)
    )

    put_oi = (
        puts["oi"]
        .fillna(0)
        .clip(lower=0)
    )

    total_call_oi = float(
        call_oi.sum()
    )

    total_put_oi = float(
        put_oi.sum()
    )

    if total_call_oi > 0:

        result["pcr"] = (
            total_put_oi
            /
            total_call_oi
        )

    if (
        not calls.empty
        and call_oi.max() > 0
    ):

        idx = call_oi.idxmax()

        result["call_wall"] = safe_float(
            calls.loc[
                idx,
                "strike"
            ]
        )

    if (
        not puts.empty
        and put_oi.max() > 0
    ):

        idx = put_oi.idxmax()

        result["put_wall"] = safe_float(
            puts.loc[
                idx,
                "strike"
            ]
        )

    put_candidates = puts[
        puts["strike"] <= atm
    ].copy()

    call_candidates = calls[
        calls["strike"] >= atm
    ].copy()

    if not put_candidates.empty:

        put_candidates = (
            put_candidates
            .sort_values(
                ["oi", "strike"],
                ascending=[
                    False,
                    False
                ]
            )
        )

        result["support"] = safe_float(
            put_candidates.iloc[
                0
            ]["strike"]
        )

    if not call_candidates.empty:

        call_candidates = (
            call_candidates
            .sort_values(
                ["oi", "strike"],
                ascending=[
                    False,
                    True
                ]
            )
        )

        result["resistance"] = safe_float(
            call_candidates.iloc[
                0
            ]["strike"]
        )

    return result


# ================================================================
# SELECT NEAREST OPTION
# ================================================================

def choose_option(
    option_df,
    option_type,
    underlying_price
):

    side = option_df[
        option_df["option_type"]
        == option_type
    ].copy()

    if side.empty:
        return {}

    if np.isfinite(
        underlying_price
    ):

        side["distance"] = (
            side["strike"]
            - underlying_price
        ).abs()

    else:

        side["distance"] = 0

    side["liquidity"] = (
        side["volume"]
        .fillna(0)
        .clip(lower=0)
        +
        side["oi"]
        .fillna(0)
        .clip(lower=0)
        * 0.10
    )

    side = side.sort_values(
        [
            "distance",
            "liquidity"
        ],
        ascending=[
            True,
            False
        ]
    )

    return side.iloc[0].to_dict()


# ================================================================
# TRADE PLAN
# ================================================================

def build_trade_plan(
    selected_option,
    direction,
    analysis_5m,
    analysis_30m,
    analysis_daily,
    risk_profile
):

    if not selected_option:

        return {
            "decision": "NO TRADE",
            "quality": 0,
            "reason":
                "No suitable option contract found.",
        }

    premium = safe_float(
        selected_option.get(
            "ltp"
        )
    )

    if (
        not np.isfinite(premium)
        or premium <= 0
    ):

        return {
            "decision": "NO TRADE",
            "quality": 0,
            "reason":
                "Selected option has no usable live premium.",
        }

    quality = int(
        0.45
        * analysis_5m.get(
            "score",
            0
        )
        +
        0.35
        * analysis_30m.get(
            "score",
            0
        )
        +
        0.20
        * analysis_daily.get(
            "score",
            0
        )
    )

    bullish_alignment = (
        analysis_5m["trend"]
        == "BULLISH"
        and
        analysis_30m["trend"]
        == "BULLISH"
    )

    bearish_alignment = (
        analysis_5m["trend"]
        == "BEARISH"
        and
        analysis_30m["trend"]
        == "BEARISH"
    )

    if (
        direction == "CALL BUY"
        and not bullish_alignment
    ):

        return {
            "decision": "NO TRADE",
            "quality": quality,
            "reason":
                "CALL BUY rejected because "
                "the 5-minute and 30-minute "
                "underlying trends are not "
                "both bullish.",
        }

    if (
        direction == "PUT BUY"
        and not bearish_alignment
    ):

        return {
            "decision": "NO TRADE",
            "quality": quality,
            "reason":
                "PUT BUY rejected because "
                "the 5-minute and 30-minute "
                "underlying trends are not "
                "both bearish.",
        }

    if quality < 55:

        return {
            "decision": "NO TRADE",
            "quality": quality,
            "reason":
                f"Setup quality is "
                f"{quality}/100, below "
                f"the minimum quality gate.",
        }

    profiles = {

        "Conservative": (
            0.65,
            1.20,
            1.60,
            2.00,
            2.40
        ),

        "Balanced": (
            0.55,
            1.35,
            1.90,
            2.50,
            3.10
        ),

        "Aggressive": (
            0.45,
            1.50,
            2.20,
            3.00,
            3.80
        ),
    }

    (
        sl_percent,
        t1_percent,
        t2_percent,
        t3_percent,
        t4_percent
    ) = profiles[risk_profile]

    entry = premium

    stop_loss = max(
        entry
        * (
            1
            - sl_percent
        ),
        0.01
    )

    target_1 = (
        entry
        * (
            1
            + t1_percent
        )
    )

    target_2 = (
        entry
        * (
            1
            + t2_percent
        )
    )

    target_3 = (
        entry
        * (
            1
            + t3_percent
        )
    )

    target_4 = (
        entry
        * (
            1
            + t4_percent
        )
    )

    lot_size = safe_float(
        selected_option.get(
            "lot_size",
            1
        ),
        1
    )

    risk_per_unit = (
        entry
        - stop_loss
    )

    profit_t1_per_unit = (
        target_1
        - entry
    )

    profit_t2_per_unit = (
        target_2
        - entry
    )

    profit_t3_per_unit = (
        target_3
        - entry
    )

    profit_t4_per_unit = (
        target_4
        - entry
    )

    max_loss = (
        risk_per_unit
        * lot_size
    )

    t1_profit = (
        profit_t1_per_unit
        * lot_size
    )

    t2_profit = (
        profit_t2_per_unit
        * lot_size
    )

    t3_profit = (
        profit_t3_per_unit
        * lot_size
    )

    t4_profit = (
        profit_t4_per_unit
        * lot_size
    )

    rr1 = (
        profit_t1_per_unit
        /
        risk_per_unit
        if risk_per_unit > 0
        else np.nan
    )

    rr2 = (
        profit_t2_per_unit
        /
        risk_per_unit
        if risk_per_unit > 0
        else np.nan
    )

    option_pop = safe_float(
        selected_option.get(
            "pop"
        )
    )

    if np.isfinite(
        option_pop
    ):

        pop = option_pop
        pop_source = (
            "Upstox Option Greek API"
        )

    else:

        pop = float(
            np.clip(
                50
                +
                (
                    quality
                    - 50
                )
                * 0.50
                +
                (
                    4
                    if analysis_5m[
                        "volume_confirmed"
                    ]
                    else 0
                ),
                50,
                82
            )
        )

        pop_source = (
            "Rule-based estimate"
        )

    return {
        "decision": direction,
        "quality": quality,
        "entry": entry,
        "sl": stop_loss,
        "t1": target_1,
        "t2": target_2,
        "t3": target_3,
        "t4": target_4,
        "lot": lot_size,
        "risk_per_unit": risk_per_unit,
        "t1_profit_per_unit":
            profit_t1_per_unit,
        "t2_profit_per_unit":
            profit_t2_per_unit,
        "t3_profit_per_unit":
            profit_t3_per_unit,
        "t4_profit_per_unit":
            profit_t4_per_unit,
        "max_loss": max_loss,
        "t1_pnl": t1_profit,
        "t2_pnl": t2_profit,
        "t3_pnl": t3_profit,
        "t4_pnl": t4_profit,
        "rr1": rr1,
        "rr2": rr2,
        "pop": pop,
        "pop_source": pop_source,
        "reason":
            f"Underlying technical alignment "
            f"supports {direction}. "
            f"The selected option is the "
            f"nearest usable CE/PE contract "
            f"to the underlying price.",
    }


# ================================================================
# SAFETY CHECKS
# ================================================================

def beginner_safety_checks(
    plan,
    analysis_5m,
    analysis_30m
):

    checks = []

    decision = plan.get(
        "decision",
        "NO TRADE"
    )

    checks.append(
        (
            "PASS"
            if decision
            in {
                "CALL BUY",
                "PUT BUY"
            }
            else "STOP",

            "Trade Direction",

            (
                "A defined option "
                "buying direction exists."
                if decision
                in {
                    "CALL BUY",
                    "PUT BUY"
                }
                else
                "No valid option-buying "
                "direction exists."
            )
        )
    )

    alignment = (
        analysis_5m["trend"]
        ==
        analysis_30m["trend"]
        and
        analysis_5m["trend"]
        in {
            "BULLISH",
            "BEARISH"
        }
    )

    checks.append(
        (
            "PASS"
            if alignment
            else "STOP",

            "Timeframe Agreement",

            (
                "5-minute and 30-minute "
                "underlying trends agree."
                if alignment
                else
                "5-minute and 30-minute "
                "underlying trends do not agree."
            )
        )
    )

    volume_ok = (
        analysis_5m[
            "volume_confirmed"
        ]
    )

    checks.append(
        (
            "PASS"
            if volume_ok
            else "WAIT",

            "Underlying Volume",

            (
                "Recent underlying "
                "volume confirms activity."
                if volume_ok
                else
                "Underlying volume has "
                "not confirmed the move."
            )
        )
    )

    rr = safe_float(
        plan.get("rr1")
    )

    checks.append(
        (
            "PASS"
            if (
                np.isfinite(rr)
                and rr >= 1
            )
            else "STOP",

            "Risk / Reward",

            (
                f"Target 1 is approximately "
                f"{rr:.1f}R."
                if np.isfinite(rr)
                else
                "Risk/reward unavailable."
            )
        )
    )

    quality = safe_float(
        plan.get(
            "quality",
            0
        ),
        0
    )

    checks.append(
        (
            "PASS"
            if quality >= 60
            else "STOP",

            "Setup Quality",

            f"Quality score is "
            f"{quality:.0f}/100."
        )
    )

    safe = all(
        status != "STOP"
        for status, _, _
        in checks
    )

    return {
        "safe": safe,
        "checks": checks,
    }


# ================================================================
# SAFE TECHNICAL DATA LOADER
# ================================================================

def load_underlying_analysis(
    underlying_key
):

    errors = []

    # -------------------------
    # 5 minute
    # -------------------------

    try:

        candles_5m = get_intraday(
            underlying_key,
            5
        )

    except Exception as exc:

        candles_5m = pd.DataFrame()

        errors.append(
            f"5-minute data: {exc}"
        )

    # -------------------------
    # 30 minute
    # -------------------------

    try:

        candles_30m = get_intraday(
            underlying_key,
            30
        )

    except Exception as exc:

        candles_30m = pd.DataFrame()

        errors.append(
            f"30-minute data: {exc}"
        )

    # -------------------------
    # Daily
    # -------------------------

    try:

        candles_daily = get_daily(
            underlying_key
        )

    except Exception as exc:

        candles_daily = pd.DataFrame()

        errors.append(
            f"Daily data: {exc}"
        )

    analysis_5m = timeframe_analysis(
        candles_5m
    )

    analysis_30m = timeframe_analysis(
        candles_30m
    )

    analysis_daily = timeframe_analysis(
        candles_daily
    )

    return (
        analysis_5m,
        analysis_30m,
        analysis_daily,
        errors,
    )


# ================================================================
# PAGE STYLE
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 18px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# HEADER
# ================================================================

st.markdown(
    '<div class="main-title">'
    '🛢️ Commodity PRO Trader Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'MCX commodity OPTIONS only • '
    'live Upstox data • CE/PE • '
    'Greeks • OI • technicals • '
    'risk plan'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "OPTION-ONLY MODE: This application does NOT "
    "search for, resolve, or analyze MCX futures "
    "for trading. It finds MCX CE/PE contracts "
    "directly and uses the option contract's "
    "underlying_key only as the reference market."
)


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header(
        "⚙️ Analysis Settings"
    )

    commodity = st.text_input(
        "Commodity",
        value="Crude Oil",
        placeholder=(
            "Gold / Crude Oil / "
            "Silver / Natural Gas"
        )
    )

    risk_profile = st.selectbox(
        "Risk Profile",
        [
            "Conservative",
            "Balanced",
            "Aggressive"
        ],
        index=1
    )

    beginner_mode = st.checkbox(
        "Beginner Safety + Explainability",
        value=True
    )

    analyze_button = st.button(
        "🔎 Analyze Commodity Options",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")

    st.caption(
        "Examples:"
    )

    st.caption(
        "Gold • Gold Mini • Silver • "
        "Silver Mini • Crude Oil • "
        "Crude Oil Mini • Natural Gas • "
        "Copper • Zinc • Aluminium • "
        "Lead • Nickel"
    )

    st.caption(
        "Decision: CALL BUY / PUT BUY / NO TRADE"
    )

    st.caption(
        "PoP is informational and is not a "
        "guarantee of profit."
    )


# ================================================================
# SESSION
# ================================================================

if "commodity_result" not in st.session_state:

    st.session_state[
        "commodity_result"
    ] = None


# ================================================================
# ANALYZE
# ================================================================

if analyze_button:

    with st.spinner(
        "Finding MCX CE/PE options and "
        "analyzing live data..."
    ):

        try:

            # ----------------------------------------------------
            # STEP 1:
            # DIRECT MCX OPTION RESOLUTION
            # ----------------------------------------------------

            resolved = resolve_mcx_options(
                commodity
            )

            # ----------------------------------------------------
            # STEP 2:
            # BUILD OPTION DATA
            # ----------------------------------------------------

            option_df = (
                build_option_dataframe(
                    resolved
                )
            )

            # ----------------------------------------------------
            # STEP 3:
            # GET UNDERLYING KEY
            #
            # THIS COMES FROM THE OPTION.
            # NO FUTURES LOOKUP.
            # ----------------------------------------------------

            underlying_key = str(
                resolved.get(
                    "underlying_key",
                    ""
                )
            ).strip()

            if not underlying_key:

                for key in (
                    option_df[
                        "underlying_key"
                    ]
                    .dropna()
                    .astype(str)
                    .tolist()
                ):

                    if key.strip():

                        underlying_key = (
                            key.strip()
                        )

                        break

            if not underlying_key:

                raise RuntimeError(
                    "MCX option contracts were "
                    "found, but Upstox did not "
                    "return their underlying_key. "
                    "Please retry after refreshing "
                    "the Upstox token."
                )

            # ----------------------------------------------------
            # STEP 4:
            # UNDERLYING LIVE QUOTE
            # ----------------------------------------------------

            underlying_quotes = get_quotes(
                [underlying_key]
            )

            underlying_quote = parse_quote(
                underlying_quotes.get(
                    underlying_key,
                    {}
                )
            )

            underlying_price = safe_float(
                underlying_quote.get(
                    "ltp"
                )
            )

            # ----------------------------------------------------
            # STEP 5:
            # TECHNICALS
            #
            # Technical failures do NOT stop
            # option analysis.
            # ----------------------------------------------------

            (
                analysis_5m,
                analysis_30m,
                analysis_daily,
                technical_errors
            ) = load_underlying_analysis(
                underlying_key
            )

            # ----------------------------------------------------
            # STEP 6:
            # OPTION STRUCTURE
            # ----------------------------------------------------

            structure = (
                calculate_option_structure(
                    option_df,
                    underlying_price
                )
            )

            # ----------------------------------------------------
            # STEP 7:
            # SELECT CE / PE
            # ----------------------------------------------------

            call_option = choose_option(
                option_df,
                "CE",
                underlying_price
            )

            put_option = choose_option(
                option_df,
                "PE",
                underlying_price
            )

            # ----------------------------------------------------
            # STEP 8:
            # DECISION
            # ----------------------------------------------------

            if (
                analysis_5m["trend"]
                == "BULLISH"
                and
                analysis_30m["trend"]
                == "BULLISH"
            ):

                direction = "CALL BUY"

                selected_option = (
                    call_option
                )

            elif (
                analysis_5m["trend"]
                == "BEARISH"
                and
                analysis_30m["trend"]
                == "BEARISH"
            ):

                direction = "PUT BUY"

                selected_option = (
                    put_option
                )

            else:

                direction = "NO TRADE"

                selected_option = {}

            # ----------------------------------------------------
            # STEP 9:
            # TRADE PLAN
            # ----------------------------------------------------

            if direction == "NO TRADE":

                quality = int(
                    0.45
                    * analysis_5m[
                        "score"
                    ]
                    +
                    0.35
                    * analysis_30m[
                        "score"
                    ]
                    +
                    0.20
                    * analysis_daily[
                        "score"
                    ]
                )

                plan = {
                    "decision":
                        "NO TRADE",

                    "quality":
                        quality,

                    "reason":
                        "The 5-minute and "
                        "30-minute underlying "
                        "trends are not aligned "
                        "strongly enough for an "
                        "option-buying setup.",
                }

            else:

                plan = build_trade_plan(
                    selected_option,
                    direction,
                    analysis_5m,
                    analysis_30m,
                    analysis_daily,
                    risk_profile
                )

            # ----------------------------------------------------
            # STEP 10:
            # SAFETY
            # ----------------------------------------------------

            safety = (
                beginner_safety_checks(
                    plan,
                    analysis_5m,
                    analysis_30m
                )
            )

            # ----------------------------------------------------
            # SAVE
            # ----------------------------------------------------

            st.session_state[
                "commodity_result"
            ] = {

                "resolved":
                    resolved,

                "option_df":
                    option_df,

                "underlying_key":
                    underlying_key,

                "underlying_quote":
                    underlying_quote,

                "underlying_price":
                    underlying_price,

                "analysis_5m":
                    analysis_5m,

                "analysis_30m":
                    analysis_30m,

                "analysis_daily":
                    analysis_daily,

                "technical_errors":
                    technical_errors,

                "structure":
                    structure,

                "call_option":
                    call_option,

                "put_option":
                    put_option,

                "selected_option":
                    selected_option,

                "plan":
                    plan,

                "safety":
                    safety,

                "analysis_time":
                    now_ist().strftime(
                        "%d %b %Y %I:%M:%S %p"
                    ),
            }

        except Exception as exc:

            st.session_state[
                "commodity_result"
            ] = {
                "error": str(exc)
            }


# ================================================================
# F&O-STYLE COMMODITY DASHBOARD — VISUAL ONLY
# ================================================================
st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#eef6ff 0%,#f7f9fc 42%,#eef8f4 100%);}
.block-container{max-width:1520px;padding-top:1.4rem;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#f0f7ff 0%,#f8fbff 48%,#f2fbf6 100%);border-right:1px solid #cbdcf2;}
section[data-testid="stSidebar"] h2{color:#12395b!important;}
.stButton>button{border:1px solid #93c5fd!important;border-radius:10px!important;font-weight:800!important;background:linear-gradient(180deg,#eff6ff,#dbeafe)!important;color:#174ea6!important;box-shadow:0 3px 10px rgba(37,99,235,.10)!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#2563eb,#1d4ed8)!important;color:#fff!important;border-color:#1d4ed8!important;}
.stTextInput input{border:2px solid #bfdbfe!important;border-radius:10px!important;background:#f8fbff!important;font-weight:700!important;}
.fo-hero{background:linear-gradient(135deg,#0b1f33 0%,#123b5d 55%,#155e75 100%);border-radius:20px;padding:24px 28px;color:#fff;box-shadow:0 10px 30px rgba(15,45,70,.14);margin:2px 0 18px;position:relative;overflow:hidden;}
.fo-hero:after{content:"";position:absolute;right:-90px;top:-120px;width:300px;height:300px;border-radius:50%;background:rgba(255,255,255,.055);}
.fo-hero-top{display:flex;align-items:center;justify-content:space-between;gap:18px;position:relative;z-index:1;}
.fo-brand{font-size:32px;font-weight:900;letter-spacing:-.6px;line-height:1.1;}.fo-tagline{font-size:15px;color:rgba(255,255,255,.72);margin-top:6px;}.fo-live{padding:10px 14px;border-radius:12px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.15);text-align:right;min-width:150px;}.fo-live b{display:block;font-size:15px}.fo-live small{display:block;margin-top:4px;font-size:13px;color:rgba(255,255,255,.68)}
.fo-section{font-size:15px;font-weight:900;letter-spacing:1px;color:#0f4c81;margin:22px 2px 9px;display:flex;align-items:center;gap:9px}.fo-section:before{content:"";width:7px;height:22px;border-radius:5px;background:linear-gradient(180deg,#2563eb,#06b6d4);display:inline-block}.fo-section:after{content:"";height:1px;background:linear-gradient(90deg,#bfdbfe,#e2e8f0,transparent);flex:1}
.fo-instrument{display:flex;align-items:end;justify-content:space-between;gap:14px;background:linear-gradient(100deg,#fff,#f0f7ff 55%,#effcf5);border:2px solid #c9dff4;border-radius:15px;padding:15px 18px;margin-bottom:14px;box-shadow:0 5px 15px rgba(15,23,42,.055)}.fo-symbol{font-size:27px;font-weight:900;color:#0f3b63}.fo-symbol-note{font-size:14px;color:#64748b;margin-top:3px}.fo-regime{font-size:13px;font-weight:850;color:#6d28d9;background:#ede9fe;border:1px solid #c4b5fd;padding:7px 10px;border-radius:20px}
.fo-metrics{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:10px}.fo-metric{background:#fff;border:1px solid #d7e2ee;border-radius:14px;padding:14px 13px;min-height:94px;box-shadow:0 5px 15px rgba(15,23,42,.055)}.fo-metric-label{font-size:12px;font-weight:900;color:#64748b;letter-spacing:.8px}.fo-metric-value{font-size:22px;font-weight:900;color:#172b4d;margin-top:8px;line-height:1.1;white-space:nowrap}.fo-metric-note{font-size:12px;color:#64748b;margin-top:7px}.fo-metric.spot{background:linear-gradient(145deg,#e0f2fe,#fff);border-color:#7dd3fc}.fo-metric.support{background:linear-gradient(145deg,#ecfdf5,#fff);border-left:6px solid #16a34a}.fo-metric.resistance{background:linear-gradient(145deg,#fff1f2,#fff);border-left:6px solid #dc2626}
.fo-decision{border-radius:19px;padding:22px;border:2px solid #e3e7eb;box-shadow:0 10px 26px rgba(15,23,42,.07);background:#fff}.fo-decision.call{background:linear-gradient(135deg,#dcfce7,#f0fdf4 45%,#fff);border-color:#4ade80}.fo-decision.put{background:linear-gradient(135deg,#ffe4e6,#fff1f2 45%,#fff);border-color:#fb7185}.fo-decision.neutral{background:linear-gradient(135deg,#fef3c7,#fffbeb 45%,#fff);border-color:#fbbf24}.fo-decision-row{display:flex;justify-content:space-between;align-items:center;gap:15px}.fo-decision-label{font-size:13px;font-weight:900;color:#64748b;letter-spacing:1px}.fo-decision-title{font-size:37px;font-weight:950;letter-spacing:-1px;margin-top:4px}.fo-decision.call .fo-decision-title{color:#087f3e}.fo-decision.put .fo-decision-title{color:#c81e3a}.fo-decision.neutral .fo-decision-title{color:#92400e}.fo-decision-note{font-size:14px;color:#667085;margin-top:6px}.fo-score-ring{min-width:100px;text-align:center;border-radius:15px;background:rgba(255,255,255,.72);border:1px solid rgba(0,0,0,.06);padding:11px 13px}.fo-score-ring b{display:block;font-size:28px;color:#182230}.fo-score-ring span{font-size:12px;color:#98a2b3;font-weight:800}
.fo-decision-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:17px}.fo-decision-cell{background:rgba(255,255,255,.72);border:1px solid rgba(16,42,67,.07);border-radius:11px;padding:11px;text-align:center}.fo-decision-cell span{display:block;font-size:12px;font-weight:900;color:#98a2b3}.fo-decision-cell b{display:block;font-size:18px;color:#182230;margin-top:5px}
.fo-plan-head{display:flex;align-items:center;justify-content:space-between;gap:14px;background:linear-gradient(100deg,#fff,#eff6ff);border:2px solid #c7d7ea;border-radius:16px 16px 0 0;padding:17px 19px}.fo-plan-action{font-size:13px;font-weight:900;color:#2563eb;letter-spacing:.8px}.fo-plan-contract{font-size:25px;font-weight:900;color:#0f3b63;margin-top:3px}.fo-plan-pop{font-size:22px;font-weight:900;color:#087f3e;background:#dcfce7;border:1px solid #86efac;border-radius:999px;padding:7px 12px;white-space:nowrap}
.fo-levels{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:7px;margin-top:9px}.fo-level{background:#fff;border:1px solid #d7e2ee;border-radius:13px;padding:11px 9px;min-height:112px;min-width:0;box-sizing:border-box;overflow:hidden;box-shadow:0 4px 12px rgba(15,23,42,.045)}.fo-level span{display:block;font-size:11px;color:#64748b;font-weight:900;letter-spacing:.45px;white-space:nowrap}.fo-level b{display:block;font-size:18px;color:#182230;margin-top:7px;white-space:nowrap}.fo-level small{display:block;font-size:11px;color:#64748b;margin-top:4px;line-height:1.25}.fo-level.entry{background:linear-gradient(180deg,#eff6ff,#fff);border-top:5px solid #2563eb}.fo-level.sl{background:linear-gradient(180deg,#fff1f2,#fff);border-top:5px solid #dc2626}.fo-level.t1{background:linear-gradient(180deg,#ecfdf5,#fff);border-top:5px solid #16a34a}.fo-level.t2{background:linear-gradient(180deg,#ecfeff,#fff);border-top:5px solid #0f766e}.fo-level.t3{background:linear-gradient(180deg,#ecfdf5,#fff);border-top:5px solid #059669}.fo-level.t4{background:linear-gradient(180deg,#d1fae5,#fff);border-top:5px solid #047857}.fo-level.greeks{background:linear-gradient(180deg,#f5f3ff,#fff);border-top:5px solid #7c3aed}.fo-level-pct{font-weight:900!important;font-size:13px!important}.fo-level-lot{font-weight:900!important;font-size:13px!important}.fo-level.sl .fo-level-pct,.fo-level.sl .fo-level-lot{color:#b91c1c!important}.fo-level.t1 .fo-level-pct,.fo-level.t1 .fo-level-lot,.fo-level.t2 .fo-level-pct,.fo-level.t2 .fo-level-lot,.fo-level.t3 .fo-level-pct,.fo-level.t3 .fo-level-lot,.fo-level.t4 .fo-level-pct,.fo-level.t4 .fo-level-lot{color:#15803d!important}
.fo-risk-box{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-top:12px}.fo-risk-cell{background:#fff;border:1px solid #d7e2ee;border-radius:11px;padding:10px;text-align:center}.fo-risk-cell span{display:block;font-size:10px;font-weight:900;color:#64748b}.fo-risk-cell b{display:block;font-size:17px;color:#172b4d;margin-top:4px}.fo-safety-banner{display:flex;align-items:center;justify-content:space-between;gap:14px;background:linear-gradient(135deg,#eff6ff,#f8fbff);border:1px solid #93c5fd;border-left:6px solid #2563eb;border-radius:14px;padding:14px 16px;margin:12px 0 16px}.fo-safety-banner b{font-size:16px;color:#12395b}.fo-safety-banner span{font-size:13px;color:#52657a;display:block;margin-top:3px}.fo-safety-badge{background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;padding:7px 10px;border-radius:20px;font-size:12px;font-weight:900}.fo-check-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.fo-check{background:#fff;border:2px solid #d7e2ee;border-radius:12px;padding:12px}.fo-check.pass{background:#ecfdf5;border-color:#4ade80}.fo-check.wait{background:#fffbeb;border-color:#fbbf24}.fo-check.fail{background:#fff1f2;border-color:#fb7185}.fo-check-top{display:flex;justify-content:space-between;gap:8px}.fo-check-name{font-weight:900;color:#172b4d}.fo-check-status{font-size:11px;font-weight:900;border-radius:8px;padding:4px 7px}.fo-check.pass .fo-check-status{background:#bbf7d0;color:#166534}.fo-check.wait .fo-check-status{background:#fde68a;color:#92400e}.fo-check.fail .fo-check-status{background:#fecdd3;color:#9f1239}.fo-check-detail{font-size:13px;color:#52657a;line-height:1.45;margin-top:7px}
.fo-card-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.fo-card{background:#fff;border:1px solid #d7e2ee;border-radius:14px;padding:14px;box-shadow:0 4px 12px rgba(15,23,42,.04)}.fo-card-label{font-size:11px;color:#64748b;font-weight:900}.fo-card-value{font-size:21px;color:#172b4d;font-weight:900;margin-top:5px}.fo-why{background:#fff;border:2px solid #d7e2ee;border-radius:15px;padding:14px 17px;box-shadow:0 3px 12px rgba(16,42,67,.03)}.fo-why-line{padding:8px 2px;border-bottom:1px solid #e2e8f0;font-size:14px;color:#344054;line-height:1.5}.fo-why-line:last-child{border-bottom:0}.fo-footer{margin-top:22px;padding:13px 15px;border-radius:12px;background:linear-gradient(90deg,#e0f2fe,#ecfdf5);border:1px solid #bfdbfe;color:#475569;font-size:12px;line-height:1.6;text-align:center}
.fo-card-value,.fo-decision-cell b,.fo-plan-contract,.fo-metric-value{overflow-wrap:anywhere;word-break:break-word;white-space:normal!important}
.fo-decision-cell{min-width:0}.fo-decision-cell b{line-height:1.25}
.fo-card{min-width:0;min-height:82px}.fo-card-grid{align-items:stretch}
.fo-snapshot-wrap{width:100%;overflow-x:auto;border:1px solid #d7e2ee;border-radius:14px;background:#fff;box-shadow:0 4px 12px rgba(15,23,42,.04)}
.fo-snapshot-table{width:100%;min-width:900px;border-collapse:collapse;font-size:13px}
.fo-snapshot-table th{position:sticky;top:0;background:#eaf3fb;color:#24425f;font-size:11px;font-weight:900;letter-spacing:.45px;padding:10px 9px;border-bottom:2px solid #c7d7ea;text-align:right;white-space:nowrap}
.fo-snapshot-table th:first-child,.fo-snapshot-table td:first-child{text-align:center}
.fo-snapshot-table td{padding:9px;border-bottom:1px solid #edf2f7;text-align:right;color:#243b53;white-space:nowrap}
.fo-snapshot-table tr:last-child td{border-bottom:0}
.fo-snapshot-table .ce{color:#087f3e;font-weight:900}.fo-snapshot-table .pe{color:#c81e3a;font-weight:900}
.fo-data-note{margin:8px 2px 0;color:#64748b;font-size:12px}
@media(max-width:1200px){.fo-metrics{grid-template-columns:repeat(4,minmax(0,1fr))}.fo-levels{grid-template-columns:repeat(4,minmax(0,1fr))}}

@media(max-width:1050px){.fo-metrics{grid-template-columns:repeat(4,minmax(0,1fr))}.fo-levels{grid-template-columns:repeat(4,minmax(0,1fr))}.fo-card-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:720px){.fo-hero-top,.fo-plan-head,.fo-decision-row{flex-direction:column;align-items:flex-start}.fo-metrics,.fo-levels,.fo-risk-box,.fo-card-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.fo-decision-grid{grid-template-columns:repeat(2,1fr)}.fo-check-grid{grid-template-columns:1fr}.fo-brand{font-size:25px}}
</style>
""", unsafe_allow_html=True)

# ================================================================
# DISPLAY
# ================================================================

result = st.session_state[
    "commodity_result"
]


if not result:
    st.info("Enter a commodity and click 'Analyze Commodity Options'.")
elif result.get("error"):
    st.error(result["error"])
else:
    resolved = result["resolved"]
    option_df = result["option_df"]
    underlying_quote = result["underlying_quote"]
    underlying_price = result["underlying_price"]
    structure = result["structure"]
    plan = result["plan"]
    selected = result["selected_option"]
    decision = plan.get("decision", "NO TRADE")

    # HERO / INSTRUMENT HEADER
    regime = "BULLISH" if result["analysis_5m"].get("trend") == "BULLISH" else ("BEARISH" if result["analysis_5m"].get("trend") == "BEARISH" else "MIXED")
    st.markdown(f"""
    <div class="fo-hero">
      <div class="fo-hero-top">
        <div><div class="fo-brand">🛢️ Commodity PRO Trader Assistant</div>
        <div class="fo-tagline">MCX commodity OPTIONS only · Live Upstox data · CE/PE · Greeks · OI · technicals · risk plan</div></div>
        <div class="fo-live"><b>● LIVE UPSTOX</b><small>{result.get('analysis_time','—')} IST</small></div>
      </div>
    </div>
    <div class="fo-instrument">
      <div><div class="fo-symbol">{resolved.get('symbol','—')} · MCX OPTIONS</div><div class="fo-symbol-note">Nearest active CE/PE series · Expiry {resolved.get('expiry','—')} · Futures are not used for trading</div></div>
      <div class="fo-regime">MARKET REGIME · {regime}</div>
    </div>
    """, unsafe_allow_html=True)

    spot = safe_float(underlying_price)
    prev = safe_float(underlying_quote.get("previous_close"), spot)
    day_change = safe_float(underlying_quote.get("change"), spot - prev if np.isfinite(prev) else 0)
    day_pct = day_change / prev * 100 if np.isfinite(prev) and prev else 0
    metrics = [
        ("SPOT", fmt_money(spot), f"{day_change:+.2f} ({day_pct:+.2f}%)", "spot"),
        ("EXPIRY", resolved.get("expiry","—"), "Nearest active option expiry", ""),
        ("PCR", fmt_number(structure.get("pcr"),2), "Put / Call OI", ""),
        ("SUPPORT", fmt_number(structure.get("support"),2), "Put OI zone", "support"),
        ("RESISTANCE", fmt_number(structure.get("resistance"),2), "Call OI zone", "resistance"),
        ("OPTIONS", str(len(option_df)), "Active CE + PE contracts", ""),
        ("LOT SIZE", fmt_number(selected.get("lot_size") if selected else np.nan,0), "1 option lot", ""),
    ]
    html="<div class='fo-metrics'>"
    for title,value,note,cls in metrics:
        html += f"<div class='fo-metric {cls}'><div class='fo-metric-label'>{title}</div><div class='fo-metric-value'>{value}</div><div class='fo-metric-note'>{note}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    if beginner_mode:
        st.markdown("<div class='fo-safety-banner'><div><b>🛡️ BEGINNER SAFETY + EXPLAINABILITY IS ON</b><span>Stricter execution filter, stronger confirmation and plain-English checks. This is decision support — not a guarantee of profit.</span></div><div class='fo-safety-badge'>SAFETY ON</div></div>", unsafe_allow_html=True)

    # DECISION
    st.markdown("<div class='fo-section'>TRADE DECISION</div>", unsafe_allow_html=True)
    action_class = "call" if decision == "CALL BUY" else ("put" if decision == "PUT BUY" else "neutral")
    pop = safe_float(plan.get("pop"),0)
    quality = safe_float(plan.get("quality"),0)
    contract = selected.get("trading_symbol","No executable option") if selected else "No executable setup"
    note = {"CALL BUY":"Directional upside setup — execute only when the displayed confirmation conditions are satisfied.","PUT BUY":"Directional downside setup — execute only when the displayed confirmation conditions are satisfied.","NO TRADE":"No option setup currently meets the minimum quality, alignment and safety gates."}.get(decision,"No executable setup.")
    st.markdown(f"""
    <div class="fo-decision {action_class}">
      <div class="fo-decision-row"><div><div class="fo-decision-label">ENGINE OUTPUT</div><div class="fo-decision-title">{decision}</div><div class="fo-decision-note">{note}</div></div><div class="fo-score-ring"><b>{quality:.0f}</b><span>QUALITY / 100</span></div></div>
      <div class="fo-decision-grid"><div class="fo-decision-cell"><span>OPTION</span><b>{contract}</b></div><div class="fo-decision-cell"><span>PoP</span><b>{pop:.1f}%</b></div><div class="fo-decision-cell"><span>MARKET</span><b>{regime}</b></div><div class="fo-decision-cell"><span>RISK PROFILE</span><b>{risk_profile.upper()}</b></div></div>
    </div>
    """, unsafe_allow_html=True)

    # OPTION / TRADE PLAN
    if selected:
        st.markdown("<div class='fo-section'>SELECTED OPTION</div>", unsafe_allow_html=True)
        selected_html=f"""<div class='fo-card-grid'>
          <div class='fo-card'><div class='fo-card-label'>OPTION TYPE</div><div class='fo-card-value'>{selected.get('option_type','—')}</div></div>
          <div class='fo-card'><div class='fo-card-label'>STRIKE</div><div class='fo-card-value'>{fmt_money(selected.get('strike'))}</div></div>
          <div class='fo-card'><div class='fo-card-label'>PREMIUM</div><div class='fo-card-value'>{fmt_money(selected.get('ltp'))}</div></div>
          <div class='fo-card'><div class='fo-card-label'>LOT SIZE</div><div class='fo-card-value'>{fmt_number(selected.get('lot_size'),0)}</div></div>
          <div class='fo-card'><div class='fo-card-label'>DELTA</div><div class='fo-card-value'>{fmt_number(selected.get('delta'),3)}</div></div>
          <div class='fo-card'><div class='fo-card-label'>IV</div><div class='fo-card-value'>{fmt_number(selected.get('iv'),2)}%</div></div>
          <div class='fo-card'><div class='fo-card-label'>OPTION PoP</div><div class='fo-card-value'>{fmt_number(selected.get('pop'),1)}%</div></div>
          <div class='fo-card'><div class='fo-card-label'>OI</div><div class='fo-card-value'>{fmt_number(selected.get('oi'),0)}</div></div>
        </div>"""
        st.markdown(selected_html, unsafe_allow_html=True)

    if decision in {"CALL BUY","PUT BUY"} and selected:
        st.markdown("<div class='fo-section'>SELECTED TRADE PLAN</div>", unsafe_allow_html=True)
        entry=safe_float(plan.get("entry")); qty=safe_float(plan.get("lot"),safe_float(selected.get("lot_size"),1))
        levels=[("ENTRY",plan.get("entry"),"Option premium","entry"),("STOP LOSS",plan.get("sl"),"Defined risk level","sl"),("TARGET 1",plan.get("t1"),"First profit level","t1"),("TARGET 2",plan.get("t2"),"Second profit level","t2"),("TARGET 3",plan.get("t3"),"Third profit level","t3"),("TARGET 4",plan.get("t4"),"Fourth profit level","t4"),("DELTA / IV",f"{safe_float(selected.get('delta')):.2f} / {safe_float(selected.get('iv')):.1f}%","Option characteristics","greeks")]
        def level_note(v):
            v=safe_float(v)
            if not np.isfinite(entry) or entry==0 or not np.isfinite(v): return ("—","—")
            delta=v-entry; pct=delta/abs(entry)*100; pnl=delta*qty
            return (f"{pct:+.2f}% from entry",f"₹{pnl:+,.0f} for 1 lot")
        cards="<div class='fo-plan-head'><div><div class='fo-plan-action'>SELECTED STRATEGY</div><div class='fo-plan-contract'>{decision} · {contract}</div></div><div class='fo-plan-pop'>PoP {pop:.1f}%</div></div><div class='fo-levels'>"
        for title,value,note_text,cls in levels:
            if cls=="greeks": pct_html=""; lot_html=""
            else:
                pct,lot=level_note(value); pct_html=f"<small class='fo-level-pct'>{pct}</small>"; lot_html=f"<small class='fo-level-lot'>{lot}</small>"
            cards += f"<div class='fo-level {cls}'><span>{title}</span><b>{fmt_money(value) if cls!='greeks' else value}</b><small>{note_text}</small>{pct_html}{lot_html}</div>"
        cards += "</div>"
        st.markdown(cards,unsafe_allow_html=True)
        st.markdown(f"<div class='fo-risk-box'><div class='fo-risk-cell'><span>PLANNED MAX LOSS · 1 LOT</span><b>{fmt_money(plan.get('max_loss'))}</b></div><div class='fo-risk-cell'><span>TARGET 1 · 1 LOT</span><b>{fmt_money(plan.get('t1_pnl'))}</b></div><div class='fo-risk-cell'><span>TARGET 2 · 1 LOT</span><b>{fmt_money(plan.get('t2_pnl'))}</b></div><div class='fo-risk-cell'><span>TARGET 3 · 1 LOT</span><b>{fmt_money(plan.get('t3_pnl'))}</b></div><div class='fo-risk-cell'><span>TARGET 4 · 1 LOT</span><b>{fmt_money(plan.get('t4_pnl'))}</b></div></div>",unsafe_allow_html=True)
        st.markdown(f"<div class='fo-why' style='margin-top:10px'><div class='fo-why-line'><b>EXIT RULE:</b> {plan.get('exit','Follow stop-loss and targets.')}</div><div class='fo-why-line'>Risk/Reward: T1 {safe_float(plan.get('rr1'),0):.2f}R · T2 {safe_float(plan.get('rr2'),0):.2f}R · T3 {safe_float(plan.get('t3'),0)-entry if np.isfinite(entry) else 0:.2f} premium move · T4 {safe_float(plan.get('t4'),0)-entry if np.isfinite(entry) else 0:.2f} premium move.</div></div>",unsafe_allow_html=True)
        st.caption(f"PoP source: {plan.get('pop_source','Model')}. PoP is informational and is not a guarantee of profit.")
    else:
        st.markdown("<div class='fo-why'><div class='fo-why-line'>⛔ No executable trade plan is displayed because the current setup did not pass the backend quality and safety gates.</div></div>",unsafe_allow_html=True)

    # BEGINNER CHECKS
    if beginner_mode:
        st.markdown("<div class='fo-section'>BEGINNER TRADE CHECK</div>", unsafe_allow_html=True)
        safe=result.get("safety",{}); checks=safe.get("checks",[])
        if checks:
            html="<div class='fo-check-grid'>"
            for status,title,message in checks:
                st_class={"PASS":"pass","WAIT":"wait","STOP":"fail"}.get(status,"wait")
                html+=f"<div class='fo-check {st_class}'><div class='fo-check-top'><div class='fo-check-name'>{title}</div><div class='fo-check-status'>{status}</div></div><div class='fo-check-detail'>{message}</div></div>"
            html+="</div>"; st.markdown(html,unsafe_allow_html=True)
        else:
            st.markdown("<div class='fo-why'><div class='fo-why-line'>No beginner safety checks were returned.</div></div>",unsafe_allow_html=True)

    # WHY
    st.markdown("<div class='fo-section'>WHY THIS DECISION?</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fo-why'><div class='fo-why-line'>{plan.get('reason','No additional explanation available.')}</div></div>",unsafe_allow_html=True)

    # TECHNICALS
    st.markdown("<div class='fo-section'>UNDERLYING TECHNICAL ANALYSIS</div>", unsafe_allow_html=True)
    for label,analysis in [("5 Minute",result["analysis_5m"]),("30 Minute",result["analysis_30m"]),("Daily",result["analysis_daily"])]:
        with st.expander(label,expanded=True):
            vals=[("TREND",analysis.get("trend","UNKNOWN")),("SCORE",f"{analysis.get('score',0)}/100"),("RSI",fmt_number(analysis.get("rsi"),1)),("ADX",fmt_number(analysis.get("adx"),1)),("ATR",fmt_money(analysis.get("atr"))), ("EMA20",fmt_money(analysis.get("ema20"))), ("EMA50",fmt_money(analysis.get("ema50"))), ("VWAP",fmt_money(analysis.get("vwap")))]
            html="<div class='fo-card-grid'>"+"".join(f"<div class='fo-card'><div class='fo-card-label'>{k}</div><div class='fo-card-value'>{v}</div></div>" for k,v in vals)+"</div>"
            st.markdown(html,unsafe_allow_html=True)

    if result.get("technical_errors"):
        st.warning("Some underlying technical data was temporarily unavailable. The option analysis was still completed using the data returned by Upstox.")

    st.markdown("<div class='fo-footer'>MCX OPTION-ONLY MODE · No MCX futures are resolved or traded. MCX CE/PE contracts are discovered from the official Upstox MCX instrument master with Instrument Search as fallback. The option contract's underlying_key is used only for underlying price and technical analysis. Data source: Upstox.</div>",unsafe_allow_html=True)

