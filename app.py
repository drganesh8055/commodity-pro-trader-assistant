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


# ================================================================
# DIRECT MCX OPTION SEARCH
#
# IMPORTANT:
# There is NO futures resolution here.
# ================================================================

@st.cache_data(
    ttl=90,
    show_spinner=False
)
def _search_instrument_rows(query, instrument_type=None, expiry=None):
    """
    Search Upstox Instrument Search API.

    MCX option discovery is deliberately broad:
    - MCX only
    - FO only
    - CE/PE only when requested
    - no futures are selected
    - expiry is optional because MCX commodity option expiry
      availability can differ by contract.

    The returned rows are filtered again locally.
    """
    params = {
        "query": query,
        "exchanges": "MCX",
        "segments": "FO",
        "page_number": 1,
        "records": 30,
    }

    if instrument_type:
        params["instrument_types"] = instrument_type

    if expiry:
        params["expiry"] = expiry

    payload = api_get(
        "/v2/instruments/search",
        params=params,
        timeout=30,
    )

    data = payload.get("data", [])
    return data if isinstance(data, list) else []


@st.cache_data(ttl=90, show_spinner=False)
def search_mcx_options(symbol, option_type, expiry_keyword=None):
    """
    Robust MCX CE/PE discovery.

    Important:
    We never ask Upstox for FUT and never return FUT rows.
    We search using several harmless textual variants because
    Upstox may index commodity names as CRUDE, CRUDE OIL,
    CRUDEOIL, etc.
    """
    symbol = normalize_symbol(symbol)
    option_type = str(option_type or "").upper().strip()

    if option_type not in ("CE", "PE"):
        return []

    # Text variants. The API supports partial, case-insensitive search.
    variants = {
        "GOLD": ["GOLD"],
        "GOLDM": ["GOLDM", "GOLD MINI", "GOLD"],
        "SILVER": ["SILVER"],
        "SILVERM": ["SILVERM", "SILVER MINI", "SILVER"],
        "CRUDEOIL": ["CRUDEOIL", "CRUDE OIL", "CRUDE"],
        "CRUDEOILMINI": [
            "CRUDEOILMINI",
            "CRUDE OIL MINI",
            "CRUDE MINI",
            "CRUDE",
        ],
        "NATURALGAS": ["NATURALGAS", "NATURAL GAS", "NAT GAS", "NATGAS"],
        "COPPER": ["COPPER"],
        "ZINC": ["ZINC"],
        "ALUMINIUM": ["ALUMINIUM", "ALUMINUM"],
        "LEAD": ["LEAD"],
        "NICKEL": ["NICKEL"],
    }.get(symbol, [symbol])

    rows = []

    # First pass: specific option type + optional expiry.
    for query in variants:
        try:
            rows.extend(
                _search_instrument_rows(
                    query,
                    instrument_type=option_type,
                    expiry=expiry_keyword,
                )
            )
        except Exception:
            continue

    # Second pass: if an expiry-filtered search returned nothing,
    # search without expiry and filter dates locally.
    if not rows:
        for query in variants:
            try:
                rows.extend(
                    _search_instrument_rows(
                        query,
                        instrument_type=option_type,
                        expiry=None,
                    )
                )
            except Exception:
                continue

    # Final safety filter: only MCX_FO + requested CE/PE.
    cleaned = []
    seen = set()

    for row in rows:
        if not isinstance(row, dict):
            continue

        exchange = str(row.get("exchange", "")).upper()
        segment = str(row.get("segment", "")).upper()
        inst_type = str(row.get("instrument_type", "")).upper()

        if exchange != "MCX":
            continue

        if segment != "MCX_FO":
            continue

        if inst_type != option_type:
            continue

        key = str(row.get("instrument_key", "")).strip()

        if not key or key in seen:
            continue

        # Explicitly reject anything that looks like a future.
        trading_symbol = str(row.get("trading_symbol", "")).upper()
        if inst_type == "FUT" or " FUT " in f" {trading_symbol} ":
            continue

        seen.add(key)
        cleaned.append(row)

    return cleaned

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
    if not value:
        return ""

    return str(value)[:10]


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
            1.60
        ),

        "Balanced": (
            0.55,
            1.35,
            1.90
        ),

        "Aggressive": (
            0.45,
            1.50,
            2.20
        ),
    }

    sl_percent, t1_percent, t2_percent = (
        profiles[
            risk_profile
        ]
    )

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
        "lot": lot_size,
        "risk_per_unit": risk_per_unit,
        "t1_profit_per_unit":
            profit_t1_per_unit,
        "t2_profit_per_unit":
            profit_t2_per_unit,
        "max_loss": max_loss,
        "t1_pnl": t1_profit,
        "t2_pnl": t2_profit,
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
# DISPLAY
# ================================================================

result = st.session_state[
    "commodity_result"
]


if not result:

    st.info(
        "Enter a commodity and click "
        "'Analyze Commodity Options'."
    )

elif result.get("error"):

    st.error(
        result["error"]
    )

else:

    resolved = result[
        "resolved"
    ]

    option_df = result[
        "option_df"
    ]

    underlying_quote = result[
        "underlying_quote"
    ]

    underlying_price = result[
        "underlying_price"
    ]

    structure = result[
        "structure"
    ]

    plan = result[
        "plan"
    ]

    selected = result[
        "selected_option"
    ]


    # ============================================================
    # OPTION SERIES
    # ============================================================

    st.markdown(
        "## 📅 Selected MCX Option Series"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Commodity",
        resolved.get(
            "symbol",
            "—"
        )
    )

    c2.metric(
        "Expiry",
        resolved.get(
            "expiry",
            "—"
        )
    )

    c3.metric(
        "Underlying LTP",
        fmt_money(
            underlying_price
        )
    )

    c4.metric(
        "Options Found",
        str(
            len(option_df)
        )
    )


    # ============================================================
    # LIVE UNDERLYING
    # ============================================================

    st.markdown(
        "## 📊 Live Underlying Market"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "LTP",
        fmt_money(
            underlying_quote.get(
                "ltp"
            )
        )
    )

    c2.metric(
        "Day High",
        fmt_money(
            underlying_quote.get(
                "high"
            )
        )
    )

    c3.metric(
        "Day Low",
        fmt_money(
            underlying_quote.get(
                "low"
            )
        )
    )

    c4.metric(
        "Volume",
        fmt_number(
            underlying_quote.get(
                "volume"
            ),
            0
        )
    )


    # ============================================================
    # OPTION STRUCTURE
    # ============================================================

    st.markdown(
        "## 🧱 Option Market Structure"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ATM Strike",
        fmt_number(
            structure.get(
                "atm"
            ),
            2
        )
    )

    c2.metric(
        "PCR (OI)",
        fmt_number(
            structure.get(
                "pcr"
            ),
            2
        )
    )

    c3.metric(
        "Put OI / Support",
        fmt_number(
            structure.get(
                "support"
            ),
            2
        )
    )

    c4.metric(
        "Call OI / Resistance",
        fmt_number(
            structure.get(
                "resistance"
            ),
            2
        )
    )


    # ============================================================
    # DECISION
    # ============================================================

    st.markdown(
        "## 🎯 Selected Option Trade Plan"
    )

    decision = plan.get(
        "decision",
        "NO TRADE"
    )

    if decision == "CALL BUY":

        st.success(
            "🟢 CALL BUY SETUP"
        )

    elif decision == "PUT BUY":

        st.warning(
            "🔴 PUT BUY SETUP"
        )

    else:

        st.info(
            "⚪ NO TRADE"
        )


    # ============================================================
    # SELECTED OPTION
    # ============================================================

    if selected:

        st.markdown(
            f"### "
            f"{selected.get('trading_symbol', 'Selected Option')}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Option",
            selected.get(
                "option_type",
                "—"
            )
        )

        c2.metric(
            "Strike",
            fmt_money(
                selected.get(
                    "strike"
                )
            )
        )

        c3.metric(
            "Premium",
            fmt_money(
                selected.get(
                    "ltp"
                )
            )
        )

        c4.metric(
            "Lot Size",
            fmt_number(
                selected.get(
                    "lot_size"
                ),
                0
            )
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Delta",
            fmt_number(
                selected.get(
                    "delta"
                ),
                3
            )
        )

        iv = safe_float(
            selected.get(
                "iv"
            )
        )

        c2.metric(
            "IV",
            f"{iv:.2f}%"
            if np.isfinite(iv)
            else "—"
        )

        pop = safe_float(
            selected.get(
                "pop"
            )
        )

        c3.metric(
            "Option PoP",
            f"{pop:.1f}%"
            if np.isfinite(pop)
            else "—"
        )

        c4.metric(
            "OI",
            fmt_number(
                selected.get(
                    "oi"
                ),
                0
            )
        )


    # ============================================================
    # TRADE LEVELS
    # ============================================================

    if decision in {
        "CALL BUY",
        "PUT BUY"
    }:

        st.markdown(
            "### 💰 Option Trade Levels"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Entry",
            fmt_money(
                plan.get(
                    "entry"
                )
            )
        )

        c2.metric(
            "Stop Loss",
            fmt_money(
                plan.get(
                    "sl"
                )
            )
        )

        c3.metric(
            "Target 1",
            fmt_money(
                plan.get(
                    "t1"
                )
            )
        )

        c4.metric(
            "Target 2",
            fmt_money(
                plan.get(
                    "t2"
                )
            )
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Max Loss / 1 Lot",
            fmt_money(
                plan.get(
                    "max_loss"
                )
            )
        )

        c2.metric(
            "T1 Potential / 1 Lot",
            fmt_money(
                plan.get(
                    "t1_pnl"
                )
            )
        )

        c3.metric(
            "T2 Potential / 1 Lot",
            fmt_money(
                plan.get(
                    "t2_pnl"
                )
            )
        )

        c4.metric(
            "Model PoP",
            f"{safe_float(plan.get('pop'), 0):.1f}%"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Risk / Reward T1",
            (
                f"{safe_float(plan.get('rr1'), 0):.2f}R"
            )
        )

        c2.metric(
            "Risk / Reward T2",
            (
                f"{safe_float(plan.get('rr2'), 0):.2f}R"
            )
        )

        c3.metric(
            "Quality",
            (
                f"{safe_float(plan.get('quality'), 0):.0f}/100"
            )
        )

        st.caption(
            "PoP source: "
            f"{plan.get('pop_source', 'Model')}. "
            "PoP is not a guarantee of profit."
        )


    # ============================================================
    # REASON
    # ============================================================

    st.markdown(
        "### 💡 Why this setup?"
    )

    st.write(
        plan.get(
            "reason",
            "No additional explanation available."
        )
    )


    # ============================================================
    # BEGINNER SAFETY
    # ============================================================

    if beginner_mode:

        st.markdown(
            "## 🛡️ Beginner Trade Check"
        )

        if result[
            "safety"
        ]["safe"]:

            st.success(
                "Passed the additional beginner "
                "safety checks. This still does "
                "not guarantee a profitable trade."
            )

        else:

            st.error(
                "Did not pass all beginner safety "
                "checks. Treat this as WAIT / "
                "NO TRADE until conditions improve."
            )

        for (
            status,
            title,
            message
        ) in result[
            "safety"
        ]["checks"]:

            if status == "PASS":

                st.success(
                    f"**{title} — PASS**\n\n"
                    f"{message}"
                )

            elif status == "WAIT":

                st.warning(
                    f"**{title} — WAIT**\n\n"
                    f"{message}"
                )

            else:

                st.error(
                    f"**{title} — STOP**\n\n"
                    f"{message}"
                )


    # ============================================================
    # OPTION SNAPSHOT
    # ============================================================

    st.markdown(
        "## 📋 MCX Option Snapshot"
    )

    display = option_df[
        [
            "option_type",
            "strike",
            "ltp",
            "oi",
            "volume",
            "iv",
            "delta",
            "pop"
        ]
    ].copy()

    display = display.rename(
        columns={
            "option_type": "Type",
            "strike": "Strike",
            "ltp": "LTP",
            "oi": "OI",
            "volume": "Volume",
            "iv": "IV",
            "delta": "Delta",
            "pop": "PoP",
        }
    )

    st.dataframe(
        display.sort_values(
            [
                "Strike",
                "Type"
            ]
        ),
        use_container_width=True,
        hide_index=True
    )


    # ============================================================
    # TECHNICALS
    # ============================================================

    st.markdown(
        "## 📈 Underlying Technical Analysis"
    )

    technical_sets = [
        (
            "5 Minute",
            result["analysis_5m"]
        ),
        (
            "30 Minute",
            result["analysis_30m"]
        ),
        (
            "Daily",
            result["analysis_daily"]
        ),
    ]

    for label, analysis in technical_sets:

        with st.expander(
            label,
            expanded=True
        ):

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Trend",
                analysis.get(
                    "trend",
                    "UNKNOWN"
                )
            )

            c2.metric(
                "Score",
                f"{analysis.get('score', 0)}/100"
            )

            c3.metric(
                "RSI",
                fmt_number(
                    analysis.get(
                        "rsi"
                    ),
                    1
                )
            )

            c4.metric(
                "ADX",
                fmt_number(
                    analysis.get(
                        "adx"
                    ),
                    1
                )
            )

            c5.metric(
                "ATR",
                fmt_money(
                    analysis.get(
                        "atr"
                    )
                )
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "EMA20",
                fmt_money(
                    analysis.get(
                        "ema20"
                    )
                )
            )

            c2.metric(
                "EMA50",
                fmt_money(
                    analysis.get(
                        "ema50"
                    )
                )
            )

            c3.metric(
                "VWAP",
                fmt_money(
                    analysis.get(
                        "vwap"
                    )
                )
            )


    # ============================================================
    # TECHNICAL DATA WARNINGS
    # ============================================================

    technical_errors = result.get(
        "technical_errors",
        []
    )

    if technical_errors:

        st.warning(
            "Some underlying technical data "
            "was temporarily unavailable. "
            "The option analysis was still "
            "completed using the data that "
            "Upstox returned."
        )


    # ============================================================
    # IMPORTANT
    # ============================================================

    st.markdown(
        "## ℹ️ Important"
    )

    st.caption(
        "This is an MCX OPTION-ONLY application. "
        "It does not resolve an MCX futures "
        "contract for trading."
    )

    st.caption(
        "MCX CE/PE contracts are discovered "
        "directly through Upstox Instrument Search."
    )

    st.caption(
        "The option contract's underlying_key "
        "is used only for underlying price and "
        "technical analysis."
    )

    st.caption(
        "Upstox currently documents the standard "
        "Put/Call Option Chain endpoint as "
        "unavailable for MCX. Therefore this "
        "application reconstructs the option view "
        "from individual MCX CE/PE contracts, "
        "live quotes and Option Greek data."
    )

    st.caption(
        "Last analysis: "
        f"{result.get('analysis_time', '—')} IST"
    )

    st.caption(
        "Data source: Upstox."
    )
