# Commodity PRO Trader Assistant v1
# Standalone Streamlit app for MCX commodity futures using Upstox APIs.
# Required Streamlit secret: UPSTOX_ACCESS_TOKEN
# Run: streamlit run Commodity_PRO_Trader_Assistant_v1.py

import time
import threading
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Commodity PRO Trader Assistant", page_icon="🛢️", layout="wide")

API_BASE = "https://api.upstox.com"
IST = ZoneInfo("Asia/Kolkata")
API_LOCK = threading.Lock()
MIN_API_GAP = 0.50
_LAST_API_CALL = 0.0

ALIASES = {
    "CRUDE": "CRUDEOIL", "CRUDE OIL": "CRUDEOIL", "CRUDEOIL": "CRUDEOIL",
    "NATURAL GAS": "NATURALGAS", "NAT GAS": "NATURALGAS", "NATGAS": "NATURALGAS",
    "GOLD": "GOLD", "SILVER": "SILVER", "COPPER": "COPPER", "ZINC": "ZINC",
    "ALUMINIUM": "ALUMINIUM", "ALUMINUM": "ALUMINIUM", "LEAD": "LEAD", "NICKEL": "NICKEL",
}

COMMON = ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "COPPER", "ZINC", "ALUMINIUM", "LEAD", "NICKEL"]


def now_ist(): return datetime.now(IST)

def sf(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def money(v):
    x = sf(v)
    return "—" if not np.isfinite(x) else f"₹{x:,.2f}"

def num(v, d=2):
    x = sf(v)
    return "—" if not np.isfinite(x) else f"{x:,.{d}f}"

def normalize(q):
    raw = " ".join(str(q or "").strip().upper().split())
    return ALIASES.get(raw, raw.replace(" ", ""))


def token():
    t = st.secrets.get("UPSTOX_ACCESS_TOKEN", "")
    if not t:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is missing. Add it under Streamlit Secrets.")
    return str(t).strip()


def api_get(path, params=None, timeout=20):
    global _LAST_API_CALL
    with API_LOCK:
        gap = time.time() - _LAST_API_CALL
        if gap < MIN_API_GAP: time.sleep(MIN_API_GAP - gap)
        r = requests.get(API_BASE + path, params=params or {}, headers={
            "Accept": "application/json", "Authorization": f"Bearer {token()}"
        }, timeout=timeout)
        _LAST_API_CALL = time.time()
    if r.status_code == 401: raise RuntimeError("Upstox access token is invalid or expired.")
    if r.status_code == 429: raise RuntimeError("Upstox API rate limit reached. Please retry shortly.")
    if not r.ok:
        try: detail = r.json()
        except Exception: detail = r.text[:500]
        raise RuntimeError(f"Upstox API error {r.status_code}: {detail}")
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def instrument_search(q):
    # The Search Instruments API returns futures contracts matching the search.
    data = api_get("/v2/instruments/search", {"query": q, "exchanges": "MCX", "segments": "MCX_FO", "page_number": 1, "records": 50})
    return data.get("data", []) if isinstance(data, dict) else []


def resolve_future(query):
    n = normalize(query)
    queries = list(dict.fromkeys([query.strip(), n]))
    all_rows = []
    for q in queries:
        if q:
            try: all_rows += instrument_search(q)
            except Exception: pass
    unique = {}
    for x in all_rows:
        k = x.get("instrument_key") or x.get("trading_symbol")
        if k: unique[k] = x
    rows = list(unique.values())
    today = date.today().isoformat()
    rows = [x for x in rows if str(x.get("segment", "")).upper() == "MCX_FO"]
    rows = [x for x in rows if x.get("expiry") and str(x.get("expiry"))[:10] >= today] or rows
    rows.sort(key=lambda x: str(x.get("expiry", "9999-99-99")))
    # Prefer a future instrument. Search result documentation identifies future contracts.
    fut = [x for x in rows if "FUT" in str(x.get("trading_symbol", "")).upper() or str(x.get("instrument_type", "")).upper() in {"FUTCOM", "FUT"}]
    return (fut[0] if fut else (rows[0] if rows else None)), rows


def quote(key):
    return api_get("/v3/market-quote/quotes", {"instrument_key": key}).get("data", {})


def extract_quote(data, key):
    x = data.get(key) if isinstance(data, dict) else None
    if x is None and data: x = next(iter(data.values()))
    x = x or {}
    o = x.get("ohlc") or {}
    return {
        "ltp": sf(x.get("last_price", x.get("ltp"))),
        "open": sf(o.get("open")), "high": sf(o.get("high")), "low": sf(o.get("low")), "close": sf(o.get("close")),
        "volume": sf(x.get("volume", o.get("volume")), 0), "oi": sf(x.get("oi")), "previous_oi": sf(x.get("previous_oi", x.get("prev_oi"))),
        "prev_close": sf(x.get("prev_close_price", o.get("close"))),
    }


def candles(key, unit, interval, days_back):
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days_back)).isoformat()
    enc = requests.utils.quote(key, safe="")
    path = f"/v3/historical-candle/{enc}/{unit}/{interval}/{end}/{start}"
    raw = api_get(path)
    rows = raw.get("data", {}).get("candles", [])
    out = []
    for c in rows:
        if len(c) < 6: continue
        out.append({"timestamp": pd.to_datetime(c[0], errors="coerce"), "open": sf(c[1]), "high": sf(c[2]), "low": sf(c[3]), "close": sf(c[4]), "volume": sf(c[5], 0), "oi": sf(c[6]) if len(c) > 6 else np.nan})
    df = pd.DataFrame(out)
    if df.empty: return df
    return df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_intraday(key, minutes=5):
    enc = requests.utils.quote(key, safe="")
    raw = api_get(f"/v3/historical-candle/intraday/{enc}/minutes/{minutes}")
    return parse_candles(raw)


def parse_candles(raw):
    rows = raw.get("data", {}).get("candles", [])
    out=[]
    for c in rows:
        if len(c)>=6:
            out.append({"timestamp":pd.to_datetime(c[0], errors="coerce"),"open":sf(c[1]),"high":sf(c[2]),"low":sf(c[3]),"close":sf(c[4]),"volume":sf(c[5],0),"oi":sf(c[6]) if len(c)>6 else np.nan})
    df=pd.DataFrame(out)
    return df.dropna(subset=["timestamp","close"]).sort_values("timestamp").reset_index(drop=True) if not df.empty else df


def add_indicators(df):
    x=df.copy()
    d=x.close.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean(); al=loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs=ag/al.replace(0,np.nan); x["rsi"]=100-100/(1+rs)
    x["ema20"]=x.close.ewm(span=20,adjust=False).mean(); x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
    pc=x.close.shift(1); tr=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1)
    x["atr14"]=tr.ewm(alpha=1/14,adjust=False).mean()
    typ=(x.high+x.low+x.close)/3; v=x.volume.fillna(0); cv=v.cumsum().replace(0,np.nan); x["vwap"]=(typ*v).cumsum()/cv
    up=x.high.diff(); dn=-x.low.diff(); plus=np.where((up>dn)&(up>0),up,0); minus=np.where((dn>up)&(dn>0),dn,0)
    atr=x["atr14"]; pdi=100*pd.Series(plus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan); mdi=100*pd.Series(minus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); x["adx"]=dx.ewm(alpha=1/14,adjust=False).mean()
    x["vol_ma20"]=x.volume.rolling(20).mean()
    return x


def resample(df, rule):
    if df.empty:return df
    x=df.set_index("timestamp").resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum","oi":"last"}).dropna(subset=["close"]).reset_index()
    return x


def tf_analysis(df):
    if df.empty or len(df)<30:return {"trend":"UNKNOWN","score":0,"rsi":np.nan,"adx":np.nan,"atr":np.nan,"ema20":np.nan,"ema50":np.nan,"vwap":np.nan,"volume_confirmed":False,"close":np.nan}
    x=add_indicators(df); z=x.iloc[-1]
    bull=z.close>z.ema20 and z.ema20>=z.ema50 and z.rsi>=52
    bear=z.close<z.ema20 and z.ema20<=z.ema50 and z.rsi<=48
    trend="BULLISH" if bull else "BEARISH" if bear else "NEUTRAL"
    score=0
    score+=20 if z.close>z.ema20 else 0; score+=20 if z.ema20>z.ema50 else 0
    score+=15 if z.rsi>=52 else 15 if z.rsi<=48 else 0; score+=15 if z.adx>=20 else 0
    score+=10 if z.close>=z.vwap else 0; score+=5 if sf(z.vol_ma20,0)>0 and z.volume>=z.vol_ma20 else 0
    return {"trend":trend,"score":min(100,int(score)),"rsi":sf(z.rsi),"adx":sf(z.adx),"atr":sf(z.atr14),"ema20":sf(z.ema20),"ema50":sf(z.ema50),"vwap":sf(z.vwap),"volume_confirmed":sf(z.vol_ma20,0)>0 and z.volume>=z.vol_ma20,"close":sf(z.close)}


def levels(df):
    if df.empty:return np.nan,np.nan
    r=df.tail(min(80,len(df))); return sf(r.low.min()),sf(r.high.max())


def build_plan(q,a5,a30,ad,lot,risk):
    p=q["ltp"]; atr=a5["atr"]
    if not np.isfinite(p): return {"decision":"NO TRADE","reason":"Live price unavailable."}
    if not np.isfinite(atr) or atr<=0: atr=max(p*0.005,0.01)
    mult={"Conservative":(.75,1.25,1.75),"Balanced":(.65,1.35,1.90),"Aggressive":(.55,1.50,2.20)}[risk]
    quality=int(.45*a5["score"]+.35*a30["score"]+.20*ad["score"])
    bull=a5["trend"]==a30["trend"]=="BULLISH"; bear=a5["trend"]==a30["trend"]=="BEARISH"
    if bull: decision="BUY"; entry=p; sl=p-atr*mult[0]; t1=p+atr*mult[1]; t2=p+atr*mult[2]
    elif bear: decision="SELL"; entry=p; sl=p+atr*mult[0]; t1=p-atr*mult[1]; t2=p-atr*mult[2]
    else: decision="NO TRADE"; entry=sl=t1=t2=np.nan
    if decision=="NO TRADE":
        why=[]
        if not (bull or bear):why.append("5-minute and 30-minute trends are not aligned.")
        if quality<55:why.append(f"Quality score is only {quality}/100.")
        return {"decision":decision,"quality":quality,"entry":entry,"sl":sl,"t1":t1,"t2":t2,"lot":lot,"reason":" ".join(why) or "Setup did not pass the trade-quality gate."}
    risk_u=abs(entry-sl); r1=abs(t1-entry)/risk_u if risk_u else np.nan; r2=abs(t2-entry)/risk_u if risk_u else np.nan
    pop=float(np.clip(50+(quality-50)*.55+(3 if a5["volume_confirmed"] else 0)+(3 if ad["trend"]==a30["trend"] and ad["trend"] in ("BULLISH","BEARISH") else 0),50,82))
    return {"decision":decision,"quality":quality,"entry":entry,"sl":sl,"t1":t1,"t2":t2,"lot":lot,"rr1":r1,"rr2":r2,"max_loss":risk_u*lot,"t1_pnl":abs(t1-entry)*lot,"t2_pnl":abs(t2-entry)*lot,"pop":pop,"reason":"Trend, momentum and timeframe alignment support the setup."}


def safety(plan,a5,a30,ad):
    checks=[]; aligned=a5["trend"]==a30["trend"] and a5["trend"] in ("BULLISH","BEARISH")
    checks.append(("PASS" if plan["decision"] in ("BUY","SELL") else "STOP","Trade direction","A clear trade direction exists." if plan["decision"] in ("BUY","SELL") else "No clear trade direction."))
    checks.append(("PASS" if aligned else "STOP","Timeframe agreement","5-minute and 30-minute trends agree." if aligned else "5-minute and 30-minute trends do not agree."))
    checks.append(("PASS" if a5["volume_confirmed"] else "WAIT","Volume confirmation","Recent volume confirms activity." if a5["volume_confirmed"] else "Volume has not confirmed the move."))
    rr=sf(plan.get("rr1")); checks.append(("PASS" if np.isfinite(rr) and rr>=1 else "STOP","Risk / Reward",f"Target 1 is about {rr:.1f}R." if np.isfinite(rr) else "Risk/reward unavailable."))
    q=sf(plan.get("quality"),0); checks.append(("PASS" if q>=60 else "STOP","Setup quality",f"Quality score is {q:.0f}/100."))
    return {"safe":all(x[0]!="STOP" for x in checks),"checks":checks}


def css():
    st.markdown("""<style>
    .title{font-size:34px;font-weight:800}.sub{color:#6b7280;margin-bottom:18px}
    .box{border:1px solid rgba(128,128,128,.22);border-radius:12px;padding:14px}
    </style>""",unsafe_allow_html=True)

css()
st.markdown('<div class="title">🛢️ Commodity PRO Trader Assistant</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">Standalone MCX futures analysis • live Upstox data • technicals • risk plan • beginner safety</div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    q=st.text_input("Commodity", "Crude Oil", placeholder="Crude Oil / Gold / Silver / Copper")
    risk=st.selectbox("Risk Profile", ["Conservative","Balanced","Aggressive"], index=1)
    beginner=st.checkbox("Beginner Safety + Explainability", True)
    analyze=st.button("🔎 Analyze Live Commodity", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Examples: Crude Oil • Natural Gas • Gold • Silver • Copper • Zinc • Aluminium • Lead • Nickel")
    st.caption("Decision-support only. Model PoP is an estimate, not a guarantee.")

if "result" not in st.session_state: st.session_state.result=None
if analyze:
    with st.spinner("Resolving MCX futures contract and analyzing live market..."):
        try:
            contract,candidates=resolve_future(q)
            if not contract: raise RuntimeError(f"Could not resolve an active MCX futures contract for '{q}'.")
            key=contract.get("instrument_key")
            qq=extract_quote(quote(key),key)
            d1=get_intraday(key,1)
            d5=get_intraday(key,5)
            # V3 historical daily data supplies a real daily timeframe; intraday is used for 5m.
            day=candles(key,"days",1,370)
            if d5.empty: d5=d1
            d30=resample(d1 if not d1.empty else d5,"30min")
            if d30.empty or len(d30)<30: d30=resample(d5,"30min")
            a5=tf_analysis(d5); a30=tf_analysis(d30); ad=tf_analysis(day)
            sup,res=levels(d5)
            lot=sf(contract.get("lot_size",contract.get("minimum_lot",1)),1)
            plan=build_plan(qq,a5,a30,ad,lot,risk)
            st.session_state.result={"contract":contract,"quote":qq,"a5":a5,"a30":a30,"ad":ad,"sup":sup,"res":res,"plan":plan,"safety":safety(plan,a5,a30,ad),"time":now_ist().strftime("%d %b %Y %I:%M:%S %p")}
        except Exception as e:
            st.session_state.result={"error":str(e)}

r=st.session_state.result
if not r:
    st.info("Enter an MCX commodity in the sidebar and click Analyze Live Commodity.")
elif r.get("error"):
    st.error(r["error"])
else:
    c=r["contract"]; qq=r["quote"]; p=r["plan"]
    st.markdown("## Selected Contract")
    a,b,c1,d=st.columns(4)
    a.metric("Commodity",c.get("name",c.get("trading_symbol","—")))
    b.metric("Contract",c.get("trading_symbol","—"))
    c1.metric("Expiry",str(c.get("expiry","—"))[:10])
    d.metric("Lot Size",num(c.get("lot_size",1),0))
    st.markdown("## Live Market")
    a,b,c1,d=st.columns(4)
    a.metric("LTP",money(qq.get("ltp"))); b.metric("Day High",money(qq.get("high"))); c1.metric("Day Low",money(qq.get("low"))); d.metric("OI",num(qq.get("oi"),0))
    st.markdown("## Trade Plan")
    if p["decision"]=="BUY": st.success("🟢 BUY SETUP")
    elif p["decision"]=="SELL": st.warning("🔴 SELL SETUP")
    else: st.info("⚪ NO TRADE")
    a,b,c1,d=st.columns(4); a.metric("Entry",money(p.get("entry"))); b.metric("Stop Loss",money(p.get("sl"))); c1.metric("Target 1",money(p.get("t1"))); d.metric("Target 2",money(p.get("t2")))
    if p["decision"] in ("BUY","SELL"):
        a,b,c1,d=st.columns(4); a.metric("Max Loss / 1 Lot",money(p.get("max_loss"))); b.metric("T1 Potential / 1 Lot",money(p.get("t1_pnl"))); c1.metric("T2 Potential / 1 Lot",money(p.get("t2_pnl"))); d.metric("Model PoP",f'{p.get("pop",0):.1f}%')
        a,b=st.columns(2); a.metric("Risk / Reward T1",f'{p.get("rr1",0):.2f}R'); b.metric("Risk / Reward T2",f'{p.get("rr2",0):.2f}R')
        st.caption("Model PoP is a rule-based estimate and is not a guarantee of profit or a market-provided probability.")
    st.markdown("### Why this setup?"); st.write(p.get("reason"))
    if beginner:
        st.markdown("## 🛡️ Beginner Trade Check")
        if r["safety"]["safe"]: st.success("Passed the additional beginner safety checks. This still does not guarantee a profitable trade.")
        else: st.error("Did not pass all beginner safety checks. Treat this as WAIT / NO TRADE until conditions improve.")
        for status,title,msg in r["safety"]["checks"]:
            if status=="PASS": st.success(f"**{title} — PASS**  \\n{msg}")
            elif status=="WAIT": st.warning(f"**{title} — WAIT**  \\n{msg}")
            else: st.error(f"**{title} — STOP**  \\n{msg}")
    st.markdown("## Market Structure")
    for label,x in [("5 Minute",r["a5"]),("30 Minute",r["a30"]),("Daily",r["ad"])]:
        with st.expander(label,expanded=True):
            a,b,c,d,e=st.columns(5); a.metric("Trend",x["trend"]); b.metric("RSI",num(x["rsi"],1)); c.metric("ADX",num(x["adx"],1)); d.metric("EMA20",money(x["ema20"])); e.metric("EMA50",money(x["ema50"]))
    a,b=st.columns(2); a.metric("Recent Support",money(r["sup"])); b.metric("Recent Resistance",money(r["res"]))
    st.caption(f"Last analysis: {r['time']} IST. Instrument, contract, quote and candle data are sourced through Upstox.")
    st.caption("Note: Upstox documents the put/call option-chain endpoint as unavailable for MCX, so this v1 deliberately focuses on MCX futures rather than pretending to have MCX option-chain data.")
