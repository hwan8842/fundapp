import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Tuple, Optional
import json, math, time
import numpy as np
import altair as alt

# ==============================
# Globals & Constants
# ==============================
DB_PATH = "fundops.db"
OPERATOR_NAME = "조석환"
SUPPORTED_CCY = ["KRW", "USD"]
USD_KRW = 1400.0

st.set_page_config(page_title="FundOps", layout="wide")

# ==============================
# Global Styles (all buttons white)
# ==============================
st.markdown("""
<style>
  :root{ --header-h:60px; --tabbar-h:64px; --page-pad:16px; }
  header[data-testid="stHeader"]{
    position:fixed; top:0; left:0; right:0; z-index:100; height:var(--header-h);
    backdrop-filter: blur(4px); background: rgba(255,255,255,.85);
    border-bottom:1px solid #e5e7eb;
  }
  .block-container{ padding-top: calc(var(--header-h) + var(--tabbar-h) + var(--page-pad)) !important; padding-bottom:1.25rem;}
  div[role="tablist"]{
    position:fixed; top:var(--header-h); left:0; right:0; z-index:95;
    display:flex; gap:10px; padding:10px 12px; background:#fff; border-bottom:1px solid #e5e7eb; overflow-x:auto; scrollbar-width:thin;
    justify-content:center;
  }
  button[role="tab"]{
    padding:14px 22px; min-height:52px; border-radius:12px; border:1px solid #e5e7eb;
    background:#f9fafb; color:#111827; font-size:1.06rem; font-weight:700; letter-spacing:.2px; white-space:nowrap;
  }
  button[role="tab"]:hover{ background:#f3f4f6;}
  button[role="tab"][aria-selected="true"]{ background:#0f172a; color:#fff; border-color:#0f172a; box-shadow:0 2px 8px rgba(0,0,0,.08);}
  .stTextInput input, .stNumberInput input, .stDateInput input{ min-height:46px; font-size:1.02rem;}
  .stButton button{ min-height:46px; font-size:1.02rem; padding:10px 18px; font-weight:700; background:white !important; color:#111 !important; border:1px solid #e5e7eb !important;}
  .stButton button:hover{ background:#f9fafb !important; }
  .disabled-dates label, .disabled-dates input{ color:#9ca3af !important; }
  .card { border:1px solid #e5e7eb; border-radius:12px; padding:16px; background:#fff; }
</style>
""", unsafe_allow_html=True)

# ==============================
# Small Utilities
# ==============================
def _rr():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

@st.cache_resource(show_spinner=False)
def _db_pragmas() -> Dict[str, str]:
    return {
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "busy_timeout": "5000",
        "foreign_keys": "ON",
        "cache_size": "-20000",
        "temp_store": "MEMORY",
    }

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False, isolation_level=None)
    try:
        pragmas = _db_pragmas()
        conn.execute(f"PRAGMA journal_mode={pragmas['journal_mode']}")
        conn.execute(f"PRAGMA synchronous={pragmas['synchronous']}")
        conn.execute(f"PRAGMA busy_timeout={pragmas['busy_timeout']}")
        conn.execute(f"PRAGMA foreign_keys={pragmas['foreign_keys']}")
        conn.execute(f"PRAGMA cache_size={pragmas['cache_size']}")
        conn.execute(f"PRAGMA temp_store={pragmas['temp_store']}")
    except Exception:
        pass
    return conn

def _to_float_safe(v):
    try: return float(str(v).replace(",","").strip())
    except: return 0.0

def truncate_amount(amount: float, ccy: str) -> float:
    amount = _to_float_safe(amount)
    if ccy == "KRW":
        return math.floor((amount or 0.0) * 100) / 100.0
    return round((amount or 0.0), 2)

def fmt_qty_2(val) -> str:
    try: return f"{float(val):,.2f}"
    except: return f"{val}"

def fmt_by_ccy(val, ccy: str, kind: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)): return ""
    if kind == "qty": return fmt_qty_2(val)
    if ccy == "KRW": return f"{float(val):,.0f}"
    return f"{float(val):,.2f}"

def apply_row_ccy_format(df: pd.DataFrame, ccy_col: str, qty_cols: List[str], price_cols: List[str], amount_cols: List[str]):
    if df.empty or ccy_col not in df.columns: return df
    out = df.copy()
    for ccy_v, idx in out.groupby(ccy_col).groups.items():
        if qty_cols:
            for c in qty_cols:
                if c in out.columns:
                    out.loc[idx, c] = out.loc[idx, c].map(lambda v: fmt_by_ccy(v, ccy_v, "qty"))
        if price_cols:
            for c in price_cols:
                if c in out.columns:
                    out.loc[idx, c] = out.loc[idx, c].map(lambda v: fmt_by_ccy(v, ccy_v, "price"))
        if amount_cols:
            for c in amount_cols:
                if c in out.columns:
                    out.loc[idx, c] = out.loc[idx, c].map(lambda v: fmt_by_ccy(v, ccy_v, "amount"))
    return out

# ==============================
# DB Bootstrapping
# ==============================
def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS investors(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cash_flows(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_id INTEGER NOT NULL,
            dt TEXT NOT NULL,
            ccy TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            note TEXT,
            source TEXT DEFAULT 'MANUAL',
            dividend_id INTEGER,
            FOREIGN KEY(investor_id) REFERENCES investors(id) ON DELETE CASCADE,
            FOREIGN KEY(dividend_id) REFERENCES dividends(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dt TEXT NOT NULL,
            symbol TEXT NOT NULL,
            ccy TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            override_json TEXT,
            note TEXT
        );
        CREATE TABLE IF NOT EXISTS investor_trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            investor_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            ccy TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            note TEXT,
            is_edit INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(trade_id) REFERENCES trades(id) ON DELETE CASCADE,
            FOREIGN KEY(investor_id) REFERENCES investors(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS dividends(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dt TEXT NOT NULL,
            symbol TEXT NOT NULL,
            ccy TEXT NOT NULL,
            total_amount REAL NOT NULL,
            note TEXT,
            record_dt TEXT
        );
        CREATE TABLE IF NOT EXISTS realized_pnl(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            investor_id INTEGER NOT NULL,
            ccy TEXT NOT NULL,
            amount REAL NOT NULL,
            note TEXT,
            FOREIGN KEY(trade_id) REFERENCES trades(id) ON DELETE CASCADE,
            FOREIGN KEY(investor_id) REFERENCES investors(id) ON DELETE CASCADE
        );
        """)
        for alter in [
            "ALTER TABLE cash_flows ADD COLUMN source TEXT DEFAULT 'MANUAL'",
            "ALTER TABLE cash_flows ADD COLUMN dividend_id INTEGER",
            "ALTER TABLE dividends ADD COLUMN record_dt TEXT",
            "ALTER TABLE investor_trades ADD COLUMN is_edit INTEGER NOT NULL DEFAULT 0",
        ]:
            try: cur.execute(alter)
            except Exception: pass
        cur.executescript("""
        CREATE INDEX IF NOT EXISTS ix_trades_dt ON trades(dt);
        CREATE INDEX IF NOT EXISTS ix_invtrades_keys ON investor_trades(investor_id, symbol, ccy);
        CREATE INDEX IF NOT EXISTS ix_invtrades_trade ON investor_trades(trade_id);
        CREATE INDEX IF NOT EXISTS ix_cash_dt_ccy ON cash_flows(dt, ccy);
        CREATE INDEX IF NOT EXISTS ix_cash_investor_ccy ON cash_flows(investor_id, ccy);
        CREATE INDEX IF NOT EXISTS ix_cash_dividend ON cash_flows(dividend_id);
        CREATE INDEX IF NOT EXISTS ix_rp_trade_investor ON realized_pnl(trade_id, investor_id);
        """)
        conn.commit()

@st.cache_data(show_spinner=False)
def load_df(query: str, params: tuple = ()):
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)

# ==============================
# Data Access Helpers
# ==============================
def get_investor_id_by_name(name: str):
    if not name: return None
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM investors WHERE name=?", (name,)).fetchone()
        return int(row[0]) if row else None

@st.cache_data(show_spinner=False)
def list_investors() -> pd.DataFrame:
    return load_df("SELECT id, name, created_at FROM investors ORDER BY id")

def add_investor(name: str):
    name = (name or "").strip()
    if not name: return False
    with get_conn() as conn:
        try:
            conn.execute("INSERT INTO investors(name, created_at) VALUES (?,?)",(name, datetime.now().isoformat()))
            conn.commit()
        except sqlite3.IntegrityError:
            return False
    list_investors.clear()
    return True

def get_cash_asof(iid: int, ccy: str, asof: date) -> float:
    with get_conn() as conn:
        row = conn.execute("""
          SELECT COALESCE(SUM(CASE
            WHEN type IN ('DEPOSIT','DIVIDEND','MGMT_FEE_IN','DIVIDEND_ROUND_ADJ') THEN amount
            WHEN type IN ('WITHDRAW','MGMT_FEE_OUT') THEN -amount ELSE 0 END),0.0)
          FROM cash_flows
          WHERE investor_id=? AND ccy=? AND date(dt) <= date(?)
        """, (iid, ccy, asof.isoformat())).fetchone()
    return float(row[0] or 0.0)

def get_cash(iid: int, ccy: str) -> float:
    with get_conn() as conn:
        row = conn.execute("""
          SELECT COALESCE(SUM(CASE
            WHEN cf.type IN ('DEPOSIT','DIVIDEND','MGMT_FEE_IN','DIVIDEND_ROUND_ADJ') THEN cf.amount
            WHEN cf.type IN ('WITHDRAW','MGMT_FEE_OUT') THEN -cf.amount ELSE 0 END),0.0)
          FROM cash_flows cf WHERE investor_id=? AND ccy=?""", (iid, ccy)).fetchone()
    return float(row[0] or 0.0)

def get_position_qty(iid: int, symbol: str, ccy: str) -> float:
    with get_conn() as conn:
        row = conn.execute("""
          SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN qty
                                   WHEN side='SELL' THEN -qty ELSE 0 END),0.0)
          FROM investor_trades WHERE investor_id=? AND symbol=? AND ccy=?""",
          (iid, symbol, ccy)).fetchone()
    return float(row[0] or 0.0)

def get_total_position_qty(symbol: str, ccy: str) -> float:
    with get_conn() as conn:
        row = conn.execute("""
          SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN qty
                                   WHEN side='SELL' THEN -qty ELSE 0 END),0.0)
          FROM investor_trades WHERE symbol=? AND ccy=?""",
          (symbol, ccy)).fetchone()
    return float(row[0] or 0.0)

def add_cashflow_retry(investor_id: int, dt: date, ccy: str, type_: str, amount: float, note: str, source: str = "MANUAL"):
    amount = truncate_amount(amount, ccy)
    for i in range(5):
        try:
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO cash_flows(investor_id, dt, ccy, type, amount, note, source) VALUES (?,?,?,?,?,?,?)",
                    (investor_id, dt.isoformat(), ccy, type_, amount, note, source)
                )
                conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < 4:
                time.sleep(0.3*(i+1)); continue
            raise

def record_trade(side: str, dt: date, symbol: str, ccy: str, total_qty: float, price: float,
                 alloc_amounts: Optional[Dict[int, float]], note: str = "", edit_mode: bool = False,
                 alloc_qtys: Optional[Dict[int, float]] = None):
    """
    alloc_amounts: 투자자별 '금액' 매핑(단가*수량). BUY/SELL 공통 사용
      - BUY: investor_trades.qty = alloc_amount/price, cash_flows WITHDRAW 금액 = alloc_amount
      - SELL: investor_trades.qty = alloc_amount/price, 현금흐름/수수료는 rebuild에서 계산(정책 반영)

    alloc_qtys: 투자자별 실주식수 매핑(선택). 제공 시 rounding 오류 없이 investor_trades.qty에 사용.
    """
    with get_conn() as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        cur = conn.cursor()
        cur.execute("""
          INSERT INTO trades(dt, symbol, ccy, side, qty, price, fee, override_json, note)
          VALUES (?,?,?,?,?,?,?,?,?)
        """, (dt.isoformat(), symbol, ccy, side, total_qty, price, 0.0, json.dumps(alloc_amounts or {}), note))
        trade_id = cur.lastrowid

        qty_alloc_pairs: List[Tuple[int, float]] = []
        if alloc_qtys:
            for iid, q in alloc_qtys.items():
                if q is None: continue
                qf = float(q)
                if abs(qf) <= 1e-12:
                    continue
                qty_alloc_pairs.append((iid, qf))
        elif alloc_amounts:
            for iid, amt in alloc_amounts.items():
                if amt is None: continue
                amt_f = float(amt)
                if abs(amt_f) <= 1e-12:
                    continue
                q = (amt_f / float(price)) if price != 0 else 0.0
                qty_alloc_pairs.append((iid, q))

        if qty_alloc_pairs:
            qty_sum = sum(q for _, q in qty_alloc_pairs)
            diff = float(total_qty) - float(qty_sum)
            if abs(diff) > 1e-8:
                last_iid, last_q = qty_alloc_pairs[-1]
                qty_alloc_pairs[-1] = (last_iid, last_q + diff)

            batch = [
                (trade_id, iid, symbol, ccy, side, q, price, 0.0, note, 1 if edit_mode else 0)
                for iid, q in qty_alloc_pairs
            ]

            cur.executemany("""
              INSERT INTO investor_trades(trade_id, investor_id, symbol, ccy, side, qty, price, fee, note, is_edit)
              VALUES (?,?,?,?,?,?,?,?,?,?)
            """, batch)
        conn.commit()
    rebuild_trade_effects(from_dt=dt.isoformat())

def rebuild_trade_effects(from_dt: Optional[str] = None):
    """
    SELL 정책(최종):
      - 투자자별 proceeds = qty*px 를 DEPOSIT.
      - profit = proceeds − avg*qty (손익). profit>0 인 경우에만 수익의 10%를 MGMT_FEE_OUT.
      - 운용자(조석환)는 모든 투자자 fee의 합계를 MGMT_FEE_IN.
      - EDIT 행은 현금흐름/손익 미기록(포지션만 반영).
    """
    with get_conn() as conn:
        cur = conn.cursor()
        if from_dt is None:
            cur.execute("DELETE FROM cash_flows WHERE source='TRADE'")
            cur.execute("DELETE FROM realized_pnl")
            base_cut = None
        else:
            cur.execute("DELETE FROM cash_flows WHERE source='TRADE' AND date(dt) >= date(?)", (from_dt,))
            cur.execute("DELETE FROM realized_pnl WHERE trade_id IN (SELECT id FROM trades WHERE date(dt) >= date(?))", (from_dt,))
            base_cut = from_dt

        # 포지션 롤업 (컷 이전)
        pos_map: Dict[Tuple[int,str,str], Tuple[float,float]] = {}
        if base_cut:
            rows = pd.read_sql_query("""
                SELECT it.investor_id, it.symbol, it.ccy, it.side, it.qty, it.price, t.dt
                FROM investor_trades it
                JOIN trades t ON t.id = it.trade_id
                WHERE date(t.dt) < date(?)
                ORDER BY t.dt ASC, it.trade_id ASC, it.id ASC
            """, conn, params=(base_cut,))
            if not rows.empty:
                for _, r in rows.iterrows():
                    key = (int(r["investor_id"]), r["symbol"], r["ccy"])
                    qpos, avg = pos_map.get(key, (0.0, 0.0))
                    qtyf, prcf = float(r["qty"]), float(r["price"])
                    if r["side"] == "BUY":
                        total_cost = avg*qpos + prcf*qtyf
                        qpos += qtyf
                        avg = (total_cost/qpos) if qpos>1e-12 else 0.0
                    else:
                        qpos -= qtyf
                        if qpos < 1e-12: qpos, avg = 0.0, 0.0
                    pos_map[key] = (qpos, avg)

        trades = pd.read_sql_query(f"""
            SELECT t.id as trade_id, t.dt, t.symbol, t.ccy, t.side, t.price, t.override_json
            FROM trades t
            {"WHERE date(t.dt) >= date(?)" if base_cut else ""}
            ORDER BY date(t.dt) ASC, t.id ASC
        """, conn, params=((base_cut,) if base_cut else ()))
        ins_cf, ins_rp = [], []

        for _, trow in trades.iterrows():
            tid = int(trow["trade_id"])
            dtv, sym, ccy, side, px = trow["dt"], trow["symbol"], trow["ccy"], trow["side"], float(trow["price"])
            its = pd.read_sql_query("""
                SELECT investor_id, side, qty, price, is_edit
                FROM investor_trades
                WHERE trade_id=?
                ORDER BY id ASC
            """, conn, params=(tid,))
            if its.empty:
                continue

            trade_cf: List[Tuple[int, str, str, str, float, str, str]] = []

            # BUY: 예수금 인출(override_json의 금액 또는 qty*px)
            if side == "BUY":
                for _, it in its.iterrows():
                    iid = int(it["investor_id"]); qty = float(it["qty"]); is_edit = int(it["is_edit"]) == 1
                    key = (iid, sym, ccy)
                    qpos, avg = pos_map.get(key, (0.0, 0.0))
                    total_cost = avg*qpos + px*qty
                    qpos += qty
                    avg = (total_cost/qpos) if qpos>1e-12 else 0.0
                    pos_map[key] = (qpos, avg)

                    if not is_edit:
                        try:
                            alloc_map = json.loads(trow.get("override_json") or "{}")
                        except Exception:
                            alloc_map = {}
                        alloc_amt = None
                        if isinstance(alloc_map, dict):
                            if str(iid) in alloc_map: alloc_amt = _to_float_safe(alloc_map[str(iid)])
                            elif iid in alloc_map: alloc_amt = _to_float_safe(alloc_map[iid])
                        if alloc_amt is None:
                            alloc_amt = qty * px
                        trade_cf.append((iid, dtv, ccy, "WITHDRAW", truncate_amount(alloc_amt, ccy), f"BUY {sym}", "TRADE"))
                ins_cf.extend(trade_cf)
                continue

            # SELL/EDIT: per-investor proceeds deposit + 10% fee out (profit>0), operator fee in
            fee_sum = 0.0
            for _, it in its.iterrows():
                iid = int(it["investor_id"])
                qty = float(it["qty"])
                is_edit = int(it.get("is_edit", 0)) == 1

                key = (iid, sym, ccy)
                qpos, avg = pos_map.get(key, (0.0, 0.0))

                proceeds = truncate_amount(qty * px, ccy)           # 매도금액
                cost     = truncate_amount(avg * qty, ccy)          # 원가
                profit   = proceeds - cost                          # 수익(손실 가능)

                # 포지션 차감(EDIT도 포지션 반영)
                qpos -= qty
                if qpos < 1e-12: qpos, avg = 0.0, 0.0
                pos_map[key] = (qpos, avg)

                if is_edit:
                    # EDIT: 현금흐름/손익 기록하지 않음
                    continue

                # 투자자 DEPOSIT: 전체 매도금액(= cost + profit)
                if abs(proceeds) > 1e-12:
                    trade_cf.append((iid, dtv, ccy, "DEPOSIT", proceeds, f"SELL {sym}", "TRADE"))

                # 실현손익: 총 profit 기록(보고용)
                if abs(profit) > 1e-12:
                    ins_rp.append((tid, iid, ccy, truncate_amount(profit, ccy), f"실현손익 {sym}"))

                # 성과수수료(10%): 이익일 때만
                fee = truncate_amount(max(0.0, profit * 0.10), ccy)
                if fee > 0:
                    trade_cf.append((iid, dtv, ccy, "MGMT_FEE_OUT", fee, f"성과수수료 {sym}", "TRADE"))
                    fee_sum += fee

            # 운용자 수취: 모든 참여자 수익 10% 합계
            op_id = get_investor_id_by_name(OPERATOR_NAME)
            if op_id and fee_sum > 0:
                trade_cf.append((op_id, dtv, ccy, "MGMT_FEE_IN", truncate_amount(fee_sum, ccy), f"성과수수료 {sym}", "TRADE"))

            if trade_cf:
                # rounding drift 보정: 투자자 매도금액 합계가 기대치를 초과하면 운용자 입금 조정
                total_qty = float(its["qty"].sum())
                expected_total = truncate_amount(total_qty * px, ccy)
                actual_total = sum(row[4] for row in trade_cf if row[3] == "DEPOSIT")
                if actual_total - expected_total > 1e-8:
                    diff = actual_total - expected_total
                    op_idx = next((idx for idx, row in enumerate(trade_cf)
                                   if row[0] == op_id and row[3] == "MGMT_FEE_IN"), None)
                    if op_idx is not None and diff > 0:
                        op_row = list(trade_cf[op_idx])
                        reducible = min(diff, op_row[4])
                        new_amt = max(0.0, op_row[4] - reducible)
                        op_row[4] = truncate_amount(new_amt, ccy)
                        diff -= reducible
                        trade_cf[op_idx] = tuple(op_row)
                        if op_row[4] <= 1e-12:
                            trade_cf.pop(op_idx)
                    # diff가 남더라도 추가 조정 불필요 (운용자 입금 외에는 지침 없음)
                ins_cf.extend(trade_cf)

        # 일괄 반영
        with get_conn() as conn2:
            if ins_cf:
                conn2.executemany(
                    "INSERT INTO cash_flows(investor_id, dt, ccy, type, amount, note, source) VALUES (?,?,?,?,?,?,?)",
                    ins_cf
                )
            if ins_rp:
                conn2.executemany(
                    "INSERT INTO realized_pnl(trade_id, investor_id, ccy, amount, note) VALUES (?,?,?,?,?)",
                    ins_rp
                )

    invalidate_all_caches()

# ==============================
# Aggregations (Cached)
# ==============================
@st.cache_data(show_spinner=False)
def investor_positions() -> pd.DataFrame:
    with get_conn() as conn:
        trades = pd.read_sql_query("""
          SELECT it.investor_id, it.symbol, it.ccy, it.side, it.qty, it.price, t.dt
          FROM investor_trades it
          JOIN trades t ON t.id = it.trade_id
          ORDER BY it.investor_id, it.symbol, it.ccy, date(t.dt), it.trade_id, it.id
        """, conn)
        inv = pd.read_sql_query("SELECT id, name FROM investors", conn)

    if trades.empty:
        return pd.DataFrame(columns=["investor_id","name","symbol","ccy","qty","avg_px","cost_basis"])

    id2name = dict(inv.values)
    rows = []
    for (iid, sym, ccy), g in trades.groupby(["investor_id","symbol","ccy"]):
        qty_pos = 0.0; avg_cost = 0.0
        for _, r in g.iterrows():
            q, p = float(r["qty"]), float(r["price"])
            if r["side"] == "BUY":
                total_cost = avg_cost*qty_pos + p*q
                qty_pos += q
                avg_cost = (total_cost/qty_pos) if qty_pos>1e-12 else 0.0
            else:
                qty_pos -= q
                if qty_pos < 1e-12: qty_pos, avg_cost = 0.0, 0.0
        if qty_pos > 1e-12:
            rows.append({
                "investor_id": int(iid),
                "name": id2name.get(int(iid), ""),
                "symbol": sym,
                "ccy": ccy,
                "qty": qty_pos,
                "avg_px": avg_cost,
                "cost_basis": avg_cost*qty_pos
            })
    return pd.DataFrame(rows, columns=["investor_id","name","symbol","ccy","qty","avg_px","cost_basis"])

@st.cache_data(show_spinner=False)
def investor_balances(ccy: str = "KRW") -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("""
          SELECT inv.name, inv.id as investor_id,
                 COALESCE(SUM(CASE
                   WHEN cf.type IN ('DEPOSIT','DIVIDEND','MGMT_FEE_IN','DIVIDEND_ROUND_ADJ') THEN cf.amount
                   WHEN cf.type IN ('WITHDRAW','MGMT_FEE_OUT') THEN -cf.amount ELSE 0 END), 0.0) as cash
          FROM investors inv
          LEFT JOIN cash_flows cf ON cf.investor_id = inv.id AND cf.ccy = ?
          GROUP BY inv.id, inv.name
          ORDER BY inv.name
        """, conn, params=(ccy,))

@st.cache_data(show_spinner=False)
def prefee_map_all() -> Dict[Tuple[int,int], float]:
    df = load_df("""
      SELECT it.trade_id, it.investor_id, it.symbol, it.ccy, it.side, it.qty, it.price, t.dt
      FROM investor_trades it
      JOIN trades t ON t.id = it.trade_id
      ORDER BY t.dt ASC, it.trade_id ASC, it.id ASC
    """)
    out: Dict[Tuple[int,int], float] = {}
    if df.empty: return out
    for (iid, sym, ccy), g in df.groupby(["investor_id","symbol","ccy"], sort=False):
        qty_pos = 0.0; avg_cost = 0.0
        for _, r in g.iterrows():
            out[(int(r["investor_id"]), int(r["trade_id"]))] = float(avg_cost)
            if r["side"] == "BUY":
                total_cost = avg_cost*qty_pos + float(r["price"])*float(r["qty"])
                qty_pos += float(r["qty"])
                avg_cost = (total_cost/qty_pos) if qty_pos>1e-12 else 0.0
            else:
                qty_pos -= float(r["qty"])
                if qty_pos < 1e-12: qty_pos, avg_cost = 0.0, 0.0
    return out


def invalidate_all_caches() -> None:
    """Invalidate frequently reused cached computations."""
    for fn in (prefee_map_all, investor_positions, investor_balances, load_df):
        try:
            fn.clear()
        except Exception:
            pass


def build_dividend_cashflows(
    dividend_id: int,
    deal_dt: date,
    symbol: str,
    ccy: str,
    total_amt: float,
    memo: str,
    pos_rec: pd.DataFrame,
    op_id: Optional[int],
) -> List[Tuple[Any, ...]]:
    if pos_rec.empty:
        return []

    total_qty = float(pos_rec["pos_qty"].sum())
    if total_qty <= 0:
        return []

    expected_total = truncate_amount(total_amt, ccy)
    rows: List[Dict[str, Any]] = []
    deposit_sum = 0.0
    fee_total = 0.0

    for rec in pos_rec.itertuples(index=False):
        iid = int(rec.investor_id)
        qty_share = float(rec.pos_qty)
        if qty_share <= 0:
            continue

        share = truncate_amount(total_amt * (qty_share / total_qty), ccy)
        if share != 0:
            rows.append(
                {
                    "investor_id": iid,
                    "type": "DIVIDEND",
                    "amount": share,
                    "note": memo,
                }
            )
            deposit_sum += share

        fee = truncate_amount(0.10 * share, ccy)
        if fee > 0:
            rows.append(
                {
                    "investor_id": iid,
                    "type": "MGMT_FEE_OUT",
                    "amount": fee,
                    "note": f"배당 성과수수료 {symbol}",
                }
            )
            fee_total += fee

    if op_id and fee_total > 0:
        rows.append(
            {
                "investor_id": op_id,
                "type": "MGMT_FEE_IN",
                "amount": truncate_amount(fee_total, ccy),
                "note": f"배당 성과수수료 합계 {symbol}",
            }
        )

    round_diff = truncate_amount(expected_total - deposit_sum, ccy)
    if op_id and abs(round_diff) > 1e-8:
        if round_diff > 0:
            rows.append(
                {
                    "investor_id": op_id,
                    "type": "DIVIDEND_ROUND_ADJ",
                    "amount": truncate_amount(round_diff, ccy),
                    "note": f"배당 반올림 조정 {symbol}",
                }
            )
        else:
            diff = -round_diff
            fee_idx = next(
                (
                    idx
                    for idx, item in enumerate(rows)
                    if item["investor_id"] == op_id and item["type"] == "MGMT_FEE_IN"
                ),
                None,
            )
            if fee_idx is not None:
                reducible = min(diff, rows[fee_idx]["amount"])
                new_amt = truncate_amount(rows[fee_idx]["amount"] - reducible, ccy)
                diff -= reducible
                if new_amt <= 1e-12:
                    rows.pop(fee_idx)
                else:
                    rows[fee_idx]["amount"] = new_amt
            if diff > 1e-8:
                rows.append(
                    {
                        "investor_id": op_id,
                        "type": "DIVIDEND_ROUND_ADJ",
                        "amount": truncate_amount(-diff, ccy),
                        "note": f"배당 반올림 조정 {symbol}",
                    }
                )

    out_rows: List[Tuple[Any, ...]] = []
    for row in rows:
        amt = truncate_amount(row["amount"], ccy)
        if abs(amt) <= 1e-12:
            continue
        out_rows.append(
            (
                row["investor_id"],
                deal_dt.isoformat(),
                ccy,
                row["type"],
                amt,
                row.get("note", ""),
                "DIVIDEND",
                dividend_id,
            )
        )
    return out_rows


def summarize_dividend_history(div_meta: pd.DataFrame, cf_recent: pd.DataFrame) -> pd.DataFrame:
    if div_meta.empty or cf_recent.empty:
        return pd.DataFrame()

    cf_recent = cf_recent.copy()
    cf_recent["note"] = cf_recent["note"].fillna("")
    records: List[Dict[str, Any]] = []

    for div in div_meta.itertuples(index=False):
        div_rows = cf_recent[cf_recent["dividend_id"] == div.id]

        if div_rows.empty:
            base_mask = (
                (cf_recent["dividend_id"].isna())
                & (cf_recent["dt"] == div.dt)
                & (cf_recent["통화"] == div.ccy)
            )
            candidate_rows = cf_recent[base_mask]

            if div.note:
                note_mask = (candidate_rows["type"] == "DIVIDEND") & (candidate_rows["note"] == div.note)
            else:
                note_mask = candidate_rows["type"] == "DIVIDEND"

            if div.symbol:
                extra_mask = candidate_rows["note"].str.contains(div.symbol, regex=False)
            else:
                extra_mask = pd.Series(False, index=candidate_rows.index)

            div_rows = candidate_rows[note_mask | extra_mask]

        if div_rows.empty:
            continue

        for iid, grp in div_rows.groupby("investor_id"):
            investor_name = grp["투자자"].iloc[0]
            base_amt = truncate_amount(
                float(grp.loc[grp["type"] == "DIVIDEND", "amount"].sum()),
                div.ccy,
            )
            fee_out = truncate_amount(
                float(grp.loc[grp["type"] == "MGMT_FEE_OUT", "amount"].sum()),
                div.ccy,
            )
            fee_in = truncate_amount(
                float(grp.loc[grp["type"] == "MGMT_FEE_IN", "amount"].sum()),
                div.ccy,
            )
            round_adj = truncate_amount(
                float(grp.loc[grp["type"] == "DIVIDEND_ROUND_ADJ", "amount"].sum()),
                div.ccy,
            )

            if investor_name == OPERATOR_NAME:
                net_amt = truncate_amount(base_amt - fee_out + fee_in - round_adj, div.ccy)
            else:
                net_amt = truncate_amount(base_amt - fee_out, div.ccy)

            records.append(
                {
                    "일자": div.dt,
                    "종목": div.symbol,
                    "투자자": investor_name,
                    "통화": div.ccy,
                    "원배당금액": base_amt if base_amt != 0 else 0.0,
                    "운용자수수료": fee_out if fee_out != 0 else 0.0,
                    "순지급금액": net_amt,
                }
            )

    if not records:
        return pd.DataFrame()

    out_df = pd.DataFrame(records)
    out_df.sort_values(["일자", "종목", "투자자"], ascending=[False, True, True], inplace=True)
    return out_df

# ==============================
# App Boot
# ==============================
init_db()
if get_investor_id_by_name(OPERATOR_NAME) is None:
    add_investor(OPERATOR_NAME)

T1, T2, T3, T4, T5, T6, T7, T8 = st.tabs([
    "대시보드", "입출금", "주문", "투자자별 잔고", "배당", "매도 공지용", "수익", "⚙️"
])

# ==============================
# Dashboard
# ==============================
with T1:
    st.subheader("요약")
    krw_df = investor_balances("KRW"); usd_df = investor_balances("USD")
    krw_cash = float(krw_df["cash"].sum() if not krw_df.empty else 0.0)
    usd_cash = float(usd_df["cash"].sum() if not usd_df.empty else 0.0)

    pos_all = investor_positions()
    inv_sum_ccy = pos_all.groupby("ccy")["cost_basis"].sum() if not pos_all.empty else pd.Series(dtype=float)
    krw_invest = float(inv_sum_ccy.get("KRW", 0.0))
    usd_invest = float(inv_sum_ccy.get("USD", 0.0))

    total_cash_k = krw_cash + usd_cash*USD_KRW
    total_invest_k = krw_invest + usd_invest*USD_KRW

    c1, c2, c3 = st.columns(3)
    c1.metric("총 현금", f"{total_cash_k:,.0f}")
    c2.metric("총 투자원가", f"{total_invest_k:,.0f}")
    c3.metric("총합", f"{(total_cash_k+total_invest_k):,.0f}")

    chart_df = pd.DataFrame([{"구분":"현금","금액": total_cash_k},{"구분":"투자원가","금액": total_invest_k}])
    st.markdown("#### 현금/투자 비중")
    pie = alt.Chart(chart_df).mark_arc(outerRadius=120, innerRadius=40).encode(
        theta=alt.Theta('금액:Q'),
        color=alt.Color('구분:N', title=None),
        tooltip=[alt.Tooltip('구분:N'), alt.Tooltip('금액:Q', format=',.0f')]
    ).properties(height=320, width=360)
    st.altair_chart(pie, use_container_width=False)
    st.caption("현금보유비중")

# ==============================
# Cash I/O
# ==============================
with T2:
    st.markdown("### 입출금")
    with st.form("cash_io"):
        inv_names = list_investors()["name"].tolist()
        sel_name = st.selectbox("투자자", inv_names)
        cA, cB = st.columns(2)
        with cA: sel_ccy = st.radio("통화", SUPPORTED_CCY, horizontal=True, key="cash_ccy")
        with cB: io_kor = st.radio("유형", ["입금", "출금"], horizontal=True, key="cash_type")
        io_type = "DEPOSIT" if io_kor == "입금" else "WITHDRAW"
        cD, cE = st.columns(2)
        with cD: dt_cf = st.date_input("입출금 일자", value=date.today(), key="cash_dt")
        with cE: amt_txt = st.text_input("금액", "0")
        amt = _to_float_safe(amt_txt)
        with st.expander("추가 옵션", expanded=False):
            note = st.text_input("비고(선택)", "")
        ok = st.form_submit_button("기록")
        if ok:
            iid = get_investor_id_by_name(sel_name)
            add_cashflow_retry(iid, dt_cf, sel_ccy, io_type, amt, note if 'note' in locals() else "", source="MANUAL")
            invalidate_all_caches()
            st.success("기록 완료"); _rr()

    st.markdown("### 최근입출금")
    cf_view = load_df("""
        SELECT cf.id,
               cf.dt   AS 일자,
               inv.name AS 투자자,
               cf.ccy  AS 통화,
               cf.type AS 유형,
               cf.amount AS 금액,
               cf.note AS 비고,
               cf.source AS 출처
        FROM cash_flows cf
        JOIN investors inv ON inv.id = cf.investor_id
        WHERE cf.source != 'TRADE'
        ORDER BY date(cf.dt) DESC, cf.id DESC
        LIMIT 200
    """)
    if cf_view.empty:
        st.info("데이터가 없습니다.")
    else:
        cf_view = cf_view.set_index("id")
        show_cols = ["일자","투자자","통화","유형","금액","비고"]
        cf_fmt = apply_row_ccy_format(cf_view.copy(), "통화", [], [], ["금액"])
        try: cf_fmt["일자"] = pd.to_datetime(cf_fmt["일자"], errors="coerce").dt.date
        except: pass
        disabled_cols = [c for c in show_cols if c not in ["일자","통화","유형","금액","비고"]]
        edited = st.data_editor(
            cf_fmt[show_cols],
            key="cash_recent_editor",
            hide_index=True,
            use_container_width=True,
            disabled=disabled_cols,
            column_config={
                "일자": st.column_config.DateColumn("일자", format="YYYY-MM-DD", width=160),
                "투자자": st.column_config.TextColumn("투자자", width=180),
                "통화": st.column_config.TextColumn("통화", width=72),
                "유형": st.column_config.TextColumn("유형", width=96),
                "금액": st.column_config.TextColumn("금액", width=140),
                "비고": st.column_config.TextColumn("비고", width=800),
            }
        )
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("변경 저장", key="cash_recent_save"):
                before = cf_view.copy()
                after = edited.copy()
                after["amount"] = after["금액"].apply(_to_float_safe)
                def _to_iso(d):
                    if pd.isna(d): return None
                    if isinstance(d, (datetime, pd.Timestamp)): return d.date().isoformat()
                    if isinstance(d, date): return d.isoformat()
                    try: return pd.to_datetime(d).date().isoformat()
                    except: return None
                after["dt"] = after["일자"].apply(_to_iso)
                after["ccy"] = after["통화"]; after["type"] = after["유형"]; after["note"] = after["비고"]
                up_cols = ["dt","ccy","type","amount","note"]
                before_up = before.assign(
                    dt=pd.to_datetime(before["일자"], errors="coerce").dt.date.astype("string"),
                    ccy=before["통화"], type=before["유형"], amount=before["금액"], note=before["비고"]
                )[up_cols]
                after_up = after[up_cols]
                changed_ids = []
                for rid in after_up.index.intersection(before_up.index):
                    if any(str(before_up.at[rid, c]) != str(after_up.at[rid, c]) for c in up_cols):
                        changed_ids.append(rid)
                if changed_ids:
                    with get_conn() as conn:
                        cur = conn.cursor()
                        for rid in changed_ids:
                            row = after_up.loc[rid]
                            cur.execute("UPDATE cash_flows SET dt=?, ccy=?, type=?, amount=?, note=? WHERE id=?",
                                        (row["dt"], row["ccy"], row["type"], float(row["amount"]), row["note"], int(rid)))
                        conn.commit()
                    invalidate_all_caches()
                    st.success(f"{len(changed_ids)}건 저장 완료"); _rr()
                else:
                    st.info("변경사항이 없습니다.")
        with c2:
            if st.button("새로고침", key="cash_recent_reload"): _rr()

# ==============================
# Orders (금액 기반, 100% 매도)
# ==============================
def _alloc_by_weights(total: float, weights: Dict[int, float]) -> Dict[int, float]:
    if total <= 0 or not weights: return {}
    s = sum(w for w in weights.values() if w > 0)
    if s <= 0: return {}
    raw = {iid: total * (w / s) for iid, w in weights.items()}
    diff = total - sum(raw.values())
    if abs(diff) > 1e-8:
        k = max(raw, key=lambda x: raw[x])
        raw[k] += diff
    return raw

with T3:
    st.markdown("### 주문")

    # 주문 모드: 카드 밖 우상단
    c_left, c_right = st.columns([6,2])
    with c_right:
        inv_all_df = list_investors()
        order_mode = st.radio("주문 모드", ["일반", "개별"], horizontal=True, key="order_mode_radio")

    # 주문 입력 카드 (빈카드 없음, 유형을 카드 내부에 배치)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("order_form", clear_on_submit=False):

        # 1행: 유형 / 결제통화 / 일자
        r1a, r1b, r1c = st.columns([1.2, 1.2, 1.2])
        with r1a:
            side = st.radio("유형", ["BUY", "SELL", "EDIT"], horizontal=True, key="ord_side")
        with r1b:
            ccy  = st.radio("결제 통화", SUPPORTED_CCY, horizontal=True, key="ord_ccy")
        with r1c:
            dtv  = st.date_input("일자", value=date.today(), key="order_date")

        # 2행: 종목명
        symbol = st.text_input("종목명", "")

        # 3행: (개별 모드일 때) 개별 투자자 - 종목명 다음 줄
        if order_mode == "개별":
            sel_name_ind = st.selectbox("개별 투자자", inv_all_df["name"].tolist(), key="order_ind_name")

        # SELL/EDIT일 때만 100% 매도
        is_sell_like = (side in ("SELL","EDIT"))
        if not is_sell_like and st.session_state.get("sell_all", False):
            st.session_state["sell_all"] = False

        # 4행: 주문 금액 / 주식수 / 100% 매도
        c_amt, c_qty, c_full = st.columns([1.2, 1.2, 1.0])
        with c_amt:
            amount_txt = st.text_input("주문 금액", "0")
            total_amount = truncate_amount(amount_txt, ccy)

        with c_qty:
            qty_input = st.number_input(
                "주식수",
                min_value=0.0, step=0.00001, format="%.5f",
                disabled=(is_sell_like and st.session_state.get("sell_all", False))
            )

        with c_full:
            sell_all = st.checkbox(
                "100% 매도",
                value=st.session_state.get("sell_all", False),
                disabled=not is_sell_like,
                key="sell_all"
            )

        # 5행: 비고 + 기록
        with st.expander("비고", expanded=False):
            note = st.text_input("비고(선택)", "")
        ok = st.form_submit_button("기록")  # Enter로 제출 가능
    st.markdown('</div>', unsafe_allow_html=True)

    if ok:
        if not symbol.strip():
            st.error("종목을 입력하세요."); st.stop()

        total_pos_qty = get_total_position_qty(symbol.strip(), ccy)
        if is_sell_like and st.session_state.get("sell_all", False):
            qty_total = float(total_pos_qty)
        else:
            qty_total = float(qty_input)

        if side != "EDIT":
            if (side == "BUY" and (total_amount <= 0 or qty_total <= 0)) or \
               (side == "SELL" and (total_amount <= 0 or qty_total <= 0)):
                st.error("주문 금액과 주식수를 입력하세요."); st.stop()

        unit_price = float(total_amount) / float(qty_total) if qty_total > 0 else 0.0

        if side == "BUY" and order_mode=="일반":
            inv_df_live = list_investors()
            op_id = get_investor_id_by_name(OPERATOR_NAME)
            cash_map: Dict[int, float] = {}
            for _, r in inv_df_live.iterrows():
                iid = int(r["id"])
                if op_id and iid == op_id:  # 운용자 제외
                    continue
                c = get_cash_asof(iid, ccy, dtv)
                if c > 0: cash_map[iid] = c
            if not cash_map:
                st.error("배분 가능한 예수금이 없습니다."); st.stop()
            total_cash_asof = sum(cash_map.values())
            if side!="EDIT" and total_amount - total_cash_asof > 1e-6:
                st.error(
                    f"총 매수금액({fmt_by_ccy(total_amount, ccy, 'amount')})이 "
                    f"{dtv.isoformat()} 기준 가용 예수금 합계({fmt_by_ccy(total_cash_asof, ccy, 'amount')})를 초과합니다."
                ); st.stop()

            alloc_amounts = _alloc_by_weights(total_amount, cash_map)
            record_trade(
                side="BUY", dt=dtv, symbol=symbol.strip(), ccy=ccy,
                total_qty=qty_total, price=unit_price,
                alloc_amounts=alloc_amounts,
                note=note if 'note' in locals() else "", edit_mode=False
            )
            st.success("기록 완료"); _rr()

        elif side == "BUY" and order_mode=="개별":
            iid = get_investor_id_by_name(st.session_state.get("order_ind_name"))
            if iid is None: st.error("투자자를 찾을 수 없습니다."); st.stop()
            if unit_price <= 0:
                st.error("주문 금액/주식수가 유효하지 않습니다."); st.stop()
            cash_asof = get_cash_asof(iid, ccy, dtv)
            if total_amount - cash_asof > 1e-6:
                st.error(f"매수 금액({fmt_by_ccy(total_amount, ccy, 'amount')})이 {st.session_state.get('order_ind_name')}의 {dtv.isoformat()} 기준 가용 예수금({fmt_by_ccy(cash_asof, ccy, 'amount')})을 초과합니다.")
                st.stop()
            record_trade(
                side="BUY", dt=dtv, symbol=symbol.strip(), ccy=ccy,
                total_qty=qty_total, price=unit_price,
                alloc_amounts={iid: total_amount},
                note=note if 'note' in locals() else "", edit_mode=False
            )
            st.success("기록 완료"); _rr()

        elif side in ("SELL","EDIT") and order_mode=="일반":
            if total_pos_qty <= 0:
                st.error("해당 종목 보유가 없습니다."); st.stop()
            if qty_total - total_pos_qty > 1e-8:
                st.error(f"매도 주식수({fmt_qty_2(qty_total)})가 전체 보유({fmt_qty_2(total_pos_qty)})를 초과합니다."); st.stop()

            inv_df = list_investors()
            hold_map: Dict[int, float] = {}
            for _, r in inv_df.iterrows():
                iid = int(r["id"])
                q = get_position_qty(iid, symbol.strip(), ccy)
                if q > 1e-12: hold_map[iid] = q
            if not hold_map:
                st.error("보유자가 없습니다."); st.stop()

            qty_by_investor = _alloc_by_weights(qty_total, hold_map)
            alloc_amounts = {iid: truncate_amount(q * unit_price, ccy) for iid, q in qty_by_investor.items()}
            record_trade(
                side="SELL", dt=dtv, symbol=symbol.strip(), ccy=ccy,
                total_qty=qty_total, price=unit_price,
                alloc_amounts=alloc_amounts,
                note=note if 'note' in locals() else "", edit_mode=(side=="EDIT"),
                alloc_qtys=qty_by_investor
            )
            st.success("기록 완료"); _rr()

        elif side in ("SELL","EDIT") and order_mode=="개별":
            iid = get_investor_id_by_name(st.session_state.get("order_ind_name"))
            if iid is None: st.error("투자자를 찾을 수 없습니다."); st.stop()
            if st.session_state.get("sell_all", False):
                qty_total = get_position_qty(iid, symbol.strip(), ccy)
            pos_qty = get_position_qty(iid, symbol.strip(), ccy)
            if qty_total - pos_qty > 1e-8:
                st.error(f"매도 주식수({fmt_qty_2(qty_total)})가 {st.session_state.get('order_ind_name')} 보유({fmt_qty_2(pos_qty)})를 초과합니다."); st.stop()
            if total_amount <= 0 or qty_total <= 0:
                st.error("주문 금액과 주식수를 입력하세요."); st.stop()
            record_trade(
                side="SELL", dt=dtv, symbol=symbol.strip(), ccy=ccy,
                total_qty=qty_total, price=unit_price,
                alloc_amounts={iid: truncate_amount(total_amount, ccy)},
                note=note if 'note' in locals() else "", edit_mode=(side=="EDIT"),
                alloc_qtys={iid: qty_total}
            )
            st.success("기록 완료"); _rr()

    # === 최근거래 (편집 가능) ===
    st.markdown("### 최근거래 (편집 가능)")
    def build_recent_trade_df(limit:int=200) -> pd.DataFrame:
        return load_df(f"""
         SELECT it.id as id, it.trade_id, it.investor_id,
                t.dt as 일자, inv.name as 투자자,
                it.symbol as 종목, it.ccy as 통화,
                it.side as 원유형, it.qty as 수량, it.price as 단가,
                (it.qty*it.price) as 금액,
                it.note as 비고, it.is_edit as is_edit
         FROM investor_trades it
         JOIN trades t ON t.id = it.trade_id
         JOIN investors inv ON inv.id = it.investor_id
         ORDER BY date(t.dt) DESC, it.id DESC
         LIMIT {int(limit)}
        """)
    it_full = build_recent_trade_df(limit=200)
    if it_full.empty:
        st.info("데이터가 없습니다.")
    else:
        it_full["유형"] = np.where(it_full["원유형"].eq("BUY"),
                                np.where(it_full["is_edit"].eq(1), "[EDIT] 매수", "매수"),
                                "매도")

        pre_map = prefee_map_all()
        def _prefee_row(row):
            if row["원유형"] != "SELL": return np.nan
            avg_before = float(pre_map.get((int(row["investor_id"]), int(row["trade_id"])), 0.0))
            return (float(row["단가"]) - avg_before) * float(row["수량"])
        it_full["사전수익"] = it_full.apply(_prefee_row, axis=1)

        it_full_fmt = it_full.copy()
        it_full_fmt["수량"] = it_full_fmt["수량"].apply(fmt_qty_2)
        it_full_fmt = apply_row_ccy_format(it_full_fmt, "통화", [], ["단가"], ["금액"])

        show = it_full_fmt[["일자","투자자","종목","통화","유형","단가","수량","금액","비고","id","trade_id","investor_id","원유형","is_edit"]].copy()
        try:
            show["일자"] = pd.to_datetime(show["일자"], errors="coerce").dt.date
        except Exception:
            pass

        editable_cols_now = ["일자","종목","통화","원유형","수량","단가","비고","is_edit"]
        disabled_cols = [c for c in show.columns if c not in editable_cols_now]

        edited = st.data_editor(
            show,
            key="it_recent_like_editor",
            hide_index=True,
            use_container_width=True,
            disabled=disabled_cols,
            column_config={
                "일자": st.column_config.DateColumn("일자", width=168, format="YYYY-MM-DD"),
                "투자자": st.column_config.TextColumn("투자자", width=182),
                "종목": st.column_config.TextColumn("종목", width=320),
                "통화": st.column_config.TextColumn("통화", width="small"),
                "유형": st.column_config.TextColumn("유형", width="small"),
                "단가": st.column_config.TextColumn("단가", width=120),
                "수량": st.column_config.TextColumn("수량", width=60),
                "금액": st.column_config.TextColumn("금액", width=120),
                "비고": st.column_config.TextColumn("비고", width=1000),
                "원유형": st.column_config.SelectboxColumn("원유형", options=["BUY","SELL"]),
                "is_edit": st.column_config.CheckboxColumn("is_edit"),
            }
        )

        c1, c2, c3 = st.columns([1,1,6])
        with c1:
            if st.button("변경 저장", key="it_recent_like_save"):
                before2 = it_full[["id","종목","통화","원유형","수량","단가","비고","is_edit"]].rename(
                    columns={"원유형":"side","수량":"qty","단가":"price","비고":"note","종목":"symbol","통화":"ccy"}
                )
                after = edited.copy()
                after["qty"] = after["수량"].apply(_to_float_safe)
                after["price"] = after["단가"].apply(_to_float_safe)
                after2 = after.rename(columns={"원유형":"side","종목":"symbol","통화":"ccy","비고":"note"})[
                    ["id","symbol","ccy","side","qty","price","note","is_edit"]
                ]

                with get_conn() as conn_u:
                    cur_u = conn_u.cursor()
                    updated_it = 0
                    for _, r in after2.iterrows():
                        rid = int(r["id"])
                        row_b = before2[before2["id"]==rid].iloc[0].to_dict()
                        changed = any(str(row_b[k]) != str(r[k]) for k in ["symbol","ccy","side","qty","price","note","is_edit"])
                        if changed:
                            cur_u.execute("""
                              UPDATE investor_trades
                              SET symbol=?, ccy=?, side=?, qty=?, price=?, note=?, is_edit=?
                              WHERE id=?
                            """, (r["symbol"], r["ccy"], r["side"], float(r["qty"]), float(r["price"]), r["note"], int(r["is_edit"]), rid))
                            updated_it += 1
                    conn_u.commit()

                before_dt = it_full[["trade_id","일자"]].copy()
                before_dt["일자"] = pd.to_datetime(before_dt["일자"], errors="coerce").dt.date
                after_dt = edited[["trade_id","일자"]].copy()
                def _to_iso(d):
                    if pd.isna(d): return None
                    if isinstance(d, (datetime, pd.Timestamp)): return d.date().isoformat()
                    if isinstance(d, date): return d.isoformat()
                    try: return pd.to_datetime(d).date().isoformat()
                    except Exception: return None
                before_dt_map = before_dt.dropna().drop_duplicates(subset=["trade_id"]).set_index("trade_id")["일자"].to_dict()
                after_dt_map_raw = after_dt.dropna().drop_duplicates(subset=["trade_id"]).set_index("trade_id")["일자"].to_dict()
                to_update = []
                for tid, new_d in after_dt_map_raw.items():
                    new_iso = _to_iso(new_d)
                    old_d = before_dt_map.get(tid, None)
                    old_iso = _to_iso(old_d) if old_d is not None else None
                    if new_iso and new_iso != old_iso:
                        to_update.append((new_iso, int(tid)))

                updated_trades = 0
                if to_update:
                    with get_conn() as conn_u2:
                        cur_u2 = conn_u2.cursor()
                        cur_u2.executemany("UPDATE trades SET dt=? WHERE id=?", to_update)
                        conn_u2.commit()
                        updated_trades = len(to_update)

                if (updated_it > 0) or (updated_trades > 0):
                    rebuild_trade_effects(from_dt=None)
                invalidate_all_caches()
                st.success(f"거래 {updated_trades}건 날짜 변경, 원장 {updated_it}건 저장 완료"); _rr()
        with c2:
            if st.button("새로고침", key="it_recent_like_reload"): _rr()

# ==============================
# Positions
# ==============================
with T4:
    st.markdown("### 투자자별 잔고")
    inv_df = list_investors()
    names_opt = ["(전체)"] + inv_df["name"].tolist() if not inv_df.empty else ["(전체)"]
    sel_local = st.selectbox("투자자별 잔고", names_opt, index=0)

    st.markdown("#### 보유 현황")
    pos = investor_positions()
    if sel_local == "(전체)":
        if not pos.empty:
            agg = pos.groupby(["symbol","ccy"], as_index=False).agg(qty=("qty","sum"), cost_basis=("cost_basis","sum"))
            agg["avg_px"] = agg.apply(lambda r: (r["cost_basis"]/r["qty"]) if r["qty"]>1e-12 else 0.0, axis=1)
            pos_view = agg.rename(columns={"symbol":"종목","ccy":"통화","qty":"보유 주식수","avg_px":"평균매수가","cost_basis":"총원가"})
            pos_view.insert(0, "투자자", "(전체)")
        else:
            pos_view = pd.DataFrame(columns=["투자자","종목","통화","보유 주식수","평균매수가","총원가"])
    else:
        if not pos.empty and "name" in pos.columns:
            pos = pos[pos["name"] == sel_local]
        pos_view = pos.rename(columns={"name":"투자자","symbol":"종목","ccy":"통화","qty":"보유 주식수","avg_px":"평균매수가","cost_basis":"총원가"}) if not pos.empty else pd.DataFrame(columns=["투자자","종목","통화","보유 주식수","평균매수가","총원가"])

    all_pos = pos if sel_local != "(전체)" else investor_positions()
    all_view = all_pos.rename(columns={"symbol":"종목","ccy":"통화","cost_basis":"총원가"}) if not all_pos.empty else pd.DataFrame(columns=["종목","통화","총원가"])
    if "총원가" not in all_view.columns and not all_view.empty:
        if "cost_basis" in all_view.columns:
            all_view = all_view.rename(columns={"cost_basis":"총원가"})
        else:
            all_view["총원가"] = 0.0

    tot_by_ccy = all_view.groupby("통화")["총원가"].sum().to_dict() if not all_view.empty else {}

    if not pos_view.empty:
        denom_local = pos_view.groupby("통화")["총원가"].sum().to_dict()
        pos_view["보유비중(%)"] = pos_view.apply(lambda r: (float(r["총원가"])/float(denom_local.get(r["통화"], 0.0))*100.0) if float(denom_local.get(r["통화"],0.0))>0 else np.nan, axis=1)
        pos_view["전체계좌비중(%)"] = pos_view.apply(lambda r: (float(r["총원가"])/float(tot_by_ccy.get(r["통화"], 0.0))*100.0) if float(tot_by_ccy.get(r["통화"],0.0))>0 else np.nan, axis=1)
        pos_view["괴리(%)"] = pos_view.apply(lambda r: (r["보유비중(%)"] - r["전체계좌비중(%)"]) if (not pd.isna(r["보유비중(%)"]) and not pd.isna(r["전체계좌비중(%)"])) else np.nan, axis=1)
        pos_view["보유 주식수"] = pos_view["보유 주식수"].apply(fmt_qty_2)
        pos_view = apply_row_ccy_format(pos_view, "통화", [], ["평균매수가"], ["총원가"])
        for c in ["보유비중(%)","전체계좌비중(%)","괴리(%)"]:
            pos_view[c] = pos_view[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):,.2f}")
        pos_view = pos_view[["투자자","종목","통화","보유 주식수","평균매수가","총원가","보유비중(%)","전체계좌비중(%)","괴리(%)"]]

    st.dataframe(
        pos_view,
        use_container_width=True,
        column_config={
            "투자자": st.column_config.TextColumn("투자자", width=144),
            "종목": st.column_config.TextColumn("종목", width=400),
            "통화": st.column_config.TextColumn("통화", width=60),
            "보유 주식수": st.column_config.TextColumn("보유 주식수", width=120),
            "평균매수가": st.column_config.TextColumn("평균매수가", width=144),
            "총원가": st.column_config.TextColumn("총원가", width=168),
            "보유비중(%)": st.column_config.TextColumn("보유비중(%)", width=84),
            "전체계좌비중(%)": st.column_config.TextColumn("전체계좌비중(%)", width=98),
            "괴리(%)": st.column_config.TextColumn("괴리(%)", width=96),
        }
    )

    st.markdown("#### 잔고(예수금)")
    bal_krw = investor_balances("KRW").rename(columns={"name":"투자자","cash":"KRW 예수금"})[["투자자","KRW 예수금"]]
    bal_usd = investor_balances("USD").rename(columns={"name":"투자자","cash":"USD 예수금"})[["투자자","USD 예수금"]]
    bal = pd.merge(bal_krw, bal_usd, on="투자자", how="outer").fillna(0)
    if not bal.empty:
        bal["KRW 예수금"] = bal["KRW 예수금"].map(lambda x: fmt_by_ccy(x, "KRW", "amount"))
        bal["USD 예수금"] = bal["USD 예수금"].map(lambda x: fmt_by_ccy(x, "USD", "amount"))
    st.dataframe(bal, use_container_width=True)

# ==============================
# Dividends  (비운용자 0.9, 운용자 = 자기지분 + 총액의 10% 추가)
# ==============================
with T5:
    st.markdown("### 배당 입력")
    with st.form("div_form"):
        syms = load_df("SELECT DISTINCT symbol FROM investor_trades ORDER BY symbol")
        sym_opts = syms["symbol"].tolist() if not syms.empty else []
        symbol_div = st.selectbox("종목", options=(sym_opts + [""])[::-1])
        ccy_div = st.radio("통화", SUPPORTED_CCY, horizontal=True, key="div_ccy_radio")
        total_txt = st.text_input("배당 총액", "0")
        total_amt = truncate_amount(total_txt, ccy_div)
        col_d1, col_d2 = st.columns(2)
        with col_d1: deal_dt = st.date_input("거래일", value=date.today(), key="div_deal_dt")
        with col_d2: record_dt = st.date_input("배당락일(보유 기준일)", value=date.today(), key="div_record_dt")
        with st.expander("추가 옵션", expanded=False):
            note_div = st.text_input("비고(선택)", "")
        ok_div = st.form_submit_button("분배 실행")
        if ok_div:
            if not symbol_div: st.warning("종목을 선택하세요.")
            elif total_amt <= 0: st.warning("배당 총액을 입력하세요.")
            else:
                # --- 여기서부터 교체 ---
                # --- 배당 분배(정정 버전): 1.0 입금 + 0.1 수수료 OUT + 운용자 1회 IN ---
                pos_rec = load_df("""
                    SELECT it.investor_id, it.symbol, it.ccy,
                           SUM(CASE WHEN it.side='BUY' THEN it.qty
                                    WHEN it.side='SELL' THEN -it.qty ELSE 0 END) AS pos_qty
                    FROM investor_trades it
                    JOIN trades t ON t.id = it.trade_id
                    WHERE it.symbol = ? AND it.ccy = ? AND date(t.dt) <= date(?)
                    GROUP BY it.investor_id, it.symbol, it.ccy
                    HAVING SUM(CASE WHEN it.side='BUY' THEN it.qty
                                    WHEN it.side='SELL' THEN -it.qty ELSE 0 END) > 0
                """, params=(symbol_div, ccy_div, record_dt.isoformat()))

                if pos_rec.empty:
                    st.warning("배당락일 기준 보유자가 없습니다.")
                else:
                    op_id = get_investor_id_by_name(OPERATOR_NAME)
                    memo = note_div if 'note_div' in locals() and note_div else f"배당 {symbol_div} (배당락일 {record_dt.isoformat()})"

                    with get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute(
                            "INSERT INTO dividends(dt, symbol, ccy, total_amount, note, record_dt) VALUES (?,?,?,?,?,?)",
                            (
                                deal_dt.isoformat(),
                                symbol_div,
                                ccy_div,
                                truncate_amount(total_amt, ccy_div),
                                memo,
                                record_dt.isoformat(),
                            ),
                        )

                        dividend_id = cur.lastrowid
                        rows_cf = build_dividend_cashflows(
                            dividend_id=dividend_id,
                            deal_dt=deal_dt,
                            symbol=symbol_div,
                            ccy=ccy_div,
                            total_amt=total_amt,
                            memo=memo,
                            pos_rec=pos_rec,
                            op_id=op_id,
                        )

                        if rows_cf:
                            cur.executemany(
                                """
                                INSERT INTO cash_flows(
                                    investor_id, dt, ccy, type, amount, note, source, dividend_id
                                ) VALUES (?,?,?,?,?,?,?,?)
                                """,
                                rows_cf,
                            )
                        conn.commit()

                    invalidate_all_caches()
                    st.success("배당 분배 완료")
                    _rr()
                # --- 교체 끝 ---

    st.markdown("### 최근 배당 분배 내역")
    div_meta = load_df(
        """
        SELECT id, dt, symbol, ccy, total_amount, note
        FROM dividends
        ORDER BY date(dt) DESC, id DESC
        LIMIT 50
        """
    )
    if div_meta.empty:
        st.info("최근 배당 분배 내역이 없습니다.")
    else:
        try:
            min_dt = pd.to_datetime(div_meta["dt"], errors="coerce").min().date()
        except Exception:
            min_dt = None

        if min_dt:
            cf_recent = load_df(
                """
                SELECT cf.investor_id, inv.name AS 투자자, cf.dt, cf.ccy AS 통화,
                       cf.type, cf.amount, cf.note, cf.dividend_id
                FROM cash_flows cf
                JOIN investors inv ON inv.id = cf.investor_id
                WHERE cf.source='DIVIDEND' AND date(cf.dt) >= date(?)
                """,
                params=(min_dt.isoformat(),),
            )
        else:
            cf_recent = load_df(
                """
                SELECT cf.investor_id, inv.name AS 투자자, cf.dt, cf.ccy AS 통화,
                       cf.type, cf.amount, cf.note, cf.dividend_id
                FROM cash_flows cf
                JOIN investors inv ON inv.id = cf.investor_id
                WHERE cf.source='DIVIDEND'
                """
            )

        if cf_recent.empty:
            st.info("최근 배당 분배 내역이 없습니다.")
        else:
            div_summary = summarize_dividend_history(div_meta, cf_recent)
            if div_summary.empty:
                st.info("최근 배당 분배 내역이 없습니다.")
            else:
                div_summary_fmt = apply_row_ccy_format(
                    div_summary.copy(),
                    "통화",
                    [],
                    [],
                    ["원배당금액", "운용자수수료", "순지급금액"],
                )
                try:
                    div_summary_fmt["일자"] = pd.to_datetime(
                        div_summary_fmt["일자"], errors="coerce"
                    ).dt.date
                except Exception:
                    pass
                st.dataframe(div_summary_fmt, use_container_width=True)

# ==============================
# Sell Notice
# ==============================
with T6:
    st.markdown("### 매도 공지용")
    preset = st.radio("기간 선택", ["1일","1주","1개월","6개월","1년","사용자 지정"], horizontal=True, index=2, key="sell_notice_range")
    today = date.today()
    delta_map = {"1일":1, "1주":7, "1개월":30, "6개월":182, "1년":365}
    if preset != "사용자 지정":
        start_dt = today - timedelta(days=delta_map[preset]); end_dt = today
        st.markdown('<div class="disabled-dates">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.date_input("시작일", value=start_dt, disabled=True, key="sell_start_disabled")
        with c2: st.date_input("종료일", value=end_dt, disabled=True, key="sell_end_disabled")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1: start_dt = st.date_input("시작일", value=today - timedelta(days=30), key="sell_start")
        with c2: end_dt = st.date_input("종료일", value=today, key="sell_end")

    sell_df = load_df("""
        SELECT it.symbol, it.ccy, date(t.dt) AS 일자,
               SUM(it.qty) AS 매도수량, SUM(it.qty * it.price) AS 매도금액,
               CASE WHEN SUM(it.qty)=0 THEN 0 ELSE SUM(it.qty * it.price)/SUM(it.qty) END AS 매도가
        FROM investor_trades it
        JOIN trades t ON t.id = it.trade_id
        WHERE it.side='SELL' AND date(t.dt) BETWEEN date(?) AND date(?)
        GROUP BY it.symbol, it.ccy, date(t.dt)
        ORDER BY 일자 DESC, 매도금액 DESC
    """, params=(start_dt.isoformat(), end_dt.isoformat()))
    buy_avg = load_df("""
        SELECT symbol, ccy,
               CASE WHEN SUM(qty)=0 THEN 0 ELSE SUM(qty * price)/SUM(qty) END AS 매수가
        FROM investor_trades
        WHERE side='BUY'
        GROUP BY symbol, ccy
    """)
    if not sell_df.empty:
        df = sell_df.merge(buy_avg, on=["symbol","ccy"], how="left").rename(columns={"symbol":"종목명"})
        df["수익률(%)"] = df.apply(
            lambda r: np.nan if (pd.isna(r["매수가"]) or r["매수가"]<=0 or pd.isna(r["매도가"]))
            else (r["매도가"]/r["매수가"]-1.0)*100.0, axis=1
        )
        krw_df = investor_balances("KRW"); usd_df = investor_balances("USD")
        total_cash_k = float((krw_df["cash"].sum() if not krw_df.empty else 0.0) + (usd_df["cash"].sum() if not usd_df.empty else 0.0)*USD_KRW)
        pos_all = investor_positions()
        inv_sum_ccy = pos_all.groupby("ccy")["cost_basis"].sum() if not pos_all.empty else pd.Series(dtype=float)
        total_invest_k = float(inv_sum_ccy.get("KRW", 0.0)) + float(inv_sum_ccy.get("USD", 0.0))*USD_KRW
        dashboard_total = max(1e-9, total_cash_k + total_invest_k)
        df["비중(%)"] = df["매도금액"].apply(lambda v: (float(v)/dashboard_total)*100.0 if dashboard_total>0 else np.nan)

        show = df[["종목명","일자","매수가","매도가","수익률(%)","비중(%)"]].copy()
        try: show["일자"] = pd.to_datetime(show["일자"], errors="coerce").dt.date
        except: pass
        for col in ["매수가","매도가"]:
            show[col] = show[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):,.0f}")
        for col in ["수익률(%)","비중(%)"]:
            show[col] = show[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):,.2f}")

        st.dataframe(
            show,
            use_container_width=True,
            column_config={
                "종목명": st.column_config.TextColumn("종목명", width=240),
                "일자": st.column_config.DateColumn("일자", format="YYYY-MM-DD", width=140),
                "매수가": st.column_config.TextColumn("매수가", width=120),
                "매도가": st.column_config.TextColumn("매도가", width=120),
                "수익률(%)": st.column_config.TextColumn("수익률(%)", width=110),
                "비중(%)": st.column_config.TextColumn("비중(%)", width=110),
            }
        )
    else:
        st.info("해당 기간 매도 데이터가 없습니다.")

# ==============================
# Profit (기간별 조회 + 실현차익)
# ==============================
with T7:
    st.markdown("### 수익")
    preset = st.radio("기간 선택", ["1일","1주","1개월","6개월","1년","사용자 지정"], horizontal=True, index=2, key="pnl_range")
    today = date.today()
    delta_map = {"1일":1, "1주":7, "1개월":30, "6개월":182, "1년":365}
    if preset != "사용자 지정":
        p_start = today - timedelta(days=delta_map[preset]); p_end = today
        st.markdown('<div class="disabled-dates">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.date_input("시작일", value=p_start, disabled=True, key="pnl_start_disabled")
        with c2: st.date_input("종료일", value=p_end, disabled=True, key="pnl_end_disabled")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1: p_start = st.date_input("시작일", value=today - timedelta(days=30), key="pnl_start")
        with c2: p_end   = st.date_input("종료일", value=today, key="pnl_end")

    pnl = load_df("""
      SELECT rp.investor_id, inv.name AS 투자자, rp.ccy AS 통화, rp.amount AS 차익, t.dt AS 일자
      FROM realized_pnl rp
      JOIN investors inv ON inv.id = rp.investor_id
      JOIN trades t ON t.id = rp.trade_id
      WHERE date(t.dt) BETWEEN date(?) AND date(?)
      ORDER BY date(t.dt) DESC, rp.investor_id
    """, params=(p_start.isoformat(), p_end.isoformat()))
    if pnl.empty:
        st.info("해당 기간 실현 차익 데이터가 없습니다.")
    else:
        try: pnl["일자"] = pd.to_datetime(pnl["일자"], errors="coerce").dt.date
        except: pass
        st.markdown("#### 투자자별 실현 차익")
        agg_inv = pnl.groupby(["투자자","통화"], as_index=False)["차익"].sum()
        def _fmt_row(row):
            return fmt_by_ccy(row["차익"], row["통화"], "amount")
        show_inv = agg_inv.copy()
        show_inv["차익"] = show_inv.apply(_fmt_row, axis=1)
        st.dataframe(show_inv, use_container_width=True)

        st.markdown("#### 거래별 실현 차익 (상세)")
        show_tx = pnl.copy()
        show_tx["표시 차익"] = show_tx.apply(lambda r: fmt_by_ccy(r["차익"], r["통화"], "amount"), axis=1)
        st.dataframe(show_tx[["일자","투자자","통화","표시 차익"]], use_container_width=True)

# ==============================
# Admin / Delete
# ==============================
def js_delete_key(trigger_btn_id: str):
    components.html(f"""
        <script>
          const sendClick = () => {{
            const btns = window.parent.document.querySelectorAll('button');
            for (const b of btns) {{
              if ((b.innerText.trim?.() ?? b.innerText).includes('{trigger_btn_id}')) {{ b.click(); break; }}
            }}
          }};
          window.addEventListener('keydown', (e) => {{
            if (e.key === 'Delete') {{ sendClick(); }}
          }});
        </script>
    """, height=0)

def delete_cash_flows(ids: List[int]) -> int:
    if not ids: return 0
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM cash_flows WHERE id IN ({','.join('?'*len(ids))})", ids)
        conn.commit()
        return cur.rowcount

def delete_investor_trades(ids: List[int]) -> int:
    if not ids: return 0
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM investor_trades WHERE id IN ({','.join('?'*len(ids))})", ids)
        cur.execute("DELETE FROM trades WHERE id NOT IN (SELECT DISTINCT trade_id FROM investor_trades)")
        cur.execute("DELETE FROM realized_pnl WHERE trade_id NOT IN (SELECT id FROM trades)")
        conn.commit()
    rebuild_trade_effects(from_dt=None)
    return len(ids)

def delete_dividends(ids: List[int]) -> int:
    if not ids: return 0
    with get_conn() as conn:
        cur = conn.cursor()
        placeholders = ','.join('?'*len(ids))
        meta_rows = cur.execute(
            f"SELECT id, dt, symbol, note FROM dividends WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

        cur.execute(f"DELETE FROM dividends WHERE id IN ({placeholders})", ids)
        cur.execute(f"DELETE FROM cash_flows WHERE dividend_id IN ({placeholders})", ids)

        for did, dtv, symbol, note in meta_rows:
            extra_notes = [note or ""]
            if symbol:
                extra_notes.extend(
                    [
                        f"배당 성과수수료 {symbol}",
                        f"배당 성과수수료 합계 {symbol}",
                        f"배당 반올림 조정 {symbol}",
                    ]
                )
            unique_notes = list(dict.fromkeys(extra_notes))
            if unique_notes:
                cur.execute(
                    f"DELETE FROM cash_flows WHERE source='DIVIDEND' AND dividend_id IS NULL AND date(dt)=date(?) AND note IN ({','.join('?'*len(unique_notes))})",
                    (dtv, *unique_notes),
                )
        conn.commit()
    invalidate_all_caches()
    return len(ids)

def delete_investors_ids(ids: List[int]) -> int:
    if not ids: return 0
    op_id = get_investor_id_by_name(OPERATOR_NAME)
    ids2 = [i for i in ids if i != op_id]
    if not ids2: return 0
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM investors WHERE id IN ({','.join('?'*len(ids2))})", ids2)
        conn.commit()
        return cur.rowcount

with T8:
    if st.button("← 나가기 (대시보드)"):
        components.html("<script>Array.from(window.parent.document.querySelectorAll('button[role=tab]')).find(b=>b.innerText.includes('대시보드'))?.click()</script>", height=0)

    st.markdown("### 데이터 삭제/ 초기화")
    tabA, tabB, tabC, tabD = st.tabs(["입출금", "투자자별 거래", "배당", "투자자"])

    def grid_with_delete(df: pd.DataFrame, id_col: str, delete_cb, title: str, key_prefix: str):
        st.markdown(f"#### {title}")
        if df.empty:
            st.info("데이터가 없습니다."); return
        view = df.copy()
        sel_col = "선택"
        view.insert(0, sel_col, False)
        ed = st.data_editor(
            view,
            key=f"{key_prefix}_grid",
            height=360,
            use_container_width=True,
            hide_index=True,
            column_config={ sel_col: st.column_config.CheckboxColumn(sel_col, help="삭제할 행 체크") },
            disabled=[c for c in view.columns if c != sel_col],
        )
        ids = []
        if isinstance(ed, pd.DataFrame) and id_col in ed.columns:
            ids = ed[ed[sel_col] == True][id_col].tolist()
        c1, c2, _ = st.columns([1,1,6])
        with c1:
            click = st.button("선택 삭제(Delete)", key=f"{key_prefix}_delete", disabled=(len(ids)==0))
        with c2:
            if st.button("새로고침", key=f"{key_prefix}_reload"): _rr()
        hidden_label = f"__{key_prefix}_DEL_TRIGGER__"
        hidden_fire = st.button(hidden_label, key=f"{key_prefix}_del_hidden", disabled=(len(ids)==0))
        js_delete_key(hidden_label)
        if click or hidden_fire:
            if not ids:
                st.warning("삭제할 행을 선택하세요.")
            else:
                cnt = delete_cb(ids)
                invalidate_all_caches()
                st.success(f"{cnt}건 삭제 완료"); _rr()

    with tabA:
        df = load_df("""
          SELECT cf.id as id, cf.dt as 일자, inv.name as 투자자, cf.ccy as 통화, cf.type as 유형,
                 cf.amount as 금액, cf.note as 비고, cf.source as 출처
          FROM cash_flows cf
          JOIN investors inv ON inv.id=cf.investor_id
          WHERE cf.source != 'TRADE'
          ORDER BY date(cf.dt) DESC, cf.id DESC LIMIT 1000
        """)
        grid_with_delete(df, "id", delete_cash_flows, "입출금 원장(삭제 가능, 매수/매도 생성분 제외)", "cf")

    with tabB:
        st.markdown("##### 기간별 삭제")
        col_d1, col_d2, col_d3 = st.columns([1,1,1])
        with col_d1: d_start = st.date_input("시작일", value=date.today() - timedelta(days=30), key="bulk_del_start")
        with col_d2: d_end   = st.date_input("종료일", value=date.today(), key="bulk_del_end")
        with col_d3:
            if st.button("기간 내 거래 삭제", key="bulk_del_btn"):
                with get_conn() as conn:
                    cur = conn.cursor()
                    tids = pd.read_sql_query("""
                        SELECT id FROM trades WHERE date(dt) BETWEEN date(?) AND date(?)
                    """, conn, params=(d_start.isoformat(), d_end.isoformat()))
                    if not tids.empty:
                        ids_list = tids["id"].tolist()
                        cur.execute(f"DELETE FROM investor_trades WHERE trade_id IN ({','.join('?'*len(ids_list))})", ids_list)
                        cur.execute("DELETE FROM trades WHERE id NOT IN (SELECT DISTINCT trade_id FROM investor_trades)")
                        cur.execute("DELETE FROM realized_pnl WHERE trade_id NOT IN (SELECT id FROM trades)")
                        conn.commit()
                    rebuild_trade_effects(from_dt=None)
                invalidate_all_caches()
                st.success("기간 내 거래 삭제 완료"); _rr()

        df = load_df("""
          SELECT it.id as id, t.dt as 일자, inv.name as 투자자, it.symbol as 종목, it.ccy as 통화,
                 it.side as 매수매도, it.qty as 수량, it.price as 단가, (it.qty*it.price) as 금액, it.note as 비고
          FROM investor_trades it
          JOIN trades t ON t.id = it.trade_id
          JOIN investors inv ON inv.id = it.investor_id
          ORDER BY date(t.dt) DESC, it.id DESC LIMIT 1000
        """)
        grid_with_delete(df, "id", delete_investor_trades, "투자자별 거래(삭제 가능)", "it")

    with tabC:
        df = load_df("""
          SELECT id, dt as 거래일, record_dt as 배당락일, symbol as 종목, ccy as 통화, total_amount as 총액, note as 비고
          FROM dividends
          ORDER BY id DESC LIMIT 1000
        """)
        grid_with_delete(df, "id", delete_dividends, "배당(삭제 가능)", "dv")

    with tabD:
        df = load_df("SELECT id, name as 투자자, created_at as 생성일 FROM investors ORDER BY id DESC")
        grid_with_delete(df, "id", delete_investors_ids, "투자자(삭제 가능)", "iv")
