"""
forward_test.py — เดินระบบ Cockpit แบบจำลอง เพื่อวัดว่ามันมี edge จริงไหม (ไม่มี UI)

กติกาที่จำลอง — ตรงกับที่ gate.py บังคับตอนเทรดจริง:
    พอร์ตตั้งต้น 5,000 USD · เสี่ยง 1.5%/ไม้ = 75 USD · สูงสุด 2 ไม้/วัน · QQQ อย่างเดียว
    เปิดไม้เมื่อ verdict = ARMED เท่านั้น · เข้าที่ "กำแพง" ตามที่ plan บอก ไม่ใช่ราคาตลาดตอนนั้น
    ถือได้ไม่เกิน 3 วันทำการ เกินนั้นปิดที่ราคาตลาด

วัดผลเป็น R ไม่ใช่ราคา option:
    ไม้นี้ชนะ = ราคาถึงเป้าก่อน invalidate → +plan_r × 75 USD
    ไม้นี้แพ้ = ราคาถึง invalidate ก่อน   → −75 USD
    เหตุผลที่ไม่จำลองราคา option: ต้องเดา IV/spread/fill ซึ่งจะกลายเป็นการวัดสมมติฐานของตัวเอง
    สิ่งที่อยากรู้จริงคือ "สัญญาณนี้ถูกทางกี่ครั้ง" ซึ่งวัดที่ underlying ได้ตรงกว่า

การตัดสินแพ้/ชนะใช้แท่ง 5 นาที ไม่ใช่ High/Low รายวัน:
    แท่งรายวันบอกไม่ได้ว่าอะไรโดนก่อน ถ้าวันนั้นแตะทั้งเป้าและ invalidate
    และมันยังนับราคาที่เกิด "ก่อน" เราเข้าไม้ด้วย ซึ่งเป็นการมองอนาคตย้อนหลัง
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

ACCOUNT_START = 5000.0
RISK_PCT = 1.5
RISK_USD = ACCOUNT_START * RISK_PCT / 100      # 75 USD — เก็บไว้อ้างอิงไม้เก่าเท่านั้น

# ขนาดไม้: หน่วยคงที่เท่ากันทั้ง LONG และ SHORT
# ไม่ใช้ความเสี่ยงคงที่ (75 USD ทุกไม้) เพราะแบบนั้นขนาดไม้จะแปรผกผันกับระยะ stop
# → ไม้ stop แคบได้ไซส์ใหญ่โดยอัตโนมัติ ทำให้ P&L รวมสะท้อน "ระยะ stop" มากกว่า "สัญญาณ"
# หน่วยคงที่ = ทุกไม้มีน้ำหนักเท่ากันในสถิติ อ่านผลเป็นเงินได้ตรง ๆ (1 จุด = QTY USD)
QTY = 100
MAX_TRADES_PER_DAY = 2
MAX_HOLD_DAYS = 3
SYMBOL = "QQQ"
LEDGER = os.path.join("forward_test", "ledger.json")


# ════════════════════════════════════════════════
# เก็บ / อ่าน
# ════════════════════════════════════════════════
def load(path: str = LEDGER) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save(trades: list[dict], path: str = LEDGER) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=1)


_CLOSED = ("win", "loss", "timeout")


def merge(mine: list[dict], theirs: list[dict]) -> list[dict]:
    """
    รวม ledger สองชุดโดยยึด id ไม่ใช่ข้อความในไฟล์

    ทำไมต้องมี: มีสอง workflow เขียนไฟล์เดียวกัน (armed-alert เปิดไม้ ·
    forward-test ปิดไม้) ถ้าทั้งคู่ commit ในช่วงเวลาใกล้กัน git จะ rebase ชนกัน
    แล้วฝั่งที่แพ้หายไปทั้งไม้ — เคยเกิดจริง 2026-08-12
    กติกา: ไม้ที่ "ปิดแล้ว" ชนะไม้ที่ยัง "เปิด" (ข้อมูลมากกว่า) · id ที่มีข้างเดียวเก็บไว้ทั้งคู่
    """
    by_id: dict[str, dict] = {}
    for t in list(theirs) + list(mine):        # mine ทีหลัง = ได้สิทธิ์เขียนทับก่อน
        tid = t.get("id")
        if not tid:
            continue
        cur = by_id.get(tid)
        if cur is None:
            by_id[tid] = t
            continue
        # ฝั่งไหนปิดแล้วเอาฝั่งนั้น ถ้าปิดทั้งคู่หรือเปิดทั้งคู่ ใช้ mine (ตัวหลัง)
        if cur.get("status") in _CLOSED and t.get("status") not in _CLOSED:
            continue
        by_id[tid] = t
    return sorted(by_id.values(), key=lambda t: (t.get("opened") or "", t.get("id") or ""))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ════════════════════════════════════════════════
# เปิดไม้
# ════════════════════════════════════════════════
def record(trades: list[dict], g: dict, now: datetime | None = None) -> dict | None:
    """
    เปิดไม้จำลองถ้า verdict = ARMED และกติกาอนุญาต — คืน trade ที่เพิ่งเปิด หรือ None

    ไม่เปิดซ้ำถ้ายังมีไม้ค้างอยู่ และไม่เปิดเกินลิมิตรายวัน
    (ตัวหลังคือกติกาเดียวกับที่ cockpit บังคับ ไม่ใช่ข้อจำกัดทางเทคนิค)
    """
    now = now or _utc_now()
    if g.get("verdict") != "ARMED":
        return None
    p = g.get("plan")
    if not p or not p.get("plan_r"):
        return None

    sym = g.get("symbol") or SYMBOL
    today = now.strftime("%Y-%m-%d")

    if any(t["symbol"] == sym and t["status"] == "open" for t in trades):
        return None                                   # ไม้เดิมยังไม่ปิด
    if sum(1 for t in trades if t["symbol"] == sym and t["date"] == today) >= MAX_TRADES_PER_DAY:
        return None                                   # ครบลิมิตของวัน

    t = {
        "id": f"{sym}-{now:%Y%m%d-%H%M%S}",
        "symbol": sym,
        "date": today,
        "opened": now.isoformat(timespec="seconds"),
        "side": p["side"],
        "entry": round(p["ideal_entry"], 2),
        "invalidation": round(p["invalidation"], 2),
        "target": round(p["target"], 2),
        "plan_r": round(p["plan_r"], 6),   # ใช้คูณเป็นเงิน — ปัดหยาบกว่านี้ P&L เพี้ยน
        "qty": QTY,
        # เงินที่เสี่ยงจริงของไม้นี้ = ระยะถึง invalidate × จำนวนหน่วย (ต่างกันได้ทุกไม้)
        "risk_usd": round(abs(round(p["ideal_entry"], 2) - round(p["invalidation"], 2)) * QTY, 2),
        "spot_at_signal": round(p["spot"], 2),
        "status": "open",
        "closed_date": None, "closed_at": None, "exit": None,
        "realized_r": None, "pnl_usd": None, "note": "",
    }
    trades.append(t)
    return t


# ════════════════════════════════════════════════
# ปิดไม้
# ════════════════════════════════════════════════
def _bars_5m(symbol: str, start: datetime, end: datetime):
    """แท่ง 5 นาทีจาก yfinance — คืน list[(ts_utc, high, low, close)] เรียงตามเวลา"""
    import yfinance as yf

    df = yf.download(symbol, interval="5m",
                     start=start.date(), end=(end + timedelta(days=1)).date(),
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return []
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)   # yfinance คืน MultiIndex เมื่อขอหลายตัว
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return [(ts.to_pydatetime(), float(h), float(l), float(c))
            for ts, h, l, c in zip(idx, df["High"], df["Low"], df["Close"])
            if h == h and l == l]                     # ตัด NaN


def resolve(trades: list[dict], bars_fn=_bars_5m, now: datetime | None = None) -> list[dict]:
    """
    เดินแท่ง 5 นาทีไปข้างหน้าจากเวลาที่เปิดไม้ — อะไรโดนก่อนอันนั้นชนะ
    คืนรายการไม้ที่เพิ่งถูกปิดในรอบนี้
    """
    now = now or _utc_now()
    closed = []
    for t in trades:
        if t["status"] != "open":
            continue
        opened = datetime.fromisoformat(t["opened"])
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        deadline = opened + timedelta(days=MAX_HOLD_DAYS)

        try:
            bars = bars_fn(t["symbol"], opened, min(now, deadline))
        except Exception as e:                          # noqa: BLE001 — ดึงราคาไม่ได้ = ไว้รอบหน้า
            t["note"] = f"ดึงราคาไม่ได้: {type(e).__name__}"
            continue

        bars = [b for b in bars if b[0] > opened]       # นับเฉพาะหลังเข้าไม้ ไม่มองย้อนหลัง
        if not bars:
            continue

        long_ = t["side"] == "LONG"
        hit = None
        for ts, hi, lo, close in bars:
            if ts > deadline:
                break
            # ตรวจ invalidate ก่อนเป้าเสมอเมื่ออยู่ในแท่งเดียวกัน — แท่ง 5 นาทีก็ยังบอกลำดับในแท่งไม่ได้
            # เลือกทางที่แย่กว่าไว้ก่อน จะได้ไม่หลอกตัวเอง
            if (long_ and lo <= t["invalidation"]) or (not long_ and hi >= t["invalidation"]):
                hit = ("loss", t["invalidation"], ts)
                break
            if (long_ and hi >= t["target"]) or (not long_ and lo <= t["target"]):
                hit = ("win", t["target"], ts)
                break

        if hit is None and now >= deadline:
            last_ts, _, _, last_close = bars[-1]
            hit = ("timeout", last_close, last_ts)

        if hit is None:
            continue

        status, exit_px, ts = hit
        # P&L คิดจากราคาที่เคลื่อน × จำนวนหน่วย — ตรงกับที่ถือจริง ไม่ต้องผ่าน R
        move = (exit_px - t["entry"]) if long_ else (t["entry"] - exit_px)
        risk_pts = abs(t["entry"] - t["invalidation"])
        qty = t.get("qty") or QTY
        r = (move / risk_pts) if risk_pts else 0.0
        t.update(status=status, exit=round(exit_px, 2),
                 closed_at=ts.isoformat(timespec="seconds"),
                 closed_date=ts.strftime("%Y-%m-%d"),
                 realized_r=round(r, 3), pnl_usd=round(move * qty, 2))
        closed.append(t)
    return closed


# ════════════════════════════════════════════════
# สถิติ
# ════════════════════════════════════════════════
def stats(trades: list[dict], month: str | None = None) -> dict:
    """month = 'YYYY-MM' หรือ None = ทั้งหมด · นับเฉพาะไม้ที่ปิดแล้ว"""
    sel = [t for t in trades if month is None or t["date"].startswith(month)]
    done = [t for t in sel if t["status"] in ("win", "loss", "timeout")
            and t.get("realized_r") is not None]
    out = {
        "month": month, "n_all": len(sel), "n_closed": len(done),
        "n_open": sum(1 for t in sel if t["status"] == "open"),
        "wins": 0, "losses": 0, "timeouts": 0, "win_rate": None,
        "expectancy_r": None, "total_r": 0.0, "pnl_usd": 0.0,
        "avg_win_r": None, "avg_loss_r": None, "rr": None, "breakeven_wr": None,
        "max_dd_usd": 0.0, "equity_start": ACCOUNT_START,
        "equity_end": ACCOUNT_START, "return_pct": 0.0, "equity_curve": [],
    }
    if not done:
        return out

    done = sorted(done, key=lambda t: t.get("closed_at") or t["opened"])
    out["wins"] = sum(1 for t in done if t["realized_r"] > 0)
    out["losses"] = sum(1 for t in done if t["realized_r"] < 0)
    out["timeouts"] = sum(1 for t in done if t["status"] == "timeout")
    out["win_rate"] = out["wins"] / len(done)
    out["total_r"] = round(sum(t["realized_r"] for t in done), 3)
    out["expectancy_r"] = round(out["total_r"] / len(done), 3)
    out["pnl_usd"] = round(sum(t["pnl_usd"] for t in done), 2)

    w = [t["realized_r"] for t in done if t["realized_r"] > 0]
    l = [abs(t["realized_r"]) for t in done if t["realized_r"] < 0]
    if w:
        out["avg_win_r"] = round(sum(w) / len(w), 3)
    if l:
        out["avg_loss_r"] = round(sum(l) / len(l), 3)
    if w and l:
        out["rr"] = round(out["avg_win_r"] / out["avg_loss_r"], 3)
        out["breakeven_wr"] = round(1 / (1 + out["rr"]), 4)

    eq, peak, dd = ACCOUNT_START, ACCOUNT_START, 0.0
    for t in done:
        eq += t["pnl_usd"]
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
        out["equity_curve"].append({"date": t.get("closed_date") or t["date"],
                                    "equity": round(eq, 2), "id": t["id"]})
    out["equity_end"] = round(eq, 2)
    out["max_dd_usd"] = round(dd, 2)
    out["return_pct"] = round((eq - ACCOUNT_START) / ACCOUNT_START * 100, 2)
    return out


def render_text(s: dict) -> str:
    """สรุปแบบข้อความ — ใช้ในอีเมลรายเดือนและปุ่มดาวน์โหลด"""
    if not s["n_closed"]:
        return (f"Forward Test {s['month'] or 'ทั้งหมด'} — ยังไม่มีไม้ที่ปิด "
                f"(เปิดค้าง {s['n_open']} ไม้)\n"
                "ยังไม่มีอะไรให้สรุป — ระบบเข้าไม้เฉพาะตอน ARMED ซึ่งเกิดไม่บ่อย")
    L = [f"Forward Test {s['month'] or 'ทั้งหมด'} — {SYMBOL} · พอร์ตจำลอง "
         f"${ACCOUNT_START:,.0f} · ไม้ละ {QTY} หน่วยเท่ากันทั้ง LONG/SHORT (1 จุด = ${QTY})",
         "=" * 74,
         f"  ไม้ที่ปิดแล้ว     {s['n_closed']}  (ชนะ {s['wins']} · แพ้ {s['losses']} "
         f"· หมดเวลา {s['timeouts']})" + (f"  · ยังค้าง {s['n_open']}" if s["n_open"] else ""),
         f"  Win rate         {s['win_rate']*100:.1f}%",
         f"  Expectancy       {s['expectancy_r']:+.3f}R ต่อไม้",
         f"  รวมทั้งเดือน      {s['total_r']:+.2f}R = ${s['pnl_usd']:+,.2f} "
         f"({s['return_pct']:+.2f}% ของพอร์ต)",
         f"  พอร์ต            ${s['equity_start']:,.0f} → ${s['equity_end']:,.2f}",
         f"  Max drawdown     ${s['max_dd_usd']:,.2f}"]
    if s["rr"]:
        L += [f"  RR จริง          1:{s['rr']:.2f}",
              f"  ต้องชนะเกิน      {s['breakeven_wr']*100:.1f}% ถึงจะเท่าทุน "
              f"({'ผ่าน' if s['win_rate'] > s['breakeven_wr'] else 'ยังไม่ผ่าน'})"]
    L += ["=" * 74]
    if s["n_closed"] < 20:
        L.append(f"⚠️ {s['n_closed']} ไม้ยังน้อยเกินกว่าจะสรุปว่ามี edge — ที่ win rate ระดับนี้ "
                 "ช่วงความเชื่อมั่นยังกว้างมาก ถือเป็นการดูว่าระบบเดินได้ ไม่ใช่ว่ามันกำไร")
    L.append(f"จำลองถือ {SYMBOL} ตรง ๆ {QTY} หน่วย/ไม้ **ไม่ได้จำลอง option** — "
             "ไม่มี theta, IV crush, สเปรด · ตัวเลขนี้ตอบว่า 'สัญญาณถูกทางมั้ย' "
             "ไม่ได้ตอบว่า 'เทรด option ตามนี้แล้วได้เท่านี้'")
    L.append("แตะ invalidate กับเป้าในแท่งเดียวกันนับเป็นแพ้")
    return "\n".join(L)
