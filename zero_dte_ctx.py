"""
zero_dte_ctx.py — บริบทต่อ snapshot ของ 0DTE: VIX · VIX1D · VIX9D · VVIX

ทำไมต้องมี
────────────────────────────────────────────────────────────────────────
chain 0DTE ถูกเก็บทุก 15 นาที แต่ไม่มีโปรเจกต์ไหนเก็บ VIX ตระกูลนี้ระหว่างวันเลย
5 พ.ย. 2026 (QQQ −6.49%) VIX1D คือตัวอ่านสดที่มีประโยชน์ที่สุดตัวหนึ่ง —
Yahoo 1m: 10.93 ตอน 09:31 → 31.03 ราว 10:53 (ช่วงก้นของ QQQ) → ~23.1–23.9 ตอนเด้ง 11:30
→ 36.80 ตอนปิด · พอวันผ่านไป แท่ง 1 นาทีของ Yahoo ก็หลุดหน้าต่าง (~7 วัน) แล้วหายถาวร

รูปแบบข้อมูล
────────────────────────────────────────────────────────────────────────
    zero_dte/<SYM>/ctx/<YYYY-MM-DD>.csv     หนึ่งแถวต่อ snapshot ของ chain (~6 KB/วัน ไม่ต้อง rollup)

**อยู่ในโฟลเดอร์ย่อย `ctx/` โดยตั้งใจ** — ตัวอ่าน chain สองตัวหยิบ *ทุก* ไฟล์ .csv/.csv.gz
ที่อยู่ตรง zero_dte/<SYM>/ : quantlab/options_data.py (ทั้งอ่านในเครื่องและ list ผ่าน GitHub API)
กับ merge_data._zdte_files() · ถ้าวางเป็น <day>_ctx.csv ข้าง chain มันจะถูกอ่านเป็น chain
(ไม่มี strike/cp) · schema ของ chain (`zero_dte.COLUMNS`) ไม่เปลี่ยนแม้แต่คอลัมน์เดียว

    ts_utc        เวลา snapshot ของ chain ที่แถวนี้ผูกอยู่ — **สตริงเดียวกับใน <day>.csv** (join ตรงตัว)
    spot          spot ของ snapshot นั้น = ค่าเดียวกับใน chain (CBOE current_price ดีเลย์ ~15 นาที)
    fetched_utc   เวลาที่ดึง quote ชุดนี้
    vix,   vix_utc     ค่าล่าสุด + **เวลาของ quote** (UTC) — แต่ละดัชนีมีเวลาของตัวเอง
    vix1d, vix1d_utc   ⋮
    vix9d, vix9d_utc
    vvix,  vvix_utc
    src           แหล่งข้อมูล ("yahoo")

เทียบเวลาด้วยคอลัมน์ *_utc ไม่ใช่ด้วยแถว — spot ของ chain กับ quote ของดัชนีมาจากคนละแหล่ง
ดีเลย์ไม่เท่ากัน (06.11.2026 03:38 ET Yahoo ให้ ^VIX ที่เวลา 03:23:46 = ช้าราว 15 นาที)

แหล่งข้อมูล — Yahoo v8 chart (เลือกหลังวัดทั้งสองเจ้า 06.11.2026)
────────────────────────────────────────────────────────────────────────
- Yahoo `^VIX1D` interval=1m range=1d: แท่ง 1 นาทีของ 5 พ.ย. ครบ 390/390 ค่าเดินจริงทั้งวัน
  และ meta บอกเวลา quote ถึงวินาที (VIX1D ปิดที่ 16:15:01 ET) · ต้องใช้ UA เบราว์เซอร์ (UA ปกติ 429)
- CBOE `quotes/_VIX1D.json`: ค่าเดียว high/low = 0 และ `last_trade_time` ของ `_VIX` ค้างที่
  16:15 ของเมื่อวานขณะที่ current_price ขยับตาม GTH ไปแล้ว → ใช้เป็นเวลา quote ไม่ได้
- CBOE `charts/intraday/_VIX1D.json`: OHLC ตรงกับ Yahoo ทุกแท่ง แต่ติดป้ายแท่งด้วยนาที *จบ*
  (Yahoo ใช้นาที *เริ่ม* → ป้ายต่างกัน 1 นาที) · แท่งแรกเป็น 0 ทั้งแท่ง · ไฟล์ 90 KB
  และของ `_VIX` timeout ที่ 10 วินาที — เก็บไว้เป็นทางสำรองตอนย้อนดูวันล่าสุด ไม่ใช้ในตัวบันทึก

กติกา: ห้ามทำให้การบันทึก chain พังหรือช้า
────────────────────────────────────────────────────────────────────────
- workflow เรียกเป็น step แยก *หลัง* chain เขียนลงดิสก์แล้ว (continue-on-error + timeout)
- ไม่มีฟังก์ชันไหนในไฟล์นี้โยน exception ออกไป
- ไม่มี snapshot ของ chain รอบนี้ → ไม่ยิงเน็ต ไม่เขียน (แถวต้องผูกกับ snapshot เสมอ)
- ดึงไม่ได้สักตัว → ไม่เขียนอะไรเลย · ได้บางตัว → เขียนแถวพร้อมช่องว่างของตัวที่ไม่ได้
- เวลารวมของการดึงมีเพดาน `DEADLINE_S` — ยิงทีละตัว เลยเพดานแล้วข้ามตัวที่เหลือ
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd

import market_clock as mc
import zero_dte as z

# ชื่อคอลัมน์ → สัญลักษณ์ของ Yahoo (ลำดับนี้คือลำดับคอลัมน์ในไฟล์)
INDEXES = {"vix": "^VIX", "vix1d": "^VIX1D", "vix9d": "^VIX9D", "vvix": "^VVIX"}

# schema ของไฟล์ ctx — ห้ามสลับลำดับ (write_row ไม่ต่อท้ายไฟล์ที่หัวตารางไม่ตรง)
COLUMNS = (["ts_utc", "spot", "fetched_utc"]
           + [c for k in INDEXES for c in (k, f"{k}_utc")]
           + ["src"])

SRC = "yahoo"
YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")   # ตัวที่สอง = ลองซ้ำอีกโฮสต์
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TIMEOUT = (4, 6)              # (connect, read) วินาที ต่อคำขอ
DEADLINE_S = 20.0             # เพดานเวลารวมของการดึงทั้งชุด
MAX_SNAPSHOT_AGE_S = 300      # snapshot ของ chain ที่เก่ากว่านี้ = รอบนี้ chain ไม่ได้บันทึก


# ════════════════════════════════════════════════
# ที่อยู่ไฟล์
# ════════════════════════════════════════════════
def ctx_path(sym: str, day) -> str:
    """zero_dte/<SYM>/ctx/<day>.csv — อ่าน z.ZDTE_DIR ตอนเรียก เทสต์ย้ายไปโฟลเดอร์ชั่วคราวได้"""
    day = day if isinstance(day, str) else day.isoformat()
    return os.path.join(z.ZDTE_DIR, sym.upper(), "ctx", f"{day}.csv")


# ════════════════════════════════════════════════
# แกะ JSON ของ Yahoo
# ════════════════════════════════════════════════
def _pos(x) -> float | None:
    """ตัวเลขบวกจริงเท่านั้น — None / NaN / inf / 0 / ติดลบ / สตริงมั่ว = ไม่มีค่า"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _iso(epoch) -> str | None:
    """epoch วินาที → '2026-11-05T21:15:01+00:00' (รูปเดียวกับ ts_utc ของ chain)"""
    try:
        e = int(epoch)
    except (TypeError, ValueError):
        return None
    if e <= 0:
        return None
    try:
        return datetime.fromtimestamp(e, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def parse_chart(payload) -> tuple[float | None, str | None]:
    """
    JSON ของ Yahoo v8 chart → (ค่าล่าสุด, เวลา quote UTC) · อ่านไม่ได้ → (None, None)

    ใช้ meta.regularMarketPrice + regularMarketTime ก่อน — ละเอียดถึงวินาที และเป็นค่าที่
    ประกาศจริง (VIX1D 5 พ.ย.: 36.8 ที่ 16:15:01 ET ขณะที่แท่ง 1m สุดท้ายติดป้าย 16:00)
    ไม่มีครบทั้งคู่ → แท่ง 1m สุดท้ายที่ไม่ว่าง + เวลา*เริ่ม*แท่ง (ค่ากับเวลาต้องมาจากที่เดียวกัน)
    """
    try:
        r0 = payload["chart"]["result"][0]
        meta = r0.get("meta") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return None, None

    v, t = _pos(meta.get("regularMarketPrice")), _iso(meta.get("regularMarketTime"))
    if v is not None and t is not None:
        return round(v, 2), t

    try:
        ts = r0.get("timestamp") or []
        closes = (r0.get("indicators") or {}).get("quote")[0].get("close") or []
    except (TypeError, IndexError, AttributeError):
        return None, None
    for i in range(min(len(ts), len(closes)) - 1, -1, -1):
        v, t = _pos(closes[i]), _iso(ts[i])
        if v is not None and t is not None:
            return round(v, 2), t
    return None, None


# ════════════════════════════════════════════════
# ดึง quote
# ════════════════════════════════════════════════
def _get_json(url: str, timeout):
    import requests                                  # import ตอนใช้ — เทสต์ไม่ต้องมีเน็ต
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "application/json"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_quotes(*, http=None, clock=time.monotonic, deadline_s: float = DEADLINE_S) -> dict:
    """
    ค่าล่าสุดของทุกดัชนีใน INDEXES · **ไม่โยน exception**

    คืน {"values": {"vix1d": (36.8, "2026-11-05T21:15:01+00:00"), ...},
         "errors": {"vvix": "HTTPError: 429 ..."}, "fetched_utc": datetime}
    ตัวที่ดึงไม่ได้อยู่ใน errors เท่านั้น ไม่มีใน values

    `http(url, timeout) -> dict` กับ `clock` ฉีดได้ เพื่อเทสต์โดยไม่แตะเน็ต
    """
    http = http or _get_json
    t0 = clock()
    values: dict = {}
    errors: dict = {}
    for key, ysym in INDEXES.items():
        last = None
        for host in YAHOO_HOSTS:
            if clock() - t0 > deadline_s:
                last = (f"{last} · " if last else "") + f"เกินเพดานเวลา {deadline_s:g} วินาที"
                break
            url = f"https://{host}/v8/finance/chart/{quote(ysym)}?interval=1m&range=1d"
            try:
                v, t = parse_chart(http(url, TIMEOUT))
            except Exception as e:                             # noqa: BLE001
                last = f"{type(e).__name__}: {e}"[:200]
                continue
            if v is None:
                last = "ตอบกลับแต่ไม่มีราคา/เวลา quote"
                continue
            values[key] = (v, t)
            break
        if key not in values:
            errors[key] = last or "ไม่ทราบสาเหตุ"
    return {"values": values, "errors": errors, "fetched_utc": datetime.now(timezone.utc)}


# ════════════════════════════════════════════════
# snapshot ล่าสุดของ chain
# ════════════════════════════════════════════════
def last_snapshot(sym: str, *, now: datetime | None = None,
                  max_age_s: float = MAX_SNAPSHOT_AGE_S) -> dict | None:
    """
    snapshot ล่าสุดของ chain วันนี้ → {"ts_utc": "...", "spot": 647.9, "day": "2026-11-05"}

    ไม่มีไฟล์ / อ่านไม่ได้ / เก่ากว่า max_age_s (= รอบนี้ chain ล้ม ยังเป็นของรอบก่อน) → None
    ts_utc คืนเป็น **สตริงเดิมจากไฟล์** ไม่แปลงไปกลับ — join กับ chain ได้ตรงตัวอักษร
    """
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(mc.ET).date().isoformat()        # กติกาเดียวกับ z.write_rows
    p = z.path_for(sym, day)
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p, usecols=["ts_utc", "spot"], dtype={"ts_utc": str})
        d["ts"] = pd.to_datetime(d["ts_utc"], utc=True, errors="coerce")
        d = d[d["ts"].notna()]
        if d.empty:
            return None
        row = d.loc[d["ts"].idxmax()]
        age = (now - row["ts"].to_pydatetime()).total_seconds()
    except Exception as e:                                     # noqa: BLE001
        print(f"  !! อ่าน snapshot จาก {p} ไม่ได้ ({type(e).__name__}: {e})", file=sys.stderr)
        return None
    if not -60 <= age <= max_age_s:
        return None
    return {"ts_utc": str(row["ts_utc"]), "spot": _pos(row["spot"]), "day": day}


# ════════════════════════════════════════════════
# เขียนไฟล์
# ════════════════════════════════════════════════
def build_row(snap: dict, quotes: dict, fetched: datetime) -> dict | None:
    """แถวตาม COLUMNS · ไม่มีค่าสักตัว → None (= ไม่เขียน)"""
    vals = (quotes or {}).get("values") or {}
    if not vals:
        return None
    row = {"ts_utc": snap["ts_utc"], "spot": snap.get("spot"),
           "fetched_utc": fetched.astimezone(timezone.utc).replace(microsecond=0).isoformat()}
    for k in INDEXES:
        v, t = vals.get(k, (None, None))
        row[k], row[f"{k}_utc"] = v, t
    row["src"] = SRC
    return row


def write_row(sym: str, day: str, row: dict) -> dict:
    """
    ต่อท้าย zero_dte/<SYM>/ctx/<day>.csv · กันซ้ำด้วย ts_utc (รอบเดิมเรียกซ้ำไม่เพิ่มแถว)

    เขียนด้วย pandas + LF ตายตัว เหมือน chain — merge_data._merge_csv อ่านแล้วเขียนไฟล์นี้ใหม่
    ทั้งไฟล์ทุกรอบ commit ถ้ารูปตัวเลขไม่ตรงกับที่ pandas เขียน diff จะบวมทุก 15 นาที
    """
    p = ctx_path(sym, day)
    fresh = not os.path.exists(p) or os.path.getsize(p) == 0
    if not fresh:
        try:
            old = pd.read_csv(p, dtype=str)
        except Exception as e:                                 # noqa: BLE001
            # ไฟล์พัง: ต่อท้ายจะพังหนักขึ้น ทับก็ทำของเก่าหาย (§6.15) → ไม่แตะเลย
            return {"written": 0, "path": p,
                    "reason": f"อ่านไฟล์เดิมไม่ได้ ({type(e).__name__}) — ไม่เขียน"}
        if list(old.columns) != COLUMNS:
            return {"written": 0, "path": p,
                    "reason": "หัวตารางของไฟล์เดิมไม่ตรง schema — ไม่ต่อท้าย (คอลัมน์จะเลื่อน)"}
        if str(row["ts_utc"]) in set(old["ts_utc"].astype(str)):
            return {"written": 0, "path": p, "reason": "snapshot นี้มีบริบทแล้ว"}

    os.makedirs(os.path.dirname(p), exist_ok=True)
    buf = io.StringIO()
    pd.DataFrame([row], columns=COLUMNS).to_csv(buf, index=False, header=fresh,
                                                lineterminator="\n")
    with open(p, "a", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    return {"written": 1, "path": p, "reason": ""}


# ════════════════════════════════════════════════
# บันทึก
# ════════════════════════════════════════════════
def record(symbols=("QQQ",), *, fetch=None, now: datetime | None = None) -> dict:
    """
    บันทึกบริบทของ snapshot ล่าสุดของแต่ละ symbol · **ไม่โยน exception**

    คืน {"ok", "results": [{"sym", "written", "reason", "path"?}], "values", "errors"}
    ok = False เมื่อมี snapshot ให้ผูกแต่ดึงดัชนีไม่ได้สักตัว หรือพังกลางทาง
    ไม่มี snapshot รอบนี้ = ok True + written 0 (chain step เตือนเรื่องตัวเองไปแล้ว)
    """
    syms = [s.strip().upper() for s in symbols if s and s.strip()]
    try:
        now = now or datetime.now(timezone.utc)
        snaps = {s: last_snapshot(s, now=now) for s in syms}
        if not any(snaps.values()):
            return {"ok": True, "values": {}, "errors": {},
                    "results": [{"sym": s, "written": 0,
                                 "reason": "ไม่มี snapshot ของ chain รอบนี้ — ไม่บันทึกบริบท"}
                                for s in syms]}

        q = (fetch or fetch_quotes)()
        values, errors = q.get("values") or {}, q.get("errors") or {}
        if not values:
            return {"ok": False, "values": {}, "errors": errors,
                    "results": [{"sym": s, "written": 0,
                                 "reason": "ดึงดัชนีไม่ได้สักตัว — ไม่เขียนอะไร"} for s in syms]}

        fetched = q.get("fetched_utc") or now
        results = []
        for s in syms:
            snap = snaps.get(s)
            if not snap:
                results.append({"sym": s, "written": 0,
                                "reason": "ไม่มี snapshot ของ chain รอบนี้ — ไม่บันทึกบริบท"})
                continue
            results.append({"sym": s, **write_row(s, snap["day"], build_row(snap, q, fetched))})
        return {"ok": True, "values": values, "errors": errors, "results": results}
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "values": {}, "errors": {"*": f"{type(e).__name__}: {e}"},
                "results": [{"sym": s, "written": 0, "reason": "พังกลางทาง — ดู errors"}
                            for s in syms]}


__all__ = ["INDEXES", "COLUMNS", "ctx_path", "parse_chart", "fetch_quotes",
           "last_snapshot", "build_row", "write_row", "record"]
