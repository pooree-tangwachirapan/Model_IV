"""
lp_store.py — ที่เก็บ context log ของแท็บ Long Premium (ไม่มี UI)   [LP]

╔═══════════════════════════════════════════════════════════════════════╗
║ [LP] LONG-PREMIUM TAB — ไฟล์ใหม่ 31 ส.ค. 2026                          ║
║ ถ้าเจอบั๊กที่ไม่เคยเกิดก่อนมีแท็บนี้ → ไฟล์นี้ลบทิ้งได้ทั้งไฟล์             ║
║ ระบบเดิม (fade/breakout/email/workflow) ไม่ import อะไรจากที่นี่เลย       ║
╚═══════════════════════════════════════════════════════════════════════╝

═══ ทำไมต้องมีไฟล์นี้ ═══

dataset ที่จับ **โครงสร้าง option (GEX/EM/IV) + tape (VWAP/VP/spike) ณ เวลาเดียวกัน**
ยังไม่มีอยู่จริงสำหรับ QQQ — CBOE ให้ snapshot ปัจจุบันเท่านั้น ไม่มีย้อนหลัง (HANDOFF §4)

ผลคือคำถามอย่าง "ตอน −GEX แล้วราคาหลุด value area มันวิ่งจริงไหม" **ตอบไม่ได้เลย**
ไม่ใช่เพราะไม่มีโค้ด แต่เพราะไม่มีใครเคยเก็บข้อมูลสองอย่างนี้ไว้พร้อมกัน
ถ้าไม่เริ่มเก็บวันนี้ อีกสามเดือนก็ยังตอบไม่ได้เหมือนเดิม

═══ ทำไมเป็น JSONL และทำไมอยู่คนละโฟลเดอร์กับ forward_test/ ═══

- **JSONL**: หนึ่งบรรทัด = หนึ่ง record · ต่อท้ายได้โดยไม่ต้องอ่านทั้งไฟล์
  และไฟล์เสียบางบรรทัดไม่ทำให้ทั้งไฟล์ใช้ไม่ได้ (ต่างจาก JSON ก้อนเดียว)
  รูปแบบเดียวกับ `forward_test/log/*.jsonl` ที่โปรเจกต์ใช้อยู่แล้ว

- **อยู่ `long_premium/` ไม่ใช่ `forward_test/`**: HANDOFF §6.4 บันทึกไว้ว่าไฟล์ที่ workflow
  สร้างหายเงียบมาแล้ว 3 ครั้ง (git add ระบุไฟล์เดียว · pull --rebase กลืน conflict · ไม่มีขั้น commit)
  ถ้าเอาไฟล์ใหม่ไปวางในโฟลเดอร์ที่ workflow แตะอยู่ จะเสี่ยงทั้งไฟล์เราหาย
  และเสี่ยงทำ merge ของระบบเดิมพัง — แยกโฟลเดอร์คือวิธีที่ไม่ต้องเดาเลยว่าใครแตะอะไร
"""

from __future__ import annotations

import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "long_premium")
LOG_PATH = os.path.join(LOG_DIR, "context_log.jsonl")


def append(record: dict, path: str = LOG_PATH) -> str:
    """
    ต่อท้าย 1 record — สร้างโฟลเดอร์ให้เองถ้ายังไม่มี

    เขียนเป็น UTF-8 เสมอ (console เครื่องงานเป็น cp874 — ถ้าไม่ระบุจะพังตอนมีภาษาไทย)
    `ensure_ascii=False` เพื่อให้เปิดไฟล์อ่านด้วยตาได้ ไม่ใช่ \\uXXXX เต็มไปหมด
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = dict(record)
    rec.setdefault("logged_at", datetime.now().isoformat(timespec="seconds"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return path


def read_all(path: str = LOG_PATH) -> list[dict]:
    """
    อ่านทุก record — **ข้ามบรรทัดที่เสียแทนที่จะโยน exception**

    เหตุผล: ไฟล์ log ที่ถูกเขียนตอนโปรแกรมถูกฆ่ากลางคัน จะมีบรรทัดสุดท้ายไม่ครบ
    ถ้าปล่อยให้ throw ข้อมูลดี ๆ ที่เก็บมาหลายเดือนจะอ่านไม่ได้เลยเพราะบรรทัดเดียว
    """
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue           # บรรทัดเสีย — ข้าม ไม่ล้มทั้งไฟล์
            if isinstance(obj, dict):
                out.append(obj)
    return out


def stats(path: str = LOG_PATH) -> dict:
    """สรุปว่าเก็บมาได้เท่าไหร่แล้ว — ใช้โชว์บนแท็บว่ายังต้องรออีกกี่วัน"""
    recs = read_all(path)
    days = sorted({r.get("session_date") for r in recs if r.get("session_date")})
    return {
        "n_records": len(recs),
        "n_sessions": len(days),
        "first_session": days[0] if days else None,
        "last_session": days[-1] if days else None,
        "path": path,
        "exists": os.path.exists(path),
    }


def flatten(snap: dict | None, ctx: dict | None,
            fade: dict | None = None, breakout_v: dict | None = None) -> dict:
    """
    รวม snapshot (โครงสร้าง option) + context (tape) + verdict ของ 2 ระบบเดิม → record เดียว

    เก็บเป็น **field แบน ไม่ซ้อน** เพื่อให้โหลดเข้า pandas ได้ตรง ๆ ตอนวิเคราะห์ทีหลัง
    ค่าไหนไม่มีให้เป็น None ไปเลย ห้ามเดาหรือเติมค่าแทน —
    ตอนวิเคราะห์ต้องแยกออกว่า "ไม่มีข้อมูล" กับ "มีข้อมูลแล้วค่าเป็นศูนย์" คนละเรื่องกัน
    """
    rec: dict = {"schema": 1}

    if ctx:
        rec.update({k: v for k, v in ctx.items() if k != "error"})
        rec["ctx_error"] = ctx.get("error")

    if snap and not snap.get("error"):
        lv = snap.get("levels") or {}
        em = snap.get("expected_move") or {}
        rv = snap.get("rv") or {}
        macro = snap.get("macro") or {}
        sk = snap.get("skew") or {}
        term = snap.get("term") or {}
        pin = snap.get("pin") or {}
        iv_atm = snap.get("atm_iv")
        hv20 = rv.get("hv20")
        rec.update({
            "spot_cboe": snap.get("spot"),
            "near_expiry": snap.get("near_expiry"), "near_dte": snap.get("near_dte"),
            "net_gex": lv.get("net"), "net_gex_profile": lv.get("net_profile"),
            "net_gex_agree": lv.get("net_agree"),          # False = เครื่องหมายเชื่อไม่ได้ (HANDOFF §6.8)
            "gamma_flip": lv.get("flip"),
            "call_wall": lv.get("call_wall"), "put_wall": lv.get("put_wall"),
            "call_wall_oi": lv.get("call_wall_oi"), "put_wall_oi": lv.get("put_wall_oi"),
            "max_pain": snap.get("max_pain"),
            "em": em.get("em"), "em_pct": em.get("em_pct"), "em_dte": em.get("dte"),
            "atm_iv": iv_atm, "hv20": hv20, "hv10": rv.get("hv10"),
            "vrp": (iv_atm - hv20) if (iv_atm and hv20) else None,
            "vix": macro.get("vix"), "vvix": macro.get("vvix"),
            "rr25": sk.get("rr25"),
            "term_slope_pct": term.get("slope_pct"), "term_state": term.get("state"),
            "pin_score": pin.get("score"),
            "zero_dte_share": snap.get("zero_dte"),
        })
    elif snap:
        rec["snap_error"] = snap.get("error")

    # verdict ของ 2 ระบบเดิม ณ ขณะนั้น — อ่านอย่างเดียว ไม่เปลี่ยนอะไรของเขา
    for tag, v in (("fade", fade), ("breakout", breakout_v)):
        if not v:
            continue
        plan = v.get("plan") or {}
        rec.update({
            f"{tag}_verdict": v.get("verdict"),
            f"{tag}_side": plan.get("side"),
            f"{tag}_entry": plan.get("ideal_entry"),
            f"{tag}_target": plan.get("target"),
            f"{tag}_invalid": plan.get("invalidation"),
            f"{tag}_plan_r": plan.get("plan_r"),
        })
    return rec
