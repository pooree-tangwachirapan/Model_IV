"""
predictions.py — บันทึก "ทุกครั้งที่ระบบตัดสิน" ไม่ใช่แค่ตอนที่มันยิง (ไม่มี UI)

ปัญหาที่แก้:
    ledger เก็บเฉพาะไม้ที่ ARMED แล้วเปิดจริง — วันที่ระบบบอก STAND_DOWN หายไปหมด
    ผ่านไปหนึ่งปีจึงตอบไม่ได้เลยว่า
      · ระบบเลือกมากแค่ไหน (ประเมินกี่ครั้ง ยิงกี่ครั้ง)
      · ประตูไหนเป็นตัวปิดบ่อยสุด — คือประตูที่กำหนดพฤติกรรมทั้งระบบ
      · วันที่ห้ามเทรด ถ้าเทรดไปจะได้หรือเสีย (ประตูนั้นคุ้มค่าที่จะมีไหม)
      · ถ้าเปลี่ยนเกณฑ์เป็นค่าอื่น ผลจะต่างไหม
    ทั้งสี่ข้อต้องใช้ "ภาพ ณ ตอนตัดสิน" ซึ่งถ้าไม่เก็บตอนนั้นก็สร้างย้อนหลังไม่ได้
    เพราะ chain ของ CBOE เป็น snapshot ปัจจุบัน ไม่มี API ย้อนหลังให้ฟรี

รูปแบบ: JSONL เดือนละไฟล์ — forward_test/log/YYYY-MM.jsonl
    เลือก JSONL เพราะเป็น append-only บรรทัดละ record
    · สอง workflow เขียนพร้อมกันแล้ว merge ง่ายกว่า JSON array (รวมบรรทัดแล้ว dedupe)
    · บรรทัดเดียวเสียไม่ทำให้ทั้งไฟล์อ่านไม่ได้
    · ไฟล์แยกรายเดือนกันไฟล์โตจนแก้ conflict ไม่ไหว

ขนาด: ~34 record/วัน (17 รอบ × 2 ระบบ) × 250 วันทำการ ≈ 8,500 record/ปี ≈ 3–4 MB/ปี

⚠️ สิ่งนี้ไม่ใช่ backtest ย้อนหลัง — มันคือ log ที่ "กลายเป็น" ชุดข้อมูล backtest เมื่อเวลาผ่านไป
   เริ่มเก็บวันนี้ = อีกหนึ่งปีมีข้อมูลหนึ่งปี · ไม่เริ่ม = อีกหนึ่งปีก็ยังไม่มีอะไร
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LOG_DIR = os.path.join("forward_test", "log")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def log_path(when: datetime | None = None) -> str:
    return os.path.join(LOG_DIR, f"{(when or _utc_now()):%Y-%m}.jsonl")


def _key(rec: dict) -> tuple:
    """หนึ่ง record = หนึ่ง (เวลา, สัญลักษณ์, ระบบ) — ใช้ dedupe ตอน merge"""
    return (rec.get("ts", ""), rec.get("sym", ""), rec.get("sys", ""))


def to_record(g: dict, snap: dict, when: datetime | None = None) -> dict:
    """
    แปลงผลตัดสิน 1 ครั้ง → record เดียว

    เก็บ "ค่าที่ใช้ตัดสิน" ไม่ใช่แค่คำตัดสิน — ไม่งั้นย้อนกลับไปคิดใหม่ด้วยเกณฑ์อื่นไม่ได้
    ใช้ key สั้นเพราะไฟล์นี้จะโตขึ้นทุกวันไปเรื่อย ๆ
    """
    lv = (snap or {}).get("levels") or {}
    em = ((snap or {}).get("expected_move") or {}).get("em")
    p = g.get("plan") or {}
    return {
        "ts": (when or _utc_now()).isoformat(timespec="seconds"),
        "sym": g.get("symbol"),
        "sys": g.get("system", "fade"),
        "v": g.get("verdict"),
        "data_issue": bool(g.get("data_issue")),
        "reason": (g.get("reason") or "")[:160],
        "spot": snap.get("spot") if snap else None,
        "lv": {
            "pw": lv.get("put_wall"), "cw": lv.get("call_wall"),
            "pw_oi": lv.get("put_wall_oi"), "cw_oi": lv.get("call_wall_oi"),
            "flip": lv.get("flip"), "net": lv.get("net"),
            "em": em, "mp": (snap or {}).get("max_pain"),
            "iv": (snap or {}).get("atm_iv"),
        },
        # เก็บผลรายประตู เพื่อตอบทีหลังว่า "ประตูไหนเป็นตัวปิด" โดยไม่ต้องเดาจาก reason
        "gates": {r["label"]: bool(r["ok"]) for r in (g.get("hard") or [])},
        "soft": {r["label"]: bool(r["ok"]) for r in (g.get("soft") or [])},
        # เก็บแผนแม้วันที่ไม่ได้เทรด — ใช้ตอบว่า "ถ้าฝืนเทรดวันนั้นจะได้หรือเสีย"
        "plan": ({"side": p.get("side"), "entry": p.get("ideal_entry"),
                  "stop": p.get("invalidation"), "target": p.get("target"),
                  "r": round(p["plan_r"], 3) if p.get("plan_r") else None}
                 if p else None),
    }


def load(path: str | None = None) -> list[dict]:
    """อ่านไฟล์เดียว — บรรทัดเสียข้ามไป ไม่ทำให้ทั้งไฟล์ตาย"""
    path = path or log_path()
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out


def load_all(directory: str = LOG_DIR) -> list[dict]:
    """อ่านทุกเดือนรวมกัน เรียงตามเวลา"""
    out = []
    try:
        names = sorted(n for n in os.listdir(directory) if n.endswith(".jsonl"))
    except OSError:
        return []
    for n in names:
        out.extend(load(os.path.join(directory, n)))
    return sorted(out, key=lambda r: r.get("ts", ""))


def append(records: list[dict], path: str | None = None) -> int:
    """
    เขียนต่อท้ายแบบ merge — อ่านของเดิมมารวมแล้ว dedupe ก่อนเขียน

    ทำไมไม่ append ตรง ๆ: อีก workflow อาจเขียนไฟล์เดียวกันระหว่างที่เราคำนวณอยู่
    (ดึง CBOE + yfinance ใช้เวลาหลายวินาที) เขียนทับตรง ๆ แล้ว record ของเขาหาย
    คืนจำนวน record ที่เพิ่มขึ้นจริง
    """
    if not records:
        return 0
    path = path or log_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing = load(path)
    seen = {_key(r) for r in existing}
    added = [r for r in records if _key(r) not in seen]
    if not added:
        return 0
    merged = sorted(existing + added, key=lambda r: (r.get("ts", ""), r.get("sym", ""), r.get("sys", "")))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)          # atomic — ไฟล์ไม่เหลือสภาพครึ่ง ๆ ถ้าโดนฆ่ากลางคัน
    return len(added)


def record_all(gates: list[dict], snaps: list[dict], when: datetime | None = None) -> int:
    """เรียกจาก send_report หลังประเมินครบทุกระบบ — จับคู่ gate กับ snapshot ตามสัญลักษณ์"""
    by_sym = {s.get("symbol"): s for s in (snaps or []) if not s.get("error")}
    recs = [to_record(g, by_sym.get(g.get("symbol")) or {}, when) for g in (gates or [])]
    return append(recs)


# ════════════════════════════════════════════════
# วิเคราะห์
# ════════════════════════════════════════════════
def summarize(records: list[dict], system: str | None = None,
              symbol: str | None = None) -> dict:
    """
    สรุปว่าระบบตัดสินอะไรไปบ้าง + ประตูไหนเป็นตัวปิด

    "ตัวปิด" นับเฉพาะตอนที่ประตูนั้นไม่ผ่าน — ประตูที่ไม่ผ่านบ่อยสุดคือประตูที่
    กำหนดพฤติกรรมของระบบจริง ๆ ส่วนประตูที่ผ่านตลอดแทบไม่มีผลอะไร
    """
    rs = [r for r in records
          if (system is None or r.get("sys") == system)
          and (symbol is None or r.get("sym") == symbol)]
    out = {"n": len(rs), "system": system, "symbol": symbol,
           "verdicts": {}, "blockers": {}, "gate_seen": {},
           "first": rs[0]["ts"][:10] if rs else None,
           "last": rs[-1]["ts"][:10] if rs else None,
           "days": len({r["ts"][:10] for r in rs}),
           "armed_days": sorted({r["ts"][:10] for r in rs if r.get("v") == "ARMED"}),
           }
    if not rs:
        return out
    for r in rs:
        v = "DATA_ISSUE" if r.get("data_issue") else (r.get("v") or "?")
        out["verdicts"][v] = out["verdicts"].get(v, 0) + 1
        for label, ok in (r.get("gates") or {}).items():
            out["gate_seen"][label] = out["gate_seen"].get(label, 0) + 1
            if not ok:
                out["blockers"][label] = out["blockers"].get(label, 0) + 1
    out["armed_rate"] = out["verdicts"].get("ARMED", 0) / len(rs)
    return out


def render_text(s: dict) -> str:
    if not s["n"]:
        return (f"ยังไม่มีบันทึกการตัดสิน"
                + (f" ของระบบ {s['system']}" if s.get("system") else "")
                + " — log จะเริ่มมีเมื่อ workflow รันรอบถัดไป")
    L = [f"บันทึกการตัดสิน — {s['system'] or 'ทุกระบบ'}"
         + (f" · {s['symbol']}" if s.get("symbol") else ""),
         f"  ช่วง             {s['first']} ถึง {s['last']}  ({s['days']} วัน · {s['n']} ครั้ง)",
         "  คำตัดสิน:"]
    for v, n in sorted(s["verdicts"].items(), key=lambda kv: -kv[1]):
        L.append(f"    {v:<12} {n:>5}  ({n/s['n']*100:.1f}%)")
    L.append(f"  ยิงสัญญาณ {s['armed_rate']*100:.1f}% ของการประเมิน"
             + (f" · วันที่ ARMED: {', '.join(s['armed_days'][-6:])}" if s["armed_days"] else ""))
    if s["blockers"]:
        L.append("  ประตูที่เป็นตัวปิด (นับครั้งที่ไม่ผ่าน / ครั้งที่ถูกตรวจ):")
        for label, n in sorted(s["blockers"].items(), key=lambda kv: -kv[1]):
            seen = s["gate_seen"].get(label, 0)
            L.append(f"    {label:<26} {n:>5} / {seen:<5} ({n/seen*100:.0f}%)" if seen else "")
    L.append("  ⚠️ log นี้บอกว่าระบบ 'ตัดสินอะไร' ไม่ได้บอกว่า 'ตัดสินถูกไหม' — "
             "ต้องเทียบกับราคาที่เกิดขึ้นจริงถึงจะรู้")
    return "\n".join(x for x in L if x)
