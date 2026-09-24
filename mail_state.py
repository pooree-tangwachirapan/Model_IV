"""
mail_state.py — จำว่า "เมลรอบไหนของวันไหนส่งไปแล้ว" เพื่อกันส่งซ้ำข้าม job

ทำไมต้องมี — `concurrency` ของ GitHub **ไม่ได้กันเมลซ้ำ**
────────────────────────────────────────────────────────────────────────
ตอนแรกเข้าใจว่า `cancel-in-progress: false` แปลว่าชั้น cron ที่มาทีหลังจะถูกทิ้ง
**ผิด** — มันแค่ "ต่อคิว" พอชั้นที่วิ่งอยู่จบ ชั้นที่รอคิวจะเริ่มทันที
แล้วเห็นว่าเวลาเป้าเพิ่งผ่านไป ~5 นาที ซึ่งยังอยู่ใน LATE_GRACE_MIN → **ส่งซ้ำ**

ไทม์ไลน์จริงของวันที่ GitHub ไม่ดีเลย์เลย (EDT):
    08:17  ชั้นนี้ได้ไป → นอนรอ
    13:00  ส่ง premarket ✔
    13:04  job จบ → ชั้น 12:17 ที่ค้างคิวเริ่มวิ่งทันที
    13:05  มันคำนวณว่า premarket สายไป 5 นาที (ยังอยู่ใน grace 20) → ส่งอีกใบ
    14:00  ส่ง open30 ✔ · 14:04 จบ → ชั้น 13:17 เริ่ม → ส่ง open30 อีกใบ
    = วันละ 4 ใบ แทนที่จะเป็น 2

เวลาอย่างเดียวแยกไม่ออกระหว่าง "ใบซ้ำ" กับ "ชั้นแรกที่เริ่มสายจริง" —
ต้องมีหลักฐานว่า **ส่งไปแล้ว** ซึ่งข้ามระหว่าง job ได้ นั่นคือไฟล์นี้

เก็บที่ไหน
────────────────────────────────────────────────────────────────────────
ไฟล์ JSON เล็ก ๆ ที่ workflow เอาเข้า/ออกผ่าน `actions/cache` — รูปแบบเดียวกับที่
`.gate-state.json` ของ armed-alert ใช้อยู่แล้วและพิสูจน์ว่าได้ผล
ไม่ commit ลง repo เพราะมันเป็นสถานะชั่วคราวรายวัน ไม่ใช่ข้อมูลที่ต้องเก็บถาวร

ถ้า cache หาย (GitHub ล้าง cache ที่ไม่ถูกใช้ 7 วัน) อย่างแย่สุดคือได้เมลซ้ำหนึ่งใบ
ไม่ใช่ไม่ได้เมล — เลือกให้พังไปทางที่ปลอดภัยกว่า

เก็บย้อนหลัง KEEP_DAYS วันแล้วตัดทิ้ง ไม่ให้ไฟล์โตไปเรื่อย ๆ
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

import market_clock as mc

STATE_FILE = os.environ.get("MAIL_STATE_FILE", ".mail-sent.json")
KEEP_DAYS = 10


def load(path: str | None = None) -> dict:
    p = path or STATE_FILE
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError) as e:
        # ไฟล์เสียต้องไม่ทำให้เมลไม่ออก — ถือว่ายังไม่เคยส่ง แล้วเขียนทับใหม่
        print(f"  !! อ่าน {p} ไม่ได้ ({type(e).__name__}: {e}) — ถือว่ายังไม่เคยส่ง",
              file=sys.stderr)
        return {}


def save(state: dict, path: str | None = None) -> None:
    p = path or STATE_FILE
    cutoff = (mc.today_et() - timedelta(days=KEEP_DAYS)).isoformat()
    trimmed = {k: sorted(set(v)) for k, v in state.items() if k >= cutoff}
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(trimmed, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write("\n")


def already_sent(slot: str, day: date | str | None = None,
                 path: str | None = None) -> bool:
    day = str(day or mc.today_et())
    return slot in load(path).get(day, [])


def mark(slot: str, day: date | str | None = None, path: str | None = None) -> dict:
    day = str(day or mc.today_et())
    state = load(path)
    state.setdefault(day, [])
    if slot not in state[day]:
        state[day].append(slot)
    save(state, path)
    return state


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="จำว่าเมลรอบไหนส่งไปแล้ว — กันส่งซ้ำเมื่อ cron หลายชั้นวิ่งต่อกัน")
    ap.add_argument("cmd", choices=["check", "mark", "show"],
                    help="check = exit 0 ถ้า **ยังไม่ได้ส่ง** (bash เอาไปใช้ต่อได้ตรง ๆ)")
    ap.add_argument("slot", nargs="?", default="", help="premarket | open30 | monthly-YYYY-MM")
    ap.add_argument("--day", default="", help="YYYY-MM-DD (ว่าง = วันนี้ตามเวลา ET)")
    ap.add_argument("--file", default="", help=f"ไฟล์สถานะ (ค่าตั้งต้น {STATE_FILE})")
    a = ap.parse_args()

    path = a.file or None
    day = a.day or None

    if a.cmd == "show":
        print(json.dumps(load(path), ensure_ascii=False, indent=1, sort_keys=True))
        return 0

    if not a.slot:
        print("ต้องระบุ slot", file=sys.stderr)
        return 2

    if a.cmd == "check":
        sent = already_sent(a.slot, day, path)
        print("sent" if sent else "not-sent")
        return 1 if sent else 0          # exit 0 = ยังไม่ส่ง = ส่งได้

    mark(a.slot, day, path)
    print(f"บันทึกแล้วว่า {a.slot} ของวัน {day or mc.today_et()} ส่งไปแล้ว")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
