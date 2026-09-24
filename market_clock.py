"""
market_clock.py — นาฬิกา/ปฏิทินตลาดหุ้นสหรัฐ (DST-aware, ไม่ต้องดูแลรายปี)

ทำไมต้องมีไฟล์นี้
────────────────────────────────────────────────────────────────────────
ก่อนหน้านี้ทุก workflow ฝังเวลา UTC ไว้ตรง ๆ เช่น `WINDOW_END=18:35` (= 14:35 ET)
ซึ่ง **จริงแค่ช่วง EDT** พอเข้า EST (พ.ย.–มี.ค.) ทุกบรรทัดคลาดไป 1 ชั่วโมง
และไม่มีใครรู้จักวันหยุดตลาด → วันหยุดก็ยังเผารันเนอร์ + เขียน log ที่ไม่มีความหมาย

ไฟล์นี้จึงเป็น "แหล่งความจริงเดียว" ของเวลาตลาด ให้ทั้ง workflow และโค้ด Python
เรียกใช้ตัวเดียวกัน — หน้าจอ อีเมล และตัวบันทึกจะไม่เถียงกันเรื่องเวลาอีก

วันหยุดคำนวณจาก "กฎ" ไม่ใช่ตารางที่ต้องมาเติมทุกปี
────────────────────────────────────────────────────────────────────────
NYSE/Nasdaq มีวันหยุด 9 ตัวที่เป็นกฎตายตัว + Good Friday ที่ผูกกับ Easter
เขียนเป็นกฎแล้วใช้ได้ทุกปีโดยไม่ต้องแก้ ถ้าใช้ตารางฝังไว้ ปีที่ไม่ได้เติม
is_trading_day() จะตอบ True ในวันหยุด ซึ่งเงียบและผิด

กฎ "วันชดเชย" ของ NYSE: ตรงวันเสาร์ หยุดศุกร์ก่อนหน้า · ตรงวันอาทิตย์ หยุดจันทร์ถัดไป
**ยกเว้นวันปีใหม่** ที่ตรงวันเสาร์ — ตลาดไม่หยุดศุกร์ที่ 31 ธ.ค.

ข้อจำกัดที่รู้ตัว: วันปิดกรณีพิเศษ (ไว้อาลัยประมุข · เฮอริเคน) คาดเดาไม่ได้
ใส่เพิ่มมือได้ที่ EXTRA_CLOSED
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)          # ครึ่งวัน

# วันปิดกรณีพิเศษที่กฎคาดเดาไม่ได้ — เติมมือเมื่อเกิด (รูปแบบ "YYYY-MM-DD")
EXTRA_CLOSED: set[str] = set()


# ════════════════════════════════════════════════
# วันหยุด — คำนวณจากกฎ
# ════════════════════════════════════════════════
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """วันที่ของ weekday ที่ n ของเดือน · weekday: 0=จันทร์ ... 6=อาทิตย์"""
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """weekday ตัวสุดท้ายของเดือน"""
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Easter Sunday (Anonymous Gregorian algorithm) — ใช้หา Good Friday"""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    lo = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lo) // 451
    month = (h + lo - 7 * m + 114) // 31
    day = ((h + lo - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date, *, shift_saturday: bool = True) -> date:
    """
    เลื่อนวันหยุดตามกฎ NYSE — เสาร์ → ศุกร์ก่อน · อาทิตย์ → จันทร์ถัดไป

    shift_saturday=False ใช้กับวันปีใหม่เท่านั้น: ถ้า 1 ม.ค. ตรงวันเสาร์
    ตลาด **ไม่** หยุดศุกร์ที่ 31 ธ.ค. ของปีก่อน (ต่างจากวันหยุดตัวอื่น)
    """
    if d.weekday() == 5:                       # เสาร์
        return d - timedelta(days=1) if shift_saturday else d
    if d.weekday() == 6:                       # อาทิตย์
        return d + timedelta(days=1)
    return d


def holidays(year: int) -> set[date]:
    """วันหยุดตลาดทั้งปี (วันที่ตลาดปิดจริง หลังเลื่อนวันชดเชยแล้ว)"""
    out = {
        _observed(date(year, 1, 1), shift_saturday=False),   # วันปีใหม่
        _nth_weekday(year, 1, 0, 3),                         # MLK — จันทร์ที่ 3 ม.ค.
        _nth_weekday(year, 2, 0, 3),                         # Washington's Birthday
        _easter(year) - timedelta(days=2),                   # Good Friday
        _last_weekday(year, 5, 0),                           # Memorial Day
        _observed(date(year, 6, 19)),                        # Juneteenth
        _observed(date(year, 7, 4)),                         # Independence Day
        _nth_weekday(year, 9, 0, 1),                         # Labor Day
        _nth_weekday(year, 11, 3, 4),                        # Thanksgiving — พฤหัสที่ 4
        _observed(date(year, 12, 25)),                       # Christmas
    }
    # ตัดตัวที่เลื่อนแล้วไปตกเสาร์/อาทิตย์ (เกิดกับวันปีใหม่ที่ไม่เลื่อน)
    return {d for d in out if d.weekday() < 5}


def early_closes(year: int) -> set[date]:
    """
    วันครึ่งวัน (ปิด 13:00 ET) — ศุกร์หลัง Thanksgiving ·
    วันก่อน Independence Day · Christmas Eve ที่เป็นวันทำการ
    """
    hol = holidays(year)
    out: set[date] = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}

    for d in (date(year, 7, 3), date(year, 12, 24)):
        if d.weekday() < 5 and d not in hol:
            out.add(d)

    return {d for d in out if d.weekday() < 5 and d not in hol}


def is_holiday(d: date) -> bool:
    return d.isoformat() in EXTRA_CLOSED or d in holidays(d.year)


def is_trading_day(d: date) -> bool:
    """วันที่ตลาดเปิด — ไม่ใช่เสาร์/อาทิตย์ และไม่ใช่วันหยุด"""
    return d.weekday() < 5 and not is_holiday(d)


def is_early_close(d: date) -> bool:
    return is_trading_day(d) and d in early_closes(d.year)


def next_trading_day(d: date) -> date:
    n = d + timedelta(days=1)
    while not is_trading_day(n):
        n += timedelta(days=1)
    return n


def prev_trading_day(d: date) -> date:
    p = d - timedelta(days=1)
    while not is_trading_day(p):
        p -= timedelta(days=1)
    return p


# ════════════════════════════════════════════════
# เวลาของ session
# ════════════════════════════════════════════════
def et_now() -> datetime:
    return datetime.now(UTC).astimezone(ET)


def today_et() -> date:
    return et_now().date()


def _et_at(d: date, t: time) -> datetime:
    """ประกอบ (วัน, เวลา ET) → datetime ที่มี tz ของ New York — DST ถูกเสมอ"""
    return datetime.combine(d, t, tzinfo=ET)


def session(d: date) -> tuple[datetime, datetime] | None:
    """(เปิด, ปิด) เป็น datetime UTC · None ถ้าวันนั้นตลาดไม่เปิด"""
    if not is_trading_day(d):
        return None
    close = EARLY_CLOSE if is_early_close(d) else REGULAR_CLOSE
    return (_et_at(d, REGULAR_OPEN).astimezone(UTC),
            _et_at(d, close).astimezone(UTC))


# ชื่อ slot → นาทีนับจากระฆังเปิด (ลบ = ก่อนเปิด)
SLOTS = {
    "premarket": -30,      # 09:00 ET — เมลเตรียมตัวก่อนตลาดเปิดครึ่งชั่วโมง
    "open": 0,             # 09:30 ET
    "open30": +30,         # 10:00 ET — สรุป opening range หลังเปิด 30 นาที
}


def slot_utc(d: date, slot: str) -> datetime | None:
    """เวลา UTC ของ slot ในวันนั้น · None ถ้าตลาดไม่เปิด"""
    s = session(d)
    if s is None:
        return None
    if slot == "close":
        return s[1]
    if slot not in SLOTS:
        raise ValueError(f"ไม่รู้จัก slot {slot!r} — มีให้ใช้: {sorted(SLOTS) + ['close']}")
    return s[0] + timedelta(minutes=SLOTS[slot])


def month_to_report(d: date | None = None) -> str | None:
    """
    ถ้า `d` เป็น **วันทำการแรกของเดือน** คืนเดือนก่อนหน้า ("2026-08") · ไม่ใช่ก็คืน None

    ใช้ตัดสินว่าวันนี้ควรส่งรายงานรายเดือนไหม — แทนการเช็ค `date -u +%d = "01"`
    ที่ของเดิมใช้ ซึ่งพังเพราะ GitHub ดีเลย์: cron ตั้ง 21:31 UTC วันที่ 1
    พอสาย 262 นาที (ค่าเฉลี่ยที่วัดได้) เวลาจริงข้ามเที่ยงคืนไปเป็นวันที่ 2
    แล้ว `date -u +%d` ตอบ "02" → ข้ามเงียบ **รายงานรายเดือนจึงแทบไม่เคยออกเลย**

    วิธีนี้ไม่ผูกกับนาฬิกา UTC และไม่ผูกกับ "วันที่ 1" ที่อาจเป็นวันหยุด/เสาร์อาทิตย์
    """
    d = d or today_et()
    if not is_trading_day(d):
        return None
    prev = prev_trading_day(d)
    if (prev.year, prev.month) == (d.year, d.month):
        return None                                # ไม่ใช่วันทำการแรกของเดือน
    return f"{prev.year:04d}-{prev.month:02d}"


def session_progress(now: datetime | None = None) -> dict:
    """
    สถานะตลาดตอนนี้ — ใช้ตัดสินใจในโค้ดและพิมพ์ให้คนอ่านได้ด้วย

    open_utc / close_utc เป็น None เมื่อวันนี้ตลาดปิด
    """
    now = (now or datetime.now(UTC)).astimezone(UTC)
    d = now.astimezone(ET).date()
    s = session(d)
    if s is None:
        return {"date": d.isoformat(), "trading": False, "phase": "closed",
                "open_utc": None, "close_utc": None, "minutes_from_open": None,
                "early_close": False}
    o, c = s
    mins = (now - o).total_seconds() / 60
    phase = "premarket" if now < o else ("regular" if now <= c else "after")
    return {"date": d.isoformat(), "trading": True, "phase": phase,
            "open_utc": o, "close_utc": c, "minutes_from_open": mins,
            "early_close": is_early_close(d)}


# ════════════════════════════════════════════════
# CLI — ให้ workflow (bash) ถามเวลาได้โดยไม่ต้องฝัง UTC ไว้ใน yml
# ════════════════════════════════════════════════
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="นาฬิกาตลาดสหรัฐ — ตอบเป็น epoch สำหรับ bash")
    ap.add_argument("what", choices=["epoch", "iso", "trading", "info", "report-month"],
                    help="epoch=วินาที UNIX · iso=ISO8601 · trading=1/0 · info=สรุปอ่านได้ · "
                         "report-month=เดือนที่ควรสรุป (ว่าง+exit 1 ถ้ายังไม่ถึงรอบ)")
    ap.add_argument("--slot", default="premarket",
                    help="premarket | open | open30 | close")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (ว่าง = วันนี้ตามเวลา ET)")
    a = ap.parse_args()

    d = date.fromisoformat(a.date) if a.date else today_et()

    if a.what == "trading":
        print("1" if is_trading_day(d) else "0")
        return 0

    if a.what == "report-month":
        m = month_to_report(d)
        print(m or "")
        return 0 if m else 1                       # exit 1 = ยังไม่ถึงรอบรายงาน

    if a.what == "info":
        # ต้องถาม `d` ไม่ใช่ session_progress() ซึ่งดูแต่ "วันนี้" —
        # ของเดิมใส่ --date เป็นวันหยุดแล้วพังด้วย TypeError เพราะ trading มาจากวันนี้
        # แต่ slot_utc ใช้วันที่ส่งมา (คนละวัน) แล้วคืน None
        p = session_progress() if d == today_et() else {
            "date": d.isoformat(), "trading": is_trading_day(d),
            "phase": "-", "early_close": is_early_close(d),
            "open_utc": session(d)[0] if is_trading_day(d) else None,
            "close_utc": session(d)[1] if is_trading_day(d) else None,
        }
        print(f"วันที่ (ET)     : {p['date']}")
        print(f"ตลาดเปิดวันนี้ : {'ใช่' if p['trading'] else 'ไม่ (เสาร์/อาทิตย์ หรือวันหยุด)'}")
        rm = month_to_report(d)
        if rm:
            print(f"รอบรายงานเดือน : {rm} (วันทำการแรกของเดือน)")
        if p["trading"]:
            print(f"ช่วง           : {p['phase']}")
            print(f"เปิด (UTC)     : {p['open_utc']:%Y-%m-%d %H:%M}")
            print(f"ปิด  (UTC)     : {p['close_utc']:%Y-%m-%d %H:%M}"
                  + ("  [ครึ่งวัน]" if p["early_close"] else ""))
            for s in ("premarket", "open", "open30", "close"):
                print(f"  slot {s:<10}: {slot_utc(d, s):%H:%M} UTC")
        return 0

    t = slot_utc(d, a.slot)
    if t is None:
        print("0" if a.what == "epoch" else "")
        return 1                                   # exit 1 = วันนี้ตลาดไม่เปิด
    print(int(t.timestamp()) if a.what == "epoch" else t.isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
