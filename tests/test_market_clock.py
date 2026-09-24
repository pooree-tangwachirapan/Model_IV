"""
ตรวจ market_clock.py ด้วยคำตอบที่รู้แน่ (known answer) ไม่ใช่แค่ "ไม่ crash"

ทำไมชุดนี้สำคัญกว่าที่คิด: ทุก workflow ตัดสินใจจากไฟล์นี้ว่า "วันนี้ทำงานไหม"
และ "เป้าหมายเวลาคือกี่โมง" — ถ้าไฟล์นี้ผิด เมลจะออกผิดเวลาแบบเงียบ ๆ
ซึ่งเป็นอาการเดียวกับบั๊กที่เพิ่งแก้ไป (เมลออก success ทุกวัน แต่สาย 4 ชั่วโมง)

วันที่ในชุดนี้เทียบกับปฏิทิน NYSE จริง ปี 2026–2028
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta, timezone

import market_clock as mc

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


print("=== วันหยุดตลาด เทียบปฏิทิน NYSE จริง ===")
EXPECT = {
    2026: ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
           "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"],
    2027: ["2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
           "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"],
    # 2028: 1 ม.ค. ตรงวันเสาร์ → NYSE **ไม่** ชดเชยเป็นศุกร์ 31 ธ.ค. (กฎเฉพาะวันปีใหม่)
    2028: ["2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19",
           "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25"],
}
for y, want in EXPECT.items():
    got = sorted(d.isoformat() for d in mc.holidays(y))
    check(f"วันหยุด {y} ครบและตรง", got == want, f"ได้ {got}")

check("2028 ไม่มีวันหยุดปีใหม่ (ตรงเสาร์ ไม่ชดเชย)",
      date(2027, 12, 31) not in mc.holidays(2027)
      and date(2028, 1, 1) not in mc.holidays(2028))

print("\n=== วันทำการ ===")
check("4 ก.ค. 2026 ตรงวันเสาร์ → หยุดศุกร์ 3 ก.ค.", not mc.is_trading_day(date(2026, 7, 3)))
check("19 มิ.ย. 2027 ตรงวันเสาร์ → หยุดศุกร์ 18 มิ.ย.", not mc.is_trading_day(date(2027, 6, 18)))
check("4 ก.ค. 2027 ตรงวันอาทิตย์ → หยุดจันทร์ 5 ก.ค.", not mc.is_trading_day(date(2027, 7, 5)))
check("เสาร์ไม่ใช่วันทำการ", not mc.is_trading_day(date(2026, 9, 26)))
check("พฤหัสปกติเป็นวันทำการ", mc.is_trading_day(date(2026, 9, 24)))
check("next_trading_day ข้ามสุดสัปดาห์",
      mc.next_trading_day(date(2026, 9, 25)) == date(2026, 9, 28))
check("next_trading_day ข้ามวันหยุด (26 พ.ย. Thanksgiving)",
      mc.next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27))
check("prev_trading_day ย้อนข้ามวันหยุด",
      mc.prev_trading_day(date(2026, 11, 27)) == date(2026, 11, 25))

print("\n=== DST: เวลาเดียวกันของ ET ต้องได้ UTC ต่างกันระหว่าง EDT กับ EST ===")
# นี่คือบั๊กที่ workflow เดิมมี — ฝัง UTC ไว้ตรง ๆ แล้วคลาด 1 ชม. ครึ่งปี
edt = mc.slot_utc(date(2026, 9, 24), "premarket")      # EDT (UTC−4)
est = mc.slot_utc(date(2026, 12, 15), "premarket")     # EST (UTC−5)
check("premarket ช่วง EDT = 13:00 UTC", edt.strftime("%H:%M") == "13:00", str(edt))
check("premarket ช่วง EST = 14:00 UTC", est.strftime("%H:%M") == "14:00", str(est))
check("open ช่วง EDT = 13:30 UTC",
      mc.slot_utc(date(2026, 9, 24), "open").strftime("%H:%M") == "13:30")
check("open30 ช่วง EST = 15:00 UTC",
      mc.slot_utc(date(2026, 12, 15), "open30").strftime("%H:%M") == "15:00")

print("\n=== slot: ระยะห่างต้องคงที่เสมอ ไม่ว่าฤดูไหน ===")
for d in (date(2026, 9, 24), date(2026, 12, 15), date(2027, 3, 10)):
    o = mc.slot_utc(d, "open")
    check(f"{d} premarket = open − 30 นาที",
          mc.slot_utc(d, "premarket") == o - timedelta(minutes=30))
    check(f"{d} open30 = open + 30 นาที",
          mc.slot_utc(d, "open30") == o + timedelta(minutes=30))

print("\n=== วันครึ่งวัน ===")
check("27 พ.ย. 2026 (ศุกร์หลัง Thanksgiving) = ครึ่งวัน", mc.is_early_close(date(2026, 11, 27)))
check("24 ธ.ค. 2026 = ครึ่งวัน", mc.is_early_close(date(2026, 12, 24)))
check("24 ก.ย. 2026 ไม่ใช่ครึ่งวัน", not mc.is_early_close(date(2026, 9, 24)))
o, c = mc.session(date(2026, 11, 27))
check("ครึ่งวันปิด 13:00 ET (18:00 UTC ช่วง EST)", c.strftime("%H:%M") == "18:00", str(c))
o, c = mc.session(date(2026, 9, 24))
check("วันปกติปิด 16:00 ET (20:00 UTC ช่วง EDT)", c.strftime("%H:%M") == "20:00", str(c))

print("\n=== วันที่ตลาดปิด ต้องคืน None ไม่ใช่เดาเวลาให้ ===")
check("session ของวันเสาร์ = None", mc.session(date(2026, 9, 26)) is None)
check("slot_utc ของวันหยุด = None", mc.slot_utc(date(2026, 12, 25), "premarket") is None)

print("\n=== session_progress ===")
p = mc.session_progress(datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc))
check("08:00 ET = ก่อนเปิด", p["phase"] == "premarket", str(p["phase"]))
p = mc.session_progress(datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc))
check("11:00 ET = ตลาดเปิดอยู่", p["phase"] == "regular", str(p["phase"]))
check("นับนาทีจากเปิดได้ถูก (11:00 ET = +90)", abs(p["minutes_from_open"] - 90) < 0.5,
      str(p["minutes_from_open"]))
p = mc.session_progress(datetime(2026, 9, 24, 21, 0, tzinfo=timezone.utc))
check("17:00 ET = หลังปิด", p["phase"] == "after", str(p["phase"]))
p = mc.session_progress(datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc))
check("วันเสาร์ = closed และไม่มีเวลาเปิด/ปิด",
      p["phase"] == "closed" and p["open_utc"] is None and not p["trading"])

print("\n=== CLI ที่ workflow เรียก ===")
import subprocess
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def run(*a):
    return subprocess.run([sys.executable, os.path.join(ROOT, "market_clock.py"), *a],
                          capture_output=True, text=True, encoding="utf-8", env=env)

r = run("trading", "--date", "2026-09-26")
check("CLI trading วันเสาร์ → 0", r.stdout.strip() == "0", r.stdout)
r = run("trading", "--date", "2026-09-24")
check("CLI trading วันทำการ → 1", r.stdout.strip() == "1", r.stdout)
r = run("epoch", "--slot", "premarket", "--date", "2026-09-24")
check("CLI epoch คืนตัวเลขที่ตรงกับ slot_utc",
      r.stdout.strip() == str(int(mc.slot_utc(date(2026, 9, 24), "premarket").timestamp())),
      r.stdout)
r = run("epoch", "--slot", "premarket", "--date", "2026-12-25")
check("CLI epoch วันหยุด → exit 1 (workflow ใช้เป็นสัญญาณข้ามวัน)",
      r.returncode == 1, f"exit {r.returncode}")

print(f"\nสรุป: {'ผ่านหมด' if not FAIL else f'ไม่ผ่าน {len(FAIL)} เคส'}")
raise SystemExit(1 if FAIL else 0)
