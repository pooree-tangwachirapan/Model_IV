"""
ตรวจตัวบันทึก 0DTE — ไม่แตะเน็ตเลย ป้อน chain ปลอมที่มีรูปร่างเหมือน CBOE ของจริง

เคสที่ต้องมีแน่ ๆ (มาจากของที่พังจริงตอนเขียน 24-25 ก.ย. 2026):
  · index-union: คำนวณคอลัมน์ก่อนกรองแล้วเอามาประกอบ DataFrame → ได้แถว NaN แถมมา
    แล้ว dedupe จับไม่ได้เพราะ NaN != NaN → ไฟล์บวมทุกครั้งที่รัน
  · expiry ที่หมดอายุแล้วต้องไม่ถูกนับเป็น 0DTE (บทเรียน HANDOFF §6.2)
  · เขียนซ้ำรอบเดิมต้องไม่เพิ่มแถว (cron มีบันไดหลายชั้น + กดมือทดสอบได้)
  · rollup แล้วต้องยังอ่านย้อนได้ครบ ไม่งั้นการประหยัดพื้นที่ = ทำข้อมูลหาย
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta, timezone

import pandas as pd

import zero_dte as z

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


# ── chain ปลอมที่มีรูปร่างเหมือน CBOE จริง ────────────────────────
def fake_chain(expiries=("260924", "260925", "260928"), spot=740.0,
               strikes=(700, 735, 740, 745, 800)):
    rows = []
    for e in expiries:
        for k in strikes:
            for cp in ("C", "P"):
                near = abs(k - spot) <= 10
                rows.append({
                    "option": f"QQQ{e}{cp}{int(k * 1000):08d}",
                    "bid": 2.0 if near else 0.0,
                    "ask": 2.2 if near else 0.05,
                    "iv": 0.19, "open_interest": 500, "volume": 1000 if near else 0,
                    "delta": 0.5, "gamma": 0.02, "vega": 0.1, "theta": -0.8,
                    "theo": 2.1, "open": 2.5, "high": 3.0, "low": 1.8,
                    "last_trade_price": 2.1, "prev_day_close": 2.4,
                })
    return pd.DataFrame(rows)


TODAY = date(2026, 9, 25)
TS = datetime(2026, 9, 25, 13, 35, tzinfo=timezone.utc)


print("=== แกะ OCC symbol ===")
check("แกะ call ถูก", z.parse_occ("QQQ260925C00740000") == ("2026-09-25", "C", 740.0))
check("แกะ put ที่มีเศษสตางค์ถูก",
      z.parse_occ("QQQ260925P00737500") == ("2026-09-25", "P", 737.5))
check("สัญลักษณ์เพี้ยนคืน None ไม่โยน exception",
      z.parse_occ("ไม่ใช่สัญลักษณ์") == (None, None, None))

print("\n=== เลือก expiry ที่เป็น 0DTE ===")
check("มี expiry วันนี้ → เลือกวันนี้",
      z.pick_expiry(["2026-09-24", "2026-09-25", "2026-09-28"], TODAY) == "2026-09-25")
# บทเรียน §6.2: ของเดิม clamp DTE แล้ว expiry ที่หมดอายุกลายเป็น "0DTE ปลอม"
check("expiry ที่ผ่านไปแล้วต้องไม่ถูกเลือก",
      z.pick_expiry(["2026-09-23", "2026-09-24"], TODAY) is None)
check("ไม่มี expiry วันนี้ → เลือกตัวถัดไปข้างหน้า",
      z.pick_expiry(["2026-09-24", "2026-09-28"], TODAY) == "2026-09-28")
check("รายการว่าง → None", z.pick_expiry([], TODAY) is None)

print("\n=== สร้างแถวจาก chain ===")
rows = z.build_rows(fake_chain(), 740.0, sym="QQQ", ts=TS, today=TODAY)
check("เลือกเฉพาะ expiry 0DTE", set(rows["expiry"]) == {"2026-09-25"}, str(set(rows["expiry"])))
check("dte = 0", set(rows["dte"]) == {0}, str(set(rows["dte"])))
check("คอลัมน์ครบตาม schema", list(rows.columns) == z.COLUMNS)

# ── บั๊กจริงที่เจอ 24 ก.ย. 2026 ──
check("ไม่มีแถวที่ strike/cp เป็น NaN (บั๊ก index-union)",
      not rows["strike"].isna().any() and not rows["cp"].isna().any())
check("จำนวนแถว = 2 × จำนวน strike ที่เก็บ พอดี",
      len(rows) == 2 * rows["strike"].nunique(), f"{len(rows)} แถว / {rows['strike'].nunique()} strike")

print("\n=== การกรอง strike ===")
r = z.build_rows(fake_chain(), 740.0, ts=TS, today=TODAY, band_pct=0.01)
check("กรอบแคบ: strike ไกลที่ไม่มี volume ถูกตัดออก",
      800.0 not in set(r["strike"]), str(sorted(set(r["strike"]))))
check("กรอบแคบ: strike ที่มี volume ยังอยู่แม้ห่างเกินกรอบ",
      735.0 in set(r["strike"]) and 745.0 in set(r["strike"]))
r = z.build_rows(fake_chain(), 740.0, ts=TS, today=TODAY, band_pct=1.0)
check("กรอบกว้าง 100%: เก็บทุก strike", len(r) == 10, f"{len(r)} แถว")

print("\n=== mid ===")
r = z.build_rows(fake_chain(), 740.0, ts=TS, today=TODAY)
atm = r[(r["strike"] == 740.0) & (r["cp"] == "C")].iloc[0]
check("mid = กลาง bid/ask เมื่อมีสองฝั่ง", abs(atm["mid"] - 2.1) < 1e-9, str(atm["mid"]))
one_side = fake_chain()
one_side.loc[one_side["option"] == "QQQ260925C00740000", "bid"] = 0.0
r2 = z.build_rows(one_side, 740.0, ts=TS, today=TODAY)
a2 = r2[(r2["strike"] == 740.0) & (r2["cp"] == "C")].iloc[0]
# มีแต่ฝั่ง ask = ราคาหลอก (0.00/2.20 → mid 1.10) ต้องถอยไปใช้ last แทน
check("มี quote ฝั่งเดียว → ถอยไปใช้ last ไม่ใช่ mid หลอก",
      abs(a2["mid"] - 2.1) < 1e-9, str(a2["mid"]))

print("\n=== ปัดทศนิยม ===")
noisy = fake_chain()
noisy["prev_day_close"] = 240.595001220703
r = z.build_rows(noisy, 740.0, ts=TS, today=TODAY)
check("ปัด prev_close เหลือ 4 ตำแหน่ง", r["prev_close"].iloc[0] == 240.595, str(r["prev_close"].iloc[0]))

print("\n=== chain ที่ใช้ไม่ได้ ต้องคืนตารางว่าง ไม่ใช่ crash ===")
check("chain ว่าง", z.build_rows(pd.DataFrame(), 740.0, today=TODAY).empty)
check("chain = None", z.build_rows(None, 740.0, today=TODAY).empty)
check("ไม่มี expiry ข้างหน้าเลย",
      z.build_rows(fake_chain(expiries=("260101",)), 740.0, today=TODAY).empty)
check("สัญลักษณ์เพี้ยนทั้งกระดาน",
      z.build_rows(pd.DataFrame({"option": ["ขยะ", "ขยะ2"]}), 740.0, today=TODAY).empty)


# ════════════════════════════════════════════════
# เขียน/อ่านไฟล์ — ใช้โฟลเดอร์ชั่วคราว ไม่แตะข้อมูลจริง
# ════════════════════════════════════════════════
REAL = z.ZDTE_DIR
TMP = tempfile.mkdtemp(prefix="zdte-test-")
z.ZDTE_DIR = TMP
try:
    print("\n=== เขียนแล้วอ่านกลับ ===")
    res = z.write_rows(rows)
    back = z.load("QQQ", "2026-09-25")
    check("เขียนได้ครบทุกแถว", res["written"] == len(rows), str(res))
    check("อ่านกลับได้จำนวนเท่าเดิม", len(back) == len(rows), f"{len(back)} vs {len(rows)}")
    check("ชื่อไฟล์ = expiry", os.path.basename(res["path"]) == "2026-09-25.csv",
          res["path"])
    check("ค่าที่อ่านกลับตรงกับที่เขียน",
          abs(float(back[(back["strike"] == 740.0) & (back["cp"] == "C")]["mid"].iloc[0]) - 2.1) < 1e-9)

    print("\n=== เขียนซ้ำต้องไม่เพิ่มแถว (idempotent) ===")
    r2 = z.write_rows(rows)
    check("รอบสองเขียน 0 แถว", r2["written"] == 0, str(r2))
    check("รอบสองข้ามครบทุกแถว", r2["skipped"] == len(rows), str(r2))
    check("ไฟล์ยังมีแถวเท่าเดิม", len(z.load("QQQ", "2026-09-25")) == len(rows))

    print("\n=== snapshot คนละเวลา = เพิ่มแถว ===")
    rows2 = z.build_rows(fake_chain(), 741.0, ts=TS + timedelta(minutes=15), today=TODAY)
    z.write_rows(rows2)
    both = z.load("QQQ", "2026-09-25")
    check("สอง snapshot รวมกันได้", len(both) == len(rows) * 2, f"{len(both)} แถว")
    check("มี 2 ช่วงเวลา", both["ts_utc"].nunique() == 2, str(both["ts_utc"].nunique()))

    print("\n=== session_frame: volume สะสม → ปริมาณต่อช่วง ===")
    # CBOE ส่ง volume เป็นยอดสะสม ถ้าไม่ทำ diff กราฟจะอ่านผิดว่า "ซื้อหนักตอนบ่าย"
    sf = z.session_frame("QQQ", "2026-09-25")
    atm = sf[(sf["strike"] == 740.0) & (sf["cp"] == "C")].sort_values("ts")
    check("snapshot แรก vol_delta = ยอดสะสมทั้งก้อน", atm["vol_delta"].iloc[0] == 1000,
          str(atm["vol_delta"].iloc[0]))
    check("snapshot ที่สอง vol_delta = 0 (ยอดสะสมไม่เปลี่ยน)",
          atm["vol_delta"].iloc[1] == 0, str(atm["vol_delta"].iloc[1]))
    check("มีคอลัมน์ที่ใช้วิเคราะห์ครบ",
          {"et", "vol_delta", "notional", "moneyness", "oi_delta"} <= set(sf.columns))

    print("\n=== day_summary ===")
    s = z.day_summary("QQQ", "2026-09-25")
    check("นับ snapshot ถูก", s["snapshots"] == 2, str(s["snapshots"]))
    check("ไม่ใช่วันว่าง", not s["empty"])
    check("spot สุดท้ายคือของ snapshot ล่าสุด", s["spot_last"] == 741.0, str(s["spot_last"]))
    check("มี top_notional", len(s["top_notional"]) > 0)
    check("วันที่ไม่มีข้อมูล → empty=True", z.day_summary("QQQ", "1999-01-04")["empty"])

    print("\n=== รวบเดือน แล้วต้องอ่านย้อนได้ครบ ===")
    before = len(z.load("QQQ", "2026-09-25"))
    days_before = z.available_days("QQQ")
    r = z.rollup_month("QQQ", "2026-09")
    check("รวบแล้วยืนยันได้", r["verified"], str(r))
    check("ลบไฟล์รายวันแล้ว", not os.path.exists(z.path_for("QQQ", "2026-09-25")))
    check("มีไฟล์คลัง", os.path.exists(z.month_path("QQQ", "2026-09")))
    check("อ่านย้อนจากคลังได้ครบทุกแถว", len(z.load("QQQ", "2026-09-25")) == before,
          f"{len(z.load('QQQ', '2026-09-25'))} vs {before}")
    check("available_days ยังเห็นวันเดิม", z.available_days("QQQ") == days_before,
          f"{z.available_days('QQQ')} vs {days_before}")
    check("สรุปจากคลังได้เหมือนเดิม", z.day_summary("QQQ", "2026-09-25")["snapshots"] == 2)
    check("คลังเล็กกว่าไฟล์ดิบ",
          os.path.getsize(z.month_path("QQQ", "2026-09")) < 33000)

    print("\n=== บันทึกต่อหลังรวบแล้ว: ต้องเห็นทั้งของใหม่และของในคลัง ===")
    # เคสนี้คือบั๊กที่เจอตอนเขียนเทสต์ — load() เดิมคืนแค่ไฟล์รายวันใหม่
    # ทำให้วันที่รวบไปแล้วดูเหมือนเหลือ snapshot เดียว ทั้งที่ข้อมูลเต็มยังอยู่ในคลัง
    rows3 = z.build_rows(fake_chain(), 742.0, ts=TS + timedelta(minutes=30), today=TODAY)
    z.write_rows(rows3)
    check("load รวมไฟล์รายวัน + คลัง (2 เก่า + 1 ใหม่ = 3 snapshot)",
          z.day_summary("QQQ", "2026-09-25")["snapshots"] == 3,
          str(z.day_summary("QQQ", "2026-09-25")["snapshots"]))
    check("ไม่มีแถวซ้ำหลังรวม",
          len(z.load("QQQ", "2026-09-25")) == len(rows) * 3,
          f"{len(z.load('QQQ', '2026-09-25'))} vs {len(rows) * 3}")
    check("เขียนซ้ำของที่อยู่ในคลังแล้ว ต้องไม่เพิ่มแถว",
          z.write_rows(rows)["written"] == 0, str(z.write_rows(rows)))
    r = z.rollup_month("QQQ", "2026-09")
    check("รวบอีกครั้งแล้วยังได้ 3 snapshot เท่าเดิม",
          z.day_summary("QQQ", "2026-09-25")["snapshots"] == 3,
          str(z.day_summary("QQQ", "2026-09-25")["snapshots"]))

    print("\n=== record() ต้องไม่โยน exception เมื่อดึง chain ไม่ได้ ===")
    def boom(sym): raise RuntimeError("CBOE ล่ม")
    out = z.record("QQQ", fetch=boom)
    check("คืน ok=False พร้อมเหตุผล", out["ok"] is False and "CBOE" in out["error"], str(out))
    out = z.record("QQQ", fetch=lambda s: (pd.DataFrame(), 0.0))
    check("chain ว่าง → ok=False ไม่ crash", out["ok"] is False, str(out))
    out = z.record("QQQ", fetch=lambda s: (fake_chain(), 740.0),
                   ts=TS + timedelta(minutes=45))
    check("fetch ปกติ → บันทึกได้", out["ok"] and out["written"] > 0, str(out))

    print("\n=== รายงาน HTML ===")
    import zero_dte_report as rep
    html = rep.build("QQQ", "2026-09-25")
    check("ได้ HTML ที่สมบูรณ์", html.startswith("<!doctype html>") and "</html>" in html)
    check("มีกราฟ plotly", "plotly" in html.lower())
    # ห้ามเช็คหัวข้อด้วยข้อความไทยตรง ๆ — plotly escape อักขระนอก ASCII เป็น \\uXXXX
    # ในก้อน JSON ที่ฝังมา เช่น "1 ·" กลายเป็น "1 \\u00b7" (เสียเวลาไล่มาแล้วรอบหนึ่ง
    # ตรงกับ HANDOFF §7: เทสต์ FAIL ให้สงสัยเทสต์ก่อนสงสัยโค้ด)
    check("วาดครบ 6 กราฟ", html.count("Plotly.newPlot") == 6,
          f"นับได้ {html.count('Plotly.newPlot')}")
    check("หัวข้อกราฟครบ 1–6",
          all((f"{i} ·" in html) or (f"{i} \\u00b7" in html) for i in range(1, 7)),
          "ขาดหัวข้อบางตัว")
    check("ฝัง plotly.js ครั้งเดียว ไม่ซ้ำ 6 รอบ",
          html.count("https://cdn.plot.ly") <= 2, str(html.count("https://cdn.plot.ly")))
    check("ไม่มี NaN โผล่ในตาราง", ">nan<" not in html.lower())
    empty = rep.build("QQQ", "1999-01-04")
    check("วันที่ไม่มีข้อมูล → หน้าบอกว่ายังไม่มีข้อมูล ไม่ crash",
          "ยังไม่มีข้อมูล" in empty)
finally:
    z.ZDTE_DIR = REAL
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\nสรุป: {'ผ่านหมด' if not FAIL else f'ไม่ผ่าน {len(FAIL)} เคส'}")
raise SystemExit(1 if FAIL else 0)
