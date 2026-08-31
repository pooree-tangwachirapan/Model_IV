"""
ตรวจคณิตศาสตร์ของแท็บ Long Premium — intraday.py · contracts.py · lp_store.py   [LP]

ใช้เคสที่คำนวณด้วยมือได้ ไม่ใช่แค่ "รันแล้วไม่ crash" (แนวเดียวกับชุดอื่นในโฟลเดอร์นี้)

[LP] ไฟล์นี้เป็นของแท็บใหม่ล้วน — ลบทิ้งได้โดยไม่กระทบชุดทดสอบเดิม
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง
import numpy as np
import pandas as pd
import intraday as it
import contracts as ct
import lp_store as st_

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

def bars(rows, day="2026-08-20"):
    """rows = [(hhmm, open, high, low, close, volume)]"""
    idx = pd.to_datetime([f"{day} {h[:2]}:{h[2:]}" for h, *_ in rows]).tz_localize("America/New_York")
    df = pd.DataFrame([{"Open": o, "High": h_, "Low": l, "Close": c, "Volume": v}
                       for _, o, h_, l, c, v in rows], index=idx)
    return it._add_session_cols(df)


# ═══════════════════════════════════════════════════════════════
print("=== 1. value_area — histogram ที่คำนวณมือได้ ===")
# tick=1.0 · แต่ละแท่งอยู่ในช่องเดียวสนิท (L,H อยู่ระหว่าง edge ไม่คร่อม)
#   ช่อง [100,101) V=10  ·  [101,102) V=50 ← POC  ·  [102,103) V=30  ·  [103,104) V=10
#   total=100 · target=70 · เริ่มที่ POC (50) แล้วขยายฝั่งที่มากกว่า: บน=30 > ล่าง=10
#   → acc = 50+30 = 80 ≥ 70 หยุด · โซนคือช่อง 1..2 → VAL=101.5 · POC=101.5 · VAH=102.5
b1 = bars([("0930", 100.5, 100.8, 100.2, 100.5, 10),
           ("0935", 101.5, 101.8, 101.2, 101.5, 50),
           ("0940", 102.5, 102.8, 102.2, 102.5, 30),
           ("0945", 103.5, 103.8, 103.2, 103.5, 10)])
va = it.value_area(b1, tick=1.0)
check("POC = 101.5 (ช่องที่ volume สูงสุด)", abs(va["poc"] - 101.5) < 1e-9, f"ได้ {va['poc']}")
check("VAH = 102.5 (ขยายขึ้นเพราะ 30 > 10)", abs(va["vah"] - 102.5) < 1e-9, f"ได้ {va['vah']}")
check("VAL = 101.5 (ไม่ขยายลง)", abs(va["val"] - 101.5) < 1e-9, f"ได้ {va['val']}")
check("total volume = 100", abs(va["total_volume"] - 100) < 1e-6, f"ได้ {va['total_volume']}")
check("VAL ≤ POC ≤ VAH เสมอ", va["val"] <= va["poc"] <= va["vah"])

# ครอบคลุม ≥70% จริงตามนิยาม
b2 = bars([("0930", 100.5, 100.8, 100.2, 100.5, 25),
           ("0935", 101.5, 101.8, 101.2, 101.5, 25),
           ("0940", 102.5, 102.8, 102.2, 102.5, 25),
           ("0945", 103.5, 103.8, 103.2, 103.5, 25)])
va2 = it.value_area(b2, tick=1.0)
n_buckets = round((va2["vah"] - va2["val"]) / 1.0) + 1
check("volume เท่ากันทุกช่อง → ต้องกิน 3 ช่อง (75% ≥ 70%)", n_buckets == 3, f"ได้ {n_buckets} ช่อง")

check("แท่งเดียวราคาไม่ขยับ → VAH=VAL=POC ไม่ crash",
      it.value_area(bars([("0930", 5, 5, 5, 5, 7)]))["poc"] == 5.0)
check("ไม่มีแท่ง → None", it.value_area(pd.DataFrame()) is None)


# ═══════════════════════════════════════════════════════════════
print("\n=== 2. session_vwap — typical price ถ่วง volume ===")
# แท่ง1 tp=(102+98+100)/3=100 V=100 → VWAP=100
# แท่ง2 tp=(105+103+104)/3=104 V=100 → VWAP=(100*100+104*100)/200=102
b3 = bars([("0930", 99, 102, 98, 100, 100), ("0935", 100, 105, 103, 104, 100)])
vw = it.session_vwap(b3)
check("VWAP หลังแท่งแรก = 100", abs(vw.iloc[0] - 100) < 1e-9, f"ได้ {vw.iloc[0]}")
check("VWAP หลังแท่งสอง = 102", abs(vw.iloc[1] - 102) < 1e-9, f"ได้ {vw.iloc[1]}")

# ถ่วงน้ำหนักจริง ไม่ใช่ค่าเฉลี่ยเปล่า: แท่งสอง volume 300 → (100*100+104*300)/400 = 103
b4 = bars([("0930", 99, 102, 98, 100, 100), ("0935", 100, 105, 103, 104, 300)])
check("volume มากกว่า → ถ่วงเข้าหาแท่งนั้น (=103)",
      abs(it.session_vwap(b4).iloc[1] - 103) < 1e-9, f"ได้ {it.session_vwap(b4).iloc[1]}")

# รีเซ็ตทุก session — วันที่สองต้องไม่เอา volume ของวันแรกมารวม
b5 = pd.concat([bars([("0930", 99, 102, 98, 100, 100)], day="2026-08-20"),
                bars([("0930", 200, 202, 198, 200, 100)], day="2026-08-21")])
check("VWAP รีเซ็ตข้ามวัน (วันที่สอง = 200 ไม่ใช่ 150)",
      abs(it.session_vwap(b5).iloc[1] - 200) < 1e-9, f"ได้ {it.session_vwap(b5).iloc[1]}")


# ═══════════════════════════════════════════════════════════════
print("\n=== 3. volume_spike — baseline ต้อง shift(1) (ห้าม lookahead) ===")
# 4 session · slot เดียวกัน · volume = 10, 20, 30, 400
#   rolling(3,min3).median() = [NaN, NaN, 20, 30]
#   .shift(1)                = [NaN, NaN, NaN, 20]   ← วันที่ 4 ใช้ median ของ 3 วันก่อนหน้า
#   spike วันที่ 4 = 400/20 = 20.0
#   **ถ้าลืม shift** จะได้ 400/30 = 13.3 (เพราะ 400 ดันค่า median ของตัวเองขึ้น) = ผิด
b6 = pd.concat([bars([("0930", 1, 1, 1, 1, v)], day=d) for d, v in
                [("2026-08-17", 10), ("2026-08-18", 20), ("2026-08-19", 30), ("2026-08-20", 400)]])
sp = it.volume_spike(b6, lookback=3)
check("spike วันสุดท้าย = 400/20 = 20.0 (baseline ไม่รวมตัวเอง)",
      abs(sp.iloc[3] - 20.0) < 1e-9, f"ได้ {sp.iloc[3]} — ถ้าได้ 13.3 แปลว่า shift(1) หาย")
check("วันแรก ๆ ที่ยังไม่มี baseline = NaN ไม่ใช่ 1.0",
      np.isnan(sp.iloc[0]) and np.isnan(sp.iloc[2]))

# ต้องเทียบ slot ต่อ slot ไม่ใช่ค่าเฉลี่ยรวมทั้งวัน
# slot 0930 volume ใหญ่ตลอด · slot 1200 เล็กตลอด → ทั้งคู่ต้องได้ spike ≈ 1 ไม่ใช่ 0930 สูงลิ่ว
b7 = pd.concat([bars([("0930", 1, 1, 1, 1, 1000), ("1200", 1, 1, 1, 1, 100)], day=d)
                for d in ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20"]])
sp7 = it.volume_spike(b7, lookback=3)
last = sp7.iloc[-2:]         # แท่ง 0930 และ 1200 ของวันสุดท้าย
check("baseline แยกตามเวลาของวัน → แท่งเปิดตลาดไม่เป็น spike ปลอม",
      abs(last.iloc[0] - 1.0) < 1e-9 and abs(last.iloc[1] - 1.0) < 1e-9,
      f"ได้ 0930={last.iloc[0]} · 1200={last.iloc[1]}")


# ═══════════════════════════════════════════════════════════════
print("\n=== 4. locate — ตำแหน่งเทียบ value area ===")
va_t = {"vah": 110.0, "poc": 105.0, "val": 100.0}
check("ราคาใน VA → inside · penetration = 0", it.locate(105, va_t)["zone"] == "inside"
      and it.locate(105, va_t)["penetration_pct"] == 0.0)
check("ราคาเหนือ VAH → above", it.locate(111, va_t)["zone"] == "above")
check("ราคาใต้ VAL → below", it.locate(99, va_t)["zone"] == "below")
check("penetration = (110→121) 9.09% ของราคา",
      abs(it.locate(121, va_t)["penetration_pct"] - (11 / 121 * 100)) < 1e-9)
check("ไม่มี VA → ทุกค่าเป็น None", it.locate(100, None)["zone"] is None)


# ═══════════════════════════════════════════════════════════════
print("\n=== 5. bs_price — เทียบค่าที่รู้คำตอบ + put-call parity ===")
# ตำราเล่มไหนก็ได้: S=K=100, T=1, iv=20%, r=5% → call = 10.4506
c = ct.bs_price(100, 100, 1.0, 0.20, 0.05, True)
check("ATM call 1y 20% 5% = 10.4506", abs(c - 10.4506) < 0.001, f"ได้ {c:.4f}")
p = ct.bs_price(100, 100, 1.0, 0.20, 0.05, False)
check("put = 5.5735", abs(p - 5.5735) < 0.001, f"ได้ {p:.4f}")
# C − P = S − K·e^(−rT)
check("put-call parity",
      abs((c - p) - (100 - 100 * np.exp(-0.05 * 1.0))) < 1e-6,
      f"C-P={c-p:.6f} vs {100 - 100*np.exp(-0.05):.6f}")
check("T=0 → เหลือแต่ intrinsic (call ITM 10)",
      abs(ct.bs_price(110, 100, 0.0, 0.20, 0.05, True) - 10.0) < 1e-9)
check("T=0 OTM → 0", abs(ct.bs_price(90, 100, 0.0, 0.20, 0.05, True)) < 1e-9)
check("ราคาไม่ต่ำกว่า intrinsic เสมอ",
      ct.bs_price(150, 100, 0.5, 0.20, 0.05, True) >= 50.0)
check("iv สูงขึ้น → call แพงขึ้น (vega บวก)",
      ct.bs_price(100, 100, 0.5, 0.30, 0.05, True) > ct.bs_price(100, 100, 0.5, 0.20, 0.05, True))
check("S ผิด/ติดลบ → None", ct.bs_price(-1, 100, 1, 0.2, 0.05, True) is None)


# ═══════════════════════════════════════════════════════════════
print("\n=== 6. contracts.pick — ตัวกรองและเหตุผลที่ตก ===")
def mkchain(rows):
    """rows = [(type, strike, days, bid, ask, iv, delta, theta, oi)]"""
    return pd.DataFrame([{"type": t, "strike": k, "days": d, "bid": b, "ask": a, "iv": iv,
                          "delta": dl, "theta": th, "gamma": 0.01, "vega": 0.5,
                          "open_interest": oi, "volume": 10, "expiry": "2026-10-16"}
                         for t, k, d, b, a, iv, dl, th, oi in rows])

plan_long = {"side": "LONG", "ideal_entry": 100.0, "invalidation": 97.0,
             "target": 110.0, "plan_r": 3.33}
ch = mkchain([
    ("call", 100, 30, 3.90, 4.10, 0.20,  0.52, -0.05, 5000),   # ดีทุกอย่าง
    ("call", 130, 30, 0.04, 0.06, 0.20,  0.02, -0.01, 5000),   # delta นอกแบนด์
    ("call", 100,  3, 1.00, 1.10, 0.20,  0.51, -0.30, 5000),   # DTE สั้น + theta สูง
    ("call", 101, 30, 3.00, 4.00, 0.20,  0.50, -0.05, 5000),   # spread 28%
    ("call", 102, 30, 3.40, 3.60, 0.20,  0.48, -0.05,    5),   # OI บาง
    ("put",  100, 30, 3.90, 4.10, 0.20, -0.48, -0.05, 5000),   # ผิดฝั่ง ต้องไม่โผล่
])
r = ct.pick(ch, 100.0, plan_long)
check("LONG → เลือกเฉพาะ call (put ไม่โผล่)", (r["type"] == "call").all())
ok = r[r["status"] == "OK"]
check("มีตัวผ่าน 1 ตัว คือ strike 100 DTE 30", len(ok) == 1 and ok.iloc[0]["strike"] == 100,
      f"ได้ {len(ok)} ตัว")

def why(strike, days):
    row = r[(r["strike"] == strike) & (r["dte"] == days)]
    return row.iloc[0]["reasons"] if len(row) else "(ไม่พบ)"

check("strike 130 ตกเพราะ Δ นอกแบนด์", "Δ นอกแบนด์" in why(130, 30), why(130, 30))
check("DTE 3 ตกเพราะ DTE สั้นไป", "DTE สั้นไป" in why(100, 3), why(100, 3))
check("DTE 3 ตกเพราะ theta สูงด้วย (เก็บครบทุกเหตุผล ไม่หยุดที่ข้อแรก)",
      "theta สูง" in why(100, 3), why(100, 3))
check("strike 101 ตกเพราะ spread กว้าง", "spread กว้าง" in why(101, 30), why(101, 30))
check("strike 102 ตกเพราะ OI บาง", "OI บาง" in why(102, 30), why(102, 30))
check("ตัวที่ตกยังอยู่ในตาราง ไม่ถูกซ่อน", len(r) >= 5, f"ได้ {len(r)} แถว")

# ตรวจ P&L ตรงกับ bs_price ที่คำนวณแยก
row = ok.iloc[0]
expect = (ct.bs_price(110.0, 100.0, (30 - 3) / 365, 0.20, ct.RISK_FREE, True) - 4.00) * 100
check("pnl_target ตรงกับ BS reprice ที่ (target, T−hold)",
      abs(row["pnl_target"] - expect) < 1e-6, f"ได้ {row['pnl_target']:.4f} คาด {expect:.4f}")
check("IV crush ทำกำไรน้อยลงเสมอ", row["pnl_target_ivdn"] < row["pnl_target"])
check("IV ขยายทำกำไรมากขึ้นเสมอ", row["pnl_target_ivup"] > row["pnl_target"])
check("pnl_invalid ติดลบ (ราคาไปผิดทาง)", row["pnl_invalid"] < 0)
check("breakeven ของ call = strike + premium", abs(row["breakeven"] - 104.0) < 1e-9)


# ═══════════════════════════════════════════════════════════════
print("\n=== 7. R ของ option ต่างจาก plan_r ของ underlying (สาระของโมดูล) ===")
check("r_option ต่ำกว่า plan_r — ธีต้ากินไป",
      row["r_option"] < plan_long["plan_r"],
      f"r_option={row['r_option']:.2f} vs plan_r={plan_long['plan_r']}")
s = ct.summarize(r, plan_long)
check("summarize รายงาน r_gap เป็นลบ", s["r_gap"] is not None and s["r_gap"] < 0,
      f"ได้ {s['r_gap']}")
check("summarize นับ n_ok ถูก", s["n_ok"] == 1, f"ได้ {s['n_ok']}")

# ── ถือนานขึ้น = จ่ายธีต้ามากขึ้น → R ต้องลดลงเสมอ ──
# นี่คือความสัมพันธ์ที่จริงเชิงกลไก (ตัวเดียวกัน strike เดียวกัน ต่างแค่เวลาที่ปล่อยให้ผ่านไป)
r_hold1 = ct.pick(ch, 100.0, plan_long, hold_days=1)
r_hold5 = ct.pick(ch, 100.0, plan_long, hold_days=5)
h1 = r_hold1[(r_hold1["strike"] == 100) & (r_hold1["dte"] == 30)].iloc[0]
h5 = r_hold5[(r_hold5["strike"] == 100) & (r_hold5["dte"] == 30)].iloc[0]
check("ถือ 5 วัน → r_option ต่ำกว่าถือ 1 วัน (ธีต้าเป็นต้นทุนล้วน)",
      h5["r_option"] < h1["r_option"], f"5d={h5['r_option']:.3f} vs 1d={h1['r_option']:.3f}")
check("ถือนานขึ้น → กำไรที่เป้าน้อยลง", h5["pnl_target"] < h1["pnl_target"])

# ── DTE ยาวกว่า → return_pct ต่ำกว่า (จ่ายทุนมากกว่าเพื่อกินระยะทางเท่ากัน) ──
# ใช้ราคาที่สอดคล้องกับ BS จริงที่ iv=20% ไม่ใช่ตัวเลขที่ตั้งมั่ว
# ไม่ยืนยันว่า DTE ยาวให้ R ดีกว่าเสมอ — **มันไม่จริงสากล** ดูหมายเหตุใน contracts.pick
m14 = ct.bs_price(100, 100, 14 / 365, 0.20, ct.RISK_FREE, True)
m60 = ct.bs_price(100, 100, 60 / 365, 0.20, ct.RISK_FREE, True)
ch2 = mkchain([("call", 100, 14, m14 - 0.05, m14 + 0.05, 0.20, 0.52, -0.09, 5000),
               ("call", 100, 60, m60 - 0.05, m60 + 0.05, 0.20, 0.55, -0.04, 5000)])
r2 = ct.pick(ch2, 100.0, plan_long, max_dte=90)
short_, long_ = r2[r2["dte"] == 14].iloc[0], r2[r2["dte"] == 60].iloc[0]
check("DTE ยาวกว่า → ทุนต่อสัญญาแพงกว่า", long_["mid"] > short_["mid"],
      f"60d={long_['mid']:.2f} vs 14d={short_['mid']:.2f}")
check("DTE ยาวกว่า → return_pct ต่ำกว่า (leverage น้อยกว่า)",
      long_["return_pct"] < short_["return_pct"],
      f"60d={long_['return_pct']:.1f}% vs 14d={short_['return_pct']:.1f}%")
check("DTE ยาวกว่า → theta ต่อวันคิดเป็น % ของพรีเมียมน้อยกว่า",
      long_["theta_day_pct"] < short_["theta_day_pct"],
      f"60d={long_['theta_day_pct']:.2f}% vs 14d={short_['theta_day_pct']:.2f}%")

# SHORT ต้องเลือก put
plan_short = {"side": "SHORT", "ideal_entry": 100.0, "invalidation": 103.0,
              "target": 90.0, "plan_r": 3.33}
r3 = ct.pick(ch, 100.0, plan_short)
check("SHORT → เลือกเฉพาะ put", len(r3) > 0 and (r3["type"] == "put").all())
check("แผนว่าง/ไม่มี target → คืนตารางว่าง ไม่ throw",
      ct.pick(ch, 100.0, {"side": "LONG"}).empty)
check("chain ว่าง → คืนตารางว่าง ไม่ throw", ct.pick(pd.DataFrame(), 100.0, plan_long).empty)


# ═══════════════════════════════════════════════════════════════
print("\n=== 8. lp_store — JSONL ทนไฟล์เสีย ===")
tmp = os.path.join(tempfile.mkdtemp(), "lp", "context_log.jsonl")
st_.append({"session_date": "2026-08-20", "spot_cboe": 700.0, "note": "ทดสอบภาษาไทย"}, tmp)
st_.append({"session_date": "2026-08-21", "spot_cboe": 705.0}, tmp)
recs = st_.read_all(tmp)
check("เขียน 2 แล้วอ่านได้ 2", len(recs) == 2, f"ได้ {len(recs)}")
check("ค่าที่อ่านกลับมาตรงกับที่เขียน", recs[0]["spot_cboe"] == 700.0)
check("ภาษาไทยไม่เพี้ยน", recs[0]["note"] == "ทดสอบภาษาไทย", f"ได้ {recs[0].get('note')}")
check("มี logged_at ให้อัตโนมัติ", "logged_at" in recs[0])

with open(tmp, "a", encoding="utf-8") as f:
    f.write('{"session_date": "2026-08-22", "spot_cbo\n')      # บรรทัดขาดกลางคัน
st_.append({"session_date": "2026-08-23", "spot_cboe": 710.0}, tmp)
recs2 = st_.read_all(tmp)
check("บรรทัดเสียถูกข้าม ไม่ทำให้ทั้งไฟล์อ่านไม่ได้", len(recs2) == 3, f"ได้ {len(recs2)}")
check("record หลังบรรทัดเสียยังอ่านได้", recs2[-1]["spot_cboe"] == 710.0)

s2 = st_.stats(tmp)
check("stats นับ session ไม่ซ้ำ", s2["n_sessions"] == 3, f"ได้ {s2['n_sessions']}")
check("stats บอกวันแรก/วันสุดท้าย",
      s2["first_session"] == "2026-08-20" and s2["last_session"] == "2026-08-23")
check("ไฟล์ไม่มี → คืน list ว่าง ไม่ throw", st_.read_all(tmp + ".nope") == [])


# ═══════════════════════════════════════════════════════════════
print("\n=== 9. lp_store.flatten — รวม snapshot + context + verdict ===")
snap = {"symbol": "QQQ", "spot": 700.0, "error": None, "near_expiry": "2026-09-05", "near_dte": 5,
        "levels": {"net": -1.5e9, "net_profile": 2.0e8, "net_agree": False, "flip": 705.0,
                   "call_wall": 710.0, "put_wall": 690.0,
                   "call_wall_oi": 715.0, "put_wall_oi": 690.0},
        "expected_move": {"em": 7.0, "em_pct": 1.0, "dte": 5},
        "atm_iv": 23.5, "rv": {"hv20": 19.0, "hv10": 21.0}, "macro": {"vix": 17.0, "vvix": 95.0},
        "skew": {"rr25": 2.0}, "term": {"slope_pct": 1.2, "state": "contango"},
        "pin": {"score": 42.0}, "zero_dte": 30.0, "max_pain": 700.0}
ctx = {"vwap": 699.0, "va_zone": "above", "session_date": "2026-08-20", "error": None,
       "above_vwap": True, "above_ema200": False}
fade = {"verdict": "ARMED", "plan": {"side": "LONG", "ideal_entry": 690.0, "target": 697.0,
                                     "invalidation": 688.25, "plan_r": 4.0}}
rec = st_.flatten(snap, ctx, fade=fade, breakout_v=None)
check("vrp = atm_iv − hv20 = 4.5", abs(rec["vrp"] - 4.5) < 1e-9, f"ได้ {rec.get('vrp')}")
check("เก็บ net_gex_agree=False ไว้ (เครื่องหมายเชื่อไม่ได้)", rec["net_gex_agree"] is False)
check("เก็บ wall ทั้งสองนิยาม", rec["call_wall"] == 710.0 and rec["call_wall_oi"] == 715.0)
check("เก็บ verdict + plan ของ fade", rec["fade_verdict"] == "ARMED" and rec["fade_plan_r"] == 4.0)
check("ไม่มี breakout → ไม่มี key ของมัน (ไม่เดาค่า)", "breakout_verdict" not in rec)
check("field ของ context ถูกรวมเข้ามา", rec["va_zone"] == "above" and rec["above_vwap"] is True)
check("record แบน ไม่ซ้อน (โหลดเข้า pandas ได้ตรง ๆ)",
      all(not isinstance(v, dict) for v in rec.values()))
check("snapshot error → เก็บเป็น snap_error ไม่ทิ้งเงียบ",
      st_.flatten({"error": "ดึง chain ไม่ได้"}, None).get("snap_error") == "ดึง chain ไม่ได้")
check("flatten รับ None ได้ทั้งคู่ ไม่ throw", isinstance(st_.flatten(None, None), dict))
check("json.dumps ได้จริง (ค่าที่ flatten คืนต้อง serialize ได้)",
      isinstance(json.dumps(rec, ensure_ascii=False, default=str), str))


print("\n" + "="*70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL))+' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
