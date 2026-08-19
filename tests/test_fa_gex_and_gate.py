"""
ตรวจ fa_gex.py (839 บรรทัด — ใหญ่สุด อยู่ต้นน้ำของทุกอย่าง) + gate.py (363)
ใช้ค่าที่คำนวณด้วยมือได้ — บั๊กแบบ IV scaling เคยรอดมาได้เพราะไม่มีเทสต์แบบนี้
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง
import fa_gex as fg
import gate
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

TODAY = datetime.now().date()
def dstr(n):  # วันที่ +n วันจากวันนี้
    return (TODAY + timedelta(days=n)).isoformat()

# ══════════════════════════════ fa_gex ══════════════════════════════
print("=== A1. compute_gex — สูตร Γ × OI × 100 × S² × 0.01 ===")
# Γ=0.01 OI=100 S=100 → 0.01*100*100*10000*0.01 = 10,000
df = pd.DataFrame([{"type":"call","strike":100.0,"open_interest":100,"gamma":0.01,
                    "iv":0.2,"days":7,"expiry":dstr(7)}])
ps = fg.compute_gex(df, 100.0)
check("call GEX = +10,000", abs(ps.call_gex.iloc[0] - 10_000) < 1, f"ได้ {ps.call_gex.iloc[0]}")
df2 = pd.DataFrame([{"type":"put","strike":100.0,"open_interest":100,"gamma":0.01,
                     "iv":0.2,"days":7,"expiry":dstr(7)}])
ps2 = fg.compute_gex(df2, 100.0)
check("put GEX = -10,000 (dealer convention)", abs(ps2.put_gex.iloc[0] + 10_000) < 1,
      f"ได้ {ps2.put_gex.iloc[0]}")
check("net = call + put", abs(ps2.net_gex.iloc[0] + 10_000) < 1, f"ได้ {ps2.net_gex.iloc[0]}")
# S² จริง ๆ: S=200 ต้องได้ 4 เท่า
ps3 = fg.compute_gex(df, 200.0)
check("S สองเท่า -> GEX สี่เท่า (S²)", abs(ps3.call_gex.iloc[0] - 40_000) < 1,
      f"ได้ {ps3.call_gex.iloc[0]}")
check("df ว่าง -> DataFrame ว่าง", fg.compute_gex(pd.DataFrame(), 100.0).empty)

print("\n=== A2. parse_cboe_for_gex — IV ต้องใช้ median ไม่ใช่ max ===")
# CBOE ส่งทศนิยม · มี deep-OTM ตัวเดียว iv=8.5 (850%) ซึ่งถูกต้อง
# ถ้าใช้ max เป็นเกณฑ์จะหาร 100 ทั้งกระดาน → ATM 0.22% แทน 22%
rows = [{"option":f"QQQ{TODAY:%y%m%d}C{int(k*1000):08d}","expiration":dstr(7),
         "option_type":"C","strike":k,"open_interest":100,"gamma":0.01,
         "iv":iv,"delta":0.5,"vega":0.1}
        for k,iv in [(100,0.22),(101,0.21),(102,0.23),(140,8.5)]]
out = fg.parse_cboe_for_gex(pd.DataFrame(rows), 100.0)
atm = out[out.strike == 100]
check("ATM IV คง 0.22 (ไม่โดนหาร 100)",
      len(atm) and abs(atm.iv.iloc[0] - 0.22) < 0.001,
      f"ได้ {atm.iv.iloc[0] if len(atm) else 'ไม่มีแถว'}")
# ถ้า CBOE ส่งเป็น % จริง ๆ (median > 5) ต้องหาร 100
rows_pct = [dict(r, iv=r["iv"]*100) for r in rows]
out_pct = fg.parse_cboe_for_gex(pd.DataFrame(rows_pct), 100.0)
atm_p = out_pct[out_pct.strike == 100]
check("ถ้าส่งเป็น % (median>5) -> หาร 100",
      len(atm_p) and abs(atm_p.iv.iloc[0] - 0.22) < 0.001,
      f"ได้ {atm_p.iv.iloc[0] if len(atm_p) else 'ไม่มีแถว'}")

print("\n=== A3. parse_cboe_for_gex — วันหมดอายุ ===")
rows_d = [{"option":"X","expiration":dstr(n),"option_type":"C","strike":100.0,
           "open_interest":100,"gamma":0.01,"iv":0.2,"delta":0.5,"vega":0.1}
          for n in (-1, 0, 1, 7)]
od = fg.parse_cboe_for_gex(pd.DataFrame(rows_d), 100.0)
days = sorted(od.days.tolist())
check("expiry เมื่อวาน (-1) ถูกตัดออก", -1 not in days, f"ได้ days={days}")
check("0DTE (วันนี้) ยังอยู่ และเป็น 0", 0 in days, f"ได้ days={days}")
check("1DTE เป็น 1 ไม่ใช่ 0", 1 in days, f"ได้ days={days}")

print("\n=== A4. gamma_profile — BS gamma ที่ ATM ===")
dfp = pd.DataFrame([{"type":"call","strike":100.0,"open_interest":1,"gamma":0.01,
                     "iv":0.2,"days":365,"expiry":dstr(365)}])
ladder, prof = fg.gamma_profile(dfp, 100.0, span=0.0001, n=3)
# gamma ATM: d1 = (0 + (0.05+0.02))/0.2 = 0.35 · N'(0.35)/(100*0.2*1)
d1 = (0.05 + 0.5*0.2**2) / 0.2
want_gamma = norm.pdf(d1) / (100 * 0.2 * 1.0)
want_gex = want_gamma * 1 * 100 * 100**2 * 0.01
mid = prof[len(prof)//2]
check("gamma profile ตรงสูตร BS", abs(mid - want_gex) / abs(want_gex) < 0.02,
      f"ได้ {mid:.2f} ควร {want_gex:.2f}")
check("ladder ยาวตามที่ขอ", len(ladder) == 3, f"ได้ {len(ladder)}")
check("iv หาย -> (None, None)",
      fg.gamma_profile(pd.DataFrame([{"type":"call","strike":100.0,"open_interest":1,
                                      "gamma":.01,"iv":np.nan,"days":7}]), 100.0)[0] is None)

print("\n=== A5. _zero_cross — interpolate จุดตัดศูนย์ ===")
x = np.array([1.0, 2.0, 3.0]); y = np.array([-1.0, 1.0, 2.0])
# ตัดระหว่าง 1 กับ 2: 1 + (0-(-1))*(2-1)/(1-(-1)) = 1.5
check("จุดตัด = 1.5", abs(fg._zero_cross(x, y, 1.5) - 1.5) < 1e-9, f"ได้ {fg._zero_cross(x,y,1.5)}")
check("ไม่มีจุดตัด -> None", fg._zero_cross(x, np.array([1.,2.,3.]), 2.0) is None)
# มีหลายจุดตัด ต้องเลือกจุดที่ใกล้ near สุด
x2 = np.array([0.,1.,2.,3.,4.]); y2 = np.array([-1.,1.,-1.,1.,2.])
near_hit = fg._zero_cross(x2, y2, 3.0)
check("หลายจุดตัด -> เลือกใกล้ near สุด", abs(near_hit - 2.5) < 1e-9, f"ได้ {near_hit}")

print("\n=== A6. find_levels — walls สองนิยาม ===")
d6 = pd.DataFrame([
    # call GEX สูงสุดที่ 110 (gamma สูง) แต่ call OI สูงสุดที่ 120
    {"type":"call","strike":110.0,"open_interest":100,"gamma":0.05,"iv":0.2,"days":7,"expiry":dstr(7)},
    {"type":"call","strike":120.0,"open_interest":500,"gamma":0.001,"iv":0.2,"days":7,"expiry":dstr(7)},
    {"type":"put","strike":90.0,"open_interest":100,"gamma":0.05,"iv":0.2,"days":7,"expiry":dstr(7)},
    {"type":"put","strike":80.0,"open_interest":500,"gamma":0.001,"iv":0.2,"days":7,"expiry":dstr(7)},
])
ps6 = fg.compute_gex(d6, 100.0)
lv = fg.find_levels(ps6, 100.0, df_contracts=d6)
check("call wall (GEX) = 110", lv["call_wall"] == 110.0, f"ได้ {lv['call_wall']}")
check("call wall (OI) = 120", lv["call_wall_oi"] == 120.0, f"ได้ {lv['call_wall_oi']}")
check("put wall (GEX) = 90", lv["put_wall"] == 90.0, f"ได้ {lv['put_wall']}")
check("put wall (OI) = 80", lv["put_wall_oi"] == 80.0, f"ได้ {lv['put_wall_oi']}")
check("สองนิยามต่างกันจริง (gamma ดึงเข้า ATM)",
      lv["call_wall"] != lv["call_wall_oi"] and lv["put_wall"] != lv["put_wall_oi"])
check("net = ผลรวม", abs(lv["net"] - ps6.net_gex.sum()) < 1, f"ได้ {lv['net']}")
check("ps ว่าง -> ทุกค่า None", fg.find_levels(pd.DataFrame(), 100.0)["call_wall"] is None)

print("\n=== A7. fmt_usd ===")
# 4500 -> "+4K" ไม่ใช่ "+5K" เพราะ Python ปัดครึ่งเข้าหาเลขคู่ (banker's rounding)
# ปล่อยไว้ตามนี้: เป็นการแสดงผลแบบย่ออยู่แล้ว ไม่มีผลกับการตัดสินใจ
# แต่ต้อง lock ไว้ด้วยเทสต์ จะได้ไม่มีใครไปแก้แล้วงงทีหลัง
for v, want in [(1.5e9,"+1.50B"), (-2.3e6,"-2.3M"), (4500,"+4K"), (5500,"+6K"),
                (12,"+12"), (None,"N/A")]:
    got = fg.fmt_usd(v)
    check(f"fmt_usd({v}) = {want}", got == want, f"ได้ {got}")

print("\n=== A8. parse_fa_gex — schema จริงของ FlashAlpha ===")
body = {"symbol":"NVDA","underlying_price":192.5,"as_of":"2026-07-30T09:37:03Z",
        "gamma_flip":196.15,"net_gex":3.0e7,"net_gex_label":"positive",
        "strikes":[{"strike":190.0,"call_gex":1.3e7,"put_gex":-3.4e7,"net_gex":-2.1e7},
                   {"strike":200.0,"call_gex":4.7e7,"put_gex":-1.0e7,"net_gex":3.7e7}]}
dfa, meta = fg.parse_fa_gex(body)
check("อ่าน spot", meta["spot"] == 192.5, f"ได้ {meta['spot']}")
check("อ่าน flip", abs(meta["flip"] - 196.15) < 0.01, f"ได้ {meta['flip']}")
check("อ่าน label", meta["label"] == "positive", f"ได้ {meta['label']}")
check("อ่าน strikes ครบ", len(dfa) == 2, f"ได้ {len(dfa)}")
check("put_gex ติดลบอยู่แล้ว ไม่โดนพลิกซ้ำ", (dfa.put_gex <= 0).all(), f"ได้ {dfa.put_gex.tolist()}")
check("derive call wall = 200", meta["call_wall"] == 200.0, f"ได้ {meta['call_wall']}")
check("derive put wall = 190", meta["put_wall"] == 190.0, f"ได้ {meta['put_wall']}")
check("ทำเครื่องหมายว่า derive เอง", meta["wall_source"] == "derived", f"ได้ {meta['wall_source']}")
# กรณี vendor ส่ง put_gex เป็นบวก -> ต้องพลิกเป็น convention เรา
body_pos = {**body, "strikes":[{"strike":190.0,"call_gex":1.0e7,"put_gex":3.0e7,"net_gex":4.0e7}]}
dfb, _ = fg.parse_fa_gex(body_pos)
check("put_gex บวกทั้งหมด -> พลิกเป็นลบ", (dfb.put_gex < 0).all(), f"ได้ {dfb.put_gex.tolist()}")
check("body ว่าง -> ไม่ crash", fg.parse_fa_gex(None)[0].empty)
check("body ไม่มี strikes -> ไม่ crash", fg.parse_fa_gex({"symbol":"X"})[0].empty)

print("\n=== A9. fa_preflight — กัน quota ===")
today = datetime.now().strftime("%Y-%m-%d")
check("QQQ (ETF) -> บล็อก", len(fg.fa_preflight("QQQ","")) == 1)
check("NVDA + วันนี้ (0DTE) -> บล็อก", len(fg.fa_preflight("NVDA",today)) == 1)
check("QQQ + วันนี้ -> บล็อก 2 เหตุผล", len(fg.fa_preflight("QQQ",today)) == 2)
check("NVDA + วันอื่น -> ผ่าน", fg.fa_preflight("NVDA","2026-12-19") == [])
check("NVDA + เว้นว่าง -> ผ่าน", fg.fa_preflight("NVDA","") == [])

print("\n=== A10. fa_error_message — 403 มีหลายสาเหตุ ===")
def emsg(code, txt):
    return fg.fa_error_message({"status":code,"body":{"message":txt},"ratelimit":{}}) or ""
check("200 -> None", fg.fa_error_message({"status":200,"body":{},"ratelimit":{}}) is None)
check("403 0DTE -> ชี้เรื่อง expiration", "expiration เป็นวันนี้" in emsg(403,"0DTE data requires Growth"))
check("403 ETF -> ชี้เรื่อง ETF", "ETF/Index ต้อง Basic" in emsg(403,"ETF data requires Basic plan"))
check("401 -> ชี้เรื่อง key", "API key" in emsg(401,"bad credentials"))
check("429 -> ชี้เรื่อง quota", "quota" in emsg(429,"rate limited"))
check("ข้อความ server อยู่ในผลเสมอ", "SOMETHING_UNIQUE" in emsg(403,"SOMETHING_UNIQUE"))

# ══════════════════════════════ gate ══════════════════════════════
print("\n=== B1. zone_geometry ===")
z = gate.zone_geometry(100.0, 90.0, 110.0)
check("pct กลางโซน = 0.5", abs(z["pct"] - 0.5) < 1e-9, f"ได้ {z['pct']}")
check("span = 20", abs(z["span"] - 20) < 1e-9, f"ได้ {z['span']}")
check("inside = True", z["inside"])
check("กลางโซนไม่ใช่ขอบ", not z["edge_near"])
check("ถึง put 10%", abs(z["d_put_pct"] - 10.0) < 1e-9, f"ได้ {z['d_put_pct']}")
z2 = gate.zone_geometry(92.0, 90.0, 110.0)   # pct = 0.1 < EDGE_FRAC 0.15
check("ใกล้ put wall -> edge_near", z2["edge_near"] and z2["side"] == "put")
z3 = gate.zone_geometry(120.0, 90.0, 110.0)  # นอกโซน
check("นอกโซน -> inside=False", not z3["inside"], f"pct={z3['pct']}")
check("wall กลับหัว -> None", gate.zone_geometry(100.0, 110.0, 90.0) is None)
check("wall เท่ากัน -> None", gate.zone_geometry(100.0, 100.0, 100.0) is None)
check("spot=None -> None", gate.zone_geometry(None, 90.0, 110.0) is None)
z4 = gate.zone_geometry(100.0, 90.0, 110.0, flip=100.2)
check("flip ใกล้ spot -> flip_near", z4["flip_near"], f"flip_pct={z4['flip_pct']}")

print("\n=== B2. zone_problem — แยกข้อมูลพัง vs สภาพตลาด ===")
_, is_data = gate.zone_problem(100.0, None, None)
check("กำแพงหายทั้งคู่ -> ข้อมูลพัง", is_data)
_, is_data2 = gate.zone_problem(100.0, 110.0, 90.0)
check("กำแพงกลับหัว -> ไม่ใช่ข้อมูลพัง", not is_data2)
_, is_data3 = gate.zone_problem(None, 90.0, 110.0)
check("ไม่มี spot -> ข้อมูลพัง", is_data3)

print("\n=== B3. build_plan — invalidate/target/R ===")
# ทุกอย่างยึด "กำแพง" เป็นจุดเข้า ไม่ใช่ราคาปัจจุบัน — ตัวเลขจึงไม่แกว่งทุกนาที
# wall 90 · invalid = 90 − 0.25×4 = 89 · target = 90 + 0.5×4 = 92
# risk = |90−89| = 1 · reward = |92−90| = 2 · R = 2.0
zz = gate.zone_geometry(92.0, 90.0, 110.0)      # ชิด put wall
p = gate.build_plan(92.0, zz, 90.0, 110.0, em=4.0, max_pain=None)
check("LONG จากขอบ put", p["side"] == "LONG", f"ได้ {p['side']}")
check("ideal_entry = กำแพง 90 ไม่ใช่ spot 92", p["ideal_entry"] == 90.0, f"ได้ {p['ideal_entry']}")
check("invalidate = 89", abs(p["invalidation"] - 89.0) < 1e-9, f"ได้ {p['invalidation']}")
check("target = 92 (นับจากกำแพง)", abs(p["target"] - 92.0) < 1e-9, f"ได้ {p['target']}")
check("R = 2.0 (นับจากกำแพง)", abs(p["plan_r"] - 2.0) < 1e-6, f"ได้ {p['plan_r']}")
check("ยังไม่ถึงจุดเข้า -> dist เป็นบวก", p["dist_to_entry_pts"] > 0, f"ได้ {p['dist_to_entry_pts']}")
check("ยังไม่ blown", not p["blown"])
# ทะลุ invalidate ไปแล้ว
p_blown = gate.build_plan(88.0, gate.zone_geometry(88.0, 90.0, 110.0) or zz,
                          90.0, 110.0, em=4.0)
check("spot ต่ำกว่า invalidate -> blown", p_blown["blown"], f"spot 88 invalid {p_blown['invalidation']}")
# Max Pain นอกกำแพงฝั่งตรงข้าม ต้องถูกตัดที่กำแพง
p2 = gate.build_plan(92.0, zz, 90.0, 110.0, em=4.0, max_pain=130.0)
check("Max Pain นอกโซน -> ตัดที่ call wall 110", abs(p2["target"] - 110.0) < 1e-9,
      f"ได้ {p2['target']}")
check("บอกว่าถูกตัด", "ตัดที่กำแพง" in p2["target_src"], f"ได้ {p2['target_src']}")
check("EM = 0 -> None", gate.build_plan(92.0, zz, 90.0, 110.0, em=0) is None)
check("ไม่มีโซน -> None", gate.build_plan(92.0, None, 90.0, 110.0, em=4.0) is None)
# ตั้งใจให้คืนแผน "เสมอ" แม้ราคายังไม่ถึงขอบ — เพื่อให้ย้อนวัดได้ว่าถ้าเข้าตามแผนจะชนะกี่ครั้ง
# ความ "ชิดขอบ" ไปอยู่ใน flag at_edge แทน ไม่ใช่เงื่อนไขว่าจะมีแผนหรือไม่
p_mid = gate.build_plan(100.0, gate.zone_geometry(100.0,90.0,110.0), 90.0, 110.0, em=4.0)
check("กลางโซน -> ยังมีแผน (ไว้ย้อนวัด)", p_mid is not None)
check("กลางโซน -> at_edge = False", p_mid and not p_mid["at_edge"], f"ได้ {p_mid['at_edge']}")
check("ชิดขอบ -> at_edge = True", p["at_edge"], f"ได้ {p['at_edge']}")

print("\n=== B4. contracts — คำนวณไซส์จากพรีเมียม ===")
c = gate.contracts(account=5000, risk_pct=1.5, entry_premium=2.0, invalid_premium=1.0)
# per = |2-1|*100 = 100 · budget = 75 · qty = 75//100 = 0
check("งบ 75 ต่อสัญญา 100 -> 0 สัญญา", c["qty"] == 0, f"ได้ {c['qty']}")
c2 = gate.contracts(account=50000, risk_pct=1.5, entry_premium=2.0, invalid_premium=1.0)
# budget 750 / per 100 = 7 -> โดน cap 5
check("เกิน cap -> ตัดที่ 5", c2["qty"] == 5 and c2["capped"], f"ได้ {c2['qty']} capped={c2['capped']}")
check("total = qty × per", abs(c2["total"] - 500) < 1e-9, f"ได้ {c2['total']}")
check("premium เท่ากัน -> qty 0 ไม่หารศูนย์",
      gate.contracts(5000, 1.5, 2.0, 2.0)["qty"] == 0)
check("account = None -> ไม่ crash", gate.contracts(None, 1.5, 2.0, 1.0)["qty"] == 0)

print("\n" + "="*70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL))+' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
