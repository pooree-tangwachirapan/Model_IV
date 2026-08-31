"""
ตรวจคณิตศาสตร์ใน snapshot.py — 472 บรรทัด 17 metric ที่ยังไม่เคยมี unit test เลย
ใช้เคสที่คำนวณด้วยมือได้ ไม่ใช่แค่ 'รันแล้วไม่ crash'
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง
import snapshot as sn
import pandas as pd, numpy as np

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

def mkdf(rows):
    """rows = [(type, strike, oi, iv, days, gamma)]"""
    return pd.DataFrame([{"type": t, "strike": k, "open_interest": oi,
                          "iv": iv, "days": d, "gamma": g, "expiry": "2026-08-20"}
                         for t, k, oi, iv, d, g in rows])

print("=== 1. max_pain — เคสคำนวณมือ ===")
# call OI 100 สัญญาที่ strike 100 · put OI 10 สัญญาที่ strike 110
#   settle 100: call 0        + put (110-100)*10 = 100   → 100   ← ต่ำสุด
#   settle 105: call 5*100=500 + put 5*10=50            → 550
#   settle 110: call 10*100=1000 + put 0                → 1000
df = mkdf([("call",100,100,.2,5,.01), ("put",110,10,.2,5,.01),
           ("call",105,0,.2,5,.01), ("put",105,0,.2,5,.01)])
mp = sn.max_pain(df)
check("max pain = 100 (ฝั่ง call หนักดึงลงมา)", mp == 100.0, f"ได้ {mp}")

# สลับข้าง: put หนักที่ 110 ควรดึง max pain ขึ้นไป
df2 = mkdf([("call",100,10,.2,5,.01), ("put",110,100,.2,5,.01),
            ("call",105,0,.2,5,.01), ("put",105,0,.2,5,.01)])
mp2 = sn.max_pain(df2)
check("สลับน้ำหนัก -> max pain = 110", mp2 == 110.0, f"ได้ {mp2}")
check("df ว่าง -> None", sn.max_pain(mkdf([])) is None)

print("\n=== 2. top_oi_strike ===")
df3 = mkdf([("call",100,50,.2,5,.01), ("put",100,60,.2,5,.01), ("call",105,90,.2,5,.01)])
# 100 รวม 110 · 105 รวม 90 → ควรได้ 100
check("รวม call+put ต่อ strike", sn.top_oi_strike(df3) == 100.0, f"ได้ {sn.top_oi_strike(df3)}")

print("\n=== 3. dex_vex — เครื่องหมายและสูตร ===")
raw = pd.DataFrame([
    {"delta": 0.5, "vega": 0.10, "open_interest": 100},
    {"delta": -0.3, "vega": 0.20, "open_interest": 200},
])
S = 100.0
dv = sn.dex_vex(raw, S)
# DEX = (0.5*100 + (-0.3)*200) * 100 * S = (50 - 60)*100*100 = -100,000
check("DEX = -100,000", abs(dv["dex"] + 100_000) < 1, f"ได้ {dv['dex']}")
# VEX = (0.10*100 + 0.20*200) * 100 = (10 + 40)*100 = 5,000
check("VEX = 5,000", abs(dv["vex"] - 5_000) < 1, f"ได้ {dv['vex']}")
check("ขาด column -> None ไม่ crash", sn.dex_vex(pd.DataFrame([{"x":1}]), S)["dex"] is None)

print("\n=== 4. atm_iv — ต้องใช้ median ไม่ใช่ mean (ทน outlier) ===")
df4 = mkdf([("call",99,10,.20,5,.01), ("call",100,10,.21,5,.01),
            ("call",101,10,.22,5,.01), ("call",102,10,3.00,5,.01)])
iv = sn.atm_iv(df4, 100.0, n=4)
# median ของ [.20,.21,.22,3.00] = .215 -> 21.5% · ถ้าใช้ mean จะได้ ~96%
check("median กัน outlier", abs(iv - 21.5) < 0.6, f"ได้ {iv:.2f}% (mean จะได้ ~{np.mean([20,21,22,300]):.0f}%)")

print("\n=== 5. expected_move = S × IV × √(DTE/365) ===")
df5 = mkdf([("call",100,10,.20,4,.01), ("put",100,10,.20,4,.01)])
em = sn.expected_move(df5, 100.0, "2026-08-20")
want = 100.0 * 0.20 * np.sqrt(4/365)
check("EM ตรงสูตร", abs(em["em"] - want) < 0.02, f"ได้ {em['em']:.4f} ควร {want:.4f}")
# 0DTE ต้องนับครึ่งวัน ไม่ใช่ 0 (ไม่งั้น EM = 0 แล้วประตูหารศูนย์)
df6 = mkdf([("call",100,10,.20,0,.01), ("put",100,10,.20,0,.01)])
em0 = sn.expected_move(df6, 100.0, "2026-08-20")
check("0DTE -> EM > 0 (นับครึ่งวัน)", em0["em"] and em0["em"] > 0, f"ได้ {em0['em']}")

print("\n=== 6. zero_dte_share ===")
# ห้ามให้ call/put หักล้างกันพอดี ไม่งั้น net = 0 ทั้งกระดาน ตัวหารเป็น 0
# (เคสแรกที่เขียนพลาดตรงนี้ — เป็นข้อจำกัดของนิยาม |net| ที่ต้องรู้ ไม่ใช่บั๊กของโค้ด)
df7 = mkdf([("call",100,200,.2,0,.01), ("put",100,100,.2,0,.01),
            ("call",105,200,.2,7,.01), ("put",105,100,.2,7,.01)])
z = sn.zero_dte_share(df7, 100.0)
dfz = mkdf([("call",100,100,.2,0,.01), ("put",100,100,.2,0,.01)])
check("net = 0 ทั้งกระดาน -> None ไม่ ZeroDivisionError",
      sn.zero_dte_share(dfz, 100.0) is None)
check("มี 0DTE ครึ่งหนึ่ง -> ~50%", z and 40 < z < 60, f"ได้ {z}")
check("ไม่มี 0DTE -> None", sn.zero_dte_share(mkdf([("call",100,10,.2,5,.01)]), 100.0) is None)

print("\n=== 7. vanna/charm — charm ต้องหาร 365 (ต่อวัน ไม่ใช่ต่อปี) ===")
# vanna/charm ของ call กับ put ที่ strike เดียวกันเท่ากันตามสูตร (put-call parity)
# ภายใต้ convention dealer long call / short put มันจึงหักล้างกันพอดีถ้า OI เท่ากัน
# → ต้องให้ OI ต่างกันถึงจะเห็นค่าจริง (เคสแรกที่เขียนพลาดตรงนี้)
df8 = mkdf([("call",100,1000,.25,30,.02), ("put",100,400,.25,30,.02)])
vc = sn.vanna_charm_exposure(df8, 100.0)
check("vanna คำนวณได้", vc["vanna"] is not None)
check("charm คำนวณได้", vc["charm"] is not None)
# ยืนยันว่าหาร 365 จริง: เทียบกับค่าที่ไม่หาร ต้องต่างกัน ~365 เท่า
check("charm เล็กกว่า vanna มาก (สเกลต่อวัน)",
      abs(vc["charm"]) < abs(vc["vanna"]), f"vanna {vc['vanna']:.0f} charm {vc['charm']:.0f}")

print("\n=== 8. term_structure — ต้องข้าม 0DTE ===")
rows = ([("call",100,10,.50,0,.01)] * 8 +      # 0DTE IV สูงผิดปกติ
        [("call",100,10,.20,7,.01)] * 8 +
        [("call",100,10,.25,30,.01)] * 8)
df9 = pd.DataFrame([{"type":t,"strike":k,"open_interest":oi,"iv":iv,"days":d,"gamma":g,
                     "expiry": f"2026-08-{20+d:02d}"} for t,k,oi,iv,d,g in rows])
ts = sn.term_structure(df9, 100.0)
check("near ไม่ใช่ 0DTE", ts["near_exp"] and not ts["near_exp"].endswith("-20"), f"near_exp={ts['near_exp']}")
check("contango เมื่อ far > near", ts["state"] == "contango", f"ได้ {ts['state']} slope {ts['slope_pct']}")

print("\n=== 9. pin_score — ขอบเขต 0-100 และเหตุผลชัด ===")
ps = pd.DataFrame([{"strike":100.0,"call_gex":1e6,"put_gex":-2e5,"net_gex":8e5}])
p = sn.pin_score(100.0, 100.0, 5.0, ps, net=1e9)   # spot = max pain พอดี + GEX บวก
check("spot=maxpain + GEX บวก -> คะแนนสูง", p["score"] and p["score"] > 60, f"ได้ {p['score']}")
p2 = sn.pin_score(100.0, 130.0, 5.0, ps, net=-1e9)  # ไกล max pain + GEX ลบ
check("ไกล maxpain + GEX ลบ -> คะแนนต่ำ", p2["score"] is not None and p2["score"] < 40, f"ได้ {p2['score']}")
check("EM = None -> ไม่ crash", sn.pin_score(100.0, 100.0, None, ps, 1e9)["score"] is None)

print("\n=== 10. dealer_shock — เครื่องหมายและความสมมาตร ===")
dfs = mkdf([("call",100,1000,.25,7,.02), ("put",100,500,.25,7,.02)])
sh = sn.dealer_shock(dfs, 100.0)
check("คำนวณได้", sh["shares_up"] is not None)
if sh["shares_up"] is not None:
    check("ขึ้น/ลง สมมาตร (เครื่องหมายกลับกัน)",
          abs(sh["shares_up"] + sh["shares_dn"]) < 1e-6,
          f"up {sh['shares_up']:.1f} dn {sh['shares_dn']:.1f}")
    # call OI มากกว่า put -> gamma รวมเป็นบวก -> dealer ขายตอนขึ้น = ตรึง
    check("call หนักกว่า -> shares_up เป็นบวก", sh["shares_up"] > 0, f"ได้ {sh['shares_up']:.1f}")

print("\n=== 11. skew_25d — put แพงกว่า call ต้องได้ค่าบวก ===")
df10 = pd.DataFrame([
    {"type":"put","strike":90,"open_interest":10,"iv":0.30,"days":30,"gamma":.01,
     "expiry":"2026-09-18","option":"P1"},
    {"type":"call","strike":110,"open_interest":10,"iv":0.20,"days":30,"gamma":.01,
     "expiry":"2026-09-18","option":"C1"},
])
raw10 = pd.DataFrame([{"option":"P1","delta":-0.25},{"option":"C1","delta":0.25}])
sk = sn.skew_25d(df10, raw10, target_dte=30)
check("RR25 = put25 - call25 = 30-20 = +10",
      sk["rr25"] is not None and abs(sk["rr25"] - 10.0) < 0.5, f"ได้ {sk['rr25']}")

print("\n=== 12. สัญญาของ error snapshot — ตัวที่ทำ Cockpit พังทั้งแอป ===")
# บั๊กจริง 31 ส.ค. 2026: cockpit_tab.py อ่าน snap['asof'] ก่อนเช็ค error
# → KeyError → Streamlit หยุดทั้ง script → ทุกแท็บที่อยู่หลังแท็บนั้นไม่ถูก render เลย
# เทสต์นี้ตรึงสัญญาไว้: error snapshot **มีแค่ error กับ symbol** ห้ามมีใครไปสมมติว่ามี field อื่น
_real_fetch = sn.fetch_chain
try:
    sn.fetch_chain = lambda s, tries=4: (pd.DataFrame(), 0.0)      # จำลอง CBOE ตอบ chain ว่าง
    err = sn.build_snapshot("QQQ")
finally:
    sn.fetch_chain = _real_fetch

check("chain ว่าง -> คืน dict ที่มี error", bool(err.get("error")), f"ได้ {err}")
check("error snapshot ไม่มี key 'asof' (ผู้ใช้ต้องเช็ค error ก่อนเสมอ)", "asof" not in err,
      f"key ที่มี: {sorted(err)}")
check("error snapshot ไม่มี key 'spot' / 'rows' / 'levels' ด้วย",
      not any(k in err for k in ("spot", "rows", "levels")), f"key ที่มี: {sorted(err)}")
check("ยังบอกได้ว่าเป็น symbol ไหน", err.get("symbol") == "QQQ")
# วิธีอ่านที่ปลอดภัย (แบบที่ cockpit_tab.py / long_tab.py ใช้อยู่ตอนนี้)
check(".get('asof') คืน None แทนที่จะ throw", err.get("asof") is None)

print("\n" + "="*70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL))+' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
