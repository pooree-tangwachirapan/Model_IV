"""
ตรวจ breakout — "อยู่นอกกำแพง" ต้องไม่ถูกนับเป็น "ทะลุกำแพง" ถ้าราคาไม่ได้ข้ามมาจริง (HANDOFF §6.17)

บั๊กจริง 9 พ.ย. 2026: GEX Put Wall กระโดด 650 → 680 ขึ้นไปอยู่เหนือราคา (15:00Z) ขณะที่ QQQ ขึ้นทั้งวัน
ระบบอ่าน "ราคาต่ำกว่า Put Wall" เป็นการหลุดลงสด → ARMED SHORT 9 ครั้ง (16:22–19:22Z) + เปิดไม้จำลอง
ทั้งที่ high ของวันจนถึงครั้งสุดท้ายคือ 678.85 · ปิดวันก่อน 656.10 — ราคาไม่เคยอยู่เหนือ 680 เลย

ตัวเลขทุกตัวมาจากข้อมูลที่บันทึกไว้จริง ไม่ได้แต่งให้เทสต์ผ่าน:
  · กำแพง/flip/net/EM/spot = forward_test/log/2026-11.jsonl และ 2026-10.jsonl (ค่าเต็มทศนิยม)
  · high/low ณ ตอนนั้น = แท่ง 1 นาที Yahoo (session ปกติ · รวมแท่งที่คร่อมเวลา record = ขอบบน)
    เพราะ log ก่อน 10.11.2026 ไม่ได้เก็บ high/low · ราคาปิด = แท่งรายวัน Yahoo
  · 5 พ.ย. ช่วงทะลุ 680 ไม่มีรอบประเมินตกหน้าต่างเลย (cron-job.org ยังไม่ปลุก armed-alert)
    → ใช้ราคาจริงจากแท่ง 1 นาที (676.01 = ปิดแท่ง 14:36Z) + spot ของ chain 15:15:36Z (667.61)
    กับกำแพงที่บันทึกไว้วันนั้น (GEX 680 · OI 670) และ EM/flip/net ของรอบ 15:00:10Z

เทสต์ชุดนี้ตั้งใจให้ "ล้ม" กับโค้ดก่อนแก้ (ข้อ 1–2) และ "ผ่าน" กับการทะลุจริง (ข้อ 3–5) —
ตรวจแล้วทั้งสองทางตอนเขียน (HANDOFF §6.17) · ไม่ใช้ breakout.CROSS_GATE และอ่าน key ใหม่ด้วย .get()
ในข้อ 1–9 โดยตั้งใจ เพื่อให้รันกับโค้ดเก่าแล้วได้ FAIL ที่อ่านออก ไม่ใช่ KeyError
"""
import warnings; warnings.filterwarnings("ignore")
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง

from datetime import date, datetime, timezone

import breakout, gate, predictions as pr, snapshot as sn

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

UTC = timezone.utc
CROSS = "ข้ามกำแพงมาจากในโซน"


def snap(spot, pw, cw, flip, net, em, *, pw_oi=None, cw_oi=None, mp=None, net_prof=None,
         session=None):
    s = {"symbol": "QQQ", "spot": spot, "asof": datetime(2026, 11, 9, 16, 22, 2, tzinfo=UTC),
         "max_pain": mp,
         "levels": {"put_wall": pw, "call_wall": cw, "flip": flip, "net": net,
                    "put_wall_oi": pw_oi, "call_wall_oi": cw_oi,
                    "net_profile": net_prof,
                    "net_agree": (None if net_prof is None else (net_prof < 0) == (net < 0))},
         "expected_move": {"em": em}, "term": {}, "rows": []}
    if session is not None:
        s["session"] = session
    return s


def failed(g):
    return [r["label"] for r in g["hard"] if not r["ok"]]


def gate_row(g, label):
    return next((r for r in g["hard"] if r["label"] == label), None)


# ════════════════════════════════════════════════════════════════════
print("=== 1. 9 พ.ย. 2026 16:22:02Z — record ARMED ตัวแรก (ค่าจริงจาก log) ===")
# ปิด 6 พ.ย. = 656.10 · high จนถึง 16:22Z = 678.85 · low = 662.02 (แท่ง 1 นาที Yahoo)
S9 = {"open": 662.64, "high": 678.85, "low": 662.02, "prev_close": 656.10}
s1 = snap(676.67, 680.0, 700.0, 697.3352764879404, -2971528429.0714283, 7.59354707218131,
          pw_oi=650.0, cw_oi=700.0, mp=660.0, net_prof=-2625668535.1992564, session=S9)
b1 = breakout.evaluate(s1)
check("ไม่ ARMED อีกแล้ว (เดิม ARMED SHORT แล้วเปิดไม้จำลอง)", b1["verdict"] == "STAND_DOWN",
      f"ได้ {b1['verdict']} — {b1['reason']}")
check("ตัวปิดมีตัวเดียวคือการข้ามกำแพง — ประตูอื่นผ่านหมดเหมือนที่ log บันทึก",
      failed(b1) == [CROSS], f"ไม่ผ่าน: {failed(b1)}")
row = gate_row(b1, CROSS)
check("ค่าในประตูบอก high ของวันกับกำแพงให้เห็น",
      row is not None and "678.85" in row["value"] and "680.00" in row["value"],
      f"ได้ {row and row['value']}")
check("headline บอกว่าไม่ใช่การทะลุ", b1["headline"].startswith("ไม่ใช่การทะลุ"),
      f"ได้ {b1['headline']!r}")
check("ยังเก็บแผนไว้ตอบ 'ถ้าฝืนเทรด' — R เท่าที่ log บันทึก (2.994)",
      (b1.get("plan") or {}).get("plan_r") is not None and round(b1["plan"]["plan_r"], 3) == 2.994,
      f"ได้ {b1.get('plan')}")
br1 = b1.get("breach") or {}
check("เรขาคณิตไม่เปลี่ยน: ยังเป็น DOWN ห่าง 0.44×EM อยู่ในหน้าต่าง",
      br1.get("direction") == "DOWN" and br1.get("fresh") is True
      and abs(br1["dist_em"] - (680 - 676.67) / 7.59354707218131) < 1e-12,
      f"ได้ {br1}")
f1 = gate.evaluate(s1)
check("fade ไม่ ARMED พร้อมกัน · conflicts_with เงียบ",
      f1["verdict"] != "ARMED" and breakout.conflicts_with(f1, b1) is None, f"fade {f1['verdict']}")

print("\n=== 2. 9 พ.ย. ARMED ทั้ง 9 record → STAND_DOWN ทุกตัว ตัวปิดคือการข้ามกำแพงตัวเดียว ===")
# (เวลา, spot, flip, net, em) จาก log — กำแพง 680/700 · OI 650/700 · max pain 660 ทุกตัว
# high จนถึงทุกเวลานี้ = 678.85 (แท่ง 1 นาที · QQQ แตะ 680 ครั้งแรก 19:38Z หลังตัวสุดท้าย)
NOV9 = [("16:22:02", 676.67, 697.3352764879404, -2971528429.0714283, 7.59354707218131),
        ("17:07:03", 676.71, 699.3933003799934, -2953815054.7857256, 6.425592549043118),
        ("17:22:03", 676.9, 699.8506490102488, -2940169084.9672136, 6.175612074287505),
        ("17:37:03", 678.01, 698.8404445931743, -2861539908.623228, 6.2597671338077125),
        ("17:52:02", 677.985, 698.1931786528277, -2846773324.3491693, 6.057534847100989),
        ("18:07:02", 677.94, 699.7296933962893, -2794289991.5797467, 5.867690566731055),
        ("18:52:03", 677.33, 700.4735251571803, -2813921153.642769, 5.141674053498543),
        ("19:07:03", 678.14, 699.6295399277456, -2710311329.1173096, 5.427677662829306),
        ("19:22:04", 677.74, 700.1249165466817, -2727411643.4230447, 5.391866586563045)]
bad = []
for ts, spot, flip, net, em in NOV9:
    g = breakout.evaluate(snap(spot, 680.0, 700.0, flip, net, em, pw_oi=650.0, cw_oi=700.0,
                               mp=660.0, session=S9))
    if g["verdict"] != "STAND_DOWN" or failed(g) != [CROSS]:
        bad.append((ts, g["verdict"], failed(g)))
check("9/9 ไม่ ARMED และปิดด้วยการข้ามกำแพงอย่างเดียว", not bad, f"เคสที่ผิด: {bad}")

# ════════════════════════════════════════════════════════════════════
print("\n=== 3. 5 พ.ย. 2026 — หลุด 680 ลงมาจริง (เปิด 683.17 · high 683.64 · ปิดวันก่อน 691.95) ===")
LV5 = dict(flip=713.4735479932032, net=-10118554151.597052, em=13.09854231588572)
S5a = {"open": 683.17, "high": 683.64, "low": 675.95, "prev_close": 691.95}
b3 = breakout.evaluate(snap(676.01, 680.0, 700.0, LV5["flip"], LV5["net"], LV5["em"],
                            pw_oi=670.0, cw_oi=700.0, mp=694.0,
                            net_prof=-9678361682.507404, session=S5a))
check("ARMED SHORT — การทะลุจริงต้องยังยิงได้", b3["verdict"] == "ARMED"
      and (b3.get("plan") or {}).get("side") == "SHORT", f"ได้ {b3['verdict']} — {b3['reason']}")
row = gate_row(b3, CROSS)
check("ประตูข้ามกำแพงผ่าน ด้วย high ของวัน 683.64",
      row is not None and row["ok"] and "683.64" in row["value"], f"ได้ {row}")
check("R ≥ 2.5 ตามเรขาคณิตเดิม", (b3.get("plan") or {}).get("plan_r", 0) >= breakout.MIN_PLAN_R,
      f"ได้ {b3.get('plan')}")

print("\n=== 4. 5 พ.ย. — หลุดต่อลง 670 (กำแพง OI ของวันนั้น) ที่ spot ของ chain 15:15:36Z ===")
S5b = {"open": 683.17, "high": 683.64, "low": 665.01, "prev_close": 691.95}
b4 = breakout.evaluate(snap(667.61, 670.0, 700.0, LV5["flip"], LV5["net"], LV5["em"],
                            pw_oi=670.0, cw_oi=700.0, mp=694.0, session=S5b))
check("ARMED SHORT ที่ 670", b4["verdict"] == "ARMED" and b4["breach"].get("wall") == 670.0,
      f"ได้ {b4['verdict']} — {b4['reason']}")
check("ข้าม 670 ยืนยันได้ (เคยอยู่เหนือ 670 วันนี้)", b4["breach"].get("crossed") is True,
      f"ได้ {b4['breach'].get('crossed')}")
# กระจกของข้อ 1: ถ้ากำแพง 670 มาอยู่เหนือราคาโดยที่ราคาไม่เคยขึ้นถึงตั้งแต่ปิดวันก่อน = ไม่ใช่การทะลุ
b4x = breakout.evaluate(snap(667.61, 670.0, 700.0, LV5["flip"], LV5["net"], LV5["em"],
                             pw_oi=670.0, cw_oi=700.0, mp=694.0,
                             session={"high": 669.5, "low": 660.0, "prev_close": 655.0}))
check("ตัวเลขเดียวกันแต่ไม่เคยขึ้นถึง 670 → STAND_DOWN", b4x["verdict"] == "STAND_DOWN"
      and failed(b4x) == [CROSS], f"ได้ {b4x['verdict']} {failed(b4x)}")

print("\n=== 5. 5 พ.ย. 15:00:10Z record จริง — ไกลเกินหน้าต่าง: ตัวแก้ต้องไม่เปลี่ยนผล ===")
b5 = breakout.evaluate(snap(672.5, 680.0, 700.0, LV5["flip"], LV5["net"], LV5["em"],
                            pw_oi=670.0, cw_oi=700.0, mp=694.0,
                            net_prof=-9678361682.507404,
                            session={"high": 683.64, "low": 671.66, "prev_close": 691.95}))
logged = {"ข้อมูลกำแพง": True, "ราคาทะลุกำแพงแล้ว": True, "ทะลุสด ไม่ตกรถ": False,
          "Net GEX ไม่เป็นบวก": True, "ฝั่งของ Gamma Flip": True, "เหลือระยะให้วิ่ง": False,
          "เรขาคณิต ≥ 2.5R": False}
got = {r["label"]: r["ok"] for r in b5["hard"]}
check("STAND_DOWN เหมือนที่บันทึก", b5["verdict"] == "STAND_DOWN", f"ได้ {b5['verdict']}")
check("ประตูเดิมให้ผลเท่าที่ log บันทึกทุกตัว + ประตูใหม่ผ่าน",
      {k: v for k, v in got.items() if k != CROSS} == logged and got.get(CROSS) is True,
      f"ได้ {got}")

print("\n=== 6. 12 ต.ค. 2026 13:52:03Z record จริง — gap ลงข้าม 750 ตอนเปิด (ปิด 751.27 · high 749.79) ===")
b6 = breakout.evaluate(snap(748.27, 750.0, 760.0, 749.2619411287219, -1325758542.527626,
                            5.224608024315206, pw_oi=700.0, cw_oi=800.0, mp=751.0,
                            net_prof=-860167861.8359218,
                            session={"high": 749.79, "low": 747.0, "prev_close": 751.27}))
check("ยัง ARMED — ปิดวันก่อนอยู่เหนือกำแพง = ข้ามมาแล้วตอน gap", b6["verdict"] == "ARMED",
      f"ได้ {b6['verdict']} — {b6['reason']}")
check("ยืนยันด้วยราคาปิดวันก่อน", (b6["breach"].get("cross") or {}).get("src") == "prev_close",
      f"ได้ {b6['breach'].get('cross')}")
check("R เท่าที่ log บันทึก (3.732)", round(b6["plan"]["plan_r"], 3) == 3.732, f"ได้ {b6.get('plan')}")

# ════════════════════════════════════════════════════════════════════
print("\n=== 7. กระจกขาขึ้น — Call Wall ย้ายลงมาใต้ราคา vs ทะลุขึ้นจริง ===")
up = lambda sess: breakout.evaluate(snap(661.5, 640.0, 660.0, 650.0, -2e9, 5.0, session=sess))
u_moved = up({"high": 663.0, "low": 660.8, "prev_close": 663.0})
u_real = up({"high": 661.9, "low": 657.0, "prev_close": 658.0})
check("ไม่เคยลงถึง 660 ตั้งแต่ปิดวันก่อน → STAND_DOWN", u_moved["verdict"] == "STAND_DOWN"
      and failed(u_moved) == [CROSS], f"ได้ {u_moved['verdict']} {failed(u_moved)}")
check("เคยลงถึง 657 วันนี้แล้วขึ้นข้าม 660 → ARMED LONG", u_real["verdict"] == "ARMED"
      and u_real["plan"]["side"] == "LONG", f"ได้ {u_real['verdict']} — {u_real['reason']}")

print("\n=== 8. ขอบ: แตะกำแพงพอดีนับเป็นเคยอยู่ในโซน (โซนรวมขอบ ตรงกับ breach_state) ===")
edge = breakout.evaluate(snap(676.67, 680.0, 700.0, 697.3352764879404, -2971528429.0714283,
                              7.59354707218131, session=dict(S9, high=680.0)))
check("high = 680.00 พอดี → ข้ามแล้ว", edge["breach"].get("crossed") is True and edge["verdict"] == "ARMED",
      f"ได้ {edge['breach'].get('crossed')} {edge['verdict']}")

print("\n=== 9. ข้อมูลไม่ครบ = ไม่ไล่ (แบบเดียวกับไม่มี flip / ไม่มี EM) แต่ไม่ใช่ data issue ===")
base = lambda sess: breakout.evaluate(snap(676.67, 680.0, 700.0, 697.3352764879404,
                                           -2971528429.0714283, 7.59354707218131, session=sess))
none_s = base(None)
check("snapshot ไม่มี session → STAND_DOWN", none_s["verdict"] == "STAND_DOWN"
      and none_s["breach"].get("crossed", "absent") is None, f"ได้ {none_s['verdict']} {none_s['breach'].get('crossed')}")
check("ไม่ตีเป็น data issue (หัวเมลจะขึ้น ⚠️ DATA ทั้งที่กำแพงปกติ)", none_s["data_issue"] is False)
check("headline บอกว่ายืนยันไม่ได้", none_s["headline"].startswith("ยืนยันไม่ได้"),
      f"ได้ {none_s['headline']!r}")
part = base({"high": None, "low": 662.0, "prev_close": 656.10})
check("มีแต่ปิดวันก่อน (ต่ำกว่ากำแพง) ไม่มี high → ยังสรุปว่า 'ไม่ข้าม' ไม่ได้ (None)",
      part["breach"].get("crossed", "absent") is None, f"ได้ {part['breach'].get('crossed')}")
zero = base({"high": 0, "low": 0, "prev_close": "x"})
check("0 / ตัวอักษร = ไม่มีข้อมูล (CBOE ส่ง 0 ได้)", zero["breach"].get("crossed", "absent") is None
      and zero["verdict"] == "STAND_DOWN", f"ได้ {zero['breach'].get('cross')}")
pc_only = base({"high": None, "low": None, "prev_close": 681.0})
check("ไม่มี high แต่ปิดวันก่อนอยู่เหนือกำแพง → ยืนยันได้", pc_only["breach"].get("crossed") is True)

# ════════════════════════════════════════════════════════════════════
print("\n=== 10. ค่าคงที่ของหน้าต่างไม่ถูกแตะ (HANDOFF §3 · §9 ห้ามจูน) ===")
check("เกณฑ์ทุกตัวยังเท่าเดิม",
      (breakout.BREACH_MIN_EM, breakout.STOP_EM_FRAC, breakout.TARGET_EM_MULT, breakout.MIN_PLAN_R,
       breakout.MIN_EM_ROOM, breakout.MAX_0DTE_SHARE) == (0.10, 0.25, 2.50, 2.50, 1.00, 60.0))
want = (breakout.TARGET_EM_MULT - breakout.MIN_PLAN_R * breakout.STOP_EM_FRAC) / (1 + breakout.MIN_PLAN_R)
check("เพดานหน้าต่างยังคำนวณจากสูตร (0.536×EM)",
      abs(breakout.BREACH_MAX_EM - want) < 1e-12 and round(breakout.BREACH_MAX_EM, 3) == 0.536,
      f"ได้ {breakout.BREACH_MAX_EM}")
EM, W = 5.0, 700.0
ok_sess = {"high": 701.0, "low": 690.0, "prev_close": 702.0}
# ขอบด้านใน 1e-9 — ที่ขอบพอดี (W − MAX×EM) ทศนิยมลอยตัวทำ d เกินเพดาน 1 ulp แล้วตกหน้าต่าง
s_edge = W - (breakout.BREACH_MAX_EM - 1e-9) * EM
at_edge = breakout.build_plan(s_edge, breakout.breach_state(s_edge, W, 720.0, EM, session=ok_sess), EM)
check("ที่ขอบหน้าต่าง R = MIN_PLAN_R พอดี (ผูกกันด้วยสูตร)", at_edge is not None
      and at_edge["plan_r"] >= breakout.MIN_PLAN_R
      and abs(at_edge["plan_r"] - breakout.MIN_PLAN_R) < 1e-6, f"ได้ {at_edge}")
for d, want_v in ((breakout.BREACH_MAX_EM + 0.01, "too_far"), (breakout.BREACH_MIN_EM - 0.01, "too_close")):
    g = breakout.evaluate(snap(W - d * EM, W, 720.0, 710.0, -2e9, EM, session=ok_sess))
    check(f"d = {d:.3f}×EM ข้ามจริงแต่ {want_v} → STAND_DOWN", g["verdict"] == "STAND_DOWN"
          and g["breach"][want_v] and g["breach"].get("crossed") is True, f"ได้ {g['verdict']}")
same = all(
    {k: v for k, v in breakout.breach_state(s, 680.0, 700.0, 7.0, session=se).items()
     if k not in ("crossed", "cross")} ==
    {k: v for k, v in breakout.breach_state(s, 680.0, 700.0, 7.0).items() if k not in ("crossed", "cross")}
    for s in (672.0, 676.0, 679.5, 703.0) for se in (S9, {"high": 690.0, "low": 650.0, "prev_close": 685.0}))
check("session ไม่เปลี่ยน direction/dist/fresh/too_far/too_close แม้แต่ตัวเดียว", same)

print("\n=== 11. ป้ายประตู = key ใน log — ห้ามเปลี่ยนชื่อเงียบ ๆ ===")
check("breakout.CROSS_GATE ตรงกับป้ายที่ log ใช้", getattr(breakout, "CROSS_GATE", None) == CROSS)
labels = [r["label"] for r in b3["hard"]]
check("ประตูใหม่อยู่ถัดจาก 'ราคาทะลุกำแพงแล้ว' ก่อน 'ทะลุสด ไม่ตกรถ'",
      CROSS in labels and labels.index(CROSS) == labels.index("ราคาทะลุกำแพงแล้ว") + 1
      and labels.index("ทะลุสด ไม่ตกรถ") == labels.index(CROSS) + 1, f"ลำดับ {labels}")

# ════════════════════════════════════════════════════════════════════
print("\n=== 12. snapshot ส่ง high/low/ปิดวันก่อนของ response เดียวกันมาให้ (ไม่แตะเน็ต) ===")
sfq = getattr(sn, "session_from_quote", None)
hdr = {"open": 662.69, "high": 682.48, "low": 662.02, "close": 679.35, "prev_day_close": 679.35,
       "current_price": 678.22, "last_trade_time": "2026-11-09T15:59:59"}   # CBOE จริง 10.11 03:54 ET
q = sfq(hdr) if sfq else {}
check("แปลง header จริงได้ครบ", q.get("high") == 682.48 and q.get("low") == 662.02
      and q.get("prev_close") == 679.35 and q.get("open") == 662.69
      and q.get("last_trade_time") == "2026-11-09T15:59:59", f"ได้ {q}")
q0 = sfq({"high": 0, "low": "", "prev_day_close": None}) if sfq else {"high": 1}
check("0 / ว่าง / None → None", q0.get("high") is None and q0.get("low") is None
      and q0.get("prev_close") is None, f"ได้ {q0}")


class _Resp:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


_real_get = sn.requests.get
try:
    sn.requests.get = lambda *a, **k: _Resp({"data": dict(hdr, options=[
        {"option": "QQQ261109P00680000", "open_interest": 10, "gamma": 0.01, "iv": 0.3}])})
    df, spot = sn.fetch_chain("QQQ", tries=1)
finally:
    sn.requests.get = _real_get
check("fetch_chain ยังคืน (df, spot) เหมือนเดิม", spot == 678.22 and len(df) == 1)
check("high/low แนบมากับ df.attrs['session']", (df.attrs.get("session") or {}).get("high") == 682.48,
      f"ได้ {df.attrs}")

# ด่านกระดานค้าง (12.11.2026 · HANDOFF §6.17 ข้อ 2): ก่อนเปิด high/low ยังเป็นของวันก่อน (เกิดก่อนปิดวันก่อน)
# รันจริง 10 พ.ย. 04:xx ET + รอบ 09:00 ET ใน log เคยได้ ARMED จาก high 682.48 ของวันที่ 9 — ต้องไม่ได้อีก
pre_args = (678.22, 680.0, 700.0, 703.01, -3.68e9, 7.55)
sfs = getattr(sn, "session_for_snapshot", None)
qs = sfs(df, date(2026, 11, 10)) if sfs else {}
check("df จาก CBOE ก่อนเปิด 10 พ.ย. → stale (t เป็นของ 9 พ.ย.)", (qs or {}).get("stale") is True, f"ได้ {qs}")
pre = breakout.evaluate(snap(*pre_args, session=qs))
row = gate_row(pre, CROSS) or {}
check("ก่อนเปิด: ไม่ใช้ high ของ session ก่อน → ยืนยันไม่ได้ → STAND_DOWN · ค่าประตูบอกวันของกระดาน",
      pre["verdict"] == "STAND_DOWN" and pre["breach"].get("crossed", "absent") is None
      and "2026-11-09" in row.get("value", ""), f"ได้ {pre['verdict']} {row}")
eve = breakout.evaluate(snap(*pre_args, session=sfs(df, date(2026, 11, 9)) if sfs else {}))
check("ตัวเลขเดียวกันตอนเย็นวันที่ 9 (ไม่ค้าง) → ใช้ high 682.48 ยืนยันได้",
      (eve["breach"].get("cross") or {}).get("src") == "high", f"ได้ {eve['breach'].get('cross')}")
unk = breakout.evaluate(snap(*pre_args, session=q))
check("ไม่รู้ว่าค้างไหม (ไม่มีธง stale) → ทำเหมือนเดิม ไม่ปิดหลักฐานทิ้งเงียบ ๆ",
      (unk["breach"].get("cross") or {}).get("src") == "high", f"ได้ {unk['breach'].get('cross')}")
gap = breakout.evaluate(snap(*pre_args, session=dict(qs or {}, prev_close=681.0)))
check("กระดานค้างแต่ปิดวันก่อนอยู่เหนือกำแพง → ยืนยันด้วยราคาปิด (ข้ามหลังปิด) → ARMED",
      (gap["breach"].get("cross") or {}).get("src") == "prev_close" and gap["verdict"] == "ARMED",
      f"ได้ {gap['verdict']} {gap['breach'].get('cross')}")

print("\n=== 13. log เก็บ high/low/ปิดวันก่อน + ธงกระดานค้างไว้ย้อนตรวจ ===")
r1 = pr.to_record(dict(b1, system="breakout"),
                  dict(s1, session=dict(S9, last_trade_time="t", stale=False)),
                  datetime(2026, 11, 9, 16, 22, 2, tzinfo=UTC))
check("record มี sess ครบ", r1.get("sess") == {"hi": 678.85, "lo": 662.02, "pc": 656.10, "t": "t",
                                                "st": False}, f"ได้ {r1.get('sess')}")
check("record มีผลประตูใหม่", r1["gates"].get(CROSS) is False, f"ได้ {r1['gates']}")
check("serialize เป็น JSON ได้", isinstance(json.dumps(r1, ensure_ascii=False), str))
r0 = pr.to_record(dict(none_s, system="breakout"), snap(676.67, 680.0, 700.0, 697.3, -1e9, 7.6))
check("snapshot ไม่มี session → sess = None (ไม่เดาค่า)", r0.get("sess", "missing") is None,
      f"ได้ {r0.get('sess', 'missing')}")

print("\n=== 14. 10 พ.ย. 14:00:11Z (09:00 ET) record จริง — ARMED จากกระดานค้าง ===")
S10 = {"high": 682.48, "low": 662.02, "prev_close": 679.35, "last_trade_time": "2026-11-09T15:59:59"}
args10 = (676.6, 680.0, 700.0, 693.914052754915, -4124194028.183152, 12.1754639153789)
kw10 = dict(pw_oi=650.0, cw_oi=700.0, mp=675.0, net_prof=-3793980169.669641)
was = breakout.evaluate(snap(*args10, **kw10, session=dict(S10, stale=None)))
check("ไม่มีธง (แบบตอนนั้น) → ARMED · R 4.196 เท่าที่ log บันทึก",
      was["verdict"] == "ARMED" and round((was.get("plan") or {}).get("plan_r", 0), 3) == 4.196,
      f"ได้ {was['verdict']} {was.get('plan')}")
f_st = getattr(sn, "session_is_stale", None)
now10 = breakout.evaluate(snap(*args10, **kw10,
                               session=dict(S10, stale=f_st(S10, date(2026, 11, 10)) if f_st else None)))
check("มีด่าน → STAND_DOWN ตัวปิดคือการข้ามกำแพงตัวเดียว (ปิดวันก่อน 679.35 < 680)",
      now10["verdict"] == "STAND_DOWN" and failed(now10) == [CROSS],
      f"ได้ {now10['verdict']} {failed(now10)}")
check("headline = ยืนยันไม่ได้ (ก่อนเปิดไม่รู้ว่า pre-market ไปถึงไหน — ไม่ใช่ 'ไม่ใช่การทะลุ')",
      now10["headline"].startswith("ยืนยันไม่ได้"), f"ได้ {now10['headline']!r}")

print("\n=== 15. session_is_stale — วันของ last_trade_time เทียบวันนี้ (ET) ===")
cases = [("2026-11-09T15:59:59", date(2026, 11, 10), True),    # ก่อนเปิด 10.11 (ของจริงใน log)
         ("2026-11-10T09:36:15", date(2026, 11, 10), False),   # รอบ 09:52 ET 10.11 (ของจริงใน log)
         ("2026-11-09T15:59:59", date(2026, 11, 9), False),    # เย็นวันเดียวกัน
         ("2026-11-06T15:59:59", date(2026, 11, 9), True),     # จันทร์ก่อนเปิด หลังเสาร์อาทิตย์
         (None, date(2026, 11, 10), None), ("", date(2026, 11, 10), None),
         ("abc", date(2026, 11, 10), None)]
bad = [(t, d, w) for t, d, w in cases if not f_st or f_st({"last_trade_time": t}, d) is not w]
check("ทุกกรณีถูก (ไม่รู้ = None ไม่ใช่ True)", not bad, f"ผิด: {bad}")
check("session เป็น None → None", f_st is not None and f_st(None, date(2026, 11, 10)) is None)
import pandas as pd
check("df ไม่มี attrs → session None (breakout จะยืนยันไม่ได้)",
      sfs is not None and sfs(pd.DataFrame(), date(2026, 11, 10)) is None)

print("\n" + "=" * 70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL)) + ' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
sys.exit(1 if FAIL else 0)
