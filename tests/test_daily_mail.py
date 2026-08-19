"""ตรวจว่าเมลรายวันแสดง breakout ครบ + state key ไม่ชนกัน"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, re, html as H
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง
import breakout, gate, send_report as sr
from datetime import datetime

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

def snap(spot, pw, cw, flip, net, em=5.0):
    return {'symbol':'QQQ','spot':spot,'asof':datetime.now(),'max_pain':None,
            'levels':{'put_wall':pw,'call_wall':cw,'flip':flip,'net':net,
                      'call_wall_oi':cw,'put_wall_oi':pw},
            'expected_move':{'em':em},'term':{'slope_pct':-5.0,'state':'backwardation'},
            'rows':[{'label':'0DTE share','value':'10.0% of |GEX|'}]}

print("=== 1. breakout ARMED ต้องโผล่ในเมล พร้อมป้ายระบบ ===")
s = snap(731.5, 700, 730, 720, -2e9)          # ทะลุขึ้นสด · −GEX · เหนือ flip
bg = breakout.evaluate(s); bg.setdefault("system", "breakout")
fg = gate.evaluate(s);     fg.setdefault("system", "fade")
check("breakout = ARMED", bg["verdict"] == "ARMED", f"ได้ {bg['verdict']} — {bg['reason']}")
check("fade ไม่ ARMED พร้อมกัน", fg["verdict"] != "ARMED", f"ได้ {fg['verdict']}")

htm = sr.build_gate_html([fg, bg])
check("HTML มีป้าย BREAKOUT", "BREAKOUT" in htm)
check("HTML มีป้าย FADE", "FADE" in htm)
check("HTML มี headline ของ breakout", bg["headline"][:25] in htm)
check("HTML มีแผนไล่ตาม (entry/target)", "invalidate" in htm and "target" in htm)

print("\n=== 2. state key ต้องแยกตามระบบ ไม่งั้นทับกัน ===")
cur = {f"{g['symbol']}·{g.get('system','fade')}":
       ("DATA_ISSUE" if g.get("data_issue") else g["verdict"]) for g in [fg, bg]}
check("มี 2 key ไม่ทับกัน", len(cur) == 2, f"ได้ {cur}")
check("key มีชื่อระบบ", all("·" in k for k in cur), f"{list(cur)}")
check("เก็บ verdict คนละค่าได้", len(set(cur.values())) == 2 or fg["verdict"] == bg["verdict"],
      f"{cur}")

print("\n=== 3. subject ต้องบอกว่าเป็นระบบไหนตอน ARMED ===")
rank = {"ARMED": 0, "WATCH": 1, "STAND_DOWN": 2}
top = min([fg, bg], key=lambda g: rank.get(g["verdict"], 9))
tag = "" if top["verdict"] == "STAND_DOWN" else f" [{top.get('system','fade').upper()}]"
check("subject ชี้ระบบ breakout", tag.strip() == "[BREAKOUT]", f"ได้ {tag!r}")

print("\n=== 4. conflicts_with จับเคสขัดกันได้ ===")
fake_f = dict(fg, verdict="ARMED")
check("ARMED พร้อมกัน -> เตือน", breakout.conflicts_with(fake_f, bg) is not None)
check("ไม่ ARMED พร้อมกัน -> เงียบ", breakout.conflicts_with(fg, bg) is None)

print("\n=== 5. gate.render_text ใช้กับทั้งสองระบบ ===")
for g in (fg, bg):
    try:
        t = gate.render_text(g)
        check(f"render_text {g['system']}", isinstance(t, str) and len(t) > 20)
    except Exception as e:
        check(f"render_text {g['system']}", False, f"{type(e).__name__}: {e}")

print("\n=== 6. กำแพงกลับหัว: ทั้งสองระบบต้องไม่ตีเป็น data issue ===")
s2 = snap(716.0, 700, 700, 710, 1e9)
f2 = gate.evaluate(s2); b2 = breakout.evaluate(s2)
check("fade ไม่ใช่ data issue", not f2.get("data_issue"), f"ได้ {f2.get('data_issue')}")
check("breakout ไม่ใช่ data issue", not b2.get("data_issue"), f"ได้ {b2.get('data_issue')}")

print("\n=== 7. กำแพงหายจริง -> ต้องเป็น data issue ทั้งคู่ ===")
s3 = snap(716.0, None, None, 710, 1e9)
f3 = gate.evaluate(s3); b3 = breakout.evaluate(s3)
check("fade เป็น data issue", f3.get("data_issue") is True, f"ได้ {f3.get('data_issue')}")
check("breakout เป็น data issue", b3.get("data_issue") is True, f"ได้ {b3.get('data_issue')}")

print("\n" + "=" * 70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL)) + ' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
