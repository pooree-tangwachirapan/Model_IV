"""ตรวจบั๊ก predictions.py — เน้นการเขียนพร้อมกัน / ไฟล์เสีย / dedupe"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง
import predictions as pr
import gate, breakout
from datetime import datetime, timezone, timedelta

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

TMP = tempfile.mkdtemp()
P = os.path.join(TMP, "t.jsonl")
T0 = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)

def rec(ts, sym="QQQ", sysname="fade", v="STAND_DOWN"):
    return {"ts": ts.isoformat(timespec="seconds"), "sym": sym, "sys": sysname,
            "v": v, "data_issue": False, "reason": "x", "spot": 700.0,
            "lv": {}, "gates": {"A": True, "B": v == "ARMED"}, "soft": {}, "plan": None}

print("=== 1. append + dedupe ===")
n1 = pr.append([rec(T0), rec(T0, sysname="breakout")], P)
check("เขียนครั้งแรก 2 record", n1 == 2, f"ได้ {n1}")
n2 = pr.append([rec(T0), rec(T0, sysname="breakout")], P)
check("เขียนซ้ำ record เดิม -> 0", n2 == 0, f"ได้ {n2}")
check("ไฟล์มี 2 บรรทัด", len(pr.load(P)) == 2, f"ได้ {len(pr.load(P))}")

print("\n=== 2. เขียนพร้อมกัน: อีกฝั่งเพิ่ม record ระหว่างที่เราคำนวณ ===")
# จำลอง: process A อ่านไฟล์ได้ 2 บรรทัด แล้ว process B เขียนเพิ่ม จากนั้น A เขียน
pr.append([rec(T0 + timedelta(minutes=15))], P)          # B เขียนแทรก
before = len(pr.load(P))
pr.append([rec(T0 + timedelta(minutes=30))], P)          # A เขียนต่อ
after = pr.load(P)
check("record ของ B ไม่หาย", len(after) == before + 1, f"{before} -> {len(after)}")
ts_set = {r["ts"] for r in after}
check("มีครบทุก timestamp", len(ts_set) == 3, f"ได้ {sorted(ts_set)}")

print("\n=== 3. ไฟล์มีบรรทัดเสีย -> ต้องข้ามไป ไม่ตายทั้งไฟล์ ===")
with open(P, "a", encoding="utf-8") as f:
    f.write("{ไม่ใช่ json เลย\n")
    f.write("\n")
loaded = pr.load(P)
check("อ่านได้ ข้ามบรรทัดเสีย", len(loaded) == 4, f"ได้ {len(loaded)}")
n3 = pr.append([rec(T0 + timedelta(minutes=45))], P)
check("เขียนต่อได้หลังเจอบรรทัดเสีย", n3 == 1, f"ได้ {n3}")
check("บรรทัดเสียถูกทิ้งตอนเขียนใหม่", len(pr.load(P)) == 5, f"ได้ {len(pr.load(P))}")

print("\n=== 4. ไฟล์ไม่มี / โฟลเดอร์ไม่มี ===")
P2 = os.path.join(TMP, "sub", "dir", "new.jsonl")
check("อ่านไฟล์ที่ไม่มี -> []", pr.load(P2) == [])
check("เขียนไฟล์ในโฟลเดอร์ที่ยังไม่มี", pr.append([rec(T0)], P2) == 1)
check("อ่านกลับได้", len(pr.load(P2)) == 1)

print("\n=== 5. to_record จาก gate/breakout จริง ===")
snap = {'symbol':'QQQ','spot':731.5,'asof':datetime.now(),'max_pain':729.0,'atm_iv':23.7,
        'levels':{'put_wall':700.0,'call_wall':730.0,'flip':720.0,'net':-2e9,
                  'call_wall_oi':700.0,'put_wall_oi':660.0},
        'expected_move':{'em':5.0},'term':{'slope_pct':-5.0,'state':'backwardation'},
        'rows':[{'label':'0DTE share','value':'10.0% of |GEX|'}]}
bg = breakout.evaluate(snap); bg.setdefault("system", "breakout")
fg = gate.evaluate(snap);     fg.setdefault("system", "fade")
for g in (fg, bg):
    r = pr.to_record(g, snap, T0)
    check(f"to_record {g['system']} มี key ครบ",
          all(k in r for k in ("ts","sym","sys","v","spot","lv","gates","plan")),
          f"ขาด {[k for k in ('ts','sym','sys','v','spot','lv','gates','plan') if k not in r]}")
    check(f"to_record {g['system']} serialize เป็น JSON ได้",
          isinstance(json.dumps(r, ensure_ascii=False), str))
r_b = pr.to_record(bg, snap, T0)
check("breakout ARMED เก็บแผนไว้ด้วย", r_b["plan"] is not None and r_b["plan"]["r"], f"ได้ {r_b['plan']}")
r_f = pr.to_record(fg, snap, T0)
check("fade STAND_DOWN ก็ยังเก็บผลรายประตู", len(r_f["gates"]) >= 2, f"ได้ {r_f['gates']}")

print("\n=== 6. summarize: นับตัวปิดถูกมั้ย ===")
rs = [rec(T0, v="STAND_DOWN"), rec(T0+timedelta(minutes=15), v="ARMED"),
      rec(T0+timedelta(minutes=30), v="STAND_DOWN")]
s = pr.summarize(rs, system="fade")
check("นับจำนวนถูก", s["n"] == 3, f"ได้ {s['n']}")
check("นับ ARMED ถูก", s["verdicts"].get("ARMED") == 1, f"ได้ {s['verdicts']}")
check("armed_rate ถูก", abs(s["armed_rate"] - 1/3) < 1e-9, f"ได้ {s['armed_rate']}")
check("ประตู B เป็นตัวปิด 2 ครั้ง", s["blockers"].get("B") == 2, f"ได้ {s['blockers']}")
check("ประตู A ไม่เคยปิด", "A" not in s["blockers"], f"ได้ {s['blockers']}")
check("gate_seen นับครบ", s["gate_seen"].get("A") == 3, f"ได้ {s['gate_seen']}")
check("render_text ไม่ crash", isinstance(pr.render_text(s), str))
check("summarize ลิสต์ว่าง ไม่ crash", pr.summarize([])["n"] == 0)
check("render_text ลิสต์ว่าง ไม่ crash", isinstance(pr.render_text(pr.summarize([])), str))

print("\n=== 7. กรองตามระบบ/สัญลักษณ์ ===")
mixed = [rec(T0, sysname="fade"), rec(T0, sysname="breakout"),
         rec(T0, sym="SPY", sysname="fade")]
check("กรอง system", pr.summarize(mixed, system="fade")["n"] == 2)
check("กรอง symbol", pr.summarize(mixed, symbol="SPY")["n"] == 1)
check("กรองทั้งคู่", pr.summarize(mixed, system="fade", symbol="QQQ")["n"] == 1)

print("\n=== 8. atomic write: ไม่เหลือไฟล์ .tmp ค้าง ===")
check("ไม่มี .tmp ค้าง", not any(f.endswith(".tmp") for f in os.listdir(TMP)),
      f"เจอ {[f for f in os.listdir(TMP) if f.endswith('.tmp')]}")

shutil.rmtree(TMP, ignore_errors=True)
print("\n" + "=" * 70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL)) + ' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
