"""ไล่ตรวจบั๊กรอบใหม่ — เน้นเคสขอบที่ข้อมูลจริงยังไม่เคยเจอ"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง
import forward_test as ft, gate, snapshot, fa_gex
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

print("=== 1. forward_test.resolve: ไม้ LONG ที่ target < entry (ไม่ควรเกิด แต่ถ้าเกิด?) ===")
t = dict(id="X", symbol="QQQ", date="2026-08-13", opened="2026-08-13T14:00:00+00:00",
         side="LONG", entry=100.0, invalidation=99.0, target=105.0, plan_r=5.0,
         qty=100, risk_usd=100.0, spot_at_signal=100.0, status="open",
         closed_date=None, closed_at=None, exit=None, realized_r=None, pnl_usd=None, note="")
bars = [(datetime(2026,8,13,14,5,tzinfo=timezone.utc), 99.5, 106.0, 99.5, 105.5)]
tr=[dict(t)]
ft.resolve(tr, bars_fn=lambda *a, **k: bars, now=datetime(2026,8,13,15,tzinfo=timezone.utc))
check("LONG แตะ target -> win, pnl = (105-100)*100 = 500",
      tr[0]["status"]=="win" and abs(tr[0]["pnl_usd"]-500)<0.01, f"ได้ {tr[0]['pnl_usd']}")

print("\n=== 2. แท่งเดียวแตะทั้ง invalidate และ target -> ต้องนับเป็นแพ้ ===")
tr=[dict(t)]
bars2=[(datetime(2026,8,13,14,5,tzinfo=timezone.utc), 98.0, 106.0, 98.0, 105.0)]
ft.resolve(tr, bars_fn=lambda *a,**k: bars2, now=datetime(2026,8,13,15,tzinfo=timezone.utc))
check("แตะทั้งคู่ในแท่งเดียว -> loss", tr[0]["status"]=="loss", f"ได้ {tr[0]['status']}")
check("loss pnl = (99-100)*100 = -100", abs(tr[0]["pnl_usd"]+100)<0.01, f"ได้ {tr[0]['pnl_usd']}")

print("\n=== 3. SHORT: pnl เครื่องหมายถูกมั้ย ===")
s = dict(t, side="SHORT", entry=100.0, invalidation=101.0, target=95.0)
tr=[dict(s)]
bars3=[(datetime(2026,8,13,14,5,tzinfo=timezone.utc), 94.0, 100.5, 94.0, 95.0)]
ft.resolve(tr, bars_fn=lambda *a,**k: bars3, now=datetime(2026,8,13,15,tzinfo=timezone.utc))
check("SHORT แตะ target 95 -> win pnl = (100-95)*100 = 500",
      tr[0]["status"]=="win" and abs(tr[0]["pnl_usd"]-500)<0.01, f"ได้ {tr[0]['pnl_usd']} {tr[0]['status']}")

print("\n=== 4. timeout: ปิดที่ราคาปิดล่าสุด ===")
tr=[dict(t)]
old = dict(t, opened="2026-08-01T14:00:00+00:00")
tr=[old]
bars4=[(datetime(2026,8,1,14,5,tzinfo=timezone.utc), 99.5, 100.5, 99.5, 100.2)]
ft.resolve(tr, bars_fn=lambda *a,**k: bars4, now=datetime(2026,8,10,tzinfo=timezone.utc))
check("เกิน MAX_HOLD_DAYS -> timeout", tr[0]["status"]=="timeout", f"ได้ {tr[0]['status']}")
check("timeout pnl = (100.2-100)*100 = 20", abs(tr[0]["pnl_usd"]-20)<0.01, f"ได้ {tr[0]['pnl_usd']}")

print("\n=== 5. merge: ไม้ที่ปิดแล้วต้องไม่ถูก open ทับ (ทุกลำดับ) ===")
o=dict(id="A",opened="2026-08-12T16:00:00+00:00",status="open")
c=dict(id="A",opened="2026-08-12T16:00:00+00:00",status="loss",pnl_usd=-116.0)
check("merge(open, closed) -> closed", ft.merge([o],[c])[0]["status"]=="loss")
check("merge(closed, open) -> closed", ft.merge([c],[o])[0]["status"]=="loss")
check("merge ไม่ทำไม้หาย", len(ft.merge([dict(id='B',opened='x',status='open')],[c]))==2)

print("\n=== 6. gate.zone_geometry: หารศูนย์เมื่อ call_wall == put_wall ===")
z = gate.zone_geometry(100, 100, 100)
check("wall เท่ากัน -> None ไม่ใช่ ZeroDivisionError", z is None, f"ได้ {z}")

print("\n=== 7. gate.build_plan: EM = 0 ===")
try:
    z2 = gate.zone_geometry(96, 95, 105)
    p = gate.build_plan(96, z2, 95, 105, em=0, max_pain=None)
    check("EM=0 -> คืน None ไม่ crash", p is None, f"ได้ {p}")
except Exception as e:
    check("EM=0 -> ไม่ crash", False, f"{type(e).__name__}: {e}")

print("\n=== 8. stats: ledger ว่าง / มีแต่ไม้เปิด ===")
try:
    s0 = ft.stats([])
    check("ledger ว่าง -> ไม่ crash", s0["n_closed"]==0)
    s1 = ft.stats([dict(t)])
    check("มีแต่ไม้เปิด -> n_closed=0 ไม่หารศูนย์", s1["n_closed"]==0)
    txt = ft.render_text(s1)
    check("render_text ไม้เปิดล้วน -> ไม่ crash", isinstance(txt,str) and len(txt)>0)
except Exception as e:
    check("stats เคสว่าง", False, f"{type(e).__name__}: {e}")

print("\n=== 9. record: plan ไม่มี ideal_entry (schema เก่า) ===")
try:
    g_bad = {"verdict":"ARMED","symbol":"QQQ","plan":{"plan_r":3.0,"side":"LONG"}}
    r = ft.record([], g_bad)
    check("plan ขาด key -> ควรไม่ crash หรือคืน None", r is None, "record คืนค่าแทนที่จะ raise")
except KeyError as e:
    check("plan ขาด key -> ไม่ crash", False, f"KeyError: {e} (record ไม่กัน schema เพี้ยน)")
except Exception as e:
    check("plan ขาด key", False, f"{type(e).__name__}: {e}")

print("\n=== 10. record: ลิมิต 2 ไม้/วัน นับถูกมั้ย ===")
now = datetime(2026,8,13,14,0,tzinfo=timezone.utc)
g_ok = {"verdict":"ARMED","symbol":"QQQ",
        "plan":{"plan_r":3.0,"side":"LONG","ideal_entry":100.0,
                "invalidation":99.0,"target":103.0,"spot":100.0}}
tr=[]
for i in range(4):
    got = ft.record(tr, g_ok, now=now+timedelta(seconds=i))
    if got: got["status"]="win"       # ปิดทันทีเพื่อให้เปิดไม้ถัดไปได้
check("เปิดได้ไม่เกิน 2 ไม้/วัน", len(tr)==2, f"เปิดไป {len(tr)} ไม้")

print("\n" + "="*70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL))+' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
