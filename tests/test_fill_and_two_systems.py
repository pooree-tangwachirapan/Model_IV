"""ตรวจบั๊กรอบที่ 2 — เน้น fill logic + สองระบบที่เพิ่งเพิ่ม"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง
import forward_test as ft, gate, breakout
from datetime import datetime, timezone, timedelta

def _try(fn):
    """คืน True ถ้า fn raise ValueError ตามที่คาด"""
    try:
        fn(); return False
    except ValueError:
        return True
    except Exception:
        return False


FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)

UTC = timezone.utc
T0 = datetime(2026, 8, 12, 14, 0, tzinfo=UTC)   # พุธ
def bar(mins, o, h, l, c): return (T0 + timedelta(minutes=mins), o, h, l, c)

def mk(system="fade", side="LONG", entry=100.0, inval=99.0, target=105.0):
    return dict(id=f"T-{system}", system=system, symbol="QQQ", date="2026-08-12",
                opened=T0.isoformat(), side=side, entry=entry, invalidation=inval,
                target=target, plan_r=abs(target-entry)/abs(entry-inval), qty=100,
                risk_usd=abs(entry-inval)*100, spot_at_signal=entry, status="open",
                filled_at=None, fill_price=None, closed_date=None, closed_at=None,
                exit=None, realized_r=None, pnl_usd=None, note="")

LATER = T0 + timedelta(days=10)

print("=== A. fill: limit ต้องแตะ entry ก่อนถึงจะนับผล ===")
tr=[mk()]
# ราคาวิ่งขึ้นไปแตะ target โดยไม่เคยลงมาแตะ entry 100 เลย (เปิด 101 ขึ้นตลอด)
bars=[bar(5,101,106,100.5,105.5)]
ft.resolve(tr, bars_fn=lambda *a,**k: bars, now=LATER)
check("limit ไม่เคยแตะ entry -> no_fill ไม่ใช่ win",
      tr[0]["status"]=="no_fill", f"ได้ {tr[0]['status']} pnl={tr[0]['pnl_usd']}")

print("\n=== B. fill: limit แตะ entry แล้วค่อยนับ ===")
tr=[mk()]
bars=[bar(5,101,101.5,99.8,100.2), bar(10,100.2,106,100,105.5)]
ft.resolve(tr, bars_fn=lambda *a,**k: bars, now=LATER)
check("แตะ entry แล้วถึง target -> win", tr[0]["status"]=="win", f"ได้ {tr[0]['status']}")
check("fill_price = entry (limit)", tr[0]["fill_price"]==100.0, f"ได้ {tr[0]['fill_price']}")
check("pnl = (105-100)*100 = 500", abs(tr[0]["pnl_usd"]-500)<.01, f"ได้ {tr[0]['pnl_usd']}")

print("\n=== C. fill: market เข้าที่ราคาเปิดแท่งถัดไป (มี slippage) ===")
tr=[mk(system="breakout", entry=100.0)]
bars=[bar(5,100.8,106,100.5,105.5)]     # เปิด 100.8 = ไล่แพงกว่าที่เห็น 0.8
ft.resolve(tr, bars_fn=lambda *a,**k: bars, now=LATER)
check("market fill ที่ open ของแท่งถัดไป", tr[0]["fill_price"]==100.8, f"ได้ {tr[0]['fill_price']}")
check("pnl นับจาก fill จริง = (105-100.8)*100 = 420",
      abs(tr[0]["pnl_usd"]-420)<.01, f"ได้ {tr[0]['pnl_usd']} (ถ้า 500 = ซ่อน slippage)")
check("R นับจาก fill จริงด้วย = 4.2/1.8",
      abs(tr[0]["realized_r"]-(4.2/1.8))<.01, f"ได้ {tr[0]['realized_r']}")

print("\n=== D. แท่งที่ fill แตะ stop ในแท่งเดียวกัน -> ต้องเป็นแพ้ ===")
tr=[mk()]
bars=[bar(5,101,101.2,98.5,99.0)]        # ลงมาแตะทั้ง entry 100 และ stop 99
ft.resolve(tr, bars_fn=lambda *a,**k: bars, now=LATER)
check("fill แล้วโดน stop ในแท่งเดียวกัน -> loss", tr[0]["status"]=="loss", f"ได้ {tr[0]['status']}")

print("\n=== E. ยังไม่หมดเวลาและยังไม่ fill -> ต้องคาไว้ ไม่ใช่ no_fill ===")
tr=[mk()]
bars=[bar(5,101,101.5,100.6,101.2)]
ft.resolve(tr, bars_fn=lambda *a,**k: bars, now=T0+timedelta(hours=1))
check("ยังไม่หมดเวลา -> คง open", tr[0]["status"]=="open", f"ได้ {tr[0]['status']}")

print("\n=== F. SHORT: market fill + เครื่องหมาย ===")
tr=[mk(system="breakout", side="SHORT", entry=100.0, inval=101.0, target=95.0)]
bars=[bar(5,99.5,99.8,94.0,95.0)]        # เปิด 99.5 = ไล่ short ได้แย่กว่าที่เห็น
ft.resolve(tr, bars_fn=lambda *a,**k: bars, now=LATER)
check("SHORT market fill = 99.5", tr[0]["fill_price"]==99.5, f"ได้ {tr[0]['fill_price']}")
check("SHORT pnl = (99.5-95)*100 = 450", abs(tr[0]["pnl_usd"]-450)<.01, f"ได้ {tr[0]['pnl_usd']}")

print("\n=== G. no_fill ต้องไม่เข้าสถิติแพ้ชนะ แต่ต้องนับให้เห็น ===")
nf = mk(); nf.update(status="no_fill", realized_r=None, pnl_usd=None)
wn = mk(); wn.update(status="win", realized_r=2.0, pnl_usd=200.0, closed_at=T0.isoformat(), closed_date="2026-08-12")
s = ft.stats([nf, wn])
check("n_closed นับเฉพาะที่ fill", s["n_closed"]==1, f"ได้ {s['n_closed']}")
check("n_no_fill นับแยก", s["n_no_fill"]==1, f"ได้ {s['n_no_fill']}")
check("win_rate ไม่ถูกเจือจางด้วย no_fill", abs(s["win_rate"]-1.0)<.001, f"ได้ {s['win_rate']}")
check("fill_rate = 1/2", abs(s["fill_rate"]-0.5)<.001, f"ได้ {s['fill_rate']}")
check("render_text ไม่ crash", isinstance(ft.render_text(s), str))

print("\n=== H. สองระบบต้องใช้ ledger คนละไฟล์ ===")
check("ledger ไม่ชนกัน", ft.ledger_path("fade") != ft.ledger_path("breakout"),
      f"{ft.ledger_path('fade')} vs {ft.ledger_path('breakout')}")
check("ระบบมั่ว -> ValueError", _try(lambda: ft.system_cfg("xxx")))

print("\n=== I. record: ลิมิตรายวันแยกกันต่อ ledger ===")
g = {"verdict":"ARMED","symbol":"QQQ",
     "plan":{"plan_r":3.0,"side":"LONG","ideal_entry":100.0,
             "invalidation":99.0,"target":103.0,"spot":100.0}}
a=[]; b=[]
for i in range(3):
    x=ft.record(a, g, now=T0+timedelta(seconds=i), system="fade")
    if x: x["status"]="win"
    y=ft.record(b, g, now=T0+timedelta(seconds=i), system="breakout")
    if y: y["status"]="win"
check("fade เปิดได้ 2 ไม้", len(a)==2, f"ได้ {len(a)}")
check("breakout เปิดได้ 2 ไม้ แยกกัน", len(b)==2, f"ได้ {len(b)}")
check("บันทึกชื่อระบบลงไม้", a[0]["system"]=="fade" and b[0]["system"]=="breakout")

print("\n=== J. ไม้เก่าที่ไม่มี field system ต้องยังทำงาน ===")
old = mk(); del old["system"]; del old["fill_price"]; del old["filled_at"]
tr=[old]
bars=[bar(5,101,101.5,99.8,100.2), bar(10,100.2,106,100,105.5)]
try:
    ft.resolve(tr, bars_fn=lambda *a,**k: bars, now=LATER)
    check("ไม้เก่าไม่มี system -> ใช้ default (limit) ได้", tr[0]["status"]=="win", f"ได้ {tr[0]['status']}")
except Exception as e:
    check("ไม้เก่าไม่มี system", False, f"{type(e).__name__}: {e}")

print("\n=== K. breakout.evaluate ให้ plan ที่ record ใช้ได้ ===")
snap = {'symbol':'QQQ','spot':731.5,'asof':datetime.now(),'max_pain':None,
        'levels':{'put_wall':700.0,'call_wall':730.0,'flip':720.0,'net':-2e9,
                  'call_wall_oi':730.0,'put_wall_oi':700.0},
        'expected_move':{'em':5.0},'term':{'slope_pct':-5.0,'state':'backwardation'},
        'rows':[{'label':'0DTE share','value':'10.0% of |GEX|'}]}
bg = breakout.evaluate(snap)
check("breakout ARMED ได้", bg["verdict"]=="ARMED", f"ได้ {bg['verdict']} — {bg['reason']}")
need = ("side","ideal_entry","invalidation","target","spot")
check("plan มี key ครบตามที่ record ต้องการ",
      all(k in (bg.get("plan") or {}) for k in need),
      f"ขาด {[k for k in need if k not in (bg.get('plan') or {})]}")
tr=[]
got = ft.record(tr, bg, now=T0, system="breakout")
check("record รับ plan ของ breakout ได้", got is not None and tr[0]["system"]=="breakout")

print("\n" + "="*70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL))+' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")


def _try(fn):
    try:
        fn(); return False
    except ValueError:
        return True
    except Exception:
        return False
