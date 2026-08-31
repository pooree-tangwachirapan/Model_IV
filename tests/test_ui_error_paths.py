"""
ตรวจว่าแท็บ UI ทนกับ "snapshot ที่ล้มเหลว" ได้ — ไม่ใช่แค่ทางที่ทุกอย่างปกติ

ทำไมชุดนี้ถึงมี ทั้งที่ HANDOFF §7 เขียนว่าไม่คุ้มจะเทสต์ UI:
  31 ส.ค. 2026 เจอบั๊กที่ `cockpit_tab.py` อ่าน `snap['asof']` ก่อนเช็ค error
  → KeyError → **Streamlit หยุดทั้ง script** → ทุกแท็บที่อยู่หลังแท็บนั้นไม่ถูก render เลยสักแท็บ
  นั่นแปลว่าบั๊กในแท็บเดียวทำแอปล่มทั้งใบ ซึ่งเปลี่ยนการประเมินความคุ้มไปเลย

ขอบเขต: เทสต์เฉพาะ **ทางที่ข้อมูลพัง** เท่านั้น ไม่ได้เทสต์ว่าหน้าจอแสดงอะไรถูกไหม
  (อันนั้นยังไม่คุ้มเหมือนเดิม) และไม่แตะเน็ต — ป้อน error snapshot เข้า session_state ตรง ๆ

ใช้ streamlit.testing.v1.AppTest ซึ่งมากับ streamlit อยู่แล้ว ไม่ได้เพิ่ม dependency
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


# ใช้ from_string ไม่ใช่ from_function — from_function ต้องอ่าน source จากไฟล์จริงผ่าน inspect
# ซึ่งใช้กับฟังก์ชันที่ประกอบขึ้นตอน runtime ไม่ได้ (OSError: could not get source code)
# ส่วน from_string รับสคริปต์เป็นข้อความตรง ๆ จึงแทนค่าพารามิเตอร์เข้าไปได้เลย
def _run(tab_module: str, render_fn: str, root: str, snap: dict):
    script = f"""
import sys
sys.path.insert(0, {root!r})
import importlib
import streamlit as st

st.session_state.setdefault("dash_snap", {snap!r})
st.session_state.setdefault("dash_snap_sym", "QQQ")

mod = importlib.import_module({tab_module!r})
getattr(mod, {render_fn!r})(sym="QQQ", name="Nasdaq 100 ETF")
"""
    return AppTest.from_string(script, default_timeout=90).run()


ERR_SNAP = {"error": "แปลง chain ไม่ได้ (ไม่มี OI/gamma)", "symbol": "QQQ"}

print("=== แท็บต้องไม่ล่มเมื่อ snapshot เป็น error (ไม่มี key 'asof') ===")
# เคสจริง: CBOE CDN ตอบ 200 พร้อม chain ว่าง (HANDOFF §6.8) แล้ว @st.cache_data แช่ไว้ 5 นาที
for mod, fn in (("cockpit_tab", "render_cockpit_tab"),
                ("dashboard", "render_dashboard_tab"),
                ("long_tab", "render_long_tab")):
    at = _run(mod, fn, ROOT, ERR_SNAP)
    exc = [e.value for e in at.exception]
    check(f"{mod}.{fn} ไม่โยน exception", not exc, f"ได้ {exc}")
    shown = " | ".join([e.value for e in at.error])
    check(f"{mod} แสดงข้อความ error ให้ผู้ใช้เห็น", "chain" in shown or "ข้อมูล" in shown,
          f"error box: {shown!r}")

print("\n=== ไม่มี snapshot เลย (session ว่าง) ก็ต้องไม่ล่ม ===")
# long_tab ทางนี้ต้องไม่ยิง CBOE เอง — บอกให้ไปกดแท็บ Dashboard แทน
at = _run("long_tab", "render_long_tab", ROOT, {"error": "ไม่มีข้อมูล", "symbol": "QQQ"})
check("long_tab ไม่ล่มเมื่อไม่มี chain ใน session", not [e.value for e in at.exception],
      f"ได้ {[e.value for e in at.exception]}")

print("\n" + "="*70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL))+' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
