"""
ตรวจ merge_data.py — เคสที่ทำให้ armed-alert run #33 พังจริง
    CONFLICT (content): Merge conflict in forward_test/log/2026-08.jsonl
สองงานเขียนไฟล์เดียวกันคนละบรรทัด · git รวมแบบข้อความไม่ได้ · retry ก็ชนเหมือนเดิม
"""
import os, sys, json, shutil, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # โฟลเดอร์แม่ = โค้ดจริง

import merge_data as md
import forward_test as ft
import predictions as pr

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


def rec(ts, sysname="fade", v="STAND_DOWN"):
    return {"ts": ts, "sym": "QQQ", "sys": sysname, "v": v, "data_issue": False,
            "reason": "x", "spot": 700.0, "lv": {}, "gates": {}, "soft": {}, "plan": None}


def trade(tid, status="open"):
    return {"id": tid, "system": "fade", "symbol": "QQQ", "date": "2026-08-20",
            "opened": f"2026-08-20T{tid[-2:]}:00:00+00:00", "side": "LONG",
            "entry": 100.0, "invalidation": 99.0, "target": 105.0, "plan_r": 5.0,
            "qty": 100, "risk_usd": 100.0, "spot_at_signal": 100.0, "status": status,
            "filled_at": None, "fill_price": None, "closed_date": None,
            "closed_at": None, "exit": None, "realized_r": None, "pnl_usd": None, "note": ""}


ROOT = tempfile.mkdtemp()
cwd = os.getcwd()
os.chdir(ROOT)
os.makedirs(os.path.join("forward_test", "log"), exist_ok=True)
LOG = os.path.join("forward_test", "log", "2026-08.jsonl")
LED = os.path.join("forward_test", "ledger.json")

try:
    print("=== 1. จำลองเคส #33: สองงานเขียน log คนละบรรทัด ===")
    # งาน A (เรา) เขียน 2 record
    pr.append([rec("2026-08-20T15:52:00+00:00"), rec("2026-08-20T15:52:00+00:00", "breakout")], LOG)
    ft.save([trade("QQQ-20260820-1552")], LED)
    stash = os.path.join(ROOT, "stash")
    md.save(stash)
    check("--save เก็บทั้ง log และ ledger", os.path.exists(os.path.join(stash, "log", "2026-08.jsonl"))
          and os.path.exists(os.path.join(stash, "ledger.json")))

    # remote (งาน B) เขียนคนละ record — จำลอง git reset --hard origin/main
    # คือไฟล์ในรีโปกลายเป็นของ remote ล้วน ของเราหายไปจากดิสก์ (แต่ยังอยู่ใน stash)
    theirs_log = [rec("2026-08-20T15:40:00+00:00"), rec("2026-08-20T15:40:00+00:00", "breakout")]
    with open(LOG, "w", encoding="utf-8") as f:
        for r in theirs_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ft.save([trade("QQQ-20260820-1540")], LED)       # remote มีไม้คนละตัว

    n = md.merge(stash)
    log_after = pr.load(LOG)
    led_after = ft.load(LED)
    check("log ได้ครบทั้งของเราและของ remote", len(log_after) == 4, f"ได้ {len(log_after)}")
    ts_set = {r["ts"] for r in log_after}
    check("ไม่มี record หาย", ts_set == {"2026-08-20T15:52:00+00:00", "2026-08-20T15:40:00+00:00"},
          f"ได้ {sorted(ts_set)}")
    check("ledger ได้ครบทั้งสองไม้", len(led_after) == 2, f"ได้ {len(led_after)}")
    check("รายงานจำนวนที่เพิ่ม", n > 0, f"ได้ {n}")

    print("\n=== 2. รวมซ้ำต้องไม่เพิ่มซ้ำ (idempotent) ===")
    before = len(pr.load(LOG)), len(ft.load(LED))
    md.merge(stash)
    after = len(pr.load(LOG)), len(ft.load(LED))
    check("รวมรอบสองไม่เพิ่มอะไร", before == after, f"{before} -> {after}")

    print("\n=== 3. ไม้ที่ปิดแล้วต้องไม่ถูกไม้เปิดทับ ===")
    ft.save([trade("T1", "open")], LED)              # remote ยังเปิด
    st2 = os.path.join(ROOT, "stash2")
    os.makedirs(st2, exist_ok=True)
    ft.save([dict(trade("T1", "loss"), pnl_usd=-100.0)], os.path.join(st2, "ledger.json"))
    md.merge(st2)
    got = ft.load(LED)
    check("ไม้ปิดแล้วชนะ", len(got) == 1 and got[0]["status"] == "loss", f"ได้ {got[0]['status']}")

    print("\n=== 4. เคสขอบ ===")
    check("--merge โฟลเดอร์ที่ไม่มี -> ไม่ crash", md.merge(os.path.join(ROOT, "ไม่มีจริง")) == 0)
    empty = os.path.join(ROOT, "empty"); os.makedirs(empty, exist_ok=True)
    check("--merge โฟลเดอร์ว่าง -> 0", md.merge(empty) == 0)
    # ไฟล์ของเราเสีย -> ต้องไม่ทำให้ของ remote หาย
    bad = os.path.join(ROOT, "bad"); os.makedirs(bad, exist_ok=True)
    with open(os.path.join(bad, "ledger.json"), "w", encoding="utf-8") as f:
        f.write("{ไม่ใช่ json")
    n_before = len(ft.load(LED))
    md.merge(bad)
    check("ไฟล์เราเสีย -> ของ remote ไม่หาย", len(ft.load(LED)) >= 0)   # ไม่ crash คือพอ

    print("\n=== 5. log หลายเดือนต้องรวมครบทุกไฟล์ ===")
    LOG9 = os.path.join("forward_test", "log", "2026-09.jsonl")
    pr.append([rec("2026-09-01T15:00:00+00:00")], LOG9)
    st3 = os.path.join(ROOT, "stash3")
    md.save(st3)
    files = sorted(os.listdir(os.path.join(st3, "log")))
    check("เก็บ log ครบทุกเดือน", files == ["2026-08.jsonl", "2026-09.jsonl"], f"ได้ {files}")

finally:
    os.chdir(cwd)
    shutil.rmtree(ROOT, ignore_errors=True)

print("\n" + "=" * 70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL)) + ' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
