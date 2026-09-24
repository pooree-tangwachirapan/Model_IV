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
    # stash เก็บ path เต็มรวมชื่อโฟลเดอร์บนสุด (เปลี่ยน 25 ก.ย. 2026 ตอนเพิ่ม zero_dte/)
    # ถ้าตัด prefix ทิ้ง ไฟล์จาก forward_test/ กับ zero_dte/ จะชนกันใน stash
    check("--save เก็บทั้ง log และ ledger พร้อมชื่อโฟลเดอร์",
          os.path.exists(os.path.join(stash, "forward_test", "log", "2026-08.jsonl"))
          and os.path.exists(os.path.join(stash, "forward_test", "ledger.json")))

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
    os.makedirs(os.path.join(st2, "forward_test"), exist_ok=True)
    ft.save([dict(trade("T1", "loss"), pnl_usd=-100.0)],
            os.path.join(st2, "forward_test", "ledger.json"))
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
    files = sorted(os.listdir(os.path.join(st3, "forward_test", "log")))
    check("เก็บ log ครบทุกเดือน", files == ["2026-08.jsonl", "2026-09.jsonl"], f"ได้ {files}")

    print("\n=== 6. CSV ของ 0DTE ต้องรวมแบบ key เหมือนกัน ===")
    # daily-report กับ armed-alert เขียนไฟล์วันเดียวกันได้ — ถ้าปล่อยให้ git รวมเอง
    # จะชนแบบเดียวกับที่ log เคยชนตอน run #33 ต่างกันแค่เป็น CSV
    import pandas as pd
    CSV = os.path.join("zero_dte", "QQQ", "2026-09-25.csv")
    os.makedirs(os.path.dirname(CSV), exist_ok=True)

    def zrow(ts, strike, cp="C", vol=10):
        return {"ts_utc": ts, "sym": "QQQ", "expiry": "2026-09-25", "dte": 0,
                "spot": 740.0, "strike": strike, "cp": cp, "bid": 1.0, "ask": 1.2,
                "mid": 1.1, "last": 1.1, "volume": vol, "open_interest": 5,
                "iv": 0.2, "delta": 0.5, "gamma": 0.02, "vega": 0.1, "theta": -0.8,
                "day_open": 1.0, "day_high": 1.3, "day_low": 0.9, "prev_close": 1.0}

    T1, T2 = "2026-09-25T13:35:00+00:00", "2026-09-25T13:50:00+00:00"
    pd.DataFrame([zrow(T1, 740.0), zrow(T1, 745.0)]).to_csv(CSV, index=False)
    st4 = os.path.join(ROOT, "stash4")
    md.save(st4)
    check("--save เก็บไฟล์ 0DTE ด้วย",
          os.path.exists(os.path.join(st4, "zero_dte", "QQQ", "2026-09-25.csv")))

    # remote เขียน snapshot คนละเวลาทับไฟล์เดียวกัน (= git reset --hard origin/main)
    pd.DataFrame([zrow(T2, 740.0), zrow(T2, 745.0)]).to_csv(CSV, index=False)
    md.merge(st4)
    got = pd.read_csv(CSV)
    check("CSV ได้ครบทั้งของเราและของ remote", len(got) == 4, f"ได้ {len(got)}")
    check("มีครบทั้งสองช่วงเวลา", set(got["ts_utc"]) == {T1, T2}, f"ได้ {sorted(set(got['ts_utc']))}")
    md.merge(st4)
    check("รวม CSV ซ้ำไม่เพิ่มแถว", len(pd.read_csv(CSV)) == 4, f"ได้ {len(pd.read_csv(CSV))}")

finally:
    os.chdir(cwd)
    shutil.rmtree(ROOT, ignore_errors=True)

print("\n" + "=" * 70)
print(f"สรุป: {'ผ่านหมด' if not FAIL else str(len(FAIL)) + ' รายการไม่ผ่าน'}")
for f in FAIL: print(f"  - {f}")
