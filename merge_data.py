"""
merge_data.py — รวมไฟล์ข้อมูลของเราเข้ากับของ remote โดยยึด "key" ไม่ใช่ข้อความในไฟล์

ปัญหาที่แก้ (เกิดจริง 2026-08-20 run #33):
    workflow สอง–สามตัวเขียน forward_test/log/YYYY-MM.jsonl กับ ledger*.json ไฟล์เดียวกัน
    พอ commit ใกล้กัน git rebase ชนที่เนื้อไฟล์:
        CONFLICT (content): Merge conflict in forward_test/log/2026-08.jsonl
    แล้ว retry loop ลองใหม่ 3 รอบ — ชนเหมือนเดิมทั้ง 3 รอบ เพราะ **ไม่ใช่ race**
    การ retry แก้ได้เฉพาะ "มีคน push แทรกตอนเรากำลัง push" ไม่ได้แก้ "เนื้อหาชนกันจริง"

วิธีที่ถูก: อย่าให้ git ตัดสินการรวม เพราะมันรวมแบบบรรทัด
    ไฟล์พวกนี้เป็น "ชุด record ที่มี key" — รวมได้แบบไม่มีทางขัดกัน
        ledger*.json   → key = trade id   · ไม้ที่ปิดแล้วชนะไม้ที่ยังเปิด
        log/*.jsonl    → key = (ts, sym, sys) · ซ้ำก็ทิ้งตัวหลัง
        zero_dte/*.csv → key = (ts_utc, strike, cp) · ซ้ำก็ทิ้งตัวหลัง

ลำดับที่ workflow ต้องทำ:
    python merge_data.py --save  "$TMP"   # เก็บของที่เราเพิ่งเขียน
    git fetch origin && git reset --hard origin/main
    python merge_data.py --merge "$TMP"   # รวมของเราเข้ากับของ remote
    git add forward_test/ && git commit && git push

reset --hard ปลอดภัยเพราะ --save ก็อปไฟล์ออกไปก่อนแล้ว และไฟล์อื่นในรีโปเราไม่ได้แก้
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import forward_test as ft
import predictions as pr

# โฟลเดอร์ข้อมูลทั้งหมดที่ workflow เขียนแล้ว commit กลับ
# **ทุก path ในไฟล์นี้อิง cwd เสมอ ห้ามอิงที่อยู่ของไฟล์ .py**
# เพราะ workflow รันจากรากรีโป และชุดทดสอบ chdir ไปโฟลเดอร์ชั่วคราว
# ถ้าอิงที่อยู่ไฟล์ .py เทสต์จะเขียนทับข้อมูลจริงของรีโป (พลาดมาแล้วตอนเพิ่ม zero_dte)
DATA_DIRS = ["forward_test", "zero_dte"]
DATA_DIR = DATA_DIRS[0]                       # คงชื่อเดิมไว้ให้ของเก่าที่ import ยังใช้ได้
ZDTE_DIR = DATA_DIRS[1]
LEDGERS = [cfg["ledger"] for cfg in ft.SYSTEMS.values()]


def _rel(path: str) -> str:
    """
    path เทียบ cwd — **เก็บชื่อโฟลเดอร์บนสุดไว้ด้วย** (forward_test/… หรือ zero_dte/…)

    ของเดิมตัด `forward_test/` ทิ้งเพราะมีโฟลเดอร์ข้อมูลเดียว พอเพิ่ม zero_dte
    แล้วยังตัดอยู่ ไฟล์สองโฟลเดอร์จะชนกันใน stash และ merge จะวางผิดที่
    """
    if os.path.isabs(path):
        return os.path.relpath(path, os.getcwd()).replace("\\", "/")
    return path.replace("\\", "/")


def _zdte_files() -> list[str]:
    """CSV 0DTE ทุกวัน/ทุกคลัง — ไล่ดูเองจาก cwd ไม่ใช้ z.ZDTE_DIR ซึ่งเป็น path สัมบูรณ์"""
    out: list[str] = []
    if not os.path.isdir(ZDTE_DIR):
        # `zero_dte.py` หาโฟลเดอร์จากที่อยู่ของไฟล์ .py แต่ไฟล์นี้อิง cwd
        # ปกติตรงกันเพราะ workflow รันจากรากรีโป — ถ้าไม่ตรงต้องดังไว้ก่อน
        # ไม่งั้น --save จะเก็บ 0 ไฟล์อย่างเงียบ ๆ แล้วข้อมูลหายตอน reset --hard
        import zero_dte as _z
        if os.path.isdir(_z.ZDTE_DIR) and os.listdir(_z.ZDTE_DIR):
            print(f"  !! มีข้อมูล 0DTE ที่ {_z.ZDTE_DIR} แต่ cwd คือ {os.getcwd()} "
                  "— รันจากรากรีโปเท่านั้น ไม่งั้นข้อมูลจะไม่ถูกเก็บ", file=sys.stderr)
        return out
    for sym in sorted(os.listdir(ZDTE_DIR)):
        d = os.path.join(ZDTE_DIR, sym)
        if not os.path.isdir(d):
            continue
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith(".csv") or f.endswith(".csv.gz")]
    return out


def _data_files() -> list[str]:
    """
    ไฟล์ข้อมูลทั้งหมดที่ workflow อาจเขียน
    — ledger ทุกระบบ + log ทุกเดือน + CSV 0DTE ทุกวัน/ทุกคลัง
    """
    out = [p for p in LEDGERS if os.path.exists(p)]
    if os.path.isdir(pr.LOG_DIR):
        out += [os.path.join(pr.LOG_DIR, f)
                for f in sorted(os.listdir(pr.LOG_DIR)) if f.endswith(".jsonl")]
    return out + _zdte_files()


def save(dest: str) -> int:
    """ก็อปไฟล์ข้อมูลไปเก็บนอกรีโป ก่อนจะ reset --hard ทับ"""
    n = 0
    for p in _data_files():
        target = os.path.join(dest, _rel(p))
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        shutil.copy2(p, target)
        n += 1
        print(f"  เก็บ {p}")
    if not n:
        print("  ไม่มีไฟล์ข้อมูลให้เก็บ")
    return n


def _merge_ledger(mine_path: str, repo_path: str) -> int:
    """รวม ledger — ยึด trade id · ไม้ปิดแล้วชนะไม้เปิด (ตรรกะเดียวกับ ft.merge)"""
    mine = ft.load(mine_path)
    theirs = ft.load(repo_path) if os.path.exists(repo_path) else []
    merged = ft.merge(mine, theirs)
    added = len(merged) - len(theirs)
    ft.save(merged, repo_path)
    return added


def _merge_log(mine_path: str, repo_path: str) -> int:
    """รวม log — pr.append() dedupe ด้วย (ts, sym, sys) ให้อยู่แล้ว"""
    return pr.append(pr.load(mine_path), repo_path)


def _merge_csv(mine_path: str, repo_path: str) -> int:
    """
    รวม CSV ของ 0DTE — key = (ts_utc, strike, cp)

    สองงาน (daily-report กับ armed-alert) เขียนไฟล์วันเดียวกันได้ ถ้าปล่อยให้ git
    รวมเองจะชนแบบเดียวกับที่ log เคยชนตอน run #33 — ต่างกันแค่เป็นไฟล์ CSV
    """
    import pandas as pd

    mine = pd.read_csv(mine_path)
    if mine.empty:
        return 0
    if os.path.exists(repo_path):
        theirs = pd.read_csv(repo_path)
    else:
        theirs = pd.DataFrame(columns=mine.columns)

    before = len(theirs)
    # key ต้องมาจาก **ทั้งสองฝั่ง** ไม่ใช่ฝั่งเราฝ่ายเดียว
    # ถ้าไฟล์เราขาดคอลัมน์ (schema คนละรุ่น) key จะหดเหลือ ["strike","cp"]
    # แล้ว drop_duplicates ไปลบ *ทุก timestamp ของ strike นั้นที่อยู่ในไฟล์ remote*
    # เหลือแถวเดียว = ลบของคนอื่นทิ้งเพราะไฟล์เราผิดรูป
    both = pd.concat([theirs, mine], ignore_index=True)
    # ต้องเป็น **union** ของคอลัมน์ทั้งสองฝั่ง (= คอลัมน์ของ both หลัง concat)
    # ไม่ใช่ intersection: ถ้าฝั่งหนึ่งขาด ts_utc แล้วใช้ intersection
    # key จะหดเหลือ (strike, cp) ซึ่งเป็นบั๊กเดิมเป๊ะ ๆ
    # ด้วย union แถวที่ขาด ts_utc จะได้ NaN ซึ่งไม่ชนกับใคร → ถูกเก็บไว้เป็นแถวแยก
    # เสียพื้นที่นิดหน่อย ดีกว่าลบข้อมูลของ remote ทิ้ง
    key = [c for c in ("day", "ts_utc", "strike", "cp") if c in both.columns]
    if key:
        both = both.drop_duplicates(subset=key)
    else:
        # ไม่มีคอลัมน์ key ร่วมกันเลย = เทียบไม่ได้ ห้ามเดา — เก็บทุกแถวไว้ก่อน
        print(f"  !! {os.path.basename(repo_path)}: ไม่มีคอลัมน์ key ร่วมกัน "
              "— รวมโดยไม่ตัดซ้ำ", file=sys.stderr)
    sort_by = [c for c in ("ts_utc", "cp", "strike") if c in both.columns]
    if sort_by:
        both = both.sort_values(sort_by)

    os.makedirs(os.path.dirname(repo_path) or ".", exist_ok=True)
    both.to_csv(repo_path, index=False, lineterminator="\n",
                compression="gzip" if repo_path.endswith(".gz") else None)
    return len(both) - before


def merge(src: str) -> int:
    """
    รวมของที่ --save ไว้ เข้ากับไฟล์ที่อยู่ในรีโปตอนนี้ (= เวอร์ชันของ remote)
    คืนจำนวน record ที่เพิ่มขึ้นจากฝั่งเรา
    """
    if not os.path.isdir(src):
        print(f"  ไม่พบโฟลเดอร์ที่เก็บไว้: {src} — ข้ามการรวม", file=sys.stderr)
        return 0

    total = 0
    for root, _dirs, files in os.walk(src):
        for f in files:
            mine_path = os.path.join(root, f)
            rel = os.path.relpath(mine_path, src).replace("\\", "/")
            repo_path = rel                       # อิง cwd เหมือนตอน --save
            os.makedirs(os.path.dirname(repo_path) or ".", exist_ok=True)
            try:
                if f.endswith(".jsonl"):
                    added = _merge_log(mine_path, repo_path)
                elif f.endswith(".json"):
                    added = _merge_ledger(mine_path, repo_path)
                elif f.endswith(".csv") or f.endswith(".csv.gz"):
                    added = _merge_csv(mine_path, repo_path)
                else:
                    continue
            except Exception as e:                       # noqa: BLE001
                # ── ห้ามก็อปทับของ remote เมื่อรวมไม่สำเร็จ ──
                # ของเดิมเขียนว่า "ใช้ของเราทับ ดีกว่าไม่ได้อะไรเลย" ซึ่งผิดเมื่อ
                # **ฝั่งที่พังคือฝั่งเรา**: ไฟล์เรา 0 ไบต์ → EmptyDataError → ก็อปทับ
                # → ไฟล์ใน repo เหลือ 0 ไบต์ แล้ว commit ทับของจริงบน origin
                # ของ remote คือความจริงร่วม เสียของเราไฟล์เดียวยังกู้ได้จากรอบหน้า
                # เสียของ remote คือเสียของทุกคน → ทับได้เฉพาะตอน remote ยังไม่มีไฟล์นั้น
                if not os.path.exists(repo_path):
                    print(f"  !! รวม {rel} ไม่สำเร็จ ({type(e).__name__}: {e}) "
                          "— remote ยังไม่มีไฟล์นี้ ใช้ของเราแทน", file=sys.stderr)
                    shutil.copy2(mine_path, repo_path)
                else:
                    print(f"::error::รวม {rel} ไม่สำเร็จ ({type(e).__name__}: {e}) "
                          "— เก็บของ remote ไว้ ข้อมูลรอบนี้ของเราไม่ถูกรวม",
                          file=sys.stderr)
                continue
            total += added
            print(f"  รวม {rel}: +{added} record จากฝั่งเรา")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="DIR", help="ก็อปไฟล์ข้อมูลไปเก็บที่ DIR")
    g.add_argument("--merge", metavar="DIR", help="รวมไฟล์จาก DIR เข้ากับรีโป")
    a = ap.parse_args()
    if a.save:
        save(a.save)
    else:
        merge(a.merge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
