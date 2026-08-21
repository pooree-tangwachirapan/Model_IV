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

DATA_DIR = "forward_test"
LEDGERS = [cfg["ledger"] for cfg in ft.SYSTEMS.values()]


def _rel(path: str) -> str:
    return os.path.relpath(path, DATA_DIR).replace("\\", "/")


def _data_files() -> list[str]:
    """ไฟล์ข้อมูลทั้งหมดที่ workflow อาจเขียน — ledger ทุกระบบ + log ทุกเดือน"""
    out = [p for p in LEDGERS if os.path.exists(p)]
    if os.path.isdir(pr.LOG_DIR):
        out += [os.path.join(pr.LOG_DIR, f)
                for f in sorted(os.listdir(pr.LOG_DIR)) if f.endswith(".jsonl")]
    return out


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
            repo_path = os.path.join(DATA_DIR, rel)
            os.makedirs(os.path.dirname(repo_path) or ".", exist_ok=True)
            try:
                if f.endswith(".jsonl"):
                    added = _merge_log(mine_path, repo_path)
                elif f.endswith(".json"):
                    added = _merge_ledger(mine_path, repo_path)
                else:
                    continue
            except Exception as e:                       # noqa: BLE001
                # รวมไฟล์เดียวพังต้องไม่ทำให้ไฟล์อื่นหายไปด้วย — ก็อปทับตรง ๆ ดีกว่าไม่ได้อะไรเลย
                print(f"  !! รวม {rel} ไม่สำเร็จ ({type(e).__name__}: {e}) — ใช้ของเราทับ",
                      file=sys.stderr)
                shutil.copy2(mine_path, repo_path)
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
