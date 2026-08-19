"""
run_tests.py — รันชุดทดสอบทั้งหมดใน tests/ แล้วสรุปผลรวม

    python run_tests.py            # รันทุกชุด
    python run_tests.py snapshot   # รันเฉพาะชุดที่ชื่อมีคำนี้

exit 1 ถ้ามีเคสไหนไม่ผ่าน — ใช้ใน CI ได้ตรง ๆ

ทำไมไม่ใช้ pytest: ชุดทดสอบเหล่านี้เขียนแบบ print + check() มาตั้งแต่ต้น
อ่านผลง่ายเวลาไล่บั๊กกับข้อมูลจริง และไม่ต้องเพิ่ม dependency ให้ workflow
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "tests")


def main() -> int:
    if not os.path.isdir(TESTS):
        print("ไม่พบโฟลเดอร์ tests/", file=sys.stderr)
        return 1

    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(f for f in os.listdir(TESTS)
                   if f.startswith("test_") and f.endswith(".py") and pick in f)
    if not files:
        print(f"ไม่มีชุดทดสอบที่ตรงกับ {pick!r}", file=sys.stderr)
        return 1

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    total = passed = failed = 0
    broken: list[str] = []
    print("=" * 74)

    for f in files:
        r = subprocess.run([sys.executable, os.path.join(TESTS, f)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env)
        out = r.stdout or ""
        n_pass = sum(1 for l in out.splitlines() if l.strip().startswith("PASS"))
        n_fail = sum(1 for l in out.splitlines() if l.strip().startswith("FAIL"))
        total += n_pass + n_fail
        passed += n_pass
        failed += n_fail

        if r.returncode != 0 and not n_fail:
            # ตายก่อนจะ check อะไรเลย (import พัง / เน็ตล่ม) — ต้องแยกจาก "เคสไม่ผ่าน"
            broken.append(f)
            print(f"  ERROR  {f:<34} จบด้วย exit {r.returncode}")
            tail = (r.stderr or out).strip().splitlines()[-3:]
            for line in tail:
                print(f"         {line[:100]}")
        else:
            mark = "ok  " if not n_fail else "FAIL"
            print(f"  {mark}   {f:<34} {n_pass:>3} ผ่าน" +
                  (f" · {n_fail} ไม่ผ่าน" if n_fail else ""))
            if n_fail:
                for line in out.splitlines():
                    if line.strip().startswith("FAIL"):
                        print(f"         {line.strip()[:100]}")

    print("=" * 74)
    print(f"  รวม {total} เคส · ผ่าน {passed} · ไม่ผ่าน {failed}"
          + (f" · ชุดที่รันไม่จบ {len(broken)}" if broken else ""))
    return 0 if (failed == 0 and not broken) else 1


if __name__ == "__main__":
    raise SystemExit(main())
