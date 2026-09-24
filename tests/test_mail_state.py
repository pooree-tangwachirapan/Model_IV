"""
ตรวจตัวกันเมลซ้ำ — จำลองลำดับเหตุการณ์จริงที่ทำให้เคยได้เมลวันละ 4 ใบ

เคสที่ต้องไม่กลับมาอีก (พบโดยการรีวิวอิสระ 25 ก.ย. 2026):
    `concurrency: cancel-in-progress: false` **ไม่ได้กันเมลซ้ำ** มันแค่ต่อคิว
    ชั้น cron ที่รอคิวจะเริ่มวิ่งทันทีที่ชั้นก่อนหน้าจบ แล้วเห็นว่าเวลาเป้า
    เพิ่งผ่านไป ~5 นาที ซึ่งยังอยู่ใน LATE_GRACE_MIN=20 → ส่งซ้ำ

    08:17 ชั้น A ได้ไป → นอนรอ → 13:00 ส่ง premarket → 13:04 จบ
    13:05 ชั้น B ที่ค้างคิวเริ่มวิ่ง → premarket สาย 5 นาที → **ส่งซ้ำ**

เวลาอย่างเดียวแยก "ใบซ้ำ" ออกจาก "ชั้นแรกที่เริ่มสายจริง" ไม่ได้
จึงต้องมีหลักฐานว่าส่งไปแล้ว ซึ่งข้ามระหว่าง job ได้
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from datetime import date, timedelta

import mail_state as ms
import market_clock as mc

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


TMP = tempfile.mkdtemp(prefix="mailstate-")
F = os.path.join(TMP, ".mail-sent.json")
TODAY = date(2026, 9, 25)

try:
    print("=== พื้นฐาน ===")
    check("ยังไม่มีไฟล์ → ถือว่ายังไม่เคยส่ง",
          not ms.already_sent("premarket", TODAY, F))
    ms.mark("premarket", TODAY, F)
    check("mark แล้ว → บอกว่าส่งไปแล้ว", ms.already_sent("premarket", TODAY, F))
    check("slot อื่นของวันเดียวกันยังส่งได้",
          not ms.already_sent("open30", TODAY, F))
    check("slot เดียวกันของวันอื่นยังส่งได้",
          not ms.already_sent("premarket", date(2026, 9, 28), F))

    print("\n=== mark ซ้ำต้องไม่บวมและไม่เปลี่ยนคำตอบ ===")
    ms.mark("premarket", TODAY, F)
    ms.mark("premarket", TODAY, F)
    st = ms.load(F)
    check("เก็บ slot เดียวไม่ซ้ำ", st[str(TODAY)] == ["premarket"], str(st))

    print("\n=== จำลองลำดับที่เคยทำให้ได้เมล 4 ใบ ===")
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)
    sent = []
    # ชั้น A: นอนรอถึง 13:00 แล้วส่ง premarket · ต่อด้วย open30 ตอน 14:00
    for slot in ("premarket", "open30"):
        if not ms.already_sent(slot, TODAY, F):
            sent.append((slot, "A")); ms.mark(slot, TODAY, F)
    # ชั้น B: เริ่มวิ่งทันทีที่ A จบ — เวลาบอกว่า "สาย 5 นาที" ซึ่งยังอยู่ใน grace
    for slot in ("premarket", "open30"):
        if not ms.already_sent(slot, TODAY, F):
            sent.append((slot, "B")); ms.mark(slot, TODAY, F)
    # ชั้น C: มาอีกชั้น
    for slot in ("premarket", "open30"):
        if not ms.already_sent(slot, TODAY, F):
            sent.append((slot, "C")); ms.mark(slot, TODAY, F)
    check("ส่งรวม 2 ใบ ไม่ใช่ 6", len(sent) == 2, f"ส่ง {sent}")
    check("เป็น premarket กับ open30 อย่างละใบจากชั้นแรก",
          sent == [("premarket", "A"), ("open30", "A")], str(sent))

    print("\n=== ชั้นแรกส่งได้แค่ slot เดียว ชั้นถัดไปต้องส่ง slot ที่เหลือได้ ===")
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)
    sent = []
    # ชั้น A มาเช้ามาก open30 เกิน MAX_WAIT จึงรับแค่ premarket
    ms.mark("premarket", TODAY, F); sent.append("premarket@A")
    for slot in ("premarket", "open30"):
        if not ms.already_sent(slot, TODAY, F):
            sent.append(f"{slot}@B"); ms.mark(slot, TODAY, F)
    check("ชั้น B ส่ง open30 ได้ และไม่ส่ง premarket ซ้ำ",
          sent == ["premarket@A", "open30@B"], str(sent))

    print("\n=== ไฟล์เสียต้องไม่ทำให้เมลไม่ออก ===")
    with open(F, "w", encoding="utf-8") as fh:
        fh.write("{ไม่ใช่ json เลย")
    check("อ่านไฟล์เสีย → ถือว่ายังไม่เคยส่ง (ยอมเมลซ้ำ ดีกว่าเมลไม่ออก)",
          not ms.already_sent("premarket", TODAY, F))
    ms.mark("premarket", TODAY, F)
    check("mark ทับไฟล์เสียได้", ms.already_sent("premarket", TODAY, F))

    print("\n=== ตัดของเก่าทิ้ง ไม่ให้ไฟล์โตไปเรื่อย ๆ ===")
    old = (mc.today_et() - timedelta(days=ms.KEEP_DAYS + 5)).isoformat()
    recent = (mc.today_et() - timedelta(days=1)).isoformat()
    ms.save({old: ["premarket"], recent: ["open30"]}, F)
    st = ms.load(F)
    check("วันที่เก่ากว่า KEEP_DAYS ถูกตัด", old not in st, str(sorted(st)))
    check("วันที่ยังใหม่ยังอยู่", recent in st, str(sorted(st)))

    print("\n=== CLI ที่ workflow เรียก (exit code คือสิ่งที่ bash ใช้ตัดสิน) ===")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    F2 = os.path.join(TMP, "cli.json")

    def run(*a):
        return subprocess.run([sys.executable, os.path.join(ROOT, "mail_state.py"),
                               *a, "--file", F2, "--day", str(TODAY)],
                              capture_output=True, text=True, encoding="utf-8", env=env)

    r = run("check", "premarket")
    check("check ตอนยังไม่ส่ง → exit 0 (bash แปลว่า 'ส่งได้')", r.returncode == 0,
          f"exit {r.returncode}")
    r = run("mark", "premarket")
    check("mark → exit 0", r.returncode == 0, f"exit {r.returncode} {r.stderr[:80]}")
    r = run("check", "premarket")
    check("check ตอนส่งแล้ว → exit 1 (bash แปลว่า 'ข้าม')", r.returncode == 1,
          f"exit {r.returncode}")
    r = run("check", "open30")
    check("check slot อื่น → exit 0", r.returncode == 0, f"exit {r.returncode}")
    r = run("show")
    check("show คืน JSON ที่อ่านได้", json.loads(r.stdout).get(str(TODAY)) == ["premarket"],
          r.stdout[:120])
    r = run("check")
    check("ไม่ใส่ slot → exit 2 ไม่ใช่ 0 (กันพลาดเป็น 'ส่งได้')", r.returncode == 2,
          f"exit {r.returncode}")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\nสรุป: {'ผ่านหมด' if not FAIL else f'ไม่ผ่าน {len(FAIL)} เคส'}")
raise SystemExit(1 if FAIL else 0)
