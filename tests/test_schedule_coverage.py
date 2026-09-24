"""
ตรวจว่า "บันได cron" ครอบดีเลย์ของ GitHub ได้จริง — ไม่ใช่เชื่อว่าคำนวณมาถูก

ทำไมต้องมีชุดนี้ (HANDOFF §6.9):
  ของเดิมมีคอมเมนต์อธิบายการคำนวณ cron ไว้ละเอียดมาก และมันก็ถูก **ณ วันที่เขียน**
  (ดีเลย์ 38 นาที) พอดีเลย์โตเป็น 4–5 ชม. ทุกอย่างพังเงียบโดยไม่มีเทสต์ไหนฟ้อง
  เพราะไม่เคยมีใครเขียนเทสต์ที่ถาม "ถ้าดีเลย์เป็น X แล้วยังทำงานไหม"

ชุดนี้อ่าน cron **จากไฟล์ yml จริง** แล้วจำลองกฎรับ/ไม่รับงานแบบเดียวกับที่ bash ทำ
ถ้าใครไปแก้ cron หรือแก้ค่า MAX_WAIT_MIN แล้วทำให้มีช่วงดีเลย์ที่ "ไม่มีชั้นไหนทำงานเลย"
เทสต์ชุดนี้จะฟ้องทันที

**ไม่ได้เทสต์ว่า bash เขียนถูก** — เทสต์ว่า *ตารางเวลา* ครอบคลุมพอ
ค่าคงที่ข้างล่างต้องตรงกับใน yml ถ้าแก้ที่หนึ่งต้องแก้อีกที่ (เทสต์เช็คให้ว่าตรงกัน)
"""
import warnings; warnings.filterwarnings("ignore")
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta, timezone

import market_clock as mc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, ".github", "workflows")

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


# ── ค่าที่ต้องตรงกับใน yml ──────────────────────────────────
DR_MAX_WAIT_MIN = 330        # daily-report: นอนรอล่วงหน้าได้มากสุด
DR_LATE_GRACE_MIN = 20       # daily-report: มาสายได้เท่านี้ยังส่ง
AA_MAX_JOB_MIN = 335         # armed-alert: เริ่มได้เมื่อเหลือถึง WINDOW_END ไม่เกินเท่านี้
AA_OBSERVE_OFFSET = -2       # armed-alert: เริ่มสังเกตที่ open − 2 นาที
AA_WINDOW_OFFSET = 305       # armed-alert: ปิดหน้าต่างที่ open + 305 นาที


def read_yml(name: str) -> str:
    with open(os.path.join(WF, name), encoding="utf-8") as fh:
        return fh.read()


def cron_minutes(text: str) -> list[int]:
    """แกะ cron ทุกบรรทัดเป็น 'นาทีของวัน (UTC)' — รองรับ '7,37 5-14 * * 1-5'"""
    out: set[int] = set()
    for line in re.findall(r'-\s*cron:\s*"([^"]+)"', text):
        mins, hours = line.split()[0], line.split()[1]

        def expand(spec: str, hi: int) -> list[int]:
            vals: list[int] = []
            for part in spec.split(","):
                if part == "*":
                    vals += list(range(hi + 1))
                elif "-" in part:
                    a, b = part.split("-")
                    vals += list(range(int(a), int(b) + 1))
                else:
                    vals.append(int(part))
            return vals

        for h in expand(hours, 23):
            for m in expand(mins, 59):
                out.add(h * 60 + m)
    return sorted(out)


def _assert_constants(text: str, pairs: list[tuple[str, int]], label: str):
    for name, want in pairs:
        m = re.search(rf"{name}=(\d+)", text)
        check(f"{label}: {name} ใน yml ตรงกับที่เทสต์สมมติ ({want})",
              m is not None and int(m.group(1)) == want,
              f"ใน yml = {m.group(1) if m else 'ไม่พบ'}")


# ════════════════════════════════════════════════
print("=== daily-report: ค่าคงที่ตรงกับ yml ===")
DR = read_yml("daily-report.yml")
_assert_constants(DR, [("MAX_WAIT_MIN", DR_MAX_WAIT_MIN),
                       ("LATE_GRACE_MIN", DR_LATE_GRACE_MIN)], "daily-report")

DR_RUNGS = cron_minutes(DR)
check("daily-report มีบันไดอย่างน้อย 6 ชั้น", len(DR_RUNGS) >= 6, f"มี {len(DR_RUNGS)}")


def dr_serves(land_min: int, target_min: int) -> bool:
    """ชั้นที่ลงตอน land_min จะส่ง slot ที่เวลา target_min ได้ไหม (กฎเดียวกับ bash)"""
    wait = target_min - land_min
    return -DR_LATE_GRACE_MIN <= wait <= DR_MAX_WAIT_MIN


print("\n=== daily-report: ดีเลย์เท่าไหร่ก็ต้องมีชั้นที่ส่ง premarket ได้ ===")
# เป้า premarket ของวันทำการช่วง EDT = 13:00 UTC · ช่วง EST = 14:00 UTC
for season, target in (("EDT", 13 * 60), ("EST", 14 * 60)):
    gaps = []
    for delay in range(0, 8 * 60 + 1, 5):            # ดีเลย์ 0–8 ชม. ทีละ 5 นาที
        if not any(dr_serves(r + delay, target) for r in DR_RUNGS):
            gaps.append(delay)
    check(f"{season}: ส่ง premarket ได้ทุกดีเลย์ 0–8 ชม.", not gaps,
          f"ดีเลย์ที่หลุด (นาที): {gaps[:12]}")

print("\n=== daily-report: รอบ open30 ต้องทนได้กว้างกว่าอีก ===")
for season, target in (("EDT", 14 * 60), ("EST", 15 * 60)):
    gaps = [d for d in range(0, 9 * 60 + 1, 5)
            if not any(dr_serves(r + d, target) for r in DR_RUNGS)]
    check(f"{season}: ส่ง open30 ได้ทุกดีเลย์ 0–9 ชม.", not gaps,
          f"ดีเลย์ที่หลุด (นาที): {gaps[:12]}")

print("\n=== daily-report: ชั้นแรกต้องไม่เช้าจนวันที่ ET เพี้ยน ===")
# market_clock ตัดสินจากวันที่ตามเวลา ET · ก่อน 05:00 UTC ช่วง EST เวลา ET ยังเป็นเมื่อวาน
check("ชั้นแรกอยู่ที่ 05:00 UTC หรือหลังจากนั้น", min(DR_RUNGS) >= 5 * 60,
      f"ชั้นแรก {min(DR_RUNGS)//60:02d}:{min(DR_RUNGS)%60:02d} UTC")

# ════════════════════════════════════════════════
print("\n=== armed-alert: ค่าคงที่ตรงกับ yml ===")
AA = read_yml("armed-alert.yml")
_assert_constants(AA, [("MAX_JOB_MIN", AA_MAX_JOB_MIN)], "armed-alert")
check("armed-alert: หน้าต่างปิดที่ open + 305 นาที ตามที่เทสต์สมมติ",
      f"OPEN + {AA_WINDOW_OFFSET} * 60" in AA, "ไม่เจอในไฟล์")

AA_RUNGS = cron_minutes(AA)
check("armed-alert มีบันไดอย่างน้อย 12 ชั้น", len(AA_RUNGS) >= 12, f"มี {len(AA_RUNGS)}")
check("ชั้นแรกอยู่ที่ 05:00 UTC หรือหลังจากนั้น", min(AA_RUNGS) >= 5 * 60,
      f"ชั้นแรก {min(AA_RUNGS)//60:02d}:{min(AA_RUNGS)%60:02d} UTC")


def aa_loss(land_min: int, open_min: int) -> int | None:
    """
    ชั้นที่ลงตอน land_min จะเก็บข้อมูลขาดไปกี่นาทีต้นทาง
    คืน None ถ้าชั้นนี้ไม่รับงาน (ออกเอง)
    """
    observe = open_min + AA_OBSERVE_OFFSET
    end = open_min + AA_WINDOW_OFFSET
    if land_min >= end:
        return None                                   # หน้าต่างปิดแล้ว
    if end - land_min > AA_MAX_JOB_MIN:
        return None                                   # เช้าเกิน จะโดน timeout ก่อน commit
    return max(0, land_min - observe)


print("\n=== armed-alert: ดีเลย์เท่าไหร่ก็ต้องมีชั้นที่เก็บได้เกือบเต็มวัน ===")
for season, open_min in (("EDT", 13 * 60 + 30), ("EST", 14 * 60 + 30)):
    worst = 0
    dead = []
    for delay in range(0, 8 * 60 + 1, 5):
        losses = [x for x in (aa_loss(r + delay, open_min) for r in AA_RUNGS)
                  if x is not None]
        if not losses:
            dead.append(delay)
        else:
            worst = max(worst, min(losses))           # ชั้นที่ดีที่สุดของดีเลย์นั้น
    check(f"{season}: มีชั้นทำงานเสมอ ทุกดีเลย์ 0–8 ชม.", not dead,
          f"ดีเลย์ที่ไม่มีชั้นไหนทำงานเลย (นาที): {dead[:12]}")
    check(f"{season}: แย่สุดเก็บขาดต้นทางไม่เกิน 5 นาที", worst <= 5,
          f"แย่สุดขาด {worst} นาที")

print("\n=== armed-alert: ชั้นที่รับงานต้องจบทันก่อน timeout ===")
TIMEOUT = int(re.search(r"timeout-minutes:\s*(\d+)", AA).group(1))
over = []
for season, open_min in (("EDT", 13 * 60 + 30), ("EST", 14 * 60 + 30)):
    for delay in range(0, 8 * 60 + 1, 5):
        for r in AA_RUNGS:
            land = r + delay
            if aa_loss(land, open_min) is None:
                continue
            end = open_min + AA_WINDOW_OFFSET
            # งานถือรันเนอร์ตั้งแต่ลงจนปิดหน้าต่าง ไม่ว่าจะนอนรอหรือวนลูปอยู่
            used = end - land
            if used > TIMEOUT - 10:                        # เว้น 10 นาทีให้ commit/setup
                over.append((season, delay, r, used))
check(f"ไม่มีชั้นไหนใช้เวลาเกิน timeout−10 นาที (timeout={TIMEOUT})", not over,
      f"เกิน: {over[:4]}")

print("\n=== ตรงกับ market_clock จริง ไม่ใช่เลขที่เดาเอง ===")
d_edt, d_est = date(2026, 9, 24), date(2026, 12, 15)
for lbl, d, want_pre, want_open in (("EDT", d_edt, 13 * 60, 13 * 60 + 30),
                                    ("EST", d_est, 14 * 60, 14 * 60 + 30)):
    pre = mc.slot_utc(d, "premarket")
    opn = mc.slot_utc(d, "open")
    check(f"{lbl}: premarket ที่เทสต์สมมติตรงกับ market_clock",
          pre.hour * 60 + pre.minute == want_pre, f"{pre:%H:%M}")
    check(f"{lbl}: open ที่เทสต์สมมติตรงกับ market_clock",
          opn.hour * 60 + opn.minute == want_open, f"{opn:%H:%M}")

print(f"\nสรุป: {'ผ่านหมด' if not FAIL else f'ไม่ผ่าน {len(FAIL)} เคส'}")
raise SystemExit(1 if FAIL else 0)
