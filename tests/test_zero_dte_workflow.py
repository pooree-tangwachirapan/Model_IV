"""
ตรวจ zero-dte-record.yml — workflow ที่ cron-job.org กดทุก 15 นาที (HANDOFF §6.16)

ไม่ได้รัน bash จริง แต่ล็อกบทเรียนที่ workflow เดิม ๆ จ่ายด้วยข้อมูลหายมาแล้ว:
  §6.13  `git add a/ b/` ล้มทั้งคำสั่ง → ต้อง add ทีละโฟลเดอร์
  §6.14  อะไรที่แก้ไฟล์ในรีโป ต้องอยู่หลัง `git reset --hard`
  §6.4   `|| true` ห้ามกลืน error ของขั้นตอน push
  §5     ตัดสินเวลา *ก่อน* pip install — รอบที่ไม่ทำงานต้องออกใน ~20 วินาที
และเช็คว่าหน้าต่างบันทึกครอบถึงหลังปิดตลาด (เหตุผลที่สร้าง workflow นี้)
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, ".github", "workflows", "zero-dte-record.yml")

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


with open(WF, encoding="utf-8") as fh:
    Y = fh.read()


def step(title: str) -> str:
    """เนื้อของ step ชื่อ `title` ถึงก่อน step ถัดไป"""
    m = re.search(rf"- name: {re.escape(title)}\n(.*?)(?=\n      - (?:name|uses):|\Z)", Y, re.S)
    if not m:
        return ""
    # ตัดบรรทัดคอมเมนต์ทิ้ง — คอมเมนต์ที่อธิบายว่า "ไม่ใช้ X" ต้องไม่ถูกนับว่าใช้ X
    return "\n".join(l for l in m.group(1).splitlines() if not l.lstrip().startswith("#"))


print("=== trigger ===")
check("มี workflow_dispatch (cron-job.org กดผ่าน API)", "workflow_dispatch:" in Y)
check("ไม่มี schedule ของ GitHub (ยิงจริง 2/26 ชั้น — §6.16)",
      re.search(r"^\s*schedule:", Y, re.M) is None)
check("timeout สั้น — งานรอบเดียวจบ ไม่นอนรอ",
      re.search(r"timeout-minutes:\s*(\d+)", Y) is not None
      and int(re.search(r"timeout-minutes:\s*(\d+)", Y).group(1)) <= 15)

print("=== ด่านเวลา ===")
plan = step("Check the window")
check("มี step ตัดสินเวลา", bool(plan))
check("ด่านเวลาอยู่ก่อน pip install",
      0 <= Y.find("- name: Check the window") < Y.find("pip install"))
check("ถาม market_clock ว่าวันนี้ตลาดเปิดไหม (วันหยุด/DST)", "market_clock.py trading" in plan)
m_until = re.search(r"UNTIL=\$\(\(\s*CLOSE\s*\+\s*(\d+)\s*\*\s*60", plan)
check("หน้าต่างปิดหลังตลาดปิด ≥ 15 นาที (quote ดีเลย์ ~15 นาที → ได้ราคาปิด)",
      m_until is not None and int(m_until.group(1)) >= 15,
      m_until.group(0) if m_until else "ไม่พบ UNTIL")
m_from = re.search(r"FROM=\$\(\(\s*OPEN\s*-\s*(\d+)\s*\*\s*60", plan)
check("หน้าต่างเปิดไม่เกิน 5 นาทีก่อนระฆัง",
      m_from is not None and int(m_from.group(1)) <= 5)
check("ทุกทางออกของด่านเวลาเขียน go=", plan.count('echo "go=no"') >= 2 and "go=yes" in plan)

print("=== บันทึก ===")
rec = step("Record 0DTE snapshot")
check("step บันทึกมีเงื่อนไข go == 'yes'", "steps.plan.outputs.go == 'yes'" in rec)
check("เรียก zero_dte_cli.py record", "zero_dte_cli.py record" in rec)
check("ดึงไม่สำเร็จ → warning ไม่ใช่เงียบ", "::warning::" in rec and "--strict" in rec)

print("=== commit ===")
com = step("Commit snapshot")
check("step commit มีเงื่อนไข go == 'yes'", "steps.plan.outputs.go == 'yes'" in com)
check("เก็บเฉพาะไฟล์ที่แก้ (ไม่ใช้ merge_data --save ที่เขียน .csv.gz ใหม่ทุกรอบ)",
      "git ls-files --modified --others" in com and "merge_data.py --save" not in com)
i_reset = com.find("git reset --hard")
i_merge = com.find("merge_data.py --merge")
i_add = com.find("git add -A -- zero_dte")
check("§6.14 merge อยู่หลัง reset --hard", 0 <= i_reset < i_merge)
check("§6.14 git add อยู่หลัง merge", 0 <= i_merge < i_add)
check("§6.13 ไม่มี `git add` หลายโฟลเดอร์ในคำสั่งเดียว",
      re.search(r"git add[^\n]*\s\S+/\s+\S+/", com) is None)
check("§6.4 push ไม่ถูกกลืนด้วย || true", re.search(r"git push[^\n]*\|\|\s*true", com) is None)
check("push ไม่สำเร็จ → ::error:: + exit 1", "::error::" in com and "exit 1" in com)
check("commit มี [skip ci] (ไม่ปลุก tests.yml ทุก 15 นาที)", "[skip ci]" in com)
check("บรรทัดสุดท้ายของ script ไม่ใช่ `[ ... ] && ...` (ทำ step แดงเมื่อเงื่อนไขเป็นเท็จ)",
      not re.search(r"\]\s*&&[^\n]*\n?\s*$", com.rstrip()))

print("=" * 60)
print(f"ไม่ผ่าน {len(FAIL)} เคส" if FAIL else "ผ่านทั้งหมด")
sys.exit(1 if FAIL else 0)
