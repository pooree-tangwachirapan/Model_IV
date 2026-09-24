"""
zero_dte_cli.py — สั่งงานตัวบันทึก 0DTE จากบรรทัดคำสั่ง (ใช้ใน GitHub Actions)

    python zero_dte_cli.py record                      # บันทึก snapshot เดี๋ยวนี้ (QQQ)
    python zero_dte_cli.py record  --symbols QQQ,SPY
    python zero_dte_cli.py report                      # ทำรายงาน HTML ของวันนี้
    python zero_dte_cli.py report  --day 2026-09-25 --out รายงาน.html
    python zero_dte_cli.py days                        # วันที่มีข้อมูลแล้ว
    python zero_dte_cli.py rollup                      # รวบเดือนที่จบแล้วเป็น .csv.gz
    python zero_dte_cli.py summary --day 2026-09-25    # สรุปหนึ่งวันเป็นข้อความ

exit code: 0 = สำเร็จ · 1 = ล้มเหลว
`record` จะ **ไม่** คืน 1 เมื่ออยู่นอกเวลาทำการ (ไม่ใช่ความผิดพลาด) นอกจากใส่ --strict
เพราะลูปใน workflow เรียกทุก 15 นาที ถ้าคืน 1 ตอนตลาดปิดจะมี annotation แดงเปล่า ๆ ทุกวัน
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import market_clock as mc
import zero_dte as z


def cmd_record(a) -> int:
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    prog = mc.session_progress()

    if not prog["trading"] and not a.force:
        print(f"วันนี้ ({prog['date']}) ตลาดไม่เปิด — ไม่บันทึก (ใส่ --force ถ้าต้องการ)")
        return 1 if a.strict else 0

    bad = 0
    for sym in syms:
        r = z.record(sym)
        if r.get("ok"):
            print(f"  {sym}: expiry {r['expiry']} · {r['strikes']} strike · "
                  f"เขียน {r['written']} แถว"
                  + (f" · ข้าม {r['skipped']} (ซ้ำ)" if r.get("skipped") else "")
                  + (f" · {r['reason']}" if r.get("reason") else "")
                  + f" → {r['path']}")
        else:
            bad += 1
            print(f"  !! {sym}: {r.get('error')}", file=sys.stderr)

    if bad and a.strict:
        return 1
    return 0


def cmd_report(a) -> int:
    import zero_dte_report as rep

    day = a.day or _latest_day(a.symbols)
    if not day:
        print("ยังไม่มีข้อมูลให้ทำรายงาน — รัน record ก่อน", file=sys.stderr)
        return 1
    sym = a.symbols.split(",")[0].strip().upper()
    p = rep.write(sym, day, a.out, standalone=a.standalone)
    print(f"เขียนรายงานแล้ว: {p}")
    return 0


def _latest_day(symbols: str) -> str | None:
    sym = symbols.split(",")[0].strip().upper()
    days = z.available_days(sym)
    return days[-1] if days else None


def cmd_rollup(a) -> int:
    """รวบไฟล์รายวันของเดือนที่จบแล้วเป็นคลัง .csv.gz"""
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    today = mc.today_et()
    this_month = today.strftime("%Y-%m")

    months = [a.month] if a.month else None
    rc = 0
    for sym in syms:
        todo = months
        if todo is None:
            # เดือนที่ยังมีไฟล์รายวันอยู่ ยกเว้นเดือนปัจจุบัน
            days = z.available_days(sym)
            open_months = {d[:7] for d in days if os.path.exists(z.path_for(sym, d))}
            todo = sorted(open_months - {this_month})
            if not todo:
                # ต้องพิมพ์เสมอ — งานที่เงียบตอนไม่ทำอะไร แยกไม่ออกจากงานที่พัง
                # (บทเรียนซ้ำของโปรเจกต์นี้: "สำเร็จ" ไม่เท่ากับ "ได้ทำ")
                print(f"  {sym}: ยังไม่มีเดือนที่จบแล้วให้รวบ"
                      + (f" (มีไฟล์รายวันของ {', '.join(sorted(open_months))}"
                         f" · เดือนปัจจุบัน {this_month} ยังไม่รวบ)" if open_months
                         else " — ยังไม่มีข้อมูลเลย"))
        for m in todo:
            if m == this_month and not a.force:
                print(f"  ข้าม {sym} {m} — เดือนนี้ยังไม่จบ (ใส่ --force ถ้าต้องการ)")
                continue
            r = z.rollup_month(sym, m, remove=not a.keep)
            if r.get("reason"):
                print(f"  {sym} {m}: {r['reason']}")
                continue
            print(f"  {sym} {m}: {r['days']} วัน · {r['rows']:,} แถว → "
                  f"{r['bytes']/1024/1024:.2f} MB · ลบไฟล์รายวัน {r['removed']} ไฟล์"
                  + ("" if r["verified"] else "  !! อ่านคลังกลับมาไม่ตรง ไม่ได้ลบอะไร"))
            if not r["verified"]:
                rc = 1
    return rc


def cmd_days(a) -> int:
    syms = z.available_symbols() or [s.strip().upper() for s in a.symbols.split(",")]
    for sym in syms:
        days = z.available_days(sym)
        print(f"{sym}: {len(days)} วัน" + (f"  {days[0]} → {days[-1]}" if days else ""))
        for d in days:
            print(f"    {d}")
    return 0


def cmd_summary(a) -> int:
    sym = a.symbols.split(",")[0].strip().upper()
    day = a.day or _latest_day(a.symbols)
    if not day:
        print("ยังไม่มีข้อมูล", file=sys.stderr)
        return 1
    s = z.day_summary(sym, day)
    if s.get("empty"):
        print(f"{sym} {day}: ไฟล์ว่าง")
        return 1
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return 0
    print(f"{sym} · {s['day']} · expiry {s['expiry']}")
    print(f"  snapshot     {s['snapshots']} ครั้ง ({s['first_et']}–{s['last_et']} ET)")
    print(f"  strike       {s['strikes']} · แถวรวม {s['rows']:,}")
    print(f"  spot         {s['spot_open']:,.2f} → {s['spot_last']:,.2f}"
          f"  (ต่ำ {s['spot_low']:,.2f} · สูง {s['spot_high']:,.2f})")
    print(f"  ปริมาณ C/P   {s['call_volume']:,.0f} / {s['put_volume']:,.0f}"
          + (f"  P/C {s['pc_ratio']:.2f}" if s["pc_ratio"] else ""))
    print(f"  พรีเมียมรวม  ${s['notional_usd']:,.0f}")
    print("  strike ที่พรีเมียมหมุนมากสุด:")
    for t in s["top_notional"][:5]:
        print(f"    {t['strike']:g}{t['cp']:<2} ${t['notional']:>14,.0f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ตัวบันทึก/รายงาน option 0DTE")
    ap.add_argument("cmd",
                    choices=["record", "report", "days", "summary", "rollup"])
    ap.add_argument("--symbols", default="QQQ", help="คั่นด้วย , (ค่าตั้งต้น QQQ)")
    ap.add_argument("--day", default="", help="YYYY-MM-DD (ว่าง = วันล่าสุดที่มีข้อมูล)")
    ap.add_argument("--out", default="", help="path ไฟล์รายงาน")
    ap.add_argument("--standalone", action="store_true",
                    help="ฝัง plotly.js ในไฟล์ (ไฟล์ใหญ่ ~4MB แต่เปิดออฟไลน์ได้)")
    ap.add_argument("--force", action="store_true", help="บันทึกแม้วันนี้ตลาดไม่เปิด")
    ap.add_argument("--strict", action="store_true",
                    help="คืน exit 1 เมื่อบันทึกไม่สำเร็จ (ค่าตั้งต้นคือกลืน)")
    ap.add_argument("--json", action="store_true", help="summary เป็น JSON")
    ap.add_argument("--month", default="", help="rollup: YYYY-MM (ว่าง = ทุกเดือนที่จบแล้ว)")
    ap.add_argument("--keep", action="store_true",
                    help="rollup: เขียนคลังแต่ไม่ลบไฟล์รายวัน")
    a = ap.parse_args()
    a.out = a.out or None
    a.day = a.day or None
    a.month = a.month or None

    return {"record": cmd_record, "report": cmd_report, "days": cmd_days,
            "summary": cmd_summary, "rollup": cmd_rollup}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
