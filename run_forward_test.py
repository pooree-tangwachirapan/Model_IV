"""
run_forward_test.py — ตัวสั่งงาน forward test จาก GitHub Actions

    python run_forward_test.py record                  เปิดไม้จำลองถ้า verdict = ARMED
    python run_forward_test.py resolve                 ปิดไม้ที่ถึงเป้า / invalidate / หมดเวลา
    python run_forward_test.py report --month 2026-08  สรุปผล (ใส่ --email เพื่อส่งเมล)

ledger เก็บเป็นไฟล์ในรีโป ไม่ใช่ actions/cache — cache หมดอายุ 7 วัน
แต่การทดสอบกินเวลาเป็นเดือน ข้อมูลต้องอยู่ให้ครบและย้อนดูได้
"""

import argparse
import sys
from datetime import datetime, timezone

import breakout
import forward_test as ft
import gate
import snapshot
from send_report import BG, CARD, DIM, LINE, TXT, send


def build_html(s: dict, trades: list[dict]) -> str:
    good = "#2ecc71" if s["pnl_usd"] >= 0 else "#e74c3c"

    def cell(label, value, color=TXT, note=""):
        return (f'<td style="padding:12px 14px;border:1px solid {LINE};vertical-align:top">'
                f'<div style="color:{DIM};font-size:10.5px;letter-spacing:.6px">{label}</div>'
                f'<div style="color:{color};font-family:monospace;font-size:19px;'
                f'font-weight:700;margin-top:3px">{value}</div>'
                + (f'<div style="color:{DIM};font-size:10.5px;margin-top:2px">{note}</div>'
                   if note else "") + '</td>')

    if not s["n_closed"]:
        body = (f'<div style="background:{CARD};border:1px solid {LINE};border-radius:8px;'
                f'padding:20px;color:{DIM};font-size:13px">'
                f'ยังไม่มีไม้ที่ปิดในช่วงนี้ (เปิดค้าง {s["n_open"]} ไม้) — '
                f'ระบบเข้าไม้เฉพาะตอน ARMED ซึ่งเกิดไม่บ่อยตามตั้งใจ</div>')
    else:
        rows = "".join(
            f'<tr>'
            f'<td style="padding:7px 9px;border-bottom:1px solid {LINE};color:{DIM};'
            f'font-size:11.5px;white-space:nowrap">{t["date"]}</td>'
            f'<td style="padding:7px 9px;border-bottom:1px solid {LINE};color:#8aadee;'
            f'font-size:11.5px;font-weight:600">{t["side"]}</td>'
            f'<td style="padding:7px 9px;border-bottom:1px solid {LINE};color:{TXT};'
            f'font-family:monospace;font-size:11.5px;white-space:nowrap">'
            f'{t["entry"]:,.2f} → {t["exit"]:,.2f}</td>'
            f'<td style="padding:7px 9px;border-bottom:1px solid {LINE};'
            f'color:{"#2ecc71" if t["realized_r"] > 0 else "#e74c3c"};'
            f'font-family:monospace;font-size:11.5px;font-weight:700">'
            f'{t["realized_r"]:+.2f}R</td>'
            f'<td style="padding:7px 9px;border-bottom:1px solid {LINE};'
            f'color:{"#2ecc71" if t["pnl_usd"] > 0 else "#e74c3c"};'
            f'font-family:monospace;font-size:11.5px">${t["pnl_usd"]:+,.2f}</td>'
            f'<td style="padding:7px 9px;border-bottom:1px solid {LINE};color:{DIM};'
            f'font-size:10.5px">{t["status"]}</td></tr>'
            for t in sorted((x for x in trades if x.get("realized_r") is not None),
                            key=lambda x: x.get("closed_at") or x["opened"]))
        pass_fail = (s["win_rate"] > s["breakeven_wr"]) if s["breakeven_wr"] else None
        body = (
            f'<table style="width:100%;border-collapse:collapse;margin-bottom:16px"><tr>'
            + cell("ไม้ที่ปิด", s["n_closed"],
                   note=f'ชนะ {s["wins"]} · แพ้ {s["losses"]} · หมดเวลา {s["timeouts"]}')
            + cell("WIN RATE", f'{s["win_rate"]*100:.1f}%',
                   note=(f'ต้องเกิน {s["breakeven_wr"]*100:.1f}% ถึงเท่าทุน'
                         if s["breakeven_wr"] else ""))
            + cell("EXPECTANCY", f'{s["expectancy_r"]:+.2f}R', good, note="ต่อไม้")
            + '</tr><tr>'
            + cell("P&L", f'${s["pnl_usd"]:+,.2f}', good, note=f'{s["return_pct"]:+.2f}% ของพอร์ต')
            + cell("พอร์ต", f'${s["equity_end"]:,.2f}', good,
                   note=f'เริ่ม ${s["equity_start"]:,.0f}')
            + cell("MAX DD", f'${s["max_dd_usd"]:,.2f}',
                   note=(f'RR 1:{s["rr"]:.2f}' if s["rr"] else ""))
            + '</tr></table>'
            + (f'<div style="background:rgba(231,76,60,.12);border-left:3px solid #e74c3c;'
               f'padding:10px 12px;color:#e8b4ae;font-size:12px;margin-bottom:16px">'
               f'<b>{s["n_closed"]} ไม้ยังน้อยเกินกว่าจะสรุปว่ามี edge</b> — ช่วงความเชื่อมั่นที่ '
               f'{s["n_closed"]} ไม้ยังกว้างมาก อ่านว่า "ระบบเดินได้" ไม่ใช่ "ระบบกำไร"</div>'
               if s["n_closed"] < 20 else "")
            + (f'<div style="color:{DIM};font-size:12px;margin-bottom:6px">'
               f'{"ผ่านจุดเท่าทุน" if pass_fail else "ยังไม่ผ่านจุดเท่าทุน"}</div>'
               if pass_fail is not None else "")
            + f'<div style="background:{CARD};border:1px solid {LINE};border-radius:8px;'
              f'padding:6px 10px 10px"><table style="width:100%;border-collapse:collapse">'
              f'{rows}</table></div>')

    return (
        f'<html><body style="margin:0;padding:18px;background:{BG};'
        f'font-family:-apple-system,Segoe UI,Roboto,sans-serif">'
        f'<div style="max-width:820px;margin:0 auto">'
        f'<div style="color:{TXT};font-size:22px;font-weight:700">📊 Forward Test — '
        f'{s["month"] or "ทั้งหมด"}</div>'
        f'<div style="color:{DIM};font-size:12px;margin:4px 0 18px">'
        f'{ft.SYMBOL} · พอร์ตจำลอง ${ft.ACCOUNT_START:,.0f} · ไม้ละ {ft.QTY} หน่วย '
        f'เท่ากันทั้ง LONG/SHORT (1 จุด = ${ft.QTY}) · '
        f'สูงสุด {ft.MAX_TRADES_PER_DAY} ไม้/วัน · '
        f'ถือไม่เกิน {ft.MAX_HOLD_DAYS} วันทำการ</div>'
        f'{body}'
        f'<div style="color:{DIM};font-size:11px;line-height:1.6;border-top:1px solid {LINE};'
        f'margin-top:16px;padding-top:12px">'
        f'จำลองที่ราคา underlying เป็นหน่วย R <b>ไม่ได้จำลองราคา option</b> — '
        f'ของจริงจะมี spread, IV crush และ fill ที่ไม่ตรงแผน ผลจริงย่อมแย่กว่านี้<br>'
        f'ตัดสินด้วยแท่ง 5 นาที นับเฉพาะหลังเวลาเข้าไม้ · แตะทั้งเป้าและ invalidate '
        f'ในแท่งเดียวกันนับเป็นแพ้เสมอ<br>'
        f'ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน</div></div></body></html>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "resolve", "report"])
    ap.add_argument("--symbol", default=ft.SYMBOL)
    ap.add_argument("--month", default=None, help="YYYY-MM · ไม่ใส่ = เดือนนี้ (report)")
    ap.add_argument("--system", default=ft.DEFAULT_SYSTEM, choices=list(ft.SYSTEMS),
                    help="ระบบที่จะเดิน — fade (รอที่กำแพง) หรือ breakout (ไล่ตามการทะลุ)")
    ap.add_argument("--ledger", default=None,
                    help="ไม่ใส่ = ใช้ ledger ประจำระบบที่เลือก")
    ap.add_argument("--email", action="store_true", help="ส่งรายงานเข้าเมล (report)")
    args = ap.parse_args()

    ledger = args.ledger or ft.ledger_path(args.system)
    trades = ft.load(ledger)
    before = len(trades)
    print(f"  ระบบ: {ft.system_cfg(args.system)['label']}  ·  ledger {ledger}")

    if args.cmd == "record":
        snap = snapshot.build_snapshot(args.symbol)
        if snap.get("error"):
            print(f"ดึงข้อมูลไม่ได้: {snap['error']}", file=sys.stderr)
            return 1
        # ตัวตัดสินคนละตัวตามระบบ — โครงของ verdict/plan เหมือนกัน record จึงใช้ร่วมได้
        evaluator = breakout.evaluate if args.system == "breakout" else gate.evaluate
        g = evaluator(snap)
        print(f"  {args.symbol}: {g['verdict']} — {g['headline']}")
        t = ft.record(trades, g, system=args.system)
        if t:
            print(f"  เปิดไม้จำลอง {t['side']} @ {t['entry']:,.2f} · "
                  f"invalidate {t['invalidation']:,.2f} · เป้า {t['target']:,.2f} "
                  f"· {t['plan_r']:.2f}R")
        else:
            print("  ไม่เปิดไม้ (ไม่ ARMED / มีไม้ค้าง / ครบลิมิตวันนี้)")

    elif args.cmd == "resolve":
        closed = ft.resolve(trades)
        print(f"  ปิดไป {len(closed)} ไม้" if closed else "  ไม่มีไม้ที่ถึงเงื่อนไขปิด")
        for t in closed:
            if t["status"] == "no_fill":
                print(f"    {t['id']} ไม่ได้ fill — {t.get('note','')}")
            else:
                print(f"    {t['id']} {t['status']} @ {t['exit']:,.2f} "
                      f"{t['realized_r']:+.2f}R ${t['pnl_usd']:+,.2f}")

    else:  # report
        month = args.month or datetime.now(timezone.utc).strftime("%Y-%m")
        s = ft.stats(trades, month)
        text = ft.render_text(s)
        print(text)
        if args.email:
            sel = [t for t in trades if t["date"].startswith(month)]
            head = (f'{s["pnl_usd"]:+,.0f} USD' if s["n_closed"] else "ยังไม่มีไม้ปิด")
            send(build_html(s, sel), text,
                 f"📊 Forward Test {month} — {ft.SYMBOL} {head}")

    if len(trades) != before or args.cmd == "resolve":
        # อ่านไฟล์ใหม่แล้ว merge ก่อนเขียนเสมอ — ระหว่างที่เราคำนวณอยู่ (ดึงราคา yfinance
        # ใช้เวลาหลายวินาที) อีก workflow อาจเขียนไฟล์นี้ไปแล้ว ถ้าเขียนทับตรง ๆ ไม้ของเขาหาย
        ft.save(ft.merge(trades, ft.load(ledger)), ledger)
        final = ft.load(ledger)
        print(f"  เขียน ledger แล้ว ({len(final)} ไม้"
              + (f" · merge เพิ่มจากดิสก์ {len(final) - len(trades)} ไม้"
                 if len(final) != len(trades) else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
