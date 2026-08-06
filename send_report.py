"""
send_report.py — สร้าง Signal Recap แล้วส่งเข้าอีเมล
รันบน GitHub Actions (cron) หรือรันมือในเครื่องก็ได้

ตั้งค่าผ่าน environment variables (ห้าม hardcode):
    SMTP_HOST     default smtp.gmail.com
    SMTP_PORT     default 587
    SMTP_USER     อีเมลผู้ส่ง เช่น you@gmail.com
    SMTP_PASS     Gmail App Password 16 หลัก (ไม่ใช่รหัสผ่านปกติ)
    MAIL_TO       ผู้รับ คั่นด้วย , ได้หลายคน
    SYMBOLS       default "QQQ" — ใส่หลายตัวคั่นด้วย , เช่น "QQQ,SPY"

ทดสอบโดยไม่ส่งจริง:
    python send_report.py --dry-run
"""

import argparse
import os
import smtplib
import sys
import traceback
from datetime import datetime
from email.message import EmailMessage

import snapshot
from dashboard import STATUS_COLOR, render_text

BG, CARD, LINE, TXT, DIM = "#0b1220", "#111d33", "#1d2f4f", "#e8f2ff", "#7e93b5"


def build_html(snaps: list[dict]) -> str:
    """HTML แบบ table + inline style — mail client ส่วนใหญ่ไม่รองรับ CSS grid/flex"""
    blocks = []
    for s in snaps:
        if s.get("error"):
            blocks.append(
                f'<div style="background:{CARD};border:1px solid {LINE};border-radius:8px;'
                f'padding:16px;margin-bottom:18px;color:#e74c3c">'
                f'<b>{s.get("symbol","?")}</b> — {s["error"]}</div>')
            continue

        rows = []
        for r in s["rows"]:
            col = STATUS_COLOR.get(r["status"], DIM)
            rows.append(
                f'<tr>'
                f'<td style="padding:8px 10px;border-bottom:1px solid {LINE};'
                f'color:#8aadee;font-size:12px;font-weight:600;white-space:nowrap">{r["label"]}</td>'
                f'<td style="padding:8px 10px;border-bottom:1px solid {LINE};'
                f'color:{TXT};font-family:monospace;font-size:14px;font-weight:600;'
                f'white-space:nowrap">{r["value"]}</td>'
                f'<td style="padding:8px 10px;border-bottom:1px solid {LINE};text-align:center">'
                f'<span style="color:{col};border:1px solid {col};border-radius:4px;'
                f'padding:2px 7px;font-size:10px;font-weight:700;white-space:nowrap">'
                f'{r["status"]}</span></td>'
                f'<td style="padding:8px 10px;border-bottom:1px solid {LINE};'
                f'color:{DIM};font-size:11.5px;line-height:1.45">{r["note"]}</td>'
                f'</tr>')

        blocks.append(
            f'<div style="background:{CARD};border:1px solid {LINE};border-radius:8px;'
            f'padding:14px 8px;margin-bottom:20px">'
            f'<div style="padding:0 10px 10px">'
            f'<span style="color:{TXT};font-size:19px;font-weight:700">{s["symbol"]}</span>'
            f'<span style="color:{TXT};font-family:monospace;font-size:16px;margin-left:10px">'
            f'${s["spot"]:,.2f}</span>'
            f'<div style="color:{DIM};font-size:11px;margin-top:4px">'
            f'expiry ใกล้สุด {s["near_expiry"]} ({s["near_dte"]}d) · '
            f'{s["n_contracts"]:,} สัญญา · ATM IV '
            f'{s["atm_iv"]:.1f}%</div></div>'
            f'<table style="width:100%;border-collapse:collapse">{"".join(rows)}</table>'
            f'</div>')

    return (
        f'<html><body style="margin:0;padding:18px;background:{BG};'
        f'font-family:-apple-system,Segoe UI,Roboto,sans-serif">'
        f'<div style="max-width:820px;margin:0 auto">'
        f'<div style="color:{TXT};font-size:22px;font-weight:700;margin-bottom:4px">'
        f'📋 Signal Recap</div>'
        f'<div style="color:{DIM};font-size:12px;margin-bottom:18px">'
        f'{datetime.now():%Y-%m-%d %H:%M} · CBOE delayed (~15 นาที) + yfinance</div>'
        f'{"".join(blocks)}'
        f'<div style="color:{DIM};font-size:11px;line-height:1.6;border-top:1px solid {LINE};'
        f'padding-top:12px">'
        f'⚠️ หน่วยของ GEX / DEX / VEX เป็น convention ของระบบนี้เอง เทียบข้ามผู้ให้บริการไม่ได้ — '
        f'ให้ดู<b>ตำแหน่ง level</b>และ<b>การเปลี่ยนแปลง</b> ไม่ใช่เลขดิบ<br>'
        f'⚠️ เครื่องหมายของ Net GEX เดี่ยว ๆ ไม่ควรใช้เป็น Gate ตัดสิน regime '
        f'(ผลต่างของเลขใหญ่สองก้อน พลิกง่าย)<br>'
        f'ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน'
        f'</div></div></body></html>')


def _env(name: str, default: str = "") -> str:
    """
    อ่าน env แบบถือว่า "ค่าว่าง = ไม่ได้ตั้ง"
    สำคัญบน GitHub Actions: secret ที่ไม่มีอยู่จะถูกส่งมาเป็นสตริงว่าง ไม่ใช่ตัวแปรที่หายไป
    → os.environ.get(name, default) จะคืน "" ไม่ใช่ default
    """
    return (os.environ.get(name) or "").strip() or default


def send(html: str, text: str, subject: str) -> None:
    user = _env("SMTP_USER")
    pw = _env("SMTP_PASS")
    if not user or not pw:
        raise SystemExit(
            "ยังไม่ได้ตั้ง SMTP_USER / SMTP_PASS\n"
            "  GitHub: repo → Settings → Secrets and variables → Actions\n"
            "  ในเครื่อง: set SMTP_USER=you@gmail.com และ SMTP_PASS=<app password 16 หลัก>")

    # ไม่ตั้ง MAIL_TO = ส่งหาตัวเอง (กรณีใช้บ่อยสุด)
    to = [a.strip() for a in _env("MAIL_TO", user).split(",") if a.strip()]
    if not to:
        raise SystemExit("MAIL_TO ไม่ถูกต้อง")

    if " " in pw:
        pw = pw.replace(" ", "")          # Google โชว์ app password เป็น 4 กลุ่มมีเว้นวรรค
    if len(pw) != 16:
        print(f"⚠️  SMTP_PASS ยาว {len(pw)} ตัว — App Password ของ Google ยาว 16 ตัว "
              f"(ถ้าใช้รหัสผ่าน Gmail ปกติ Google จะปฏิเสธ)", file=sys.stderr)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    host = _env("SMTP_HOST", "smtp.gmail.com")
    port = int(_env("SMTP_PORT", "587"))
    print(f"เชื่อมต่อ {host}:{port} เป็น {user} → ส่งหา {', '.join(to)}")
    try:
        with smtplib.SMTP(host, port, timeout=45) as srv:
            srv.starttls()
            srv.login(user, pw)
            srv.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        raise SystemExit(
            f"เข้าสู่ระบบไม่ผ่าน ({e.smtp_code}) — เช็ค 2 อย่าง:\n"
            "  1) SMTP_PASS ต้องเป็น App Password 16 หลัก ไม่ใช่รหัสผ่าน Gmail ปกติ\n"
            "  2) บัญชีต้องเปิด 2-Step Verification ก่อนถึงจะสร้าง App Password ได้")
    print(f"✅ ส่งแล้ว → {', '.join(to)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="คำนวณและเขียนไฟล์ preview.html แต่ไม่ส่งเมล")
    ap.add_argument("--symbols", default=os.environ.get("SYMBOLS", "QQQ"))
    args = ap.parse_args()

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    snaps, failed = [], []
    for s in syms:
        try:
            snaps.append(snapshot.build_snapshot(s))
            print(f"  {s}: OK")
        except Exception as e:
            traceback.print_exc()
            failed.append(s)
            snaps.append({"symbol": s, "error": f"{type(e).__name__}: {e}"})

    ok = [s for s in snaps if not s.get("error")]
    if not ok:
        print("คำนวณไม่สำเร็จสักตัว — ไม่ส่งเมล", file=sys.stderr)
        return 1

    html = build_html(snaps)
    text = "\n\n".join(render_text(s) for s in snaps)
    tag = "⚠️ " if failed else ""
    head = ok[0]
    subject = (f"{tag}Signal Recap {', '.join(syms)} — "
               f"{head['symbol']} ${head['spot']:,.2f} · {datetime.now():%d %b %H:%M}")

    if args.dry_run:
        with open("preview.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n[dry-run] ไม่ส่งเมล · subject: {subject}")
        print("เขียน preview.html แล้ว\n")
        print(text[:1500])
        return 0

    send(html, text, subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
