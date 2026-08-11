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
import json
import os
import smtplib
import sys
import traceback
from datetime import datetime
from email.message import EmailMessage

import gate
import snapshot
from dashboard import STATUS_COLOR, render_text

BG, CARD, LINE, TXT, DIM = "#0b1220", "#111d33", "#1d2f4f", "#e8f2ff", "#7e93b5"

GATE_COLOR = {"ARMED": "#2ecc71", "WATCH": "#e67e22", "STAND_DOWN": "#e74c3c"}
GATE_WORD = {"ARMED": "ARMED", "WATCH": "WATCH", "STAND_DOWN": "STAND DOWN"}


def build_gate_html(gates: list[dict]) -> str:
    """
    บล็อก "วันนี้เข้าได้ไหม" — วางบนสุดของเมลเพราะเป็นส่วนเดียวที่ต้องตัดสินใจ
    ARMED = โครงสร้างของกระดานไม่ได้ห้ามไว้ ไม่ใช่คำสั่งให้เข้า
    """
    if not gates:
        return ""
    cards = []
    for g in gates:
        bad = g.get("data_issue")
        col = "#f1c40f" if bad else GATE_COLOR.get(g["verdict"], DIM)
        word = "DATA ISSUE" if bad else GATE_WORD[g["verdict"]]
        lines = []
        for row in g["hard"] + g["soft"]:
            c = "#2ecc71" if row["ok"] else col
            mark = "✓" if row["ok"] else "✕"
            lines.append(
                f'<tr><td style="padding:5px 8px;color:{c};font-size:13px;'
                f'width:22px;text-align:center">{mark}</td>'
                f'<td style="padding:5px 8px;color:#8aadee;font-size:12px;'
                f'font-weight:600;white-space:nowrap">{row["label"]}</td>'
                f'<td style="padding:5px 8px;color:{TXT};font-family:monospace;'
                f'font-size:12px">{row["value"]}</td></tr>')

        plan = ""
        p = g.get("plan")
        if p:
            plan = (
                f'<div style="margin:10px 10px 0;padding:10px 12px;background:{BG};'
                f'border-left:3px solid {col};border-radius:0 6px 6px 0">'
                f'<div style="color:{TXT};font-size:13px;font-weight:600">{p["direction"]}</div>'
                f'<div style="color:{DIM};font-size:11.5px;line-height:1.6;margin-top:4px;'
                f'font-family:monospace">'
                f'entry {p["ideal_entry"]:,.2f} · invalidate {p["invalidation"]:,.2f} · '
                f'target {p["target"]:,.2f}'
                + (f' · {p["plan_r"]:.2f}R' if p.get("plan_r") else "")
                + f'</div><div style="color:{DIM};font-size:11px;margin-top:3px">'
                  f'ราคาตอนนี้ {p["spot"]:,.2f} · '
                  + ("เลย invalidate ไปแล้ว — แผนนี้ตาย" if p["blown"]
                     else "อยู่ที่ขอบพอดี" if p["at_edge"]
                     else f'ทะลุจุดเข้าไปแล้ว {abs(p["dist_to_entry_pct"]):.2f}% — ตกรถ'
                     if p["dist_to_entry_pts"] < 0
                     else f'ยังห่างจุดเข้า {p["dist_to_entry_pct"]:.2f}%')
                  + f' · {p["target_src"]}</div></div>')

        todo = "".join(f'<div style="color:{DIM};font-size:11.5px;line-height:1.7">☐ {m}</div>'
                       for m in g["manual"])

        cards.append(
            f'<div style="background:{CARD};border:1px solid {LINE};border-left:4px solid {col};'
            f'border-radius:8px;padding:14px 6px;margin-bottom:14px">'
            f'<div style="padding:0 10px 8px">'
            f'<span style="color:{col};border:1px solid {col};border-radius:4px;padding:2px 8px;'
            f'font-size:11px;font-weight:700;letter-spacing:.5px">{word}</span>'
            f'<span style="color:{TXT};font-size:17px;font-weight:700;margin-left:10px">'
            f'{g["symbol"]}</span>'
            f'<div style="color:{TXT};font-size:14px;font-weight:600;margin-top:8px">'
            f'{g["headline"]}</div>'
            f'<div style="color:{DIM};font-size:11.5px;line-height:1.5;margin-top:3px">'
            f'{g["reason"]}</div></div>'
            f'<table style="width:100%;border-collapse:collapse">{"".join(lines)}</table>'
            f'{plan}'
            f'<div style="margin:12px 10px 0;padding-top:10px;border-top:1px solid {LINE}">'
            f'<div style="color:#8aadee;font-size:11px;font-weight:600;margin-bottom:5px">'
            f'ยังต้องยืนยันเองก่อนกด</div>{todo}</div>'
            f'</div>')

    return (
        f'<div style="color:{TXT};font-size:22px;font-weight:700;margin-bottom:4px">'
        f'🎯 วันนี้เข้าได้ไหม</div>'
        f'<div style="color:{DIM};font-size:12px;margin-bottom:14px">'
        f'ประตูอัตโนมัติจาก gate.py · <b>ARMED = โครงสร้างไม่ได้ห้ามไว้ ไม่ใช่คำสั่งให้เข้า</b></div>'
        f'{"".join(cards)}'
        f'<div style="color:{DIM};font-size:11px;line-height:1.6;margin:-4px 0 22px">'
        f'⚠️ เมลนี้ส่งตามเวลา cron — ถ้ารันก่อนตลาดเปิด level ที่เห็นคือ OI ของเมื่อคืน '
        f'ยังไม่ใช่สภาพตอนราคาวิ่งจริง ต้อง re-check ระหว่างวันก่อนเข้าไม้เสมอ</div>')


def build_cme_html(cme: dict) -> str:
    """บล็อก CME/COT สำหรับอีเมล — inline style ล้วน"""
    if not cme:
        return ""
    parts = []

    for s in cme.get("cot", []):
        import cme_reports as cr
        rows = []
        for r in cr.cot_rows(s):
            col = {"NET LONG": "#2ecc71", "NET LONG ⚠": "#f39c12",
                   "NET SHORT": "#e74c3c", "NET SHORT ⚠": "#e67e22"}.get(r["status"], DIM)
            note = r["note"].replace("**", "")
            rows.append(
                f'<tr><td style="padding:7px 10px;border-bottom:1px solid {LINE};'
                f'color:#8aadee;font-size:12px;font-weight:600;white-space:nowrap">{r["label"]}</td>'
                f'<td style="padding:7px 10px;border-bottom:1px solid {LINE};color:{TXT};'
                f'font-family:monospace;font-size:14px;font-weight:600;text-align:right">{r["value"]}</td>'
                f'<td style="padding:7px 10px;border-bottom:1px solid {LINE};text-align:center">'
                f'<span style="color:{col};border:1px solid {col};border-radius:4px;padding:2px 7px;'
                f'font-size:10px;font-weight:700;white-space:nowrap">{r["status"]}</span></td>'
                f'<td style="padding:7px 10px;border-bottom:1px solid {LINE};color:{DIM};'
                f'font-size:11px;line-height:1.4">{note}</td></tr>')
        oi = f'{s["open_interest"]:,}'
        chg = s.get("oi_change")
        parts.append(
            f'<div style="padding:0 10px 8px"><span style="color:{TXT};font-size:15px;'
            f'font-weight:700">{s["label"]}</span>'
            f'<div style="color:{DIM};font-size:11px;margin-top:3px">'
            f'report {s["date"]:%Y-%m-%d} · OI {oi}'
            + (f' ({chg:+,})' if chg is not None else '') + '</div></div>'
            f'<table style="width:100%;border-collapse:collapse;margin-bottom:14px">'
            f'{"".join(rows)}</table>')

    rates = {k: v for k, v in cme.get("rates", {}).items() if isinstance(v, dict)}
    p = cme.get("policy", {})
    chips = []
    for k, v in rates.items():
        chips.append(f'<span style="display:inline-block;border:1px solid {LINE};'
                     f'border-radius:5px;padding:5px 10px;margin:0 6px 6px 0">'
                     f'<span style="color:{DIM};font-size:10px">{k.upper()}</span> '
                     f'<span style="color:{TXT};font-family:monospace;font-size:13px;'
                     f'font-weight:600">{v["rate"]:.2f}%</span></span>')
    if p.get("implied") is not None:
        note = (p.get("note") or "").replace("**", "")
        chips.append(f'<span style="display:inline-block;border:1px solid {LINE};'
                     f'border-radius:5px;padding:5px 10px;margin:0 6px 6px 0">'
                     f'<span style="color:{DIM};font-size:10px">FF futures implied</span> '
                     f'<span style="color:{TXT};font-family:monospace;font-size:13px;'
                     f'font-weight:600">{p["implied"]:.3f}%</span>'
                     + (f' <span style="color:{DIM};font-size:10px">'
                        f'({p["spread_bp"]:+.1f}bp · {note})</span>'
                        if p.get("spread_bp") is not None else '') + '</span>')
    if chips:
        parts.append(f'<div style="padding:4px 10px 10px">{"".join(chips)}</div>')

    if not parts:
        return ""
    return (f'<div style="background:{CARD};border:1px solid {LINE};border-radius:8px;'
            f'padding:14px 8px;margin-bottom:20px">'
            f'<div style="color:{TXT};font-size:17px;font-weight:700;padding:0 10px 10px">'
            f'🏛️ CME / Macro</div>{"".join(parts)}'
            f'<div style="color:{DIM};font-size:10.5px;padding:6px 10px 0;'
            f'border-top:1px solid {LINE};line-height:1.55">'
            f'COT จาก CFTC (รายสัปดาห์ · ช้า 3 วัน) · อัตราจาก NY Fed · futures จาก yfinance<br>'
            f'เครื่องมือของ CME เอง (FedWatch / QuikStrike / Daily Volume Report / Term SOFR) '
            f'ดึงอัตโนมัติไม่ได้ — CME ห้าม scraping ใน Terms of Use</div></div>')


def build_html(snaps: list[dict], cme: dict | None = None,
               gates: list[dict] | None = None) -> str:
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
        f'{build_gate_html(gates or [])}'
        f'<div style="color:{TXT};font-size:22px;font-weight:700;margin-bottom:4px">'
        f'📋 Signal Recap</div>'
        f'<div style="color:{DIM};font-size:12px;margin-bottom:18px">'
        f'{datetime.now():%Y-%m-%d %H:%M} · CBOE delayed (~15 นาที) + yfinance</div>'
        f'{"".join(blocks)}'
        f'{build_cme_html(cme) if cme else ""}'
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

    pw = pw.replace(" ", "").replace(" ", "")   # Google โชว์เป็น 4 กลุ่มมีเว้นวรรค
    gmail = "gmail" in _env("SMTP_HOST", "smtp.gmail.com")

    # ตรวจก่อนยิง — บอกสาเหตุได้ตรงกว่ารอ error 535 จาก server
    if "@" not in user:
        raise SystemExit(
            f"SMTP_USER = {user!r} ไม่ใช่อีเมลเต็ม\n"
            "  ต้องใส่ทั้ง @gmail.com เช่น you@gmail.com ไม่ใช่แค่ you")
    if gmail and len(pw) != 16:
        print(f"⚠️  SMTP_PASS ยาว {len(pw)} ตัวอักษร แต่ App Password ของ Google ยาว 16 ตัวพอดี\n"
              f"   ถ้าใช้รหัสผ่าน Gmail ปกติ Google จะปฏิเสธด้วย error 535 แน่นอน",
              file=sys.stderr, flush=True)
    elif gmail and not pw.isalpha():
        print("⚠️  App Password ของ Google เป็นตัวอักษร a-z ล้วน 16 ตัว "
              "แต่ค่าที่ใส่มามีตัวเลข/สัญลักษณ์ปน — เช็คว่าคัดลอกถูกตัวมั้ย",
              file=sys.stderr, flush=True)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    host = _env("SMTP_HOST", "smtp.gmail.com")
    port = int(_env("SMTP_PORT", "587"))
    # flush เพื่อให้บรรทัดนี้ออกก่อน error เสมอ (GitHub Actions สลับ stdout/stderr ได้)
    print(f"เชื่อมต่อ {host}:{port} · ผู้ส่ง {user} · ผู้รับ {', '.join(to)} · "
          f"รหัสยาว {len(pw)} ตัว", flush=True)
    try:
        with smtplib.SMTP(host, port, timeout=45) as srv:
            srv.starttls()
            srv.login(user, pw)
            srv.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        detail = e.smtp_error.decode("utf-8", "replace") if isinstance(e.smtp_error, bytes) \
            else str(e.smtp_error)
        low = detail.lower()
        if "application-specific password" in low or "app password" in low:
            why = ("👉 Google บอกตรง ๆ ว่าต้องใช้ **App Password** — ค่าที่ใส่คือรหัสผ่าน Gmail ปกติ\n"
                   "   สร้างที่ Google Account → Security → 2-Step Verification → App passwords")
        elif "username and password not accepted" in low:
            why = ("👉 user หรือ password ไม่ตรง — สาเหตุที่เจอบ่อยเรียงตามความน่าจะเป็น:\n"
                   "   1) SMTP_PASS เป็นรหัสผ่าน Gmail ปกติ ไม่ใช่ App Password 16 ตัว\n"
                   "   2) App Password สร้างจาก Google account คนละอันกับ SMTP_USER\n"
                   "   3) App Password ถูกลบ/เพิกถอนไปแล้ว → สร้างใหม่\n"
                   "   4) คัดลอกไม่ครบ (ต้อง 16 ตัว ไม่นับเว้นวรรค)")
        elif "smtp not enabled" in low or "not enabled for smtp" in low or "disabled" in low:
            why = "👉 บัญชีนี้ถูกปิดการใช้ SMTP (Google Workspace ต้องให้ admin เปิดให้)"
        else:
            why = "👉 ดูข้อความจาก Google ด้านบนประกอบ"
        raise SystemExit(
            f"❌ Gmail ปฏิเสธการเข้าสู่ระบบ (SMTP {e.smtp_code})\n\n"
            f"ข้อความจาก Google:\n  {detail}\n\n{why}")
    except smtplib.SMTPException as e:
        raise SystemExit(f"❌ ส่งไม่สำเร็จ: {type(e).__name__}: {e}")
    except (OSError, TimeoutError) as e:
        raise SystemExit(
            f"❌ ต่อ {host}:{port} ไม่ได้ — {type(e).__name__}: {e}\n"
            "👉 มักเกิดจากเน็ตออฟฟิศ/ไฟร์วอลล์บล็อก outbound port 587\n"
            "   ลองรันบน GitHub Actions แทน (เน็ตของ GitHub ไม่บล็อก) "
            "หรือใช้ SMTP_PORT=465 ถ้าเครือข่ายอนุญาต")
    print(f"✅ ส่งแล้ว → {', '.join(to)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="คำนวณและเขียนไฟล์ preview.html แต่ไม่ส่งเมล")
    ap.add_argument("--test-smtp", action="store_true",
                    help="ทดสอบล็อกอิน + ส่งเมลสั้น ๆ อย่างเดียว (ข้ามการคำนวณ) — ใช้ debug")
    ap.add_argument("--symbols", default=os.environ.get("SYMBOLS", "QQQ"))
    ap.add_argument("--no-cme", action="store_true",
                    help="ไม่ต้องแนบส่วน CME/COT")
    ap.add_argument("--no-gate", action="store_true",
                    help="ไม่ต้องแนบบล็อก 'วันนี้เข้าได้ไหม' (gate.py)")
    ap.add_argument("--armed-only", action="store_true",
                    help="ส่งเมลเฉพาะตอนมีตัวไหนขึ้น ARMED — ใช้กับ cron ระหว่างวันที่ยิงถี่")
    ap.add_argument("--only-on-change", action="store_true",
                    help="ส่งเฉพาะตอน verdict ต่างจากรอบก่อน (ต้องมี --state-file) "
                         "— กันเมลซ้ำตอน ARMED ค้างอยู่หลายรอบ")
    ap.add_argument("--state-file", default=os.environ.get("GATE_STATE_FILE", ""),
                    help="ไฟล์ JSON เก็บ verdict ล่าสุดต่อ symbol")
    ap.add_argument("--cot-markets",
                    default=os.environ.get("COT_MARKETS", "NASDAQ-100 (รวม),VIX Futures"),
                    help="ตลาด COT คั่นด้วย , (ดูรายชื่อใน cme_reports.COT_MARKETS)")
    args = ap.parse_args()

    if args.test_smtp:
        send("<html><body style='font-family:sans-serif'>"
             "<h3>✅ SMTP ใช้งานได้</h3><p>ถ้าเห็นเมลนี้ แปลว่าตั้งค่าถูกแล้ว "
             "รายงานจริงจะส่งตามเวลาที่ตั้งใน workflow</p></body></html>",
             "SMTP ใช้งานได้ — ตั้งค่าถูกแล้ว",
             "[TEST] Model_IV — ทดสอบการส่งเมล")
        return 0

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

    # ── CME/Macro (COT + rates + futures) — ล้มเหลวได้ ไม่ทำให้เมลไม่ถูกส่ง ──
    cme = None
    if not args.no_cme:
        try:
            import cme_reports
            mkts = [m.strip() for m in args.cot_markets.split(",") if m.strip()]
            cme = cme_reports.build_cme_snapshot(mkts, weeks=52)
            print(f"  CME: COT {len(cme.get('cot', []))} ตลาด · "
                  f"rates {len([1 for v in cme.get('rates', {}).values() if isinstance(v, dict)])}")
        except Exception as e:
            print(f"  CME: ข้ามไป ({type(e).__name__}: {e})", file=sys.stderr)

    # ── ประตูก่อนเข้าไม้ — ล้มเหลวได้ ไม่ทำให้ Signal Recap ไม่ถูกส่ง ──
    gates = []
    if not args.no_gate:
        for s in ok:
            try:
                gates.append(gate.evaluate(s))
            except Exception as e:
                print(f"  gate {s['symbol']}: ข้ามไป ({type(e).__name__}: {e})", file=sys.stderr)
        for g in gates:
            print(f"  gate {g['symbol']}: {g['verdict']} — {g['headline']}")

    # ── สถานะรอบก่อน — ต้องเขียนกลับทุกทางออก ไม่งั้นการตรวจ "เปลี่ยนสถานะ" เพี้ยน ──
    prev_state = {}
    if args.state_file:
        try:
            with open(args.state_file, encoding="utf-8") as f:
                prev_state = json.load(f)
        except (OSError, ValueError):
            prev_state = {}          # รอบแรก / ไฟล์เสีย = ถือว่าไม่เคยมีสถานะ
    # ปัญหาข้อมูลถือเป็นสถานะแยก ไม่ใช่ STAND_DOWN — ไม่งั้น pipeline พังแล้วเงียบสนิท
    # และเก็บลง state ด้วย เพื่อให้ --only-on-change เตือนครั้งเดียวตอนเริ่มพัง ไม่ใช่ทุก 15 นาที
    cur_state = {g["symbol"]: ("DATA_ISSUE" if g.get("data_issue") else g["verdict"])
                 for g in gates}

    def save_state():
        if not args.state_file or not cur_state:
            return
        try:
            with open(args.state_file, "w", encoding="utf-8") as f:
                json.dump(cur_state, f, ensure_ascii=False)
        except OSError as e:
            print(f"เขียน state file ไม่ได้: {e}", file=sys.stderr)

    armed = any(g["verdict"] == "ARMED" for g in gates)
    data_bad = any(g.get("data_issue") for g in gates)
    if args.armed_only and not armed and not data_bad:
        print("--armed-only: ไม่มีตัวไหนขึ้น ARMED — ไม่ส่งเมล")
        save_state()
        return 0
    if data_bad and not armed:
        # ผ่านด่าน --armed-only มาได้เพราะข้อมูลมีปัญหา ไม่ใช่เพราะเข้าเกณฑ์
        # (--only-on-change ด้านล่างยังหยุดได้อีกชั้น ถ้าพังแบบเดิมมาตั้งแต่รอบก่อน)
        print("ตรวจพบปัญหาข้อมูล ผ่านด่าน --armed-only: "
              + ", ".join(f"{g['symbol']}: {g['reason'][:60]}" for g in gates if g.get("data_issue")))

    if args.only_on_change and cur_state:
        if all(prev_state.get(s) == v for s, v in cur_state.items()):
            print(f"--only-on-change: verdict เท่ารอบก่อน ({cur_state}) — ไม่ส่งเมล")
            save_state()
            return 0
        moved = {s: f"{prev_state.get(s, '—')}→{v}" for s, v in cur_state.items()
                 if prev_state.get(s) != v}
        print(f"--only-on-change: เปลี่ยนสถานะ {moved} — ส่งเมล")

    save_state()
    html = build_html(snaps, cme, gates)
    text = "\n\n".join(render_text(s) for s in snaps)
    if gates:
        text = ("=" * 78 + "\nวันนี้เข้าได้ไหม\n" + "=" * 78 + "\n"
                + "\n\n".join(gate.render_text(g) for g in gates)
                + "\n\n" + "=" * 78 + "\n\n" + text)
    if cme:
        try:
            import cme_reports
            lines = cme_reports.summary_lines(cme)
            if lines:
                text += "\n\n" + "=" * 78 + "\nCME / MACRO\n" + "=" * 78 + "\n" + "\n".join(lines)
        except Exception:
            pass
    tag = "⚠️ " if failed else ""
    head = ok[0]
    # หัวเรื่องขึ้นต้นด้วยคำตัดสิน — บนล็อกสกรีนมือถือเห็นแค่ 40 ตัวแรกก็พอตัดสินใจได้ว่าต้องเปิดไหม
    lead = ""
    if gates:
        bad = [g for g in gates if g.get("data_issue")]
        if bad:
            lead = f"⚠️ DATA {', '.join(g['symbol'] for g in bad)} · "
        else:
            rank = {"ARMED": 0, "WATCH": 1, "STAND_DOWN": 2}
            top = min(gates, key=lambda g: rank.get(g["verdict"], 9))
            icon = {"ARMED": "🎯", "WATCH": "👀", "STAND_DOWN": "⛔"}[top["verdict"]]
            lead = f"{icon} {GATE_WORD[top['verdict']]} {top['symbol']} · "
    subject = (f"{tag}{lead}{head['symbol']} ${head['spot']:,.2f} · "
               f"Signal Recap {', '.join(syms)} · {datetime.now():%d %b %H:%M}")

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
