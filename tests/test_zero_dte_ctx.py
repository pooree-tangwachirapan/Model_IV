"""
ตรวจ zero_dte_ctx.py — บริบท VIX · VIX1D · VIX9D · VVIX ต่อ snapshot ของ 0DTE · ไม่แตะเน็ตเลย

สิ่งที่ต้องไม่พังเด็ดขาด (เหตุผลที่ไฟล์นี้มีอยู่):
  · ดึงไม่ได้ = ไม่เขียนอะไรเพิ่ม และไม่โยน exception (step ใน workflow ต้องไม่ทำ chain หาย)
  · ไฟล์ ctx ต้องไม่ไปโผล่ในที่ที่ตัวอ่าน chain มองหา — QuantLab กับ merge_data หยิบ *ทุก*
    .csv ที่อยู่ตรง zero_dte/<SYM>/ ถ้า ctx ไปอยู่ตรงนั้นจะถูกอ่านเป็น chain
  · schema ของ chain ไม่เปลี่ยน
  · merge_data._merge_csv อ่านแล้วเขียนไฟล์ ctx ใหม่ทุกรอบ commit → ไบต์ต้องไม่เปลี่ยนถ้าเนื้อไม่เปลี่ยน
JSON ปลอมลอกรูปจาก Yahoo ของจริง (^VIX1D 5 พ.ย. 2026 ดึงเมื่อ 06.11.2026)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta, timezone

import pandas as pd

import merge_data as md
import zero_dte as z
import zero_dte_cli as cli
import zero_dte_ctx as zc

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(name)


# ── JSON ของ Yahoo ที่ตัดมาจากของจริง (meta + 3 แท่งสุดท้ายของวัน) ──────────────
def yahoo(price=36.8, qtime=1793913301, closes=(33.31999969482422, 33.15999984741211,
                                                36.79999923706055),
          ts=(1793912280, 1793912340, 1793912400), symbol="^VIX1D"):
    meta = {"symbol": symbol, "exchangeName": "WCB", "dataGranularity": "1m",
            "chartPreviousClose": 12.08}
    if price is not None:
        meta["regularMarketPrice"] = price
    if qtime is not None:
        meta["regularMarketTime"] = qtime
    return {"chart": {"result": [{"meta": meta, "timestamp": list(ts),
                                  "indicators": {"quote": [{"close": list(closes)}]}}],
                      "error": None}}


DAY = "2026-11-05"
TS1 = datetime(2026, 11, 5, 21, 0, 56, tzinfo=timezone.utc)      # snapshot 16:00 ET
TS2 = datetime(2026, 11, 5, 21, 15, 39, tzinfo=timezone.utc)     # snapshot 16:15 ET (ของจริง)
NOW = TS2 + timedelta(seconds=8)                                 # step ctx วิ่งหลัง chain ไม่กี่วินาที
FETCHED = NOW + timedelta(seconds=2)


def fake_chain(spot):
    rows = []
    for k in (645, 650):
        for cp in ("C", "P"):
            rows.append({"option": f"QQQ261105{cp}{k * 1000:08d}", "bid": 1.0, "ask": 1.2,
                         "iv": 0.4, "open_interest": 100, "volume": 50, "delta": 0.5,
                         "gamma": 0.02, "vega": 0.1, "theta": -1.0, "open": 2.0, "high": 3.0,
                         "low": 0.5, "last_trade_price": 1.1, "prev_day_close": 2.2})
    return pd.DataFrame(rows)


def quotes(values, errors=None):
    return lambda: {"values": values, "errors": errors or {}, "fetched_utc": FETCHED}


ALL = {"vix": (27.79, "2026-11-05T21:15:01+00:00"), "vix1d": (36.8, "2026-11-05T21:15:01+00:00"),
       "vix9d": (32.66, "2026-11-05T21:15:01+00:00"), "vvix": (134.99, "2026-11-05T21:15:01+00:00")}


print("=== schema ===")
# schema ของ chain ต้องไม่ขยับ — QuantLab (quantlab/options_data.py) อ่านด้วยชื่อคอลัมน์ชุดนี้
check("schema ของ chain (zero_dte.COLUMNS) ไม่เปลี่ยน",
      z.COLUMNS == ["ts_utc", "sym", "expiry", "dte", "spot", "strike", "cp",
                    "bid", "ask", "mid", "last", "volume", "open_interest", "iv",
                    "delta", "gamma", "vega", "theta",
                    "day_open", "day_high", "day_low", "prev_close"])
check("schema ของ ctx: ts_utc · spot · เวลา fetch · ค่า+เวลา quote ครบ 4 ดัชนี · src",
      zc.COLUMNS == ["ts_utc", "spot", "fetched_utc", "vix", "vix_utc", "vix1d", "vix1d_utc",
                     "vix9d", "vix9d_utc", "vvix", "vvix_utc", "src"], str(zc.COLUMNS))

print("\n=== แกะ JSON ของ Yahoo ===")
check("meta → ค่าล่าสุด + เวลา quote ถึงวินาที (VIX1D ปิด 36.8 ที่ 16:15:01 ET)",
      zc.parse_chart(yahoo()) == (36.8, "2026-11-05T21:15:01+00:00"), str(zc.parse_chart(yahoo())))
check("meta ไม่มีราคา → ใช้แท่ง 1m สุดท้าย + เวลาเริ่มแท่ง (16:00 ET)",
      zc.parse_chart(yahoo(price=None)) == (36.8, "2026-11-05T21:00:00+00:00"),
      str(zc.parse_chart(yahoo(price=None))))
check("meta ไม่มีเวลา → ค่ากับเวลามาจากแท่งเดียวกัน ไม่ผสมกัน",
      zc.parse_chart(yahoo(price=99.0, qtime=None)) == (36.8, "2026-11-05T21:00:00+00:00"))
check("แท่งสุดท้ายว่าง → ถอยไปแท่งก่อนหน้าที่มีค่า",
      zc.parse_chart(yahoo(price=None, closes=(33.32, 33.16, None)))
      == (33.16, "2026-11-05T20:59:00+00:00"))
check("ราคา 0 / NaN ใน meta ไม่นับเป็นค่า",
      zc.parse_chart(yahoo(price=0, closes=(None, None, None))) == (None, None)
      and zc.parse_chart(yahoo(price=float("nan"), closes=(None, None, None))) == (None, None))
check("Yahoo ตอบ error (result = None) → (None, None)",
      zc.parse_chart({"chart": {"result": None,
                                "error": {"code": "Not Found", "description": "x"}}}) == (None, None))
check("ของมั่ว (None · list · สตริง · dict ว่าง) → (None, None) ไม่โยน exception",
      all(zc.parse_chart(x) == (None, None) for x in (None, [], "ไม่ใช่ json", {}, {"chart": {}})))

print("\n=== ดึง quote (http ปลอม) ===")
calls = []
def http_ok(url, timeout):
    calls.append(url)
    return yahoo()
q = zc.fetch_quotes(http=http_ok)
check("ได้ครบ 4 ดัชนี ไม่มี error", set(q["values"]) == set(zc.INDEXES) and not q["errors"],
      str(q["errors"]))
check("ยิงตัวละครั้ง · ใช้ interval=1m range=1d · ^ ถูก encode",
      len(calls) == 4 and all("interval=1m&range=1d" in u for u in calls)
      and any("%5EVIX1D" in u for u in calls), str(calls))

def http_partial(url, timeout):
    if "VVIX" in url:
        raise RuntimeError("HTTP 429 Too Many Requests")
    return yahoo()
q = zc.fetch_quotes(http=http_partial)
check("ตัวหนึ่งพัง → ตัวอื่นยังได้ · ตัวที่พังมีเหตุผลใน errors",
      set(q["values"]) == {"vix", "vix1d", "vix9d"} and "429" in q["errors"].get("vvix", ""),
      str(q))

seen = []
def http_flaky_host(url, timeout):
    seen.append(url)
    if "query1." in url:
        raise ConnectionError("reset")
    return yahoo()
q = zc.fetch_quotes(http=http_flaky_host)
check("โฮสต์แรกพัง → ลองโฮสต์ที่สองแล้วได้ค่า",
      len(q["values"]) == 4 and sum("query2." in u for u in seen) == 4, str(q["errors"]))

T = [0.0]
def http_slow(url, timeout):
    T[0] += 15.0                       # คำขอละ 15 วินาที
    return yahoo()
q = zc.fetch_quotes(http=http_slow, clock=lambda: T[0], deadline_s=20)
check("เกินเพดานเวลา → หยุดยิงตัวที่เหลือ (ไม่ทำให้ commit ช้า)",
      set(q["values"]) == {"vix", "vix1d"} and set(q["errors"]) == {"vix9d", "vvix"}
      and "เพดานเวลา" in q["errors"]["vix9d"], str(q))

def http_dead(url, timeout):
    raise TimeoutError("read timed out")
q = zc.fetch_quotes(http=http_dead)
check("พังทุกตัว → values ว่าง ไม่โยน exception",
      q["values"] == {} and set(q["errors"]) == set(zc.INDEXES))


# ════════════════════════════════════════════════
# ส่วนที่แตะไฟล์ — ทำในโฟลเดอร์ชั่วคราวทั้งหมด
# ════════════════════════════════════════════════
REAL = z.ZDTE_DIR
ROOT = tempfile.mkdtemp(prefix="zdte-ctx-test-")
z.ZDTE_DIR = os.path.join(ROOT, "zero_dte")
cwd = os.getcwd()
try:
    z.write_rows(z.build_rows(fake_chain(650.2), 650.2, ts=TS1, today=date(2026, 11, 5)))
    z.write_rows(z.build_rows(fake_chain(647.9), 647.9, ts=TS2, today=date(2026, 11, 5)))
    chain_path = z.path_for("QQQ", DAY)
    chain_before = open(chain_path, "rb").read()
    days_before = z.available_days("QQQ")

    print("\n=== snapshot ล่าสุดของ chain ===")
    s = zc.last_snapshot("QQQ", now=NOW)
    check("ได้ snapshot ล่าสุด (16:15) พร้อม spot ของ chain",
          s is not None and s["spot"] == 647.9 and s["day"] == DAY, str(s))
    check("ts_utc เป็นสตริงเดียวกับในไฟล์ chain ตรงตัว (join ได้)",
          s is not None and s["ts_utc"] == "2026-11-05T21:15:39+00:00"
          and s["ts_utc"] in set(pd.read_csv(chain_path, dtype=str)["ts_utc"]), str(s))
    check("snapshot เก่ากว่า 5 นาที (รอบนี้ chain ล้ม) → None",
          zc.last_snapshot("QQQ", now=TS2 + timedelta(minutes=10)) is None)
    check("ยังไม่มีไฟล์ chain ของวันนั้น → None",
          zc.last_snapshot("QQQ", now=NOW + timedelta(days=1)) is None)

    print("\n=== ไม่มี snapshot หรือดึงไม่ได้ → ไม่เขียนอะไรเลย ===")
    called = []
    def fetch_spy():
        called.append(1)
        return {"values": ALL, "errors": {}, "fetched_utc": FETCHED}
    r = zc.record(["QQQ"], fetch=fetch_spy, now=TS2 + timedelta(minutes=10))
    p = zc.ctx_path("QQQ", DAY)
    check("ไม่มี snapshot ของรอบนี้ → ไม่ยิงเน็ต ไม่สร้างไฟล์ · ok (chain step เตือนเองแล้ว)",
          r["ok"] and not called and not os.path.exists(p), str(r))

    r = zc.record(["QQQ"], fetch=quotes({}, {"vix": "x", "vix1d": "x", "vix9d": "x", "vvix": "x"}),
                  now=NOW)
    check("ดึงไม่ได้สักตัว → ok False · ไม่สร้างไฟล์ · ไม่สร้างโฟลเดอร์ ctx",
          not r["ok"] and not os.path.exists(p) and not os.path.exists(os.path.dirname(p)), str(r))

    def fetch_boom():
        raise RuntimeError("ระเบิดกลางทาง")
    try:
        r = zc.record(["QQQ"], fetch=fetch_boom, now=NOW)
        no_raise = True
    except Exception:                                          # noqa: BLE001
        no_raise = False
    check("fetch โยน exception → record ไม่โยนต่อ · ok False · ไม่เขียน",
          no_raise and not r["ok"] and not os.path.exists(p), str(r))

    print("\n=== เขียนแถว ===")
    r = zc.record(["QQQ"], fetch=quotes(ALL), now=NOW)
    check("เขียนลง zero_dte/QQQ/ctx/<วัน>.csv",
          r["ok"] and r["results"][0]["written"] == 1
          and os.path.normpath(r["results"][0]["path"]) == os.path.normpath(
              os.path.join(z.ZDTE_DIR, "QQQ", "ctx", f"{DAY}.csv")), str(r))
    d = pd.read_csv(p, dtype=str)
    check("หัวตาราง = schema · 1 แถว", list(d.columns) == zc.COLUMNS and len(d) == 1)
    row = d.iloc[0]
    check("ts_utc/spot ตรงกับ snapshot ของ chain",
          row["ts_utc"] == "2026-11-05T21:15:39+00:00" and float(row["spot"]) == 647.9, str(row))
    check("ค่า + เวลา quote ของทุกดัชนีถูกที่",
          float(row["vix1d"]) == 36.8 and row["vix1d_utc"] == "2026-11-05T21:15:01+00:00"
          and float(row["vvix"]) == 134.99 and float(row["vix"]) == 27.79, str(row.to_dict()))
    check("fetched_utc + src", row["fetched_utc"] == "2026-11-05T21:15:49+00:00"
          and row["src"] == "yahoo", str(row.to_dict()))
    raw = open(p, "rb").read()
    check("บรรทัดจบด้วย LF ล้วน (ไม่ใช่ CRLF)", b"\r\n" not in raw and raw.endswith(b"\n"))

    r = zc.record(["QQQ"], fetch=quotes(ALL), now=NOW)
    check("รอบเดิมซ้ำ → ไม่เพิ่มแถว", r["results"][0]["written"] == 0
          and len(pd.read_csv(p)) == 1, str(r))

    print("\n=== ได้บางตัว ===")
    TS3 = TS2 + timedelta(minutes=15)
    z.write_rows(z.build_rows(fake_chain(646.0), 646.0, ts=TS3, today=date(2026, 11, 5)))
    r = zc.record(["QQQ"], fetch=quotes({"vix1d": (37.1, "2026-11-05T21:30:01+00:00")},
                                        {"vix": "429", "vix9d": "429", "vvix": "429"}),
                  now=TS3 + timedelta(seconds=8))
    d = pd.read_csv(p, dtype=str)
    last = d.iloc[-1]
    check("VIX1D ได้ตัวเดียว → ยังเขียนแถว · ตัวที่ไม่ได้เป็นช่องว่าง",
          r["ok"] and len(d) == 2 and last["vix1d"] == "37.1" and pd.isna(last["vix"])
          and pd.isna(last["vvix_utc"]), str(last.to_dict()))
    check("errors ของตัวที่ไม่ได้ส่งกลับให้ CLI เตือน", set(r["errors"]) == {"vix", "vix9d", "vvix"})

    print("\n=== ตัวอ่าน chain ต้องมองไม่เห็นไฟล์ ctx ===")
    sym_dir = os.path.join(z.ZDTE_DIR, "QQQ")
    # กติกาเดียวกับ quantlab/options_data.py: ทุกไฟล์ .csv/.csv.gz ที่อยู่ตรง zero_dte/<SYM>/
    ql_like = sorted(f for f in os.listdir(sym_dir) if f.endswith((".csv", ".csv.gz")))
    check("รายการแบบ QuantLab เห็นแค่ไฟล์ chain", ql_like == [f"{DAY}.csv"], str(ql_like))
    check("available_days ไม่เปลี่ยน", z.available_days("QQQ") == days_before == [DAY])
    loaded = z.load("QQQ", DAY)
    check("load() ยังได้ schema ของ chain และแถวครบ 3 snapshot",
          list(loaded.columns) == z.COLUMNS and loaded["ts_utc"].nunique() == 3, str(loaded.shape))
    os.chdir(ROOT)                                    # merge_data อิง cwd เหมือนตอนรันใน workflow
    zfiles = [f.replace("\\", "/") for f in md._zdte_files()]
    check("merge_data._zdte_files() (ที่ workflow อื่นใช้ --save) ไม่หยิบ ctx",
          zfiles == [f"zero_dte/QQQ/{DAY}.csv"], str(zfiles))

    print("\n=== merge_data._merge_csv กับไฟล์ ctx (step commit ทำทุกรอบ) ===")
    stash = os.path.join(ROOT, "stash")
    rel = os.path.relpath(p, ROOT)
    os.makedirs(os.path.join(stash, os.path.dirname(rel)), exist_ok=True)
    shutil.copy2(p, os.path.join(stash, rel))
    before = open(p, "rb").read()
    added = md._merge_csv(os.path.join(stash, rel), rel)
    check("เนื้อเท่ากัน → merge แล้วไบต์ไม่เปลี่ยน (ไม่มี diff ปลอมทุก 15 นาที)",
          added == 0 and open(p, "rb").read() == before)

    # ฝั่งเรามีแถวใหม่ (รอบ 16:30) · remote ยังไม่มี — จำลอง reset --hard แล้ว merge
    z.ZDTE_DIR = os.path.join(stash, "zero_dte")
    TS4 = TS3 + timedelta(minutes=15)
    zc.write_row("QQQ", DAY, zc.build_row({"ts_utc": TS4.isoformat(), "spot": 645.5},
                                          {"values": ALL}, TS4 + timedelta(seconds=9)))
    z.ZDTE_DIR = os.path.join(ROOT, "zero_dte")
    added = md._merge_csv(os.path.join(stash, rel), rel)
    m = pd.read_csv(p, dtype=str)
    check("แถวใหม่ของเราเข้าไป · ไม่ซ้ำ · เรียงตามเวลา",
          added == 1 and len(m) == 3 and m["ts_utc"].is_unique
          and list(m["ts_utc"]) == sorted(m["ts_utc"]), str(list(m["ts_utc"])))
    check("ไฟล์ chain ไม่ถูกแตะเลยตลอดชุดนี้ (นอกจากที่เทสต์เขียนเอง 3 snapshot)",
          open(chain_path, "rb").read().startswith(chain_before))

    print("\n=== กันไฟล์เดิมผิดรูป ===")
    bad = zc.ctx_path("QQQ", "2026-11-06")
    os.makedirs(os.path.dirname(bad), exist_ok=True)
    with open(bad, "w", encoding="utf-8", newline="") as fh:
        fh.write("ts_utc,vix\n2026-11-06T15:00:00+00:00,20.0\n")
    snap_bad = {"ts_utc": "2026-11-06T15:15:00+00:00", "spot": 640.0}
    res = zc.write_row("QQQ", "2026-11-06", zc.build_row(snap_bad, {"values": ALL}, NOW))
    check("หัวตารางไม่ตรง schema → ไม่ต่อท้าย (คอลัมน์จะเลื่อน) · ไฟล์เดิมไม่ถูกแก้",
          res["written"] == 0 and open(bad, encoding="utf-8").read()
          == "ts_utc,vix\n2026-11-06T15:00:00+00:00,20.0\n", str(res))

    print("\n=== CLI: zero_dte_cli.py context ===")
    real_record = zc.record
    argv = sys.argv
    try:
        def run(ret, *args):
            zc.record = lambda syms, **kw: ret
            sys.argv = ["zero_dte_cli.py", "context", "--symbols", "QQQ", *args]
            return cli.main()
        full = {"ok": True, "values": ALL, "errors": {},
                "results": [{"sym": "QQQ", "written": 1, "path": "x", "reason": ""}]}
        part = {"ok": True, "values": {"vix1d": ALL["vix1d"]}, "errors": {"vvix": "429"},
                "results": [{"sym": "QQQ", "written": 1, "path": "x", "reason": ""}]}
        none = {"ok": False, "values": {}, "errors": {"vix": "x"},
                "results": [{"sym": "QQQ", "written": 0, "reason": "ดึงไม่ได้"}]}
        check("ได้ครบ → exit 0 แม้ใส่ --strict", run(full, "--strict") == 0)
        check("ได้บางตัว + --strict → exit 1 (workflow แปลงเป็น warning)", run(part, "--strict") == 1)
        check("ไม่ได้เลย + --strict → exit 1", run(none, "--strict") == 1)
        check("ไม่ใส่ --strict → exit 0 เสมอ (เหมือน record)", run(none) == 0 and run(part) == 0)
    finally:
        zc.record = real_record
        sys.argv = argv
finally:
    os.chdir(cwd)
    z.ZDTE_DIR = REAL
    shutil.rmtree(ROOT, ignore_errors=True)

check("คืน ZDTE_DIR ของจริงแล้ว (เทสต์ไม่เขียนทับข้อมูลรีโป)", z.ZDTE_DIR == REAL)

print("=" * 60)
print(f"ไม่ผ่าน {len(FAIL)} เคส" if FAIL else "ผ่านทั้งหมด")
sys.exit(1 if FAIL else 0)
