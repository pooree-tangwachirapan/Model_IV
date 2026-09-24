"""
zero_dte_report.py — ทำรายงาน HTML จากข้อมูล 0DTE ที่ zero_dte.py เก็บไว้

เป้าหมายของรายงานนี้คือ **ตั้งคำถามเรื่อง edge ให้ตอบได้** ไม่ใช่บอกว่าให้เทรดอะไร
(§9 ของ HANDOFF: ห้ามเชียร์ว่าระบบแม่นยำ · ห้ามตั้ง threshold ก่อนมีข้อมูล)
ทุกกราฟจึงเป็นการ "แสดงสิ่งที่เกิดขึ้น" ไม่มีสัญญาณซื้อขายในไฟล์นี้

กราฟที่มีและเหตุผลที่เลือก
────────────────────────────────────────────────────────────────────────
1. เส้นราคา underlying ตลอดวัน            — บริบท ดูอย่างอื่นไม่รู้เรื่องถ้าไม่มีอันนี้
2. ปริมาณต่อ strike (call vs put)          — "คนเล่นที่ไหน" ของทั้งวัน
3. พรีเมียมที่เปลี่ยนมือต่อ strike          — "เงินไปไหน" ต่างจากจำนวนสัญญา
                                             เพราะ 1,000 สัญญาที่ $0.05 ไม่เท่า 100 สัญญาที่ $5
4. ราคา mid ของ strike ที่คึกที่สุด         — นี่คือ "กราฟราคา" ของตัว option เอง
5. heatmap ปริมาณ (strike × เวลา)           — เห็นการย้ายฐานของ flow ระหว่างวัน
6. put/call ratio ตามเวลา                   — regime ภายในวัน
7. ตารางต่อ strike (OHLC วัน · vol · OI · IV)

กราฟใช้ Plotly โหลดจาก CDN (`include_plotlyjs="cdn"`) — ไฟล์เล็ก ~300 KB แทน ~4 MB
ข้อแลก: ต้องต่อเน็ตตอนเปิดรายงาน ถ้าอยากได้ไฟล์ที่เปิดออฟไลน์ได้ ใช้ `--standalone`

หมายเหตุเรื่องหน่วย: พรีเมียมคูณ 100 เพราะ 1 สัญญา = 100 หุ้น (ETF option มาตรฐาน)
"""

from __future__ import annotations

import html
import os
from datetime import date, datetime, timezone

import pandas as pd

import market_clock as mc
import zero_dte as z

TOP_STRIKES = 8          # กี่ strike ในกราฟราคา — มากกว่านี้เส้นทับกันอ่านไม่ออก


def _fig_html(fig, *, standalone: bool, first: bool) -> str:
    """แปลง figure → HTML · ฝัง plotly.js แค่ครั้งเดียวในไฟล์"""
    if standalone:
        js = True if first else False
    else:
        js = "cdn" if first else False
    return fig.to_html(full_html=False, include_plotlyjs=js,
                       config={"displaylogo": False, "responsive": True})


def _empty_page(sym: str, day: str, reason: str) -> str:
    return f"""<!doctype html><html lang="th"><meta charset="utf-8">
<title>0DTE {html.escape(sym)} {html.escape(day)}</title>
<body style="font-family:system-ui,sans-serif;max-width:720px;margin:48px auto;padding:0 16px">
<h1>ยังไม่มีข้อมูล</h1>
<p>{html.escape(sym)} · {html.escape(day)}</p>
<p style="color:#b00">{html.escape(reason)}</p>
<p>ตัวบันทึกจะเก็บข้อมูลเมื่อ workflow <code>armed-alert</code> รันในวันทำการ
หรือกดปุ่ม <b>บันทึก snapshot เดี๋ยวนี้</b> ในแท็บ 0DTE ของแอป
หรือรัน <code>python zero_dte_cli.py record</code></p>
</body></html>"""


def build(sym: str, day: date | str, *, standalone: bool = False) -> str:
    """คืน HTML ของรายงานหนึ่งวัน (สตริงเดียว เขียนลงไฟล์ได้เลย)"""
    import plotly.graph_objects as go       # import ตอนใช้ — CI ที่ไม่ทำรายงานไม่ต้องมี

    day = str(day)
    df = z.session_frame(sym, day)
    if df.empty:
        return _empty_page(sym, day, "ไม่พบไฟล์ของวันนี้ หรือไฟล์ว่าง")

    s = z.day_summary(sym, day)
    parts: list[str] = []
    n = [0]                                  # ตัวนับ เพื่อฝัง plotly.js แค่ figure แรก

    def add(fig, height=380):
        fig.update_layout(height=height, margin=dict(l=50, r=20, t=44, b=44),
                          template="plotly_white", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2))
        parts.append(_fig_html(fig, standalone=standalone, first=(n[0] == 0)))
        n[0] += 1

    # ── 1. ราคา underlying ────────────────────────────────
    spot_path = df.groupby("ts", as_index=False).agg(spot=("spot", "first"))
    spot_path["et"] = spot_path["ts"].dt.tz_convert(mc.ET).dt.strftime("%H:%M")
    f = go.Figure(go.Scatter(x=spot_path["et"], y=spot_path["spot"],
                             mode="lines+markers", name=sym,
                             line=dict(width=2, color="#1f77b4")))
    f.update_layout(title=f"1 · ราคา {sym} ตลอดวัน (เวลา ET · CBOE ดีเลย์ ~15 นาที)",
                    yaxis_title="ราคา ($)", xaxis_title="เวลา ET")
    add(f)

    # ── 2. ปริมาณต่อ strike ───────────────────────────────
    by = (df.groupby(["strike", "cp"], as_index=False)
            .agg(vol=("vol_delta", "sum"), notional=("notional", "sum")))
    c = by[by["cp"] == "C"].sort_values("strike")
    p = by[by["cp"] == "P"].sort_values("strike")
    f = go.Figure()
    f.add_bar(x=c["strike"], y=c["vol"], name="Call", marker_color="#2ca02c")
    f.add_bar(x=p["strike"], y=p["vol"], name="Put", marker_color="#d62728")
    f.add_vline(x=s["spot_last"], line_dash="dash", line_color="#333",
                annotation_text=f"spot {s['spot_last']:,.2f}")
    f.update_layout(title="2 · ปริมาณที่เทรดต่อ strike (รวมทั้งวัน)",
                    barmode="group", xaxis_title="strike",
                    yaxis_title="สัญญา", hovermode="closest")
    add(f)

    # ── 3. พรีเมียมที่เปลี่ยนมือต่อ strike ─────────────────
    f = go.Figure()
    f.add_bar(x=c["strike"], y=c["notional"], name="Call", marker_color="#2ca02c")
    f.add_bar(x=p["strike"], y=p["notional"], name="Put", marker_color="#d62728")
    f.add_vline(x=s["spot_last"], line_dash="dash", line_color="#333")
    f.update_layout(title="3 · พรีเมียมที่เปลี่ยนมือต่อ strike ($ = สัญญา × mid × 100)",
                    barmode="group", xaxis_title="strike",
                    yaxis_title="USD", hovermode="closest")
    add(f)

    # ── 4. กราฟราคา option ของ strike ที่คึกที่สุด ─────────
    top = (by.sort_values("notional", ascending=False)
             .head(TOP_STRIKES)[["strike", "cp"]].itertuples(index=False))
    f = go.Figure()
    for strike, cp in top:
        d = df[(df["strike"] == strike) & (df["cp"] == cp)].sort_values("ts")
        if d["mid"].notna().sum() < 2:
            continue
        f.add_scatter(x=d["et"], y=d["mid"], mode="lines+markers",
                      name=f"{strike:g}{cp}")
    f.update_layout(title=f"4 · ราคา mid ของ {TOP_STRIKES} สัญญาที่มีพรีเมียมหมุนมากสุด",
                    xaxis_title="เวลา ET", yaxis_title="พรีเมียม ($/หุ้น)")
    add(f, 440)

    # ── 5. heatmap ปริมาณ (strike × เวลา) ─────────────────
    pivot = (df.pivot_table(index="strike", columns="et", values="vol_delta",
                            aggfunc="sum", fill_value=0)
               .sort_index(ascending=True))
    if not pivot.empty and pivot.shape[1] > 1:
        f = go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns),
                                 y=[f"{v:g}" for v in pivot.index],
                                 colorscale="YlOrRd",
                                 colorbar=dict(title="สัญญา")))
        f.update_layout(title="5 · ปริมาณในแต่ละช่วง แยกตาม strike (เข้ม = คึก)",
                        xaxis_title="เวลา ET", yaxis_title="strike",
                        hovermode="closest")
        add(f, max(420, min(900, 18 * len(pivot.index))))

    # ── 6. put/call ratio ตามเวลา ─────────────────────────
    per = (df.pivot_table(index="ts", columns="cp", values="vol_delta",
                          aggfunc="sum", fill_value=0).sort_index())
    if {"C", "P"} <= set(per.columns):
        per["et"] = per.index.tz_convert(mc.ET).strftime("%H:%M")
        per["pc"] = (per["P"] / per["C"].replace(0, pd.NA))
        f = go.Figure(go.Scatter(x=per["et"], y=per["pc"], mode="lines+markers",
                                 line=dict(color="#9467bd", width=2), name="P/C"))
        f.add_hline(y=1.0, line_dash="dot", line_color="#888",
                    annotation_text="สมดุล")
        f.update_layout(title="6 · Put/Call ratio ของปริมาณในแต่ละช่วง",
                        xaxis_title="เวลา ET", yaxis_title="put ÷ call")
        add(f, 320)

    # ── 7. ตารางต่อ strike ────────────────────────────────
    last_ts = df["ts"].max()
    last = df[df["ts"] == last_ts]
    tbl = (last[["strike", "cp", "day_open", "day_high", "day_low", "mid",
                 "volume", "open_interest", "iv", "delta", "gamma"]]
           .merge(by[["strike", "cp", "notional"]], on=["strike", "cp"], how="left")
           .sort_values("notional", ascending=False).head(40))
    rows = []
    for r in tbl.itertuples(index=False):
        def fm(v, nd=2):
            return "—" if pd.isna(v) else f"{v:,.{nd}f}"
        rows.append(
            f"<tr><td>{r.strike:g}</td><td>{r.cp}</td>"
            f"<td>{fm(r.day_open)}</td><td>{fm(r.day_high)}</td><td>{fm(r.day_low)}</td>"
            f"<td>{fm(r.mid)}</td><td>{fm(r.volume, 0)}</td><td>{fm(r.open_interest, 0)}</td>"
            f"<td>{fm(r.iv * 100 if pd.notna(r.iv) else r.iv, 1)}</td>"
            f"<td>{fm(r.delta, 3)}</td><td>{fm(r.gamma, 4)}</td>"
            f"<td>{fm(r.notional, 0)}</td></tr>")
    table = f"""<table>
<thead><tr><th>strike</th><th>C/P</th><th>เปิด</th><th>สูง</th><th>ต่ำ</th>
<th>mid ล่าสุด</th><th>vol</th><th>OI</th><th>IV %</th><th>delta</th><th>gamma</th>
<th>พรีเมียม $</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""

    # ── หัวรายงาน ─────────────────────────────────────────
    pc = "—" if s["pc_ratio"] is None else f"{s['pc_ratio']:.2f}"
    chg = s["spot_last"] - s["spot_open"]
    cards = [
        ("expiry", s["expiry"]),
        ("snapshot", f"{s['snapshots']} ครั้ง · {s['first_et']}–{s['last_et']} ET"),
        ("strike ที่เก็บ", f"{s['strikes']}"),
        ("spot", f"{s['spot_open']:,.2f} → {s['spot_last']:,.2f} ({chg:+.2f})"),
        ("ช่วงราคา", f"{s['spot_low']:,.2f} – {s['spot_high']:,.2f}"),
        ("ปริมาณ C / P", f"{s['call_volume']:,.0f} / {s['put_volume']:,.0f}"),
        ("P/C ratio", pc),
        ("พรีเมียมรวม", f"${s['notional_usd']:,.0f}"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(str(v))}</div></div>' for k, v in cards)

    days = z.available_days(sym)
    others = "".join(f"<li>{html.escape(d)}</li>" for d in days[-15:])
    built = datetime.now(timezone.utc).astimezone(mc.ET)

    return f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>0DTE {html.escape(sym)} · {html.escape(day)}</title>
<style>
 :root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#f7f7f9}}
 @media(prefers-color-scheme:dark){{:root{{--bg:#14161a;--fg:#e9e9ea;--mut:#9aa0a6;
   --line:#2b2f36;--card:#1c1f25}}}}
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);
   color:var(--fg);margin:0;padding:24px 16px;line-height:1.55}}
 main{{max-width:1120px;margin:0 auto}}
 h1{{font-size:1.5rem;margin:0 0 4px}} h2{{font-size:1.05rem;margin:32px 0 8px}}
 .sub{{color:var(--mut);margin:0 0 20px;font-size:.9rem}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
 .k{{color:var(--mut);font-size:.76rem;text-transform:uppercase;letter-spacing:.04em}}
 .v{{font-size:1.02rem;font-weight:600;margin-top:2px}}
 .fig{{margin:18px 0;border:1px solid var(--line);border-radius:10px;overflow:hidden}}
 table{{width:100%;border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums}}
 th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}}
 th:nth-child(2),td:nth-child(2){{text-align:center}}
 th{{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--bg)}}
 .note{{background:var(--card);border-left:3px solid #f0ad4e;padding:10px 14px;
   border-radius:0 8px 8px 0;font-size:.87rem;margin:20px 0}}
 ul{{columns:3;font-size:.85rem;color:var(--mut)}}
 @media(max-width:640px){{ul{{columns:1}}}}
</style></head><body><main>
<h1>0DTE Option — {html.escape(sym)}</h1>
<p class="sub">วัน {html.escape(day)} · expiry {html.escape(s['expiry'])} ·
 {s['rows']:,} แถว · สร้างรายงาน {built:%Y-%m-%d %H:%M} ET</p>
<div class="grid">{card_html}</div>

<div class="note"><b>อ่านอย่างไร:</b> นี่คือข้อมูลที่บันทึกไว้ ไม่ใช่สัญญาณ ·
CBOE ดีเลย์ ~15 นาที ค่าที่ติดป้ายเวลา 10:00 คือภาพราว 09:45 ·
ความละเอียดเท่ากับความถี่ที่บันทึก ({s['snapshots']} ครั้งในวันนี้) ·
volume ที่แสดงเป็น <i>ส่วนต่างระหว่าง snapshot</i> แล้ว ไม่ใช่ยอดสะสม</div>

{''.join(f'<div class="fig">{p}</div>' for p in parts)}

<h2>7 · ตารางต่อ strike (40 อันดับแรกตามพรีเมียมที่หมุน · ค่าจาก snapshot สุดท้าย)</h2>
<div style="overflow:auto;max-height:560px">{table}</div>

<h2>วันที่มีข้อมูลแล้ว ({len(days)} วัน)</h2>
<ul>{others}</ul>
</main></body></html>"""


def write(sym: str, day: date | str, out_path: str | None = None, *,
          standalone: bool = False) -> str:
    """สร้างรายงานแล้วเขียนลงไฟล์ · คืน path"""
    day = str(day)
    out_path = out_path or os.path.join(z.ZDTE_DIR, sym.upper(),
                                        f"report-{day}.html")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(build(sym, day, standalone=standalone))
    return out_path


__all__ = ["build", "write", "TOP_STRIKES"]
