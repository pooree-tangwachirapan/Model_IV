"""cme_tab.py — Tab "CME / Macro" : COT + อัตราอ้างอิง + futures"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import cme_reports as cr

BADGE = {
    "NET LONG": "#2ecc71", "NET LONG ⚠": "#f39c12",
    "NET SHORT": "#e74c3c", "NET SHORT ⚠": "#e67e22",
    "FLAT": "#7f8c9a",
}

CSS = """
<style>
.cot-row { display:grid; grid-template-columns:180px 130px 128px 1fr; gap:10px;
  align-items:center; padding:9px 12px; border-bottom:1px solid #16243f; }
.cot-row:hover { background:#0e1b33; }
.cot-l { color:#8aadee; font-size:12.5px; font-weight:600; }
.cot-v { color:#e8f2ff; font-family:monospace; font-size:15px; font-weight:600; }
.cot-b { font-size:10.5px; font-weight:700; letter-spacing:.4px; padding:3px 8px;
  border-radius:4px; text-align:center; border:1px solid currentColor; }
.cot-n { color:#7e93b5; font-size:11.5px; line-height:1.45; }
@media (max-width:760px){ .cot-row{grid-template-columns:1fr 1fr;} .cot-n{grid-column:1/-1;} }
</style>
"""


def _net_chart(g: dict, title: str) -> go.Figure:
    """เส้น net position ย้อนหลัง + แถบ percentile"""
    hist = g["history"]
    dates = g["dates"][-len(hist):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=hist, mode="lines", name=g["name"],
        line=dict(color="#00e0ff", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>net %{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    if len(hist) > 4:
        fig.add_hline(y=float(np.percentile(hist, 90)), line_color="#e74c3c",
                      line_dash="dot", line_width=1, annotation_text="p90")
        fig.add_hline(y=float(np.percentile(hist, 10)), line_color="#2ecc71",
                      line_dash="dot", line_width=1, annotation_text="p10")
    fig.add_trace(go.Scatter(
        x=[dates[-1]], y=[hist[-1]], mode="markers",
        marker=dict(size=10, color="#f1c40f", symbol="diamond"),
        name="ล่าสุด", hovertemplate="ล่าสุด %{y:,.0f}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
        title=title, xaxis_title="", yaxis_title="Net position (สัญญา)",
        font=dict(family="monospace", size=11, color="#c8d8f0"),
        height=300, margin=dict(l=10, r=10, t=42, b=10), showlegend=False)
    return fig


def render_cme_tab():
    st.subheader("🏛️ CME / Macro — COT · อัตราอ้างอิง · Futures")

    st.warning(
        "**ทำไมไม่มี FedWatch / QuikStrike / Daily Volume Report ของ CME โดยตรง**\n\n"
        "CME Group บล็อกการดึงข้อมูลอัตโนมัติ และระบุใน Data Terms of Use ว่าห้ามใช้ "
        "script/bot/scraper กับเว็บเขา (ยืนยันจากการยิงจริง — ทุก endpoint ตอบ 403 "
        "พร้อมข้อความห้าม) เครื่องมือที่เป็นของ CME เอง — **FedWatch Tool, Daily Volume & OI "
        "Report, Term SOFR, QuikStrike, Pace of Trading** — จึงดึงอัตโนมัติไม่ได้ "
        "ต้องเปิดดูบนเว็บ CME เองครับ\n\n"
        "แท็บนี้ใช้ **ข้อมูลต้นทางเดียวกันจากหน่วยงานที่เปิดให้เรียกอย่างเป็นทางการ** แทน: "
        "COT จาก CFTC · SOFR/EFFR จาก NY Fed · ราคา futures จาก yfinance")

    c1, c2, c3 = st.columns([2.2, 1.1, 1.1])
    picks = c1.multiselect("ตลาด COT", list(cr.COT_MARKETS.keys()),
                           default=["NASDAQ-100 (รวม)", "VIX Futures"], key="cme_mkts")
    weeks = c2.slider("ย้อนหลัง (สัปดาห์)", 12, 156, 52, key="cme_weeks",
                      help="ใช้คำนวณ percentile — ยิ่งยาว ยิ่งรู้ว่าสุดโต่งจริงมั้ย")
    go_btn = c3.button("🔄 ดึงข้อมูล", type="primary", key="cme_go", width="stretch")

    if go_btn or "cme_snap" not in st.session_state:
        if not picks:
            st.info("เลือกตลาดอย่างน้อย 1 ตัว")
            return
        with st.spinner("กำลังดึง COT (CFTC) + rates (NY Fed) + futures ..."):
            try:
                st.session_state["cme_snap"] = cr.build_cme_snapshot(picks, weeks=weeks)
            except Exception as e:
                st.error(f"ดึงข้อมูลไม่สำเร็จ: {type(e).__name__}: {e}")
                return

    snap = st.session_state.get("cme_snap")
    if not snap:
        return
    for err in snap.get("errors", []):
        st.caption(f"⚠️ {err}")

    st.markdown(CSS, unsafe_allow_html=True)

    # ── COT ──
    st.markdown("### 📊 Commitment of Traders (CFTC · รายสัปดาห์)")
    st.caption("รายงานทุกวันศุกร์ 15:30 ET สะท้อนโพซิชั่น ณ วันอังคารก่อนหน้า — "
               "**ข้อมูลช้า 3 วันเสมอ** ใช้ดูโครงสร้าง ไม่ใช่สัญญาณ intraday")

    for s in snap.get("cot", []):
        oi_chg = s.get("oi_change")
        st.markdown(
            f"**{s['label']}** · `{s['market']}`  \n"
            f"report {s['date']:%Y-%m-%d} · OI `{s['open_interest']:,}`"
            + (f" ({oi_chg:+,})" if oi_chg is not None else "")
            + f" · ข้อมูล {s['weeks']} สัปดาห์")

        rows = cr.cot_rows(s)
        html = []
        for r in rows:
            col = BADGE.get(r["status"], "#7f8c9a")
            note = r["note"].replace("**", "")
            html.append(
                f'<div class="cot-row"><div class="cot-l">{r["label"]}</div>'
                f'<div class="cot-v">{r["value"]}</div>'
                f'<div class="cot-b" style="color:{col}">{r["status"]}</div>'
                f'<div class="cot-n">{note}</div></div>')
        st.markdown("".join(html), unsafe_allow_html=True)

        lev = next((g for g in s["groups"] if g["name"] == "Leveraged Money"), None)
        if lev and len(lev["history"]) > 4:
            st.plotly_chart(_net_chart(lev, f"{s['label']} — Leveraged Money net position"),
                            width="stretch")
        st.divider()

    # ── Rates ──
    st.markdown("### 💵 อัตราอ้างอิง (NY Fed · ทางการ)")
    rates = {k: v for k, v in snap.get("rates", {}).items() if isinstance(v, dict)}
    if rates:
        cols = st.columns(len(rates))
        for col, (k, v) in zip(cols, rates.items()):
            col.metric(k.upper(), f"{v['rate']:.2f}%",
                       help=f"{v['label']} · {v['date']}"
                            + (f" · vol ${v['volume']}B" if v.get("volume") else ""))
        st.caption("แทน **CME Term SOFR** ที่เป็น licensed product ดึงอัตโนมัติไม่ได้ — "
                   "SOFR ที่นี่คือ overnight ตัวจริงจาก NY Fed (ต้นทางของ Term SOFR อีกที)")

    # ── Policy pricing ──
    p = snap.get("policy", {})
    if p.get("implied") is not None:
        st.markdown("### 🏦 ตลาดคิดราคาดอกเบี้ยไว้เท่าไหร่")
        c1, c2, c3 = st.columns(3)
        c1.metric("Implied (Fed Funds futures)", f"{p['implied']:.3f}%")
        c2.metric("EFFR จริง", f"{p['effr']:.2f}%" if p.get("effr") else "N/A")
        c3.metric("ส่วนต่าง", f"{p['spread_bp']:+.1f} bp" if p.get("spread_bp") is not None else "N/A")
        st.info(f"{p.get('note','')}\n\n"
                "⚠️ **นี่ไม่ใช่ FedWatch** — FedWatch ใช้ Fed Funds futures **ทั้ง curve รายเดือน** "
                "มากระจายเป็น % ความน่าจะเป็นของแต่ละการประชุม เรามีแค่ front month ต่อเนื่อง "
                "จึงบอกได้แค่**ทิศทางรวมระยะใกล้** ไม่ใช่ % ต่อการประชุม "
                "ถ้าต้องการตัวเลขต่อการประชุมจริง ๆ ต้องเปิด FedWatch บนเว็บ CME")

    # ── Futures ──
    fut = {k: v for k, v in snap.get("futures", {}).items() if isinstance(v, dict)}
    if fut:
        st.markdown("### 📈 Futures (ราคา + volume)")
        df = pd.DataFrame([{
            "Symbol": k, "สินค้า": v["label"], "ราคา": f"{v['last']:,.4f}",
            "เปลี่ยน %": f"{v['chg_pct']:+.2f}%", "Volume": f"{v['volume']:,}",
            "เฉลี่ย 5 วัน": f"{v['vol_avg5']:,.0f}",
            "vs เฉลี่ย": f"{v['volume']/v['vol_avg5']*100:.0f}%" if v["vol_avg5"] else "—",
            "วันที่": v["date"],
        } for k, v in fut.items()])
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption("แทน **CME Daily Volume & Open Interest Report** ที่ดึงอัตโนมัติไม่ได้ — "
                   "volume ที่นี่มาจาก yfinance เป็นสัญญา front month ไม่ใช่ทั้ง complex "
                   "และ**ไม่มี open interest** (ตัวเลข OI ของ futures อยู่ในรายงาน CME เท่านั้น "
                   "ส่วน OI ของ options เราได้จาก CBOE อยู่แล้วในแท็บอื่น)")

    st.download_button(
        "⬇️ ดาวน์โหลดสรุป CME (text)",
        data="\n".join(cr.summary_lines(snap)),
        file_name=f"cme_{snap['asof']:%Y%m%d_%H%M}.txt",
        mime="text/plain", width="stretch")
