"""
pnl_tab.py — Tab "P&L" : ดูผล forward test ระหว่างทาง ไม่ต้องรอสิ้นเดือน

อ่าน forward_test/ledger.json ที่ GitHub Actions เขียนไว้ — ไฟล์นี้อยู่ในรีโป
เพราะฉะนั้นทุกครั้งที่ Streamlit Cloud redeploy จะได้ ledger ล่าสุดติดมาด้วย
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forward_test as ft


def _stat_row(s: dict):
    a, b, c, d = st.columns(4)
    a.metric("ไม้ที่ปิดแล้ว", s["n_closed"],
             delta=f"ค้าง {s['n_open']}" if s["n_open"] else None, delta_color="off")
    b.metric("Win rate", f"{s['win_rate']*100:.1f}%" if s["win_rate"] is not None else "—",
             help=(f"ต้องเกิน {s['breakeven_wr']*100:.1f}% ถึงจะเท่าทุนที่ RR นี้"
                   if s["breakeven_wr"] else None))
    c.metric("Expectancy /ไม้", f"{s['expectancy_r']:+.2f}R" if s["expectancy_r"] is not None else "—")
    d.metric("P&L", f"${s['pnl_usd']:+,.2f}",
             delta=f"{s['return_pct']:+.2f}%" if s["n_closed"] else None)


def render_pnl_tab():
    st.subheader("📊 P&L — Forward Test")

    # สองระบบเดินคู่กัน แต่ **ห้ามรวมสถิติ** — fade ชนะบ่อยแพ้ใหญ่ / breakout แพ้บ่อยชนะใหญ่
    # รวมกันแล้ว win rate กับ expectancy อ่านไม่ได้ทั้งคู่
    labels = {k: v["label"] for k, v in ft.SYSTEMS.items()}
    system = st.radio("ระบบ", list(ft.SYSTEMS), horizontal=True, key="pnl_system",
                      format_func=lambda k: labels[k])
    cfg = ft.system_cfg(system)

    st.caption(f"{cfg['label']} · {ft.SYMBOL} · พอร์ต ${ft.ACCOUNT_START:,.0f} · "
               f"**ไม้ละ {ft.QTY} หน่วยเท่ากันทั้ง LONG/SHORT** (1 จุด = ${ft.QTY}) · "
               f"สูงสุด {ft.MAX_TRADES_PER_DAY} ไม้/วัน · ถือไม่เกิน {ft.MAX_HOLD_DAYS} วันทำการ · "
               + ("เข้าด้วย **limit ที่กำแพง** (ไม่แตะ = ไม่ได้ไม้)" if cfg["entry"] == "limit"
                  else "เข้าด้วย **market** ที่ราคาเปิดแท่งถัดไป (กิน slippage เต็ม ๆ)"))
    st.caption("⚠️ จำลองถือ underlying ตรง ๆ **ไม่ได้จำลอง option** — ไม่มี theta / IV crush / สเปรด · "
               "ตอบว่า *สัญญาณถูกทางมั้ย* ไม่ได้ตอบว่า *เทรด option ตามนี้แล้วได้เท่านี้*")
    st.caption("⚠️ **ห้ามเทียบ win rate ข้ามระบบ** — สองระบบนี้ออกแบบมาให้ชนะคนละแบบ "
               "เทียบกันได้ที่ expectancy เท่านั้น")

    trades = ft.load(cfg["ledger"])
    if not trades:
        st.info(f"ยังไม่มีข้อมูลใน `{cfg['ledger']}` — "
                "จะเริ่มเขียนเมื่อ verdict ของระบบนี้ขึ้น ARMED ครั้งแรก")
        return

    months = sorted({t["date"][:7] for t in trades}, reverse=True)
    pick = st.selectbox("ช่วงเวลา", ["ทั้งหมด"] + months, index=0)
    month = None if pick == "ทั้งหมด" else pick
    s = ft.stats(trades, month)
    sel = [t for t in trades if month is None or t["date"].startswith(month)]

    _stat_row(s)

    if s["n_closed"] < 20:
        st.warning(f"**{s['n_closed']} ไม้ยังน้อยเกินกว่าจะสรุปว่ามี edge** — "
                   "ช่วงความเชื่อมั่นยังกว้างมาก อ่านตัวเลขนี้ว่า 'ระบบเดินได้' "
                   "ไม่ใช่ 'ระบบกำไร' และอย่าเพิ่งเอาไปปรับพารามิเตอร์")

    # ── equity curve ──
    if s["equity_curve"]:
        st.markdown("**เส้นพอร์ต**")
        eq = pd.DataFrame(s["equity_curve"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(eq) + 1)), y=eq["equity"], mode="lines+markers",
            line=dict(color="#00e0ff", width=2.5), marker=dict(size=6),
            customdata=eq[["date", "id"]],
            hovertemplate="ไม้ที่ %{x}<br>%{customdata[0]}<br>$%{y:,.2f}<extra></extra>"))
        fig.add_hline(y=ft.ACCOUNT_START, line_color="rgba(255,255,255,.35)",
                      line_dash="dash", annotation_text=f"ต้นทุน ${ft.ACCOUNT_START:,.0f}")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            xaxis_title="ไม้ที่ (เรียงตามเวลาปิด)", yaxis_title="พอร์ต (USD)",
            font=dict(family="monospace", size=11, color="#c8d8f0"),
            height=330, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        a.metric("Max drawdown", f"${s['max_dd_usd']:,.2f}")
        b.metric("RR จริง", f"1:{s['rr']:.2f}" if s["rr"] else "—",
                 help="กำไรเฉลี่ยตอนชนะ ÷ ขาดทุนเฉลี่ยตอนแพ้ (หน่วย R)")

    # ── ไม้ที่ยังค้าง ──
    op = [t for t in sel if t["status"] == "open"]
    if op:
        st.markdown("**ไม้ที่ยังไม่ปิด**")
        st.dataframe(pd.DataFrame([{
            "เปิดเมื่อ": t["opened"][:16].replace("T", " "), "ทิศ": t["side"],
            "เข้า": t["entry"], "invalidate": t["invalidation"], "เป้า": t["target"],
            "แผน R": round(t["plan_r"], 2), "หมายเหตุ": t.get("note", ""),
        } for t in op]), width="stretch", hide_index=True)

    # ── ไม้ที่ปิดแล้ว ──
    done = [t for t in sel if t.get("realized_r") is not None]
    if done:
        st.markdown("**ไม้ที่ปิดแล้ว**")
        st.dataframe(pd.DataFrame([{
            "วันที่": t["date"], "ทิศ": t["side"], "เข้า": t["entry"], "ออก": t["exit"],
            "invalidate": t["invalidation"], "เป้า": t["target"],
            "แผน R": round(t["plan_r"], 2), "R จริง": t["realized_r"],
            "P&L": t["pnl_usd"], "ผล": t["status"],
        } for t in sorted(done, key=lambda x: x.get("closed_at") or x["opened"], reverse=True)]),
            width="stretch", hide_index=True)

        st.download_button("⬇️ ดาวน์โหลดสรุป (text)", data=ft.render_text(s),
                           file_name=f"forward_test_{month or 'all'}.txt",
                           mime="text/plain", width="stretch")

    st.caption("⚠️ จำลองที่ราคา **underlying** เป็นหน่วย R ไม่ได้จำลองราคา option — "
               "ของจริงมี spread, IV crush และ fill ที่ไม่ตรงแผน **ผลจริงย่อมแย่กว่านี้** · "
               "แตะทั้งเป้าและ invalidate ในแท่งเดียวกันนับเป็นแพ้เสมอ")
