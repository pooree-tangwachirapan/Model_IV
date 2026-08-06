"""
dashboard.py — Tab "Dashboard" : signal recap หน้าเดียวจากแหล่งฟรีทั้งหมด
คำนวณอยู่ใน snapshot.py (ไม่มี UI) เพื่อให้สคริปต์อีเมลใช้ซ้ำได้
"""

import streamlit as st
import pandas as pd
import snapshot

# สีตามสถานะ — กลุ่มเดียวกันใช้สีเดียวกันทั้งแอปและอีเมล
STATUS_COLOR = {
    "TREND RISK": "#e74c3c", "STRESS": "#e74c3c",
    "KNIFE EDGE": "#e67e22", "ELEVATED": "#e67e22", "HIGH": "#e67e22",
    "PIN ZONE": "#2ecc71", "ALIGNED": "#2ecc71", "CALM": "#2ecc71", "RICH": "#2ecc71",
    "MAGNET": "#9b59b6", "DEFENSIVE": "#3498db", "CALL BID": "#3498db",
    "WATCH": "#f1c40f", "CHEAP": "#f39c12",
    "NEUTRAL": "#7f8c9a",
}

CSS = """
<style>
.sig-row {
  display: grid; grid-template-columns: 168px 210px 118px 1fr;
  gap: 10px; align-items: center;
  padding: 9px 12px; border-bottom: 1px solid #16243f;
}
.sig-row:hover { background: #0e1b33; }
.sig-label { color: #8aadee; font-size: 12.5px; font-weight: 600; }
.sig-value { color: #e8f2ff; font-family: monospace; font-size: 15px; font-weight: 600; }
.sig-badge {
  font-size: 10.5px; font-weight: 700; letter-spacing: .4px;
  padding: 3px 9px; border-radius: 4px; text-align: center;
  border: 1px solid currentColor;
}
.sig-note { color: #7e93b5; font-size: 11.5px; line-height: 1.45; }
.sig-head {
  display: grid; grid-template-columns: 168px 210px 118px 1fr; gap: 10px;
  padding: 6px 12px; color: #55708f; font-size: 10.5px;
  letter-spacing: .8px; border-bottom: 1px solid #1e3358;
}
@media (max-width: 760px) {
  .sig-row, .sig-head { grid-template-columns: 1fr 1fr; }
  .sig-note { grid-column: 1 / -1; }
}
</style>
"""


def render_dashboard_tab(sym: str, name: str):
    st.subheader("📋 Signal Recap — หน้าเดียวจบ")
    st.caption("ทุกค่าคำนวณจากแหล่งฟรี: CBOE delayed (chain + greeks) · yfinance (VIX/VVIX/realized vol) · "
               "ไม่มีค่าไหนต้องจ่ายเงิน")

    c1, c2, c3 = st.columns([1.4, 1.4, 1.2])
    win = c1.slider("หน้าต่าง strike รอบ spot (±%)", 3, 25, 10, key="dash_win",
                    help="ใช้กับ GEX/Walls/Flip — แคบ = เน้น level ใกล้ราคา")
    hv_override = c2.text_input("Ticker สำหรับ realized vol", value="",
                                key="dash_hv", placeholder="เว้นว่าง = ใช้ symbol เดียวกัน",
                                help="CBOE ใช้ _SPX แต่ yfinance ใช้ ^GSPC — แมปให้อัตโนมัติแล้ว")
    go = c3.button("🔄 คำนวณ Snapshot", type="primary", key="dash_go", width="stretch")

    if go or st.session_state.get("dash_snap_sym") != sym:
        with st.spinner(f"กำลังคำนวณ {sym} ..."):
            try:
                snap = snapshot.build_snapshot(sym, hv_ticker=hv_override.strip() or None,
                                               window_pct=float(win))
            except Exception as e:
                st.error(f"คำนวณไม่สำเร็จ: {type(e).__name__}: {e}")
                return
        st.session_state["dash_snap"] = snap
        st.session_state["dash_snap_sym"] = sym

    snap = st.session_state.get("dash_snap")
    if not snap:
        st.info("👆 กด **คำนวณ Snapshot** เพื่อเริ่ม")
        return
    if snap.get("error"):
        st.error(snap["error"])
        return

    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"**{snap['symbol']}** · spot `${snap['spot']:,.2f}` · "
        f"expiry ใกล้สุด `{snap['near_expiry']}` ({snap['near_dte']}d) · "
        f"`{snap['n_contracts']:,}` สัญญา · {snap['asof']:%Y-%m-%d %H:%M}")

    if snap["macro"].get("error"):
        st.caption(f"⚠️ ดึง macro ไม่ได้: {snap['macro']['error'][:90]}")
    if snap["rv"].get("error"):
        st.caption(f"⚠️ ดึง realized vol ไม่ได้: {snap['rv']['error'][:90]}")

    html = ['<div class="sig-head"><div>METRIC</div><div>VALUE</div>'
            '<div>STATUS</div><div>NOTE</div></div>']
    for r in snap["rows"]:
        col = STATUS_COLOR.get(r["status"], "#7f8c9a")
        html.append(
            f'<div class="sig-row">'
            f'<div class="sig-label">{r["label"]}</div>'
            f'<div class="sig-value">{r["value"]}</div>'
            f'<div class="sig-badge" style="color:{col}">{r["status"]}</div>'
            f'<div class="sig-note">{r["note"]}</div></div>')
    st.markdown("".join(html), unsafe_allow_html=True)

    st.divider()
    lv = snap["levels"]
    if lv.get("profile"):
        ladder, prof = lv["profile"]
        st.plotly_chart(
            __import__("fa_gex").plot_gamma_profile(ladder, prof, snap["spot"], lv["flip"]),
            width="stretch")

    with st.expander("🧮 ที่มาของแต่ละตัวเลข (สูตรที่ใช้)"):
        st.markdown("""
| Metric | สูตร / แหล่ง |
|---|---|
| Dealer cushion (Net GEX) | `Σ Γ × OI × 100 × S² × 0.01` (call +, put −) — CBOE |
| Gamma Flip | reprice BS gamma ที่ spot สมมติ 121 ระดับ → จุดตัดศูนย์ |
| Walls | strike ที่ call_gex / \\|put_gex\\| สูงสุด |
| Max Pain | strike ที่ payoff รวมของผู้ถือ option ต่ำสุด |
| DEX | `Σ delta × OI × 100 × S` — delta จาก CBOE |
| VEX | `Σ vega × OI × 100` — vega จาก CBOE |
| Vanna / Charm | Black-Scholes เอง (CBOE ไม่ให้) · charm หารด้วย 365 = ต่อวัน |
| 25Δ skew (RR25) | `IV(25Δ put) − IV(25Δ call)` หา contract จาก delta จริง |
| Expected move 1σ | `S × IV_atm × √(DTE/365)` · 0DTE นับครึ่งวัน |
| Dealer shock ±1% | `Σ(signed Γ × OI × 100) × ΔS` = หุ้นที่ dealer ต้อง hedge |
| Pin score | 50% ใกล้ Max Pain + 30% GEX บวก + 20% GEX กระจุก |
| IV vs HV | ATM IV (median รอบ spot) vs close-to-close 20 วัน + Parkinson |
| VIX / VVIX | yfinance `^VIX` / `^VVIX` |

⚠️ **หน่วยของ GEX/DEX/VEX เป็น convention ของเราเอง** — เทียบข้ามเจ้าไม่ได้
(ผู้ให้บริการแต่ละรายคูณ scale ต่างกัน) ให้ดู **ตำแหน่ง level และการเปลี่ยนแปลง** ไม่ใช่เลขดิบ
        """)

    with st.expander("📋 ข้อมูลดิบ (per-strike GEX)"):
        ps = snap["per_strike"].copy()
        import fa_gex as _fg
        for c in ("call_gex", "put_gex", "net_gex"):
            ps[c] = ps[c].apply(_fg.fmt_usd)
        st.dataframe(ps, width="stretch", height=300)

    # ── export ──
    txt = render_text(snap)
    st.download_button("⬇️ ดาวน์โหลด Snapshot (text)", data=txt,
                       file_name=f"snapshot_{snap['symbol']}_{snap['asof']:%Y%m%d_%H%M}.txt",
                       mime="text/plain", width="stretch")


def render_text(snap: dict) -> str:
    """เวอร์ชันข้อความล้วน — ใช้ทั้งปุ่มดาวน์โหลดและอีเมล fallback"""
    if snap.get("error"):
        return f"ERROR: {snap['error']}"
    L = [f"{snap['symbol']} Signal Recap — {snap['asof']:%Y-%m-%d %H:%M}",
         f"spot ${snap['spot']:,.2f} | expiry {snap['near_expiry']} ({snap['near_dte']}d) "
         f"| {snap['n_contracts']:,} contracts", "=" * 78]
    for r in snap["rows"]:
        L.append(f"{r['label']:<20} {r['value']:<30} [{r['status']}]")
        L.append(f"{'':<20} {r['note']}")
    L += ["=" * 78,
          "ที่มา: CBOE delayed (~15 นาที) + yfinance | หน่วย GEX/DEX/VEX เป็น convention ของเราเอง",
          "ไม่ใช่คำแนะนำการลงทุน"]
    return "\n".join(L)
