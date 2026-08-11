"""
cockpit_tab.py — Tab "Cockpit" : จาก "ตลาดเป็นยังไง" → "วันนี้เข้าได้ไหม ใส่เท่าไหร่"

ตรรกะทั้งหมดอยู่ใน gate.py (ไม่มี UI) เพื่อให้สคริปต์อีเมลใช้ตัวเดียวกัน
ไฟล์นี้มีแต่การแสดงผล + ช่องกรอกที่เครื่องตัดสินแทนไม่ได้
"""

import streamlit as st

import gate
import snapshot

LAMP = {"ARMED": "#2ecc71", "WATCH": "#e67e22", "STAND_DOWN": "#e74c3c"}

CSS = """
<style>
.ck-lamp { display:flex; align-items:center; gap:16px; padding:16px 18px;
  border:1px solid #1e3358; border-radius:12px; background:#0b1730; margin-bottom:6px; }
.ck-orb { width:46px; height:46px; border-radius:50%; flex:0 0 auto; }
.ck-head { font-size:21px; font-weight:700; color:#e8f2ff; line-height:1.2; }
.ck-sub { font-size:12.5px; color:#8aa6cc; margin-top:4px; }
.ck-g { display:grid; grid-template-columns:64px 190px 200px 1fr; gap:10px;
  align-items:center; padding:8px 12px; border-bottom:1px solid #16243f; }
.ck-tag { font-size:10.5px; font-weight:700; letter-spacing:.4px; padding:3px 0;
  border-radius:4px; text-align:center; border:1px solid currentColor; }
.ck-l { color:#8aadee; font-size:12.5px; font-weight:600; }
.ck-v { color:#e8f2ff; font-family:monospace; font-size:13px; }
.ck-n { color:#7e93b5; font-size:11.5px; line-height:1.45; }
@media (max-width:760px){ .ck-g{grid-template-columns:64px 1fr;} .ck-n{grid-column:1/-1;} }
</style>
"""


def _gate_rows(items, ok_word="PASS", no_word="FAIL"):
    html = []
    for g in items:
        col = "#2ecc71" if g["ok"] else "#e74c3c"
        html.append(
            f'<div class="ck-g"><div class="ck-tag" style="color:{col}">'
            f'{ok_word if g["ok"] else no_word}</div>'
            f'<div class="ck-l">{g["label"]}</div>'
            f'<div class="ck-v">{g["value"]}</div>'
            f'<div class="ck-n">{g["note"]}</div></div>')
    return "".join(html)


def render_cockpit_tab(sym: str, name: str):
    st.subheader("🎯 Cockpit — ประตูก่อนเข้าไม้")
    st.caption("แปลง 17 สัญญาณให้เหลือคำตอบเดียว: **วันนี้ fade กำแพงได้ไหม และใส่กี่สัญญา** · "
               "ตรรกะเดียวกับที่ส่งเข้าอีเมล (`gate.py`)")

    # ── snapshot: ใช้ของ Dashboard ถ้ามีอยู่แล้ว จะได้ไม่ยิง CBOE ซ้ำ ──
    c1, c2 = st.columns([3, 1])
    reuse = st.session_state.get("dash_snap") if st.session_state.get("dash_snap_sym") == sym else None
    c1.caption(f"ใช้ snapshot จากแท็บ Dashboard ({reuse['asof']:%H:%M})" if reuse
               else "ยังไม่มี snapshot ของ symbol นี้ — กดคำนวณ")
    if c2.button("🔄 คำนวณใหม่", key="ck_go", width="stretch") or reuse is None:
        with st.spinner(f"กำลังคำนวณ {sym} ..."):
            try:
                reuse = snapshot.build_snapshot(sym)
            except Exception as e:
                st.error(f"คำนวณไม่สำเร็จ: {type(e).__name__}: {e}")
                return
        st.session_state["dash_snap"] = reuse
        st.session_state["dash_snap_sym"] = sym

    snap = reuse
    if not snap or snap.get("error"):
        st.error(snap.get("error") if snap else "ไม่มีข้อมูล")
        return

    g = gate.evaluate(snap)
    color = "#f1c40f" if g.get("data_issue") else LAMP[g["verdict"]]

    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="ck-lamp"><div class="ck-orb" style="background:{color};'
        f'box-shadow:0 0 0 6px {color}22, 0 0 26px {color}66"></div>'
        f'<div><div class="ck-head">{g["headline"]}</div>'
        f'<div class="ck-sub">{g["reason"]}</div></div></div>', unsafe_allow_html=True)
    st.caption(f"**{snap['symbol']}** spot `${snap['spot']:,.2f}` · "
               f"expiry {snap['near_expiry']} ({snap['near_dte']}d) · {snap['asof']:%Y-%m-%d %H:%M} · "
               "CBOE delayed ~15 นาที · OI อัปเดตข้ามคืน")

    if g.get("data_issue"):
        st.warning("⚠️ **นี่เป็นปัญหาข้อมูล ไม่ใช่สภาพตลาด** — verdict ที่เห็นตัดสินจากกระดานที่ไม่ครบ "
                   "อย่าเพิ่งเชื่อว่าตลาดไม่เหมาะเทรด ให้ไปเช็คว่า CBOE ตอบอะไรมาก่อน")

    # ── ประตู ──
    st.markdown("**ประตูแข็ง — ตกข้อเดียวจบ**")
    st.markdown(_gate_rows(g["hard"]), unsafe_allow_html=True)
    if g["soft"]:
        st.markdown("")
        st.markdown("**ประตูอ่อน — ไม่ผ่าน = ลดไซส์ครึ่ง ไม่ใช่ห้ามเทรด**")
        st.markdown(_gate_rows(g["soft"], "OK", "WARN"), unsafe_allow_html=True)

    z = g["zone"]
    if z:
        a, b, c, d = st.columns(4)
        a.metric("ถึง Put Wall", f"{z['d_put_pct']:.2f}%")
        b.metric("ตำแหน่งในโซน", f"{z['pct']*100:.0f}%",
                 help="0% = ชิด Put Wall · 100% = ชิด Call Wall · ขอบ 15% สองฝั่งคือที่ที่มี edge")
        c.metric("ถึง Call Wall", f"{z['d_call_pct']:.2f}%")
        if z.get("zone_em"):
            d.metric("ความกว้างโซน", f"{z['zone_em']:.2f}×EM",
                     help=f"ต้อง ≥ {gate.MIN_ZONE_EM:g}×EM · แคบกว่านี้ราคาเดินข้ามทั้งโซนได้เองใน 1 วัน")

        # กำแพงอีกนิยาม — GEX ถ่วงด้วย gamma (สูงสุดที่ ATM) จึงถูกดึงเข้าหาราคา
        # ส่วน OI ล้วนคือภูเขาสัญญาคงค้างจริง สองอันตรงกัน = level แข็ง
        lv = snap.get("levels") or {}
        cw_oi, pw_oi = lv.get("call_wall_oi"), lv.get("put_wall_oi")
        if cw_oi and pw_oi:
            same = (cw_oi == lv.get("call_wall") and pw_oi == lv.get("put_wall"))
            msg = (f"กำแพงที่ใช้ตัดสินคือนิยาม **GEX** (γ×OI) = "
                   f"`{lv['put_wall']:,.0f} / {lv['call_wall']:,.0f}` · "
                   f"ถ้าใช้ **OI ล้วน** จะได้ `{pw_oi:,.0f} / {cw_oi:,.0f}`")
            if same:
                st.success(msg + " — **ตรงกันทั้งสองนิยาม = level แข็ง**")
            else:
                st.info(msg + "\n\nต่างกันเพราะ GEX ถ่วงด้วย gamma ซึ่งสูงสุดที่ ATM "
                        "จึงดึงกำแพงเข้าหาราคาปัจจุบัน ส่วน OI ล้วนคือภูเขาสัญญาคงค้างจริง "
                        "ที่นิ่งกว่า — **สองนิยามไม่ตรงกัน = level ยังไม่ได้รับการยืนยัน** "
                        "ให้ลดความเชื่อมั่นลง")

    # ── แผนไม้ ──
    p = g.get("plan")
    if p:
        st.divider()
        st.markdown(f"**แผนไม้ — {p['direction']}**")
        a, b, c, d = st.columns(4)
        a.metric("จุดเข้าที่ควรเป็น", f"{p['ideal_entry']:,.2f}",
                 delta=f"ราคาตอนนี้ {p['spot']:,.2f}", delta_color="off",
                 help="อ้างอิงที่กำแพง ไม่ใช่ราคาปัจจุบัน — ตัวเลขจะได้ไม่เปลี่ยนทุกนาที")
        b.metric("Invalidate", f"{p['invalidation']:,.2f}",
                 help=f"ทะลุกำแพงไปอีก {gate.INVALID_EM_FRAC:g}×EM — ผูกกับความผันผวนจริง ไม่ใช่ % ตายตัว")
        c.metric("เป้า", f"{p['target']:,.2f}", help=p["target_src"])
        d.metric("เรขาคณิต", f"{p['plan_r']:.2f}R" if p.get("plan_r") else "—",
                 help=f"ต่ำกว่า {gate.MIN_PLAN_R:.1f}R ถือว่าไม่คุ้มค่าเสี่ยง")

        if p["blown"]:
            st.error(f"ราคาเลยจุด invalidate ไปแล้ว ({p['spot']:,.2f}) — แผนนี้ตายแล้ว")
        elif p["at_edge"]:
            st.success(f"ราคาอยู่ที่ขอบพอดี ({p['spot']:,.2f})")
        elif p["dist_to_entry_pts"] < 0:
            st.warning(f"ราคาทะลุจุดเข้าไปแล้ว {abs(p['dist_to_entry_pts']):,.2f} "
                       f"({abs(p['dist_to_entry_pct']):.2f}%) — ตกรถ ไม่ต้องไล่")
        else:
            st.info(f"ราคายังห่างจุดเข้าอีก {p['dist_to_entry_pts']:,.2f} "
                    f"({p['dist_to_entry_pct']:.2f}%) — รอ")

        st.caption("ตัวเลขทั้งหมดเป็นราคา **underlying** ไม่ใช่ราคา option · "
                   "แสดงทุก verdict แม้ราคาจะเลยจุดเข้าไปแล้ว เพื่อให้ย้อนวัดผลได้")

    # ── ประตูที่เครื่องตัดสินแทนไม่ได้ ──
    st.divider()
    st.markdown("**ต้องยืนยันด้วยตัวเองก่อนกด**")
    checked = [st.checkbox(m, key=f"ck_man_{i}") for i, m in enumerate(g["manual"])]
    all_manual = all(checked)

    if g["verdict"] == "ARMED" and all_manual:
        st.success("ผ่านครบทุกประตู — ไปคำนวณไซส์ แล้วเข้าที่ขอบกำแพง")
    elif g["verdict"] == "ARMED":
        st.warning(f"โครงสร้างเข้าเกณฑ์แล้ว แต่ยังเหลือ {checked.count(False)} ข้อที่ต้องยืนยันเอง")

    # ── ไซส์ ──
    st.divider()
    st.markdown("**ขนาดไม้**")
    s1, s2, s3 = st.columns(3)
    acct = s1.number_input("ขนาดพอร์ต (USD)", min_value=0.0, value=10000.0, step=500.0, key="ck_acct")
    risk = s2.number_input("ความเสี่ยงต่อไม้ (%)", min_value=0.1, max_value=10.0,
                           value=gate.RISK_PCT_DEFAULT, step=0.1, key="ck_risk")
    cap = s3.number_input("เพดานสัญญา", min_value=1, value=5, step=1, key="ck_cap")
    s4, s5, s6 = st.columns(3)
    entry = s4.number_input("พรีเมียมที่เข้า", min_value=0.0, value=2.50, step=0.05, key="ck_entry")
    inval = s5.number_input("พรีเมียมที่ invalidate", min_value=0.0, value=1.25, step=0.05, key="ck_inval")
    mult = s6.number_input("ตัวคูณสัญญา", min_value=1, value=100, step=1, key="ck_mult")

    if risk > 4:
        st.error("เกิน 4% ต่อไม้ — แพ้ติดกัน 8 ไม้ (เกิดแทบแน่นอนที่ WR 30–50%) จะกินพอร์ตเกินครึ่ง")

    sz = gate.contracts(acct, risk, entry, inval, mult, cap)
    a, b, c = st.columns(3)
    a.metric("เสี่ยงต่อสัญญา", f"${sz['per_contract']:,.0f}" if sz["per_contract"] else "—")
    b.metric("จำนวนสัญญา", sz["qty"], help="ถูกจำกัดด้วยเพดาน" if sz["capped"] else None)
    c.metric("เสี่ยงจริง", f"${sz['total']:,.0f} ({sz['total']/acct*100:.2f}%)" if acct and sz["qty"] else "$0")
    if sz["qty"] == 0 and sz["per_contract"]:
        st.warning("งบความเสี่ยงไม่พอแม้แต่สัญญาเดียว — ไม้นี้ใหญ่เกินพอร์ต ข้ามไป")

    # ── ส่งต่อให้ cockpit ──
    st.divider()
    st.markdown("**เปิดใน Gamma Cockpit (พร้อมค่าที่กรอกให้แล้ว)**")
    # ไม่ต้องส่ง value= — Streamlit จำค่าให้เองผ่าน key อยู่แล้ว
    # (ส่งทั้ง key และ value ที่อ่านมาจาก session_state ซ้ำ เสี่ยงโดนเตือนเรื่อง default vs session state)
    base = st.text_input("URL ของ gamma-cockpit.html", key="ck_base",
                         placeholder="https://pooree-tangwachirapan.github.io/Model_IV/gamma-cockpit.html")
    q = gate.cockpit_query(snap, account=acct or None, risk_pct=risk)
    if base.strip():
        st.link_button("🎯 เปิด Cockpit", f"{base.strip()}?{q}", width="stretch")
    else:
        st.caption("ยังไม่ได้ตั้ง URL — คอมมิต `gamma-cockpit.html` เข้ารีโปนี้แล้วเปิด GitHub Pages "
                   "(Settings → Pages → Deploy from branch) จะได้ URL ถาวรมาใส่ตรงนี้")
    st.code(f"?{q}", language="text")

    st.download_button("⬇️ ดาวน์โหลดคำตัดสิน (text)", data=gate.render_text(g),
                       file_name=f"gate_{snap['symbol']}_{snap['asof']:%Y%m%d_%H%M}.txt",
                       mime="text/plain", width="stretch")

    st.caption("⚠️ **ARMED ไม่ใช่คำสั่งให้เข้า** — แปลว่าโครงสร้างของกระดานไม่ได้ห้ามไว้เท่านั้น "
               "ตัวจับจังหวะจริงยังเป็นการที่ราคาไปถึงกำแพงแล้วปฏิเสธ ซึ่งต้องดูบนกราฟเอง · ไม่ใช่คำแนะนำการลงทุน")
