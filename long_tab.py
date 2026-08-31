"""
long_tab.py — แท็บที่ 8: Long Premium (Contract Picker + Context Recorder)   [LP]

╔═══════════════════════════════════════════════════════════════════════╗
║ [LP] LONG-PREMIUM TAB — ไฟล์ใหม่ 31 ส.ค. 2026                          ║
║ ถ้าเจอบั๊กที่ไม่เคยเกิดก่อนมีแท็บนี้ → ไฟล์นี้ลบทิ้งได้ทั้งไฟล์             ║
║ ระบบเดิม (fade/breakout/email/workflow) ไม่ import อะไรจากที่นี่เลย       ║
╚═══════════════════════════════════════════════════════════════════════╝

═══ กฎเหล็กของไฟล์นี้: ห้ามกระทบระบบเดิม ═══

1. **อ่าน st.session_state อย่างเดียว ห้ามเขียนทับ key ของคนอื่น**
   ต่างจาก cockpit_tab.py:65-66 ที่เขียน `dash_snap` กลับ — เราจะไม่ทำ
   เพื่อรับประกันว่าแท็บเดิมเห็นข้อมูลเหมือนเดิมทุกกรณี ไม่ว่าจะกดอะไรในแท็บนี้

2. **key ของเราเองขึ้นต้นด้วย `lp_` เสมอ** — grep เจอทันทีว่าอันไหนของแท็บใหม่

3. **ไม่ยิง CBOE ซ้ำ** ถ้ายังไม่มี snapshot ให้บอกผู้ใช้ไปกดแท็บ Dashboard
   (ยิงเองจะทำให้โควตา/เวลาโหลดของระบบเดิมเปลี่ยนไปโดยที่เจ้าตัวไม่ได้สั่ง)

4. **เรียก gate/breakout แบบอ่านอย่างเดียว** — evaluate() เป็น pure function
   รับ snapshot คืน dict ไม่มี side effect

═══ สถานะของแท็บนี้ ═══

**ยังไม่ใช่ระบบให้สัญญาณ** — เป็นเครื่องคำนวณ + ตัวเก็บข้อมูล
เหตุผลอยู่ในหัวไฟล์ intraday.py ข้อ 4 (สถิติ 31 ส.ค. 2026 บอกว่าประตูทิศทางที่เคยคิดไว้ไม่มีหลักฐานรองรับ)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import contracts as ct
import intraday as it
import lp_store as store

# ── import ระบบเดิมเพื่ออ่านแผนเท่านั้น ไม่แก้ไขอะไรในนั้น ──
import breakout
import gate


def _f(x, nd=2, dash="—"):
    """ตัวเลขที่อาจเป็น None — คืนขีดแทน 'None' หรือ nan"""
    try:
        if x is None or x != x:
            return dash
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return dash


def render_long_tab(sym: str, name: str):
    st.subheader("🎯 Long Premium — เลือกสัญญา + เก็บบริบท")
    st.caption(
        f"แปลงแผนที่เป็น **ราคา {sym}** (จาก `gate.py` / `breakout.py`) ให้เป็น **สัญญา option จริง** "
        "แล้วตอบว่า R ที่เหลือหลังหักธีต้าคือเท่าไหร่ · ใช้ข้อมูลที่แอปดึงมาแล้ว ไม่ยิง CBOE ซ้ำ")

    st.warning(
        "**ยังไม่มีหลักฐานว่าระบบนี้มี edge** — แท็บนี้คือเครื่องคำนวณกับตัวเก็บข้อมูล ไม่ใช่ตัวให้สัญญาณ  \n"
        "ตัวกรองทิศทาง (VWAP / EMA200) ถูก **บันทึกไว้เฉย ๆ ไม่ได้ใช้ตัดสิน** เพราะทดสอบกับ QQQ 60 sessions "
        "(31 ส.ค. 2026) แล้วพบว่าไม่ช่วย และทำให้ไม้ช้าลงซึ่งเป็นต้นทุนล้วนของ long premium")

    # ── อ่าน state แบบอ่านอย่างเดียว ────────────────────────────
    snap = st.session_state.get("dash_snap") if st.session_state.get("dash_snap_sym") == sym else None
    chain = st.session_state.get("df_parsed")
    spot_cboe = st.session_state.get("S")

    if snap is None:
        st.info("ยังไม่มี snapshot ของ symbol นี้ — ไปกดคำนวณที่แท็บ **📋 Dashboard** ก่อน "
                "(แท็บนี้ตั้งใจไม่ยิง CBOE เอง เพื่อไม่ให้กระทบเวลาโหลดของระบบเดิม)")
    elif snap.get("error"):
        st.error(f"snapshot มีปัญหา: {snap['error']}")
        snap = None

    # ── 1. แผนที่จะใช้ ─────────────────────────────────────────
    st.markdown("#### 1 · แผนที่จะใช้")
    src = st.radio("ที่มาของแผน", ["fade (gate.py)", "breakout (breakout.py)", "กรอกมือ"],
                   horizontal=True, key="lp_plan_src")

    plan, verdict_txt = None, ""
    fade_v = bo_v = None
    if snap:
        try:
            fade_v = gate.evaluate(snap)
            bo_v = breakout.evaluate(snap)
        except Exception as e:                       # noqa: BLE001
            st.error(f"เรียก evaluate ของระบบเดิมไม่สำเร็จ: {type(e).__name__}: {e}")

    if src.startswith("fade") and fade_v:
        plan, verdict_txt = fade_v.get("plan"), f"fade: **{fade_v.get('verdict')}** — {fade_v.get('headline') or ''}"
    elif src.startswith("breakout") and bo_v:
        plan, verdict_txt = bo_v.get("plan"), f"breakout: **{bo_v.get('verdict')}** — {bo_v.get('headline') or ''}"

    if src == "กรอกมือ":
        base = float(spot_cboe or 700.0)
        c1, c2, c3, c4 = st.columns(4)
        side = c1.selectbox("ทิศ", ["LONG", "SHORT"], key="lp_side")
        entry = c2.number_input("จุดเข้า", value=base, step=0.5, format="%.2f", key="lp_entry")
        invalid = c3.number_input("จุด invalidate", value=base - 5.0, step=0.5, format="%.2f", key="lp_inv")
        target = c4.number_input("เป้า", value=base + 10.0, step=0.5, format="%.2f", key="lp_tgt")
        risk, rew = abs(entry - invalid), abs(target - entry)
        plan = {"side": side, "ideal_entry": entry, "invalidation": invalid, "target": target,
                "plan_r": (rew / risk) if risk else None}
        verdict_txt = "แผนกรอกมือ — ไม่ผ่านประตูของระบบไหนทั้งสิ้น"

    if verdict_txt:
        st.caption(verdict_txt)

    if not plan:
        st.info("ระบบที่เลือกยังไม่มีแผนตอนนี้ (ปกติแปลว่าโครงสร้างวันนี้ไม่เข้าเงื่อนไข) "
                "— เลือก **กรอกมือ** เพื่อลองคำนวณสัญญาด้วยตัวเลขสมมติได้")
        _render_context(sym, spot_cboe, snap, fade_v, bo_v)
        return

    a, b, c, d = st.columns(4)
    a.metric("ทิศ", plan.get("side", "—"))
    b.metric("จุดเข้า", _f(plan.get("ideal_entry")))
    c.metric("invalidate", _f(plan.get("invalidation")))
    d.metric("เป้า", _f(plan.get("target")))
    st.caption(f"plan_r (ของ **ราคา {sym}** ไม่คิดเวลาและ IV) = **{_f(plan.get('plan_r'))}** · "
               f"ที่มาเป้า: {plan.get('target_src', '—')}")

    # ── 2. Contract Picker ─────────────────────────────────────
    st.markdown("#### 2 · เลือกสัญญา")
    if chain is None or (hasattr(chain, "empty") and chain.empty):
        st.info("ยังไม่มี chain ใน session — กด **โหลดข้อมูล** ที่ Sidebar ก่อน")
        _render_context(sym, spot_cboe, snap, fade_v, bo_v)
        return

    p1, p2, p3, p4 = st.columns(4)
    hold = p1.number_input("วันที่ตั้งใจถือ (วันทำการ)", 1, 30, ct.DEFAULT_HOLD_DAYS,
                           key="lp_hold",
                           help="ใช้คิดว่าธีต้ากินไปเท่าไหร่ก่อนถึงเป้า · ค่าเริ่ม 3 = forward_test.MAX_HOLD_DAYS")
    dmin = p2.number_input("DTE ต่ำสุด", 1, 120, ct.MIN_DTE, key="lp_dmin",
                           help=f"ค่าเริ่ม {ct.MIN_DTE} = {ct.DTE_MULT}× วันที่ถือ — ซื้อสั้นกว่านี้ธีต้ากินค่าเวลาหมดก่อนออก")
    dmax = p3.number_input("DTE สูงสุด", 2, 365, 60, key="lp_dmax")
    acct = p4.number_input("พอร์ต (USD)", 0, 10_000_000, 50_000, step=1000, key="lp_acct")
    risk_pct = st.slider("ความเสี่ยงต่อไม้ (% ของพอร์ต)", 0.25, 5.0, gate.RISK_PCT_DEFAULT, 0.25,
                         key="lp_riskpct")

    try:
        picks = ct.pick(chain, float(spot_cboe), plan, hold_days=int(hold),
                        account=float(acct) or None, risk_pct=float(risk_pct),
                        min_dte=int(dmin), max_dte=int(dmax))
    except Exception as e:                           # noqa: BLE001
        st.error(f"เลือกสัญญาไม่สำเร็จ: {type(e).__name__}: {e}")
        picks = pd.DataFrame()

    summ = ct.summarize(picks, plan)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ผ่านเกณฑ์", f"{summ['n_ok']} / {summ['n_total']}")
    m2.metric(f"plan_r (ราคา {sym})", _f(summ["plan_r"]))
    m3.metric("R จริงของ option", _f(summ["best_r_option"]),
              delta=_f(summ["r_gap"]) if summ["r_gap"] is not None else None,
              help="R หลังหักธีต้าและตีราคาด้วย BS — ตัวเลขที่ตัดสินจริง")
    m4.metric("ผลตอบแทนที่เป้า", f"{_f(summ['best_return_pct'], 0)}%",
              help="กำไรคิดเป็น % ของพรีเมียมที่จ่าย — ดูคู่กับ R เสมอ เพราะ R เพียว ๆ จะเชียร์ DTE ยาว")

    if summ["r_gap"] is not None and summ["r_gap"] < 0:
        st.caption(f"⚠️ ธีต้ากิน R ไป **{_f(abs(summ['r_gap']))}** จาก {_f(summ['plan_r'])} "
                   f"เหลือ {_f(summ['best_r_option'])} — นี่คือส่วนที่แผนราคาเปล่า ๆ มองไม่เห็น")

    if picks.empty:
        st.warning("ไม่มีสัญญาที่ตีราคาได้เลย (ปกติแปลว่ากระดานฝั่งนั้นไม่มี bid/ask หรือ IV)")
    else:
        show_all = st.checkbox("แสดงสัญญาที่ตกด้วย (พร้อมเหตุผล)", value=False, key="lp_showall")
        df = picks if show_all else picks[picks["status"] == "OK"]
        if df.empty:
            st.warning("ไม่มีสัญญาไหนผ่านเกณฑ์ — ติ๊กช่องข้างบนเพื่อดูว่าตกเพราะอะไร")
        else:
            view = pd.DataFrame({
                "strike": df["strike"], "DTE": df["dte"], "Δ": df["delta"].round(2),
                "mid": df["mid"].round(2), "spread%": df["spread_pct"].round(1),
                "IV%": df["iv_pct"].round(1), "θ/วัน%": df["theta_day_pct"].round(2),
                "P&L เป้า": df["pnl_target"].round(0),
                "IV−3": df["pnl_target_ivdn"].round(0),
                "IV+3": df["pnl_target_ivup"].round(0),
                "P&L invalid": df["pnl_invalid"].round(0),
                "R option": df["r_option"].round(2),
                "ผลตอบแทน%": df["return_pct"].round(0),
                "จำนวน": df["qty"], "OI": df["open_interest"].astype("int64"),
                "สถานะ": df["status"], "เหตุผลที่ตก": df["reasons"],
            })
            st.dataframe(view.head(40), width="stretch", hide_index=True)
            st.caption(
                "**IV−3 / IV+3** = กำไรที่เป้าถ้า IV ลด/เพิ่ม 3 จุด — ช่องว่างระหว่างสองค่านี้คือความเสี่ยง vega  \n"
                "ราคาเข้าใช้ **mid** ของจริงต้องข้าม spread · CBOE ช้า ~15 นาที · "
                "สมมติว่าขาชนะกับขาแพ้ใช้เวลาเท่ากัน (ของจริง stop มักมาเร็วกว่า)")

    st.divider()
    _render_context(sym, spot_cboe, snap, fade_v, bo_v)


# ════════════════════════════════════════════════════════════════
def _render_context(sym, spot_cboe, snap, fade_v, bo_v):
    """ส่วนที่ 3–4: บริบทของวันนี้ + ปุ่มบันทึกลง log"""
    st.markdown("#### 3 · บริบทวันนี้ (โครงสร้าง option + tape)")

    cc1, cc2 = st.columns([1, 3])
    if cc1.button("📈 คำนวณ tape", key="lp_ctx_go", width="stretch",
                  help="ดึงแท่ง 5 นาที 60 วันจาก yfinance แล้วคำนวณ VWAP / Volume Profile / EMA200 / volume spike"):
        with st.spinner("กำลังดึงแท่ง 5 นาที ..."):
            st.session_state["lp_ctx"] = it.build_context(sym, spot=spot_cboe)
            st.session_state["lp_ctx_sym"] = sym

    ctx = st.session_state.get("lp_ctx") if st.session_state.get("lp_ctx_sym") == sym else None
    if not ctx:
        cc2.caption("ยังไม่ได้คำนวณ — กดปุ่มซ้ายมือ (ใช้ yfinance ไม่เกี่ยวกับโควตา CBOE)")
        _render_recorder(snap, None, fade_v, bo_v)
        return
    if ctx.get("error"):
        st.error(f"คำนวณ tape ไม่สำเร็จ: {ctx['error']}")
        _render_recorder(snap, ctx, fade_v, bo_v)
        return

    cc2.caption(f"แท่ง 5m {ctx.get('bars_5m_n', 0):,} แท่ง · {ctx.get('sessions_n', 0)} sessions · "
                f"session ล่าสุด {ctx.get('session_date')} ({ctx.get('today_bars', 0)} แท่ง)")
    if (ctx.get("today_bars") or 0) < 5:
        st.caption("⚠️ session วันนี้เพิ่งเริ่ม — VWAP กับ volume spike ยังไม่มีความหมาย")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("VWAP", _f(ctx.get("vwap")),
              delta=("เหนือ" if ctx.get("above_vwap") else "ใต้") if ctx.get("above_vwap") is not None else None)
    g2.metric("EMA200 (วัน)", _f(ctx.get("ema200_daily")),
              delta=("เหนือ" if ctx.get("above_ema200") else "ใต้") if ctx.get("above_ema200") is not None else None)
    g3.metric("Volume spike", _f(ctx.get("vspike_last")) + "×",
              help="volume ของแท่งล่าสุด ÷ median ของช่วงเวลาเดียวกันย้อนหลัง 20 วัน")
    g4.metric("อยู่ตรงไหนของ VA", ctx.get("va_zone") or "—",
              delta=(f"พ้นออกไป {_f(ctx.get('va_penetration_pct'))}%"
                     if (ctx.get("va_penetration_pct") or 0) > 0 else None))

    lv = (snap or {}).get("levels") or {}
    rows = [
        ("Value Area เมื่อวาน (VAH / POC / VAL)",
         f"{_f(ctx.get('prev_vah'))} / {_f(ctx.get('prev_poc'))} / {_f(ctx.get('prev_val'))}", "tape"),
        ("Value Area วันนี้ (ยังไม่จบ)",
         f"{_f(ctx.get('today_vah'))} / {_f(ctx.get('today_poc'))} / {_f(ctx.get('today_val'))}", "tape"),
        ("กำแพง GEX (call / put)", f"{_f(lv.get('call_wall'))} / {_f(lv.get('put_wall'))}", "option"),
        ("กำแพง OI ล้วน (call / put)", f"{_f(lv.get('call_wall_oi'))} / {_f(lv.get('put_wall_oi'))}", "option"),
        ("Gamma flip", _f(lv.get("flip")), "option"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["ระดับ", "ค่า", "มาจาก"]),
                 width="stretch", hide_index=True)
    st.caption("เส้นสองชุดนี้มาคนละที่ — **tape** คือที่ที่มีคนเทรดจริง · **option** คือที่ที่มีสัญญาคงค้าง "
               "ตรงกัน = level แข็ง · ไม่ตรง = ยังไม่ยืนยัน")

    _render_recorder(snap, ctx, fade_v, bo_v)


def _render_recorder(snap, ctx, fade_v, bo_v):
    """ส่วนที่ 4: บันทึกลง JSONL"""
    st.markdown("#### 4 · เก็บบริบทไว้ใช้ทีหลัง")
    s = store.stats()
    st.caption(
        f"เก็บมาแล้ว **{s['n_records']} record / {s['n_sessions']} session** "
        + (f"({s['first_session']} → {s['last_session']})" if s["n_sessions"] else "")
        + "  \nชุดข้อมูลที่จับ **โครงสร้าง option กับ tape พร้อมกัน** ยังไม่มีใครมี — "
          "CBOE ไม่ให้ย้อนหลัง ถ้าไม่เริ่มเก็บวันนี้ อีก 3 เดือนก็ยังตอบคำถามประตูไม่ได้ "
          "(เป้าหมาย 30–50 session ถึงจะเริ่มพูดอะไรได้)")

    b1, b2 = st.columns([1, 3])
    if b1.button("💾 บันทึกตอนนี้", key="lp_save", width="stretch",
                 disabled=(snap is None and ctx is None)):
        try:
            rec = store.flatten(snap, ctx, fade=fade_v, breakout_v=bo_v)
            path = store.append(rec)
            st.success(f"บันทึกแล้ว → `{path}` (รวม {len(rec)} field)")
        except Exception as e:                       # noqa: BLE001
            st.error(f"บันทึกไม่สำเร็จ: {type(e).__name__}: {e}")

    b2.caption("บันทึกเมื่อกดเท่านั้น — **ไม่มี workflow ไม่มี cron ไม่มี auto-commit** "
               "ไฟล์อยู่ `long_premium/` แยกจาก `forward_test/` เพื่อไม่ให้ชนกับ merge ของระบบเดิม")

    if s["n_records"]:
        with st.expander(f"ดู record ล่าสุด ({s['n_records']} ทั้งหมด)", expanded=False):
            recs = store.read_all()[-10:]
            st.dataframe(pd.DataFrame(recs), width="stretch", hide_index=True)
            st.download_button("⬇️ ดาวน์โหลด context_log.jsonl",
                               data="\n".join(pd.Series(recs).astype(str)),
                               file_name="context_log_preview.txt",
                               mime="text/plain", width="stretch")
