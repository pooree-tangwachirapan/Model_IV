"""
zero_dte_tab.py — แท็บที่ 9: 0DTE Recorder (ดูข้อมูลที่เก็บไว้ + ปุ่มออกรายงาน HTML)   [ZDTE]

╔═══════════════════════════════════════════════════════════════════════╗
║ [ZDTE] แท็บใหม่ 25 ก.ย. 2026 — ถ้าเจอบั๊กที่ไม่เคยเกิดก่อนมีแท็บนี้      ║
║ ถอนได้ที่ iv_surface_real.py 3 จุด แล้วลบ zero_dte*.py ทิ้ง             ║
║ ระบบเดิม (fade/breakout/email) ไม่ import อะไรจากที่นี่เลย              ║
╚═══════════════════════════════════════════════════════════════════════╝

กฎเดียวกับแท็บ Long Premium:
  1. อ่าน st.session_state อย่างเดียว · key ของเราขึ้นต้น `zdte_` เสมอ
  2. เช็ค `.get("error")` ก่อนแตะ field อื่นของ snapshot — HANDOFF §6.6
     (บั๊กในแท็บเดียวทำ Streamlit หยุดทั้ง script แล้วทุกแท็บหลังจากนั้นไม่ render)
  3. ปุ่มที่ยิงเน็ต (บันทึก snapshot) ต้องกดเองเท่านั้น ไม่ยิงตอน render

หน้าที่ของแท็บ: **ดูว่าเก็บอะไรไว้แล้ว และออกรายงานไปวิเคราะห์ต่อ**
ไม่มีสัญญาณซื้อขายในแท็บนี้ (§9 — ข้อมูลยังไม่พอจะตั้ง threshold อะไรทั้งนั้น)
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

import market_clock as mc
import zero_dte as z


@st.cache_data(ttl=60, show_spinner=False)
def _summary_cached(sym: str, day: str, mtime: float) -> dict:
    """
    สรุปหนึ่งวัน · cache ผูกกับ mtime ของไฟล์ ไม่ใช่แค่ (sym, day)

    ถ้า cache ด้วย (sym, day) เฉย ๆ พอกดบันทึก snapshot ใหม่แล้วหน้าจอจะยังโชว์ของเก่า
    จนกว่า ttl จะหมด — ผู้ใช้กดปุ่มแล้วไม่เห็นอะไรเปลี่ยน = เข้าใจว่าปุ่มพัง
    """
    return z.day_summary(sym, day)


def _mtime(sym: str, day: str) -> float:
    for p in (z.path_for(sym, day), z.month_path(sym, day[:7])):
        if os.path.exists(p):
            return os.path.getmtime(p)
    return 0.0


def render_zero_dte_tab(sym: str, name: str):
    st.subheader(f"0DTE Recorder — {sym}")
    st.caption(
        "เก็บ option 0DTE ทุก strike ลง CSV ตลอดวัน เพื่อให้ย้อนกลับมาวิเคราะห์ edge ได้ภายหลัง · "
        "CBOE ให้แต่ภาพปัจจุบัน ไม่มีย้อนหลัง — ถ้าไม่เก็บเองก็ถามย้อนหลังไม่ได้เลย"
    )

    prog = mc.session_progress()
    phase = {"premarket": "ก่อนเปิด", "regular": "ตลาดเปิดอยู่",
             "after": "หลังปิด", "closed": "วันนี้ตลาดไม่เปิด"}[prog["phase"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("สถานะตลาด", phase)
    c2.metric("วันนี้ (ET)", prog["date"])
    c3.metric("เวลาเก็บอัตโนมัติ", "ทุก 15 นาทีตอนตลาดเปิด")

    # ── ปุ่มบันทึกเดี๋ยวนี้ (ยิงเน็ต — กดเองเท่านั้น) ─────────────────
    b1, b2 = st.columns([1, 3])
    if b1.button("📥 บันทึก snapshot เดี๋ยวนี้", key="zdte_record",
                 use_container_width=True):
        with st.spinner("กำลังดึง chain จาก CBOE…"):
            try:
                r = z.record(sym)
            except Exception as e:                        # noqa: BLE001
                r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if r.get("ok"):
            msg = (f"บันทึกแล้ว · expiry {r['expiry']} · {r['strikes']} strike · "
                   f"เขียน {r['written']} แถว")
            if r.get("skipped"):
                msg += f" · ข้าม {r['skipped']} แถวที่บันทึกไปแล้ว"
            st.success(msg)
        else:
            st.error(f"บันทึกไม่สำเร็จ: {r.get('error')}")

    b2.caption("ปุ่มนี้ยิง CBOE หนึ่งครั้ง · ปกติ workflow `armed-alert` เก็บให้อัตโนมัติ"
               " ทุก 15 นาทีอยู่แล้ว ใช้ตอนอยากได้จุดเพิ่มนอกตาราง")

    # ── เลือกวัน ─────────────────────────────────────────────────
    days = z.available_days(sym)
    if not days:
        st.info(
            "ยังไม่มีข้อมูลสักวัน — กดปุ่มด้านบนเพื่อเก็บจุดแรก "
            "หรือรอ workflow เก็บให้ในวันทำการถัดไป"
        )
        return

    st.divider()
    left, right = st.columns([1, 2])
    day = left.selectbox("วันที่", days[::-1], key="zdte_day")
    arch = z.archived_months(sym)
    right.caption(
        f"มีข้อมูล {len(days)} วัน ({days[0]} → {days[-1]})"
        + (f" · เดือนที่รวบเป็นคลังแล้ว: {', '.join(arch)}" if arch else "")
    )

    try:
        s = _summary_cached(sym, day, _mtime(sym, day))
    except Exception as e:                                # noqa: BLE001
        st.error(f"อ่านข้อมูลวันนี้ไม่สำเร็จ: {type(e).__name__}: {e}")
        return

    if s.get("empty"):
        st.warning(f"ไฟล์ของวัน {day} ว่าง — ยังไม่มี snapshot ที่ใช้ได้")
        return

    m = st.columns(4)
    m[0].metric("snapshot", f"{s['snapshots']}", f"{s['first_et']}–{s['last_et']} ET")
    m[1].metric("strike ที่เก็บ", f"{s['strikes']}", f"{s['rows']:,} แถว")
    m[2].metric("spot", f"{s['spot_last']:,.2f}",
                f"{s['spot_last'] - s['spot_open']:+.2f}")
    m[3].metric("P/C ratio",
                "—" if s["pc_ratio"] is None else f"{s['pc_ratio']:.2f}",
                f"พรีเมียม ${s['notional_usd']:,.0f}")

    if s["snapshots"] < 4:
        st.warning(
            f"วันนี้มีแค่ {s['snapshots']} snapshot — กราฟตามเวลาจะอ่านอะไรไม่ได้มาก "
            "วันที่เก็บครบจะมีราว 26 จุด (ทุก 15 นาที)"
        )

    # ── ปุ่มออกรายงาน HTML ───────────────────────────────────────
    st.divider()
    st.markdown("#### รายงาน HTML")
    r1, r2 = st.columns([1, 2])
    standalone = r2.checkbox(
        "ฝัง plotly.js ในไฟล์ (เปิดออฟไลน์ได้ · ไฟล์ใหญ่ ~4 MB)",
        key="zdte_standalone",
        help="ไม่ติ๊ก = โหลด plotly จาก CDN ไฟล์เล็กกว่ามากแต่ต้องต่อเน็ตตอนเปิด",
    )

    if r1.button("📊 สร้างรายงาน", key="zdte_build", type="primary",
                 use_container_width=True):
        with st.spinner("กำลังประกอบกราฟ…"):
            try:
                import zero_dte_report as rep
                html = rep.build(sym, day, standalone=standalone)
                st.session_state["zdte_html"] = html
                st.session_state["zdte_html_key"] = f"{sym}-{day}"
            except Exception as e:                        # noqa: BLE001
                st.session_state.pop("zdte_html", None)
                st.error(f"สร้างรายงานไม่สำเร็จ: {type(e).__name__}: {e}")

    html = st.session_state.get("zdte_html")
    if html and st.session_state.get("zdte_html_key") == f"{sym}-{day}":
        st.download_button(
            f"⬇️ ดาวน์โหลด 0dte-{sym}-{day}.html",
            data=html.encode("utf-8"),
            file_name=f"0dte-{sym}-{day}.html",
            mime="text/html",
            key="zdte_dl_html",
            use_container_width=True,
        )
        with st.expander("ดูรายงานในหน้านี้เลย", expanded=False):
            # height คงที่ — ปล่อยให้ยืดตามเนื้อหาไม่ได้ Streamlit ต้องรู้ความสูงล่วงหน้า
            st.components.v1.html(html, height=900, scrolling=True)

    # ── ดาวน์โหลด CSV ดิบ ────────────────────────────────────────
    st.markdown("#### ข้อมูลดิบ")
    p = z.path_for(sym, day)
    if os.path.exists(p):
        with open(p, "rb") as fh:
            st.download_button(f"⬇️ {os.path.basename(p)}", data=fh.read(),
                               file_name=os.path.basename(p), mime="text/csv",
                               key="zdte_dl_csv")
    else:
        arch_p = z.month_path(sym, day[:7])
        st.caption(f"วันนี้ถูกรวบเข้าคลังรายเดือนแล้ว — ไฟล์อยู่ที่ `{arch_p}`")

    top = pd.DataFrame(s["top_notional"])
    if not top.empty:
        top["notional"] = top["notional"].map(lambda v: f"${v:,.0f}")
        top.columns = ["strike", "C/P", "พรีเมียมที่หมุน"]
        st.markdown("**strike ที่พรีเมียมเปลี่ยนมือมากที่สุด**")
        st.dataframe(top, hide_index=True, use_container_width=True)

    st.caption(
        "ข้อจำกัดที่ต้องอ่านคู่กันเสมอ: CBOE ดีเลย์ ~15 นาที · "
        "ความละเอียดเท่าความถี่ที่บันทึก · เก็บเฉพาะ strike ที่มีคนเทรด"
        f" หรืออยู่ในกรอบ ±{z.BAND_PCT*100:.0f}% รอบราคา"
    )
