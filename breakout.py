"""
breakout.py — ประตูสำหรับไม้ "ไล่ตามการทะลุกำแพง" (ไม่มี UI)

คู่ตรงข้ามของ gate.py โดยตั้งใจ:

    gate.py (fade)      ต้องการ +GEX · ราคาอยู่ในโซน · ห่าง flip · โซนกว้าง
                        → เดิมพันว่ากำแพง "ถือ" ราคาถูกดูดกลับ
    breakout.py         ต้องการ −GEX หรือกำแพงแตกแล้ว · ราคาอยู่นอกโซน · ใกล้/ต่ำกว่า flip
                        → เดิมพันว่ากำแพง "แตก" แล้ววิ่งต่อ

สองระบบนี้ **ห้าม ARMED พร้อมกัน** — ถ้าเกิดขึ้นแปลว่าตรรกะขัดกันเอง มี assert กันไว้

ฐานคิด (SKILL v2 · Wall-Breach Protocol + Trend-Day Scorecard):
    • dealer short gamma = ไล่ตามราคา → การเคลื่อนไหวถูกขยาย ไม่ถูกกด
    • กำแพงที่แตกเปลี่ยนหน้าที่จาก "แนวต้าน" เป็น "ตัวเร่ง" จนกว่าราคาจะ reclaim + ยืนได้
    • ระบบนี้ **แพ้บ่อย ชนะน้อยครั้งแต่ใหญ่** — ตรงข้ามกับ fade ที่ชนะบ่อยแพ้ใหญ่
      ต้องวัดที่ expectancy ไม่ใช่ win rate ห้ามเอาไปเทียบ win rate กับ gate.py

ข้อจำกัดที่ต้องรู้ (เหมือน gate.py):
    ตัดสินได้เฉพาะประตูที่วัดจากข้อมูลได้ · ข้อมูลเป็น CBOE delayed ~15 นาที
    OI อัปเดตข้ามคืน → ก่อนตลาดเปิดคือ "แผนที่เมื่อคืน"
"""

from __future__ import annotations

# ── เกณฑ์ ──
# ทุกระยะวัดเป็น "เท่าของ Expected Move" ไม่ใช่ % ของราคา
# เพราะวันผันผวนสูงกำแพงถูกทะลุลึกกว่าปกติเป็นเรื่องปกติ — ใช้ % ตายตัวจะกรองผิดทั้งสองทาง
BREACH_MIN_EM = 0.10      # ต้องพ้นกำแพงอย่างน้อย 0.10×EM ถึงนับว่า "แตก" ไม่ใช่แค่แหย่
STOP_EM_FRAC = 0.25       # stop = ราคากลับเข้ามาในกำแพงลึก 0.25×EM (reclaim = สมมติฐานตาย)
TARGET_EM_MULT = 2.50     # เป้า = 2.5×EM จากกำแพงที่แตก (ขนาดของ trend day)
MIN_PLAN_R = 2.50         # ระบบแพ้บ่อย ต้องการ R สูงกว่า fade (gate.py ใช้ 2.0)
MIN_EM_ROOM = 1.00        # ต้องเหลือระยะถึงเป้าอย่างน้อย 1×EM ไม่งั้นไม่มีอะไรให้กิน
MAX_0DTE_SHARE = 60.0     # 0DTE เกินนี้ = level ขยับเร็วเกินจะไล่


def max_viable_breach_em() -> float:
    """
    ไล่ตามได้ไกลสุดกี่ ×EM ก่อนที่เรขาคณิตจะไม่ผ่าน MIN_PLAN_R

    ที่ระยะ d (หน่วย EM) จากกำแพง:
        risk   = (d + STOP_EM_FRAC) × EM
        reward = (TARGET_EM_MULT − d) × EM
        R      = (TARGET − d) / (d + STOP)
    ต้องการ R ≥ MIN_PLAN_R  →  d ≤ (TARGET − MIN_R×STOP) / (1 + MIN_R)

    ทำไมต้องคำนวณ ไม่ตั้งเป็นค่าคงที่: เวอร์ชันแรกตั้ง "ทะลุอย่างน้อย 0.15%" กับ
    "ต้องได้ 2.5R" แยกกัน แล้วสองเงื่อนไขนั้นขัดกันเอง — เพดานจากเรขาคณิตอยู่ที่ 0.147%
    ซึ่งต่ำกว่าพื้น 0.15% ระบบจึงยิงไม่ได้เลยแม้แต่เคสที่ดีที่สุด (ตรวจเจอตอนเทสต์ 2026-08-13)
    ผูกไว้ด้วยกันแบบนี้แล้วมันขัดกันเองไม่ได้อีก
    """
    return (TARGET_EM_MULT - MIN_PLAN_R * STOP_EM_FRAC) / (1.0 + MIN_PLAN_R)


BREACH_MAX_EM = max_viable_breach_em()

if BREACH_MAX_EM <= BREACH_MIN_EM:      # กันไม่ให้ตั้งค่าจนระบบยิงไม่ได้อีก
    raise ValueError(
        f"เกณฑ์ขัดกันเอง: เพดานจากเรขาคณิต {BREACH_MAX_EM:.3f}×EM "
        f"ต่ำกว่าพื้น BREACH_MIN_EM {BREACH_MIN_EM:.3f}×EM — ไม่มีระยะไหนที่ผ่านทั้งคู่")

MANUAL_GATES = [
    "ไม่ใช่วัน FOMC / CPI / NFP — วันข่าวการทะลุมักเป็นกับดัก",
    "มี volume ยืนยันตอนทะลุ (ไม่ใช่ทะลุเงียบ ๆ ตอนสภาพคล่องบาง)",
    "ไม่ได้ไล่ในชั่วโมงแรก (09:30–10:30 ET) — whipsaw สูงสุด",
    "ยอมรับล่วงหน้าว่าไม้แบบนี้แพ้ 60–70% และนั่นคือปกติ",
]


def _g(label, ok, value, note):
    return {"label": label, "ok": bool(ok), "value": value, "note": note}


def _fmt_usd(x):
    if x is None:
        return "—"
    a = abs(x)
    if a >= 1e9:
        return f"{x/1e9:+.2f}Bn"
    if a >= 1e6:
        return f"{x/1e6:+.1f}M"
    return f"{x:+,.0f}"


def breach_state(spot, put_wall, call_wall, em=None) -> dict | None:
    """
    ราคาอยู่ตรงไหนเทียบกำแพง — คืน None ถ้ายังอยู่ในโซน (ไม่มีอะไรให้ breakout)
    ต้องมี em ถึงจะตัดสิน fresh/too_far ได้ เพราะหน้าต่างวัดเป็น ×EM
    """
    if None in (spot, put_wall, call_wall) or call_wall <= put_wall:
        return None
    if spot > call_wall:
        wall, side, direction = call_wall, "call", "UP"
    elif spot < put_wall:
        wall, side, direction = put_wall, "put", "DOWN"
    else:
        return None
    dist_pts = abs(spot - wall)
    out = {
        "wall": wall, "side": side, "direction": direction,
        "dist_pts": dist_pts, "dist_pct": dist_pts / spot * 100,
        "dist_em": None, "fresh": False, "too_far": False, "too_close": False,
    }
    if em:
        d = dist_pts / em
        out.update(dist_em=d,
                   fresh=BREACH_MIN_EM <= d <= BREACH_MAX_EM,
                   too_far=d > BREACH_MAX_EM,
                   too_close=d < BREACH_MIN_EM)
    return out


def build_plan(spot, br, em, max_pain=None, next_level=None) -> dict | None:
    """
    แผนไล่ตามการทะลุ — stop คือ "ราคากลับเข้ามาในกำแพง" ไม่ใช่ % ตายตัว

    ต่างจาก fade ตรงที่ entry คือราคาตลาดตอนนี้ (ไล่) ไม่ใช่รอราคามาหา
    เพราะรอแล้วมันไม่กลับมา — นั่นคือธรรมชาติของ breakout และเป็นจุดอ่อนที่ยอมรับ
    """
    if not br or not br["fresh"] or not em:
        return None
    up = br["direction"] == "UP"
    wall = br["wall"]

    # stop: กลับเข้ามาในกำแพงลึก 0.5×EM = การทะลุนั้นเป็นของปลอม
    stop = wall - STOP_EM_FRAC * em if up else wall + STOP_EM_FRAC * em

    # เป้า: level ถัดไปถ้ามี ไม่งั้น 2×EM จากจุดที่ทะลุ
    if next_level and ((up and next_level > spot) or (not up and next_level < spot)):
        target, src = next_level, "level ถัดไป"
    else:
        target = wall + TARGET_EM_MULT * em if up else wall - TARGET_EM_MULT * em
        src = f"{TARGET_EM_MULT:g}×EM จากกำแพงที่แตก"

    risk_pts = abs(spot - stop)
    reward_pts = abs(target - spot)
    if risk_pts <= 0:
        return None
    return {
        "direction": ("LONG — ไล่ขึ้นหลังทะลุ Call Wall" if up
                      else "SHORT — ไล่ลงหลังหลุด Put Wall"),
        "side": "LONG" if up else "SHORT",
        "wall": wall, "spot": spot,
        "ideal_entry": spot,          # ไล่ที่ราคาตลาด ไม่มีจุดเข้าในอุดมคติ
        "invalidation": stop, "target": target,
        "risk_pts": risk_pts, "reward_pts": reward_pts,
        "plan_r": reward_pts / risk_pts,
        "target_src": src,
        "room_em": reward_pts / em,
        # ── ฟิลด์ให้ schema ตรงกับ gate.build_plan ──
        # เพื่อให้ตัวเรนเดอร์ (gate.render_text / send_report.build_gate_html) ใช้ตัวเดียวกันได้
        # ไม่ต้องแยกทางเขียนสองชุด ซึ่งจะเพี้ยนออกจากกันเมื่อแก้ข้างเดียว
        # ระบบนี้เข้าที่ราคาตลาดทันที → ระยะถึงจุดเข้าเป็น 0 และอยู่ที่ "ขอบ" เสมอ
        # blown เป็น False เสมอโดยโครงสร้าง: stop อยู่ในกำแพง ส่วน spot อยู่นอกกำแพงแล้ว
        "dist_to_entry_pts": 0.0,
        "dist_to_entry_pct": 0.0,
        "at_edge": True,
        "blown": False,
    }


def evaluate(snap: dict) -> dict:
    """
    รับ snapshot.build_snapshot() → คำตัดสินฝั่ง breakout
    verdict: STAND_DOWN / WATCH / ARMED (ความหมายเดียวกับ gate.py)
    """
    out = {
        "system": "breakout",
        "symbol": snap.get("symbol"), "spot": snap.get("spot"), "asof": snap.get("asof"),
        "verdict": "STAND_DOWN", "headline": "", "reason": "",
        "hard": [], "soft": [], "manual": list(MANUAL_GATES),
        "breach": None, "plan": None, "data_issue": False,
    }
    if snap.get("error"):
        out["headline"] = "ไม่มีข้อมูลพอจะตัดสิน"
        out["reason"] = str(snap["error"])
        out["data_issue"] = True
        return out

    lv = snap.get("levels") or {}
    S = snap.get("spot")
    pw, cw, flip = lv.get("put_wall"), lv.get("call_wall"), lv.get("flip")
    net = lv.get("net", 0.0)
    em = (snap.get("expected_move") or {}).get("em")
    br = breach_state(S, pw, cw, em)
    out["breach"] = br

    H = out["hard"]
    walls = ((f"put {pw:,.2f}" if pw else "put —") + " / " +
             (f"call {cw:,.2f}" if cw else "call —"))
    H.append(_g("ข้อมูลกำแพง", br is not None or (pw and cw), walls,
                "ต้องมีกำแพงทั้งสองฝั่งถึงจะรู้ว่าทะลุอันไหน"))
    # แยก "ดึงข้อมูลมาไม่ครบ" ออกจาก "กำแพงทับกันจริง" ให้ตรงกับ gate.zone_problem
    # ทั้งคู่ห้ามเทรดเหมือนกัน แต่สิ่งที่ต้องทำต่อคนละเรื่อง — อันแรกต้องไปไล่ว่า pipeline พังตรงไหน
    # อันหลังคือสภาพตลาดที่ถูกต้องแล้ว ถ้าตีเป็น data issue หัวเรื่องเมลจะขึ้น ⚠️ DATA ทั้งที่ข้อมูลปกติ
    if pw is None or cw is None:
        out["data_issue"] = True
        out["headline"] = "ข้อมูลไม่พอจะตัดสิน"
        miss = [n for n, v in (("Put Wall", pw), ("Call Wall", cw)) if v is None]
        out["reason"] = f"หา {' และ '.join(miss)} ไม่เจอ — กระดานว่างหรือดึงข้อมูลมาไม่ครบ"
        return out
    if cw <= pw:
        out["headline"] = "ไม่มีโซนให้อ้างอิง"
        out["reason"] = (f"Call Wall {cw:,.2f} ไม่ได้อยู่เหนือ Put Wall {pw:,.2f} — "
                         "ไม่มีขอบเขตให้บอกว่า 'ทะลุ' คืออะไร · นี่คือสภาพตลาดจริง ไม่ใช่ข้อมูลพัง")
        return out

    # ── ประตูแข็ง ──
    H.append(_g("ตำแหน่งเทียบกำแพง", br is not None,
                f"{br['direction']} ทะลุ {br['wall']:,.2f} ไป {br['dist_pct']:.2f}%"
                if br else f"ยังอยู่ในโซน {pw:,.2f}–{cw:,.2f}",
                "ระบบนี้เข้าเฉพาะตอนกำแพงแตกแล้ว — อยู่ในโซนคือเกมของ fade ไม่ใช่เกมนี้"))
    if br is None:
        out["headline"] = "ยังไม่มีการทะลุ"
        out["reason"] = f"ราคาอยู่ในโซน {pw:,.2f}–{cw:,.2f} — ระบบ fade คือตัวที่ควรดู ไม่ใช่ breakout"
        return out

    if not em:
        H.append(_g("ความสดของการทะลุ", False, "ไม่มี Expected Move",
                    "หน้าต่างไล่ตามวัดเป็น ×EM — ไม่มี EM ก็ตัดสินไม่ได้"))
    else:
        H.append(_g("ความสดของการทะลุ", br["fresh"],
                    f"{br['dist_em']:.2f}×EM ({br['dist_pct']:.2f}%) จากกำแพง",
                    f"ต้องอยู่ระหว่าง {BREACH_MIN_EM:.2f}–{BREACH_MAX_EM:.2f}×EM · "
                    + ("ต่ำกว่านี้ = ยังแค่แหย่ อาจเด้งกลับ"
                       if br["too_close"] else
                       f"เกินนี้เรขาคณิตให้ไม่ถึง {MIN_PLAN_R:g}R แล้ว — ตกรถ ไม่ต้องไล่")))

    # regime: −GEX หนุน breakout · +GEX แรงกดจะสวนเรา
    H.append(_g("Net GEX", net <= 0, _fmt_usd(net),
                "ระบบ breakout ต้องการ **ลบ** · ลบ = dealer short gamma ไล่ตามราคา "
                "ขยายการเคลื่อนไหว หนุนการทะลุ · บวก = dealer สวนราคา ดูดกลับเข้าโซน "
                "ทำให้การทะลุเป็นของปลอม"))

    if flip is None:
        H.append(_g("ฝั่งของ Gamma Flip", False, "หา flip ไม่เจอ",
                    "ไม่รู้ว่าอยู่ regime ไหน — ไม่ควรไล่ตามอะไรทั้งนั้น"))
    else:
        up = br["direction"] == "UP"
        # ทะลุขึ้นควรอยู่เหนือ flip / หลุดลงควรอยู่ต่ำกว่า flip — ไม่งั้นสวน regime
        ok_flip = (S > flip) if up else (S < flip)
        H.append(_g("ฝั่งของ Gamma Flip", ok_flip,
                    f"flip {flip:,.2f} · spot {'เหนือ' if S > flip else 'ต่ำกว่า'}",
                    "ทะลุขึ้นต้องอยู่เหนือ flip · หลุดลงต้องอยู่ต่ำกว่า flip "
                    "ไม่งั้นกำลังไล่สวน regime ที่ dealer ยังกดอยู่"))

    plan = build_plan(S, br, em, snap.get("max_pain"))
    out["plan"] = plan
    if em:
        room = (plan or {}).get("room_em")
        H.append(_g("ระยะเหลือถึงเป้า", bool(room and room >= MIN_EM_ROOM),
                    f"{room:.2f}×EM ถึงเป้า" if room else "คำนวณไม่ได้",
                    f"ต้อง ≥ {MIN_EM_ROOM:g}×EM · น้อยกว่านี้เป้าอยู่ในระยะที่ราคาแกว่งถึงเองอยู่แล้ว"))
    H.append(_g(f"เรขาคณิต ≥ {MIN_PLAN_R:g}R", bool(plan and plan["plan_r"] >= MIN_PLAN_R),
                f"{plan['plan_r']:.2f}R" if plan else "—",
                f"ระบบนี้แพ้บ่อย ต้องการ R สูงกว่า fade (ที่ใช้ {2.0:g}R) ถึงจะคุ้ม"))

    # ── ประตูอ่อน ──
    Sf = out["soft"]
    z0 = None
    for r in snap.get("rows", []):
        if r["label"] == "0DTE share":
            try:
                z0 = float(r["value"].split("%")[0])
            except (ValueError, IndexError):
                pass
    if z0 is not None:
        Sf.append(_g("0DTE share", z0 <= MAX_0DTE_SHARE, f"{z0:.1f}% ของ |GEX|",
                     f"เกิน {MAX_0DTE_SHARE:g}% = กำแพงขยับทุกชั่วโมง ไล่ตามไม่ทัน"))
    if lv.get("call_wall_oi") and lv.get("put_wall_oi"):
        agree = (lv["call_wall_oi"] == cw and lv["put_wall_oi"] == pw)
        Sf.append(_g("กำแพงสองนิยาม", agree,
                     f"OI ล้วน {lv['put_wall_oi']:,.0f}/{lv['call_wall_oi']:,.0f}",
                     "ไม่ตรงกัน = ไม่แน่ใจว่ากำแพงที่ 'แตก' คือกำแพงจริง"))
    term = snap.get("term") or {}
    if term.get("slope_pct") is not None:
        back = term["slope_pct"] < 0
        Sf.append(_g("Term structure", back,
                     f"{term['state']} ({term['slope_pct']:+.1f}%)",
                     "backwardation = ตลาดกลัวระยะสั้น มักมาคู่กับการเคลื่อนไหวแรง หนุน breakout"))

    # ── ตัดสิน ──
    failed = [r["label"] for r in H if not r["ok"]]
    if failed:
        out["verdict"] = "STAND_DOWN"
        out["headline"] = "ไม่เข้าเกณฑ์ไล่ตาม"
        out["reason"] = " · ".join(failed)
    else:
        out["verdict"] = "ARMED"
        out["headline"] = f"เข้าเกณฑ์ — {br['direction']} หลังทะลุ {br['wall']:,.2f}"
        weak = [r["label"] for r in Sf if not r["ok"]]
        out["reason"] = ("ผ่านทุกประตู" if not weak
                         else "ผ่านประตูแข็ง แต่ควรลดไซส์ครึ่ง: " + " · ".join(weak))
    return out


def conflicts_with(fade_verdict: dict, breakout_verdict: dict) -> str | None:
    """
    กันตรรกะขัดกันเอง — สองระบบนี้ต้องไม่ ARMED พร้อมกันเด็ดขาด
    fade ต้องการราคาอยู่ในโซน · breakout ต้องการราคาอยู่นอกโซน → เป็นไปไม่ได้ทั้งคู่
    ถ้าเกิดขึ้นแปลว่ามีบั๊ก ไม่ใช่โอกาสทำเงินสองเด้ง
    """
    if fade_verdict.get("verdict") == "ARMED" and breakout_verdict.get("verdict") == "ARMED":
        return ("ทั้ง fade และ breakout ขึ้น ARMED พร้อมกัน — เป็นไปไม่ได้ตามนิยาม "
                "(อันหนึ่งต้องการราคาในโซน อีกอันต้องการนอกโซน) มีบั๊กในการคำนวณ level")
    return None
