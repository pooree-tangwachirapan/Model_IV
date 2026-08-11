"""
gate.py — ประตูความเสี่ยงก่อนเข้าไม้ fade กำแพงแกมมา (ไม่มี UI)

ทำไมแยกไฟล์: เหมือน snapshot.py — ตรรกะล้วน เพื่อให้ Streamlit tab กับสคริปต์อีเมล
เรียกฟังก์ชันตัวเดียวกัน จะได้ไม่มีทางที่หน้าจอกับอีเมลตัดสินไม่ตรงกัน

ขอบเขต — อ่านให้จบก่อนใช้:
  โมดูลนี้ตัดสินได้เฉพาะประตูที่ "วัดจากข้อมูลได้" เท่านั้น
  ประตูที่เหลือ (วันข่าว / สเปรด bid-ask / เขียนจุด invalidate) มนุษย์ต้องยืนยันเอง
  → ผลลัพธ์สูงสุดที่เป็นไปได้คือ ARMED = "โครงสร้างเข้าเกณฑ์" ไม่ใช่ "เข้าเลย"

  และข้อมูลที่ป้อนเข้ามาคือ CBOE delayed ~15 นาที ส่วน OI อัปเดตข้ามคืนเท่านั้น
  → ถ้ารันก่อนตลาดเปิด ผลที่ได้คือ "แผนที่ของเมื่อคืน" ไม่ใช่สภาพตลาดตอนนี้
"""

from __future__ import annotations

from urllib.parse import urlencode

# ── เกณฑ์ (ตรงกับ gamma-cockpit.html — แก้ที่นี่แล้วต้องแก้ที่นั่นด้วย) ──
EDGE_FRAC = 0.15          # "ชิดกำแพง" = อยู่ในขอบ 15% ของความกว้างโซน
FLIP_FRAC = 0.08          # ต้องห่าง flip ≥ 8% ของความกว้างโซน
FLIP_MIN_PCT = 0.30       # และ ≥ 0.30% ของ spot (buffer ขั้นต่ำ ห้ามรับที่ flip พอดี)
INVALID_EM_FRAC = 0.25    # จุด invalidate = ทะลุกำแพงไปอีก 0.25 × EM
TARGET_EM_FRAC = 0.50     # เป้าขั้นต่ำถ้าไม่มี Max Pain = 0.5 × EM เข้าหากลางโซน

ZERO_DTE_HIGH = 35.0      # % ของ |GEX| ที่ถือว่า 0DTE คุมเกม
PIN_LOW = 35.0            # pin score ต่ำกว่านี้ = โน้มไป trend ไม่ใช่ pin
MIN_PLAN_R = 2.0          # เรขาคณิตต้องให้อย่างน้อย 2R ถึงจะคุ้มค่าเสี่ยง

# โซนต้องกว้างพอเทียบ "ระยะที่ราคาเดินได้เองใน 1 วัน" ไม่งั้นเรขาคณิตไร้ความหมาย
# ที่มา: เข้าที่ขอบ 15% → ระยะถึงกลางโซน = 0.35 × ความกว้าง
#        อยากให้ระยะนั้น ≥ 1σ ต้องได้ความกว้าง ≥ 2.9×EM → ปัดขึ้นเป็น 3.0 เผื่อ margin
# เคสจริงที่จับได้ (2026-08-11): SPY 0DTE กำแพง 770/775 กว้าง 5.00 จุด EM 3.52
#        = 1.42×EM · ราคาเดินข้ามทั้งโซนได้เองโดยไม่มีอะไรเกิดขึ้น
MIN_ZONE_EM = 3.0

DAY_MAX_TRADES = 2
DAY_MAX_LOSS_PCT = 3.0
RISK_PCT_DEFAULT = 1.5

# ประตูที่เครื่องตัดสินแทนไม่ได้ — ต้องขึ้นในทุก output เสมอ
MANUAL_GATES = [
    "ไม่ใช่วัน FOMC / CPI / NFP / งบตัวนั้น",
    "bid-ask ≤ 5% ของ mid และ OI ที่ strike ที่จะเข้าเพียงพอ",
    "เขียนจุด invalidate ไว้แล้วก่อนเข้า (ราคาที่สมมติฐานพัง ไม่ใช่ราคาที่ทนไม่ไหว)",
    "ยังไม่ถึงลิมิตของวัน (%d ไม้ / ขาดทุน %.0f%% ของพอร์ต)" % (DAY_MAX_TRADES, DAY_MAX_LOSS_PCT),
]


def _g(label, ok, value, note):
    return {"label": label, "ok": bool(ok), "value": value, "note": note}


def zone_geometry(spot, put_wall, call_wall, flip=None) -> dict | None:
    """
    เรขาคณิตของโซน — สูตรเดียวกับ gamma-cockpit.html บรรทัด zone()
    pct: 0 = ชิด Put Wall, 1 = ชิด Call Wall
    """
    if None in (spot, put_wall, call_wall) or call_wall <= put_wall:
        return None
    span = call_wall - put_wall
    pct = (spot - put_wall) / span
    out = {
        "span": span, "pct": pct,
        "inside": 0 < pct < 1,
        "edge_near": (0 < pct < 1) and (pct < EDGE_FRAC or pct > 1 - EDGE_FRAC),
        "side": "put" if pct < 0.5 else "call",
        "d_put_pct": (spot - put_wall) / spot * 100,
        "d_call_pct": (call_wall - spot) / spot * 100,
        "flip_frac": None, "flip_pct": None, "flip_near": False, "below_flip": False,
    }
    if flip is not None:
        out["flip_frac"] = abs(spot - flip) / span
        out["flip_pct"] = abs(spot - flip) / spot * 100
        out["flip_near"] = (out["flip_frac"] < FLIP_FRAC) or (out["flip_pct"] < FLIP_MIN_PCT)
        out["below_flip"] = spot < flip
    return out


def zone_problem(spot, put_wall, call_wall) -> tuple[str, bool]:
    """
    บอกให้ชัดว่า "ไม่มีโซน" เพราะอะไร — คืน (เหตุผล, เป็นปัญหาข้อมูลไหม)

    ทำไมต้องแยก: "ดึงข้อมูลมาไม่ครบ" กับ "กำแพงกลับหัวจริง ๆ" ให้ verdict เดียวกันคือห้ามเทรด
    แต่สิ่งที่ต้องทำต่อคนละเรื่อง — อันแรกต้องไปไล่หาว่า pipeline พังตรงไหน
    อันหลังคือสภาพตลาดที่ถูกต้องแล้ว ไม่ต้องแก้อะไร
    """
    if spot is None:
        return "ไม่มีราคา spot — ดึงกระดานไม่สำเร็จ", True
    miss = [n for n, v in (("Put Wall", put_wall), ("Call Wall", call_wall)) if v is None]
    if len(miss) == 2:
        return ("หากำแพงไม่เจอทั้งสองฝั่ง — กระดานว่างหรือดึงข้อมูลมาไม่ครบ "
                "ไม่ใช่สภาพตลาด ให้ไปเช็คว่า CBOE ตอบอะไรมา", True)
    if miss:
        return (f"หา {miss[0]} ไม่เจอ — ไม่มี OI ฝั่งนั้นเลยในหน้าต่างรอบ spot "
                "หรือดึงข้อมูลมาไม่ครบ", True)
    if call_wall <= put_wall:
        return (f"Call Wall {call_wall:,.2f} ไม่ได้อยู่เหนือ Put Wall {put_wall:,.2f} — "
                "กำแพงกลับหัว ไม่มีช่องให้ราคาถูกดูดกลับ · นี่คือสภาพตลาดจริง ไม่ใช่ข้อมูลพัง", False)
    return "ไม่ทราบสาเหตุ", True


def build_plan(spot, z, put_wall, call_wall, em, max_pain=None) -> dict | None:
    """
    แผนไม้ในหน่วยราคาของ underlying — ยังไม่ใช่ราคา option
    invalidate ผูกกับ EM ไม่ใช่ % ตายตัว เพราะวันผันผวนสูงกำแพงถูกแหย่ลึกกว่าปกติเป็นเรื่องปกติ
    """
    if not z or not z["edge_near"] or not em:
        return None
    buf = INVALID_EM_FRAC * em
    if z["side"] == "put":
        direction, wall = "LONG — fade ขึ้นจาก Put Wall", put_wall
        invalid = wall - buf
        target = max_pain if (max_pain and max_pain > spot) else spot + TARGET_EM_FRAC * em
    else:
        direction, wall = "SHORT — fade ลงจาก Call Wall", call_wall
        invalid = wall + buf
        target = max_pain if (max_pain and max_pain < spot) else spot - TARGET_EM_FRAC * em

    # เป้าต้องอยู่ในโซน — Max Pain ที่หลุดออกไปนอกกำแพงฝั่งตรงข้ามทำให้ R:R ปลอมสูง
    # แล้วผ่านประตู MIN_PLAN_R ทั้งที่แผนคือ "ถือข้ามกำแพงที่ควรจะหยุดราคา"
    src = "Max Pain" if (max_pain and target == max_pain) else f"{TARGET_EM_FRAC:g}×EM เข้าหากลางโซน"
    clamped = min(max(target, put_wall), call_wall)
    if clamped != target:
        src += f" (ตัดที่กำแพง {clamped:,.2f} — เป้าเดิม {target:,.2f} อยู่นอกโซน)"
        target = clamped

    risk_pts = abs(spot - invalid)
    reward_pts = abs(target - spot)
    plan_r = (reward_pts / risk_pts) if risk_pts else None
    return {
        "direction": direction, "wall": wall, "entry_ref": spot,
        "invalidation": invalid, "target": target,
        "risk_pts": risk_pts, "reward_pts": reward_pts, "plan_r": plan_r,
        "target_src": src,
    }


def evaluate(snap: dict) -> dict:
    """
    รับ snapshot.build_snapshot() → คืนคำตัดสิน + เหตุผล
    verdict: STAND_DOWN (แดง) / WATCH (เหลือง) / ARMED (เขียว)
    """
    out = {
        "symbol": snap.get("symbol"), "spot": snap.get("spot"), "asof": snap.get("asof"),
        "verdict": "STAND_DOWN", "headline": "", "reason": "",
        "hard": [], "soft": [], "manual": list(MANUAL_GATES),
        "zone": None, "plan": None, "cockpit_query": "", "data_issue": False,
    }
    if snap.get("error"):
        out["headline"] = "ไม่มีข้อมูลพอจะตัดสิน"
        out["reason"] = str(snap["error"])
        out["data_issue"] = True
        return out

    lv = snap.get("levels") or {}
    S = snap.get("spot")
    pw, cw, flip, net = lv.get("put_wall"), lv.get("call_wall"), lv.get("flip"), lv.get("net", 0.0)
    em = (snap.get("expected_move") or {}).get("em")
    z = zone_geometry(S, pw, cw, flip)
    out["zone"] = z
    out["cockpit_query"] = cockpit_query(snap)

    # ── ประตูแข็ง: ตกข้อเดียว = STAND_DOWN ──
    H = out["hard"]
    # โชว์ทีละฝั่ง — เห็นทันทีว่าฝั่งไหนหาย ไม่ใช่แค่ "ไม่ครบ"
    walls = ((f"put {pw:,.2f}" if pw is not None else "put —") + " / " +
             (f"call {cw:,.2f}" if cw is not None else "call —"))
    H.append(_g("ข้อมูลกำแพง", z is not None, walls,
                "ต้องมี Put Wall < Call Wall ถึงจะมีโซนให้ fade"))
    if z is None:
        why, is_data = zone_problem(S, pw, cw)
        out["data_issue"] = is_data
        out["headline"] = "ข้อมูลไม่พอจะตัดสิน" if is_data else "ไม่มีโซนให้ fade"
        out["reason"] = why
        return out

    H.append(_g("ราคาอยู่ในโซน", z["inside"], f"{z['pct']*100:.0f}% ของโซน",
                "อยู่นอกกำแพง = แรง hedge ไม่ดูดกลับแล้ว · กำแพงที่แตกเปลี่ยนหน้าที่จาก support เป็นตัวเร่ง"))

    # โซนแคบกว่าระยะที่ราคาเดินได้เอง = ไม่มีอะไรให้ fade ต่อให้รูปร่างถูกทุกอย่าง
    if em:
        zone_em = z["span"] / em
        z["zone_em"] = zone_em
        H.append(_g("โซนกว้างพอ", zone_em >= MIN_ZONE_EM,
                    f"{z['span']:,.2f} จุด = {zone_em:.2f}×EM (EM 1σ ±{em:,.2f})",
                    f"ต้อง ≥ {MIN_ZONE_EM:g}×EM · แคบกว่านี้ราคาคาดว่าจะกวาดทั้งโซนได้ในวันเดียว "
                    f"กำแพงทั้งสองฝั่งโดนทดสอบ สมมติฐาน 'ถูกดูดกลับ' อ่อนลง "
                    f"และเข้าที่ขอบก็ไม่ต่างจากเข้ากลาง"))
    H.append(_g("Net GEX เป็นบวก", net > 0, _fmt_usd(net),
                "บวก = dealer long gamma ซื้อ dip ขาย rip · ลบ = ไล่ตามราคา ห้าม fade"))

    if flip is None:
        H.append(_g("ห่าง Gamma Flip", False, "หา flip ไม่เจอ",
                    f"ไม่มี zero-crossing ใน ±12% รอบ spot ({lv.get('flip_method') or '—'}) — ไม่รู้ว่าอยู่ฝั่งไหนของ regime"))
    else:
        ok_flip = (not z["flip_near"]) and not (net < 0 and z["below_flip"])
        H.append(_g("ห่าง Gamma Flip", ok_flip,
                    f"{flip:,.2f} (ห่าง {z['flip_pct']:.2f}% / {z['flip_frac']*100:.0f}% ของโซน)",
                    f"ต้องห่าง ≥ {FLIP_MIN_PCT}% ของ spot และ ≥ {FLIP_FRAC*100:.0f}% ของความกว้างโซน — "
                    "ที่ flip พอดีคือ regime กำลังพลิก ไม่ใช่ที่ที่จะรับ"))

    # ── ประตูอ่อน: ผ่านได้แต่ลดความมั่นใจ / ลดไซส์ ──
    Sf = out["soft"]
    pin = (snap.get("pin") or {}).get("score")
    if pin is not None:
        Sf.append(_g("Pin score", pin >= PIN_LOW, f"{pin:.0f}/100",
                     f"ต่ำกว่า {PIN_LOW:.0f} = โน้มไปทาง trend มากกว่า pin — fade ได้แต่ครึ่งไซส์"))
    term = snap.get("term") or {}
    if term.get("slope_pct") is not None:
        Sf.append(_g("Term structure", term["slope_pct"] > 0,
                     f"{term.get('state')} ({term['slope_pct']:+.1f}%)",
                     "backwardation = ตลาดกลัวระยะสั้น ความน่าจะเป็นที่กำแพงถูกทะลุสูงขึ้น"))
    z0 = snap.get("zero_dte")
    if z0 is not None:
        Sf.append(_g("0DTE share", z0 <= ZERO_DTE_HIGH, f"{z0:.1f}% ของ |GEX|",
                     f"เกิน {ZERO_DTE_HIGH:.0f}% = 0DTE คุมเกม level ขยับเร็วระหว่างวัน ต้อง re-check บ่อยขึ้น"))
    # ความกว้างโซนเทียบ EM ย้ายไปเป็น "ประตูแข็ง" แล้ว (ดู MIN_ZONE_EM ด้านบน)
    # เดิมมีทั้ง soft (≥2×EM) และ hard (≥3×EM) วัดเลขตัวเดียวกัน → IWM ที่ 2.53×EM
    # ขึ้น ok กับ FAIL พร้อมกันบนหน้าจอเดียว ตัวชี้วัดเดียวต้องมีคำตัดสินเดียว
    if lv.get("flip_method") and lv["flip_method"] != "spot-ladder":
        Sf.append(_g("คุณภาพค่า Flip", False, str(lv["flip_method"]),
                     "ค่า flip มาจากวิธี fallback ที่หยาบกว่า — ใช้เป็นตัวเลขคร่าว ๆ เท่านั้น"))

    # ── คำตัดสิน ──
    failed = [g for g in H if not g["ok"]]
    if failed:
        out["verdict"] = "STAND_DOWN"
        out["headline"] = "ไม่เข้าเกณฑ์ — อย่า fade วันนี้"
        out["reason"] = " · ".join(g["label"] for g in failed)
        return out

    out["plan"] = build_plan(S, z, pw, cw, em, snap.get("max_pain"))
    weak = [g for g in Sf if not g["ok"]]

    if not z["edge_near"]:
        out["verdict"] = "WATCH"
        out["headline"] = "ผ่านประตู แต่ยังไม่ถึงขอบกำแพง"
        out["reason"] = (f"ราคาอยู่ที่ {z['pct']*100:.0f}% ของโซน — กลางโซนไม่มี edge ให้ fade · "
                         f"เฝ้ารอราคาเข้าใกล้ {pw:,.2f} หรือ {cw:,.2f}")
    else:
        out["verdict"] = "ARMED"
        side = "Put Wall" if z["side"] == "put" else "Call Wall"
        out["headline"] = f"โครงสร้างเข้าเกณฑ์ — ราคาชิด {side}"
        out["reason"] = (f"ผ่านประตูอัตโนมัติครบ {len(H)} ข้อ · "
                         + (f"ประตูอ่อนไม่ผ่าน {len(weak)} ข้อ ({', '.join(g['label'] for g in weak)}) — ลดไซส์ครึ่ง"
                            if weak else "ประตูอ่อนผ่านหมด"))
        pr = (out["plan"] or {}).get("plan_r")
        if pr is not None and pr < MIN_PLAN_R:
            out["verdict"] = "WATCH"
            out["headline"] = "ชิดกำแพงแล้ว แต่เรขาคณิตไม่คุ้ม"
            out["reason"] = (f"ระยะถึงเป้าเทียบระยะถึงจุด invalidate ได้แค่ {pr:.2f}R "
                             f"(ต้องการอย่างน้อย {MIN_PLAN_R:.1f}R) — กำแพงอยู่ใกล้เป้าเกินไป")
    return out


def cockpit_query(snap: dict, account: float | None = None,
                  risk_pct: float = RISK_PCT_DEFAULT) -> str:
    """query string สำหรับส่งค่าต่อให้ gamma-cockpit.html"""
    lv = snap.get("levels") or {}
    q = {"sym": snap.get("symbol") or "", "spot": _r(snap.get("spot")),
         "put": _r(lv.get("put_wall")), "call": _r(lv.get("call_wall")),
         "flip": _r(lv.get("flip")),
         "gex": "pos" if lv.get("net", 0) > 0 else ("neg" if lv.get("net", 0) < 0 else "unk"),
         "risk": risk_pct}
    if account:
        q["acct"] = account
    return urlencode({k: v for k, v in q.items() if v not in (None, "")})


def contracts(account, risk_pct, entry_premium, invalid_premium,
              multiplier=100, cap=5) -> dict:
    """
    สูตรเดียวกับ gamma-cockpit.html — entry/invalid เป็น "ราคา option" ไม่ใช่ราคา underlying
    """
    out = {"per_contract": None, "budget": None, "qty": 0, "total": 0.0, "capped": False}
    if not account or not risk_pct or entry_premium is None or invalid_premium is None:
        return out
    per = abs(entry_premium - invalid_premium) * multiplier
    if per <= 0:
        return out
    budget = account * risk_pct / 100
    qty = int(budget // per)
    if cap and qty > cap:
        qty, out["capped"] = cap, True
    out.update(per_contract=per, budget=budget, qty=qty, total=qty * per)
    return out


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def _fmt_usd(x) -> str:
    if x is None:
        return "—"
    a = abs(x)
    for div, suf in ((1e9, "Bn"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{'-' if x < 0 else ''}${a/div:,.2f}{suf}"
    return f"{'-' if x < 0 else ''}${a:,.0f}"


def render_text(res: dict) -> str:
    """ข้อความล้วน — ใช้ในอีเมล fallback และปุ่มดาวน์โหลด"""
    mark = {"ARMED": "[ARMED]", "WATCH": "[WATCH]", "STAND_DOWN": "[STAND DOWN]"}
    tag = "[DATA ISSUE]" if res.get("data_issue") else mark.get(res["verdict"], "")
    L = [f"{tag} {res.get('symbol')} — {res['headline']}", res["reason"], "-" * 70]
    for g in res["hard"]:
        L.append(f"  {'PASS' if g['ok'] else 'FAIL'}  {g['label']:<22} {g['value']}")
    if res["soft"]:
        L.append("  -- ประตูอ่อน (ไม่ผ่าน = ลดไซส์ ไม่ใช่ห้ามเทรด) --")
        for g in res["soft"]:
            L.append(f"  {'ok  ' if g['ok'] else 'WARN'}  {g['label']:<22} {g['value']}")
    p = res.get("plan")
    if p:
        L += ["-" * 70,
              f"  {p['direction']}",
              f"  อ้างอิงราคาเข้า {p['entry_ref']:,.2f} · invalidate {p['invalidation']:,.2f} "
              f"· เป้า {p['target']:,.2f} ({p['target_src']})",
              f"  เรขาคณิต {p['plan_r']:.2f}R" if p.get("plan_r") else None]
    L += ["-" * 70, "ยังต้องยืนยันด้วยตัวเองก่อนกด:"]
    L += [f"  [ ] {m}" for m in res["manual"]]
    L += ["", "ARMED = โครงสร้างเข้าเกณฑ์ ไม่ใช่คำสั่งให้เข้า · ข้อมูล CBOE delayed ~15 นาที, OI อัปเดตข้ามคืน",
          "ไม่ใช่คำแนะนำการลงทุน"]
    return "\n".join(x for x in L if x is not None)
