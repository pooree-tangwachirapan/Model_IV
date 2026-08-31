"""
contracts.py — เลือกสัญญา option ให้กับแผนที่คิดมาแล้ว (ไม่มี UI)   [LP]

╔═══════════════════════════════════════════════════════════════════════╗
║ [LP] LONG-PREMIUM TAB — ไฟล์ใหม่ 31 ส.ค. 2026                          ║
║ ถ้าเจอบั๊กที่ไม่เคยเกิดก่อนมีแท็บนี้ → ไฟล์นี้ลบทิ้งได้ทั้งไฟล์             ║
║ ระบบเดิม (fade/breakout/email/workflow) ไม่ import อะไรจากที่นี่เลย       ║
╚═══════════════════════════════════════════════════════════════════════╝

═══ ปัญหาที่ไฟล์นี้แก้ ═══

`gate.py` (fade) กับ `breakout.py` ออกแผนเป็น **ราคา underlying** เท่านั้น —
entry / invalidation / target / plan_r ทั้งหมดเป็นจุดราคาของ QQQ
แล้ว `gate.contracts()` ให้มนุษย์กรอก entry_premium กับ invalid_premium เอง

ช่องว่างคือ: **plan_r ของราคา ≠ R ที่ได้จริงจาก option**
plan_r สมมติว่าเวลาฟรีและ IV คงที่ ซึ่งผิดทั้งสองข้อสำหรับ long premium
ไฟล์นี้ตอบว่า "ซื้อสัญญาไหน แล้ว R จริงหลังหักธีต้าเหลือเท่าไหร่"

═══ ทำไมไม่คำนวณ Greeks เอง ═══

CBOE ส่ง delta / gamma / vega / theta / bid / ask / open_interest มาให้ครบอยู่แล้ว
และ `parse_options()` ใน iv_surface_real.py ใช้ `df_raw.copy()` → คอลัมน์พวกนี้อยู่ครบใน
`st.session_state["df_parsed"]` ที่แอปดึงมาแล้ว
→ **โมดูลนี้ยิงเน็ต 0 ครั้ง** ใช้ของที่มีอยู่แล้วล้วน ๆ

หลักการเดียวกับ `fa_gex.parse_cboe_for_gex()` ที่คำนวณ BS gamma เฉพาะเป็น fallback

BS ถูกใช้เฉพาะตอน **ตีราคาฉากอนาคต** (ราคาไปถึงเป้า / ไปถึง invalid) ซึ่ง CBOE ให้ไม่ได้

═══ ข้อสมมติที่ต้องรู้ก่อนเชื่อตัวเลข ═══

1. **Sticky strike** — ตอน reprice เราคง IV ของสัญญานั้นไว้เท่าเดิม
   ของจริงอยู่ระหว่าง sticky strike กับ sticky delta → ฉาก "IV คงที่" จึงเป็นค่ากลาง
   ไม่ใช่คำทำนาย ให้ดูฉาก −3/+3 ควบคู่เสมอ

2. **เวลาที่ใช้ถือ** — สมมติว่าทั้งขาชนะและขาแพ้ใช้เวลาเท่ากัน (`hold_days`)
   ของจริง stop มักโดนเร็วกว่าเป้า → ตัวเลขฝั่งขาดทุนจึงมองโลกในแง่ร้ายไปนิด

3. **ราคาเข้าใช้ mid** — ของจริงต้องข้าม spread ครึ่งหนึ่งเป็นอย่างน้อย
   คอลัมน์ `spread_pct` บอกว่าค่าผ่านทางแพงแค่ไหน

4. **CBOE delayed ~15 นาที** — bid/ask ที่เห็นไม่ใช่ราคาที่จะได้จริง
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── ค่าคงที่ ──
# ค่าเดียวกับ fa_gex.RISK_FREE / snapshot.RF — ทั้งโปรเจกต์ใช้ 5% เหมือนกัน
RISK_FREE = 0.05

# forward_test.MAX_HOLD_DAYS = 3 วันทำการ
# กฎ DTE ≥ 3 × วันที่ถือ: ถ้าซื้อ 3DTE แล้วถือ 3 วัน = ธีต้ากินค่าเวลาหมดพอดี
# ตัวคูณ 3 ทำให้ยังเหลือค่าเวลาตอนออก ไม่ใช่ขายซากที่หมดอายุ
DEFAULT_HOLD_DAYS = 3
DTE_MULT = 3
MIN_DTE = DEFAULT_HOLD_DAYS * DTE_MULT        # = 9 · ไม่ใช่ 0–2DTE

# แบนด์ delta: ต่ำกว่านี้ option ไม่ขยับตามแผน · สูงกว่านี้จ่ายค่า intrinsic ที่ไม่ได้ leverage
DELTA_MIN, DELTA_MAX = 0.35, 0.55

MAX_SPREAD_PCT = 5.0       # ตรงกับ manual gate ข้อ 2 ของ gate.py ("bid-ask ≤ 5% ของ mid")
MIN_OI = 100               # manual gate ข้อเดียวกัน — OI ที่ strike ที่จะเข้าเพียงพอ
MAX_THETA_DAY_PCT = 3.0    # เสียค่าเวลาเกิน 3%/วัน = ไม้มีอายุสั้นกว่าที่แผนต้องการ
IV_SHOCK = 3.0             # จุด IV ที่ใช้จำลอง crush / expansion

MULTIPLIER = 100           # 1 สัญญา = 100 หุ้น

# ─────────────────────────────────────────────────────────────────
# หมายเหตุเรื่องเกณฑ์: ตัวเลขข้างบนเป็น "คุณภาพการเข้าออก" (สภาพคล่อง เรขาคณิต)
# ไม่ใช่ threshold ของสัญญาณ จึงตั้งได้โดยไม่ต้องรอ forward test
# แต่ยังไม่มีตัวไหนถูกพิสูจน์กับข้อมูลจริงของพอร์ตนี้ → pick() จึง **ไม่ตัดใครทิ้ง**
# ทุกสัญญาถูกคืนกลับมาพร้อมเหตุผลที่ตก เพื่อให้เห็นว่าทำไมถึงไม่มีตัวไหนผ่าน
# (HANDOFF §9 — อย่าตั้ง/ขยับ threshold ก่อนมีข้อมูล)
# ─────────────────────────────────────────────────────────────────


def bs_price(S: float, K: float, T: float, iv: float,
             r: float = RISK_FREE, is_call: bool = True) -> float | None:
    """
    Black-Scholes ราคาเต็มสูตร — ใช้ตีราคาฉากอนาคต ไม่ใช่ Greeks expansion

    ทำไมไม่ใช้ delta+gamma ประมาณ: เป้าของ breakout คือ 2.5×EM ซึ่งไกลเกินกว่าที่
    การกระจายเทย์เลอร์อันดับสองจะแม่น (ยิ่งใกล้หมดอายุยิ่งเพี้ยน)
    ที่นี่ input ครบอยู่แล้ว (K, iv, T) จึงตีราคาตรง ๆ ได้เลย ไม่ต้องประมาณ

    T ≤ 0 หรือ iv ≤ 0 → คืนมูลค่าที่แท้จริง (intrinsic) เพราะค่าเวลาหมดแล้ว
    """
    if S is None or K is None or not np.isfinite(S) or not np.isfinite(K) or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if T is None or iv is None or T <= 0 or iv <= 0:
        return float(intrinsic)

    vol = iv * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / vol
    d2 = d1 - vol
    if is_call:
        px = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        px = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(px, intrinsic))


def _mid(bid, ask) -> float | None:
    """ราคากลาง — ต้องมีทั้งสองฝั่งถึงจะเชื่อได้ ฝั่งเดียวคือกระดานที่ไม่มีคนทำราคา"""
    if bid is None or ask is None or not np.isfinite(bid) or not np.isfinite(ask):
        return None
    if ask <= 0 or bid < 0 or ask < bid:
        return None
    m = (bid + ask) / 2
    return float(m) if m > 0 else None


def pick(chain: pd.DataFrame, spot: float, plan: dict,
         hold_days: int = DEFAULT_HOLD_DAYS,
         account: float | None = None, risk_pct: float | None = None,
         min_dte: int = MIN_DTE, max_dte: int = 60,
         delta_band: tuple[float, float] = (DELTA_MIN, DELTA_MAX),
         iv_shock: float = IV_SHOCK) -> pd.DataFrame:
    """
    หาสัญญาที่เข้ากับแผน แล้วตีราคาทุกฉาก → DataFrame เรียงตาม r_option (ดีสุดอยู่บน)

    `chain` = st.session_state["df_parsed"] (มี delta/theta/bid/ask/iv/days/strike/type ครบ)
    `plan`  = ผลจาก gate.build_plan() หรือ breakout.build_plan() — schema เดียวกันทั้งคู่
              ต้องมี side / ideal_entry / invalidation / target

    **ไม่ตัดสัญญาไหนทิ้ง** — คืนมาทุกตัวพร้อมคอลัมน์ `status` และ `reasons`
    เพื่อให้เห็นว่าถ้าไม่มีตัวไหนผ่าน มันตกเพราะอะไร ไม่ใช่ตารางว่าง ๆ ที่บอกอะไรไม่ได้

    ⚠️ **"DTE ยาวให้ R ดีกว่า" ไม่จริงสากล — อย่า hardcode สมมติฐานนี้เข้าไป**
    วัดจริง 31 ส.ค. 2026 บนกระดาน QQQ:
      · เป้า +2.1% (ระยะปกติของ fade)  → 60DTE R=1.67 ชนะ 18DTE R=1.25
      · เป้า +10%  (ระยะแบบ breakout)  → 14DTE R=3.62 ชนะ 60DTE R=1.67
    เพราะ move ใหญ่ทำให้สัญญาสั้นกลายเป็น intrinsic เกือบล้วน convexity จึงชนะธีต้า
    ส่วน move เล็กไม่พอจะชดเชยธีต้าของสัญญาสั้น
    → **ให้ตารางจัดอันดับเอง อย่าไปเดาแทนมัน** และดู `return_pct` คู่กับ `r_option` เสมอ
    """
    cols = ["strike", "days", "type", "bid", "ask", "iv", "delta", "theta",
            "gamma", "vega", "open_interest", "volume", "expiry"]
    if chain is None or chain.empty or not plan or not spot:
        return pd.DataFrame(columns=cols)

    side = plan.get("side")
    if side not in ("LONG", "SHORT"):
        return pd.DataFrame(columns=cols)

    # LONG = ซื้อ call (อยากให้ขึ้น) · SHORT = ซื้อ put (อยากให้ลง) — ฝั่ง long premium ล้วน
    want = "call" if side == "LONG" else "put"
    is_call = want == "call"

    entry_px = plan.get("ideal_entry") or spot
    target_px = plan.get("target")
    invalid_px = plan.get("invalidation")
    if target_px is None or invalid_px is None:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[df["type"] == want]
    df = df[df["days"].notna() & (df["days"] >= 1)]
    df = df[(df["days"] >= min_dte - 30) & (df["days"] <= max_dte + 30)]  # เผื่อไว้ให้เห็นตัวที่ตก
    if df.empty:
        return pd.DataFrame(columns=cols)

    dlo, dhi = delta_band
    rows = []
    for _, r in df.iterrows():
        K = float(r["strike"])
        dte = int(r["days"])
        iv = float(r["iv"]) if r["iv"] == r["iv"] and r["iv"] > 0 else None
        mid = _mid(r["bid"], r["ask"])
        if mid is None or iv is None:
            continue                       # ไม่มีราคา/IV = ตีราคาไม่ได้ ข้ามไปเลย ไม่ใช่ "ตก"

        spread_pct = (float(r["ask"]) - float(r["bid"])) / mid * 100
        delta = abs(float(r["delta"])) if r["delta"] == r["delta"] else np.nan
        theta_day = abs(float(r["theta"])) if r["theta"] == r["theta"] else np.nan
        theta_day_pct = theta_day / mid * 100 if (theta_day == theta_day and mid) else np.nan
        oi = float(r["open_interest"]) if r["open_interest"] == r["open_interest"] else 0.0

        # ── ตีราคาฉากอนาคต: เวลาเดินไป hold_days วัน ราคาไปถึงเป้า / ถึง invalid ──
        T_now = dte / 365.0
        T_then = max((dte - hold_days) / 365.0, 0.0)

        def px_at(S_, iv_shift=0.0):
            return bs_price(S_, K, T_then, max(iv + iv_shift / 100.0, 1e-6),
                            RISK_FREE, is_call)

        p_tgt = px_at(target_px)
        p_tgt_dn = px_at(target_px, -iv_shock)      # IV crush — ถูกทางแต่ยังขาดทุนได้
        p_tgt_up = px_at(target_px, +iv_shock)      # vol ขยาย — ของแถมของ long premium
        p_inv = px_at(invalid_px)

        # P&L ต่อ 1 สัญญา (USD) — เข้าที่ mid วันนี้ ออกที่ราคาที่ตีได้
        pnl_tgt = (p_tgt - mid) * MULTIPLIER if p_tgt is not None else None
        pnl_tgt_dn = (p_tgt_dn - mid) * MULTIPLIER if p_tgt_dn is not None else None
        pnl_tgt_up = (p_tgt_up - mid) * MULTIPLIER if p_tgt_up is not None else None
        pnl_inv = (p_inv - mid) * MULTIPLIER if p_inv is not None else None

        # R ของ option — ตัวเลขที่ตัดสินจริง ต่างจาก plan_r ซึ่งเป็น R ของราคา underlying
        r_opt = (pnl_tgt / abs(pnl_inv)) if (pnl_tgt and pnl_inv and pnl_inv < 0) else None

        # ผลตอบแทนต่อทุนที่จ่าย — ต้องดูคู่กับ r_option เสมอ
        # เพราะ R เพียว ๆ จะเชียร์ DTE ยาวเสมอ (ธีต้าน้อยกว่า) โดยไม่บอกว่าต้องจ่ายแพงกว่ามาก
        # 60DTE อาจให้ R ดีกว่า 14DTE แต่ใช้ทุนต่อสัญญาเป็นเท่าตัวเพื่อกินระยะทางเท่าเดิม
        ret_pct = (pnl_tgt / (mid * MULTIPLIER) * 100) if pnl_tgt is not None else None

        # ── เหตุผลที่ตก — เก็บทุกข้อ ไม่หยุดที่ข้อแรก ให้เห็นภาพรวมว่าตกกี่ด่าน ──
        reasons = []
        if dte < min_dte:
            reasons.append(f"DTE สั้นไป ({dte} < {min_dte})")
        if dte > max_dte:
            reasons.append(f"DTE ยาวไป ({dte} > {max_dte})")
        if delta == delta and not (dlo <= delta <= dhi):
            reasons.append(f"Δ นอกแบนด์ ({delta:.2f})")
        if spread_pct > MAX_SPREAD_PCT:
            reasons.append(f"spread กว้าง ({spread_pct:.1f}%)")
        if oi < MIN_OI:
            reasons.append(f"OI บาง ({oi:,.0f})")
        if theta_day_pct == theta_day_pct and theta_day_pct > MAX_THETA_DAY_PCT:
            reasons.append(f"theta สูง ({theta_day_pct:.1f}%/วัน)")
        if r_opt is None:
            reasons.append("ตีราคา R ไม่ได้")

        # ── จำนวนสัญญา: เรียก gate.contracts() ที่มีอยู่แล้ว ไม่เขียนสูตร sizing ใหม่ ──
        qty = cost = None
        if account and risk_pct and p_inv is not None:
            import gate                       # import ในฟังก์ชัน — อ่านอย่างเดียว ไม่แก้ gate.py
            sz = gate.contracts(account, risk_pct, entry_premium=mid, invalid_premium=p_inv)
            qty, cost = sz["qty"], sz["total"]

        rows.append({
            "strike": K, "dte": dte, "expiry": r["expiry"], "type": want,
            "bid": float(r["bid"]), "ask": float(r["ask"]), "mid": mid,
            "spread_pct": spread_pct, "iv_pct": iv * 100,
            "delta": delta, "theta_day": theta_day, "theta_day_pct": theta_day_pct,
            "open_interest": oi,
            "volume": float(r["volume"]) if r["volume"] == r["volume"] else 0.0,
            "px_at_target": p_tgt, "px_at_invalid": p_inv,
            "pnl_target": pnl_tgt, "pnl_target_ivdn": pnl_tgt_dn,
            "pnl_target_ivup": pnl_tgt_up, "pnl_invalid": pnl_inv,
            "r_option": r_opt, "return_pct": ret_pct,
            "breakeven": K + mid if is_call else K - mid,
            "qty": qty, "cost_total": cost,
            "status": "OK" if not reasons else "ตก",
            "reasons": " · ".join(reasons),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # เรียง: ผ่านก่อน แล้วค่อยเรียงตาม R ของ option
    out["_ok"] = (out["status"] == "OK").astype(int)
    out = out.sort_values(["_ok", "r_option"], ascending=[False, False],
                          na_position="last").drop(columns=["_ok"])
    return out.reset_index(drop=True)


def summarize(picks: pd.DataFrame, plan: dict) -> dict:
    """
    สรุปสั้น ๆ สำหรับหัวตาราง — เน้นเปรียบเทียบ R ของ underlying กับ R ของ option

    ตัวเลขคู่นี้คือสาระทั้งหมดของโมดูล: แผนอาจให้ 2.5R บนกระดาษ
    แต่พอซื้อเป็น option แล้วธีต้ากินจนเหลือ 1.2R ซึ่งเปลี่ยนคำตอบว่าควรเข้าไหม
    """
    plan_r = plan.get("plan_r") if plan else None
    ok = picks[picks["status"] == "OK"] if (picks is not None and not picks.empty) else None
    best = ok.iloc[0] if (ok is not None and not ok.empty) else None
    return {
        "n_total": 0 if picks is None or picks.empty else int(len(picks)),
        "n_ok": 0 if ok is None else int(len(ok)),
        "plan_r": plan_r,
        "best_r_option": float(best["r_option"]) if best is not None and best["r_option"] else None,
        "r_gap": (float(best["r_option"]) - plan_r)
                 if (best is not None and best["r_option"] and plan_r) else None,
        "best_strike": float(best["strike"]) if best is not None else None,
        "best_dte": int(best["dte"]) if best is not None else None,
        "best_mid": float(best["mid"]) if best is not None else None,
        "best_return_pct": (float(best["return_pct"])
                            if best is not None and best["return_pct"] else None),
    }
