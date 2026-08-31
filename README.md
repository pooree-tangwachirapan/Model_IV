> 🧭 **เริ่มงานกับโปรเจกต์นี้ครั้งแรก / session ใหม่ → อ่าน [HANDOFF.md](HANDOFF.md) ก่อน**
> รวมสถานะปัจจุบัน แผนที่โค้ด ข้อจำกัดของแหล่งข้อมูล และบทเรียนที่จ่ายด้วยของพังมาแล้ว

1️⃣ เลือก Underlying   → SPY / QQQ / IWM / SPX (dropdown)

2️⃣ โหลด Expiry List  → กดปุ่ม 1 ครั้ง (เร็ว ไม่โหลด chain)
                         แสดงทันทีว่ามี expiry กี่วัน

3️⃣ เลือก Expiry Mode:
   🗓️  1 วัน           → dropdown เลือกวันเดียว
   📅  ช่วงวัน          → slider กำหนด min/max วัน + จำนวนสูงสุด
   ⭐  Preset           → Short-term / Near-term / Long-term
   ✏️  เลือกเองหลายวัน  → multiselect checkbox

4️⃣ กดปุ่ม "🚀 โหลดและคำนวณ"
   → มี progress bar แสดงทุก expiry ที่กำลังโหลด

═══════════════════════════════════════════

## 📋 Tab Dashboard + 🧲 GEX + ⚖️ เทียบข้อมูล

แอปมี 6 tabs:
- **📋 Dashboard** — Signal Recap 17 ตัวหน้าเดียว: Flip distance, Level stack (flip/Max Pain/top OI),
  Dealer cushion, Walls, VIX/VVIX, Vol premium, IV vs HV, VEX, DEX, Vanna/Charm, 25Δ skew,
  Term structure, 0DTE share, Expected Move 1σ, Dealer shock ±1%, Pin score
  → คำนวณอยู่ใน `snapshot.py` (ไม่มี UI) เพื่อให้สคริปต์อีเมลใช้ซ้ำได้
- **🎯 Cockpit** — แปลง 17 สัญญาณให้เหลือคำตอบเดียว: *วันนี้ fade กำแพงได้ไหม ใส่กี่สัญญา*
  → ตรรกะอยู่ใน `gate.py` (ไม่มี UI) แบบเดียวกัน เพื่อให้หน้าจอกับอีเมลตัดสินตรงกันเสมอ
  → **เป็นระบบ mean-reversion ล้วน ไม่มี breakout** — ประตู "ราคาอยู่ในโซน" ตัดทิ้งทันที
  ที่ราคาหลุดกำแพง ซึ่งเป็นจุดที่ระบบ breakout จะเข้าพอดี
- **📊 P&L** — ผล forward test ระหว่างทาง: equity curve, win rate, expectancy, ไม้ที่ยังค้าง
- **🏛️ CME / Macro** — COT (Commitment of Traders) แยก Dealer / Asset Manager / Leveraged Money
  พร้อม net position + Δ รายสัปดาห์ + percentile ย้อนหลัง, อัตราอ้างอิง SOFR/EFFR/BGCR/TGCR,
  ราคา+volume futures (ES/NQ/ZQ/ZN), และอัตราดอกเบี้ยที่ตลาดคิดราคาไว้
- **📈 IV Surface** — ของเดิม (CBOE Delayed, ฟรี)
- **🧲 GEX** — Dealer Gamma Exposure: Net GEX / Gamma Flip / Call Wall / Put Wall
  - แหล่ง CBOE: คำนวณเอง (Γ × OI × 100 × S² × 0.01) ฟรี unlimited ใช้กับ QQQ/SPX ได้
  - แหล่ง FlashAlpha: `lab.flashalpha.com` — Free tier = หุ้นรายตัวเท่านั้น 5 req/วัน (ETF/Index ต้อง Basic+)
- **⚖️ เทียบข้อมูล** — เทียบ CBOE (delayed 15 นาที) vs FlashAlpha สด ๆ: spot / Net GEX / Flip / Walls / correlation per strike

## 🔑 ตั้งค่า FlashAlpha API Key (ห้าม commit key ลง repo!)

รันในเครื่อง:
```
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# แล้วแก้ใส่ key จริง (ไฟล์นี้อยู่ใน .gitignore แล้ว)
```

บน Streamlit Cloud: App → Settings → Secrets แล้ววาง
```
FLASHALPHA_API_KEY = "your-key"
```

## ▶️ รัน
```
pip install -r requirements.txt
streamlit run iv_surface_real.py
```

## 📧 อีเมลรายวันอัตโนมัติ (GitHub Actions — ฟรี)

`send_report.py` สร้าง Signal Recap แล้วส่งเข้าเมล · `.github/workflows/daily-report.yml`
ตั้ง cron จันทร์–ศุกร์ 13:00 UTC (= 09:00 ET ก่อนตลาดเปิด = 20:00 ไทย)

**ทดสอบก่อนโดยไม่ส่งจริง:**
```
python send_report.py --dry-run
```
จะได้ไฟล์ `preview.html` เปิดดูหน้าตาเมลได้

**ตั้งค่าให้ส่งจริง** — ไปที่ repo → Settings → Secrets and variables → Actions → New repository secret
เพิ่ม 3 ตัวนี้ (ค่าที่เหลือมี default ให้แล้ว):

| Secret | ค่า |
|---|---|
| `SMTP_USER` | อีเมล Gmail ผู้ส่ง |
| `SMTP_PASS` | **App Password 16 หลัก** (ไม่ใช่รหัสผ่าน Gmail ปกติ) |
| `MAIL_TO` | ผู้รับ คั่นด้วย `,` ได้หลายคน |

วิธีสร้าง App Password: Google Account → Security → เปิด 2-Step Verification → App passwords
→ สร้างใหม่ → คัดลอก 16 หลักมาใส่ `SMTP_PASS`

⚠️ **อย่าใส่รหัสผ่านลงในโค้ดหรือ commit เด็ดขาด** — repo นี้เป็น public

กดรันเองได้ที่แท็บ **Actions → Daily Signal Recap → Run workflow** (เลือก symbols ได้)

### 🎯 บล็อก "วันนี้เข้าได้ไหม" ในเมล

เมลรายวันขึ้นต้นด้วยคำตัดสินจาก `gate.py` และหัวเรื่องขึ้นต้นด้วย **ARMED / WATCH / STAND DOWN**
เพื่อให้ตัดสินใจได้จากล็อกสกรีนมือถือโดยไม่ต้องเปิดอ่าน

| verdict | แปลว่า |
|---|---|
| ⛔ `STAND_DOWN` | ประตูแข็งตกอย่างน้อย 1 ข้อ — อย่า fade วันนี้ |
| 👀 `WATCH` | ผ่านประตูแล้วแต่ราคายังไม่ถึงขอบกำแพง (หรือเรขาคณิตไม่ถึง 2R) |
| 🎯 `ARMED` | โครงสร้างเข้าเกณฑ์ + ราคาชิดกำแพง |

⚠️ **ARMED ไม่ใช่คำสั่งให้เข้า** — เครื่องตัดสินได้เฉพาะประตูที่วัดจากข้อมูลได้
ประตูที่เหลือ (วันข่าว / bid-ask / เขียนจุด invalidate / ลิมิตรายวัน) ยังต้องยืนยันเอง
และเมลนี้ใช้ CBOE delayed ~15 นาที + OI ที่อัปเดตข้ามคืน

**แจ้งเตือนระหว่างวันเฉพาะตอนเพิ่งเข้าเกณฑ์** — `.github/workflows/armed-alert.yml`

เช็ค **ทุก 15 นาที ตรงนาที 7/22/37/52 · 09:28–14:35 ET** แบ่งเป็นสองช่วง:

| ช่วง ET | ทำอะไร | ทำไม |
|---|---|---|
| 09:28–10:18 | **สังเกตอย่างเดียว** — ประเมิน + เขียน log · ไม่ส่งเมล ไม่เปิดไม้ | ชั่วโมงแรกเป็นช่วง gamma whipsaw ที่ไม่ควรเข้า แต่เป็นช่วงที่ราคาขยับแรงที่สุด ไม่เก็บไว้ก็ตอบทีหลังไม่ได้ว่าเกิดอะไร |
| 10:18–14:35 | ทำงานเต็ม — เตือนเมื่อเข้าเกณฑ์ + บันทึก forward test | พ้นช่วงผันผวนเปิดตลาดแล้ว |

โหมดสังเกตใช้ `--log-only` ซึ่ง **จงใจไม่แตะ state** — ถ้าเขียน state ตอนนั้น พอถึงเวลาเทรดได้จริง
`--only-on-change` จะเห็นว่า "ไม่เปลี่ยน" แล้วเงียบ ทั้งที่นั่นคือนาทีแรกที่ควรเตือน

*เหตุที่ต้องมีช่วงสังเกต: 2026-08-19 ราคา QQQ ไหล 721→712 ตอน 09:45 ET
แต่ log มีแค่ 09:38 แล้วเว้นยาวถึง 10:18 — ตอบไม่ได้ว่า GEX ตอนนั้นเป็นยังไง*

ความถี่ 15 นาที = ความถี่เดียวกับที่ข้อมูล CBOE
จะใหม่ขึ้นจริง (delayed ~15 นาที) ถี่กว่านี้ไม่ได้ข้อมูลใหม่

**ลูปอยู่ในตัวงาน ไม่ใช่ตั้ง cron 17 รอบ** — ของเดิมตั้ง cron 17 รอบแต่ GitHub ยิงจริงแค่ 5
(จันทร์ 10 ส.ค.: 15:27 · 16:06 · 17:06 · 18:01 · 19:11 UTC) อีก 12 รอบหายไปเฉย ๆ ไม่ใช่ cancelled
GitHub ไม่ได้แค่ดีเลย์ **มันทิ้งรอบ** และยิงจริงราวชั่วโมงละครั้งไม่ว่าจะตั้งถี่แค่ไหน
→ ให้ cron ยิงครั้งเดียว แล้วงานนั้นนอนรอเองทุก 15 นาที ซึ่งควบคุมได้จริง
มี cron สำรอง 2 ตัว (15:18 · 16:18 UTC) เผื่อตัวแรกโดนทิ้ง — ตัวที่มาทีหลังเห็นว่าหน้าต่างปิดแล้วก็ออกเอง

เงื่อนไขการส่ง 2 ชั้น — เงียบสนิทถ้าไม่เข้าทั้งคู่:

| flag | ทำอะไร |
|---|---|
| `--armed-only` | ส่งเฉพาะตอนมีตัวไหนขึ้น ARMED |
| `--only-on-change` | ส่งเฉพาะตอน verdict **เปลี่ยน** — ARMED ค้าง 2 ชม. ได้เมลใบเดียว ไม่ใช่ 8 ใบ |

สถานะรอบก่อนเก็บใน `.gate-state.json` ผ่าน `actions/cache` · cache พลาดได้ (หมดอายุ 7 วัน)
→ ถือว่าไม่เคยมีสถานะแล้วส่งเมล — พลาดทางได้เมลเกินดีกว่าพลาดทางเงียบตอนควรเตือน

⚠️ ต่อให้หลบคิวแล้ว GitHub **ก็ยังไม่รับประกันเวลา และข้ามรอบได้** — ที่ยิงถี่คือเพื่อให้
*บางรอบ* ตกในหน้าต่าง ไม่ใช่เพื่อความแม่นของรอบใดรอบหนึ่ง ยังเป็นตัวเตือนให้ไปดู ไม่ใช่ตัวจับจังหวะเข้า

⚠️ cron เป็น UTC ไม่รู้จัก DST — ตารางตั้งไว้สำหรับ **EDT** ช่วง EST ต้องบวก 1 ชม. ทุกบรรทัด

```
python send_report.py --dry-run --no-cme                        # ดูหน้าตาบล็อก gate
python send_report.py --dry-run --armed-only                    # เงียบถ้าไม่ ARMED
python send_report.py --dry-run --armed-only --only-on-change \
       --state-file .gate-state.json                            # แบบเดียวกับที่ workflow รัน
```

## ⚠️ Net GEX มีสองสูตร และมันไม่ตรงกันเสมอไป

| ค่า | คำนวณยังไง | ใช้ทำอะไร |
|---|---|---|
| `net` | Σ ของ **gamma ที่ CBOE ส่งมา** (greeks จากโมเดลของ CBOE) | ตัวเลขที่รายงานบนหน้าจอ/เมล |
| `net_profile` | ค่าบน **เส้นที่คำนวณ gamma ด้วย Black-Scholes เอง** | เส้นเดียวกับที่ใช้หา `flip` |

`flip` คือจุดที่เส้น BS ตัดศูนย์ แต่ `net` ที่โชว์มาจาก gamma ของ CBOE
**สองสูตรจึงมีจุดศูนย์คนละที่** ทำให้กฎ *"spot เหนือ flip → GEX บวก"* ไม่จริงเสมอไป

เคสจริง 2026-08-19 09:38 ET — QQQ spot 721.52 อยู่**เหนือ** flip 718.60 ถึง 2.92 จุด
แต่ `net` (CBOE γ) ยังเป็น **−0.68Bn**

ไม่ทิ้งอันไหน เพราะ gamma ของ CBOE คือค่าจากผู้ให้บริการจริง ส่วนเส้น BS เป็นสิ่งเดียว
ที่ reprice ที่ spot สมมติได้ (จำเป็นต่อการหา flip) → **เก็บทั้งคู่ แล้วเตือนเมื่อเครื่องหมายต่างกัน**

เห็นได้ 3 ที่: แถว `⚠️ GEX สองสูตรขัดกัน` ใน Dashboard · ประตูอ่อน `เครื่องหมาย GEX สองสูตร`
ในทั้งสองระบบ · และเก็บลง prediction log (`net_prof`, `net_agree`) เพื่อย้อนดูได้ว่าวันไหนคาบเกี่ยว

**ไม่ตรงกัน = อยู่ในช่วงที่จุดศูนย์ของสองวิธีคร่อมราคาอยู่พอดี = regime ยังไม่ชัด**
ตอนนั้นอย่าเชื่อเครื่องหมาย GEX ให้ดูระยะถึง flip แทน (ประตู `ห่าง Gamma Flip` กันไว้อยู่แล้ว)

## 📊 Forward Test — วัดว่าระบบมี edge จริงไหม

เดินระบบ Cockpit แบบจำลอง **QQQ อย่างเดียว · พอร์ต $5,000 · เสี่ยง 1.5%/ไม้ ($75) ·
สูงสุด 2 ไม้/วัน · ถือไม่เกิน 3 วันทำการ** — กติกาเดียวกับที่ `gate.py` บังคับตอนเทรดจริง

| ไฟล์ / workflow | ทำอะไร |
|---|---|
| `forward_test.py` | เครื่องยนต์: เปิดไม้ · ปิดไม้ · คิดสถิติ (ไม่มี UI) |
| `run_forward_test.py` | ตัวสั่งงาน: `record` / `resolve` / `report` |
| `armed-alert.yml` | เปิดไม้ระหว่างวันเมื่อ ARMED (อยู่ในลูปเดียวกับการแจ้งเตือน) |
| `forward-test.yml` | 17:10 ET ปิดไม้ประจำวัน · วันที่ 1 ของเดือนส่งรายงานเข้าเมล |
| `forward_test/ledger.json` | ผลทั้งหมด commit อยู่ในรีโป |

**วัดเป็น R ไม่ใช่ราคา option** — จำลองราคา option ต้องเดา IV/spread/fill ซึ่งกลายเป็น
การวัดสมมติฐานตัวเอง สิ่งที่อยากรู้จริงคือ *สัญญาณนี้ถูกทางกี่ครั้ง* ซึ่งวัดที่ underlying ตรงกว่า
ชนะ = `+plan_r × $75` · แพ้ = `−$75`

**ตัดสินด้วยแท่ง 5 นาที** นับเฉพาะหลังเวลาเข้าไม้ — แท่งรายวันบอกไม่ได้ว่าเป้าหรือ invalidate
โดนก่อน และมันนับราคาที่เกิดก่อนเราเข้าไม้ด้วย ซึ่งเป็นการมองอนาคตย้อนหลัง
**แตะทั้งสองในแท่งเดียวกันนับเป็นแพ้เสมอ** เพื่อไม่ให้ผลออกมาสวยเกินจริง

⚠️ **ผลจริงย่อมแย่กว่าตัวเลขนี้** — ของจริงมี spread, IV crush และ fill ที่ไม่ตรงแผน
และต่ำกว่า 20 ไม้ยังสรุปไม่ได้ว่ามี edge ให้อ่านว่า "ระบบเดินได้" ไม่ใช่ "ระบบกำไร"

```
python run_forward_test.py record                   # เปิดไม้ถ้า ARMED
python run_forward_test.py resolve                  # ปิดไม้ที่ถึงเงื่อนไข
python run_forward_test.py report --month 2026-08   # สรุป (เติม --email เพื่อส่งเมล)
```

ดูระหว่างทางได้ที่แท็บ **📊 P&L** ไม่ต้องรอสิ้นเดือน

## 📊 แหล่งข้อมูล (ฟรีทั้งหมด)

| ข้อมูล | แหล่ง |
|---|---|
| Options chain + Greeks (delta/gamma/vega/theta/rho) + OI + IV | CBOE delayed (~15 นาที) |
| VIX / VVIX / ราคาย้อนหลัง (realized vol) / futures | yfinance |
| COT (Commitment of Traders) | CFTC public reporting API |
| SOFR / EFFR / BGCR / TGCR | NY Fed markets API |
| GEX / Flip / Walls / DEX / VEX / Vanna / Charm / Pin | คำนวณเอง |

### ❌ ที่ดึงอัตโนมัติไม่ได้ (และจะไม่ทำ)

CME Group **ห้าม scraping ใน Data Terms of Use** และบล็อกจริง (ทุก endpoint ตอบ 403
พร้อมข้อความห้าม) เครื่องมือเหล่านี้จึงต้องเปิดดูบนเว็บ CME เอง:

| เครื่องมือของ CME | ที่เราใช้แทน |
|---|---|
| FedWatch Tool | implied rate จาก Fed Funds futures (front month) + EFFR จริง — บอกทิศทาง ไม่ใช่ % ต่อการประชุม |
| Daily Volume & OI Report | volume futures จาก yfinance (ไม่มี OI ของ futures) |
| Term SOFR Reference Rates | SOFR overnight จาก NY Fed (ต้นทางของ Term SOFR) |
| QuikStrike Options Analytics | Greeks/GEX ที่เราคำนวณเองจาก CBOE |
| Pace of Trading | — |

⚠️ หน่วยของ GEX/DEX/VEX เป็น convention ของระบบนี้เอง **เทียบข้ามผู้ให้บริการไม่ได้**
ให้ดูตำแหน่ง level และการเปลี่ยนแปลง ไม่ใช่เลขดิบ
