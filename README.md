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

ยิง **ทุก 15 นาที · 17 รอบ/วัน · 10:22–14:22 ET** = ความถี่เดียวกับที่ข้อมูล CBOE จะใหม่ขึ้นจริง
(delayed ~15 นาที) ถี่กว่านี้ไม่ได้ข้อมูลใหม่ ได้แค่เผา quota

ใช้นาที **7 / 22 / 37 / 52** ไม่ใช่ 0/15/30/45 — GitHub บอกเองว่า cron ตรงต้นชั่วโมง/เลขกลม
คือช่วงคิวแน่นที่สุด หลักฐานในรีโปนี้: `daily-report` ตั้ง `0 13` แต่จริงรัน 14:07 UTC = **ช้า 67 นาที**
ระยะห่างยังเป็น 15 นาทีเท่าเดิม แค่เลื่อนหลบคิว

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
