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

แอปมี 4 tabs:
- **📋 Dashboard** — Signal Recap 17 ตัวหน้าเดียว: Flip distance, Level stack (flip/Max Pain/top OI),
  Dealer cushion, Walls, VIX/VVIX, Vol premium, IV vs HV, VEX, DEX, Vanna/Charm, 25Δ skew,
  Term structure, 0DTE share, Expected Move 1σ, Dealer shock ±1%, Pin score
  → คำนวณอยู่ใน `snapshot.py` (ไม่มี UI) เพื่อให้สคริปต์อีเมลใช้ซ้ำได้
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
