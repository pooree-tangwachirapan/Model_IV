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

## 🧲 Tab GEX + ⚖️ Tab เทียบข้อมูล (ใหม่)

แอปมี 3 tabs:
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
