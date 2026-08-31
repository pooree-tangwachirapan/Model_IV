# HANDOFF — อ่านไฟล์นี้ก่อนเริ่มงานกับ Model_IV

เอกสารส่งต่อระหว่าง session · อัปเดตล่าสุด 2026-08-25

เขียนไว้เพราะบทเรียนส่วนใหญ่ในโปรเจกต์นี้**มองไม่เห็นจากการอ่านโค้ด** —
มันคือสิ่งที่รู้ได้ตอนของพังเท่านั้น ถ้าไม่จดไว้ session ถัดไปจะพลาดซ้ำ

---

## 1. โปรเจกต์นี้คืออะไร

เครื่องมือวิเคราะห์ options QQQ ของ Pooree · repo public
[github.com/pooree-tangwachirapan/Model_IV](https://github.com/pooree-tangwachirapan/Model_IV)
· clone อยู่ที่ `Desktop\Cluade_Code\Model_IV`

เริ่มจากแอป Streamlit ดู IV Surface แล้วโตเป็น: แดชบอร์ด GEX → ประตูตัดสินใจ 2 ระบบ
→ forward test อัตโนมัติ → บันทึกทุกการตัดสิน → เมลรายวัน

**สถานะจริง (25 ส.ค. 2026):** โค้ดแข็ง ข้อมูลยังน้อยมาก
- fade: 2 ไม้ (แพ้ 1 · ไม่ได้ fill 1)
- breakout: 1 ไม้ (ยังเปิด)
- prediction log: 270 record / 8 วัน
- **ยังบอกไม่ได้เลยว่าระบบมี edge** — เป้าคือ 30–50 ไม้

---

## 2. แผนที่โค้ด

กติกาเดียวที่ถือมาตลอด: **ตรรกะแยกจาก UI** เพื่อให้หน้าจอกับอีเมลตัดสินตรงกันเสมอ

| ไฟล์ | บทบาท | มี UI |
|---|---|---|
| `snapshot.py` | ดึง chain + คำนวณ 17 metric | ไม่ |
| `fa_gex.py` | GEX/flip/walls + FlashAlpha client | บางส่วน |
| `gate.py` | ประตูระบบ **fade** | ไม่ |
| `breakout.py` | ประตูระบบ **breakout** | ไม่ |
| `forward_test.py` | จำลองไม้ · fill · P&L | ไม่ |
| `predictions.py` | บันทึกทุกการตัดสิน | ไม่ |
| `merge_data.py` | รวมไฟล์ข้อมูลด้วย key | ไม่ |
| `send_report.py` | ประกอบ + ส่งเมล | ไม่ |
| `run_forward_test.py` | CLI: record / resolve / report | ไม่ |
| `iv_surface_real.py` | แอป Streamlit (7 tabs) | ใช่ |
| `*_tab.py` | หน้าจอแต่ละแท็บ | ใช่ |

ไหลแบบนี้: `snapshot.build_snapshot()` → `gate.evaluate()` / `breakout.evaluate()`
→ `forward_test.record()` + `predictions.record_all()` → `send_report`

---

## 3. สองระบบที่ตรงข้ามกันโดยตั้งใจ

| | fade (`gate.py`) | breakout (`breakout.py`) |
|---|---|---|
| เดิมพันว่า | กำแพง **ถือ** | กำแพง **แตก** |
| ต้องการ | +GEX · ราคา**ใน**โซน · โซน ≥3×EM | −GEX · ราคา**นอก**โซน · ถูกฝั่ง flip |
| เข้ายังไง | limit ที่กำแพง (รอราคามาหา) | market ทันที (ไล่) |
| โปรไฟล์ | ชนะบ่อย แพ้ใหญ่ | แพ้บ่อย ชนะใหญ่ |
| ledger | `forward_test/ledger.json` | `ledger_breakout.json` |

**ค่าคงที่ที่ห้ามละเมิด:** สองระบบ **ARMED พร้อมกันไม่ได้** (อันหนึ่งต้องการในโซน
อีกอันนอกโซน) — `breakout.conflicts_with()` เช็คให้ ถ้าเจอ = **บั๊ก ไม่ใช่โอกาสสองเด้ง**

**ห้ามเทียบ win rate ข้ามระบบ** — คนละโปรไฟล์โดยการออกแบบ เทียบได้ที่ expectancy เท่านั้น

### หน้าต่างของ breakout คำนวณมา ไม่ได้ตั้งมือ

```
d ≤ (TARGET_EM_MULT − MIN_PLAN_R × STOP_EM_FRAC) / (1 + MIN_PLAN_R)
```
ปัจจุบัน = 0.10–0.536 ×EM · มี `raise ValueError` ตอน import ถ้าตั้งค่าจนขัดกันเอง

> เวอร์ชันแรกตั้ง "ทะลุ ≥0.15%" กับ "ต้อง 2.5R" **แยกกัน** แล้วสองอันขัดกันเอง
> (เพดานเรขาคณิต 0.147% < พื้น 0.15%) → ระบบยิงไม่ได้เลยแม้แต่เคสที่ดีที่สุด
> **บทเรียน: ค่าที่สัมพันธ์กันทางคณิตศาสตร์ ต้องผูกไว้ด้วยกัน ห้ามตั้งแยก**

---

## 4. แหล่งข้อมูลและข้อจำกัด

| ข้อมูล | แหล่ง | ข้อจำกัด |
|---|---|---|
| chain + greeks ครบ (delta/gamma/vega/theta/rho) + OI + IV | **CBOE delayed** | ฟรี unlimited · ช้า ~15 นาที · **snapshot ปัจจุบันเท่านั้น ไม่มีย้อนหลัง** |
| VIX/VVIX · realized vol · futures | yfinance | ฟรี |
| COT | CFTC public API | รายสัปดาห์ · **ช้า 3 วันเสมอ** |
| SOFR/EFFR | NY Fed API | ทางการ |
| GEX/DEX/VEX/Vanna/Charm/Pin | คำนวณเอง | หน่วยเป็น convention เราเอง |

**FlashAlpha (Free tier) — 403 มี 2 สาเหตุคนละเรื่อง:**
- ETF/Index (QQQ/SPY/SPX) → ต้อง Basic $63/เดือน
- **expiration = วันนี้ (0DTE)** → ต้อง Growth

มี `fa_preflight()` บล็อกทั้งสองเคสก่อนยิง (5 req/วันหมดง่ายมาก)

**❌ CME ห้าม scrape** — Data Terms of Use ห้ามชัดเจน ทุก endpoint ตอบ 403 พร้อมข้อความห้าม
FedWatch / QuikStrike / Daily Volume Report / Term SOFR **จะไม่ดึงอัตโนมัติ** ใช้ของทดแทนจากหน่วยงานรัฐแทน

---

## 5. Workflow (GitHub Actions)

| ไฟล์ | ทำอะไร | เมื่อไหร่ |
|---|---|---|
| `tests.yml` | รัน `run_tests.py` | ทุก push (ยกเว้น `forward_test/**`) |
| `daily-report.yml` | เมล Signal Recap 2 ระบบ + log | 13:00 UTC จ–ศ |
| `armed-alert.yml` | ลูปในตัวงาน เช็คทุก 15 นาที · เปิดไม้ 2 ระบบ | ตลาดเปิด |
| `forward-test.yml` | ปิดไม้ + รายงานรายเดือน | ตามตาราง |

**ลูปอยู่ในตัวงาน ไม่ใช่ cron 17 รอบ** — GitHub **ทิ้งรอบ** ไม่ใช่แค่ดีเลย์
(ตั้ง 17 รอบ ยิงจริง 5) ให้ cron ยิงครั้งเดียวแล้วงานนอนรอเองทุก 15 นาที

`gh auth refresh -h github.com -s workflow` ต้องให้ **Pooree กด device code ในเบราว์เซอร์เอง**
(รัน background แล้วอ่าน one-time code จาก output ส่งให้เขา) — ตอนนี้ token มี scope แล้ว

---

## 6. 🔥 บทเรียนที่จ่ายด้วยของพัง — ส่วนสำคัญที่สุดของเอกสารนี้

### 6.1 อย่าใช้ max/min ตัดสิน scale ของข้อมูลการเงิน

```python
if iv.max() > 5: iv /= 100          # ❌ deep-OTM มี IV 850% ได้จริง
if iv.median() > 5: iv /= 100       # ✅
```
บั๊กเดิมทำ ATM IV เพี้ยน **100 เท่า** (0.24% แทน 23.7%) กระทบ gamma + flip ทั้งกระดาน
และ **ยิงเป็นบางวัน** แล้วแต่ว่าวันนั้นมีสัญญาหางยาวโผล่มั้ย → จับยากมาก

### 6.2 วันที่ต้องเทียบเป็น `.date()` และห้าม clamp

```python
max((expiry - now).days, 0)                    # ❌
(expiry.date() - now.date()).days              # ✅
```
ของเดิมทำ expiry ที่หมดอายุแล้วกลายเป็น "0DTE ปลอม" และ 1DTE กลายเป็น 0
แล้วโดน filter `days > 0` ทิ้ง → **near-term expiry หายเงียบมาตลอด**

เช่นเดียวกัน `MAX_HOLD_DAYS` ต้องนับ**วันทำการ** ไม่ใช่ `timedelta(days=3)` —
ไม้ที่เปิดวันศุกร์ได้ถือจริงวันเดียว แล้วโดน timeout → สถิติเอียงไปทาง "ไม่มี edge"

### 6.3 Forward test ต้องเช็คว่าไม้ได้ fill จริง

fade ตั้ง limit ที่กำแพงซึ่งราคาอาจไม่เคยแตะ ของเดิมไม่เช็ค → นับไม้ที่ไม่เคยเข้าเป็นชนะ

> เคสจริง 17 ส.ค.: SHORT limit 735 · ราคาขึ้นสูงสุด **734.41 (ขาด 0.59)** แล้วลง 715.92
> ทะลุเป้า → โค้ดเดิมจะบันทึก **ชนะ +$598** ทั้งที่ไม่เคยได้เข้าไม้

ตอนนี้: limit ต้องมีแท่งคลุม entry · market fill ที่ราคาเปิดแท่งถัดไป (กิน slippage จริง)
· ไม่ fill = `no_fill` แยกจากสถิติ + รายงาน **fill rate**

### 6.4 ไฟล์ที่ workflow สร้าง ต้องเช็คว่ามีคน commit จริง — พลาดมาแล้ว 3 ครั้ง

1. `git pull --rebase ... || true` → **กลืน conflict** แล้ว push ทับด้วย HEAD ของ remote
2. `git add forward_test/ledger.json` ระบุไฟล์เดียว → `ledger_breakout.json` + `log/` ถูกทิ้งทุกรอบ
3. `daily-report.yml` เขียน log แต่**ไม่มีขั้นตอน commit เลย**

รูปแบบเดียวกันทั้งสามครั้ง: **สร้างถูกต้อง แล้วหายเงียบตอนจบ job**

### 6.5 Retry แก้ได้แค่ race ไม่ได้แก้ content conflict

armed-alert #29/#30/#33 พังเพราะ:
```
CONFLICT (content): Merge conflict in forward_test/log/2026-08.jsonl
```
retry 3 รอบชนเหมือนกันทั้ง 3 รอบ — ไม่ใช่ race แต่เนื้อไฟล์ต่างกันจริง

**วิธีถูก:** ไฟล์พวกนี้เป็นชุด record ที่มี key → รวมด้วย key ไม่ใช่ด้วยข้อความ
```
merge_data.py --save $STASH → git reset --hard origin/main
→ merge_data.py --merge $STASH → commit → push (fast-forward ชนไม่ได้)
```

### 6.6 อื่น ๆ

- **`$` ใน Streamlit markdown = LaTeX** — `$5,000 ... $100` กลืนตัวหนาระหว่างกลาง ต้อง escape `\$`
- **Gamma Flip ต้องใช้ spot-ladder** ไม่ใช่ cumsum ข้าม strike (cumsum ให้ N/A แทบทุกครั้ง
  เพราะ index chain put-dominated ไม่มี zero-crossing)
- **Walls มี 2 นิยาม** — GEX (γ×OI ถูกดึงเข้า ATM) vs OI ล้วน (ภูเขาสัญญาจริง) แสดงทั้งคู่
  ตรงกัน = level แข็ง · ไม่ตรง = ยังไม่ยืนยัน
- **Net GEX มี 2 สูตร** — CBOE gamma vs BS gamma มีจุดศูนย์คนละที่ กฎ "spot เหนือ flip → GEX บวก"
  จึงไม่จริงเสมอ เก็บทั้งคู่แล้วเตือนเมื่อเครื่องหมายต่างกัน
- **Net GEX รวมเป็นผลต่างของเลขใหญ่สองก้อน** — พลิกเครื่องหมายง่ายมาก
  (เทียบ FlashAlpha: correlation per-strike 0.98 · walls ตรงเป๊ะ · แต่ net รวมเครื่องหมายสวนกัน)
  **ห้ามใช้เครื่องหมาย Net GEX เดี่ยว ๆ เป็น Gate**
- **`fetch_chain` ต้อง retry** — CBOE CDN พลาดเป็นครั้งคราวตอนเปิดตลาด และตอบ 200
  พร้อมข้อมูลว่างได้ (นับเป็นล้มเหลวด้วย)

---

## 7. การทดสอบ

```bash
python run_tests.py            # ทุกชุด · exit 1 ถ้าพัง
python run_tests.py snapshot   # เฉพาะชุดที่ชื่อมีคำนี้
```

7 ชุด **210+ เคส** ใน `tests/` · CI รันทุก push · ใช้ known-answer ไม่ใช่แค่ "ไม่ crash"

**เวลาเทสต์ FAIL ให้สงสัยเทสต์ก่อนสงสัยโค้ด** — รอบที่ผ่านมา 6 จาก 6 เคสที่ FAIL
เป็นเทสต์เขียนผิดเอง (เคสเสื่อมจนตัวหารเป็น 0 · คาดหวังตาม schema เก่า) ไม่ใช่บั๊กจริงสักอัน

ยังไม่มีเทสต์: `cme_reports.py` (I/O ล้วน) · UI tabs · `run_forward_test.py` (รันมือครบแล้ว)
— ประเมินแล้วว่าไม่คุ้ม ความเสี่ยงต่ำและมีการตรวจทางอื่นครอบอยู่

---

## 8. กฎการทำงานกับ Pooree

- **ภาษาไทย** · ศัพท์เทคนิคอังกฤษปนได้
- ชอบให้ลงมือทำจริง ไม่ถามยืนยันเยอะ — งานย้อนกลับได้ทำเลย
- **push ขึ้น GitHub ได้เลยเมื่อเสร็จ** แต่ scan secret ก่อนทุกครั้ง (repo public)
- API key อยู่ `.streamlit/secrets.toml` + `~/.streamlit/secrets.toml` (untracked ทั้งคู่)
  · ไฟล์ต้นทาง `Option_Sufet/API_flashalpha.txt` มี **2 key ใช้บรรทัดล่างสุด**
- เครื่องงาน TH20170: Python 3.12 ที่ `%LOCALAPPDATA%\Programs\Python\Python312`
  · console เป็น cp874 → **รัน script ภาษาไทยต้อง `PYTHONIOENCODING=utf-8`**
- launch config `iv-surface` port 8511

---

## 9. อย่าเพิ่งทำอะไรกับสิ่งเหล่านี้

- **อย่าขยับ threshold ใด ๆ** จนกว่าจะมีไม้พอ — ตอนนี้ 3 ไม้ ไม่มีข้อมูลพอบอกว่าตึงหรือหลวมเกิน
- **อย่าเชียร์ว่าระบบแม่นยำ** — ยังไม่มีหลักฐานสักชิ้น โค้ดแข็งไม่เท่ากับระบบทำเงิน
- **อย่าเพิ่มฟีเจอร์ก่อนข้อมูลจะพอ** — สิ่งที่ขาดคือเวลา ไม่ใช่โค้ด

**สิ่งที่ควรจับตาแทน:** `fill_rate` — ตอนนี้ fade fill ได้ 1 ใน 2 ถ้าครบ 10 ไม้แล้วยังครึ่งเดียว
แปลว่าการตั้ง entry ที่กำแพงพอดีตึงเกินไป ทางแก้ที่สมเหตุผลคือถอย entry เข้ามาในโซน
(เช่น กำแพง − 0.1×EM) แลกกับ R ที่ลดลง — **แต่รอข้อมูลก่อน**

---

## 10. คำสั่งที่ใช้บ่อย

```bash
python run_tests.py                                    # เทสต์ทั้งหมด
python send_report.py --dry-run                        # ดูเมลโดยไม่ส่ง (ได้ preview.html)
python run_forward_test.py record  --system breakout   # เปิดไม้จำลอง
python run_forward_test.py resolve --system fade       # ปิดไม้ที่ถึงเงื่อนไข
python run_forward_test.py report  --system fade       # สถิติ
streamlit run iv_surface_real.py                       # แอป
gh workflow run "Daily Signal Recap" --ref main        # ยิงเมลทดสอบ
gh workflow run "Armed Alert (intraday)" -f minutes=1  # ทดสอบ 1 รอบแล้วจบ
```
