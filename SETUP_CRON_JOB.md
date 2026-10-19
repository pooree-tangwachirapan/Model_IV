# ตั้ง cron-job.org ให้กดบันทึก 0DTE ทุก 15 นาที

ทำครั้งเดียว ใช้เวลาประมาณ 15 นาที · ต้องทำเองทั้งหมด เพราะเป็นการสร้าง token และบัญชีของคุณ

## ภาพรวม

```
cron-job.org (นาฬิกาปลุกที่ตรงเวลา)
   │  ทุก 15 นาที จ–ศ 09:00–16:45 ตามเวลานิวยอร์ก
   ▼
GitHub API → สั่งรัน workflow "0DTE Recorder (external trigger)"
   │  เริ่มทันที
   ▼
workflow เช็คเองว่าตลาดเปิดอยู่ไหม
   ├─ วันหยุด / นอก 09:28–16:20 ET → จบเลย (~20 วินาที)
   └─ ในเวลา → ดึง chain 0DTE ทุก strike จาก CBOE → commit ลง zero_dte/QQQ/<วัน>.csv
```

**ทำไมต้องทำ:** cron ของ GitHub ยิงจริงแค่ 2 จาก 26 รอบต่อวัน
ข้อมูล 24 ก.ย. – 16 ต.ค. จึงได้ 1–21 snapshot ต่อวัน และไม่มีข้อมูลหลัง 14:22 ET เลย (HANDOFF §6.16)

**ผลที่ควรได้:** วันละ 28 snapshot (09:30, 09:45, … 16:15 ET)
- quote ของ CBOE ดีเลย์ราว 15 นาที snapshot 16:15 จึงเป็นราคาตอนตลาดปิด
- เวลาไทยคือ 20:30–03:15 น. และตั้งแต่ 2 พ.ย. (หลังสหรัฐฯ เลิกเวลาฤดูร้อน 1 พ.ย.) จะเป็น 21:30–04:15 น.

---

## ขั้นที่ 1: สร้าง GitHub token (5 นาที)

1. เปิด https://github.com/settings/personal-access-tokens/new
   (หรือเดินเอง: รูปโปรไฟล์ → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**)
2. กรอกตามนี้:

   | ช่อง | ใส่ |
   |---|---|
   | Token name | `cron-job-0dte` |
   | Resource owner | `pooree-tangwachirapan` |
   | Expiration | 1 ปี (จดวันหมดอายุลงปฏิทินด้วย) |
   | Repository access | **Only select repositories** → เลือก **Model_IV** ตัวเดียว |
   | Permissions → Repository permissions → **Actions** | **Read and write** |

   - สิทธิ์อื่นไม่ต้องเปิดเลย โดยเฉพาะ **Contents**
   - **Metadata: Read-only** จะถูกเลือกให้เอง ปล่อยไว้แบบนั้น
3. กด **Generate token** แล้ว copy ค่าที่ขึ้นต้นด้วย `github_pat_` เก็บไว้ชั่วคราว (หน้าเว็บแสดงค่านี้ครั้งเดียว)

> ⚠️ **อย่าวาง token ในแชทกับ Claude หรือในไฟล์ใด ๆ ในรีโป** (Model_IV เป็น public)
> ที่เดียวที่ token ควรอยู่คือช่อง Header ใน cron-job.org

token นี้แตะได้แค่เรื่อง Actions ของ Model_IV (สั่งรัน ยกเลิก ลบประวัติ run) แก้โค้ดหรือเข้ารีโปอื่นไม่ได้

---

## ขั้นที่ 2: สมัคร cron-job.org (2 นาที)

1. เปิด https://console.cron-job.org/signup สมัครด้วยอีเมล แล้วกดยืนยันในเมล
2. ใช้แผนฟรีได้เลย

---

## ขั้นที่ 3: สร้าง cron job (5 นาที)

กด **Create cronjob**

### แท็บ Common

| ช่อง | ใส่ |
|---|---|
| Title | `Model_IV 0DTE record` |
| URL | `https://api.github.com/repos/pooree-tangwachirapan/Model_IV/actions/workflows/zero-dte-record.yml/dispatches` |
| Enable job | เปิด |
| Save responses in job history | เปิด (ใช้ดูข้อความ error ของ GitHub ตอนแก้ปัญหา) |
| Execution schedule | **Custom** |

ตั้งค่าในส่วน Custom:

| ช่อง | เลือก |
|---|---|
| Days of month | ทุกวัน |
| Days of week | **Monday – Friday** |
| Months | ทุกเดือน |
| Hours | **9, 10, 11, 12, 13, 14, 15, 16** |
| Minutes | **0, 15, 30, 45** |

ถ้าหน้าเว็บมีช่องให้พิมพ์ crontab ตรง ๆ ใส่ `0,15,30,45 9-16 * * 1-5` ได้เลย

**Time zone: `America/New_York`** ← สำคัญที่สุดในหน้านี้
- ช่องนี้อาจอยู่ใกล้ส่วนตั้งเวลา หรืออยู่ในแท็บ Advanced แล้วแต่หน้าตาเว็บรุ่นนั้น
- ใช้เวลานิวยอร์กแล้ว cron-job.org จะเลื่อนตามเวลาฤดูร้อนของสหรัฐฯ ให้เอง
- **ห้ามตั้งเป็น Asia/Bangkok** ไม่งั้นตั้งแต่ 1 พ.ย. ทุกอย่างจะเพี้ยนไป 1 ชั่วโมง

รอบที่ยิงเกินมา (09:00, 09:15, 16:30, 16:45) ไม่เสียหาย workflow จะเช็คแล้วออกเองภายในราว 20 วินาที

### แท็บ Advanced

| ช่อง | ใส่ |
|---|---|
| Request method | **POST** |
| Request body | `{"ref":"main"}` |
| Timeout | 30 วินาที (ค่าเริ่ม) |

**Headers** (กดเพิ่มทีละแถว 5 แถว):

| Key | Value |
|---|---|
| `Authorization` | `Bearer github_pat_xxxxxxxx` ← คำว่า `Bearer` เว้นวรรค แล้วตามด้วย token จากขั้นที่ 1 |
| `Accept` | `application/vnd.github+json` |
| `X-GitHub-Api-Version` | `2022-11-28` |
| `Content-Type` | `application/json` |
| `User-Agent` | `model-iv-cron` |

### แท็บ Notifications

เปิดทั้งสามข้อ:
- แจ้งเมื่อรันไม่สำเร็จ
- แจ้งเมื่อกลับมาสำเร็จหลังจากล้ม
- แจ้งเมื่อ job ถูกปิดอัตโนมัติ (cron-job.org ปิด job เองเมื่อล้มติดกันหลายครั้ง ส่วนใหญ่เป็นเพราะ token หมดอายุ)

---

## ขั้นที่ 4: ทดสอบก่อนกด Save (2 นาที)

1. กด **TEST RUN** ด้านล่างของหน้า
   - ✅ สำเร็จ: ได้ **HTTP 204 No Content** (บางครั้งเป็น 200)
   - ❌ ได้ตัวเลขอื่น: ดูตารางแก้ปัญหาด้านล่าง
2. เปิด https://github.com/pooree-tangwachirapan/Model_IV/actions
   - ควรมี run ใหม่ชื่อ **0DTE Recorder (external trigger)** ขึ้นภายใน 1 นาที และเป็นสีเขียว
   - ถ้าทดสอบตอนตลาดปิด ใน log จะเขียนว่า **"วันนี้ตลาดไม่เปิด"** หรือ **"นอกหน้าต่างบันทึก"** ถือว่าปกติ แปลว่าปุ่มที่ cron-job.org กดใช้ได้แล้ว
3. กด **Save**

---

## ขั้นที่ 5: คืนแรกที่ตลาดเปิด

- ราว 20:45 น. เวลาไทย เปิดหน้า Actions ควรเห็น run ใหม่ทุก 15 นาที
  - กดเข้าไปที่ step **Record 0DTE snapshot** จะเห็นบรรทัด `QQQ: expiry … เขียน N แถว`
  - หน้า commit ของรีโปจะมี `0DTE: snapshot HH:MM ET` เพิ่มขึ้นทุก 15 นาที
- เช้าวันถัดไป บอก Claude ว่า **"เช็ค 0DTE"** แล้ว Claude จะนับให้ว่าได้ครบ 28 snapshot ไหม

---

## แก้ปัญหา

| อาการ (ตอน TEST RUN หรือในประวัติของ cron-job.org) | สาเหตุ | วิธีแก้ |
|---|---|---|
| **401** | token ผิด, หมดอายุ หรือลืมคำว่า `Bearer ` | สร้าง token ใหม่ (ขั้นที่ 1) แล้วแก้ Header |
| **403** | token ไม่มีสิทธิ์ Actions แบบ Read and write | แก้สิทธิ์ token ที่ GitHub ไม่ต้องสร้างใหม่ |
| **404** | URL พิมพ์ผิด หรือ token ไม่ได้เลือกรีโป Model_IV | GitHub ตอบ 404 แทน 403 เมื่อ token มองไม่เห็นรีโป ให้เช็คทั้งสองข้อ |
| **422** | Request body ไม่ใช่ `{"ref":"main"}` | แก้ body |
| ได้ 204 แต่ไม่มี run ขึ้นที่ GitHub | URL ชี้ไป workflow ไฟล์อื่น | ชื่อไฟล์ต้องเป็น `zero-dte-record.yml` |
| run ขึ้นทุกรอบ แต่ log บอก "นอกหน้าต่าง" ทั้งที่ตลาดเปิด | Time zone ของ job ไม่ใช่ America/New_York | แก้ Time zone |
| run เป็นสีเหลืองพร้อมคำเตือน "บันทึก 0DTE รอบนี้ไม่สำเร็จ" | CBOE ไม่ตอบรอบนั้น | ถ้าเป็นแค่บางรอบไม่ต้องทำอะไร ถ้าเป็นทุกรอบให้บอก Claude |
| cron-job.org ส่งเมลว่าปิด job แล้ว | ล้มติดกันหลายครั้ง | ดู response ใน job history แล้วแก้ตามแถวด้านบน จากนั้นเปิด job ใหม่ |

## ปิดหรือยกเลิก

- **หยุดบันทึกชั่วคราว:** ปิด Enable job ใน cron-job.org ไม่กระทบ workflow อื่นของ Model_IV
- **ยกเลิก token:** https://github.com/settings/personal-access-tokens → เลือก token → **Revoke** มีผลทันที
- **ต่ออายุ token:** ก่อนวันหมดอายุ สร้าง token ใหม่ตามขั้นที่ 1 แล้วแก้ Header `Authorization` ใน cron-job.org

## ตัวบันทึกเดิมยังทำงานอยู่

- `armed-alert.yml` ยังบันทึก 0DTE ช่วงที่มันทำงานอยู่ (นาทีที่ :07, :22, :37, :52 ถึง 14:22 ET)
- `daily-report.yml` ยังบันทึกตอน 09:00 และ 10:00 ET
- ช่วงที่ตัวบันทึกทำงานซ้อนกัน ข้อมูลจะถี่ขึ้น ซึ่งไม่เสียหาย เพราะ `merge_data.py` ตัดแถวซ้ำด้วย key `(ts_utc, strike, cp)`
- เมื่อยืนยันแล้วว่าตัวใหม่เก็บได้ครบวัน ค่อยตัดสินว่าจะเอาการบันทึกออกจาก armed-alert หรือไม่
