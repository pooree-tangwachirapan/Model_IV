# workflows_pending — ที่พักไฟล์ workflow เมื่อ push ตรงไม่ได้

**ตอนนี้ว่าง — ทุกอย่าง deploy เข้า `.github/workflows/` แล้ว (2026-08-19)**

## โฟลเดอร์นี้มีไว้ทำไม

GitHub ปฏิเสธการเขียนไฟล์ใน `.github/workflows/` ถ้า token ที่ push ไม่มี `workflow` scope:

```
refusing to allow an OAuth App to create or update workflow ... without workflow scope
```

session ที่เจอข้อจำกัดนี้ให้วางไฟล์ที่แก้แล้วไว้ที่นี่แทน แล้วบอกให้เอาไปวางทับทีหลัง
**ไฟล์ในโฟลเดอร์นี้ไม่ทำงานเอง** — GitHub Actions อ่านเฉพาะ `.github/workflows/`

## วิธีขอ scope (ทำครั้งเดียว)

```bash
gh auth refresh -h github.com -s workflow
```

จะได้ one-time code กับลิงก์ https://github.com/login/device — **เจ้าของบัญชีต้องกดยืนยันเอง
ในเบราว์เซอร์** (เป็นการอนุญาตบัญชี ไม่ใช่สิ่งที่สั่งแทนกันได้) เสร็จแล้ว push ได้ตามปกติ

## อย่าทิ้งไฟล์ค้างไว้นาน

ระหว่างที่ยังไม่ deploy **ของที่รันอยู่จริงคือเวอร์ชันเก่า**

บทเรียนจริง 2 ครั้ง:
- ค้างไว้ 1 วัน → ระบบ breakout ไม่เคยบันทึกไม้เลย และบั๊ก push ledger หายยังทำงานอยู่
- `git add forward_test/ledger.json` ระบุไฟล์เดียว → `ledger_breakout.json` กับ `log/`
  ถูกสร้างในรันเนอร์แล้วทิ้งทุกรอบ กว่าจะรู้ก็ต่อเมื่อไม้แรกของ breakout หายไปเฉย ๆ
