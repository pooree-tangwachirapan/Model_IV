# workflows_pending — ที่พักไฟล์ workflow เมื่อ push ตรงไม่ได้

**ตอนนี้ว่าง — ทุกอย่างถูก deploy เข้า `.github/workflows/` แล้ว (2026-08-15)**

## โฟลเดอร์นี้มีไว้ทำไม

GitHub ปฏิเสธการเขียนไฟล์ใน `.github/workflows/` ถ้า token ที่ push ไม่มี `workflow` scope:

```
refusing to allow an OAuth App to create or update workflow ... without workflow scope
```

session ที่เจอข้อจำกัดนี้ให้วางไฟล์ที่แก้แล้วไว้ที่นี่แทน แล้วบอกให้เอาไปวางทับทีหลัง
**ไฟล์ในโฟลเดอร์นี้ไม่ทำงานเอง** — GitHub Actions อ่านเฉพาะ `.github/workflows/`

## เอาเข้าที่ยังไง

```bash
cp workflows_pending/*.yml .github/workflows/
git rm workflows_pending/*.yml
git add -A && git commit -m "deploy pending workflows" && git push
```

ถ้าติด scope: `gh auth refresh -h github.com -s workflow` (ต้องรันในเทอร์มินัลที่กดยืนยันได้)
หรือแก้ผ่านหน้าเว็บ GitHub → เปิดไฟล์ → ✏️ Edit → วางทับ → Commit

## อย่าทิ้งไฟล์ค้างไว้นาน

ระหว่างที่ยังไม่ deploy **ของที่รันอยู่จริงคือเวอร์ชันเก่า** — รอบนี้ค้างไว้ 1 วัน
ทำให้ระบบ breakout ไม่เคยบันทึกไม้เลยสักไม้ และบั๊ก push ledger หายยังทำงานอยู่
