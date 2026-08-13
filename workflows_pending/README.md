# workflows_pending — ไฟล์ workflow ที่แก้แล้ว รอเอาเข้าที่

โฟลเดอร์นี้ **ไม่ทำงานเอง** — GitHub Actions อ่านเฉพาะไฟล์ใน `.github/workflows/`
ไฟล์ที่นี่คือเวอร์ชันที่แก้แล้ว รอให้เอาไปวางทับของเดิม

## ทำไมไม่วางให้เลย

token ที่ใช้ push ไม่มี `workflow` scope — GitHub ปฏิเสธการเขียนไฟล์ใน `.github/workflows/`
(`refusing to allow an OAuth App to create or update workflow ... without workflow scope`)

## วิธีเอาเข้าที่ (เลือกทางใดทางหนึ่ง)

**ทาง A — ผ่านเว็บ GitHub**
เปิดไฟล์เดิมใน `.github/workflows/` → ปุ่ม ✏️ Edit → ลบทั้งหมด →
คัดลอกเนื้อไฟล์จากที่นี่ไปวางทับ → Commit

**ทาง B — ในเครื่อง**
```bash
cp workflows_pending/armed-alert.yml  .github/workflows/
cp workflows_pending/forward-test.yml .github/workflows/
git add .github/workflows/
git commit -m "fix: retry ledger push instead of swallowing rebase conflicts"
git push
```
ถ้าติด scope ให้รัน `gh auth refresh -h github.com -s workflow` ก่อน

พอเอาเข้าที่แล้ว **ลบโฟลเดอร์นี้ทิ้งได้เลย**

---

## สิ่งที่แก้

ทั้งสองไฟล์แก้จุดเดียวกัน: ขั้นตอน commit `forward_test/ledger.json`

**เดิม**
```bash
git commit -m "..."
git pull --rebase --autostash origin "${GITHUB_REF_NAME}" || true
git push origin "HEAD:${GITHUB_REF_NAME}"
```

`|| true` ทำให้ rebase ที่ล้มเหลว "ผ่าน" ไปได้ — repo ค้างกลาง rebase แล้วบรรทัดถัดไป
push ด้วย HEAD ซึ่งตอนนั้นคือ commit ของ remote **ไม่ใช่ของเรา** → ไม้ที่เพิ่งบันทึกหายเงียบ
ไม่มี error ให้เห็น เพราะทุกคำสั่งคืน exit 0

เกิดจริง 2026-08-12 (run 31621321284):
```
[main b20b467] forward test: เปิดไม้จำลอง [skip ci]
error: could not apply b20b467... forward test: เปิดไม้จำลอง [skip ci]
Could not apply b20b467...
```

**ใหม่** — rebase ต้องสำเร็จถึงจะ push · ล้มก็ `rebase --abort` แล้วลองใหม่ (หน่วง 5/10/15 วินาที)
ครบ 3 รอบยังไม่ได้ = ให้ job ล้มพร้อม `::error::` จะได้เห็นว่ามีปัญหา ไม่ใช่เงียบ

## ทำไมถึงชนกันตั้งแต่แรก

มีสอง workflow เขียนไฟล์เดียวกัน:

| workflow | ทำอะไรกับ ledger | เวลา |
|---|---|---|
| `armed-alert` | **เปิด** ไม้เมื่อ ARMED | ทุก 15 นาที ระหว่างตลาดเปิด |
| `forward-test` | **ปิด** ไม้ที่ถึงเงื่อนไข | ตามตารางของตัวเอง |

ถ้าสองอันนี้ commit ใกล้กัน rebase จะชนที่ `ledger.json`

⚠️ การ retry แก้ได้แค่ "push ไม่หลุด" — ส่วนกันไม้หายจริง ๆ อยู่ที่ฝั่ง Python แล้ว
(`forward_test.merge()` รวมโดยยึด trade id + ตัวเขียนอ่านดิสก์ใหม่แล้ว merge ก่อนบันทึกเสมอ)
สองชั้นนี้ทำงานแยกกัน ต่อให้ยังไม่เอาไฟล์นี้เข้าที่ ข้อมูลก็ไม่หายแล้ว
แค่จะมี run สีแดงให้เห็นเป็นครั้งคราว
