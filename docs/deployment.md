# Deployment Guide / คู่มือการติดตั้ง

The app deploys as two separate hosts:

- **Frontend** — static files on iHost KMITL (`https://www.it.kmitl.ac.th/~<username>/`)
- **Backend** — FastAPI on Render free tier (`https://<app>.onrender.com`)

iHost only runs PHP, so the FastAPI backend cannot live there. The frontend
is static HTML/JS and calls the backend over HTTPS via `config.js`.

```
Browser
  │
  ▼
https://www.it.kmitl.ac.th/~<username>   ← iHost KMITL (static HTML/CSS/JS)
  │   fetch() / EventSource ผ่าน window.FINANCE_API_BASE
  ▼
https://<app>.onrender.com                ← Render free tier (FastAPI, 1 worker)
```

---

## ไทย

### สถาปัตยกรรม

- **Frontend** — ไฟล์ static บน iHost KMITL (`https://www.it.kmitl.ac.th/~<username>/`)
- **Backend** — FastAPI บน Render แผนฟรี (`https://<app>.onrender.com`)

iHost รันได้เฉพาะ PHP จึงใส่ FastAPI ลงไปไม่ได้ frontend เป็น HTML/JS ล้วน
แล้วเรียก backend ผ่าน HTTPS ด้วย `config.js`

### ติดตั้ง Backend (Render)

1. Push repository ขึ้น GitHub
2. สร้าง **Web Service** บน Render (แผน free) เชื่อมกับ repo — สามารถใช้
   `render.yaml` ใน repo ได้เลย (Blueprint) หรือตั้งค่าเองตามตาราง:

   | การตั้งค่า | ค่า |
   |---|---|
   | Runtime | Python 3.13 |
   | Build command | `pip install uv && uv sync --frozen` |
   | Start command | `uv run python scripts/bootstrap_runtime.py && uv run uvicorn finance_ai.main:app --host 0.0.0.0 --port $PORT` |

   Free plan has no pre-deploy/release command (`preDeployCommand`),
   so bootstrap is folded into the start command.

   ใช้ worker เดียวเท่านั้น — SQLite + RAM ของแผนฟรีไม่พอสำหรับ `--workers 4`
   Render กำหนด `$PORT` ให้เอง
3. ตั้ง Environment variables ในหน้า Dashboard (ค่ามาจาก `.env` ในเครื่อง
   ห้าม commit): ตัวที่ไม่ใช่ความลับ (`APP_ENV`, `DEBUG`, `LOG_LEVEL`,
   `DB_URL`, `LLM_PROVIDER=ollama`, `LLM_MAX_TOKENS`, `OLLAMA_BASE_URL`,
   `OLLAMA_MODEL`, `OCR_PROVIDER`, `OCR_MODEL`, `RAG_KNOWLEDGE_BASE_DIRECTORY`)
   อยู่ใน `render.yaml` อยู่แล้ว — เหลือแค่ secret สองตัวที่ต้องกรอกตอน sync:
   `OLLAMA_API_KEY` และ `OCR_API_KEY` (ใช้ค่า Ollama Cloud key เดียวกันได้)
   ส่วน `GOOGLE_API_KEY` เป็นทางเลือก (เปิดใช้ RAG) — เพิ่มทีหลังได้
   ถ้าไม่มี bootstrap จะข้าม RAG แต่ API ยังทำงานปกติ
4. ตรวจสอบการติดตั้ง:

   ```bash
   curl https://<app>.onrender.com/health
   # {"status":"healthy","service":"personal-finance-ai"}
   ```

**Bootstrap ทำอะไร** — `scripts/bootstrap_runtime.py` (logic อยู่ใน
`finance_ai.core.bootstrap`) รันได้ซ้ำได้ (idempotent) ทุกครั้งที่ service เริ่มทำงาน:
(1) `alembic upgrade head` แล้ว (2) สร้าง RAG index — ข้ามอัตโนมัติถ้ามี index อยู่แล้ว

### ติดตั้ง Frontend (iHost)

1. คัดลอกไฟล์ทั้งหมดใน `src/finance_ai/static/` (`index.html`, `styles.css`,
   `app.js`, `config.js`) ขึ้น web root ของ iHost ผ่าน FTP
   (ลักษณะโฟลเดอร์มักเป็น `domains/<site>/public_html`)
2. แก้ `config.js` **บนเซิร์ฟเวอร์** ให้ชี้ไปที่ backend:

   ```js
   window.FINANCE_API_BASE = "https://<app>.onrender.com";
   ```

   เว้นเป็น `null` เฉพาะตอน dev ในเครื่อง (same-origin)
3. ตรวจว่าเปิดได้ที่ `https://www.it.kmitl.ac.th/~<username>/` และ health chip
   เป็นสีเขียว (backend ตอบสนอง)

Asset paths ใน `index.html` เป็นแบบ relative (`styles.css`, `app.js`)
จึงทำงานได้ใน sub-path `/~<username>/`

### ข้อจำกัดของแผนฟรี

- บริการ **หลับหลังไม่มีการใช้งาน 15 นาที** — request แรกหลังตื่นช้า ~50 วินาที
  (cold start) แก้ได้ด้วย uptime pinger (เช่น cron-job.org ping `/health`)
- **ดิสก์เป็นแบบ ephemeral** — SQLite และ ChromaDB หายทุกครั้งที่ deploy/ตื่นจากหลับ
  migration กับ RAG index สร้างใหม่อัตโนมัติตอน deploy แต่ **ข้อมูลผู้ใช้ไม่อยู่รอด**
  (เหมาะกับ demo เท่านั้น)

### ความปลอดภัย

- **CORS** ล็อกเฉพาะ `https://www.it.kmitl.ac.th` กับ `http://localhost:8080`
  และปิด `allow_credentials` (ไม่ใช้ cookies; `user_id` ส่งใน request body)
  ดู `src/finance_ai/main.py`
- **Secrets อยู่บนเซิร์ฟเวอร์เท่านั้น** — `OLLAMA_API_KEY`, `OCR_API_KEY`
  (และ `GOOGLE_API_KEY` ถ้าเปิด RAG)
  มีเฉพาะใน environment variables ของ Render ห้ามใส่ใน repo หรือไฟล์ frontend
- **ไม่มีระบบ login** — `user_id` เป็น UUID ที่ browser สร้างเอง (เก็บใน localStorage)
  ใครได้ UUID นั้นไปอ่าน/แก้ข้อมูลของ user คนนั้นได้ รับได้สำหรับ demo เพราะ
  UUID เดายาก แต่ **อย่าเก็บข้อมูลการเงินจริงที่ละเอียดอ่อน** ในระบบที่ deploy ไว้

### การแก้ปัญหา (Troubleshooting)

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| CORS error ใน console | `allow_origins` ไม่ครอบ origin ที่เรียก — เทียบกับ `src/finance_ai/main.py` |
| 503 / timeout ทั้งที่ deploy สำเร็จ | Cold start (~50 s) — รอแล้วลองใหม่ หรือใช้ uptime pinger |
| OCR คืน 503 | `OCR_*` env vars ใน Render dashboard ยังไม่ครบ/ผิด |
| Mixed content (API calls ถูก block) | `FINANCE_API_BASE` ต้องเป็น HTTPS เสมอ |
| RAG ไม่มีข้อมูลหลังตื่นจากหลับ | ดิสก์ ephemeral — bootstrap รันตอน start ทุกครั้ง จะสร้าง index ให้ใหม่ถ้าหาย |
| Migration error ตอน deploy | ดู release log ตรวจ `DB_URL` ใน env vars |

### การพัฒนาต่อในอนาคต

Authentication จริง, PostgreSQL, persistent disk, rate limiting, custom domain
— ดูรายการเต็มใน `websitehosting.md` ส่วน "Future improvements"

---

## English

### Architecture

- **Frontend** — static files on iHost KMITL (`https://www.it.kmitl.ac.th/~<username>/`)
- **Backend** — FastAPI on Render free tier (`https://<app>.onrender.com`)

iHost only runs PHP, so the FastAPI backend cannot live there. The frontend
is static HTML/JS and calls the backend over HTTPS via `config.js`.

### Backend deployment (Render)

1. Push the repository to GitHub.
2. Create a **Web Service** on Render (free plan) connected to the repo —
   the repo's `render.yaml` works as a Blueprint, or configure manually:

   | Setting | Value |
   |---|---|
   | Runtime | Python 3.13 |
   | Build command | `pip install uv && uv sync --frozen` |
   | Start command | `uv run python scripts/bootstrap_runtime.py && uv run uvicorn finance_ai.main:app --host 0.0.0.0 --port $PORT` |

   Free plan has no pre-deploy/release command (`preDeployCommand`),
   so bootstrap is folded into the start command.

   One worker only: SQLite plus free-tier RAM cannot afford `--workers 4`
   (write contention). Render injects `$PORT`.
3. Set environment variables in the Render dashboard (values mirror local
   `.env`; never commit them): all non-secrets (`APP_ENV`, `DEBUG`,
   `LOG_LEVEL`, `DB_URL`, `LLM_PROVIDER=ollama`, `LLM_MAX_TOKENS`,
   `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OCR_PROVIDER`, `OCR_MODEL`,
   `RAG_KNOWLEDGE_BASE_DIRECTORY`) already live in `render.yaml` — only two
   secrets are prompted at sync: `OLLAMA_API_KEY` and `OCR_API_KEY`
   (both can hold the same Ollama Cloud key). `GOOGLE_API_KEY` is optional
   (enables RAG) — add it later via the Environment tab. Without it,
   bootstrap skips RAG and the API still works.
4. Verify the deploy:

   ```bash
   curl https://<app>.onrender.com/health
   # {"status":"healthy","service":"personal-finance-ai"}
   ```

**What the start-command bootstrap does** — `scripts/bootstrap_runtime.py` (logic in
`finance_ai.core.bootstrap`) is idempotent and safe on every service start:
(1) `alembic upgrade head`, then (2) index the RAG knowledge base — skipped
automatically when the vector store is already populated.

### Frontend deployment (iHost)

1. Copy everything in `src/finance_ai/static/` (`index.html`, `styles.css`,
   `app.js`, `config.js`) to the iHost web root via FTP (typically
   `domains/<site>/public_html`).
2. Edit `config.js` **on the server** and set the backend URL:

   ```js
   window.FINANCE_API_BASE = "https://<app>.onrender.com";
   ```

   Leave it `null` only for same-origin local development.
3. Verify the site loads at `https://www.it.kmitl.ac.th/~<username>/` and the
   health chip turns green (the backend responds).

Asset paths in `index.html` are relative (`styles.css`, `app.js`) so the site
works under the `/~<username>/` sub-path.

### Free-tier limitations

- The service **sleeps after 15 minutes of inactivity**; the first request
  after waking takes ~50 s (cold start). Mitigate with an uptime pinger
  (e.g., cron-job.org hitting `/health`).
- **Disk is ephemeral**: SQLite and ChromaDB reset on every deploy/wake.
  Migrations and the RAG index re-apply automatically on deploy, but user
  data does **not** survive. Suitable for a demo only.

### Security notes

- **CORS** is locked to `https://www.it.kmitl.ac.th` and
  `http://localhost:8080`, with `allow_credentials=False` (no cookies;
  `user_id` travels in request bodies). See `src/finance_ai/main.py`.
- **Secrets stay server-side.** `OLLAMA_API_KEY` and `OCR_API_KEY`
  (plus `GOOGLE_API_KEY` when RAG is enabled) exist
  only as Render environment variables — never in the repository or the
  frontend files served from iHost.
- **There is no authentication.** `user_id` is a client-generated UUID kept
  in the browser's `localStorage`; anyone who obtains a UUID can read and
  modify that user's data. Acceptable for a public demo because UUIDs are
  unguessable — but do not store real, sensitive financial data on the
  deployed instance.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| CORS error in console | `allow_origins` does not cover the calling origin — check `src/finance_ai/main.py` |
| 503 / timeout despite successful deploy | Cold start (~50 s) — retry, or add an uptime pinger |
| OCR returns 503 | `OCR_*` env vars in the Render dashboard are missing/wrong |
| Mixed content (API calls blocked) | `FINANCE_API_BASE` must always be HTTPS |
| RAG empty after wake from sleep | Ephemeral disk — the start-command bootstrap re-indexes automatically on next start |
| Migration error during deploy | Check the release log and the `DB_URL` env var |

### Future improvements

Real authentication, PostgreSQL, persistent disk, rate limiting, custom
domain — see the full list in `websitehosting.md`, "Future improvements".
