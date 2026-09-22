# Deployment Guide / คู่มือการติดตั้ง

The app deploys as two separate hosts:

- **Frontend** — static files on iHost KMITL (`https://www.it.kmitl.ac.th/~<username>/`)
- **Backend** — FastAPI on Render free tier (`https://<app>.onrender.com`)

iHost only runs PHP, so the FastAPI backend cannot live there. The frontend
is static HTML/JS and calls the backend over HTTPS via `config.js`.

```
Browser                                    LINE app
  │                                           │
  ▼                                           ▼
https://www.it.kmitl.ac.th/~<username>   POST /line/webhook
  │   fetch() / EventSource ผ่าน window.FINANCE_API_BASE
  ▼                                           ▼
https://<app>.onrender.com                ← Render free tier (FastAPI, 1 worker)
  │
  ▼
Neon Postgres (free tier)                ← persistent database (`DB_URL` secret)
```

**Database** — production data lives on **Neon Postgres** (free tier, no
expiry, no idle pause) via the `DB_URL` environment variable. SQLite is used
only for local development. Render's free-tier disk is ephemeral, so a
local SQLite file would lose all user data on every deploy/restart/spin-down —
Neon solves this (set up with the Neon CLI; migrations run via
`alembic upgrade head` against the Neon URL).

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
   | Build command | `pip install uv && uv sync --frozen --all-extras --no-dev` |
   | Start command | `uv run python scripts/bootstrap_runtime.py && uv run uvicorn finance_ai.main:app --host 0.0.0.0 --port $PORT` |

   Free plan has no pre-deploy/release command (`preDeployCommand`),
   so bootstrap is folded into the start command.

   ใช้ worker เดียวเท่านั้น — RAM ของแผนฟรีไม่พอสำหรับ `--workers 4`
   Render กำหนด `$PORT` ให้เอง
3. ตั้ง Environment variables ในหน้า Dashboard (ค่ามาจาก `.env` ในเครื่อง
   ห้าม commit): ตัวที่ไม่ใช่ความลับ (`APP_ENV`, `DEBUG`, `LOG_LEVEL`,
   `LLM_PROVIDER=google`, `LLM_MAX_TOKENS`, `GOOGLE_MODEL`,
   `OCR_PROVIDER`, `OCR_MODEL`, `RAG_KNOWLEDGE_BASE_DIRECTORY`)
   อยู่ใน `render.yaml` อยู่แล้ว — ต้องกรอกเองในหน้า Environment:

   | Secret | ค่า |
   |---|---|
   | `DB_URL` | connection string ของ **Neon Postgres** (ใช้ตัว pooled จาก Neon dashboard / `DATABASE_URL` ใน `.env`) — **สำคัญที่สุด** ถ้าไม่ใส่ ข้อมูลจะหายทุกครั้งที่ deploy เพราะดิสก์ Render เป็นแบบ ephemeral |
   | `GOOGLE_API_KEY` | จำเป็น — ใช้ทั้ง LLM/agent, router, OCR และ embedding ของ RAG (key เดียวจบ) |
   | `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN` | จาก LINE Developers Console — ต้องมีเพื่อเปิด `/line/webhook` |
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
- **ดิสก์เป็นแบบ ephemeral** — มีผลกับไฟล์บน Render เท่านั้น ข้อมูลผู้ใช้อยู่บน
  Neon Postgres ผ่าน `DB_URL` จึง **ไม่หาย** แต่ **index ของ ChromaDB (RAG) ยังหาย**
  ทุกครั้งที่ deploy/ตื่นจากหลับ — bootstrap สร้างใหม่อัตโนมัติตอน start
- **LINE Push quota** — แผนฟรีได้ 200 push messages/ต่อเดือน (แชร์กันทุก user)
  reply API ไม่จำกัดแต่ใช้ไม่ได้กับ agent ที่ตอบช้า จึงต้องใช้ Push

### ความปลอดภัย

- **CORS** ล็อกเฉพาะ `https://www.it.kmitl.ac.th` กับ `http://localhost:8080`
  และปิด `allow_credentials` (ไม่ใช้ cookies; `user_id` ส่งใน request body)
  ดู `src/finance_ai/main.py`
- **Secrets อยู่บนเซิร์ฟเวอร์เท่านั้น** — `GOOGLE_API_KEY`
  (LLM/agent, router, OCR, RAG embeddings ใช้ key เดียวกัน)
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
| ข้อมูลหายหลัง deploy | `DB_URL` ไม่ได้ตั้ง (ยังใช้ SQLite บนดิสก์ ephemeral) — ใส่ connection string ของ Neon |
| เชื่อมต่อ Neon ไม่ได้ / connection timeout | ใช้ connection string ตัว **pooled** (`...-pooler...`) ไม่ใช่ตัว unpooled |
| ส่ง/บันทึกข้อมูลไม่ได้ (500 เฉพาะ endpoint ที่เขียน, อ่านได้ปกติ) | บน Postgres FK ถูกบังคับจริง (ต่างจาก SQLite) — ระบบจะ auto-create user ให้เองตั้งแต่ `4ac76c0`; ถ้ายังพังดู release log ว่า deploy ล่าสุดรวมโค้ดนี้แล้ว |
| `/chat` ตอบ 500 ทันที (อ่าน DB ได้ปกติ) | LLM call พัง — เช็ค `GOOGLE_API_KEY` ใน Render dashboard (key หาย/หมดอายุ), ทดสอบ key เดียวกันจากเครื่อง local ก่อน |

### การพัฒนาต่อในอนาคต

Authentication จริง, persistent disk (สำหรับ RAG index), rate limiting, custom domain
— ดูรายการเต็มใน `websitehosting.md` ส่วน "Future improvements"
(ย้ายไป Postgres เสร็จแล้ว — ดูส่วน Database ด้านบน)

---

## English

### Architecture

- **Frontend** — static files on iHost KMITL (`https://www.it.kmitl.ac.th/~<username>/`)
- **Backend** — FastAPI on Render free tier (`https://<app>.onrender.com`)

iHost only runs PHP, so the FastAPI backend cannot live there. The frontend
is static HTML/JS and calls the backend over HTTPS via `config.js`.

**Database** — production data lives on **Neon Postgres** (free tier, no expiry,
no idle pause) via the `DB_URL` environment variable. SQLite is used only for
local development. Render's free-tier disk is ephemeral, so a local SQLite file
would lose all user data on every deploy/restart/spin-down — Neon solves this
(set up with the Neon CLI; migrations run via `alembic upgrade head` against
the Neon URL).

### Backend deployment (Render)

1. Push the repository to GitHub.
2. Create a **Web Service** on Render (free plan) connected to the repo —
   the repo's `render.yaml` works as a Blueprint, or configure manually:

   | Setting | Value |
   |---|---|
   | Runtime | Python 3.13 |
   | Build command | `pip install uv && uv sync --frozen --all-extras --no-dev` |
   | Start command | `uv run python scripts/bootstrap_runtime.py && uv run uvicorn finance_ai.main:app --host 0.0.0.0 --port $PORT` |

   Free plan has no pre-deploy/release command (`preDeployCommand`),
   so bootstrap is folded into the start command.

   One worker only: free-tier RAM cannot afford `--workers 4`
   (write contention). Render injects `$PORT`.
3. Set environment variables in the Render dashboard (values mirror local
   `.env`; never commit them): all non-secrets (`APP_ENV`, `DEBUG`,
   `LOG_LEVEL`, `LLM_PROVIDER=google`, `LLM_MAX_TOKENS`,
   `GOOGLE_MODEL`, `OCR_PROVIDER`, `OCR_MODEL`,
   `RAG_KNOWLEDGE_BASE_DIRECTORY`) already live in `render.yaml` — enter
   these secrets yourself in the Environment tab:

   | Secret | Value |
   |---|---|
   | `DB_URL` | the **Neon Postgres** connection string (use the pooled one from the Neon dashboard / `DATABASE_URL` in `.env`) — **most important**; without it, every deploy wipes the data because Render's disk is ephemeral |
   | `GOOGLE_API_KEY` | required — powers the LLM/agent, router, OCR, and RAG embeddings (one key for everything) |
   | `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN` | from the LINE Developers Console — required for `/line/webhook` |
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
- **Disk is ephemeral** — this only affects files on Render. User data lives
  on Neon Postgres via `DB_URL`, so it **survives** deploys and spin-downs.
  The **ChromaDB (RAG) index is still ephemeral** and is rebuilt automatically
  by the bootstrap on every start.
- **LINE Push quota** — the free plan allows 200 push messages/month
  (shared across all users). The reply API is unlimited but unusable for
  slow agents, hence Push.

### Security notes

- **CORS** is locked to `https://www.it.kmitl.ac.th` and
  `http://localhost:8080`, with `allow_credentials=False` (no cookies;
  `user_id` travels in request bodies). See `src/finance_ai/main.py`.
- **Secrets stay server-side.** `GOOGLE_API_KEY` (one key for the
  LLM/agent, router, OCR, and RAG embeddings) exists
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
| Data lost after deploy | `DB_URL` is not set (still on ephemeral-disk SQLite) — set the Neon connection string |
| Neon connection fails / times out | Use the **pooled** connection string (`...-pooler...`), not the unpooled one |
| Writes fail with 500 (reads work fine) | Postgres enforces foreign keys (SQLite did not) — `ensure_user_exists` auto-creates missing users since `4ac76c0`; if it still fails, check the release log includes that commit |
| `/chat` returns 500 instantly (DB reads fine) | The LLM call is failing — check `GOOGLE_API_KEY` in the Render dashboard (missing/expired key); test the same key locally first |

### Future improvements

Real authentication, persistent disk (for the RAG index), rate limiting,
custom domain — see the full list in `websitehosting.md`,
"Future improvements". (The Postgres migration is done — see the Database
section above.)
