# Deployment Guide / คู่มือการติดตั้ง

The whole app runs on **one host**: FastAPI on Render free tier
(`https://<app>.onrender.com`) serves both the API and the web UI
(`src/finance_ai/static/`). The browser calls the API on the same origin,
so no CORS setup and no frontend config are needed.

```
Browser                                    LINE app
  │                                           │
  ▼                                           ▼
https://<app>.onrender.com  (web UI + API) ◀── POST /line/webhook
  │                                        ← Render free tier (FastAPI, 1 worker)
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

แอปทั้งหมดอยู่บน **Render ที่เดียว** — FastAPI ให้บริการทั้ง API และหน้าเว็บ
(`src/finance_ai/static/`) ที่ `https://<app>.onrender.com` หน้าเว็บเรียก API
จาก origin เดียวกัน จึงไม่ต้องตั้ง CORS หรือไฟล์ config ฝั่ง frontend

### ติดตั้งบน Render

1. Push repository ขึ้น GitHub
2. สร้าง **Web Service** บน Render (แผน free) เชื่อมกับ repo — สามารถใช้
   `render.yaml` ใน repo ได้เลย (Blueprint) หรือตั้งค่าเองตามตาราง:

   | การตั้งค่า | ค่า |
   |---|---|
   | Runtime | Python 3.13 |
   | Build command | `pip install uv && uv sync --frozen --extra openrouter --extra ollama --no-dev` |
   | Start command | `uv run --no-sync python scripts/bootstrap_runtime.py && uv run --no-sync uvicorn finance_ai.main:app --host 0.0.0.0 --port $PORT --workers 1` |

   Free plan has no pre-deploy/release command (`preDeployCommand`),
   so bootstrap is folded into the start command.

   ใช้ worker เดียวเท่านั้น — RAM ของแผนฟรีไม่พอสำหรับ `--workers 4`
   Render กำหนด `$PORT` ให้เอง
3. ตั้ง Environment variables ในหน้า Dashboard (ค่ามาจาก `.env` ในเครื่อง
   ห้าม commit): ตัวที่ไม่ใช่ความลับ (`APP_ENV`, `DEBUG`, `LOG_LEVEL`,
   `LLM_PROVIDER=google`, `GOOGLE_MODEL`, `LLM_MAX_TOKENS`,
   `OCR_PROVIDER`, `OCR_MODEL`, `RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL`,
   `RAG_KNOWLEDGE_BASE_DIRECTORY`, `RATE_LIMIT_PER_MINUTE`)
   อยู่ใน `render.yaml` อยู่แล้ว — ต้องกรอกเองในหน้า Environment:

   | Secret | ค่า |
   |---|---|
   | `DB_URL` | connection string ของ **Neon Postgres** (ใช้ตัว pooled จาก Neon dashboard / `DATABASE_URL` ใน `.env`) — **สำคัญที่สุด** ถ้าไม่ใส่ ข้อมูลจะหายทุกครั้งที่ deploy เพราะดิสก์ Render เป็นแบบ ephemeral |
   | `GOOGLE_API_KEY` | จำเป็น — ใช้ทั้ง LLM/agent, router, กูรู และ RAG embeddings (Gemini, จาก https://aistudio.google.com/apikey) |
   | `OCR_API_KEY` | จำเป็น — ใส่ค่าเดียวกับ `GOOGLE_API_KEY` (ใช้อ่านสลิป/ใบเสร็จ) |
   | `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN` | จาก LINE Developers Console — ต้องมีเพื่อเปิด `/line/webhook` |

   **อย่าตั้ง `API_KEY`** — หน้าเว็บไม่ได้ส่ง header `X-API-Key` ถ้าตั้งค่าไว้
   หน้าเว็บจะใช้งานไม่ได้ (ดูหัวข้อความปลอดภัย)
4. ตรวจสอบการติดตั้ง:

   ```bash
   curl https://<app>.onrender.com/health
   # {"status":"healthy","service":"personal-finance-ai"}
   ```

   แล้วเปิด `https://<app>.onrender.com/` ในเบราว์เซอร์ — health chip ต้องเป็นสีเขียว

**Bootstrap ทำอะไร** — `scripts/bootstrap_runtime.py` (logic อยู่ใน
`finance_ai.core.bootstrap`) รันได้ซ้ำได้ (idempotent) ทุกครั้งที่ service เริ่มทำงาน:
(1) `alembic upgrade head` แล้ว (2) สร้าง RAG index — ข้ามอัตโนมัติถ้ามี index อยู่แล้ว
(รวม index ของกูรูจากเอกสารสรุปใน `docs/guru_knowledge/`)

**แหล่งข้อมูลกูรู (GURU_FETCH_ON_STARTUP=true)** — repo เก็บแค่ลิงก์ใน
`docs/gurus/sources.yaml` ไม่เก็บตัวบทความ (repo เป็น public) หลัง API เริ่มทำงาน
จะมี background thread ดาวน์โหลดลิงก์เหล่านั้น (เคารพ robots.txt, เว้น 1 วินาทีต่อคำขอ)
ตัดเหลือเฉพาะเนื้อบทความ เขียนลง `docs/guru_knowledge/*/fetched/` แล้ว index เพิ่ม
— ไม่ทำให้ start ช้าลง ระหว่างนั้นกูรูตอบจากเอกสารสรุปไปก่อน ถ้าเว็บต้นทางเปลี่ยน
layout หรือดาวน์โหลดไม่ได้ แหล่งนั้นจะถูกข้าม (ดู log) โดยไม่กระทบ API
ในเครื่องให้ใช้ `make guru-sync` แทน

**Embedding cache (RAG_EMBEDDING_CACHE_ENABLED=true)** — ดิสก์ Render ถูกล้างทุกครั้งที่ start
ปกติจึงต้อง embed เอกสารทั้งหมดใหม่ทุกครั้ง (Gemini free tier ให้แค่ 1,000 ครั้ง/วัน)
เมื่อเปิด cache เวกเตอร์จะถูกเก็บในตาราง `embedding_cache` บน Neon Postgres
แต่ละ chunk จึง embed แค่ครั้งแรกครั้งเดียว ครั้งต่อไปอ่านจากฐานข้อมูล (ไม่เรียก API)
ถ้าเปลี่ยนโมเดล embedding เวกเตอร์ชุดเก่าจะไม่ถูกใช้ (key แยกตาม provider:model)

### ข้อจำกัดของแผนฟรี

- บริการ **หลับหลังไม่มีการใช้งาน 15 นาที** — request แรกหลังตื่นช้า ~50 วินาที
  (cold start) แก้ได้ด้วย uptime pinger ที่ ping `/health` เป็นระยะ
- **ดิสก์เป็นแบบ ephemeral** — มีผลกับไฟล์บน Render เท่านั้น ข้อมูลผู้ใช้อยู่บน
  Neon Postgres ผ่าน `DB_URL` จึง **ไม่หาย** แต่ **index ของ ChromaDB (RAG) ยังหาย**
  ทุกครั้งที่ deploy/ตื่นจากหลับ — bootstrap สร้างใหม่อัตโนมัติตอน start
- **LINE Push quota** — แผนฟรีได้ 200 push messages/ต่อเดือน (แชร์กันทุก user)
  reply API ไม่จำกัดแต่ใช้ไม่ได้กับ agent ที่ตอบช้า จึงต้องใช้ Push

### ความปลอดภัย

- **ไม่มี CORS** — หน้าเว็บกับ API อยู่ origin เดียวกัน เบราว์เซอร์จึงบล็อก
  การเรียก API จากเว็บไซต์อื่นทั้งหมดโดยอัตโนมัติ (`src/finance_ai/main.py`)
- **Secrets อยู่บนเซิร์ฟเวอร์เท่านั้น** — `GOOGLE_API_KEY` / `OCR_API_KEY` (Gemini),
  `DB_URL` และ LINE secrets มีเฉพาะใน environment variables ของ Render
  ห้ามใส่ใน repo หรือไฟล์หน้าเว็บ
- **Rate limiting** — จำกัด 30 requests/นาที/IP ผ่าน `RATE_LIMIT_PER_MINUTE`
  (`src/finance_ai/core/api_security.py`) ส่วน `API_KEY` (header `X-API-Key`)
  ปล่อยว่างไว้: key ที่ใส่ในหน้าเว็บสาธารณะไม่เป็นความลับ และถ้าตั้งไว้หน้าเว็บจะใช้ไม่ได้
  ใช้เฉพาะเมื่อเปิด API ให้สคริปต์หรือระบบอื่นที่ไม่ผ่านหน้าเว็บ
- **ไม่มีระบบ login** — `user_id` เป็น UUID ที่ browser สร้างเอง (เก็บใน localStorage)
  ใครได้ UUID นั้นไปอ่าน/แก้ข้อมูลของ user คนนั้นได้ รับได้สำหรับ demo เพราะ
  UUID เดายาก แต่ **อย่าเก็บข้อมูลการเงินจริงที่ละเอียดอ่อน** ในระบบที่ deploy ไว้

### การแก้ปัญหา (Troubleshooting)

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| หน้าเว็บขึ้น 401/403 ทุกคำขอ | ตั้ง `API_KEY` ไว้ใน Render — ลบออก (หน้าเว็บไม่ส่ง `X-API-Key`) |
| 503 / timeout ทั้งที่ deploy สำเร็จ | Cold start (~50 s) — รอแล้วลองใหม่ หรือใช้ uptime pinger |
| OCR คืน 503 | `OCR_*` env vars ใน Render dashboard ยังไม่ครบ/ผิด |
| แชทตอบว่าโควตา AI หมด | key ของ Gemini ใช้ครบโควตาแผนฟรี — รอรีเซ็ตหรือเปลี่ยนเป็น key แบบเสียเงิน (ทั้ง `GOOGLE_API_KEY` และ `OCR_API_KEY`) |
| RAG ไม่มีข้อมูลหลังตื่นจากหลับ | ดิสก์ ephemeral — bootstrap รันตอน start ทุกครั้ง จะสร้าง index ให้ใหม่ถ้าหาย |
| Migration error ตอน deploy | ดู release log ตรวจ `DB_URL` ใน env vars |
| ข้อมูลหายหลัง deploy | `DB_URL` ไม่ได้ตั้ง (ยังใช้ SQLite บนดิสก์ ephemeral) — ใส่ connection string ของ Neon |
| เชื่อมต่อ Neon ไม่ได้ / connection timeout | ใช้ connection string ตัว **pooled** (`...-pooler...`) ไม่ใช่ตัว unpooled |
| ส่ง/บันทึกข้อมูลไม่ได้ (500 เฉพาะ endpoint ที่เขียน, อ่านได้ปกติ) | บน Postgres FK ถูกบังคับจริง (ต่างจาก SQLite) — ระบบจะ auto-create user ให้เองตั้งแต่ `4ac76c0`; ถ้ายังพังดู release log ว่า deploy ล่าสุดรวมโค้ดนี้แล้ว |
| `/chat` ตอบ 500 ทันที (อ่าน DB ได้ปกติ) | LLM call พัง — เช็ค `GOOGLE_API_KEY` ใน Render dashboard (key หาย/หมดอายุ), ทดสอบ key เดียวกันจากเครื่อง local ก่อน |

### การพัฒนาต่อในอนาคต

Authentication จริง, persistent disk (สำหรับ RAG index), custom domain
(ย้ายไป Postgres เสร็จแล้ว — ดูส่วน Database ด้านบน)

---

## English

### Architecture

Everything runs on **one Render host**: FastAPI serves both the API and the
web UI (`src/finance_ai/static/`) at `https://<app>.onrender.com`. The page
calls the API on its own origin, so there is no CORS setup and no frontend
config file.

**Database** — production data lives on **Neon Postgres** (free tier, no expiry,
no idle pause) via the `DB_URL` environment variable. SQLite is used only for
local development. Render's free-tier disk is ephemeral, so a local SQLite file
would lose all user data on every deploy/restart/spin-down — Neon solves this
(set up with the Neon CLI; migrations run via `alembic upgrade head` against
the Neon URL).

### Deploying on Render

1. Push the repository to GitHub.
2. Create a **Web Service** on Render (free plan) connected to the repo —
   the repo's `render.yaml` works as a Blueprint, or configure manually:

   | Setting | Value |
   |---|---|
   | Runtime | Python 3.13 |
   | Build command | `pip install uv && uv sync --frozen --extra openrouter --extra ollama --no-dev` |
   | Start command | `uv run --no-sync python scripts/bootstrap_runtime.py && uv run --no-sync uvicorn finance_ai.main:app --host 0.0.0.0 --port $PORT --workers 1` |

   Free plan has no pre-deploy/release command (`preDeployCommand`),
   so bootstrap is folded into the start command.

   One worker only: free-tier RAM cannot afford `--workers 4`
   (write contention). Render injects `$PORT`.
3. Set environment variables in the Render dashboard (values mirror local
   `.env`; never commit them): all non-secrets (`APP_ENV`, `DEBUG`,
   `LOG_LEVEL`, `LLM_PROVIDER=google`, `GOOGLE_MODEL`, `LLM_MAX_TOKENS`,
   `OCR_PROVIDER`, `OCR_MODEL`, `RAG_EMBEDDING_PROVIDER`,
   `RAG_EMBEDDING_MODEL`, `RAG_KNOWLEDGE_BASE_DIRECTORY`,
   `RATE_LIMIT_PER_MINUTE`) already live in `render.yaml` — enter
   these secrets yourself in the Environment tab:

   | Secret | Value |
   |---|---|
   | `DB_URL` | the **Neon Postgres** connection string (use the pooled one from the Neon dashboard / `DATABASE_URL` in `.env`) — **most important**; without it, every deploy wipes the data because Render's disk is ephemeral |
   | `GOOGLE_API_KEY` | required — powers the LLM/agents, router, gurus and RAG embeddings (Gemini, from https://aistudio.google.com/apikey) |
   | `OCR_API_KEY` | required — the same value as `GOOGLE_API_KEY` (slip/receipt reading) |
   | `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN` | from the LINE Developers Console — required for `/line/webhook` |

   **Do not set `API_KEY`** — the web UI does not send an `X-API-Key`
   header, so setting it blocks the site (see Security notes).
4. Verify the deploy:

   ```bash
   curl https://<app>.onrender.com/health
   # {"status":"healthy","service":"personal-finance-ai"}
   ```

   Then open `https://<app>.onrender.com/` in a browser; the health chip
   should turn green.

**What the start-command bootstrap does** — `scripts/bootstrap_runtime.py` (logic in
`finance_ai.core.bootstrap`) is idempotent and safe on every service start:
(1) `alembic upgrade head`, then (2) index the RAG knowledge base — skipped
automatically when the vector store is already populated.

### Free-tier limitations

- The service **sleeps after 15 minutes of inactivity**; the first request
  after waking takes ~50 s (cold start). Mitigate with an uptime pinger
  hitting `/health`.
- **Disk is ephemeral** — this only affects files on Render. User data lives
  on Neon Postgres via `DB_URL`, so it **survives** deploys and spin-downs.
  The **ChromaDB (RAG) index is still ephemeral** and is rebuilt automatically
  by the bootstrap on every start.
- **LINE Push quota** — the free plan allows 200 push messages/month
  (shared across all users). The reply API is unlimited but unusable for
  slow agents, hence Push.

### Security notes

- **No CORS.** The web UI and the API share one origin, so browsers block
  calls to the API from any other website by default
  (`src/finance_ai/main.py`).
- **Secrets stay server-side.** `GOOGLE_API_KEY` / `OCR_API_KEY` (Gemini),
  `DB_URL` and the LINE secrets exist only as Render environment
  variables — never in the repository or the web UI files.
- **Rate limiting** caps each IP at 30 requests/minute via
  `RATE_LIMIT_PER_MINUTE` (`src/finance_ai/core/api_security.py`).
  `API_KEY` (`X-API-Key` header) stays unset: a key embedded in a public web
  page is not secret, and setting it would block the web UI. Use it only if
  the API is opened to scripts or other systems that bypass the web UI.
- **There is no authentication.** `user_id` is a client-generated UUID kept
  in the browser's `localStorage`; anyone who obtains a UUID can read and
  modify that user's data. Acceptable for a public demo because UUIDs are
  unguessable — but do not store real, sensitive financial data on the
  deployed instance.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Every request from the web UI gets 401/403 | `API_KEY` is set on Render — remove it (the web UI does not send `X-API-Key`) |
| 503 / timeout despite successful deploy | Cold start (~50 s) — retry, or add an uptime pinger |
| OCR returns 503 | `OCR_*` env vars in the Render dashboard are missing/wrong |
| Chat replies that the AI quota is used up | The Gemini key hit its free-tier quota — wait for the reset or switch to a billed key (both `GOOGLE_API_KEY` and `OCR_API_KEY`) |
| RAG empty after wake from sleep | Ephemeral disk — the start-command bootstrap re-indexes automatically on next start |
| Migration error during deploy | Check the release log and the `DB_URL` env var |
| Data lost after deploy | `DB_URL` is not set (still on ephemeral-disk SQLite) — set the Neon connection string |
| Neon connection fails / times out | Use the **pooled** connection string (`...-pooler...`), not the unpooled one |
| Writes fail with 500 (reads work fine) | Postgres enforces foreign keys (SQLite did not) — `ensure_user_exists` auto-creates missing users since `4ac76c0`; if it still fails, check the release log includes that commit |
| `/chat` returns 500 instantly (DB reads fine) | The LLM call is failing — check `GOOGLE_API_KEY` in the Render dashboard (missing/expired key); test the same key locally first |

### Future improvements

Real authentication, persistent disk (for the RAG index), custom domain.
(The Postgres migration is done — see the Database section above.)
