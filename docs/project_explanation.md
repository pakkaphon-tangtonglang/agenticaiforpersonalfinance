# Personal Finance AI — คู่มืออธิบายระบบฉบับสมบูรณ์

> เอกสารนี้อธิบายระบบ **Personal Finance AI** ทั้งหมด ตั้งแต่แนวคิด สถาปัตยกรรม ไปจนถึงรายละเอียดโค้ดแต่ละส่วน เขียนสำหรับนักศึกษาระดับปริญญาตรีที่มีพื้นฐาน Python และ Data Science

---

## สารบัญ

1. [ภาพรวมโปรเจกต์](#1-ภาพรวมโปรเจกต์)
2. [สถาปัตยกรรมระบบ](#2-สถาปัตยกรรมระบบ)
3. [Core Layer — โครงสร้างพื้นฐาน](#3-core-layer--โครงสร้างพื้นฐาน)
4. [Database Layer — ฐานข้อมูล](#4-database-layer--ฐานข้อมูล)
5. [Tax Calculator — เครื่องคำนวณภาษี](#5-tax-calculator--เครื่องคำนวณภาษี)
6. [Agent Layer — ระบบ AI Agent](#6-agent-layer--ระบบ-ai-agent)
7. [แนวคิดการพัฒนา](#7-แนวคิดการพัฒนา)
8. [LangSmith Tracing — การ monitor AI](#8-langsmith-tracing--การ-monitor-ai)
9. [แผนพัฒนาต่อ](#9-แผนพัฒนาต่อ)

---

## 1. ภาพรวมโปรเจกต์

### ปัญหาที่ต้องการแก้

คนไทยจำนวนมากไม่รู้วิธีคำนวณภาษีเงินได้บุคคลธรรมดา ไม่รู้ว่าตัวเองมีสิทธิ์ลดหย่อนอะไรบ้าง (เช่น RMF, SSF, ประกันชีวิต) และมักจ่ายภาษีเกินจริง ระบบภาษีของไทยมีขั้นบันไดหลายขั้น (Progressive Tax) และมีเงื่อนไขค่าลดหย่อนที่ซับซ้อน ทำให้การคำนวณด้วยตัวเองมีโอกาสผิดพลาดสูง

### วิธีแก้

สร้าง **AI Chatbot ที่คุยภาษาไทย** — ผู้ใช้แค่พิมพ์ว่า "คำนวณภาษีปี 2024 เงินเดือน 1.2 ล้าน มีลูก 2 คน ซื้อ RMF 100,000" แล้ว AI จะ:

1. **เข้าใจ** คำถามภาษาไทย (ผ่าน LLM)
2. **สกัดข้อมูล** ตัวเลขจากคำถาม (รายได้, ค่าลดหย่อน, จำนวนลูก)
3. **คำนวณภาษี** ถูกต้องตามกฎหมาย (ใช้ฟังก์ชันคำนวณจริง ไม่ใช่ให้ AI เดา)
4. **ตอบกลับ** เป็นภาษาไทยที่เข้าใจง่าย พร้อมรายละเอียดแต่ละขั้นภาษี

### Tech Stack — ทำไมเลือกเทคโนโลยีเหล่านี้

| เทคโนโลยี | ทำไมเลือก |
|-----------|----------|
| **Python 3.11+** | ภาษาหลักของ Data Science/AI มี ecosystem ใหญ่ที่สุดสำหรับ ML/LLM |
| **LangGraph** | Framework สำหรับสร้าง AI Agent ที่ควบคุม flow ได้ (ไม่ใช่แค่ส่ง prompt แล้วรอตอบ) |
| **LangChain** | Library มาตรฐานสำหรับเชื่อมต่อกับ LLM หลายตัว (Gemini, OLLAMA) ด้วยโค้ดเดียวกัน |
| **Google Gemini** | LLM ที่รองรับภาษาไทยดี มี API ฟรีสำหรับทดสอบ |
| **OLLAMA/THALLE** | LLM ที่รันบนเครื่องตัวเอง (local) สำหรับเปรียบเทียบประสิทธิภาพกับ Gemini |
| **SQLAlchemy** | ORM มาตรฐานของ Python เขียนโค้ดจัดการ DB แทนการเขียน SQL ดิบ |
| **Pydantic** | Validate ข้อมูลอัตโนมัติ + Type Safety ช่วยลด bug |
| **SQLite** | ฐานข้อมูลไฟล์เดียว ง่ายสำหรับ development (Production จะเปลี่ยนเป็น PostgreSQL) |
| **Poetry** | จัดการ dependencies และ virtual environment |
| **pytest** | Framework สำหรับเขียน test อัตโนมัติ |

### สถิติของโปรเจกต์

- **219 tests** ทดสอบอัตโนมัติ
- **96.07% code coverage** (โค้ดถูกทดสอบเกือบทุกบรรทัด)
- **mypy strict mode** — ตรวจ type ทุกจุด ไม่มี error
- **pylint 9.72/10** — คุณภาพโค้ดสูง

---

## 2. สถาปัตยกรรมระบบ

### ภาพรวมทั้งระบบ — ผู้ใช้พิมพ์คำถาม → ได้คำตอบ

```
ผู้ใช้พิมพ์: "คำนวณภาษีปี 2024 เงินเดือน 1.2 ล้าน มีลูก 2 คน ซื้อ RMF 100k"
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Router Agent  (จำแนกประเภทคำถาม)                    │
│  "นี่คือคำถามเกี่ยวกับภาษี → ส่งไป Tax Agent"        │
└──────────────────────┬──────────────────────────────┘
                       │ intent = "tax"
                       ▼
┌─────────────────────────────────────────────────────┐
│  Tax Agent  (LangGraph ReAct)                        │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ LLM Node │───▶│Tool Node │───▶│ LLM Node │       │
│  │(คิดว่าจะ │    │(คำนวณภาษี│    │(สรุปผล   │       │
│  │ ทำอะไร)  │    │ จริงๆ)   │    │ เป็นไทย) │       │
│  └──────────┘    └──────────┘    └──────────┘       │
│       │                               │              │
│       │  เรียก calculate_thai_tax      │              │
│       ▼                               ▼              │
│  ┌──────────────────────────────────────────┐        │
│  │  Tax Calculator  (คณิตศาสตร์ล้วน)        │        │
│  │  - ขั้นบันไดภาษี 8 ขั้น                   │        │
│  │  - ค่าลดหย่อน 12 ประเภท                  │        │
│  │  - ใช้ Decimal (แม่นยำ ไม่ปัดเศษผิด)     │        │
│  └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
       │
       ▼
ผลลัพธ์: "ภาษีที่ต้องจ่าย 89,000 บาท อัตราภาษีที่แท้จริง 7.42%
         ลดหย่อนรวม 250,000 บาท (ส่วนตัว 60k + ลูก 60k + RMF 100k + ...)"
```

### Layered Architecture — ทำไมแยกเป็นชั้น

โปรเจกต์นี้ออกแบบเป็น 4 ชั้น (Layer) ที่แยกอิสระจากกัน:

```
┌─────────────────────────────────────────┐
│    Layer 4: Agents (AI ฉลาด)            │  ← คุยกับผู้ใช้ เข้าใจภาษาไทย
│    Router Agent, Tax Agent              │
├─────────────────────────────────────────┤
│    Layer 3: Tools (เครื่องคำนวณ)         │  ← คำนวณภาษีจริง ไม่พึ่ง AI
│    Tax Calculator, Tax Service          │
├─────────────────────────────────────────┤
│    Layer 2: Database (ฐานข้อมูล)        │  ← เก็บข้อมูลผู้ใช้ รายได้ ค่าลดหย่อน
│    Models, CRUD, Migrations             │
├─────────────────────────────────────────┤
│    Layer 1: Core (โครงสร้างพื้นฐาน)      │  ← Config, Logging, LLM Clients
│    Config, Logging, LLM Factory         │
└─────────────────────────────────────────┘
```

**ทำไมต้องแยกชั้น? (Separation of Concerns)**

เปรียบเสมือนบริษัท — แผนกบัญชี ไม่จำเป็นต้องรู้ว่าแผนก IT ทำงานอย่างไร แต่ละแผนกทำหน้าที่ของตัวเอง ถ้าจะเปลี่ยนวิธีเก็บข้อมูล (เช่น เปลี่ยนจาก SQLite เป็น PostgreSQL) ก็แก้แค่ Layer 2 โดยไม่กระทบ Layer อื่น

ข้อดี:
- **ทดสอบง่าย** — ทดสอบแต่ละชั้นแยกกันได้ ไม่ต้อง setup ทั้งระบบ
- **แก้ไขง่าย** — แก้ชั้นหนึ่งไม่พังอีกชั้น
- **ขยายง่าย** — เพิ่ม Agent ใหม่ (เช่น Investment Agent) โดยไม่แก้ชั้นอื่น
- **เข้าใจง่าย** — แต่ละไฟล์ทำหน้าที่เดียว อ่านแล้วเข้าใจทันที

### โครงสร้างโฟลเดอร์

```
personal-finance-ai/
├── src/finance_ai/
│   ├── core/            ← Layer 1: Config, Logging, LLM Clients
│   │   ├── config.py         การตั้งค่า (อ่านจาก .env)
│   │   ├── logging.py        ระบบ log
│   │   └── llm/              LLM clients ระดับต่ำ
│   │       ├── base.py            Abstract base class
│   │       ├── factory.py         Factory สร้าง client
│   │       ├── google_client.py   Google Gemini client
│   │       └── ollama_client.py   OLLAMA client
│   ├── database/        ← Layer 2: ฐานข้อมูล
│   │   ├── base.py           Base class + Mixins
│   │   ├── session.py        จัดการ connection
│   │   ├── models/           7 ตาราง (User, Income, Deduction, ...)
│   │   └── crud/             CRUD operations สำหรับแต่ละตาราง
│   ├── tools/           ← Layer 3: เครื่องคำนวณ
│   │   ├── tax_constants.py  ค่าคงที่ภาษี (ขั้นบันได, เพดานลดหย่อน)
│   │   ├── tax_calculator.py คำนวณภาษี (pure functions)
│   │   └── tax_service.py    เชื่อม DB กับ calculator
│   ├── agents/          ← Layer 4: AI Agents
│   │   ├── schemas.py        State schemas (TypedDict, Pydantic)
│   │   ├── prompts.py        System prompts ภาษาไทย
│   │   ├── llm_factory.py    LangChain ChatModel factory
│   │   ├── tax_tools.py      @tool wrapper สำหรับ LangGraph
│   │   ├── tax_agent.py      LangGraph StateGraph
│   │   └── router_agent.py   จำแนกคำถาม + route
│   └── rag/             ← (ยังไม่ได้ทำ — อนาคต)
├── tests/               ← ทดสอบอัตโนมัติ (mirror src/ structure)
├── alembic/             ← Database migrations
├── scripts/             ← สคริปต์สาธิต
└── docs/                ← เอกสารนี้
```

---

## 3. Core Layer — โครงสร้างพื้นฐาน

Core Layer คือพื้นฐานที่ทุก Layer อื่นต้องใช้: การตั้งค่า (Config), การ log (Logging), และการเชื่อมต่อ LLM

### 3.1 Config — การจัดการตั้งค่า

**ปัญหา**: ถ้า hardcode API key, database URL, model name ไว้ในโค้ดตรงๆ จะ:
- เปลี่ยนค่าทุกครั้งต้องแก้โค้ดแล้ว deploy ใหม่
- เผลอ commit API key ขึ้น GitHub ทำให้ถูก hack
- แต่ละเครื่อง (dev, production) ต้องใช้ค่าต่างกัน

**วิธีแก้**: ใช้ **Pydantic Settings** อ่านค่าจากไฟล์ `.env` (environment variables)

```python
# src/finance_ai/core/config.py

class Settings(BaseSettings):
    """ตั้งค่าทั้งหมดของแอป อ่านจาก .env อัตโนมัติ"""

    model_config = SettingsConfigDict(
        env_file=".env",        # อ่านจากไฟล์ .env
        case_sensitive=False,   # ไม่สนตัวพิมพ์ใหญ่เล็ก
        extra="ignore",         # ข้ามค่าที่ไม่รู้จัก
    )

    # เลือก LLM Provider: "google" หรือ "ollama"
    llm_provider: Literal["google", "ollama"] = Field(default="google")

    # Google Gemini Config
    google_api_key: Optional[str] = Field(default=None)
    google_model: str = Field(default="gemini-pro")

    # OLLAMA Config (สำหรับ THALLE — LLM ภาษาไทยที่รันบนเครื่อง)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="THALLE")

    # ฐานข้อมูล
    db_url: str = Field(default="sqlite:///./finance_ai.db")
```

**แนวคิดที่ใช้**:
- **Pydantic Settings** — validate ค่าอัตโนมัติ เช่น `llm_temperature` ต้องอยู่ระหว่าง 0.0-1.0 ถ้าใส่ 5.0 จะ error ทันที
- **Literal Type** — `llm_provider` รับได้แค่ `"google"` กับ `"ollama"` เท่านั้น ป้องกันพิมพ์ผิด
- **Environment Variables** — แยกค่า config ออกจากโค้ด ปลอดภัยกว่า

### 3.2 Logging — ระบบบันทึก log

**ทำไมต้อง log?** เวลาระบบมีปัญหา (เช่น AI ตอบผิด, API ล่ม) ถ้าไม่มี log จะหาสาเหตุไม่ได้เลย เหมือนรถไม่มี dashboard — ไม่รู้ว่าเครื่องยนต์มีปัญหาตรงไหน

```python
# src/finance_ai/core/logging.py

def get_logger(name: str) -> logging.Logger:
    """สร้าง logger สำหรับแต่ละ module"""
    return logging.getLogger(name)

# ใช้งาน:
logger = get_logger(__name__)
logger.info("Routed query to: tax (confidence: 0.95)")
# output: 2024-01-15 10:30:45 - finance_ai.agents.router_agent - INFO - Routed query to: tax
```

ทุก module ในระบบเรียก `get_logger(__name__)` เพื่อบันทึกว่าทำอะไรอยู่ ช่วยให้ debug ได้ง่าย

### 3.3 LLM Clients — เชื่อมต่อ AI

ระบบมี LLM client อยู่ **2 ชุด** ที่แยกกัน:

#### ชุดที่ 1: Low-Level Clients (ใน `core/llm/`)

เป็น client ที่เรียก API ตรงๆ ใช้สำหรับงานง่ายๆ ที่ไม่ต้องการ tool calling

```python
# src/finance_ai/core/llm/base.py

class BaseLLMClient(ABC):
    """Abstract Base Class — กำหนด interface ที่ทุก LLM client ต้องทำตาม"""

    @abstractmethod
    def create_message(self, messages, max_tokens, temperature) -> LLMResponse:
        """ทุก client ต้อง implement method นี้"""
        ...
```

**แนวคิด: Abstract Base Class (ABC)**

ABC เปรียบเสมือน "สัญญา" (contract) — ถ้าสร้าง LLM client ใหม่ ต้อง implement `create_message()` ถ้าลืม Python จะ error ตอนสร้าง object ทันที ไม่ต้องรอจน runtime แล้วค่อยพัง

```python
# ตัวอย่าง: GoogleClient ทำตามสัญญา
class GoogleClient(BaseLLMClient):
    def create_message(self, messages, max_tokens, temperature) -> LLMResponse:
        # เรียก Google Gemini API จริง
        response = self.client.models.generate_content(...)
        return {"content": response.text, "model": self.model, ...}

# ตัวอย่าง: OLLAMAClient ก็ทำตามสัญญาเดียวกัน
class OLLAMAClient(BaseLLMClient):
    def create_message(self, messages, max_tokens, temperature) -> LLMResponse:
        # เรียก OLLAMA HTTP API ที่รันบนเครื่อง
        response = self.http_client.post("http://localhost:11434/api/chat", ...)
        return {"content": response["message"]["content"], ...}
```

#### ชุดที่ 2: LangChain ChatModels (ใน `agents/llm_factory.py`)

เป็น client ที่ใช้กับ LangGraph Agent — รองรับ **tool calling** (ให้ AI เรียกฟังก์ชันได้)

```python
# src/finance_ai/agents/llm_factory.py

def create_chat_model(settings=None) -> BaseChatModel:
    """Factory — สร้าง ChatModel ตาม provider ที่ตั้งค่าไว้"""
    if settings.llm_provider == "google":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", ...)
    if settings.llm_provider == "ollama":
        return ChatOllama(model="THALLE", ...)
    raise ValueError(f"ไม่รองรับ provider: {settings.llm_provider}")
```

**แนวคิด: Factory Pattern**

Factory Pattern คือ "โรงงาน" ที่สร้าง object ให้โดยอัตโนมัติ — ผู้เรียกไม่ต้องรู้ว่าข้างในสร้างอย่างไร แค่บอกว่าต้องการ "google" หรือ "ollama"

```
เปรียบเทียบ:
    ❌ ไม่ใช้ Factory:
       if provider == "google":
           model = ChatGoogleGenerativeAI(api_key=..., model=..., temperature=...)
       elif provider == "ollama":
           model = ChatOllama(base_url=..., model=..., temperature=...)
       # ต้องเขียนซ้ำทุกที่ที่ต้องการสร้าง model

    ✅ ใช้ Factory:
       model = create_chat_model()  # แค่บรรทัดเดียว ได้ model ที่ถูกต้อง
```

**ทำไมต้องมี 2 ชุด?** ชุดที่ 1 (low-level) ใช้กับงานทั่วไปที่ไม่ต้องการ agent ชุดที่ 2 (LangChain) ใช้กับ LangGraph Agent ที่ต้องการความสามารถ tool calling — LangChain จัดการ message format, tool binding, state management ให้หมด

**Lazy Import สำหรับ OLLAMA**:

```python
def import_chat_ollama() -> type:
    """Import ChatOllama เฉพาะเมื่อต้องการใช้เท่านั้น"""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for OLLAMA provider. "
            "Install with: pip install langchain-ollama"
        ) from exc
    return ChatOllama
```

ทำไม Lazy Import? เพราะ `langchain-ollama` เป็น optional dependency — คนที่ใช้แค่ Google Gemini ไม่จำเป็นต้องติดตั้ง ถ้า import ตอน startup ทุกครั้ง จะ error ทั้งที่ไม่ได้ใช้

---

## 4. Database Layer — ฐานข้อมูล

### 4.1 ทำไมใช้ ORM (Object-Relational Mapping)

**ปัญหาของ Raw SQL**:
```python
# ❌ เขียน SQL ตรง — อ่านยาก, เสี่ยง SQL Injection
cursor.execute("SELECT * FROM users WHERE email = '" + email + "'")
```

**วิธีแก้ — ใช้ SQLAlchemy ORM**:
```python
# ✅ เขียนเป็น Python class — อ่านง่าย, ปลอดภัย
user = session.get(User, user_id)
user.full_name = "สมชาย"
session.commit()
```

ORM แปลง Python object เป็น SQL ให้อัตโนมัติ — เขียนโค้ด Python ปกติ ไม่ต้องเขียน SQL

### 4.2 Base Classes และ Mixins

```python
# src/finance_ai/database/base.py

class Base(DeclarativeBase):
    """ฐานของทุก model — SQLAlchemy ใช้ class นี้รู้ว่าต้องสร้างตาราง"""

class TimestampMixin:
    """Mixin — เพิ่ม created_at และ updated_at ให้ทุกตาราง"""
    created_at = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime, onupdate=lambda: datetime.now(timezone.utc))

class UUIDPrimaryKeyMixin:
    """Mixin — ใช้ UUID เป็น primary key แทนเลข 1, 2, 3"""
    id = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
```

**แนวคิด: Mixin**

Mixin คือ "ชิ้นส่วน" ที่เอาไป "ผสม" กับ class ได้ เพื่อเพิ่มความสามารถโดยไม่ต้องเขียนซ้ำ

```python
# ทุก model สืบทอด 3 class:
class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    # ได้ id (UUID), created_at, updated_at มาฟรี ไม่ต้องเขียนเอง
    email = mapped_column(String(255), unique=True)
    full_name = mapped_column(String(255))
    ...
```

**ทำไมใช้ UUID แทนเลข auto-increment?** UUID (เช่น `"a1b2c3d4-..."`) ไม่ซ้ำกันแม้จะสร้างจากหลายเครื่องพร้อมกัน เหมาะกับระบบที่อาจมีหลาย server ในอนาคต และไม่เปิดเผยจำนวนข้อมูล (ถ้าเป็น id=42 คนนอกจะรู้ว่ามี user 42 คน)

### 4.3 Database Models — 7 ตาราง

ระบบมี 7 ตารางที่เชื่อมกัน:

```
┌──────────────────┐
│      User        │ ← ศูนย์กลาง ทุก model เชื่อมมาที่นี่
│  email, name,    │
│  tax_id,         │
│  marital_status, │
│  children, ...   │
└──────┬───────────┘
       │ has many (1 → N)
       ├──── Income          ← รายได้ (เงินเดือน, โบนัส, freelance)
       ├──── Deduction       ← ค่าลดหย่อน (ประกัน, RMF, SSF)
       ├──── InvestmentHolding ← พอร์ตลงทุน (หุ้น, กองทุน)
       ├──── Transaction     ← ธุรกรรม (ซื้อ/ขายหุ้น, โอนเงิน)
       ├──── TaxFiling       ← ผลการคำนวณภาษี (ปีละ 1 รายการ)
       └──── FinancialGoal   ← เป้าหมายการเงิน (เก็บเงินซื้อบ้าน)
```

**ตัวอย่าง User Model**:
```python
# src/finance_ai/database/models/user.py

class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255))
    tax_id: Mapped[Optional[str]] = mapped_column(String(13), unique=True)
    marital_status: Mapped[str] = mapped_column(String(20), default="single")
    number_of_children: Mapped[int] = mapped_column(Integer, default=0)
    number_of_parents: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships — เชื่อมกับตารางอื่น
    incomes: Mapped[list["Income"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    # cascade="all, delete-orphan" = ลบ user → ลบ income ทั้งหมดของ user ด้วย
```

**สิ่งที่ควรรู้**:
- `unique=True` — email ซ้ำไม่ได้
- `index=True` — สร้าง index ค้นหาเร็วขึ้น (เหมือนสารบัญหนังสือ)
- `cascade="all, delete-orphan"` — ลบ parent (User) แล้ว child (Income) ถูกลบตาม

### 4.4 CRUD Pattern — Generic BaseCRUD

**CRUD** = **C**reate, **R**ead, **U**pdate, **D**elete — 4 การกระทำพื้นฐานกับฐานข้อมูล

ทุก model ต้องมี CRUD เหมือนกัน แทนที่จะเขียนซ้ำ 7 ครั้ง ใช้ **Generic Programming**:

```python
# src/finance_ai/database/crud/base_crud.py

ModelType = TypeVar("ModelType", bound=Base)  # ตัวแปร type ที่เป็น model อะไรก็ได้

class BaseCRUD(Generic[ModelType]):
    """CRUD กลาง ใช้ได้กับทุก model"""

    def __init__(self, model: type[ModelType]) -> None:
        self.model = model

    def get_by_id(self, session, record_id) -> Optional[ModelType]:
        """อ่าน 1 record ตาม id"""
        return session.get(self.model, record_id)

    def create(self, session, **kwargs) -> ModelType:
        """สร้าง record ใหม่"""
        instance = self.model(**kwargs)
        session.add(instance)
        session.commit()
        return instance

    def update(self, session, record_id, **kwargs) -> Optional[ModelType]:
        """แก้ไข record"""
        instance = self.get_by_id(session, record_id)
        for key, value in kwargs.items():
            setattr(instance, key, value)
        session.commit()
        return instance

    def delete(self, session, record_id) -> bool:
        """ลบ record"""
        instance = self.get_by_id(session, record_id)
        session.delete(instance)
        session.commit()
        return True
```

**แนวคิด: Generic Programming**

`Generic[ModelType]` คือ "template" — เขียนโค้ดครั้งเดียว ใช้ได้กับหลาย type:

```python
# ใช้งาน:
user_crud = UserCRUD()           # BaseCRUD สำหรับ User
income_crud = IncomeCRUD()       # BaseCRUD สำหรับ Income

user_crud.get_by_id(session, id)   # คืน User object
income_crud.get_by_id(session, id) # คืน Income object
```

แต่ละ CRUD สามารถเพิ่ม method เฉพาะทางได้ เช่น `UserCRUD` มี `get_by_email()` เพิ่ม

### 4.5 Alembic Migrations — Version Control ของฐานข้อมูล

**ปัญหา**: เวลาเพิ่มคอลัมน์ใหม่ในตาราง ถ้าแก้โค้ดแล้ว DB เดิมไม่อัปเดตตาม ระบบจะพัง

**วิธีแก้**: Alembic เป็นเครื่องมือ migration — เหมือน git แต่สำหรับ database schema

```bash
# สร้าง migration ใหม่เมื่อเปลี่ยน model
alembic revision --autogenerate -m "add phone_number to users"

# รัน migration (อัปเดต DB)
alembic upgrade head
```

migration file บันทึกว่า "เพิ่มคอลัมน์อะไร" และ "ถ้าจะ rollback ต้องลบอะไร" — ทำให้อัปเดต DB ได้อย่างปลอดภัย ไม่ต้องลบ DB แล้วสร้างใหม่

---

## 5. Tax Calculator — เครื่องคำนวณภาษี

### 5.1 ภาษีเงินได้แบบขั้นบันได (Progressive Tax)

ภาษีเงินได้ของไทยใช้ระบบ **ขั้นบันได** — รายได้ส่วนแรกเสียภาษีน้อย ส่วนที่มากขึ้นเสียภาษีมากขึ้น:

```
ขั้นบันไดภาษีไทย (2024):
┌──────────────────────────────┬──────────┐
│ เงินได้สุทธิ (บาท)            │ อัตราภาษี │
├──────────────────────────────┼──────────┤
│ 0 - 150,000                 │  0% (ยกเว้น) │
│ 150,001 - 300,000           │  5%      │
│ 300,001 - 500,000           │ 10%      │
│ 500,001 - 750,000           │ 15%      │
│ 750,001 - 1,000,000         │ 20%      │
│ 1,000,001 - 2,000,000       │ 25%      │
│ 2,000,001 - 5,000,000       │ 30%      │
│ 5,000,001 ขึ้นไป             │ 35%      │
└──────────────────────────────┴──────────┘

ตัวอย่าง: เงินได้สุทธิ 500,000 บาท
  - 0 - 150,000         → ภาษี 0 บาท      (ยกเว้น)
  - 150,001 - 300,000   → ภาษี 7,500 บาท  (150,000 × 5%)
  - 300,001 - 500,000   → ภาษี 20,000 บาท (200,000 × 10%)
  รวม: 27,500 บาท
```

### 5.2 ค่าลดหย่อนภาษี 12 ประเภท

```python
# src/finance_ai/tools/tax_constants.py

DEDUCTION_LIMITS = {
    "personal_allowance": Decimal("60000"),    # ค่าลดหย่อนส่วนตัว (ทุกคนได้)
    "spouse_allowance": Decimal("60000"),      # คู่สมรส
    "child_allowance": Decimal("30000"),       # ลูก (ต่อคน)
    "parent_allowance": Decimal("30000"),      # พ่อแม่ (ต่อคน)
    "social_security": Decimal("9000"),        # ประกันสังคม
    "life_insurance": Decimal("100000"),       # ประกันชีวิต
    "health_insurance": Decimal("25000"),      # ประกันสุขภาพ
    "provident_fund": Decimal("500000"),       # กองทุนสำรองเลี้ยงชีพ
    "rmf": Decimal("500000"),                  # กองทุนรวมเพื่อการเลี้ยงชีพ
    "ssf": Decimal("200000"),                  # กองทุนรวมเพื่อการออม
    "mortgage_interest": Decimal("100000"),    # ดอกเบี้ยบ้าน
}

# บาง deduction มีเพดานเป็น % ของรายได้ด้วย
PERCENTAGE_CAPPED_DEDUCTIONS = {
    "provident_fund": Decimal("0.15"),   # ไม่เกิน 15% ของรายได้
    "rmf": Decimal("0.30"),              # ไม่เกิน 30% ของรายได้
    "ssf": Decimal("0.30"),              # ไม่เกิน 30% ของรายได้
}
```

### 5.3 Pure Functions — ฟังก์ชันบริสุทธิ์

**แนวคิดสำคัญ**: Tax Calculator ถูกออกแบบเป็น **Pure Functions** — ฟังก์ชันที่:
1. **ไม่ขึ้นกับ state ภายนอก** — รับ input เข้ามา คืน output ออกไป ไม่ยุ่งกับ DB, ไม่ยุ่งกับ API
2. **ให้ผลลัพธ์เดิมเสมอ** สำหรับ input เดียวกัน — ใส่รายได้ 1 ล้าน ได้ภาษีเท่าเดิมทุกครั้ง

```python
# src/finance_ai/tools/tax_calculator.py

def calculate_tax(
    gross_income: Decimal,                    # รายได้รวม
    deductions_by_type: dict[str, Decimal],   # ค่าลดหย่อนแยกตามประเภท
    withholding_tax_paid: Decimal = Decimal("0"),  # ภาษีหัก ณ ที่จ่ายแล้ว
) -> TaxCalculationResult:
    """คำนวณภาษีเงินได้บุคคลธรรมดา"""

    # 1. ตรวจสอบ input
    validate_non_negative_income(gross_income)

    # 2. รวมค่าลดหย่อน (พร้อม cap ตามกฎหมาย)
    total_deductions, _ = sum_capped_deductions(deductions_by_type, gross_income)

    # 3. คำนวณเงินได้สุทธิ
    net_income = max(gross_income - total_deductions, Decimal("0"))

    # 4. คำนวณภาษีตามขั้นบันได
    breakdown = calculate_progressive_tax(net_income)
    total_tax = calculate_total_tax_from_breakdown(breakdown)

    # 5. คำนวณอัตราภาษีที่แท้จริง
    effective_rate = calculate_effective_rate(total_tax, gross_income)

    # 6. ภาษีที่ต้องจ่ายเพิ่ม/ได้คืน
    tax_due_or_refund = total_tax - withholding_tax_paid

    return TaxCalculationResult(
        gross_income=gross_income,
        total_deductions=total_deductions,
        net_income=net_income,
        tax_breakdown=breakdown,
        total_tax=total_tax,
        effective_tax_rate=effective_rate,
        withholding_tax_paid=withholding_tax_paid,
        tax_due_or_refund=tax_due_or_refund,
    )
```

**ทำไม Pure Functions ดี?**
- **ทดสอบง่ายมาก** — ไม่ต้อง setup database ไม่ต้อง mock API แค่ใส่ตัวเลขเข้าไปแล้วเช็คผล
- **ใช้ซ้ำได้** — Agent เรียกได้ API เรียกได้ test เรียกได้
- **ไม่มี side effect** — ไม่เปลี่ยนอะไรในระบบ ปลอดภัย

**ตัวอย่างการ cap ค่าลดหย่อน**:
```python
def cap_deduction_amount(deduction_type, claimed_amount, gross_income) -> Decimal:
    """ตัด cap ค่าลดหย่อนตามกฎหมาย"""

    # Cap 1: เพดานสูงสุดตามประเภท (เช่น ประกันชีวิตไม่เกิน 100,000)
    absolute_limit = DEDUCTION_LIMITS.get(deduction_type, claimed_amount)
    capped = min(claimed_amount, absolute_limit)

    # Cap 2: บาง type มีเพดานเป็น % ของรายได้ด้วย
    # เช่น RMF ไม่เกิน 30% ของรายได้ AND ไม่เกิน 500,000
    if deduction_type in PERCENTAGE_CAPPED_DEDUCTIONS:
        percentage_limit = gross_income * PERCENTAGE_CAPPED_DEDUCTIONS[deduction_type]
        capped = min(capped, percentage_limit)

    return capped

# ตัวอย่าง:
# รายได้ 1,000,000 ซื้อ RMF 600,000
# cap_deduction_amount("rmf", 600000, 1000000)
# → min(600000, 500000) = 500000    (เพดานแน่นอน)
# → min(500000, 300000) = 300,000   (30% ของรายได้ = 300,000)
# ผลลัพธ์: ลดหย่อน RMF ได้แค่ 300,000
```

### 5.4 Tax Service — เชื่อม DB กับ Calculator

Tax Service เป็นตัวกลางที่ดึงข้อมูลจาก DB แล้วส่งให้ Calculator:

```python
# src/finance_ai/tools/tax_service.py

def calculate_tax_for_user(session, user_id, tax_year) -> TaxCalculationResult:
    """คำนวณภาษีโดยดึงข้อมูลจาก DB"""

    # 1. ดึงข้อมูล user
    user = UserCRUD().get_by_id(session, user_id)

    # 2. ดึงรายได้รวม
    gross_income = IncomeCRUD().get_total_income_for_year(session, user_id, tax_year)

    # 3. ดึงค่าลดหย่อนจาก DB
    user_deductions = aggregate_deductions_by_type(
        DeductionCRUD().get_by_user_and_year(session, user_id, tax_year)
    )

    # 4. สร้างค่าลดหย่อนอัตโนมัติจากโปรไฟล์ (มีลูก → child_allowance, แต่งงาน → spouse_allowance)
    auto_allowances = build_auto_allowances(user)

    # 5. รวม deductions
    all_deductions = merge_deductions(auto_allowances, user_deductions)

    # 6. คำนวณภาษี (เรียก pure function)
    result = calculate_tax(gross_income, all_deductions, withholding)

    # 7. บันทึกผลลง DB
    store_tax_filing_result(session, user_id, tax_year, result)

    return result
```

**แนวคิด: Separation of Concerns**
- **Calculator** (pure functions) — รับตัวเลข คำนวณ คืนผล ไม่รู้จัก DB
- **Service** — ดึงข้อมูลจาก DB ส่งให้ Calculator บันทึกผลกลับ DB
- **Agent** — เข้าใจภาษาไทย สกัดตัวเลข ส่งให้ Calculator แสดงผลเป็นไทย

ทั้ง 3 ส่วนทำคนละงาน ไม่ปนกัน

---

## 6. Agent Layer — ระบบ AI Agent

### 6.1 ทำไมต้องมี AI Agent?

Calculator คำนวณได้ถูกต้อง **แต่ผู้ใช้ต้องรู้ว่าต้องใส่อะไร** — ต้องรู้ชื่อ parameter, ต้องแปลงค่าเป็น Decimal, ต้องรู้ว่า "ลูก 2 คน" ต้องใส่เป็น `child_allowance = 60000`

AI Agent แก้ปัญหานี้ — ผู้ใช้แค่พิมพ์ภาษาไทยธรรมชาติ Agent จะ:
1. **เข้าใจ** ว่าผู้ใช้พูดอะไร
2. **สกัด** ตัวเลขจากข้อความ
3. **เรียก Calculator** ด้วย parameter ที่ถูกต้อง
4. **อธิบายผล** กลับเป็นภาษาไทย

### 6.2 Router Agent — จำแนกประเภทคำถาม

Router Agent ทำหน้าที่เป็น "ประชาสัมพันธ์" — ฟังคำถามแล้วส่งไปยังแผนกที่ถูกต้อง

```
ผู้ใช้: "คำนวณภาษีปี 2024"  → Router: intent="tax"       → Tax Agent
ผู้ใช้: "ดูพอร์ตหุ้น"         → Router: intent="investment" → (ยังไม่พร้อม)
ผู้ใช้: "วันนี้อากาศดี"       → Router: intent="unknown"    → "ขออภัย ไม่เกี่ยวกับการเงิน"
```

```python
# src/finance_ai/agents/router_agent.py

def classify_query(query: str, chat_model=None) -> RouterDecision:
    """จำแนกคำถามด้วย LLM"""
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    response = chat_model.invoke(messages)  # ส่งให้ LLM
    return parse_router_response(response.content)
    # ผลลัพธ์: RouterDecision(intent="tax", confidence=0.95)

def route_query(query: str, chat_model=None) -> dict:
    """จำแนก + ส่งไปยัง agent ที่ถูกต้อง"""
    decision = classify_query(query, chat_model)
    if decision.intent == "tax":
        return execute_tax_agent(query, chat_model)  # → Tax Agent
    return build_unsupported_response(decision)       # → "ขออภัย..."
```

**System Prompt ของ Router** (สั่ง LLM ให้ทำหน้าที่จำแนก):
```
คุณเป็นตัวจำแนกคำถามทางการเงิน (Financial Query Classifier)
จำแนกคำถามของผู้ใช้เป็นหมวดหมู่:
- "tax": คำถามเกี่ยวกับภาษี, การคำนวณภาษี, ลดหย่อนภาษี
- "investment": คำถามเกี่ยวกับการลงทุน, หุ้น, กองทุน
- "general": คำถามทั่วไปเกี่ยวกับการเงิน
- "unknown": ไม่เกี่ยวกับการเงิน

ตอบเป็น JSON เท่านั้น: {"intent": "<category>", "confidence": <0.0-1.0>}
```

**ทำไม Router เป็นแค่ LLM call ธรรมดา ไม่ใช่ LangGraph?** เพราะมันทำแค่อย่างเดียว — จำแนกข้อความ ไม่ต้องวน loop ไม่ต้องเรียก tool ใช้ LangGraph จะ overengineering

**การ parse response ที่มี code fence**:
```python
def _strip_code_fence(text: str) -> str:
    """ลบ ```json ... ``` ที่ LLM ชอบครอบ JSON"""
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text
```

LLM บางครั้งตอบเป็น ` ```json {"intent": "tax"} ``` ` แทนที่จะตอบ JSON เปล่าๆ ฟังก์ชันนี้จัดการปัญหานั้น

### 6.3 Tax Agent — หัวใจของระบบ (LangGraph ReAct)

Tax Agent ใช้ **LangGraph** สร้างเป็น **State Machine** (ระบบที่มีสถานะและเปลี่ยนสถานะตามเงื่อนไข)

#### ReAct Pattern คืออะไร?

**ReAct** = **Re**asoning + **Act**ing — AI จะ "คิด" (Reason) ก่อนว่าต้องทำอะไร แล้วค่อย "ทำ" (Act) วนไปเรื่อยๆ จนเสร็จ

```
ตัวอย่าง flow จริง:

Step 1 (Reasoning): LLM คิด...
  "ผู้ใช้ต้องการคำนวณภาษี รายได้ 1.2 ล้าน มีลูก 2 คน ซื้อ RMF 100k
   ฉันต้องเรียก calculate_thai_tax ด้วยข้อมูลเหล่านี้"

Step 2 (Acting): LLM เรียก tool
  → calculate_thai_tax(
      gross_income="1200000",
      deductions_by_type={"personal_allowance": "60000", "child_allowance": "60000", "rmf": "100000"},
      withholding_tax_paid="0"
    )

Step 3 (Tool Result): Tool คืนผลลัพธ์
  → {"total_tax": "89000", "effective_tax_rate": "0.0742", "net_income": "980000", ...}

Step 4 (Reasoning → Answer): LLM อ่านผลแล้วตอบภาษาไทย
  "ผลการคำนวณภาษีปี 2024:
   - รายได้รวม: 1,200,000 บาท
   - ค่าลดหย่อนรวม: 220,000 บาท
   - เงินได้สุทธิ: 980,000 บาท
   - ภาษีที่ต้องจ่าย: 89,000 บาท
   - อัตราภาษีที่แท้จริง: 7.42%"
```

#### แผนภาพ LangGraph StateGraph

```
                    ┌─────────────────────────────────────┐
                    │         TaxAgentState                │
                    │  {                                   │
                    │    messages: [...],  ← ประวัติสนทนา   │
                    │    tax_result: null  ← ผลภาษี        │
                    │  }                                   │
                    └─────────────────────────────────────┘

Entry Point
     │
     ▼
┌──────────┐     มี tool_calls?     ┌──────────┐
│  agent   │──── ใช่ (tools) ──────▶│  tools   │
│  (LLM)   │                        │(Tool     │
│          │◀────────────────────────│ Node)    │
└──────────┘     ส่งผลกลับ          └──────────┘
     │
     │ ไม่มี tool_calls (end)
     ▼
   [END]
```

**อธิบายทีละ node**:

**1. agent node** (LLM ตัดสินใจ):
```python
def create_llm_node(chat_model):
    """สร้าง node ที่ LLM คิดและตัดสินใจ"""
    model_with_tools = chat_model.bind_tools(TAX_TOOLS)  # "บอก" LLM ว่ามี tool อะไรให้เรียก

    def llm_node(state: TaxAgentState) -> dict:
        # ใส่ system prompt + ข้อความทั้งหมด
        messages = [SystemMessage(content=TAX_AGENT_SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node
```

`bind_tools(TAX_TOOLS)` คือหัวใจ — มันบอก LLM ว่า "คุณมี tool ชื่อ `calculate_thai_tax` ที่รับ parameter แบบนี้" LLM จะ **ตัดสินใจเอง** ว่าจะเรียก tool หรือตอบตรงๆ

**2. tools node** (ToolNode — เรียกฟังก์ชันจริง):
```python
graph.add_node("tools", ToolNode(TAX_TOOLS))
# ToolNode เป็น prebuilt component ของ LangGraph
# ถ้า LLM ตัดสินใจเรียก calculate_thai_tax
# ToolNode จะเรียกฟังก์ชันจริง แล้วส่งผลกลับ
```

**3. Conditional Edge** (เลือกเส้นทาง):
```python
def should_continue(state: TaxAgentState) -> str:
    """ถ้า LLM เรียก tool → ไป tools, ถ้าไม่ → จบ"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
```

**สร้าง Graph ทั้งหมด**:
```python
# src/finance_ai/agents/tax_agent.py

def build_tax_agent_graph(chat_model=None):
    """สร้าง LangGraph StateGraph สำหรับ Tax Agent"""

    graph = StateGraph(TaxAgentState)

    # เพิ่ม node
    graph.add_node("agent", create_llm_node(chat_model))
    graph.add_node("tools", ToolNode(TAX_TOOLS))

    # กำหนดจุดเริ่มต้น
    graph.set_entry_point("agent")

    # เพิ่ม edge
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")  # หลัง tool ทำเสร็จ → กลับไป agent

    return graph.compile()

# ใช้งาน:
graph = build_tax_agent_graph()
result = graph.invoke({"messages": [("user", "คำนวณภาษี เงินเดือน 1.2 ล้าน")]})
print(result["messages"][-1].content)  # คำตอบภาษาไทย
```

### 6.4 State Management — จัดการสถานะ

```python
# src/finance_ai/agents/schemas.py

class TaxAgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]  # ← สำคัญมาก!
    tax_result: dict[str, Any] | None
```

`Annotated[list[Any], add_messages]` คือ "annotation" ที่บอก LangGraph ว่า:
- `messages` เป็น list
- ใช้ `add_messages` reducer — เวลา node คืน `{"messages": [new_msg]}` จะ **เพิ่ม** ข้อความเข้า list ไม่ใช่แทนที่

เปรียบเสมือน chat history — ทุก node เพิ่มข้อความเข้า list ทำให้ node ถัดไปเห็นประวัติทั้งหมด

### 6.5 Tax Tools — @tool wrapper

LLM ส่งค่ามาเป็น string (เช่น `"1200000"` ไม่ใช่ `Decimal("1200000")`) ต้องแปลงก่อน:

```python
# src/finance_ai/agents/tax_tools.py

@tool
def calculate_thai_tax(
    gross_income: str,                     # LLM ส่งมาเป็น string
    deductions_by_type: dict[str, str],    # value เป็น string
    withholding_tax_paid: str = "0",
) -> dict[str, Any]:
    """คำนวณภาษีเงินได้บุคคลธรรมดาของไทย"""

    # แปลง string → Decimal
    income = parse_decimal_value(gross_income, "gross_income")
    deductions = parse_deductions_input(raw_deductions=deductions_by_type)
    withholding = parse_decimal_value(withholding_tax_paid, "withholding_tax_paid")

    # เรียก calculator จริง
    result = calculate_tax(income, deductions, withholding)

    # แปลงเป็น dict ที่ JSON-serializable (Decimal → string)
    return result.model_dump(mode="json")
```

**ทำไมรับ string ไม่รับ Decimal?** เพราะ LLM output เป็น text/JSON — ไม่มี concept ของ Decimal LLM จะส่ง `"1200000"` มาเป็น string เสมอ tool wrapper จึงต้องแปลงให้

**`@tool` decorator** คือ LangChain decorator ที่แปลง function ธรรมดาเป็น tool ที่ LLM เรียกได้ — มันอ่าน docstring, parameter names, และ type hints แล้วสร้าง schema ให้ LLM รู้วิธีเรียก

### 6.6 System Prompt — สั่ง AI ให้ทำงานถูกต้อง

```python
# src/finance_ai/agents/prompts.py

TAX_AGENT_SYSTEM_PROMPT = """
คุณเป็นผู้เชี่ยวชาญภาษีเงินได้บุคคลธรรมดาของไทย

หน้าที่:
1. สกัดข้อมูลภาษีจากคำถามผู้ใช้ (รายได้, ค่าลดหย่อน, ปีภาษี)
2. เรียกใช้เครื่องมือ calculate_thai_tax
3. อธิบายผลลัพธ์เป็นภาษาไทยที่เข้าใจง่าย

กฎสำคัญ:
- ถ้าผู้ใช้ไม่ระบุค่าลดหย่อนส่วนตัว ให้ใส่ personal_allowance = 60000 เสมอ
- ถ้าผู้ใช้ระบุจำนวนลูก ให้คำนวณ child_allowance = 30000 × จำนวนลูก
- ถ้าผู้ใช้ไม่ระบุปีภาษี ให้ใช้ปีปัจจุบัน
- ถ้าขาดข้อมูลสำคัญ (เช่น รายได้) ให้ถามกลับ

ประเภทค่าลดหย่อนที่รองรับ:
- personal_allowance: ค่าลดหย่อนส่วนตัว (60,000)
- spouse_allowance: คู่สมรส (60,000)
- child_allowance: บุตร (30,000 × จำนวน)
- rmf: กองทุน RMF (สูงสุด 30% ไม่เกิน 500,000)
- ssf: กองทุน SSF (สูงสุด 30% ไม่เกิน 200,000)
... (12 ประเภท)
"""
```

System Prompt เปรียบเสมือน "คู่มือพนักงาน" — บอก AI ว่ามีหน้าที่อะไร ต้องทำอะไรบ้าง มีกฎอะไร ถ้า prompt ไม่ดี AI จะตอบผิดหรือไม่ครบ

---

## 7. แนวคิดการพัฒนา

### 7.1 TDD — Test-Driven Development

**TDD** คือ "เขียน test ก่อน เขียนโค้ดทีหลัง":

```
1. เขียน test ที่ fail (Red)     → ระบุว่าต้องการอะไร
2. เขียนโค้ดให้ test ผ่าน (Green) → ทำให้มันทำงานได้
3. Refactor (Refactor)            → ปรับปรุงโค้ดให้สวย
```

**ตัวอย่างจริงจากโปรเจกต์**:

```python
# Step 1: เขียน test ก่อน (ยังไม่มี code จริง)
def test_calculate_tax_basic_income():
    """ทดสอบคำนวณภาษีรายได้พื้นฐาน"""
    result = calculate_tax(
        gross_income=Decimal("500000"),
        deductions_by_type={"personal_allowance": Decimal("60000")},
    )
    assert result.gross_income == Decimal("500000")
    assert result.total_tax > Decimal("0")
    assert result.effective_tax_rate > Decimal("0")

# Step 2: เขียน calculate_tax() ให้ test ผ่าน
# Step 3: Refactor ถ้าจำเป็น
```

**ทำไม TDD ดี?**
- **มั่นใจว่าโค้ดทำงานถูก** — ทุกฟังก์ชันถูกทดสอบ
- **กล้าแก้โค้ด** — ถ้าแก้แล้วพัง test จะบอกทันที
- **เป็นเอกสารอัตโนมัติ** — อ่าน test เข้าใจว่าฟังก์ชันทำอะไร

### 7.2 Type Safety — Decimal vs float

**ปัญหาของ float**:
```python
# ❌ อันตราย! float มีปัญหา floating point precision
>>> 0.1 + 0.2
0.30000000000000004  # ไม่ใช่ 0.3!

# ถ้าใช้ float คำนวณภาษี:
>>> tax = 1200000 * 0.1
>>> tax
120000.00000000001  # ผิด!
```

**วิธีแก้ — ใช้ Decimal**:
```python
# ✅ Decimal แม่นยำ 100%
>>> from decimal import Decimal
>>> Decimal("0.1") + Decimal("0.2")
Decimal('0.3')  # ถูกต้อง!

>>> Decimal("1200000") * Decimal("0.1")
Decimal('120000.0')  # ถูกต้อง!
```

**กฎ**: ทุกค่าที่เกี่ยวกับเงินในโปรเจกต์นี้ใช้ `Decimal` เท่านั้น — รายได้, ค่าลดหย่อน, ภาษี, อัตราภาษี

### 7.3 Clean Code Principles

โปรเจกต์นี้ยึดหลัก Clean Code:

**1. ฟังก์ชันทำอย่างเดียว (Single Responsibility)**:
```python
# ✅ แต่ละฟังก์ชันทำอย่างเดียว
validate_non_negative_income(gross_income)     # ตรวจสอบ
calculate_bracket_tax(net_income, ...)         # คำนวณ 1 ขั้น
calculate_progressive_tax(net_income)          # คำนวณทุกขั้น
cap_deduction_amount(type, amount, income)     # ตัด cap
calculate_effective_rate(tax, income)          # คำนวณอัตรา
```

**2. ชื่อสื่อความหมาย**:
```python
# ✅ อ่านแล้วรู้เลยว่าทำอะไร
calculate_total_withholding_tax()
build_auto_allowances()
merge_deductions()
store_tax_filing_result()

# ❌ ไม่ทำแบบนี้
calc_twt()
build_aa()
merge()
store()
```

**3. ฟังก์ชันไม่เกิน 20 บรรทัด** — ถ้ายาวกว่านี้ แยกเป็นฟังก์ชันย่อย

**4. Error messages ที่ชัดเจน**:
```python
# ✅ ดี — บอกว่าอะไรผิด และค่าที่ได้รับคืออะไร
raise ValueError(f"Income cannot be negative. Received: {gross_income} THB")
raise ValueError(f"Cannot parse gross_income as Decimal. Received: {value}")

# ❌ ไม่ดี
raise ValueError("Invalid input")
```

### 7.4 Testing Strategy

```
219 tests ครอบคลุม:
├── Core Layer:        26 tests (config, logging, LLM clients)
├── Database Layer:   ~80 tests (7 models × CRUD tests)
├── Tools Layer:      ~57 tests (constants, calculator, service)
└── Agent Layer:       56 tests (schemas, tools, factory, agent, router)

Coverage: 96.07% — โค้ดเกือบทุกบรรทัดถูกทดสอบ
```

**วิธีทดสอบ Agent โดยไม่เรียก API จริง** — Mock เฉพาะ LLM:

```python
# tests/agents/test_tax_agent.py

def test_full_react_loop(mock_chat_model):
    """ทดสอบ ReAct loop ทั้งหมด: LLM → tool → LLM → END"""

    # Mock ให้ LLM ตอบ 2 ครั้ง:
    # ครั้งที่ 1: เรียก tool (มี tool_calls)
    # ครั้งที่ 2: ตอบเป็นภาษาไทย (ไม่มี tool_calls → จบ)
    mock_chat_model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{
            "name": "calculate_thai_tax",
            "args": {"gross_income": "1200000", ...},
            "id": "call_1"
        }]),
        AIMessage(content="ภาษีที่ต้องจ่าย 89,000 บาท"),
    ]

    graph = build_tax_agent_graph(chat_model=mock_chat_model)
    result = graph.invoke({"messages": [("user", "คำนวณภาษี")]})

    # ตรวจสอบว่า tool ถูกเรียกจริง และได้ผลลัพธ์
    assert "89,000" in result["messages"][-1].content
```

**จุดสำคัญ**: Mock **เฉพาะ LLM** — Calculator, Parser, State Management ทำงานจริงทั้งหมด ทำให้มั่นใจว่าทุกส่วนทำงานร่วมกันได้

---

## 8. LangSmith Tracing — การ monitor AI

### Observability คืออะไร?

Observability คือ "ความสามารถในการดูว่าระบบข้างในทำอะไร" — เหมือนกล้องวงจรปิดของ AI

**ทำไมสำคัญ?** AI ไม่ได้ทำงานเหมือนโค้ดปกติ — มันตัดสินใจแบบ non-deterministic (ตอบไม่เหมือนกันทุกครั้ง) ถ้าไม่มี tracing จะไม่รู้ว่า:
- LLM ได้รับ prompt อะไร?
- ตัดสินใจเรียก tool ไหม?
- Tool คืนผลอะไร?
- LLM ตอบอะไรกลับไป?
- ใช้ token เท่าไหร่? (ค่าใช้จ่าย)

### LangSmith Integration

LangGraph มี LangSmith tracing ในตัว — แค่ตั้งค่า environment variables:

```bash
# ใน .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=finance-ai-tax-agent
```

เมื่อเปิดใช้งาน ทุก node execution จะถูกบันทึกใน LangSmith dashboard:

```
Trace Timeline:
─────────────────────────────────────────────────
│ agent (LLM)    │ 1.2s │ tokens: 850        │
│  ├── input: "คำนวณภาษี เงินเดือน 1.2 ล้าน"  │
│  └── output: tool_call → calculate_thai_tax  │
─────────────────────────────────────────────────
│ tools (Tool)   │ 0.1s │                      │
│  ├── input: {gross_income: "1200000", ...}   │
│  └── output: {total_tax: "89000", ...}       │
─────────────────────────────────────────────────
│ agent (LLM)    │ 0.8s │ tokens: 420        │
│  ├── input: tool_result                      │
│  └── output: "ภาษีที่ต้องจ่าย 89,000 บาท..." │
─────────────────────────────────────────────────
```

ดูได้แบบ real-time ที่ [smith.langchain.com](https://smith.langchain.com)

---

## 9. แผนพัฒนาต่อ

### สิ่งที่ทำเสร็จแล้ว

| Milestone | รายละเอียด | สถานะ |
|-----------|-----------|-------|
| 0.1 | โครงสร้างพื้นฐาน (Config, Logging, LLM Clients) | เสร็จ |
| 1.1 | Database Layer (Models, CRUD, Alembic) | เสร็จ |
| 1.2 | Tax Calculator (Pure Functions + DB Service) | เสร็จ |
| 1.3 | Tax Agent (LangGraph ReAct + Router Agent) | เสร็จ |

### สิ่งที่จะทำต่อ

| Milestone | รายละเอียด | สถานะ |
|-----------|-----------|-------|
| 2.x | **Investment Agent** — ติดตามพอร์ตลงทุน, ราคาหุ้น, กำไร/ขาดทุน | ยังไม่เริ่ม |
| 2.x | **Expense Agent** — ติดตามรายจ่าย, งบประมาณ | ยังไม่เริ่ม |
| 3.x | **RAG (Retrieval Augmented Generation)** — ค้นหากฎหมายภาษีจาก ChromaDB | ยังไม่เริ่ม |
| 4.x | **FastAPI** — REST API สำหรับ frontend | ยังไม่เริ่ม |
| 4.x | **Planning Agent** — วางแผนการเงินระยะยาว | ยังไม่เริ่ม |

### การเปรียบเทียบ LLM

เป้าหมายระยะยาวคือ **benchmark** ระหว่าง:
- **Google Gemini** — LLM cloud ที่รองรับภาษาไทย
- **THALLE** — LLM ภาษาไทยสำหรับการเงิน ที่รันบนเครื่อง (ผ่าน OLLAMA)

แต่ละ Agent อาจใช้ LLM ตัวที่แตกต่างกันได้ (เช่น Router ใช้ Gemini เพราะเร็ว, Tax Agent ใช้ THALLE เพราะเข้าใจภาษาไทยดีกว่า)

---

## สรุป — ทำไมถึงพัฒนาแบบนี้

| แนวคิด | ทำไม | ประยุกต์ใช้ตรงไหน |
|--------|------|-------------------|
| **Layered Architecture** | แยกส่วน ทดสอบง่าย ขยายง่าย | Core → DB → Tools → Agents |
| **Factory Pattern** | เปลี่ยน LLM/DB provider ได้ง่าย | LLM Factory, CRUD Factory |
| **Abstract Base Class** | กำหนด interface มาตรฐาน | BaseLLMClient, BaseCRUD |
| **Pure Functions** | ทดสอบง่าย ไม่มี side effect | Tax Calculator |
| **ReAct Pattern** | AI คิดก่อนทำ วนจนเสร็จ | Tax Agent (LangGraph) |
| **Tool Calling** | AI เรียกฟังก์ชันจริง ไม่เดาคำตอบ | calculate_thai_tax |
| **Pydantic Validation** | ตรวจข้อมูลอัตโนมัติ | Settings, TaxCalculationResult |
| **TDD** | มั่นใจว่าทุกส่วนทำงานถูก | 219 tests, 96% coverage |
| **Decimal** | คำนวณเงินแม่นยำ 100% | ทุก field ที่เป็นเงิน |
| **ORM + Migrations** | จัดการ DB ปลอดภัย | SQLAlchemy + Alembic |
| **Observability** | debug AI ได้ ดู trace ทีละ node | LangSmith Tracing |

ระบบนี้ไม่ได้ "ให้ AI เดาคำตอบ" — AI เป็นแค่ตัวกลางที่เข้าใจภาษาไทย ส่วนการคำนวณจริงใช้ฟังก์ชันคณิตศาสตร์ที่ถูกทดสอบแล้ว ทำให้ผลลัพธ์ **ถูกต้องตามกฎหมาย** ไม่ใช่การเดาของ AI
