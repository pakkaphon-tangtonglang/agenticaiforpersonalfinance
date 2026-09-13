# Architecture & Sequence Diagrams

Diagrams for thesis chapter 3, generated from the actual code
(`src/finance_ai/`). Rendered as Mermaid — GitHub displays them natively.

## 1. System Architecture (Layered View)

```mermaid
flowchart TB
    subgraph clients["Clients"]
        LINE["LINE app<br/>(chatbot)"]
        WEB["Web frontend<br/>(HTML/JS served by FastAPI)"]
    end

    subgraph api["FastAPI Backend (Render)"]
        ROUTE["REST endpoints<br/>/chat, /dashboard, /upload,<br/>/assets/*, /line/webhook"]
        ORCH["orchestrate_query<br/>(LangGraph graph)"]
    end

    subgraph agents["Multi-Agent System (LangGraph 1.x)"]
        ROUTER["Router Agent<br/>(classify + confidence threshold<br/>+ symbol pre-search)"]
        TAX["Tax Agent"]
        EXP["Expense Agent"]
        ASSET["Asset Monitoring Agent"]
        PLAN["Planning Agent"]
        RECO["Recommendation Agent"]
        REPORT["Report Agent"]
    end

    subgraph services["Service / Tool Layer (src/finance_ai/tools)"]
        TAXC["tax_calculator<br/>(Decimal, Thai brackets)"]
        EXPC["expense_service"]
        MARKET["market_data_tools<br/>(yfinance)"]
        GOALC["planning_tools"]
        RAGT["rag_tool"]
        CONVS["conversation_service"]
    end

    subgraph storage["Data Layer"]
        DB[("SQLite / PostgreSQL<br/>SQLAlchemy ORM")]
        CHROMA[("ChromaDB<br/>23 Thai finance docs")]
    end

    subgraph external["External APIs"]
        LINEAPI["LINE Messaging API<br/>(webhook + push)"]
        LLM["LLM providers<br/>(Gemini / Ollama / OpenRouter)"]
    end

    LINE <--> LINEAPI
    LINEAPI --> ROUTE
    WEB <--> ROUTE
    ROUTE --> ORCH
    ORCH --> ROUTER
    ROUTER --> TAX & EXP & ASSET & PLAN & RECO & REPORT
    TAX --> TAXC
    EXP --> EXPC
    ASSET --> MARKET
    PLAN --> GOALC
    TAX & EXP & RECO & REPORT --> RAGT
    ORCH --> CONVS
    services --> DB
    RAGT --> CHROMA
    ORCH --> LLM
```

## 2. Chat Query Sequence (Web /chat)

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI /chat
    participant Conv as conversation_service
    participant Router as Router Agent
    participant Agent as Specialist Agent
    participant Tools as Tool Layer
    participant DB as SQLite

    User->>API: POST /chat {query, user_id, conversation_id}
    API->>Conv: load recent history
    Conv->>DB: SELECT messages
    API->>Router: orchestrate_query(query, chat_history)
    Router->>Router: symbol pre-search + LLM classify
    alt confidence >= threshold
        Router->>Agent: route to specialist
        Agent->>Tools: call domain tools (InjectedState: user_id, db)
        Tools->>DB: read/write user data
        Tools-->>Agent: structured results
    else confidence < threshold
        Router-->>API: clarify question (Thai menu)
    end
    Agent-->>API: {intent, response}
    API->>Conv: save user + assistant messages
    API-->>User: JSON response (or SSE stream)
```

## 3. LINE Chatbot Sequence (Push Model)

The webhook answers within LINE's ~1 second window; the agent runs as a
FastAPI background task because slow models (10–30 s on Ollama) outlive
reply tokens (~30 s). Answers are delivered via the Push Message API.

```mermaid
sequenceDiagram
    actor User
    participant LINE as LINE app
    participant MA as LINE Messaging API
    participant WH as POST /line/webhook
    participant BG as BackgroundTask
    participant Map as mapping_service
    participant Orch as orchestrate_query
    participant Push as Push Message API

    User->>LINE: ส่งข้อความ "ภาษีของฉัน"
    LINE->>MA: message event
    MA->>WH: POST (X-Line-Signature = HMAC-SHA256)
    WH->>WH: verify signature (channel secret)
    WH-->>MA: 200 OK (< 1 s)
    WH->>BG: handle_line_event(...)
    BG->>Map: get_or_create_line_mapping(line_user_id)
    Map->>Map: auto-create user + conversation on first message
    BG->>Orch: query + chat history + user_id
    Orch-->>BG: Thai response
    BG->>Push: push answer + Quick Reply menu
    Push->>LINE: message delivered
    LINE-->>User: คำตอบจากเอเจนต์
```

## 4. LangGraph Agent Graph

```mermaid
flowchart LR
    START((START)) --> ROUTER["Router Agent"]
    ROUTER -->|tax| TAX["Tax Agent"]
    ROUTER -->|expense| EXP["Expense Agent"]
    ROUTER -->|asset| ASSET["Asset Monitoring Agent"]
    ROUTER -->|planning| PLAN["Planning Agent"]
    ROUTER -->|recommendation| RECO["Recommendation Agent"]
    ROUTER -->|report| REPORT["Report Agent"]
    ROUTER -->|unknown / low confidence| CLARIFY["Clarify question"]
    TAX & EXP & ASSET & PLAN & RECO & REPORT & CLARIFY --> END((END))
```

## 5. RAG Retrieval Flow

```mermaid
flowchart LR
    DOC["docs/knowledge_base/<br/>(23 Thai finance docs)"] -->|chunk + embed| CHROMA[("ChromaDB<br/>vector store")]
    Q["User question"] -->|embed| RET["retriever<br/>(top-k similarity)"]
    CHROMA --> RET
    RET --> CTX["cited context<br/>(source + passage)"]
    CTX --> AGENT["Specialist agent prompt"]
```

## 6. Entity-Relationship Diagram

```mermaid
erDiagram
    users ||--o{ incomes : "earns"
    users ||--o{ deductions : "claims"
    users ||--o{ transactions : "spends"
    users ||--o{ financial_goals : "sets"
    users ||--o{ watched_assets : "watches"
    users ||--o{ investment_holdings : "holds"
    users ||--o{ tax_filings : "files"
    users ||--o{ risk_assessments : "assessed"
    users ||--o{ conversations : "chats"
    users ||--o| line_user_mappings : "linked from LINE"
    conversations ||--o{ conversation_messages : "contains"
    line_user_mappings }o--|| conversations : "continues"
    asset_schedules ||--o{ asset_notifications : "triggers"

    users {
        string id PK
        string email UK
        string full_name
        datetime created_at
    }
    line_user_mappings {
        string id PK
        string line_user_id UK "LINE platform userId"
        string user_id FK
        string conversation_id FK
    }
    incomes {
        string id PK
        string user_id FK
        decimal amount
        string income_type
    }
    deductions {
        string id PK
        string user_id FK
        string deduction_type "rmf/ssf/social_security/..."
        decimal amount
    }
    transactions {
        string id PK
        string user_id FK
        decimal amount
        string category
        date occurred_at
    }
    financial_goals {
        string id PK
        string user_id FK
        decimal target_amount
        date deadline
    }
    watched_assets {
        string id PK
        string user_id FK
        string symbol "SYMBOL.BK"
    }
    investment_holdings {
        string id PK
        string user_id FK
        string symbol
        decimal quantity
        decimal average_cost
    }
    tax_filings {
        string id PK
        string user_id FK
        int year
        decimal tax_owed
    }
    risk_assessments {
        string id PK
        string user_id FK
        int total_score
        string risk_category
    }
    conversations {
        string id PK
        string user_id FK
        string title
        bool is_active
    }
    conversation_messages {
        string id PK
        string conversation_id FK
        string role "user/assistant"
        string content
        string intent
    }
    asset_schedules {
        string id PK
        string user_id FK
        string symbol
        string frequency
    }
    asset_notifications {
        string id PK
        string user_id FK
        string message
        bool is_read
    }
```

## 7. Deployment View

```mermaid
flowchart LR
    subgraph render["Render (live deployment)"]
        SVC["FastAPI service<br/>+ static frontend"]
        SCHED["APScheduler<br/>(asset price refresh)"]
        SVC --- SCHED
    end
    subgraph env["Environment variables (secrets, never committed)"]
        SECRET["LINE_CHANNEL_SECRET"]
        TOKEN["LINE_CHANNEL_ACCESS_TOKEN"]
        KEY["GOOGLE_API_KEY / LLM keys"]
    end
    LINE["LINE servers"] -->|webhook push| SVC
    SVC -->|push replies| LINE
    YF["Yahoo Finance"] -.->|blocked on Render IP,<br/>documented limitation| SVC
    env -.-> SVC
```
