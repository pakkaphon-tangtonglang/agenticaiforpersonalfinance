# Personal Finance AI

> Multi-Agent AI system for personal finance management designed for Thai users

[![CI](https://img.shields.io/github/actions/workflow/status/yourusername/personal-finance-ai/ci.yml?branch=main)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## What is This?

An intelligent personal finance assistant that helps you:

- **Optimize Taxes** - Calculate Thai personal income tax with all deductions
- **Track Expenses** - Record and categorize spending, get monthly summaries
- **Monitor Assets** - Watch stock prices, search finance news, manage watchlists
- **Plan Ahead** - Set financial goals, calculate saving plans, track progress
- **Get Recommendations** - Receive proactive, personalized financial advice
- **Generate Reports** - Full financial reports with health scores and visualizations

## Key Features

### Multi-Agent System (LangGraph)
Powered by LangGraph with a Router + 6 specialized agents:

- **Router Agent** - Classifies user query and routes to the right specialist
- **Tax Agent** - Thai personal income tax calculations with all deductions
- **Expense Agent** - Income/expense tracking, monthly summaries, category queries
- **Asset Monitoring Agent** - Stock prices, finance news, watchlist management
- **Planning Agent** - Financial goals, saving plans, psychological cue detection
- **Recommendation Agent** - Proactive financial recommendations and health scoring
- **Report Agent** - Comprehensive financial reports with PDF/CSV export

### MCP Tool Layer
Agent tools are exposed as MCP (Model Context Protocol) servers:

- **knowledge** - RAG search over Thai finance knowledge base (ChromaDB)
- **market_data** - Stock prices and finance news (Bright Data / yfinance)
- **tax_calc** - Stateless tax and saving plan calculators
- **finance_db** - User-scoped database operations (expenses, goals, reports)

### RAG Knowledge Base
ChromaDB vector store with 16+ Thai finance documents covering:
- Personal income tax, deductions, filing guides, VAT/withholding
- Thai stocks, mutual funds, ETFs, bonds, DCA strategy
- Budgeting, emergency funds, debt management
- Life/health insurance, social security
- Retirement and financial planning

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Poetry (for dependency management)
- Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))
  - Or use Ollama / OpenRouter as alternative LLM providers

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/personal-finance-ai.git
cd personal-finance-ai

# Install dependencies
make install

# Copy environment variables template
cp .env.example .env

# Edit .env and add your API keys
nano .env

# Run tests to verify setup
make test
```

### Running the Application

```bash
# Start the FastAPI backend (development mode with auto-reload)
make dev

# Or run production server
make run
```

The API will be available at `http://localhost:8000`
Interactive API docs at `http://localhost:8000/docs`

MCP servers are mounted at `/mcp/*` endpoints for external MCP clients.

## Usage Examples

### Calculate Taxes

```python
from finance_ai.agents.llm_factory import create_chat_model
from finance_ai.agents.router_agent import orchestrate_query

result = orchestrate_query(
    "คำนวณภาษี เงินเดือน 1,200,000 บาท มีบุตร 2 คน ซื้อ RMF 100,000",
    chat_model=create_chat_model(),
    user_id="your-user-id",
)

print(result["intent"])    # "tax"
print(result["response"])  # Thai-language tax breakdown
```

### Chat with Streaming

```python
from finance_ai.agents.stream_utils import orchestrate_query_stream

for event in orchestrate_query_stream(
    "สรุปค่าใช้จ่ายเดือนนี้",
    chat_model=create_chat_model(),
    user_id="your-user-id",
):
    if event.event_type == "token":
        print(event.content, end="", flush=True)
```

## Project Structure

```
personal-finance-ai/
├── src/finance_ai/
│   ├── agents/              # LangGraph agent implementations
│   │   ├── router_agent.py       # Intent classification + dispatch
│   │   ├── tax_agent.py          # Tax Agent (ReAct graph)
│   │   ├── expense_agent.py      # Expense Agent
│   │   ├── asset_monitoring_agent.py
│   │   ├── planning_agent.py
│   │   ├── recommendation_agent.py
│   │   ├── report_agent.py
│   │   ├── llm_factory.py        # LLM provider factory
│   │   ├── stream_utils.py       # Streaming agent responses
│   │   └── prompts.py            # System prompts
│   ├── mcp/                # MCP server implementations
│   │   ├── knowledge_server.py   # RAG search server
│   │   ├── market_data_server.py # Stock price + news server
│   │   ├── tax_calc_server.py    # Stateless calculators
│   │   └── finance_db_server.py  # User-scoped DB operations
│   ├── tools/              # Service layer (business logic)
│   ├── rag/                # ChromaDB vector store + embeddings
│   ├── database/           # SQLAlchemy models + CRUD + Alembic
│   ├── evaluation/         # Evaluation framework (6 dimensions)
│   ├── core/               # Config, logging, LLM clients
│   ├── ui/                 # Charts, export, constants (non-Streamlit)
│   └── main.py             # FastAPI app entry point
├── tests/                  # Test suite (mirrors src/)
├── alembic/                # Database migrations
├── docs/
│   └── knowledge_base/     # RAG source documents (Thai finance)
├── scripts/                # Utility scripts (eval, PDF generation)
├── .github/workflows/      # CI/CD pipeline
├── pyproject.toml          # Dependencies and tool configs
├── Makefile                # Common commands
└── README.md
```

## Development

### Setup Development Environment

```bash
# Install pre-commit hooks
pre-commit install

# Run all quality checks
make check

# Run specific checks
make format      # Auto-format code
make lint        # Run linter
make typecheck   # Check type hints
make test        # Run tests
```

### Code Quality Standards

This project follows strict code quality standards:

- Type hints on all functions
- Test coverage >90%
- Linting score >9.0/10
- Functions <20 lines
- Comprehensive docstrings

See [CLAUDE.md](./CLAUDE.md) for full coding standards.

### Available Make Commands

```bash
make install     # Install dependencies
make dev         # Run development server (auto-reload)
make run         # Run production server
make test        # Run tests with coverage
make coverage    # Generate HTML coverage report
make lint        # Check code quality
make format      # Format code with black
make typecheck   # Type checking with mypy
make check       # Run all checks (format, lint, typecheck, test)
make clean       # Clean generated files
make evaluate    # Run evaluation framework
```

## LLM Providers

The system supports multiple LLM providers (configured in `.env`):

| Provider | Config key | Default model |
|---|---|---|
| Google Gemini | `llm_provider=google` | `gemini-2.5-flash` |
| Ollama (local) | `llm_provider=ollama` | `THALLE` |
| OpenRouter | `llm_provider=openrouter` | `deepseek/deepseek-chat-v3-0324` |

## Roadmap

### Completed
- [x] Project foundation, dev tools, CI/CD config
- [x] Database models + Alembic migrations
- [x] Tax calculation engine (all Thai deductions + brackets)
- [x] RAG knowledge base (ChromaDB + 16 Thai finance docs)
- [x] Multi-agent system (Router + 6 specialized agents)
- [x] Cross-agent collaboration tools
- [x] Conversation history + memory (DB persistence)
- [x] Evaluation framework (routing, RAG, accuracy, hallucination, performance)
- [x] Dashboard, file upload, PDF/CSV export
- [x] Asset monitoring with scheduled fetching + notifications
- [x] Proactive recommendations + financial health scoring

### In Progress
- [ ] MCP tool layer (agents consume tools via MCP servers)
- [ ] FastAPI backend (replacing Streamlit)
- [ ] Dependency upgrades (LangGraph 1.x, LangChain 1.x, google-genai 2.x)

### Future
- [ ] Web frontend (Next.js or similar)
- [ ] Mobile app
- [ ] Real-time market data streaming
- [ ] Advanced tax strategies (scenario modeling)
- [ ] Broker statement auto-import

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

All PRs must:
- Pass all tests (`make check`)
- Have >90% coverage
- Pass linting (score >9.0)
- Pass type checking

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by [Google Gemini](https://ai.google.dev/) for LLM capabilities
- Tools exposed via [MCP](https://modelcontextprotocol.io/) (Model Context Protocol)
- RAG with [ChromaDB](https://www.trychroma.com/)
- Market data from [Bright Data](https://brightdata.com/) and yfinance
- Tax data from [Thai Revenue Department](https://www.rd.go.th/)
- Market data from [SET](https://www.set.or.th/) and [AIMC](https://www.aimc.or.th/)

---

**Disclaimer**: This tool is for informational purposes only and does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions. Tax calculations are based on current Thai tax laws and may not reflect the latest changes.
