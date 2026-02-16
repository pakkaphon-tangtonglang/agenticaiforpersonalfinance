# 🤖 Personal Finance AI

> Multi-Agent AI system for personal finance management designed for Thai users

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](./htmlcov)
[![Code Quality](https://img.shields.io/badge/code%20quality-9.5%2F10-brightgreen.svg)](./pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## 🎯 What is This?

An intelligent personal finance assistant that helps you:

- 💰 **Optimize Taxes** - Calculate Thai personal income tax with all deductions
- 📊 **Track Investments** - Monitor stocks, mutual funds, and portfolio performance
- 💡 **Get AI Recommendations** - Receive personalized financial advice
- 📈 **Plan Ahead** - Set goals and track progress

## ✨ Key Features

### 🧮 Tax Agent
- Accurate Thai tax calculations (2024 tax year)
- All standard deductions supported (RMF, SSF, insurance, etc.)
- Tax optimization suggestions
- Multi-year comparisons

### 📈 Investment Agent
- Stock portfolio tracking (SET/MAI)
- Mutual fund monitoring
- Real-time gain/loss calculations
- Portfolio allocation analysis

### 🤖 Multi-Agent System
Powered by LangGraph with specialized agents:
- **Router Agent** - Understands your query and routes to the right specialist
- **Tax Agent** - Expert in Thai tax laws and calculations
- **Investment Agent** - Tracks and analyzes your portfolio
- **Planning Agent** - Long-term financial planning

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Poetry (for dependency management)
- Anthropic API key ([Get one here](https://console.anthropic.com/))

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
# Start the API server
make run

# Or run in development mode with auto-reload
make dev
```

The API will be available at `http://localhost:8000`

## 📖 Usage Examples

### Calculate Taxes

```python
from finance_ai.agents import TaxAgent

agent = TaxAgent()

result = agent.calculate_tax(
    gross_income=1_200_000,  # 1.2M THB/year
    deductions={
        "personal": 60_000,
        "spouse": 60_000,
        "children": 2,  # 2 kids = 60,000 deduction
        "rmf": 100_000,
        "social_security": 9_000,
    }
)

print(f"Total Tax: {result.total_tax:,.2f} THB")
print(f"Effective Rate: {result.effective_rate:.2f}%")
```

### Track Portfolio

```python
from finance_ai.agents import InvestmentAgent

agent = InvestmentAgent()

portfolio = agent.get_portfolio_summary(user_id=1)

print(f"Total Value: {portfolio.total_value:,.2f} THB")
print(f"Total Gain/Loss: {portfolio.total_gain_loss:,.2f} THB ({portfolio.total_gain_loss_pct:.2f}%)")
```

## 🏗️ Project Structure

```
personal-finance-ai/
├── src/
│   └── finance_ai/
│       ├── agents/          # LangGraph agent implementations
│       │   ├── router.py
│       │   ├── tax_agent.py
│       │   └── investment_agent.py
│       ├── tools/           # Tools used by agents
│       │   ├── tax_calculator.py
│       │   └── portfolio_tracker.py
│       ├── rag/             # RAG system for knowledge retrieval
│       │   ├── vectorstore.py
│       │   └── embeddings.py
│       ├── database/        # Database models and CRUD
│       │   ├── models.py
│       │   └── crud.py
│       └── core/            # Core utilities
│           ├── config.py
│           ├── logging.py
│           └── llm/
├── tests/                   # Test suite (mirrors src/)
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── .github/workflows/       # CI/CD pipelines
├── pyproject.toml          # Dependencies and tool configs
├── Makefile                # Common commands
└── README.md
```

## 🛠️ Development

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

- ✅ Type hints on all functions
- ✅ Test coverage >90%
- ✅ Linting score >9.0/10
- ✅ Functions <20 lines
- ✅ Comprehensive docstrings

See [CLAUDE.md](./CLAUDE.md) for full coding standards.

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make coverage

# Run specific test file
pytest tests/agents/test_tax_agent.py

# Run tests in watch mode
pytest-watch
```

### Available Make Commands

```bash
make install     # Install dependencies
make dev         # Run in development mode
make run         # Run production server
make test        # Run tests
make coverage    # Generate coverage report
make lint        # Check code quality
make format      # Auto-format code
make typecheck   # Type checking
make check       # Run all checks (format, lint, typecheck, test)
make clean       # Clean generated files
```

## 📚 Documentation

- [Development Milestones](./development_milestones.md) - Project roadmap and implementation plan
- [Architecture Design](./docs/architecture.md) - System architecture and design decisions
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
- [Code Standards](./CLAUDE.md) - Coding guidelines and best practices

## 🧪 Testing

We maintain >90% test coverage with comprehensive test suites:

```bash
# Run all tests with coverage
make coverage

# View coverage report
open htmlcov/index.html
```

Test types:
- **Unit Tests** - Individual function testing
- **Integration Tests** - Component interaction testing
- **End-to-End Tests** - Full workflow testing
- **Property-Based Tests** - Edge case discovery with Hypothesis

## 🔒 Security

- 🔐 Database encrypted at rest (SQLCipher)
- 🔒 HTTPS only in production
- 🛡️ Input validation on all endpoints
- 🚫 No API keys stored in database
- ⚠️ Rate limiting enabled
- 📝 Audit logging for sensitive operations

## 🗺️ Roadmap

### ✅ Milestone 0: Foundation (Complete)
- [x] Project structure
- [x] Development tools setup
- [x] CI/CD pipeline

### 🔄 Milestone 1: Core Features (In Progress)
- [x] Database models
- [x] Tax calculation engine
- [ ] RAG system for tax laws
- [ ] Basic API endpoints

### 📅 Milestone 2: Multi-Agent System (Next)
- [ ] LangGraph orchestration
- [ ] Investment tracking
- [ ] Portfolio analysis
- [ ] Web interface

### 🔮 Future
- [ ] Mobile app
- [ ] Real-time market data
- [ ] Advanced tax strategies
- [ ] Expense tracking
- [ ] Budgeting tools

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](./CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

All PRs must:
- ✅ Pass all tests
- ✅ Have >90% coverage
- ✅ Pass linting (score >9.0)
- ✅ Pass type checking
- ✅ Include documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by [Anthropic Claude](https://www.anthropic.com/) for LLM capabilities
- Tax data from [Thai Revenue Department](https://www.rd.go.th/)
- Market data from [SET](https://www.set.or.th/) and [AIMC](https://www.aimc.or.th/)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/personal-finance-ai/issues)
- **Email**: your.email@example.com
- **Discord**: [Join our community](https://discord.gg/yourinvite)

---

**⚠️ Disclaimer**: This tool is for informational purposes only and does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions. Tax calculations are based on current Thai tax laws and may not reflect the latest changes.
