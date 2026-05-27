# ✈️ United Airlines AI Assistant

An AI-powered customer support chatbot built with **RAG (Retrieval-Augmented Generation)** and **LangChain Agents** — demonstrating production-grade AI engineering patterns.

---

## 🏗️ Architecture

```
User Query
    │
    ├── /chat/rag ──► ChromaDB Retriever ──► GPT-4o-mini ──► Answer + Sources
    │                 (policy documents)      (LangChain RAG chain)
    │
    └── /chat/agent ─► LangChain Agent ──► Tool Selection ──► Answer
                       (GPT-4o-mini)         ├── check_flight_status()
                                             ├── calculate_baggage_fee()
                                             └── get_airport_info()
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| LLM Orchestration | LangChain |
| Vector Database | ChromaDB (FAISS-backed) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| Observability | LangSmith |
| Agent Pattern | OpenAI Tools Agent |

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/yourusername/flight-ai-assistant
cd flight-ai-assistant
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# 3. Run the server
uvicorn main:app --reload

# 4. Open browser
# Chat UI  →  http://localhost:8000
# API Docs →  http://localhost:8000/docs
```

## 📡 API Endpoints

### `POST /chat/rag`
Answers policy questions using RAG from embedded airline documents.
```json
Request:  { "message": "What is the baggage fee for Basic Economy?", "session_id": "user_1" }
Response: { "answer": "...", "sources": ["baggage_policy.pdf"], "latency_ms": 823.4 }
```

### `POST /chat/agent`
Uses tool-calling agent for dynamic queries (flight status, fee calculation).
```json
Request:  { "message": "Check status of flight UA456", "session_id": "user_1" }
Response: { "answer": "...", "tools_used": ["check_flight_status"], "latency_ms": 1243.1 }
```

### `GET /health`
```json
{ "status": "healthy", "rag_ready": true, "agent_ready": true, "langsmith_enabled": true }
```

## 🧠 Key Features

- **RAG Pipeline** — Documents chunked, embedded, stored in ChromaDB. Retriever fetches top-3 relevant chunks per query.
- **Tool-Calling Agent** — LangChain agent autonomously selects and calls tools based on user intent.
- **Multi-turn Memory** — Agent maintains per-session conversation history (last 5 turns).
- **LangSmith Observability** — Full trace visibility: token usage, latency, retrieval quality, tool calls.
- **FastAPI** — Auto-generated Swagger UI at `/docs`. CORS enabled for frontend integration.

## 📊 LangSmith Dashboard

Enable tracing to monitor:
- RAG retrieval quality (which chunks were fetched)
- Agent reasoning steps and tool call decisions
- Latency breakdown per chain step
- Token usage and cost tracking

Set `LANGCHAIN_API_KEY` in `.env` and visit [smith.langchain.com](https://smith.langchain.com).

## 📁 Project Structure

```
flight-ai-assistant/
├── main.py           # FastAPI app, endpoints, chat UI
├── rag_pipeline.py   # ChromaDB vector store + LangChain RAG chain
├── agent.py          # LangChain agent + tool definitions
├── requirements.txt
└── .env.example
```

## 🔮 Future Improvements

- [ ] Connect to real United Airlines API (FlightAware / Amadeus)
- [ ] Load actual PDF policy documents instead of mock text
- [ ] Add Redis for persistent session memory
- [ ] Deploy on AWS Lambda + API Gateway
- [ ] Add re-ranking (Cohere Rerank) for better RAG quality
- [ ] Streaming responses via Server-Sent Events

---

Built as a demonstration of AI engineering patterns — RAG, vector stores, LangChain agents, and FastAPI.
