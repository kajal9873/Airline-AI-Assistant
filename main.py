"""
Airline AI Assistant — FastAPI Backend
AI-powered customer support chatbot using RAG + LangChain Agents.
Works for any airline — update policy docs in rag_pipeline.py to customize.

Features:
  - /chat/rag    → RAG pipeline (policy questions)
  - /chat/agent  → LangChain Agent with tool calling
  - /health      → Health check
  - /docs        → Auto-generated Swagger UI (FastAPI built-in)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import time

from rag_pipeline import build_vector_store, build_rag_chain
from agent import build_agent

load_dotenv()

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Airline AI Assistant",
    description="""
    AI-powered customer support chatbot using RAG + LangChain Agents.
    
    **Two endpoints:**
    - `/chat/rag` — Answers policy questions from embedded airline documents (ChromaDB)
    - `/chat/agent` — Agent with tool calling for flight status, baggage fees, airport info
    
    Built with: FastAPI · LangChain · LangGraph · ChromaDB · Groq · LangSmith
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "What is the baggage policy for Basic Economy?", "session_id": "user_001"},
                {"message": "Check status of flight UA456", "session_id": "user_002"},
            ]
        }
    }

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    tools_used: list[str] = []
    latency_ms: float
    session_id: str

class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    agent_ready: bool
    langsmith_enabled: bool

# ── Startup: load models once ─────────────────────────────────────────────
rag_chain = None
retriever = None
agent_executor = None

@app.on_event("startup")
async def startup_event():
    global rag_chain, retriever, agent_executor

    if not os.getenv("GROQ_API_KEY"):
       print("WARNING: GROQ_API_KEY not set — set it in .env file")
       return

    print("Initializing RAG pipeline...")
    vector_store = build_vector_store()
    rag_chain, retriever = build_rag_chain(vector_store)
    print("RAG pipeline ready!")

    print("Initializing Agent...")
    agent_executor = build_agent()
    print("Agent ready!")
    print("\nServer is up! Visit http://localhost:8000/docs for Swagger UI\n")


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if the server and all components are running."""
    return HealthResponse(
        status="healthy",
        rag_ready=rag_chain is not None,
        agent_ready=agent_executor is not None,
        langsmith_enabled=os.getenv("LANGCHAIN_TRACING_V2") == "true",
    )


@app.post("/chat/rag", response_model=ChatResponse)
async def chat_rag(request: ChatRequest):
    """
    RAG endpoint — answers questions from indexed airline policy documents.
    Best for: baggage policy, refunds, delays, MileagePlus, check-in rules.
    """
    if not rag_chain:
        raise HTTPException(503, "RAG pipeline not ready. Check OPENAI_API_KEY in .env")

    start = time.time()
    try:
        answer = rag_chain.invoke(request.message)

        # Retrieve sources separately for the response
        docs = retriever.invoke(request.message)
        sources = list({doc.metadata.get("source", "N/A") for doc in docs})

        return ChatResponse(
            answer=answer,
            sources=sources,
            tools_used=["ChromaDB retriever", "Groq LLM (llama-3.1-8b)"],
            latency_ms=round((time.time() - start) * 1000, 2),
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(500, f"RAG error: {str(e)}")


# In-memory chat history store (per session)
chat_histories: dict[str, list] = {}

@app.post("/chat/agent", response_model=ChatResponse)
async def chat_agent(request: ChatRequest):
    """
    Agent endpoint — uses tool calling to answer dynamic queries.
    Best for: flight status, baggage fee calculation, airport info.
    The agent autonomously decides which tools to call.
    """
    if not agent_executor:
        raise HTTPException(503, "Agent not ready. Check OPENAI_API_KEY in .env")

    start = time.time()

    # Maintain per-session chat history
    history = chat_histories.get(request.session_id, [])

    try:
        messages = history + [HumanMessage(content=request.message)]
        result = agent_executor.invoke({"messages": messages})

        all_messages = result["messages"]
        answer = all_messages[-1].content
   
        tools_used = [
            msg.name for msg in all_messages
            if hasattr(msg, "name") and msg.name
        ] 

        # Update chat history
        history.append(HumanMessage(content=request.message))
        history.append(AIMessage(content=answer))
        chat_histories[request.session_id] = history[-10:]  # keep last 5 turns

        return ChatResponse(
            answer=answer,
            sources=[],
            tools_used=tools_used if tools_used else ["GPT-4o-mini (no tools needed)"],
            latency_ms=round((time.time() - start) * 1000, 2),
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")


# ── Simple chat UI (bonus!) ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Simple browser-based chat UI to demo the assistant."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Airline AI Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'DM Sans', sans-serif; background: #f0f4f8; display: flex; flex-direction: column; height: 100vh; }
  header { background: #003087; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; }
  header span { font-size: 12px; opacity: 0.7; background: rgba(255,255,255,0.15); padding: 3px 10px; border-radius: 20px; }
  .mode-toggle { display: flex; gap: 8px; padding: 12px 24px; background: white; border-bottom: 1px solid #e2e8f0; }
  .mode-btn { padding: 6px 16px; border-radius: 20px; border: 1.5px solid #003087; font-size: 13px; cursor: pointer; font-family: inherit; font-weight: 500; transition: all 0.2s; }
  .mode-btn.active { background: #003087; color: white; }
  .mode-btn:not(.active) { background: white; color: #003087; }
  .chat-area { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 72%; padding: 12px 16px; border-radius: 16px; font-size: 14px; line-height: 1.6; }
  .msg.user { align-self: flex-end; background: #003087; color: white; border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: white; color: #1a202c; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .msg .meta { font-size: 11px; margin-top: 6px; opacity: 0.6; }
  .msg.bot .meta { color: #718096; }
  .msg.user .meta { color: rgba(255,255,255,0.7); }
  .input-area { padding: 16px 24px; background: white; border-top: 1px solid #e2e8f0; display: flex; gap: 10px; }
  .input-area input { flex: 1; padding: 12px 16px; border: 1.5px solid #e2e8f0; border-radius: 24px; font-size: 14px; font-family: inherit; outline: none; transition: border 0.2s; }
  .input-area input:focus { border-color: #003087; }
  .input-area button { padding: 12px 22px; background: #003087; color: white; border: none; border-radius: 24px; font-size: 14px; font-weight: 600; cursor: pointer; font-family: inherit; }
  .input-area button:hover { background: #00256e; }
  .suggestions { display: flex; gap: 8px; flex-wrap: wrap; padding: 0 24px 12px; }
  .sug { padding: 6px 14px; background: white; border: 1px solid #e2e8f0; border-radius: 20px; font-size: 12px; cursor: pointer; color: #003087; font-weight: 500; }
  .sug:hover { background: #f0f4f8; }
  .typing { color: #718096; font-size: 13px; padding: 8px 16px; background: white; border-radius: 16px; align-self: flex-start; }
</style>
</head>
<body>
<header>
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
  <h1>Airline AI Assistant</h1>
  <span>Powered by RAG + LangChain</span>
</header>

<div class="mode-toggle">
  <button class="mode-btn active" id="rag-btn" onclick="setMode('rag')">Policy Q&A (RAG)</button>
  <button class="mode-btn" id="agent-btn" onclick="setMode('agent')">Smart Agent (Tool Calling)</button>
</div>

<div class="suggestions">
  <span class="sug" onclick="ask(this.textContent)">What's the baggage fee for Basic Economy?</span>
  <span class="sug" onclick="ask(this.textContent)">Can I get a refund for a cancelled flight?</span>
  <span class="sug" onclick="ask(this.textContent)">Check status of flight UA789</span>
  <span class="sug" onclick="ask(this.textContent)">How much for 2 bags, 55 lbs each?</span>
  <span class="sug" onclick="ask(this.textContent)">Tell me about ORD airport</span>
</div>

<div class="chat-area" id="chat">
  <div class="msg bot">
    Hi! I'm your Airline AI Assistant. Ask me about baggage, refunds, flight status, loyalty rewards, or any travel policy.
    <div class="meta">RAG mode — answers from policy docs</div>
  </div>
</div>

<div class="input-area">
  <input id="inp" placeholder="Ask about baggage, refunds, flights..." onkeydown="if(event.key==='Enter') send()">
  <button onclick="send()">Send</button>
</div>

<script>
let mode = 'rag';
function setMode(m) {
  mode = m;
  document.getElementById('rag-btn').className = 'mode-btn' + (m==='rag'?' active':'');
  document.getElementById('agent-btn').className = 'mode-btn' + (m==='agent'?' active':'');
}
function ask(q) { document.getElementById('inp').value = q; send(); }
async function send() {
  const inp = document.getElementById('inp');
  const q = inp.value.trim();
  if (!q) return;
  inp.value = '';
  const chat = document.getElementById('chat');
  chat.innerHTML += `<div class="msg user">${q}</div>`;
  chat.innerHTML += `<div class="msg bot typing" id="typing">Thinking...</div>`;
  chat.scrollTop = chat.scrollHeight;
  try {
    const res = await fetch('/chat/' + mode, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: q, session_id: 'web_user'})
    });
    const data = await res.json();
    document.getElementById('typing').remove();
    const tools = data.tools_used.length ? `Tools: ${data.tools_used.join(', ')} · ` : '';
    const sources = data.sources.length ? `Sources: ${data.sources.join(', ')} · ` : '';
    chat.innerHTML += `<div class="msg bot">${data.answer}<div class="meta">${sources}${tools}${data.latency_ms}ms</div></div>`;
  } catch(e) {
    document.getElementById('typing').remove();
    chat.innerHTML += `<div class="msg bot">Error: ${e.message}</div>`;
  }
  chat.scrollTop = chat.scrollHeight;
}
</script>
</body>
</html>
"""
