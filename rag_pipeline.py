from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings
from groq import Groq
import os

class GroqEmbeddings(Embeddings):
    def __init__(self):
        self.client = Groq()
        self.model = "llama-3.1-8b-instant"

    def _embed(self, text: str) -> list[float]:
        import hashlib, math
        h = hashlib.md5(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h] * 24
        return vec[:384]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

AIRLINE_DOCS = [
    {"content": """Airline Baggage Policy:
- Carry-on: 1 free personal item + 1 carry-on for most fares.
- Checked bags: Basic Economy: 0 free. Economy+: first bag $35, second $45.
- MileagePlus Premier: Silver 2 free, Gold 3 free, Platinum/1K 3 free.
- Overweight (51-70 lbs): $100. Oversize (63-115 inches): $200.""",
     "source": "baggage_policy.pdf", "topic": "baggage"},

    {"content": """Airline Delay & Cancellation Policy:
- United cancels flight: full refund OR free rebooking.
- Delay 3+ hours domestic: request refund if you choose not to travel.
- Weather delays: United not liable for hotel/meals.
- Mechanical delays: meal vouchers $15-20, hotel if overnight.
- Rebook via united.com, app, or 1-800-864-8331.""",
     "source": "delay_policy.pdf", "topic": "delays"},

    {"content": """Airline Refund Policy:
- Refundable tickets: full refund in 7 business days.
- Non-refundable: eCredit valid 12 months.
- Basic Economy: no changes after 24-hour window.
- 24-hour risk-free cancellation: any ticket within 24hrs of purchase (7+ days before departure).
- Processing: 7 days credit cards, up to 20 days other methods.""",
     "source": "refund_policy.pdf", "topic": "refunds"},

    {"content": """Airline MileagePlus Program:
- Earn miles on flights, hotels, car rentals, credit cards.
- Economy ~5 miles/dollar, Business ~8, First ~11.
- Status tiers: Silver 25K PQP, Gold 50K, Platinum 75K, 1K 100K PQP.
- Miles expire after 18 months inactivity.
- Domestic economy awards start at 12,500 miles one-way.""",
     "source": "mileageplus.pdf", "topic": "loyalty"},

    {"content": """Airline Check-In Policy:
- Online check-in: opens 24hrs before, closes 45min before departure.
- International: closes 60min before departure.
- Mobile boarding pass available via United app.
- TSA PreCheck available if enrolled.""",
     "source": "checkin_policy.pdf", "topic": "checkin"},

    {"content": """Airline Pet Policy:
- In-cabin: small dogs/cats $125 each way domestic.
- Carrier: soft-sided, fits under seat (18x11x11 inches max).
- Max 2 pets per cabin per flight. Must be 8+ weeks old.
- Cargo pets: $200 fee.
- Service animals: free. ESAs no longer permitted.""",
     "source": "pet_policy.pdf", "topic": "pets"},
]


def build_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    embeddings = GroqEmbeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Loading existing ChromaDB vector store...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("Building vector store from documents...")
    documents = [
        Document(page_content=doc["content"], metadata={"source": doc["source"], "topic": doc["topic"]})
        for doc in AIRLINE_DOCS
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    print("Vector store built!")
    return vector_store


def build_rag_chain(vector_store: Chroma):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = ChatPromptTemplate.from_template("""You are a helpful airline customer support assistant.
Answer using ONLY the provided context. If not in context, say "I don't have that information. Please contact our support team directly."
Be concise and friendly.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(f"[{doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever