"""Central configuration for SonaRAG.

Every tunable knob lives here. Other modules `import config` and read these
values, so you change behaviour in ONE place instead of hunting through code.
This is a small habit that separates hobby scripts from maintainable projects.
"""
from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"     # source documents live here
VECTOR_DIR = PROJECT_ROOT / "vector_store"     # Chroma will persist here (Phase 2)
RATES_PATH = PROJECT_ROOT / "rates.json"       # daily gold/silver rates (cache)
RATES_CURRENCY = "INR"                          # currency for GoldAPI live rates

# --- Identity ---------------------------------------------------------------
APP_NAME = "Aurum"
APP_TAGLINE = "Gold & Silver Intelligence"

# --- Chunking (Phase 1) -----------------------------------------------------
# Size is measured in CHARACTERS. Rough rule of thumb: 1 token ~= 4 characters
# of English, so CHUNK_SIZE = 800 chars ~= 200 tokens. We'll verify real token
# counts later with Claude's count_tokens endpoint (never tiktoken — that's
# OpenAI's tokenizer and it miscounts for Claude).
CHUNK_SIZE = 800        # target characters per chunk
CHUNK_OVERLAP = 120     # characters shared between neighbouring chunks

# --- Retrieval (Phase 3) ----------------------------------------------------
TOP_K = 5               # how many candidate chunks to pull from the vector DB
SIMILARITY_THRESHOLD = 0.35   # legacy strict gate (used by retriever.py demo)

# Aurum is an EXPERT, not a strict refuser:
#   * top similarity below OFF_DOMAIN_FLOOR  -> clearly not about metals -> refuse
#   * passages at/above GROUNDING_MIN_SIM    -> trusted enough to cite as sources
# Re-measured after expanding the knowledge base: off-domain tops out ~0.20
# (weather/IPL drift up as the gold corpus grows), in-domain floors ~0.48.
# 0.30 sits cleanly between them, with margin on both sides.
GROUNDING_MIN_SIM = 0.30
OFF_DOMAIN_FLOOR = 0.30

# --- Embeddings (Phase 2) ---------------------------------------------------
# A MULTILINGUAL model so Hindi/Marwari queries work later, not just English.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Generation (Phase 4) ---------------------------------------------------
# The LLM is a SWAPPABLE component. `llm.py` hides the backend behind one
# function, so switching engines is a one-line change here.
LLM_BACKEND = "groq"                    # "groq" | "claude" | "ollama"
GROQ_MODEL = "llama-3.3-70b-versatile"  # reliable at reproducing exact price figures (8B garbled them)
CLAUDE_MODEL = "claude-opus-4-8"        # used if you set LLM_BACKEND = "claude"
OLLAMA_MODEL = "llama3.2:3b"            # used if you set LLM_BACKEND = "ollama"

MAX_ANSWER_TOKENS = 900
TEMPERATURE = 0.2                       # low = factual & grounded, not creative
