import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-placeholder")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "sk-ant-placeholder")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "gemini-placeholder")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk-placeholder")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Models
DEFAULT_OPENAI_MODEL = "gpt-4.1"
SMART_OPENAI_MODEL = "o3"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
DEFAULT_OPENROUTER_MODEL = "openai/o3"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

TICKET_TAGS = [
    "billing", "refund", "shipping", "damaged_item", "late_delivery",
    "account_access", "password_reset", "subscription", "cancellation",
    "bug_report", "feature_request", "complaint", "praise", "urgent",
    "vip_customer", "legal_threat", "chargeback", "warranty",
    "international", "gift_order",
]


def load_text(name: str) -> str:
    with open(DATA_DIR / name, "r") as f:
        return f.read()
