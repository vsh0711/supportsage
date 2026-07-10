# app/guardrails/output_guard.py
import re
import logging

logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "PHONE": r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
    "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
}

HALLUCINATION_SIGNALS = [
    "as of my knowledge cutoff",
    "i don't have access to real-time",
    "as an ai language model",
    "i'm just an ai",
    "my training data",
]

REFUSAL_TRIGGERS = [
    "i don't have enough information",
    "i cannot answer",
    "no information available",
]

class OutputGuard:
    def __init__(self):
        self.pii_patterns = {k: re.compile(v) for k, v in PII_PATTERNS.items()}

    def redact_pii(self, text: str) -> str:
        for pii_type, pattern in self.pii_patterns.items():
            text = pattern.sub(f"[{pii_type}_REDACTED]", text)
        return text

    def check_hallucination(self, text: str) -> tuple[bool, str]:
        lower = text.lower()
        for signal in HALLUCINATION_SIGNALS:
            if signal in lower:
                return True, signal
        return False, ""

    def is_refusal(self, text: str) -> bool:
        lower = text.lower()
        return any(t in lower for t in REFUSAL_TRIGGERS)

    def validate(self, response: str) -> dict:
        is_hallucinating, signal = self.check_hallucination(response)
        if is_hallucinating:
            return {
                "cleaned_response": "",
                "is_hallucination": True,
                "is_refusal": False,
                "override_response": "I couldn't find a confident answer. Let me connect you with a human agent."
            }
        cleaned = self.redact_pii(response)
        return {
            "cleaned_response": cleaned,
            "is_hallucination": False,
            "is_refusal": self.is_refusal(cleaned),
            "override_response": None
        }

output_guard = OutputGuard()