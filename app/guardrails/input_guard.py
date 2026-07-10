# app/guardrails/input_guard.py
import re
import logging

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"forget\s+(everything|all|previous)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are|a)",
    r"new\s+instructions?:",
    r"system\s*:",
    r"<\s*system\s*>",
    r"disregard\s+(your|all|previous)",
    r"bypass\s+(safety|guidelines|restrictions)",
    r"jailbreak",
    r"do\s+anything\s+now",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\b(politics|election|vote|president|government)\b",
    r"\b(religion|god|allah|jesus|bible|quran)\b",
    r"\b(stock\s+tip|invest\s+in|buy\s+stock)\b",
]

PII_PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "PHONE": r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
    "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
}

class InputGuard:
    def __init__(self):
        self.injection_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self.scope_patterns = [re.compile(p, re.IGNORECASE) for p in OUT_OF_SCOPE_PATTERNS]
        self.pii_patterns = {k: re.compile(v) for k, v in PII_PATTERNS.items()}

    def check_injection(self, text: str) -> tuple[bool, str]:
        for pattern in self.injection_patterns:
            if pattern.search(text):
                logger.warning(f"Injection detected: {pattern.pattern}")
                return False, "prompt_injection"
        return True, ""

    def check_scope(self, text: str) -> tuple[bool, str]:
        for pattern in self.scope_patterns:
            if pattern.search(text):
                return False, "out_of_scope"
        return True, ""

    def redact_pii(self, text: str) -> tuple[str, list]:
        detected = []
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(text):
                text = pattern.sub(f"[{pii_type}_REDACTED]", text)
                detected.append(pii_type)
        return text, detected

    def validate(self, query: str) -> dict:
        is_safe, reason = self.check_injection(query)
        if not is_safe:
            return {"allowed": False, "reason": reason, "cleaned_query": "", "pii_detected": []}

        in_scope, reason = self.check_scope(query)
        if not in_scope:
            return {"allowed": False, "reason": reason, "cleaned_query": "", "pii_detected": []}

        cleaned_query, pii_detected = self.redact_pii(query)
        return {"allowed": True, "reason": "", "cleaned_query": cleaned_query, "pii_detected": pii_detected}

input_guard = InputGuard()