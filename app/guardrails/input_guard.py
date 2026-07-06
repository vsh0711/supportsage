# Input validation + PII redaction for user queries to the support agent + injection defense
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import logging

logger = logging.getLogger(__name__)

# Prompt injection patterns — common attack signatures
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
    r"do\s+anything\s+now",  # DAN attack (Do-Anything-now Attack)
]

# Out of scope topics for a support agent
OUT_OF_SCOPE_PATTERNS = [
    r"\b(politics|election|vote|president|government)\b",
    r"\b(religion|god|allah|jesus|bible|quran)\b",
    r"\b(stock\s+tip|invest\s+in|buy\s+stock)\b",
]


class InputGuard:
    """
    Three checks on every incoming query:
    
    1. Prompt injection detection — regex pattern matching against known attacks
       Why regex and not LLM? Speed. LLM check = 500ms. Regex = <1ms.
       For security checks, latency matters more than nuance.
    
    2. PII detection — finds emails, phone numbers, SSNs, credit cards
       Why redact input PII? 
       - User might accidentally paste their SSN in a query
       - We don't want that stored in our logs or sent to external APIs
    
    3. Scope check — is this even a support question?
       Prevents the bot from becoming a general-purpose chatbot
    """

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self.scope_patterns = [
            re.compile(p, re.IGNORECASE) for p in OUT_OF_SCOPE_PATTERNS
        ]

    def check_injection(self, text: str) -> tuple[bool, str]:
        """Returns (is_safe, reason)"""
        for pattern in self.injection_patterns:
            if pattern.search(text):
                logger.warning(f"Prompt injection detected: {pattern.pattern}")
                return False, "prompt_injection"
        return True, ""

    def check_scope(self, text: str) -> tuple[bool, str]:
        """Returns (is_in_scope, reason)"""
        for pattern in self.scope_patterns:
            if pattern.search(text):
                return False, "out_of_scope"
        return True, ""

    def redact_pii(self, text: str) -> tuple[str, list]:
        """
        Returns (redacted_text, list_of_detected_entities)
        Detected entities are logged for audit but not stored with text.
        """
        results = self.analyzer.analyze(
            text=text,
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                      "US_SSN", "PERSON", "LOCATION"],
            language="en"
        )
        if not results:
            return text, []

        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        detected = [r.entity_type for r in results]
        logger.info(f"PII redacted: {detected}")
        return anonymized.text, detected

    def validate(self, query: str) -> dict:
        """
        Main entry point. Returns:
        {
            "allowed": bool,
            "reason": str,          # why blocked, empty if allowed
            "cleaned_query": str,   # PII-redacted version
            "pii_detected": list    # what PII types were found
        }
        """
        # Check 1: Injection
        is_safe, reason = self.check_injection(query)
        if not is_safe:
            return {
                "allowed": False,
                "reason": reason,
                "cleaned_query": "",
                "pii_detected": []
            }

        # Check 2: Scope
        in_scope, reason = self.check_scope(query)
        if not in_scope:
            return {
                "allowed": False,
                "reason": reason,
                "cleaned_query": "",
                "pii_detected": []
            }

        # Check 3: Redact PII before further processing
        cleaned_query, pii_detected = self.redact_pii(query)

        return {
            "allowed": True,
            "reason": "",
            "cleaned_query": cleaned_query,
            "pii_detected": pii_detected
        }


# Singleton
input_guard = InputGuard()