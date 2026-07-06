#output validation guardrails for the support agent

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import logging

logger = logging.getLogger(__name__)

# Hallucination signals — phrases that indicate LLM is making things up
HALLUCINATION_SIGNALS = [
    "as of my knowledge cutoff",
    "i don't have access to real-time",
    "i cannot browse the internet",
    "as an ai language model",
    "i'm just an ai",
    "my training data",
]

# Refusal triggers — responses that mean retrieval failed
REFUSAL_TRIGGERS = [
    "i don't have enough information",
    "i cannot answer",
    "i don't know",
    "no information available",
]


class OutputGuard:
    """
    Two checks on every LLM response before it reaches the user:
    
    1. PII redaction — LLM might echo back PII from retrieved docs
       (e.g. a support doc contains a customer's example email)
    
    2. Hallucination detection — catch LLM breaking character
       and referring to its own training data instead of our KB
    """

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact_pii(self, text: str) -> str:
        results = self.analyzer.analyze(
            text=text,
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"],
            language="en"
        )
        if not results:
            return text
        return self.anonymizer.anonymize(text=text, analyzer_results=results).text

    def check_hallucination(self, text: str) -> tuple[bool, str]:
        """Returns (is_hallucinating, signal_found)"""
        lower = text.lower()
        for signal in HALLUCINATION_SIGNALS:
            if signal in lower:
                logger.warning(f"Hallucination signal detected: {signal}")
                return True, signal
        return False, ""

    def is_refusal(self, text: str) -> bool:
        lower = text.lower()
        return any(trigger in lower for trigger in REFUSAL_TRIGGERS)

    def validate(self, response: str) -> dict:
        """
        Returns:
        {
            "cleaned_response": str,
            "is_hallucination": bool,
            "is_refusal": bool,
            "override_response": str or None  # if we need to replace the response
        }
        """
        # Check hallucination first
        is_hallucinating, signal = self.check_hallucination(response)
        if is_hallucinating:
            return {
                "cleaned_response": "",
                "is_hallucination": True,
                "is_refusal": False,
                "override_response": (
                    "I'm sorry, I couldn't find a confident answer for that. "
                    "Let me connect you with a human agent who can help."
                )
            }

        # Redact any PII in output
        cleaned = self.redact_pii(response)

        return {
            "cleaned_response": cleaned,
            "is_hallucination": False,
            "is_refusal": self.is_refusal(cleaned),
            "override_response": None
        }


# Singleton
output_guard = OutputGuard()