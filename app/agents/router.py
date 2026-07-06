#building the LLM Answer layer, which orchestrates the flow of a support query through 
#input validation, routing, retrieval, generation, and output validation.
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings
from app.core.prompts import QA_PROMPT, ROUTER_PROMPT
from app.rag.retriever import retriever
from app.guardrails.input_guard import input_guard
from app.guardrails.output_guard import output_guard
import logging

logger = logging.getLogger(__name__)


class SupportAgent:
    """
    The orchestration layer — decides what to do with each query.

    Flow:
    1. Input guard — check injection, scope, redact PII
    2. Route — SIMPLE / COMPLEX / ESCALATE
    3. Retrieve — hybrid search + rerank
    4. Generate — LLM answers using retrieved context only
    5. Output guard — redact PII, catch hallucinations
    6. Return structured response

    Model tiering:
    - SIMPLE queries → llama-3.1-8b-instant (fast, cheap)
    - COMPLEX queries → llama-3.3-70b-versatile (stronger reasoning)
    - ESCALATE → skip LLM, return human handoff message

    Why this matters:
    70% of support queries are simple FAQs.
    Routing them to the small model cuts cost by ~60%
    while maintaining quality where it matters.
    """

    def __init__(self):
        self.primary_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.PRIMARY_MODEL,
            temperature=0.1,
            max_tokens=512
        )
        self.fallback_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.FALLBACK_MODEL,
            temperature=0.1,
            max_tokens=512
        )
        self.router_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.PRIMARY_MODEL,
            temperature=0  # deterministic routing
        )

    def _route(self, query: str) -> str:
        """Classify query as SIMPLE / COMPLEX / ESCALATE."""
        try:
            prompt = ROUTER_PROMPT.format(query=query)
            result = self.router_llm.invoke([HumanMessage(content=prompt)])
            decision = result.content.strip().upper()
            if decision not in ["SIMPLE", "COMPLEX", "ESCALATE"]:
                return "SIMPLE"  # safe default
            return decision
        except Exception as e:
            logger.warning(f"Router failed: {e}. Defaulting to SIMPLE.")
            return "SIMPLE"

    def _generate(self, query: str, context_docs: list[dict], route: str) -> str:
        """Generate answer using retrieved context."""
        # Build context string from retrieved docs
        context = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{doc['document']}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = QA_PROMPT.format(context=context, question=query)

        # Model tiering — pick LLM based on route
        llm = self.fallback_llm if route == "COMPLEX" else self.primary_llm

        try:
            result = llm.invoke([HumanMessage(content=prompt)])
            return result.content.strip()
        except Exception as e:
            logger.warning(f"Primary generation failed: {e}. Trying fallback...")
            try:
                # Fallback to other model if primary fails
                fallback = self.primary_llm if route == "COMPLEX" else self.fallback_llm
                result = fallback.invoke([HumanMessage(content=prompt)])
                return result.content.strip()
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return (
                    "I'm experiencing technical difficulties right now. "
                    "Please try again in a moment or contact our support team directly."
                )

    def run(self, query: str) -> dict:
        """
        Main entry point.

        Returns:
        {
            "answer": str,
            "route": str,           # SIMPLE / COMPLEX / ESCALATE
            "sources": list,        # retrieved doc metadata
            "pii_detected": list,   # what PII was found in input
            "is_refusal": bool,     # did we refuse to answer?
            "escalate": bool        # should this go to human agent?
        }
        """
        # Step 1: Input validation
        guard_result = input_guard.validate(query)

        if not guard_result["allowed"]:
            reason = guard_result["reason"]
            if reason == "prompt_injection":
                return {
                    "answer": "I detected an attempt to manipulate my instructions. This has been logged.",
                    "route": "BLOCKED",
                    "sources": [],
                    "pii_detected": [],
                    "is_refusal": True,
                    "escalate": False
                }
            elif reason == "out_of_scope":
                return {
                    "answer": "I'm a customer support assistant and can only help with order, billing, account, and delivery questions.",
                    "route": "BLOCKED",
                    "sources": [],
                    "pii_detected": [],
                    "is_refusal": True,
                    "escalate": False
                }

        cleaned_query = guard_result["cleaned_query"]
        pii_detected = guard_result["pii_detected"]

        # Step 2: Route the query
        route = self._route(cleaned_query)
        logger.info(f"Query routed as: {route}")

        # Step 3: Handle escalation immediately
        if route == "ESCALATE":
            return {
                "answer": "I understand this is an important issue. Let me connect you with a human agent who can give you the attention you deserve.",
                "route": "ESCALATE",
                "sources": [],
                "pii_detected": pii_detected,
                "is_refusal": False,
                "escalate": True
            }

        # Step 4: Retrieve relevant docs
        docs = retriever.retrieve(cleaned_query)

        if not docs:
            return {
                "answer": "I don't have enough information to answer that. Let me connect you with a human agent.",
                "route": route,
                "sources": [],
                "pii_detected": pii_detected,
                "is_refusal": True,
                "escalate": True
            }

        # Step 5: Generate answer
        raw_answer = self._generate(cleaned_query, docs, route)

        # Step 6: Output validation
        output_result = output_guard.validate(raw_answer)
        final_answer = (
            output_result["override_response"]
            if output_result["override_response"]
            else output_result["cleaned_response"]
        )

        return {
            "answer": final_answer,
            "route": route,
            "sources": [d["metadata"] for d in docs],
            "pii_detected": pii_detected,
            "is_refusal": output_result["is_refusal"],
            "escalate": output_result["is_refusal"]
        }


# Singleton
agent = SupportAgent()
