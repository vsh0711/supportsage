
# Used during ingestion — enriches each chunk with document-level context
# This is Anthropic's Contextual Retrieval technique, now running on Groq
CONTEXTUAL_RETRIEVAL_PROMPT = """You are preparing a customer support knowledge base.
Given the following support interaction, write 1-2 sentences that describe what topic 
this interaction covers and what kind of customer problem it addresses.
This context will be prepended to the chunk to make it retrievable.

Support Interaction:
Category: {category}
Intent: {intent}
Customer Question: {instruction}
Agent Response: {response}

Write only the context sentences, nothing else."""


# Used at query time — generates the final answer
QA_PROMPT = """You are SupportSage, a helpful customer support AI assistant.
Answer the customer's question using ONLY the context provided below.
If the context does not contain enough information to answer, say: 
"I don't have enough information to answer that. Let me connect you with a human agent."

Do NOT make up information. Do NOT reference the context explicitly (don't say "according to the context").
Be concise, empathetic, and professional.

Context:
{context}

Customer Question: {question}

Answer:"""


# Used by the agent router to classify query complexity
ROUTER_PROMPT = """Classify this customer support query into one of three categories:

1. SIMPLE — FAQs, straightforward questions with direct answers (e.g. "how do I cancel my order")
2. COMPLEX — Multi-part questions, complaints, edge cases requiring detailed reasoning
3. ESCALATE — Angry customers, legal threats, sensitive issues, anything needing a human

Query: {query}

Respond with ONLY one word: SIMPLE, COMPLEX, or ESCALATE"""