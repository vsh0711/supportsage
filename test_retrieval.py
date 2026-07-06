# test_retrieval.py
import sys
sys.path.insert(0, ".")

from app.rag.retriever import retriever

# Test queries — covers different intents in our dataset
test_queries = [
    "How do I cancel my order?",
    "I want a refund for my purchase",
    "My account is locked and I can't login",
    "Where is my delivery?",
    "How do I update my payment method?"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    
    results = retriever.retrieve(query, top_k=10, top_k_rerank=3)
    
    for i, r in enumerate(results):
        print(f"\n[Result {i+1}] Score: {r['score']:.4f}")
        print(f"Category: {r['metadata']['category']}")
        print(f"Intent: {r['metadata']['intent']}")
        print(f"Original Q: {r['metadata']['original_question']}")
        print(f"Doc preview: {r['document'][:150]}...")