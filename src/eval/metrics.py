def compute_groundedness(answer: str, retrieved_chunks: list) -> float:
    """
    Simple groundedness metric: returns the fraction of answer sentences that overlap with retrieved context.
    """
    import re
    answer_sents = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
    context = ' '.join([c.get('text', '') for c in retrieved_chunks])
    grounded = 0
    for sent in answer_sents:
        if sent and sent.lower() in context.lower():
            grounded += 1
    return grounded / len(answer_sents) if answer_sents else 0.0

def compute_answer_relevance(answer: str, query: str) -> float:
    """
    Simple answer relevance: returns the fraction of query keywords found in the answer.
    """
    import re
    from collections import Counter
    query_words = set(re.findall(r'\w+', query.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    if not query_words:
        return 0.0
    overlap = query_words & answer_words
    return len(overlap) / len(query_words)
