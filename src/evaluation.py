"""Train/test split and ranking metrics.

Written before any model, so every model is scored identically.
"""

import numpy as np
from scipy.sparse import csr_matrix


def leave_k_out(matrix, k=5, min_interactions=10, seed=42):
    """Hide k songs from each user who has enough of them.

    Users below min_interactions stay in training but are never scored --
    holding items out from them leaves too little signal to learn from.
    Returns (train, test, eval_users).
    """
    rng = np.random.default_rng(seed)
    per_user = np.diff(matrix.indptr)
    eval_users = np.where(per_user >= min_interactions)[0]

    mask = np.ones(matrix.nnz, dtype=bool)
    test = {}

    for u in eval_users:
        start, end = matrix.indptr[u], matrix.indptr[u + 1]
        positions = rng.choice(np.arange(start, end), size=k, replace=False)
        test[u] = matrix.indices[positions]
        mask[positions] = False

    rows = np.repeat(np.arange(matrix.shape[0]), per_user)
    train = csr_matrix(
        (matrix.data[mask], (rows[mask], matrix.indices[mask])),
        shape=matrix.shape,
    )
    return train, test, eval_users


def precision_at_k(recommended, relevant, k):
    """Of the k we recommended, what fraction were right?"""
    return len(set(recommended[:k]) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    """Of the songs they wanted, what fraction did we find?"""
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Like precision, but a hit at rank 1 counts more than one at rank 10."""
    rel = set(relevant)
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in rel)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def evaluate(recommend_fn, test, eval_users, k=10, sample=2000, seed=42):
    """Score any recommend function against the held-out songs."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eval_users, size=min(sample, len(eval_users)), replace=False)

    p, r, n = [], [], []
    for u in chosen:
        recs = recommend_fn(u, k)
        truth = test[u]
        p.append(precision_at_k(recs, truth, k))
        r.append(recall_at_k(recs, truth, k))
        n.append(ndcg_at_k(recs, truth, k))

    return {
        f"precision@{k}": round(float(np.mean(p)), 4),
        f"recall@{k}": round(float(np.mean(r)), 4),
        f"ndcg@{k}": round(float(np.mean(n)), 4),
    }