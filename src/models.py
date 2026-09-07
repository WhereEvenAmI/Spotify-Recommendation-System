"""Recommendation models.

Every model exposes the same two methods -- fit(train) and recommend(user_row, k)
-- so evaluate() can score any of them without knowing which one it has.
"""

import numpy as np
from implicit.als import AlternatingLeastSquares


class BaseRecommender:
    """Shared behaviour: remember the training data, know what a user already has."""

    def fit(self, train):
        self.train = train
        return self

    def seen(self, user_row):
        start, end = self.train.indptr[user_row], self.train.indptr[user_row + 1]
        return set(self.train.indices[start:end])

    def recommend(self, user_row, k):
        raise NotImplementedError


class RandomRecommender(BaseRecommender):
    """The floor. Anything that cannot beat this is broken, not undertuned."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def recommend(self, user_row, k):
        seen = self.seen(user_row)
        pool = self.rng.choice(self.train.shape[1], size=k + len(seen) + 50, replace=False)
        return [i for i in pool if i not in seen][:k]


class PopularityRecommender(BaseRecommender):
    """The same top songs for everybody. Deceptively strong on music data."""

    def fit(self, train):
        super().fit(train)
        popularity = np.asarray(train.sum(axis=0)).ravel()
        self.ranking = np.argsort(-popularity)
        return self

    def recommend(self, user_row, k):
        seen = self.seen(user_row)
        return [i for i in self.ranking if i not in seen][:k]


class ALSRecommender(BaseRecommender):
    """Matrix factorisation for implicit feedback (Hu, Koren & Volinsky 2008).

    alpha scales observed interactions so the model treats them as high-confidence
    and the zeros as weak evidence -- a zero may mean dislike, or simply never heard.
    """

    def __init__(self, factors=64, regularization=0.05, iterations=20, alpha=40, seed=42):
        self.alpha = alpha
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
        )

    def fit(self, train):
        super().fit(train)
        self.model.fit((train * self.alpha).astype("float32"))
        return self

    def recommend(self, user_row, k):
        ids, _ = self.model.recommend(
            user_row,
            self.train[user_row],
            N=k,
            filter_already_liked_items=True,
        )
        return list(ids)