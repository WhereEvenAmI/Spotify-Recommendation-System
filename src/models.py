"""Recommendation models.

Every model exposes the same two methods -- fit(train) and recommend(user_row, k)
-- so evaluate() can score any of them without knowing which one it has.
"""

import numpy as np
from implicit.als import AlternatingLeastSquares
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import AUDIO_FEATURES


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

class ContentBasedRecommender(BaseRecommender):
    """Recommends songs that sound like what the user already listens to.

    Knows nothing about other users -- which makes it weaker on accuracy but
    able to recommend a track nobody in the dataset has ever played.
    """

    def fit(self, train, features):
        super().fit(train)

        numeric = StandardScaler().fit_transform(
            features[AUDIO_FEATURES].to_numpy(dtype="float64")
        )
        genres = pd.get_dummies(features["track_genre"]).to_numpy(dtype="float64")
        vectors = np.hstack([numeric, genres])

        # unit-length rows, so a dot product is exactly cosine similarity
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

        self.item_vectors = np.zeros((train.shape[1], vectors.shape[1]))
        self.item_vectors[features.index.to_numpy()] = vectors
        return self

    def recommend(self, user_row, k):
        seen = list(self.seen(user_row))
        if not seen:
            return []

        profile = self.item_vectors[seen].mean(axis=0)
        norm = np.linalg.norm(profile)
        if norm == 0:
            return []
        profile /= norm

        scores = self.item_vectors @ profile
        scores[seen] = -np.inf

        top = np.argpartition(-scores, k)[:k]
        return list(top[np.argsort(-scores[top])])

from sklearn.cluster import KMeans


class ClusteredContentRecommender(ContentBasedRecommender):
    """Content-based with several taste profiles per user instead of one.

    A single averaged vector puts a metal-and-hip-hop listener in a region
    where neither genre lives. Clustering keeps the modes separate.
    """

    def __init__(self, n_clusters=3, seed=42):
        self.n_clusters = n_clusters
        self.seed = seed

    def recommend(self, user_row, k):
        seen = list(self.seen(user_row))
        if not seen:
            return []

        vectors = self.item_vectors[seen]
        n = min(self.n_clusters, len(seen))

        if n == 1:
            centroids = vectors.mean(axis=0, keepdims=True)
            weights = np.array([1.0])        
        else:
            km = KMeans(n_clusters=n, n_init=1, random_state=self.seed).fit(vectors)
            centroids = km.cluster_centers_
            weights = np.bincount(km.labels_, minlength=n) / len(seen)
            
        centroids = centroids / np.maximum(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9
        )

        scores = ((self.item_vectors @ centroids.T) * weights).max(axis=1)
        scores[seen] = -np.inf

        top = np.argpartition(-scores, k)[:k]
        return list(top[np.argsort(-scores[top])])

class HybridRecommender(BaseRecommender):
    """Blends ALS and content scores, and falls back to content for cold users.

    alpha is the weight on ALS. cold_threshold is the number of interactions
    below which ALS has too little signal and we use content alone.
    """

    def __init__(self, als, content, alpha=0.8, cold_threshold=5):
        self.als = als
        self.content = content
        self.alpha = alpha
        self.cold_threshold = cold_threshold

    def _als_scores(self, user_row):
        return self.als.model.item_factors @ self.als.model.user_factors[user_row]

    def _content_scores(self, user_row):
        seen = list(self.seen(user_row))
        profile = self.content.item_vectors[seen].mean(axis=0)
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(self.train.shape[1])
        return self.content.item_vectors @ (profile / norm)

    @staticmethod
    def _rescale(scores):
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)

    def recommend(self, user_row, k):
        seen = list(self.seen(user_row))
        if not seen:
            return []

        if len(seen) < self.cold_threshold:
            return self.content.recommend(user_row, k)

        combined = (
            self.alpha * self._rescale(self._als_scores(user_row))
            + (1 - self.alpha) * self._rescale(self._content_scores(user_row))
        )
        combined[seen] = -np.inf

        top = np.argpartition(-combined, k)[:k]
        return list(top[np.argsort(-combined[top])])