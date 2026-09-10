"""Streamlit demo: pick songs, get recommendations from four models."""

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"

ALPHA = 40.0
REG = 0.05


@st.cache_resource
def load_model():
    m = joblib.load(ARTIFACT)
    m["col_to_song"] = {v: k for k, v in m["song_to_col"].items()}
    m["YtY"] = m["item_factors"].T @ m["item_factors"]
    return m


def als_user_vector(item_factors, YtY, chosen):
    """Solve for a new user's factors with the item factors held fixed.

    This is one half-step of ALS, applied to a single user -- the same
    calculation training runs across every user, every iteration.
    """
    Y = item_factors[chosen]
    c = 1.0 + ALPHA
    A = YtY + (c - 1.0) * (Y.T @ Y) + REG * np.eye(item_factors.shape[1])
    b = c * Y.sum(axis=0)
    return np.linalg.solve(A, b)


def rank(scores, chosen, k):
    scores = scores.copy()
    scores[chosen] = -np.inf
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


def normalise(scores):
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("Hybrid Music Recommender")
st.caption(
    "Trained on 1.5M playlist interactions across 21,219 tracks. "
    "Pick a few songs and the model works out your taste from scratch."
)

M = load_model()
titles = sorted(M["song_to_col"])

label = {s: s.replace("|", "  —  ").title() for s in titles}

picked = st.multiselect(
    "Songs you like",
    options=titles,
    format_func=lambda s: label[s],
    default=titles[:0],
    help="Three to five is enough",
)

model_name = st.radio(
    "Model",
    ["Hybrid", "Collaborative filtering (ALS)", "Content-based", "Most popular"],
    horizontal=True,
)

if not picked:
    st.info("Pick at least one song to see recommendations.")
    st.stop()

chosen = [M["song_to_col"][s] for s in picked]

if model_name == "Most popular":
    scores = M["item_popularity"].astype(float)
else:
    als_scores = M["item_factors"] @ als_user_vector(
        M["item_factors"], M["YtY"], chosen
    )
    profile = M["item_vectors"][chosen].mean(axis=0)
    n = np.linalg.norm(profile)
    content_scores = (
        M["item_vectors"] @ (profile / n) if n else np.zeros(len(M["item_popularity"]))
    )

    if model_name == "Collaborative filtering (ALS)":
        scores = als_scores
    elif model_name == "Content-based":
        scores = content_scores
    else:
        scores = 0.9 * normalise(als_scores) + 0.1 * normalise(content_scores)

st.subheader("Recommended")
for i, col in enumerate(rank(scores, chosen, 10), 1):
    st.write(f"**{i}.** {label[M['col_to_song'][col]]}")