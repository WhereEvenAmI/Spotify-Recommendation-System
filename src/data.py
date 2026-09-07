"""Loading the cleaned interaction data and building the user-item matrix."""

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_interactions():
    """One row per user-song pair, with strength = number of playlists."""
    data = pd.read_pickle(DATA_DIR / "interactions_clean.pkl")
    return (
        data.groupby(["user_id", "key"])
            .size()
            .reset_index(name="strength")
    )


def build_matrix(interactions):
    """Return the sparse user-item matrix and the two index maps."""
    users = interactions["user_id"].unique()
    songs = interactions["key"].unique()

    user_to_row = {u: i for i, u in enumerate(users)}
    song_to_col = {s: i for i, s in enumerate(songs)}

    matrix = csr_matrix(
        (interactions["strength"],
         (interactions["user_id"].map(user_to_row),
          interactions["key"].map(song_to_col))),
        shape=(len(users), len(songs)),
    )
    return matrix, user_to_row, song_to_col