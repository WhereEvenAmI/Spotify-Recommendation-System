"""Loading the cleaned interaction data and building the user-item matrix."""

from pathlib import Path

import re
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

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


def normalise(text):
    """Strip an artist or track name down to a comparable form.

    Uses \\w with the unicode flag rather than [a-z0-9] -- an ASCII-only
    pattern silently empties every non-Latin-script title.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"\s-\s.*$", " ", t)
    t = re.sub(r"\b(feat|ft|featuring|with)\b.*$", " ", t)
    t = t.replace("'", "").replace("\u2019", "")
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def load_track_features(song_to_col):
    """Audio features for the songs in our matrix, indexed by column number."""
    tracks = pd.read_csv(DATA_DIR / "track_features.csv")

    tracks["key"] = (
        tracks["artists"].astype(str).str.split(";").str[0].map(normalise)
        + "|"
        + tracks["track_name"].map(normalise)
    )

    tracks = tracks.drop_duplicates(subset="key")
    tracks = tracks[tracks["key"].isin(song_to_col)].copy()
    tracks["col"] = tracks["key"].map(song_to_col)

    return tracks.set_index("col").sort_index()