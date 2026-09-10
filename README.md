# Spotify Hybrid Music Recommender

A recommendation system built on 1.5M playlist interactions, comparing five approaches
under one evaluation harness. Best model reaches **NDCG@10 of 0.117** — 5.7× a
most-popular baseline — while covering 66× more of the catalogue.

---

## Why this dataset is awkward

Most public music-recommender projects use Spotify's `audio-features` API or the
Million Playlist Dataset. Neither is available: the API endpoint was deprecated in
late 2024, and the MPD is no longer a public download.

So this project reconstructs the problem from two unrelated Kaggle scrapes:

| | Source | Contents |
|---|---|---|
| **Interactions** | `andrewmvd/spotify-playlists` | 12.9M rows of user / playlist / artist / track |
| **Content** | `maharshipandya/-spotify-tracks-dataset` | 114K tracks with audio features and genre |

They share no key. Interactions reference tracks by free text, features by Spotify ID.
Reconciling them is the first real piece of work.

---

## Data pipeline

```
12,902,577  raw lines in the interactions file
12,901,979  loaded            (598 malformed, 0.005%)
 2,823,960  distinct songs, 15,918 users
    57,117  matched to the feature catalogue (2.02% of songs)
    21,219  distinct songs after normalisation collapsed spelling variants
 1,531,635  interactions kept (11.87%), 14,857 users
```

**The 598 malformed rows** were caused by unescaped double-quote characters inside
track titles — mostly vinyl formats like `Original 12"`. The quote prematurely closed
the CSV field and scrambled the row's boundaries. Quantified before discarding.

**The 2% song match rate** looks alarming but reflects catalogue size, not a broken
join: those 21,219 songs represent **74% of the entire feature catalogue**. People
listen to 36× more songs than the features file contains.

**Why only 12% of interactions were kept.** The catalogue was deliberately scoped to
the intersection of both datasets. Content-based and collaborative models must be
evaluated on identical items or the comparison is meaningless.

**Text normalisation** strips bracketed suffixes, `feat.` clauses and version markers
so `"Hello"`, `"Hello - Remastered"` and `"Hello (Radio Edit)"` collapse to one key.
It uses `\w` with the Unicode flag rather than `[a-z0-9]` — an ASCII-only pattern
silently empties every non-Latin-script title, which cost ~2,900 songs before it was
caught.

---

## Evaluation

Built **before** any model, so every model is scored identically.

- **Matrix**: 14,857 × 21,219, 1.2M filled cells, 0.384% dense
- **Split**: leave-5-out per user, for the 12,052 users with 10+ interactions
- **Metrics**: Precision@10, Recall@10, NDCG@10, plus catalogue coverage
- **Feedback type**: implicit — playlist membership, not ratings

Users with fewer than 10 interactions stay in **training** but are excluded from
**scoring**. Holding items out from them leaves too little signal to learn from, but
deleting them would discard real co-occurrence information. Training and evaluation
populations don't have to match.

---

## Results

Top-10 recommendations, 2,000 sampled evaluation users, seed 42.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Random | 0.0003 | 0.0005 | 0.0004 | — |
| Content-based (clustered) | 0.0050 | 0.0100 | 0.0080 | — |
| Content-based | 0.0065 | 0.0130 | 0.0108 | 25.0% |
| Most popular | 0.0112 | 0.0225 | 0.0205 | 0.2% |
| ALS | 0.0636 | 0.1271 | 0.1135 | 14.4% |
| **Hybrid (α=0.9)** | **0.0636** | **0.1272** | **0.1149** | **14.6%** |

Re-checked on all 12,052 evaluation users: hybrid **0.1166** vs pure ALS **0.1149**.

**Random validates the harness.** With 5 held-out songs from 21,219, random should
score ≈ 5/21,219 = 0.00024. It scored 0.0003.

**Coverage needs the accuracy column beside it.** A random recommender would cover
~100% of the catalogue and be useless. The meaningful result is that ALS beats
most-popular on *both* axes: 5.6× the NDCG and 66× the coverage. Most-popular
reached only **46 distinct songs across 12,000 users**.

### Blend weight sweep

| α (weight on ALS) | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| NDCG@10 | 0.011 | 0.059 | 0.086 | 0.103 | 0.113 | **0.115** | 0.113 |

Even at α=0.2 — 80% content weight — NDCG is already 5× pure content. The
behavioural signal dominates almost immediately.

---

## Qualitative check

A user with eight Blink-182 tracks and two All Time Low tracks. ALS recommendations:

```
paramore | aint it fun          good charlotte | the anthem
paramore | still into you       fall out boy   | alone together
sum 41   | in too deep          sum 41         | fat lip
```

Pop punk and emo, almost every one — **from a model that was never given genre
information**. It only saw which songs appear in the same people's playlists. Genre
emerged from co-occurrence alone.

---

## What didn't work

**Content-based lost to a non-personalised baseline** — 0.0108 vs 0.0205 NDCG.
Audio similarity is a weak proxy for what someone wants next. Its recommendations for
the pop-punk user included Czech and Argentine punk bands: acoustically similar,
culturally irrelevant. The model worked correctly and the recommendations were still
wrong.

**Clustering user profiles made it worse.** Averaging a user's whole history into one
vector misrepresents split taste — someone who listens to metal and hip-hop gets a
profile representing neither. Per-user KMeans with best-cluster matching scored
**0.0094**; a size-weighted variant scored **0.0080**, against **0.0108** for the
single average.

The results were monotonic in how much history informed each score, which suggests the
features were too noisy to support splitting: thinner centroids hurt more than
separating the modes helped. The simpler model was kept, and both variants remain in
the code as evidence.

**Artist was deliberately excluded from content features.** Adding it would have
raised the score — recommending another Blink-182 track to someone with eight is
nearly a lookup — but it wouldn't help the case the model exists for. A genuinely new
artist releasing a new song gets nothing from artist features. The metric would have
improved while the model's actual cold-start capability stayed flat.

**The blend gain is real but marginal.** α=0.9 beats pure ALS consistently across all
three metrics on the full population, but by ~1.5% relative. That doesn't justify
shipping two models. What justifies the hybrid is the **switching fallback** for cold
users and items — and the main evaluation can't measure that benefit, because it
excludes low-interaction users by design.

---

## Structure

```
src/
├── data.py         load interactions, normalise text, build the sparse matrix
├── evaluation.py   leave-k-out split, ranking metrics, coverage
└── models.py       shared interface: random, popularity, ALS, content, hybrid
notebooks/
├── 01_explore.ipynb   data exploration and the join
└── 02_models.ipynb    training and comparison
```

Every model exposes `fit(train)` and `recommend(user_row, k)`, so `evaluate()` scores
any of them without knowing which it has. Adding a model is one dictionary entry.

---

## Running it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download both datasets from the links above into `data/`, then run
`notebooks/01_explore.ipynb` followed by `notebooks/02_models.ipynb`.

Datasets aren't committed — they're public and reproducible, and the interactions file
is 1.2 GB.

---

## What I'd do next

**Neural collaborative filtering.** ALS learns a linear factorisation; a denoising or
variational autoencoder (Mult-VAE, Liang et al. 2018) learns a non-linear compression
of the same interaction vectors and typically outperforms matrix factorisation on
implicit feedback. Runs on the existing matrix with no additional data.

**Ranking-optimised training.** ALS minimises reconstruction error, not ranking
quality. BPR optimises pairwise ranking directly — closer to what Precision@K and NDCG
actually measure. A direct comparison would show whether the objective mismatch costs
anything.

**Richer cold-start features.** The content model is limited by nine audio features.
Audio embeddings from a pretrained model, or lyric embeddings, would give it a better
notion of similarity than tempo and energy.

**What this dataset can't support.** Reinforcement learning from implicit feedback —
plays as reward, skips as penalty — is how production systems adapt to immediate user
response. It isn't possible here: playlist membership carries no negative signal and
no timestamps, so there's no reward to learn a policy from. It would need
session-level interaction logs.
