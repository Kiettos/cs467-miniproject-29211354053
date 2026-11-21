"""Recommendation engine: Popularity, NLP Content-Based, User-Based CF, Hybrid.

Usage:
    nlp = NLPPipeline().fit(items)
    rec = Recommender(nlp, ratings)
    rec.recommend("U07", method="hybrid", n=5)
    rec.recommend_query("ancient maritime trade, ceramics, maps", n=5)
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

METHODS = ("popularity", "content", "cf", "hybrid")


def load_data(data_dir="data"):
    items = pd.read_csv(f"{data_dir}/items.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    return items, ratings


def _minmax(s):
    """Scale a Series to [0, 1] (NaN stays NaN)."""
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


class Recommender:
    def __init__(self, nlp, ratings, min_count=2, k_neighbors=5, alpha=0.5):
        self.nlp = nlp                  # fitted NLPPipeline (shared vocabulary)
        self.min_count = min_count      # popularity threshold
        self.k = k_neighbors            # neighbors for user-based CF
        self.alpha = alpha              # hybrid weight: alpha * CF + (1 - alpha) * content
        self.fit(ratings)

    # ------------------------------------------------------------------ setup
    def fit(self, ratings):
        self.ratings = ratings.copy()
        self._refresh()
        return self

    def _refresh(self):
        """Rebuild everything that depends on items / ratings."""
        self.item_ids = list(self.nlp.items["item_id"])
        self.idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self.title = dict(zip(self.nlp.items["item_id"], self.nlp.items["exhibit_title"]))
        # user x item matrix (NaN = not rated); new/cold items become all-NaN columns
        self.M = (self.ratings.pivot_table(index="user_id", columns="item_id", values="rating")
                  .reindex(columns=self.item_ids))
        self.user_mean = self.M.mean(axis=1)
        self.global_mean = self.ratings["rating"].mean()
        # user-user similarity on mean-centered ratings (missing -> 0)
        centered = self.M.sub(self.user_mean, axis=0).fillna(0.0)
        S = cosine_similarity(centered.values) if len(centered) else np.zeros((0, 0))
        self.user_sim = pd.DataFrame(np.nan_to_num(S), index=centered.index, columns=centered.index)
        # item-item content similarity (TF-IDF cosine)
        self.item_sim = cosine_similarity(self.nlp.matrix)

    def add_new_items(self, new_df):
        """Cold-start: register new exhibits (placard text only, no ratings)."""
        self.nlp.add_items(new_df)
        self._refresh()

    # ------------------------------------------------------------- helpers
    def rated_items(self, user_id):
        if user_id not in self.M.index:
            return set()
        return set(self.M.columns[self.M.loc[user_id].notna()])

    def _profile_vec(self, user_id):
        """User profile = rating-weighted sum of TF-IDF vectors of items the user rated."""
        row = self.M.loc[user_id].dropna()
        w = (row - 3.0).clip(lower=0).values          # only liked items (rating > 3) pull the profile
        if w.sum() == 0:
            w = row.values
        rows = [self.idx[i] for i in row.index]
        vec = self.nlp.matrix[rows].T @ w
        return csr_matrix(np.asarray(vec).reshape(1, -1))

    # ------------------------------------------------------------ 1. popularity
    def popularity_scores(self):
        """Weighted rating (IMDB formula): (v/(v+m))*R + (m/(v+m))*C, only items with >= min_count ratings."""
        g = self.ratings.groupby("item_id")["rating"].agg(["mean", "count"])
        g = g[g["count"] >= self.min_count]
        m, C = self.min_count, self.global_mean
        score = g["count"] / (g["count"] + m) * g["mean"] + m / (g["count"] + m) * C
        return score.reindex(self.item_ids)

    # ------------------------------------------------------------ 2. content-based
    def content_scores(self, user_id):
        """Cosine similarity between user profile and every exhibit."""
        sims = cosine_similarity(self._profile_vec(user_id), self.nlp.matrix).ravel()
        return pd.Series(sims, index=self.item_ids)

    # ------------------------------------------------------------ 3. user-based CF
    def cf_scores(self, user_id):
        """Predicted rating = user_mean + sum(sim * (r_vi - mean_v)) / sum(|sim|) over top-k similar users who rated i."""
        out = pd.Series(np.nan, index=self.item_ids)
        sims = self.user_sim.loc[user_id].drop(user_id)
        sims = sims[sims > 0]
        for item in self.item_ids:
            col = self.M[item].reindex(sims.index).dropna()
            if col.empty:
                continue
            s = sims[col.index].nlargest(self.k)
            r = col[s.index] - self.user_mean[s.index]
            out[item] = self.user_mean[user_id] + (s * r).sum() / s.abs().sum()
        return out.clip(1, 5)

    # ------------------------------------------------------------ hybrid
    def hybrid_scores(self, user_id):
        """alpha * CF + (1-alpha) * content (both scaled to [0,1]); items with no CF signal use content only."""
        cf = _minmax(self.cf_scores(user_id))
        ct = _minmax(self.content_scores(user_id))
        return (self.alpha * cf + (1 - self.alpha) * ct).where(cf.notna(), ct)

    def scores(self, user_id, method):
        if user_id not in self.M.index:          # unknown user -> popularity fallback
            return self.popularity_scores()
        return {"popularity": lambda: self.popularity_scores(),
                "content": lambda: self.content_scores(user_id),
                "cf": lambda: self.cf_scores(user_id),
                "hybrid": lambda: self.hybrid_scores(user_id)}[method]()

    # ------------------------------------------------------------ rating prediction (for RMSE/MAE)
    def predict_rating(self, user_id, item_id, method):
        mu = self.user_mean.get(user_id, self.global_mean)
        if method == "popularity":
            g = self.ratings[self.ratings.item_id == item_id]["rating"]
            return (g.sum() + 2 * self.global_mean) / (len(g) + 2)   # smoothed item mean
        if user_id not in self.M.index:
            return self.global_mean
        if method == "cf":
            p = self.cf_scores(user_id).get(item_id, np.nan)
            return mu if np.isnan(p) else p
        # content: similarity-weighted average of the user's own ratings (item-kNN on TF-IDF)
        rated = self.M.loc[user_id].dropna()
        sims = np.array([self.item_sim[self.idx[item_id], self.idx[i]] for i in rated.index])
        content = (sims * rated.values).sum() / sims.sum() if sims.sum() > 0 else mu
        if method == "content":
            return content
        cf = self.cf_scores(user_id).get(item_id, np.nan)          # hybrid
        return content if np.isnan(cf) else self.alpha * cf + (1 - self.alpha) * content

    # ------------------------------------------------------------ public API
    def _format(self, s, n, kw_fn):
        top = s.dropna().sort_values(ascending=False).head(n)
        rows = [{"rank": r, "item_id": iid, "exhibit_title": self.title[iid],
                 "score": round(float(sc), 3), "keywords": ", ".join(kw_fn(iid))}
                for r, (iid, sc) in enumerate(top.items(), 1)]
        return pd.DataFrame(rows)

    def recommend(self, user_id, method="hybrid", n=5):
        """Top-n UNVISITED exhibits for a visitor, with scores and keyword explanations."""
        assert method in METHODS, f"method must be one of {METHODS}"
        s = self.scores(user_id, method)
        s = s.drop(labels=list(self.rated_items(user_id)), errors="ignore")
        profile = self._profile_vec(user_id) if user_id in self.M.index else None

        def kw(iid):
            i = self.idx[iid]
            if profile is not None:
                k = self.nlp.explain_match(profile, i)
                if k:
                    return k
            return self.nlp.top_keywords(i)
        return self._format(s, n, kw)

    def recommend_query(self, text, n=5):
        """Top-n exhibits for a free-text query (content-based), with matching keywords."""
        q = self.nlp.transform_text(text)
        s = pd.Series(cosine_similarity(q, self.nlp.matrix).ravel(), index=self.item_ids)
        return self._format(s[s > 0], n,
                            lambda iid: self.nlp.explain_match(q, self.idx[iid]) or self.nlp.top_keywords(self.idx[iid]))

    def similar_items(self, item_id, n=5):
        """Item-to-item content recommendation."""
        i = self.idx[item_id]
        s = pd.Series(self.item_sim[i], index=self.item_ids).drop(item_id)
        v = self.nlp.matrix[i]
        return self._format(s, n, lambda iid: self.nlp.explain_match(v, self.idx[iid]))


if __name__ == "__main__":
    from nlp_pipeline import NLPPipeline
    items, ratings = load_data()
    rec = Recommender(NLPPipeline().fit(items), ratings)
    print(rec.recommend_query("ancient maritime trade, traditional ceramics, and maps").to_string(index=False))
    for m in METHODS:
        print(f"\n[{m}] U07")
        print(rec.recommend("U07", m).to_string(index=False))