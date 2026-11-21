"""NLP pipeline: text cleaning, TF-IDF features, keyword extraction, query vectorization.

Used by recommender.py. Only needs numpy, pandas, scikit-learn.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Standard English stop words + a few filler words common in museum placards
EXTRA_STOPWORDS = {"used", "made", "like", "known", "century", "bce", "show", "shows",
                   "many", "one", "also", "long", "small", "large"}
STOPWORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS


# ---------- 1. Text preprocessing ----------
def clean_text(text):
    """Lowercase, remove punctuation/digits, collapse extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Split cleaned text into word tokens."""
    return text.split()


def normalize(token):
    """Very light plural stemming: ships -> ship, maps -> map (keeps 'glass', 'bronze')."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def preprocess(text):
    """Full pipeline: clean -> tokenize -> stop-word removal -> normalize. Returns a string."""
    tokens = [normalize(t) for t in tokenize(clean_text(text))]
    return " ".join(remove_stopwords(tokens))


def build_document(row, tag_weight=2):
    """One exhibit = placard text + theme tags (tags repeated to give them more weight)."""
    tags = str(row["theme_tags"]).replace(",", " ")
    return preprocess(f'{row["placard_text"]} {" ".join([tags] * tag_weight)}')


# ---------- 2. TF-IDF + keywords ----------
class NLPPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self.items = None
        self.matrix = None  # item x term TF-IDF (sparse)

    def fit(self, items_df):
        """Fit vocabulary on the exhibit dataset."""
        self.items = items_df.reset_index(drop=True).copy()
        docs = self.items.apply(build_document, axis=1)
        self.matrix = self.vectorizer.fit_transform(docs)
        self.terms = np.array(self.vectorizer.get_feature_names_out())
        return self

    def transform_text(self, text):
        """Vectorize any new query / description with the SAME fitted vocabulary."""
        return self.vectorizer.transform([preprocess(text)])

    def add_items(self, new_df):
        """Cold-start: add new exhibits using the existing vocabulary (no refit)."""
        new_df = new_df.reset_index(drop=True)
        docs = new_df.apply(build_document, axis=1)
        new_matrix = self.vectorizer.transform(docs)
        from scipy.sparse import vstack
        self.items = pd.concat([self.items, new_df], ignore_index=True)
        self.matrix = vstack([self.matrix, new_matrix]).tocsr()
        return self

    def top_keywords(self, item_idx, n=5):
        """Highest-weighted TF-IDF terms of one exhibit."""
        row = self.matrix[item_idx].toarray().ravel()
        top = row.argsort()[::-1][:n]
        return [self.terms[i] for i in top if row[i] > 0]

    def explain_match(self, query_vec, item_idx, n=5):
        """Terms that contribute most to cosine similarity between a query/profile and an exhibit."""
        q = query_vec.toarray().ravel()
        d = self.matrix[item_idx].toarray().ravel()
        contrib = q * d
        top = contrib.argsort()[::-1][:n]
        return [self.terms[i] for i in top if contrib[i] > 0]


# ---------- 3. Simple sentiment helper for visitor comments ----------
POSITIVE = {"loved", "best", "superb", "beautiful", "fascinating", "interesting", "good",
            "worth", "exactly", "liked", "learned", "pleasant"}
NEGATIVE = {"boring", "skipped", "dry", "hard", "not", "nothing", "too", "didn"}


def sentiment_score(comment):
    """Lexicon sentiment in [-1, 1]: (pos - neg) / (pos + neg)."""
    toks = tokenize(clean_text(comment))
    pos = sum(t in POSITIVE for t in toks)
    neg = sum(t in NEGATIVE for t in toks)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


if __name__ == "__main__":
    items = pd.read_csv("data/items.csv")
    nlp = NLPPipeline().fit(items)
    print("TF-IDF matrix:", nlp.matrix.shape)
    q = nlp.transform_text("ancient maritime trade, traditional ceramics, and maps")
    sims = (nlp.matrix @ q.T).toarray().ravel()
    for i in sims.argsort()[::-1][:5]:
        print(f'{items.item_id[i]} {items.exhibit_title[i]:40s} {sims[i]:.3f}  {nlp.explain_match(q, i)}')
