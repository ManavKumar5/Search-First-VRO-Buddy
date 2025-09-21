# search_api.py
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.idx")
META_PATH = os.path.join(DATA_DIR, "catalog_meta.pkl")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"


class SearchEngine:
    def __init__(self):
        print("Loading models and index...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.df = pickle.load(f).reset_index(drop=True)
        self.tfidf = load(TFIDF_PATH)
        # precompute tfidf matrix for all docs to speed-up scoring for candidates
        self._doc_tfidf = self.tfidf.transform(self.df["text"].tolist())
        print("Loaded.")

    def _normalize(self, arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn == 0:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    def search(
        self,
        query,
        top_k=10,
        alpha=0.6,
        category=None,
        min_price=None,
        max_price=None,
        personalization_boost=None,
    ):
        """
        query: string
        top_k: number of results
        alpha: weight given to vector score (vs tfidf score)
        personalization_boost: dict(category->click_count) to boost certain categories
        """
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        D, I = self.index.search(
            q_emb.astype("float32"), top_k * 5
        )  # retrieve more candidates
        candidate_idxs = I[0]
        candidate_scores = D[0]

        # compute tfidf similarity for query vs candidates
        q_tfidf = self.tfidf.transform([query])
        candidate_tfidf = self._doc_tfidf[candidate_idxs]
        tfidf_sims = cosine_similarity(q_tfidf, candidate_tfidf)[0]

        # normalize scores
        vec_norm = self._normalize(candidate_scores)
        tfidf_norm = self._normalize(tfidf_sims)

        final_scores = alpha * vec_norm + (1 - alpha) * tfidf_norm

        results = []
        for idx, score in zip(candidate_idxs, final_scores):
            row = self.df.iloc[idx].to_dict()
            row["raw_vec_score"] = float(
                self._normalize([self.df.index.get_loc(idx) if False else score])[0]
            )  # placeholder
            row["score"] = float(score)
            results.append(row)

        # apply filters
        if category and category.lower() != "all":
            results = [
                r
                for r in results
                if str(r.get("category", "")).lower() == str(category).lower()
            ]
        if min_price is not None:
            results = [
                r
                for r in results
                if r.get("price") is not None
                and r.get("price") != ""
                and float(r.get("price")) >= float(min_price)
            ]
        if max_price is not None:
            results = [
                r
                for r in results
                if r.get("price") is not None
                and r.get("price") != ""
                and float(r.get("price")) <= float(max_price)
            ]

        # personalization boost (simple multiplicative boosting by category click count)
        if personalization_boost:
            for r in results:
                cat = r.get("category")
                boost = personalization_boost.get(cat, 0)
                r["score"] = r["score"] * (1 + 0.1 * boost)

        # sort and return top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# Optional quick test
if __name__ == "__main__":
    engine = SearchEngine()
    q = "blue sneakers under 5000"
    res = engine.search(q, top_k=5)
    for r in res:
        print(r["title"], r["price"], r["category"], r["score"])
