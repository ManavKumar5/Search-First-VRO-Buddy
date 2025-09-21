# build_index.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pickle
from joblib import dump

DATA_DIR = "data"
CATALOG_CSV = os.path.join(DATA_DIR, "catalog_sample.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.idx")
META_PATH = os.path.join(DATA_DIR, "catalog_meta.pkl")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_catalog(csv_path=CATALOG_CSV):
    df = pd.read_csv(csv_path)
    # ensure columns
    for c in ["product_id", "title", "description", "category", "price", "image_url"]:
        if c not in df.columns:
            df[c] = ""
    # create a combined text field
    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(
        str
    )
    return df


def build_embeddings(df, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    texts = df["text"].tolist()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # normalize for cosine similarity with IndexFlatIP
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs = embs / norms
    return embs


def build_faiss_index(embs, faiss_path=FAISS_INDEX_PATH):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product with normalized vectors => cosine
    index.add(embs.astype("float32"))
    faiss.write_index(index, faiss_path)
    print(f"Wrote FAISS index to {faiss_path}")


def build_tfidf(df, tfidf_path=TFIDF_PATH):
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    vect.fit(df["text"].tolist())
    dump(vect, tfidf_path)
    print(f"Saved TF-IDF vectorizer to {tfidf_path}")
    return vect


def save_metadata(df, meta_path=META_PATH):
    with open(meta_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_catalog()
    print(f"Loaded catalog with {len(df)} products")
    embs = build_embeddings(df)
    build_faiss_index(embs)
    build_tfidf(df)
    save_metadata(df)
    print("All done. Run the app with: streamlit run app.py")
