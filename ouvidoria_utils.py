import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_manifestacoes(path: str | Path = "manifestacoes.json") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "data" not in df.columns:
        df["data"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    return df


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    v = _normalize_vectors(np.asarray(vectors, dtype=float))
    return v @ v.T


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: Sequence[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    model = get_model(model_name)
    embeddings = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=float)


def compare_representations(df: pd.DataFrame, pair_ids: Sequence[Tuple[str, str]], model_name: str = DEFAULT_MODEL) -> pd.DataFrame:
    corpus = df["texto"].tolist()
    bow = CountVectorizer(lowercase=True, stop_words=None)
    tfidf = TfidfVectorizer(lowercase=True, stop_words=None)

    bow_matrix = bow.fit_transform(corpus)
    tfidf_matrix = tfidf.fit_transform(corpus)
    emb_matrix = embed_texts(corpus, model_name=model_name)

    sim_bow = cosine_similarity(bow_matrix)
    sim_tfidf = cosine_similarity(tfidf_matrix)
    sim_emb = cosine_similarity_matrix(emb_matrix)

    rows = []
    index_map = {doc_id: idx for idx, doc_id in enumerate(df["id"].tolist())}

    for id_a, id_b in pair_ids:
        i, j = index_map[id_a], index_map[id_b]
        rows.append(
            {
                "par": f"{id_a} × {id_b}",
                "BoW": round(float(sim_bow[i, j]), 4),
                "TF-IDF": round(float(sim_tfidf[i, j]), 4),
                "Embeddings": round(float(sim_emb[i, j]), 4),
            }
        )
    return pd.DataFrame(rows)


def detectar_duplicatas(textos: Sequence[str], limiar: float = 0.85, model_name: str = DEFAULT_MODEL) -> List[dict]:
    if len(textos) < 2:
        return []
    embeddings = embed_texts(textos, model_name=model_name)
    sim = cosine_similarity_matrix(embeddings)
    pares = []
    for i in range(len(textos)):
        for j in range(i + 1, len(textos)):
            score = float(sim[i, j])
            if score >= limiar:
                pares.append(
                    {
                        "i": i,
                        "j": j,
                        "similaridade": round(score, 4),
                        "texto_1": textos[i],
                        "texto_2": textos[j],
                    }
                )
    return pares


def detect_duplicatas(textos: Sequence[str], limiar: float = 0.85, model_name: str = DEFAULT_MODEL) -> List[dict]:
    return detectar_duplicatas(textos, limiar=limiar, model_name=model_name)


def get_longest_manifestations(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.assign(_len=df["texto"].str.len()).sort_values("_len", ascending=False).head(n).drop(columns="_len")


def chunk_text(text: str, chunk_size: int = 220, chunk_overlap: int = 40) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 220, chunk_overlap: int = 40) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["texto"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(chunks):
            rows.append({"id": row["id"], "chunk_index": idx, "chunk": chunk})
    return pd.DataFrame(rows)


def project_embeddings(vectors: np.ndarray, method: str = "pca") -> np.ndarray:
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, max(5, len(vectors) - 1)), random_state=42, init="pca")
        return reducer.fit_transform(vectors)
    reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(vectors)


def create_similarity_heatmap(similarity_matrix: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(similarity_matrix, index=labels, columns=labels)
