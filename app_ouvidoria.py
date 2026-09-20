from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from ouvidoria_utils import DEFAULT_MODEL, cosine_similarity_matrix, embed_texts, load_manifestacoes, project_embeddings


@st.cache_data
def get_dataset():
    return load_manifestacoes("manifestacoes.json")


@st.cache_resource
def get_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@st.cache_data
def get_embeddings(df: pd.DataFrame, model_name: str):
    return embed_texts(df["texto"].tolist(), model_name=model_name)


def color_for_score(score: float) -> str:
    if score > 0.7:
        return "🟢"
    if score > 0.5:
        return "🟡"
    return "🔴"


def build_similarity_score(query: str, df: pd.DataFrame, model_name: str, top_k: int):
    model = get_model(model_name)
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    corpus_vectors = get_embeddings(df, model_name)
    sims = cosine_similarity(query_vector, corpus_vectors)[0]
    ranked = df.copy()
    ranked["score"] = sims
    ranked = ranked.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    ranked["indicador"] = ranked["score"].apply(color_for_score)
    return ranked


st.set_page_config(page_title="Ouvidoria Inteligente", layout="wide")

st.title("Ouvidoria Inteligente — Triagem Semântica")

with st.sidebar:
    st.header("Configurações")
    selected_model = st.selectbox("Modelo de embedding", [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2",
    ])
    top_k = st.slider("Top-k", min_value=3, max_value=10, value=5)


df = get_dataset()


tab1, tab2, tab3, tab4 = st.tabs(["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"])

with tab1:
    query = st.text_area("Descreva o problema do cidadão:", height=150, placeholder="Ex.: falta de iluminação em praça pública e buraco enorme na rua")
    if st.button("Buscar manifestações similares"):
        if query.strip():
            results = build_similarity_score(query, df, selected_model, top_k)
            st.dataframe(
                results[["id", "categoria_oficial", "score", "texto", "indicador"]].style.format({"score": "{:.3f}"}),
                use_container_width=True,
            )
        else:
            st.warning("Digite uma manifestação para iniciar a busca.")

with tab2:
    st.subheader("Base completa")
    show_matrix = st.button("Gerar matriz de similaridade")
    st.dataframe(df, use_container_width=True)
    if show_matrix:
        vectors = get_embeddings(df, selected_model)
        matrix = cosine_similarity_matrix(vectors)
        pd.options.display.float_format = "{:.3f}".format
        st.dataframe(pd.DataFrame(matrix, index=df["id"], columns=df["id"]), use_container_width=True)

with tab3:
    st.subheader("Espaço vetorial 2D")
    method = st.radio("Método de redução", ["pca", "tsne"], horizontal=True)
    vectors = get_embeddings(df, selected_model)
    coords = project_embeddings(vectors, method=method)
    df_plot = df.copy()
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]

    st.scatter_chart(
        df_plot[["x", "y", "categoria_oficial"]].assign(label=df_plot["id"]),
        x="x",
        y="y",
        color="categoria_oficial",
    )
    st.caption("Os clusters semânticos nem sempre coincidem perfeitamente com as categorias oficiais, especialmente quando há duplicatas e temas cruzados.")

with tab4:
    st.subheader("Chunking de manifestação longa")
    long_text = st.text_area("Cole uma manifestação longa:", height=220, placeholder="Texto longo...")
    chunk_size = st.slider("Tamanho do chunk", min_value=80, max_value=500, value=220)
    overlap = st.slider("Overlap", min_value=0, max_value=120, value=40)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if st.button("Gerar chunks") and long_text.strip():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(long_text)
        embed = embed_texts(chunks, model_name=selected_model)
        coords = project_embeddings(embed, method="pca")
        chunk_df = pd.DataFrame({"chunk": chunks, "x": coords[:, 0], "y": coords[:, 1]})
        st.dataframe(chunk_df, use_container_width=True)
        st.scatter_chart(chunk_df[["x", "y"]], x="x", y="y")

st.caption("Protótipo de ouvidoria inteligente com busca semântica, clusterização e chunking.")
