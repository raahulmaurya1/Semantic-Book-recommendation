import os
import pandas as pd
import numpy as np
import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import time

# Set page config
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")

# Load secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]
huggingface_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

@st.cache_data
def load_books():
    books_df = pd.read_csv("books_with_emotions.csv")
    books_df["large_thumbnail"] = books_df["thumbnail"].fillna("") + "&fife=w800"
    books_df["large_thumbnail"] = np.where(
        books_df["large_thumbnail"].isna(),
        "cover_not_found.jpg",
        books_df["large_thumbnail"],
    )
    books_df["small_thumbnail"] = books_df["thumbnail"].fillna("") + "&fife=w200"
    return books_df

books = load_books()

def batch_embed(documents, embeddings, batch_size=100):
    vectors = []
    metadatas = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        embeddings_batch = embeddings.embed_documents([doc.page_content for doc in batch])

        for embedding in embeddings_batch:
            if isinstance(embedding, float):
                raise ValueError("Embedding result is a scalar (float), expected a list or array.")
            vectors.append(embedding)
        metadatas.extend([doc.metadata for doc in batch])

    vectors = np.array(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array of embeddings, got {vectors.ndim}D array.")

    return FAISS.from_embeddings(vectors, documents, embeddings)

@st.cache_resource
def get_or_create_faiss():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index/index.faiss"

    if not os.path.exists(index_path):
        st.info("Creating new FAISS index...")
        loader = UnstructuredFileLoader("tagged_description.txt")
        raw_documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50, separator="\n")
        documents = splitter.split_documents(raw_documents)

        db = batch_embed(documents, embeddings)
        db.save_local("faiss_index")
        st.success("FAISS index created and saved.")
    else:
        try:
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except Exception:
            st.warning("Failed to load FAISS index. Rebuilding it.")
            loader = UnstructuredFileLoader("tagged_description.txt")
            raw_documents = loader.load()
            splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50, separator="\n")
            documents = splitter.split_documents(raw_documents)
            db = batch_embed(documents, embeddings)
            db.save_local("faiss_index")
            st.success("FAISS index rebuilt.")
    return db

db_books = get_or_create_faiss()

def retrieve_semantic_recommendations(query, category=None, tone=None,
                                      initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

# UI
st.title("📚 Semantic Book Recommender")

query = st.text_input("Describe a book you'd love (e.g., 'A story about forgiveness'):")
categories = ["All"] + sorted(books["categories"].dropna().astype(str).unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Category:", categories, index=0)
with col2:
    tone = st.selectbox("Emotional Tone:", tones, index=0)

if st.button("🔍 Recommend Books") and query:
    with st.spinner("Fetching recommendations..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        recs = retrieve_semantic_recommendations(query, category, tone)
        st.subheader("Recommended Books")

        if recs.empty:
            st.warning("No recommendations found. Try a different query.")
        else:
            for _, row in recs.iterrows():
                with st.container():
                    col_img, col_txt = st.columns([1, 5])
                    with col_img:
                        st.image(row["large_thumbnail"], width=500)
                    with col_txt:
                        authors = row["authors"].split(";")
                        if len(authors) == 2:
                            authors_str = f"{authors[0]} and {authors[1]}"
                        elif len(authors) > 2:
                            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
                        else:
                            authors_str = row["authors"]
                        description = " ".join(row["description"].split()[:100]) + "..."
                        st.markdown(f"**{row['title']}** by *{authors_str}*  \n{description}")

    st.success("✔️ Done!")
