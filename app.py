import re
import io
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()


def extract_pdf_pages(pdf_bytes):
    pages = []
    pdf = PdfReader(io.BytesIO(pdf_bytes))

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()

        if extracted_text:
            pages.append({
                "page": page_number,
                "text": clean_text(extracted_text)
            })

    return pages


def create_chunks_from_pages(pages, chunk_size=80, overlap=20):
    chunks = []

    for page in pages:
        words = page["text"].split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])

            if chunk_text.strip():
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

    return chunks


def semantic_search(question, chunks, model, top_k=5):
    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = model.encode(chunk_texts)
    question_embedding = model.encode([question])

    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": similarities[index]
        })

    return results


st.set_page_config(page_title="Study Chatbot")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

st.title("Study Chatbot")
st.write("Upload your study notes and ask questions.")

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["txt", "pdf"]
)

text = ""
pages = []
pdf_bytes = None

if uploaded_file:

    if uploaded_file.type == "text/plain":
        text = clean_text(uploaded_file.read().decode("utf-8"))
        pages = [{"page": 1, "text": text}]

    elif uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        pages = extract_pdf_pages(pdf_bytes)
        text = " ".join([page["text"] for page in pages])

    st.subheader("Uploaded File")
    st.info(f"File uploaded: {uploaded_file.name}")

    if uploaded_file.type == "application/pdf" and pdf_bytes:
        st.download_button(
            label="Open / Download PDF",
            data=pdf_bytes,
            file_name=uploaded_file.name,
            mime="application/pdf"
        )

        st.subheader("PDF Preview")

        first_page = PdfReader(io.BytesIO(pdf_bytes)).pages[0]

        page_width = float(first_page.mediabox.width)
        page_height = float(first_page.mediabox.height)

        ratio = page_width / page_height

        # Portrait PDFs (A4 books, papers)
        if ratio < 1:
            viewer_width = 500
            viewer_height = 850

        # Landscape PDFs (slides)
        else:
            viewer_width = 1000
            viewer_height = 650

        st.pdf(pdf_bytes, height=800)

    if pages:
        page_numbers = [page["page"] for page in pages]

        selected_page = st.sidebar.number_input(
            "Jump to page",
            min_value=1,
            max_value=len(page_numbers),
            value=1,
            step=1
        )

        selected_page_text = ""

        for page in pages:
            if page["page"] == selected_page:
                selected_page_text = page["text"]
                break

        st.subheader(f"Page {selected_page} Extracted Text")

        st.text_area(
            "Page Content",
            selected_page_text,
            height=250
        )

    with st.expander("Show all extracted text"):
        st.text_area(
            "Extracted Content",
            text,
            height=300
        )

    question = st.text_input(
        "Ask a question about your notes:"
    )

    if question:

        chunks = create_chunks_from_pages(
            pages,
            chunk_size=80,
            overlap=20
        )

        results = semantic_search(
            question,
            chunks,
            model,
            top_k=5
        )

        st.success("Most relevant results:")

        found_result = False

        for index, result in enumerate(results):

            if result["score"] > 0.45:
                found_result = True

                st.markdown(
                    f"""
### Page {result["page"]}

**Similarity:** {result["score"]:.2f}

> {result["text"]}
"""
                )

                st.button(
                    f"Go to page {result['page']}",
                    key=index
                )

        if not found_result:
            st.warning("No strong semantic match found.")