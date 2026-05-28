import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Study Chatbot")

st.title("Study Chatbot")
st.write("Upload your study notes and ask questions.")

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["txt", "pdf"]
)

text = ""

if uploaded_file:

    # TXT FILE
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    # PDF FILE
    elif uploaded_file.type == "application/pdf":

        pdf = PdfReader(uploaded_file)

        for page in pdf.pages:

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text

    st.subheader("Your Notes")

    st.text_area(
        "Content",
        text,
        height=300
    )

    question = st.text_input(
        "Ask a question about your notes:"
    )

    if question:

        found_sentences = []

        sentences = text.split(".")

        for sentence in sentences:

            if question.lower() in sentence.lower():

                found_sentences.append(sentence.strip())

        if found_sentences:

            st.success("Relevant information found:")

            for result in found_sentences[:5]:

                st.write("- " + result)

        else:

            st.warning("No direct match found.")