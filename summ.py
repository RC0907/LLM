import streamlit as st
import base64
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# Set page config at the very beginning
st.set_page_config(layout="wide")

# Constants
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
DATA_DIR = Path("data")

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model, device

# File preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return [text.page_content for text in texts]

# Summarization function
def summarize_text(text, tokenizer, model, device, max_length=100):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=250, truncation=True).to(device)
    summary_ids = model.generate(
        inputs,
        max_length=max_length,  # Reduced from 150 to 100
        min_length=30,  # Adjust if needed
        length_penalty=3.0,  # Increased from 2.0 to 3.0 for shorter summaries
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# LLM pipeline
@st.cache_data(show_spinner=False)
def llm_pipeline(filepath, _tokenizer, _model, _device):  # Note the leading underscores
    texts = file_preprocessing(filepath)
    summaries = []
    for text in tqdm(texts, desc="Summarizing chunks"):
        summary = summarize_text(text, _tokenizer, _model, _device)
        summaries.append(summary)
    return " ".join(summaries)

# Function to display the PDF
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main function
def main():
    st.title("Document Summarization App")

    # Load model
    tokenizer, model, device = load_model()

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        if st.button("Summarize"):
            # Create a safe filename
            safe_filename = "".join([c for c in uploaded_file.name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            filepath = DATA_DIR / safe_filename

            # Save uploaded file
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())

            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Uploaded File")
                displayPDF(str(filepath))
            
            with col2:
                with st.spinner("Summarizing... This may take a few minutes."):
                    try:
                        summary = llm_pipeline(str(filepath), tokenizer, model, device)
                        st.info("Summarization Complete")
                        st.success(summary)
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {str(e)}")

if __name__ == "__main__":
    main()
