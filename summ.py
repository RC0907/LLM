import streamlit as st
import base64
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set page config at the very beginning
st.set_page_config(layout="wide")

# Constants
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
DATA_DIR = Path("data")
CHUNK_SIZE = 500
MAX_CHUNKS = 20  # Limit the number of chunks to process

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# File preprocessing
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Summarization function
def summarize_text(text, tokenizer, model, device, max_length=100):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=250, truncation=True).to(device)
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to limit the summary to a specific word count
def truncate_summary(summary, word_limit=500):
    words = summary.split()
    if len(words) > word_limit:
        summary = " ".join(words[:word_limit]) + "..."
    return summary

# LLM pipeline
def llm_pipeline(text, tokenizer, model, device):
    chunks = split_text(text)
    chunks = chunks[:MAX_CHUNKS]  # Limit the number of chunks
    
    with ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(summarize_text, chunk, tokenizer, model, device): chunk for chunk in chunks}
        summaries = []
        for future in as_completed(future_to_chunk):
            summaries.append(future.result())
    
    full_summary = " ".join(summaries)
    return truncate_summary(full_summary, word_limit=500)

# Function to display the PDF
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main function
def main():
    st.title("Research Paper Summarization App")

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
                with st.spinner("Summarizing... This may take a few moments."):
                    try:
                        text = extract_text_from_pdf(filepath)
                        summary = llm_pipeline(text, tokenizer, model, device)
                        st.info("Summarization Complete")
                        st.success(summary)
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {str(e)}")

if __name__ == "__main__":
    main()