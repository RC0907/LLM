import streamlit as st
from transformers import BartTokenizerFast, BartForConditionalGeneration
import PyPDF2
import io
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the smaller model
model_name = "facebook/bart-base"
tokenizer = BartTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Limit the number of pages to process
PAGE_LIMIT = 5
CHUNK_SIZE = 512  # Number of tokens per chunk

def extract_text_from_pdf(file, page_limit=PAGE_LIMIT):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            if i >= page_limit:
                break
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=CHUNK_SIZE, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = tokenizer.tokenize(text)
    for i in range(0, len(tokens), chunk_size):
        yield tokenizer.convert_tokens_to_string(tokens[i:i + chunk_size])

def main():
    st.title("Research Paper Summarization App")
    st.write(f"Upload a research paper (PDF) to get a summary. The first {PAGE_LIMIT} pages will be processed.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            summary = ""
            with st.spinner("Generating summary..."):
                for chunk in chunk_text(text):
                    summary += summarize_text(chunk) + " "
                
            st.write("### Summary")
            st.write(summary)
        else:
            st.error("Unable to extract text from the PDF.")

if __name__ == "__main__":
    main()
