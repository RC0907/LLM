import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import PyPDF2
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Use the smaller model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def summarize_text(text):
    try:
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=500, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return "An error occurred while generating the summary."

def main():
    st.title("Research Paper Summarization App")
    st.write("Upload a research paper (PDF) to get a summary.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            with st.spinner("Generating summary..."):
                summary = summarize_text(text)
                
            st.write("### Summary")
            st.write(summary)
        else:
            st.error("Unable to extract text from the PDF.")

if __name__ == "__main__":
    main()
