import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import PyPDF2
import io
import time

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=500, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    # Set the page configuration
    st.set_page_config(page_title="Research Paper Summarization App", layout="wide")

    # Header section
    st.title("Research Paper Summarization App")
    st.write("Upload a research paper (PDF) to get a summary using state-of-the-art language models.")

    # File uploader with instructions
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="visible")

    # Process the file if uploaded
    if uploaded_file is not None:
        st.sidebar.header("Processing Status")
        st.write("### Extracting and Summarizing...")
        
        # Show a progress bar for text extraction and summarization
        progress_bar = st.progress(0)
        progress_text = st.empty()  # Placeholder for progress text

        # Simulate text extraction and summarization process (if needed)
        # Update the progress bar
        progress_text.text("Extracting text from PDF...")
        time.sleep(1)  # Simulate a delay for text extraction
        progress_bar.progress(50)  # Update progress to 50%

        # Extract text
        text = extract_text_from_pdf(uploaded_file)
        if text:
            progress_text.text("Generating summary...")
            time.sleep(1)  # Simulate a delay for summarization
            progress_bar.progress(100)  # Update progress to 100%

            # Generate summary
            summary = summarize_text(text)
            
            # Display the summary
            st.write("### Summary")
            st.write(summary)
        else:
            st.error("Unable to extract text from the PDF.")

        # Hide the progress bar and text after completion
        progress_bar.empty()
        progress_text.empty()
    
    # Footer section
    st.sidebar.write("Built with ❤️ using Streamlit and Transformers.")
    st.sidebar.write("© 2024 Research Paper Summarization App")

if __name__ == "__main__":
    main()