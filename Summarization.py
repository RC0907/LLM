import streamlit as st

# Setting Page Config
st.set_page_config(page_title="Research Paper Summarizer", page_icon="📚", layout="wide")

import PyPDF2
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import torch
from streamlit_lottie import st_lottie
import requests

# Custom CSS to enhance the app's appearance
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        color: #333;  /* Set a darker text color for contrast */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stRadio>label {
        background-color: #e1e1e1;
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stRadio>label:hover {
        background-color: #d1d1d1;
    }
    .css-1v0mbdj.etr89bj1 {
        display: block;
        margin-left: auto;
        margin-right: auto;
        min-width: 180px;
    }
    /* Add styling for headers and paragraphs to ensure readability */
    h1, h2, h3, h4, h5, h6 {
        color: #333; /* Dark text color for headers */
    }
    p {
        color: #333; /* Dark text color for paragraphs */
    }
</style>
""", unsafe_allow_html=True)

# Lottie Animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_1a8dx7zj.json')

# Mapping quality options to model names with descriptions
model_mapping = {
    "Low Quality (Fast)": {
        "model_name": "t5-small",
        "description": "Fast and efficient but may sacrifice some accuracy."
    },
    "Medium Quality (Balanced)": {
        "model_name": "t5-base",
        "description": "A balanced option providing good accuracy and speed."
    },
    "High Quality (Accurate)": {
        "model_name": "t5-large",
        "description": "High accuracy but may be slower and require more resources."
    }
}

# Loading Model
@st.cache_resource
def Load_model(model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Extracting Text
def ExtractingText(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Cleaning Text
def clean_text(text):
    # Remove references
    text = re.sub(r'\[\d+\]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Summarization
def Summarizer(text, model, tokenizer, max_length=250, chunk_size=1000, stride=200):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - stride)]
    summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        inputs_text = "summarize: " + chunk
        input_ids = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).input_ids
        summary_ids = model.generate(input_ids,
            max_length=max_length,
            min_length=40,
            num_beams=4,
            repetition_penalty=1.5,
            temperature=0.7,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        progress_bar.progress((i + 1) / len(chunks))
    final_summary = " ".join(summaries)
    return final_summary

# Two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("📚 Research Paper Summarization")
    st.markdown("---")

    # Description and Model Quality Options
    st.markdown("""
    **Welcome to the Research Paper Summarization Tool!**
    
    This tool allows you to upload a PDF document and get a concise summary. Choose a model quality option based on your needs:
    
    - **Low Quality (Fast)**: Quick summarization with lower accuracy. Ideal for fast results.
    - **Medium Quality (Balanced)**: Provides a good balance between speed and accuracy.
    - **High Quality (Accurate)**: Detailed and accurate summaries but may take more time and resources.
    
    Please choose the model quality option below to get started.
    """)
    st.markdown("---")

    # Model selection in terms of quality
    quality = st.radio(
        "Choose Model Quality:",
        list(model_mapping.keys()),
        index=1
    )
    st.markdown("---")

    # Display selected model description
    model_info = model_mapping[quality]
    st.info(f"**Selected Model:** {quality}\n\n{model_info['description']}")
    st.markdown("---")

    # Get the model name based on user selection
    model_name = model_info['model_name']
    tokenizer, model = Load_model(model_name)

    uploaded_file = st.file_uploader("Choose PDF file to Upload", type="pdf")
    
    if uploaded_file is not None:
        text = ExtractingText(uploaded_file)
        cleaned_text = clean_text(text)
        if st.button("Summarize"):
            with st.spinner("Generating Summary..."):
                summary = Summarizer(cleaned_text, model, tokenizer, max_length=250)
            st.success("Summary generated successfully!")
            st.text_area("Summary", summary, height=400)
            
            # Download button for the summary
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="research_summary.txt",
                mime="text/plain"
            )

with col2:
    st_lottie(lottie_book, height=300, key="book")
    
    st.markdown("### How it works")
    st.markdown("""
    1. **Upload** your research paper in PDF format.
    2. **Choose** the quality of summarization you need.
    3. Click **Summarize** to generate a concise summary.
    4. **Download** the summary for later use.
    """)
    
    st.markdown("### Tips for best results")
    st.markdown("""
    - Ensure your PDF is text-based, not scanned images.
    - For longer papers, consider using the high-quality option.
    - The summary length is optimized for readability.
    """)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Rahul Chauhan | [GitHub](https://github.com/RC0907)")
