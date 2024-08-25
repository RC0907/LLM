Research Paper Summarization App

Overview

The Research Paper Summarization App is a web application that allows users to upload PDF files of research papers and receive a concise summary of the content. It uses advanced natural language processing techniques to extract and summarize the text from the uploaded documents.

Features

PDF Upload: Upload a PDF file of a research paper.
Text Extraction: Extracts text from the uploaded PDF.
Text Summarization: Generates a summary of the extracted text using a pre-trained model.
Error Handling: Provides informative messages if something goes wrong during processing.
How It Works

Upload PDF: Use the file uploader to choose and upload a PDF file.
Extract Text: The app reads and extracts text from the PDF.
Generate Summary: The extracted text is then summarized using a smaller, efficient machine learning model.
View Summary: The generated summary is displayed for you to review.
Technologies Used

Streamlit: A framework for building interactive web apps.
PyPDF2: A library for reading and extracting text from PDF files.
Transformers: A library by Hugging Face for working with machine learning models, specifically used here for text summarization.
Pre-trained Model: The app uses the facebook/bart-base model for generating summaries.
