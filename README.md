# Interactive-Document-Analysis-Chatbot-leveraging-NLP
Overview:
This project is a web-based chatbot designed to extract text from PDF documents, generate embeddings for text chunks, and respond to user queries using GPT-2. The system is built using Flask, with PDFMiner for text extraction, SentenceTransformer for embedding generation, and ngrok for tunneling.

Features:
Extracts and processes text from PDF files.
Generates vector embeddings using sentence-transformers/paraphrase-distilroberta-base-v1.
Provides contextually relevant responses to user queries using GPT-2.
User-friendly interface with a chatbox for query input and response display.
Requirements:
Flask
PDFMiner
SentenceTransformer
Transformers (Hugging Face)
pyngrok
How to Run:
Clone the repository.
Install the required Python packages using pip install -r requirements.txt.
Set your ngrok authtoken.
Run the application: python app.py.
Access the chatbot via the ngrok-provided URL.
Usage:
Upload a PDF to extract and chunk text.
Ask questions related to the PDF content.
Receive accurate, context-based responses from the chatbot.
