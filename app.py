# Import necessary libraries
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from pyngrok import ngrok  # Ngrok for tunneling

# Initialize Flask application
app = Flask(__name__)

# Set your Ngrok authtoken here
ngrok.set_auth_token("2iS0H0DngQX0wU4gSHdf002mwvX_YQRkM5SRBaB2kuG2F8eS")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Function to chunk text
def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

# Function to generate embeddings
def generate_embeddings(chunks, model_name='sentence-transformers/paraphrase-distilroberta-base-v1'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings, model

# Function to save embeddings and chunks to a file
def save_embeddings_and_chunks(embeddings, chunks, embeddings_path='embeddings.pkl', chunks_path='chunks.pkl'):
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

# Function to load embeddings and chunks from a file
def load_embeddings_and_chunks(embeddings_path='embeddings.pkl', chunks_path='chunks.pkl'):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return embeddings, chunks

# Function to query the chatbot
def query_chatbot(query, embeddings, chunks, model, tokenizer, hf_model):
    # Generate embedding for the query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    # Calculate cosine similarity between query embedding and all document embeddings
    similarities = cosine_similarity(query_embedding, embeddings)
    most_similar_index = np.argmax(similarities)

    # Retrieve the corresponding chunk
    retrieved_chunk = chunks[most_similar_index]

    # Prepare the prompt for the LLM
    prompt = f"Answer the question based on the following context:\n\n{retrieved_chunk}\n\nQuestion: {query}\nAnswer:"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)

    # Generate the answer using the Hugging Face model with adjusted parameters
    outputs = hf_model.generate(**inputs, max_length=1024, num_return_sequences=1, temperature=0.7, top_k=50)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove repeated sentences
    answer = remove_repeated_sentences(answer)

    return answer

# Function to remove repeated sentences
def remove_repeated_sentences(text):
    sentences = text.split('. ')
    seen = set()
    result = []

    for sentence in sentences:
        if sentence.strip() not in seen:
            seen.add(sentence.strip())
            result.append(sentence.strip())

    return '. '.join(result)

# Route for home page and chatbot interface
@app.route('/')
def home():
    return render_template_string("""
   <!DOCTYPE html>
   <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chatbot</title>

            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background: rgb(38, 51, 61);
                    background: -webkit-linear-gradient(to right, rgb(38, 51, 61), rgb(50, 55, 65), rgb(33, 33, 78));
                    background: linear-gradient(to right, rgb(38, 51, 61), rgb(50, 55, 65), rgb(33, 33, 78));
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    width: 100%;
                    max-width: 600px;
                    background-color: rgba(0, 0, 0, 0.4);
                    border-radius: 8px;
                    box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }
                 .chatbox {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
        }
        /* Customize scrollbar for WebKit browsers */
        .chatbox::-webkit-scrollbar {
            width: 12px;
        }

        .chatbox::-webkit-scrollbar-thumb {
            background-color: white; /* Scrollbar thumb color */
            border-radius: 6px; /* Rounded corners */
        }

        .chatbox::-webkit-scrollbar-track {
            background: transparent; /* Transparent background */
        }

        /* Customize scrollbar for Firefox */
        .chatbox {
            scrollbar-width: thin;
            scrollbar-color: white transparent;
        }

                .message {
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 8px;
                    max-width: 80%;
                }
                .user {
                    background-color: #7742ac;
                    color: #fff;
                    align-self: flex-end;
                    text-align: right;
                    margin-left: auto;
                }
                .bot {
                    background-color: rgba(0, 0, 0, 0.3);
                    color: #fff;
                    align-self: flex-start;
                    text-align: left;
                    margin-right: auto;
                }
                .input-group {
                    display: flex;
                    background-color: rgba(0, 0, 0, 0.3);
                    padding: 10px;
                }
                .input-group input {
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px 0 0 5px;
                    outline: none;
                    background-color: rgba(0, 0, 0, 0.3);
                    color: white;

                }
                .input-group button {
                    padding: 12px 20px; /* Adjusted padding */
                    background-color: transparent; /* Transparent background */
                    border: none; /* No border */
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }
                .input-group button svg {
                    width: 24px; /* Adjust size of the SVG icon */
                    height: 24px;
                    fill: #7742ac; /* Color of the arrow and circle */
                }
                .input-group button.loading {
                    pointer-events: none; /* Disable clicking during loading */
                }
                .input-group button.loading svg {
                    display: none; /* Hide SVG icon during loading */
                }
                .input-group button.loading::after {
                    content: "";
                    display: inline-block;
                    width: 24px; /* Adjust size of loading animation */
                    height: 24px;
                    border: 2px solid #7742ac; /* Loading animation color */
                    border-radius: 50%; /* Rounded shape */
                    border-top-color: transparent; /* Hide top part of the border */
                    border-right-color: transparent; /* Hide right part of the border */
                    animation: spin 1s linear infinite; /* Animation effect */
                    margin-left: 10px; /* Adjust spacing */
                    vertical-align: middle;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .input-group button:hover svg {
                    fill: #7742ac; /* Darker blue on hover */
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="chatbox" id="chatbox"></div>
                <div class="input-group">
                    <input type="text" id="query-input" placeholder="Type your message..." autocomplete="off">
                    <button id="send-btn">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="10" stroke="#007bff" stroke-width="2" fill="none" />
                            <path stroke="#58cc71" stroke-width="2" d="M12 8v8M8 12l4-4 4 4" />
                        </svg>
                    </button>
                </div>
            </div>

            <script>
                document.getElementById('send-btn').addEventListener('click', async () => {
                    const queryInput = document.getElementById('query-input');
                    const query = queryInput.value.trim();
                    if (query === '') return;

                    const chatbox = document.getElementById('chatbox');

                    // Add user message to chatbox
                    const userMessage = document.createElement('div');
                    userMessage.classList.add('message', 'user');
                    userMessage.textContent = query;
                    chatbox.appendChild(userMessage);
                    queryInput.value = '';

                    // Disable the send button while processing
                    const sendBtn = document.getElementById('send-btn');
                    sendBtn.classList.add('loading');


                    // Send the query to the backend
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query })
                    });
                    const data = await response.json();

                    // Add bot response to chatbox
                    const botMessage = document.createElement('div');
                    botMessage.classList.add('message', 'bot');
                    botMessage.textContent = data.answer;
                    chatbox.appendChild(botMessage);

                    // Enable the send button
                    sendBtn.classList.remove('loading');

                    // Scroll to the bottom of the chatbox
                    chatbox.scrollTop = chatbox.scrollHeight;
                });
            </script>
        </body>
        </html>


    """)

# Route for querying the chatbot
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data['query']
    answer = query_chatbot(query, embeddings, chunks, model, hf_tokenizer, hf_model)
    return jsonify({'answer': answer})

# Main function to run the Flask app
def main():
    global embeddings, chunks, model, hf_tokenizer, hf_model

    # Path to PDF file and pickle files
    pdf_path = 'data/b.pdf'
    embeddings_path = 'embeddings.pkl'
    chunks_path = 'chunks.pkl'
    recreate_embeddings = True

    # If embeddings and chunks need to be recreated or don't exist, recreate them
    if recreate_embeddings or not os.path.exists(embeddings_path) or not os.path.exists(chunks_path):
        # Step 1: Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)

        # Step 2: Chunk the text
        chunks = chunk_text(pdf_text)

        # Step 3: Generate embeddings
        embeddings, model = generate_embeddings(chunks)

        # Step 4: Save embeddings and chunks
        save_embeddings_and_chunks(embeddings, chunks, embeddings_path, chunks_path)
    else:
        # Step 5: Load embeddings and chunks
        embeddings, chunks = load_embeddings_and_chunks(embeddings_path, chunks_path)
        # Load the model
        model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')

    # Step 6: Load the Hugging Face model and tokenizer
    global hf_model, hf_tokenizer
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    hf_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Establish a tunnel to the Flask app on port 5000 using Ngrok
    public_url = ngrok.connect(5000)
    print(' * Tunnel URL:', public_url)

    # Start the Flask app
    app.run(port=5000)

if __name__ == '__main__':
    main()
