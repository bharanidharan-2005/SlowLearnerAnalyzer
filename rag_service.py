import sqlite3
import json
import os
import faiss
import numpy as np
import pickle
import requests  
import time  
import socket # NEW: Imported to patch the cloud network
from openai import OpenAI
from dotenv import load_dotenv

# --- CLOUD INFRASTRUCTURE PATCH ---
# Render's free tier frequently suffers from broken IPv6 DNS resolution.
# This overrides Python's default socket behavior to strictly use IPv4,
# completely bypassing the '[Errno -5] No address associated with hostname' crash.
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [res for res in responses if res[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ----------------------------------

# Load environment variables (API Key from .env file)
load_dotenv()

# Initialize LLM Client to use Groq's free API
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://api.groq.com/openai/v1" 
)

# Define where to save our vector database
VECTOR_DIR = "vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(VECTOR_DIR, "students.index")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")

def get_embeddings(texts):
    """
    Generates vector embeddings by calling the free Hugging Face API
    with automatic network retries for cloud stability.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is missing. Please add it to your .env file and Render.")
        
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Try to connect 3 times before giving up
    for attempt in range(3):
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": texts}, timeout=15)
            
            if response.status_code != 200:
                raise Exception(f"Hugging Face API Error: {response.status_code} - {response.text}")
                
            return np.array(response.json(), dtype=np.float32)
            
        except requests.exceptions.ConnectionError:
            if attempt == 2: 
                raise Exception("Network failure: Could not connect to Hugging Face after 3 attempts.")
            print(f"Network glitch detected. Retrying in 2 seconds... (Attempt {attempt + 1})")
            time.sleep(2)


def build_vector_db():
    """
    ETL PIPELINE:
    1. EXTRACT data from SQLite
    2. TRANSFORM into text chunks and vectors
    3. LOAD into FAISS vector database
    """
    # 1. EXTRACT
    conn = sqlite3.connect("instance/students.db")
    c = conn.cursor()
    c.execute("SELECT register_no, student_name, marks_json, failed_subjects, total, average, status FROM student_performance")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return False

    texts = []
    metadata = []

    # 2. TRANSFORM
    for row in rows:
        reg_no, name, marks_json, fails, total, avg, status = row
        marks = json.loads(marks_json)
        
        chunk = f"Student Name: {name}. Register Number: {reg_no}. "
        chunk += f"Overall Status: {status}. Total Marks: {total}. Average: {avg}%. "
        if status == "Fail":
            chunk += f"This student failed in the following subjects: {fails}. "
        
        chunk += "Subject-wise performance: " + ", ".join([f"{sub}: {m}" for sub, m in marks.items()]) + "."
        
        texts.append(chunk)
        metadata.append({
            "name": name,
            "reg_no": reg_no,
            "text": chunk
        })

    # Convert text chunks to vector embeddings using the external API
    print(f"Generating vector embeddings for {len(texts)} student records via Hugging Face API...")
    embeddings = get_embeddings(texts)

    # 3. LOAD (Save to FAISS)
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    print("FAISS Vector database built successfully!")
    return True

def ask_ai(user_question):
    """
    RAG RETRIEVAL PIPELINE:
    1. Embed user question
    2. Search FAISS vector DB for closest matches
    3. Send retrieved data + question to LLM
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return "Error: No data found. Please upload a student report first."

    try:
        question_embedding = get_embeddings([user_question])
    except Exception as e:
        return f"Embedding API Error: {str(e)}"

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)

    distances, indices = index.search(question_embedding, k=5)
    
    retrieved_context = ""
    sources = []
    for i in indices[0]:
        if i != -1: 
            retrieved_context += metadata[i]["text"] + "\n\n"
            sources.append(metadata[i]["name"])

    system_prompt = f"""
    You are an intelligent Academic Assistant for Mount Zion College faculty. 
    Answer the faculty's question using ONLY the provided student context below.
    Do not invent marks, names, or statistics. 
    If the context does not contain the answer, politely state that you cannot find the information in the current database.
    
    RETRIEVED CONTEXT:
    {retrieved_context}
    """

    try:
        response = client.chat.completions.create(
          model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.1 
        )
        answer = response.choices[0].message.content
        
        unique_sources = list(set(sources))
        if unique_sources:
            citation = f"\n\n📚 *Sources: Student Records ({', '.join(unique_sources)})*"
            return answer + citation
        else:
            return answer
            
    except Exception as e:
        return f"LLM API Error: Check your API Key and internet connection. Details: {str(e)}"