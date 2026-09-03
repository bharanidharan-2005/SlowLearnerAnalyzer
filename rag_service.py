import sqlite3
import json
import os
import faiss
import numpy as np
import pickle
import requests  # NEW: Used to call the Hugging Face API
from openai import OpenAI
from dotenv import load_dotenv

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
    instead of using local server memory.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is missing. Please add it to your .env file and Render.")
        
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Send the batch of text chunks to Hugging Face
    response = requests.post(api_url, headers=headers, json={"inputs": texts})
    
    if response.status_code != 200:
        raise Exception(f"Hugging Face API Error: {response.status_code} - {response.text}")
        
    # The API returns a list of lists. FAISS strictly requires a 32-bit float NumPy array.
    return np.array(response.json(), dtype=np.float32)


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
        
        # Create a descriptive text chunk for the LLM context window
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
    dimension = embeddings.shape[1] # The size of the vector array (384 for MiniLM)
    index = faiss.IndexFlatL2(dimension) # L2 measures the mathematical distance between vectors
    index.add(embeddings)

    # Save the FAISS index and the text metadata to disk
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

    # 1. Convert the user's question into a mathematical vector via API
    try:
        question_embedding = get_embeddings([user_question])
    except Exception as e:
        return f"Embedding API Error: {str(e)}"

    # 2. Search the Vector Database for the top 5 most relevant student records
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)

    # Retrieve distances and index numbers of the top 5 matches
    distances, indices = index.search(question_embedding, k=5)
    
    # 3. Build the Context (The exact student records the AI must read)
    retrieved_context = ""
    sources = []
    for i in indices[0]:
        if i != -1: # -1 means no match was found
            retrieved_context += metadata[i]["text"] + "\n\n"
            sources.append(metadata[i]["name"])

    # 4. Construct the Prompt for the LLM
    system_prompt = f"""
    You are an intelligent Academic Assistant for Mount Zion College faculty. 
    Answer the faculty's question using ONLY the provided student context below.
    Do not invent marks, names, or statistics. 
    If the context does not contain the answer, politely state that you cannot find the information in the current database.
    
    RETRIEVED CONTEXT:
    {retrieved_context}
    """

    # 5. Send to Groq LLM
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
        
        # Add source citations to build trust
        unique_sources = list(set(sources))
        if unique_sources:
            citation = f"\n\n📚 *Sources: Student Records ({', '.join(unique_sources)})*"
            return answer + citation
        else:
            return answer
            
    except Exception as e:
        return f"LLM API Error: Check your API Key and internet connection. Details: {str(e)}"