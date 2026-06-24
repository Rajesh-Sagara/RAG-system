from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load models once at startup
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def ask_question(question):
    docs = db.similarity_search_with_score(question, k=3)
    relevant_docs = [doc for doc, score in docs if score < 2.0]
    
    if not relevant_docs:
        return "I'm here to answer questions about the company. Please ask something related to our services."

    context = "\n".join([d.page_content for d in relevant_docs])
    prompt = f"""
    You are a helpful assistant for a company website.

    Answer ONLY using the provided context.

    Rules:
    - If the question is unrelated to the context, say you cannot answer.
    - Do not make up information.
    - Keep answers short and professional.
    Context:
    {context}

    Question:
    {question}
    """
    return generate_answer(prompt)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'answer': 'Please ask a valid question.'})
    
    answer = ask_question(question)
    return jsonify({'answer': answer})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)
