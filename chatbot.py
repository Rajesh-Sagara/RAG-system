from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

def load_db():
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def ask_question(question,db):    
    docs = db.similarity_search_with_score(question, k=3)
    ##print(f"DEBUG - Scores: {[score for doc, score in docs]}")
    relevant_docs = [doc for doc, score in docs if score < 1.8]  # Adjust threshold as needed
    if not relevant_docs:
        return "I'm here to answer questions about the company. Please ask something related to our services."


    context = "\n".join([d.page_content for d in relevant_docs])
    prompt = f"""
    You are a helpful assistant for a company website.

    Answer ONLY using the provided context.

    Rules:
    - If the question is unrelated to the context, respond with "I cannot answer that question."
    - Do not make up information.
    - Keep answers short and professional.
    Context:
    {context}

    Question:
    {question}
    """
    return generate_answer(prompt)
def main():
    print("Website chatbot ready. Type 'exit' to quit.\n")
    db=load_db()
    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if question:
            response = ask_question(question,db)
            print("\nBot:", response, "\n")
        else:
            print("Please enter a valid question.")

if __name__ == "__main__":
    main()