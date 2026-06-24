from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import ask_question, load_db
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = load_db()

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Question):
    answer = ask_question(q.question, db)
    return {"answer": answer}