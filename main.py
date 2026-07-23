from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import workflow, get_custom_graph_png
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()

server = FastAPI()

# Enable CORS for frontend integration
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history = []

class QuestionRequest(BaseModel):
    question: str


@server.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>KYC Assistant Frontend - index.html not found</h3>"


@server.post("/ask")
def ask(request: QuestionRequest):
    # 1. invoke rag_chain
    response = workflow.invoke({
        "question": request.question,
        "answer": "",
        "chat_history": chat_history
    })
    # 2. append HumanMessage to chat_history
    chat_history.append(HumanMessage(content=request.question))
    # 3. append AIMessage to chat_history
    chat_history.append(AIMessage(content=response["answer"]))
    # 4. return the answer
    return {"answer": response["answer"]}


@server.get("/graph")
def get_graph():
    try:
        png_data = get_custom_graph_png()
        return Response(content=png_data, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}