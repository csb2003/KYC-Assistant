from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import workflow
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

server = FastAPI()
chat_history = []

class QuestionRequest(BaseModel):
    question: str


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