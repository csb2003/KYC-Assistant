import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from IPython.display import Image, display

# Import LangChain and LangGraph modules
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

print("LLM ready")

tavily_search = TavilySearch(max_results=3)

# Load the PDF----------------------------------------------------------------
loader = PyPDFLoader("data/RBIs-Master-Direction-KYC-Direction-2016_compressed.pdf")
pages = loader.load()

# Split into chunks-----------------------------------------------------------------
# Initialize a RecursiveCharacterTextSplitter with a chunk size of 500 characters and an overlap of 50 characters.
# This helps maintain context between chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 30
)

chunks = splitter.split_documents(pages)
print(f"Split {len(pages)} into {len(chunks)} chunks")

# create embeddings of chunks-

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

#create FAISS vector store for similarity search-
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created successfully")

#CREATE A COntext aware retriever-

system_prompt = """Use the given context to answer the question.\nIf you don't know the answer, say you don't know.\n\nContext:\n{context}"""

#Take chat history + current question --> frame proper question with all context
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system","Given an optional chat history and a question, rephrase the question to be self-contained and understandable without the chat history"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


history_aware_retriever = create_history_aware_retriever(
    llm,
    vectorstore.as_retriever(search_kwargs={"k": 5}),
    contextualize_prompt
)
    

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Use the given context to answer the question.
     You are a helpful KYC assistant.
     If the answer is not found in the context, respond with exactly: "IDK"
     Do not make up answers.
     Context: {context}"""),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")
])


# 2. question_answer_chain: Combines the retrieved documents with the qa_prompt to generate the final answer.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 3. rag_chain: The complete RAG pipeline that integrates the retriever and the Q&A chain.
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# AgentState + nodes + graph  ← add here
class AgentState(TypedDict):
    question: str
    answer: str
    chat_history:List

#node 1
def check_scope(state:AgentState):
    question = state["question"]
    system_prompt = """You are a helpful assistant.
        Is this question - "{question}" related to KYC, AML, fintech, banking regulations, 
        or is it a follow-up to a previous conversation about these topics?
        But
        Reply with only one word: RELEVANT or IRRELEVANT.
        When in doubt, reply RELEVANT."""

    response = llm.invoke([
        HumanMessage(content=system_prompt.format(question=question))
    ])
    return {"answer": response.content, "question": question}

#node 2
def retrieve_and_answer(state:AgentState):
    question = state["question"]
    chat_history = state["chat_history"]
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return {"answer": response["answer"],"chat_history": chat_history + [HumanMessage(content= question),AIMessage(content= response["answer"])]}

#node 3
def ask_clarification(state: AgentState):
  print("ask_clarification node")
  return {"answer": f"Could you give me more details about: '{state['question']}'?"}


#node 4
def web_search(state: AgentState):
    question = state["question"]
    results = tavily_search.invoke(question)
    
    if results["answer"]:
        raw = results["answer"]
    elif results["results"]:
        raw = "\n".join([r["content"] for r in results["results"][:2]])
    else:
        return {"answer": "No results found from web search."}
    
    # Pass through LLM to clean up
    response = llm.invoke([
        HumanMessage(content=f"""
        You are a precise information cleanup assistant.
        Based on the following raw web search results, construct a clean, concise, and highly structured answer to the question: "{question}"

        Search Results:
        {raw}

        Follow these cleanup instructions strictly:
        1. Extract only the factual points, specific dates, and guidelines directly answering the question.
        2. Remove all introductory meta-sentences (e.g., "The Reserve Bank of India has issued...", "Here are the key guidelines...").
        3. Remove all conversational filler, boilerplate text, and generic concluding sentences (e.g., "These guidelines demonstrate the RBI's commitment...").
        4. Organize the information with clean markdown bullet points and bold section names if appropriate.
        5. Keep the tone professional, objective, and direct.
        """)
    ])
    return {"answer": response.content}


#conditional edge functions-
def clarify(state: AgentState):
    if "IRRELEVANT" in state["answer"]:
        return "ask_clarification"
    return "retrieve_and_answer"

def should_web_search(state: AgentState):
    if "IDK" in state["answer"]:
        return "web_search"
    return END

#build graph
builder = StateGraph(AgentState)
builder.set_entry_point("check_scope")
builder.add_node("check_scope",check_scope)
builder.add_node("ask_clarification",ask_clarification)
builder.add_node("retrieve_and_answer",retrieve_and_answer)
builder.add_node("web_search",web_search)

builder.add_edge("ask_clarification", END)
builder.add_edge("web_search",END)
builder.add_conditional_edges("check_scope", clarify, {
    "ask_clarification": "ask_clarification",
    "retrieve_and_answer": "retrieve_and_answer"
})

builder.add_conditional_edges("retrieve_and_answer",should_web_search,
{
    "web_search": "web_search",
    END : END
})


workflow = builder.compile()

def get_custom_graph_png() -> bytes:
    """Generate the graph PNG image with conditional edges labeled by their condition function names."""
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png
    
    syntax = workflow.get_graph().draw_mermaid()
    
    # Dynamically inject condition function names on conditional edges
    for source_node, branches in builder.branches.items():
        for cond_name, branch_spec in branches.items():
            for target_node in branch_spec.ends.keys():
                # Replace standard dotted edge with a labeled one using standard Mermaid syntax
                old_edge = f"{source_node} -.-> {target_node};"
                new_edge = f"{source_node} -.->|{cond_name}| {target_node};"
                syntax = syntax.replace(old_edge, new_edge)
                
    return draw_mermaid_png(syntax)

if __name__ == "__main__":
    # Save the graph image locally when run directly
    try:
        with open("graph.png", "wb") as f:
            f.write(get_custom_graph_png())
        print("Graph image saved to graph.png")
    except Exception as e:
        print(f"Failed to save graph image: {e}")

    chat_history = []

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break
        response = workflow.invoke({
            "question": question,
            "answer": "",
            "chat_history": chat_history
        })
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response["answer"]))
        print(response["answer"])