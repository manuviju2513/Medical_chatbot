from flask import Flask, render_template, request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🔹 Load embeddings
embeddings = download_embedding()

# 🔹 In-memory session store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 🔹 Vector DB
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 🔹 LLM
chatmodel = ChatOpenAI(model="gpt-4o")

# ✅ FIXED PROMPT (Proper memory injection)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 🔹 Chains
question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ✅ FIXED MEMORY WRAPPER
rag_chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"   # 🔥 critical fix
)

# 🔹 Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    # ✅ Dynamic session id (per user)
    session_id = request.remote_addr

    response = rag_chain_with_memory.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}}
    )

    print("Session:", session_id)
    print("Memory:", store[session_id].messages)  # debug
    print("Response:", response["answer"])

    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)