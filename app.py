import glob
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Zyro Dynamics HR Help Desk")

@st.cache_resource
def build_rag():

    documents = []

    for pdf in glob.glob("data/*.pdf"):
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 25}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

    return retriever, llm

retriever, llm = build_rag()

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are the Zyro Dynamics HR Help Desk Assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, reply exactly:

I can only answer questions based on the Zyro Dynamics HR policy documents provided.

Context:
{context}

Question:
{question}

Answer:
""")

REFUSAL_MESSAGE = (
    "I can only answer questions based on the Zyro Dynamics HR policy documents provided."
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_bot(question):

    docs = retriever.invoke(question)

    if len(docs) == 0:
        return REFUSAL_MESSAGE

    context = format_docs(docs)

    prompt = RAG_PROMPT.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(prompt)

    return response.content

st.title("🤖 Zyro Dynamics HR Help Desk")

question = st.chat_input("Ask an HR question")

if question:
    st.chat_message("user").write(question)

    answer = ask_bot(question)

    st.chat_message("assistant").write(answer)