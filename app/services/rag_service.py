import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.prompts import SYSTEM_PROMPT


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def summarize_content(file_path: str = None, text: str = None, user_prompt: str = ""):
    """Handles PDF only, Text only, or Both combined."""
    docs = []

   
    if file_path:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    
    if text and text.strip():
        docs.append(Document(page_content=text.strip()))

    
    if not docs:
        return "No content provided to summarize."

   
    return _run_rag(docs, user_prompt)

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def _run_rag(docs, user_prompt):
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "User Request:\n{question}")
    ])

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_prompt)

    vectorstore.delete_collection()

    return response