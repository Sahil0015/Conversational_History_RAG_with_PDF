import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os


## set up the streamlit app
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

## input the Groq API key
api_key = st.text_input("Enter your Groq API key", type="password")

## input the Hugging Face API key
Hugging_Face_api_key = st.text_input("Enter your Hugging Face API key", type="password")

os.environ['HF_TOKEN']=Hugging_Face_api_key
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Check if Groq API key is provided
if api_key and Hugging_Face_api_key:
    llm = ChatGroq(groq_api_key = api_key, model_name = "llama3-8b-8192")

    ## chat interface
    session_id = st.text_input("Session ID", value = "default")

    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    ## process uploaded PDFs
    if uploaded_files:
        document = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            document.extend(docs)

        ## Split and create embeddinbgs for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(document)
        vector_store = Chroma(
            collection_name="test_collection",
            embedding_function=embeddings,
            persist_directory="./chroma"
        )
        vector_store.add_documents(splits)
        retriever = vector_store.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id]=ChatMessageHistory()
                return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
                rag_chain,get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")
