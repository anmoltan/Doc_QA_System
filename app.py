import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import tiktoken

load_dotenv()
# PDF processing function
def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    store_name = pdf.name[:-4]

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key= "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")
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    return VectorStore

# Chatbot response function
def get_chatbot_response(query, VectorStore):
    docs = VectorStore.similarity_search(query=query, k=3)

    llm = OpenAI(openai_api_key= "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")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
    return response

# Main function for Streamlit app
def main():
    st.title("AI QA System")

    # PDF processing and QA
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        VectorStore = process_pdf(pdf)

    # Chatbot UI
    if pdf:
        st.header("Chat with PDF")


        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask Me"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            query = prompt  # Use the chatbot's query as the PDF QA query
            response = get_chatbot_response(query, VectorStore)
            st.write(response)

            # Display the PDF QA answer in the chat
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
