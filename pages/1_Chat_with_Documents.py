import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text,chunk_size=1000,chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore,llm_model,temperature,k):
    llm = ChatOpenAI(model_name=llm_model,temperature=temperature)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(search_kwargs={'k': k}),
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
             with st.chat_message("user"):
                 st.markdown(message.content)
        else:
             with st.chat_message("assistant"):
                 st.markdown(message.content)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")
    with st.sidebar:
        st.image("img/logo_sq.png")
        st.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
        st.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Documents :books:")
  
    with st.expander("Database Options"):
        pdf_docs = st.file_uploader("Upload your files (only PDF)",accept_multiple_files=True)
        chunk_size = st.number_input('Chunk Size:',min_value=200, max_value=2000,value=1000)
        overlap_size = st.number_input('Overlap:', min_value = 40, max_value=400, value=200)

        if st.button("Process",key='database'):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                    
                # get the text chunks
                text_chunks = get_text_chunks(raw_text, chunk_size, overlap_size)

                # create the vector store
                st.session_state.vectorstore = get_vectorstore(text_chunks)

            st.success('Done! Proceed to Model Options')

    with st.expander("Model Options"):
        llm_model = st.radio('LLM model:',["gpt-3.5-turbo","gpt-4",],index=0,)
        temperature = st.slider('Temperature:', 0.0, 1.0, step=0.1, value=0.7)
        k = st.slider('Chunks to Retrieve:', 0, 10, step=1, value=4)

        if st.button("Process",key='chat'):
            with st.spinner("Processing"):
                # create converstion chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore,llm_model,temperature,k)
            st.success('Now you can use the chat!')
  
    user_question = st.chat_input("Ask a question about your documents.")
    if user_question:
        handle_userinput(user_question)     

if __name__ == '__main__':
    main()