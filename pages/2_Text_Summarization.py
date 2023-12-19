import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def txt_splt(text,split_size):
    text_splitter = TokenTextSplitter(chunk_size=split_size, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    return docs

def token_count(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def summarizer(llm_model, temperature, prompt, docs, type,refine_prompt=None):
    llm = ChatOpenAI(model_name=llm_model,temperature=temperature)
    if type == 'stuff':
        chain = load_summarize_chain(llm, chain_type=type, prompt=prompt)
    elif type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=type, map_prompt=prompt, combine_prompt=prompt)
    else:
        chain = load_summarize_chain(llm, chain_type=type, question_prompt=prompt, refine_prompt=refine_prompt)
    return chain.run(docs)

def main():
    load_dotenv()
    st.set_page_config(page_title="Summarize", page_icon=":receipt:")
    with st.sidebar:
        st.image("img/logo_sq.png")
        st.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
        st.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

    st.header("Text Summarization :receipt:")
    with st.expander("Model Options"):
        st.session_state.llm_model = st.radio('LLM model:',["gpt-3.5-turbo","gpt-4"],index=0,)
        st.session_state.temperature = st.slider('Temperature:', 0.0, 1.0, step=0.1, value=0.7)
        if st.session_state.llm_model == "gpt-3.5-turbo":
            st.session_state.split_size = st.slider('Split Size (tokens):', 200, 4000, step=100, value=2000)
        else:
            st.session_state.split_size = st.slider('Split Size (tokens):', 2000, 8000, step=100, value=4000)   

    st.write("""If your text is smaller than the split size, the method used for summarization is going to be 'stuff'. 
            If it is longer, the option to chose between 'map_reduce' and 'refine' is going to be available.""")
    
    txt_sum = st.text_area(
            "Text to summarize.",
            "Inser your text here."
            )
    st.write(f"""Your text has '{len(txt_sum)}' characters and '{token_count(txt_sum)}' tokens.""")

    txt_prompt = st.text_area(
            "Define your prompt.",
            """Write a concise summary of the following text:
            '{text}'""")
    prompt = PromptTemplate.from_template(txt_prompt)

    if token_count(txt_sum) > st.session_state.split_size:
        st.session_state.type = st.radio('Summarization Method:',["map_reduce","refine"],index=0,)
        if st.session_state.type == "refine":
            refine_txt = st.text_area(
                "Define your refine prompt.",
                """Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_answer}\n
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {text}
                ------------
                Given the new context, refine the original summary. If the context isn't useful, return the original summary.""")
            refine_prompt = PromptTemplate.from_template(refine_txt)
    else:
        st.session_state.type = 'stuff'
 

    if st.button("Process",key='runmodel'):
        with st.spinner("Processing"):
            if st.session_state.type == "stuff":
                summarized = summarizer(llm_model=st.session_state.llm_model, temperature=st.session_state.temperature, prompt=prompt,docs=txt_splt(txt_sum,st.session_state.split_size),type=st.session_state.type)
                txt_prompt = st.text_area("Here is your summarization and you can edit it. \n\n", summarized)
            elif st.session_state.type == "map_reduce":
                summarized = summarizer(llm_model=st.session_state.llm_model, temperature=st.session_state.temperature, prompt=prompt,docs=txt_splt(txt_sum,st.session_state.split_size),type=st.session_state.type)
                txt_prompt = st.text_area("Here is your summarization and you can edit it. \n\n", summarized)
            else:
                summarized = summarizer(llm_model=st.session_state.llm_model, temperature=st.session_state.temperature, prompt=prompt,docs=txt_splt(txt_sum,st.session_state.split_size),type=st.session_state.type,refine_prompt=refine_prompt)
                txt_prompt = st.text_area("Here is your summarization and you can edit it. \n\n", summarized)

if __name__ == '__main__':
    main()
    