import streamlit as st
from st_pages import Page, show_pages

def main():
    st.set_page_config(page_title="MAIA", page_icon="🏠")
    left_co, cent_co,last_co = st.columns([1,5,1])
    with cent_co:
        st.image("img/logo_wd.png")
        hide_img_fs = '''
            <style>
            button[title="View fullscreen"]{
                visibility: hidden;}
            </style>
            '''

        st.markdown(hide_img_fs, unsafe_allow_html=True)
    
    st.write(
    """
    Introducing MAIA, your all-in-one solution for seamless interaction with information and creativity! 
    Our app harnesses the power of AI to bring you a unique set of features designed to enhance your productivity and creativity.

    **1. Chat with Documents:**
        Say goodbye to the hassle of searching through lengthy PDF documents! With our Chat with Documents feature, you can effortlessly 
        ask questions and receive instant, relevant answers. Whether you're a student, researcher, or professional, this tool transforms 
        the way you interact with information. Simply type your query, and let our AI navigate through documents to provide you with the 
        precise information you need.

    **2. Summarization:**
        Save time and enhance your understanding with our Summarization feature. No more drowning in information overload! Our AI-powered 
        summarization tool meticulously analyzes text, distilling it down to its key points and delivering concise, informative summaries. 
        Ideal for research, studying, or quick knowledge absorption, this feature empowers you to focus on what truly matters.

    **3. Image Creation:**
        Unleash your creativity with our Image Creation feature. Transform ideas into visual masterpieces effortlessly! Whether you're designing 
        presentations, social media posts, or just want to add a visual touch to your content, our AI-driven image creation tool has you covered. 
    """
    )

    with st.sidebar: 
        show_pages(
        [
            Page("Home.py", "Home", "🏠"),
            Page("pages/1_Chat_with_Documents.py", "Chat with Documents", ":books:"),
            Page("pages/2_Text_Summarization.py", "Text Summarization", ":receipt:"),
            Page("pages/3_Image_Creation.py", "Image Creation", ":frame_with_picture:")
        ])
        st.image("img/logo_sq.png")
        hide_img_fs = '''
            <style>
            button[title="View fullscreen"]{
                visibility: hidden;}
            </style>
            '''

        st.markdown(hide_img_fs, unsafe_allow_html=True)
        st.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
        st.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

if __name__ == '__main__':
    main()