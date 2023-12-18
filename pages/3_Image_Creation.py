import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from openai import OpenAI as OPA
from langchain.prompts import PromptTemplate
from langchain.utilities.dalle_image_generator import DallEAPIWrapper

def prompt_enhancer(model,temperature,text):
    llm = ChatOpenAI(model_name=model,temperature=temperature)
    prompt = PromptTemplate(input_variables=["image_desc"], template="You are a professional assistant that help people who wants to create images in a generative AI model. Some times their prompt is not good and lack information. This way you are going to receive their promtp and enhance it to a better prompt for the model. Make it simples, not more than 400 words. User prompt: {image_desc}. Your promtp:")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text)

def get_images_2(model,prompt,size,n=1):
    client = OPA()
    response = client.images.generate(model=model,prompt=prompt,size=size,n=n)
    caption =[]
    urls = []
    if n > 1:
        for i in range(0,n-1):
            caption =[caption,i+1]
            urls = [urls,response.data[i].url]          
            return caption, urls
    else:
        return response.data[0].url

def get_images_3(model,prompt,size,quality,style):
    client = OPA()
    response = client.images.generate(model=model,prompt=prompt,size=size,quality=quality,style=style)
    return response.data[0].url       

def main():
    load_dotenv()
    st.set_page_config(page_title="Image Creation", page_icon=":frame_with_picture:")

    with st.sidebar:
        st.image("img/logo_sq.png")
        st.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
        st.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

    st.header("Image Creation :frame_with_picture:")
    
    with st.expander("Model Options"):
        st.session_state.i_model = st.radio('Image model:',["dall-e-2","dall-e-3"],index=0)
        if st.session_state.i_model == "dall-e-2":
            st.session_state.i_size = st.selectbox('Size of the image:', ("256x256", "512x512", "1024x1024"))
            #st.session_state.n_images = st.slider('Number of images:', 1, 10, 1)
        else:
            st.session_state.i_size = st.selectbox('Size of the image:', ("1024x1024", "1792x1024","1024x1792")) 
            st.session_state.i_qual = st.radio('Image quality:',["standard","hd",],index=0) 
            st.session_state.i_style = st.radio('Image style:',["vivid","natural",],index=0)
        st.session_state.enh_bot = st.toggle('Activate Enhanced Prompt')
        if st.session_state.enh_bot:
            st.session_state.pen_model = st.radio('Model:',["gpt-3.5-turbo","gpt-4"],index=0)
            st.session_state.pen_temp = st.slider('Temperature:', 0.0, 1.0, 0.1)

    txt_prompt = st.text_area("Describe your image.")

#    if "enh_bot" in st.session_state:
    if st.session_state.enh_bot:
        if st.button("Enhance",key='enhancer_2'):
            with st.spinner("Processing"):
                st.session_state.p_enhanced = prompt_enhancer(st.session_state.pen_model,st.session_state.pen_temp,txt_prompt)
    else:
        if st.button("Generate Image",key='runmodel'):
                with st.spinner("Processing"):
                    if st.session_state.i_model == "dall-e-2":
                        st.image(get_images_2(model=st.session_state.i_model,prompt=txt_prompt,size=st.session_state.i_size))
                    else:
                        st.image(get_images_3(model=st.session_state.i_model,prompt=txt_prompt,size=st.session_state.i_size,quality=st.session_state.i_qual,style=st.session_state.i_style))

    if "p_enhanced" in st.session_state:
        p_enhanced_box = st.text_area("Enhanced Prompt",st.session_state.p_enhanced)
        st.write(f"""Your prompt has '{len(p_enhanced_box)}' characters. The limit is 1000.""")
 
        if st.button("Generate Image (Enhanced)",key='runmodel_enh'):
            with st.spinner("Processing"):
                if st.session_state.i_model == "dall-e-2":
                    st.image(get_images_2(model=st.session_state.i_model,prompt=p_enhanced_box,size=st.session_state.i_size))
                else:
                    st.image(get_images_3(model=st.session_state.i_model,prompt=p_enhanced_box,size=st.session_state.i_size,quality=st.session_state.i_qual,style=st.session_state.i_style))
    
if __name__ == '__main__':
    main()
    