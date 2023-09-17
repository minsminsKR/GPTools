import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon=":tongue:"
)

st.title("Welcome to My World! 👋")

st.sidebar.success("👆 Select a page above.")

st.markdown(
    """
    ##### LLM ai projects. 
    **👈 Select a project from the sidebar** to see more about projects or me?
    	😁 Check it out with below ***"Contact me"***!!
    ### 📚 GPTools
    Easily find the information you need in your pile of documents.
    
    Supported formats are **pdf, docx, doc** and **txt**.
    - [streamlit.io](https://streamlit.io)
    - [langchain](https://www.langchain.com/) 
    - [OpenAI](https://openai.com/)
    
    ### 🤭 YourChatbot
    Make your own simple Chat bot!! using your **Kakao Talk** message file.
    - [Bert](https://huggingface.co/jhgan/ko-sroberta-multitask)
    - [Pandas](https://pandas.pydata.org/)

    ### 📞 Contact me
    I'm listening to you guys opinion. Feel free to contact me 🤙
    - Welcome to my [github](https://github.com/minsminsKR)
    - e-mail : gjwjdjq@gmail.com
"""
)
