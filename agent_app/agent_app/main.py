import os
from pathlib import Path


package_dir = Path(os.path.dirname(os.path.dirname(__file__))) # retrieve package directory path
root = package_dir.parent # retrieve root directory path
file_dirs = root / "files"


# load enviroment variales
from dotenv import load_dotenv
load_dotenv()

# import library for our UI
import streamlit as st

# load the chat template
from chat_template import template

# import the langchain agent executor
from agent import client

st.set_page_config(page_title="LangChain Agent Q&A")
st.header("LangChain Agent Application")

# request user to upload pdf file
uploaded_file = st.file_uploader(label= 'Upload a pdf file')

user_question = st.text_input("Ask a question about the file: ")
submit=st.button("Enter")


def chat_program(prompt: str,):
    
    agent_executor = client(file_directory= file_dirs, chat_template=template)

    output = agent_executor.invoke(
        {
            "input": prompt,
            "context": "based on the knowledge base",
            }
        )
    return output["output"]


if __name__ == "__main__":

    if uploaded_file:
        with open(os.path.join(file_dirs, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
    else:
        st.write("No file uploaded")


    if user_question:
        if submit:
            response = chat_program(prompt = user_question)
            st.subheader("The Response is")
            st.write(response)
