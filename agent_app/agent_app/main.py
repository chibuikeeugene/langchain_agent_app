# load enviroment variales
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from chat_template import template

from agent import client



def chat_program(dir: str, chat_template):

    agent_executor = client(file_directory= dir, chat_template=chat_template)

    output = agent_executor.invoke({"input": input})
    return output



if __name__ == "__main__":

    template_assigned =  template
    st.set_page_config(page_title="LangChain Agent Q&A")
    st.header("LangChain Agent Application")
    
    input = st.text_input("Ask a question about the file: ")
    submit=st.button("Enter")

    if input:
        if submit:
            response = chat_program(dir= "data", chat_template=template_assigned)
            st.subheader("The Response is")
            st.write(response)
