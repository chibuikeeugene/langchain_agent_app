from langserve import RemoteRunnable
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="LangChain Agent Q&A")
st.header("LangChain Agent Application")

chat_history = []
user_input  = st.text_input(
    "Ask a question",
    key="input",
)

submit = st.button('Enter')



if __name__ == "__main__":
    # adding chain client
    if user_input and submit:
        remote_chain = RemoteRunnable("http://localhost:8000/agent/")
        response = remote_chain.invoke({
            "input": user_input,
            "chat_history": chat_history  # Providing an empty list as this is the first call
        })
        st.subheader("Answer:")
        st.write(response['output'])
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response['output'])
        ])
    


