# from langchain_core.messages import HumanMessage, AIMessage


template = """Use the document saved as embeddings from the vectorstore as the knowledge base, as well as the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {input}

{agent_scratchpad}

Helpful Answer:"""