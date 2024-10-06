from langchain.agents import AgentExecutor

from langchain_ollama import ChatOllama

from langchain.agents import create_openai_functions_agent

from langchain.tools.retriever import create_retriever_tool

from langchain_community.tools.tavily_search.tool import TavilySearchResults

from document_retriever import document_retriever

# to design the prompt template
from langchain_core.prompts import ChatPromptTemplate



def client(file_directory:str, chat_template ):
    # get retriever objects from document retriever
    retriever, retriever_from_existing_doc = document_retriever(
        directory=file_directory
        )

    if retriever_from_existing_doc == None:
    #================= create the tools ======================#

        retriever_tool = create_retriever_tool(
            retriever,
            name="retriever",
            description="Searches and returns documents based on their content and metadata.",
        )
    else:
        retriever_tool = create_retriever_tool(
            retriever_from_existing_doc,
            name="retriever",
            description="Searches and returns documents based on their content and metadata.",
        )

    # setting up the search tool
    search = TavilySearchResults()

    # creating the final tool to be used by our agent
    tool = [retriever_tool, search]


    #================= create the agent ======================#
   
    prompt = ChatPromptTemplate.from_template(template=chat_template)

    llm =  ChatOllama(model="llama3", temperature=0.8, num_predict = 256, )

    agent = create_openai_functions_agent(llm=llm, prompt=prompt, tools=tool,)

    agent_executor = AgentExecutor(agent=agent, tools = tool, verbose=True)

    return agent_executor


