# load enviroment variales
from dotenv import load_dotenv
load_dotenv()


# to load the document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
# to convert document contents into embeddings
from langchain_community.embeddings import OllamaEmbeddings
# save our embeddings into a vector and create a vectorstore retriver object
from langchain_community.vectorstores import FAISS
# document content splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.tools.tavily_search.tool import TavilySearchResults
# for observability and monitoring
from langsmith import traceable
from langsmith.wrappers import  wrap_openai

# to load the model
from langchain_community.llms import Ollama

# to design the prompt template
from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_ollama import ChatOllama

from fastapi import FastAPI, Request
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
#================= load the retriever =================#

# load a sample website
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# instantiate the embedding object
embedding_obj = OllamaEmbeddings(model="llama3")

# performing the text splitting
text_splitter = RecursiveCharacterTextSplitter()

splitted_doc = text_splitter.split_documents(docs)

vector = FAISS.from_documents(splitted_doc, embedding_obj)
retriever = vector.as_retriever()

#================= create the tools ======================#

retriever_tool = create_retriever_tool(
    retriever,
    name="retriever",
    description="Searches and returns documents based on their content and metadata.",
)

# setting up the search tool
search = TavilySearchResults()

# creating the final tool to used by our agent
tool = [retriever_tool, search]


#================= create the agent ======================#
# creating a dummy prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

llm =  ChatOllama(model="llama3", temperature=0.8, num_predict = 256, )

agent = create_openai_functions_agent(llm=llm, prompt=prompt, tools=tool,)

agent_executor = AgentExecutor(agent=agent, tools = tool, verbose=True)


#================== app definition =======================#
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)


#================= adding chain routes ====================#
class Input(BaseModel):
    input: str
    chat_history: list[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )

class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)