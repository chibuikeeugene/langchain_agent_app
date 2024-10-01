# to load the document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
# to convert document contents into embeddings
from langchain_community.embeddings import OllamaEmbeddings
# save our embeddings into a vector and create a vectorstore retriver object
from langchain_community.vectorstores import FAISS
# document content splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# for logging purposes
from loguru import logger



#================= load the retriever =================#

def document_retriever(directory:str):
    
    """this function receives a file document, creates the vector embeddings and uploads to qdrant 
    and then instantiates a retriever object of our qdrant vectortore"""

    retriever_from_existing_doc = None
    retriever = None

    # load a sample file from a directory
    loader = DirectoryLoader(
        directory,
        show_progress = True,
        sample_size = 1,
        use_multithreading = True
    )

    logger.info(f"Loading document from location {directory}...")
    docs = loader.load()

   # performing the text splitting
    logger.info("Splitting the document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter()
    splitted_doc = text_splitter.split_documents(docs)
   
   # instantiate the embedding object
    embedding_obj = OllamaEmbeddings(model="llama3")

    # create a qdrant client
    logger.info("Creating a Qdrant client...") 
    client = QdrantClient(
        path= "./qdrant_embedded_data_store",
        # url = "http://localhost:6333",
    )


    if client.collection_exists("agent_doc"):
        # use an existing document vector instance
        qdrant = QdrantVectorStore.from_existing_collection(
        collection_name="agent_doc",
        client = client,
        embedding=embedding_obj,
        distance=Distance.EUCLID,
        )
        retriever_from_existing_doc = qdrant.as_retriever()

    # if False, create a new collection
    # and use it for the vector store instance
    else: 
        client.create_collection(
            collection_name="agent_doc",
            vectors_config=VectorParams(size=4096, distance=Distance.EUCLID),
        )
        # instantiate a new vectorstore object
        vectore_store = QdrantVectorStore(
        collection_name="agent_doc",
        client=client,
        embedding=embedding_obj,
        distance=Distance.EUCLID,
    )
        
        # add the splitted docs to the vector store
        logger.info("Adding documents to the vector store...")
        vectore_store.add_documents(splitted_doc)

        retriever = vectore_store.as_retriever()

    return retriever, retriever_from_existing_doc