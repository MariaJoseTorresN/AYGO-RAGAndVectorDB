from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "0c0d123c-4051-451a-af86-11144db419f8"
os.environ["PINECONE_ENV"] = "gcp-starter"

def loadText():
    loader = TextLoader("awedfirstpaper.txt")
    documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )


    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    import pinecone

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("https://langchain-demo-54nkuoa.svc.gcp-starter.pinecone.io"),  # find at app.pinecone.io
        environment=os.getenv("gcp-starter"),  # next to api key in console
    )

    index_name = "langchain-demo"

    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def search():
    embeddings = OpenAIEmbeddings()
    import pinecone

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    index_name = "langchain-demo"
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    query = "What is a distributed pointcut"
    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)

search()