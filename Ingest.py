from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading documents...")
loader = DirectoryLoader("documents/", glob="**/*.xml", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

print("Creating embeddings and storing...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_db")
print("Done! Database created at ./faiss_db")