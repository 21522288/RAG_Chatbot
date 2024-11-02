# from langchain_community.embeddings import BedrockEmbeddings

# def get_embedding_function():
#    embeddings = BedrockEmbeddings(
#     credentials_profile_name="bedrock-admin", region_name="us-east-1"
#    )
#     # embeddings = OllamaEmbeddings(model="nomic-embed-text")
#    return embeddings
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# def get_embedding_function():
#    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     # embeddings = OllamaEmbeddings(model="nomic-embed-text")
#    return embeddings

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def get_embedding_function():
   embeddings = FastEmbedEmbeddings()
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
   return embeddings