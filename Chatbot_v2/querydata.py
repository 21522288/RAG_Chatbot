import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

from get_embedding_function import get_embedding_function
from dotenv import load_dotenv
import os
# from getpass import getpass
# HUGGINGFACEHUB_API_TOKEN = getpass()
load_dotenv()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# from langchain_community.llms import NLPCloud
# llm = NLPCloud(nlpcloud_api_key="e4c513eeddeda22f6e014b7a39c1124b82c9f668")
llm = HuggingFaceHub(repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1',model_kwargs={'temprature':0.5, 'max_length':500})

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Context: {context}

Question: {question}

Answer the question based on above context:
"""
# PROMPT_TEMPLATE = """
# Bạn là trợ lý hỗ trợ trả lời các câu hỏi dựa trên dữ liệu từ tài liệu sau đây.
# Hãy trả lời câu hỏi dưới đây bằng tiếng Việt, với câu trả lời ngắn gọn, đầy đủ ý và không ngắt giữa chừng.

# Tài liệu tham khảo:
# {context}

# Câu hỏi: {question}

# Trả lời đầy đủ và giống thông tin từ tài liệu tham khảo:
# """

def main():
    query_rag("Giới thiệu tóm tắt về nha khoa BestSmile?")
    


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = llm.invoke(prompt)
    answer = response_text.split("Answer the question based on above context:")[-1].strip()
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(answer)
    return response_text


if __name__ == "__main__":
    main()
