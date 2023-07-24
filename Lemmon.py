import json
import pandas as pd
import hashlib
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from tqdm.auto import tqdm
from uuid import uuid4
import streamlit as st

pinecone.init(api_key = "acc51f3e-7f9f-4cd7-81f1-f0935e6a3dc1", environment="asia-northeast1-gcp")
new_embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese", model_kwargs={'device': "cuda"})

class chatGLM():
    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda().eval()
        self.history = []

    def chat(self, prompt):
        response, self.history = self.model.chat(self.tokenizer, prompt, self.history)
        return response, self.history


def retrieve(input_query, embeddings):
    query_result = embeddings.embed_query(input_query)
    res = index.query(query_result, top_k=1, include_metadata=True)

    retrieve_contexts = [x['metadata']['答案'] for x in res['matches']]

    prompt_start = (
            "请你根据以下内容对问题作答.\n\n"+
            "内容:\n"
    )

    prompt_end = (
        f"\n\问题: {query}\回答:"
    )
    prompt = ""
    for i in range(0, len(retrieve_contexts)):
        if len("\n\n---\n\n".join(retrieve_contexts[:i])) >= limit:
            prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(retrieve_contexts[:i-1]) +
                    prompt_end
            )
            break
        elif i == len(retrieve_contexts)-1:
            prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(retrieve_contexts) +
                    prompt_end
            )

    return prompt

llm =  chatGLM(model_name="chatglm2")

index_name = "demo"
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1024  # 1024 dim of text2vec-large-chinese
    )

food_index = "food-index"
if food_index not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=food_index,
        metric='cosine',
        dimension=1024  # 1024 dim of text2vec-large-chinese
    )

index = pinecone.Index(index_name=index_name)
limit = 3750

st.title("LemmonBox AI 机器人")

query = st.text_input("请输入您的问题")

r = retrieve(input_query=query, embeddings=new_embeddings)
if st.button("查询"):
    r = retrieve(input_query=query, embeddings=new_embeddings)
    response, history = llm.chat(r)
    st.write(response)