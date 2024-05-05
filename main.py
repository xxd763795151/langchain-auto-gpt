import gradio as gr

import os

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
# from langchain_openai import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI

os.environ["SERPAPI_API_KEY"] = "qew"
os.environ["OPENAI_API_KEY"] = "123"

# 构造 AutoGPT 的工具集
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]
# OpenAI Embedding 模型
embeddings_model = OpenAIEmbeddings()
# OpenAI Embedding 向量维数
embedding_size = 1536
# 使用 Faiss 的 IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True),
    memory=vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8}),# 实例化 Faiss 的 VectorStoreRetriever
)

def gpt_chat(message, history):
    enable_chat = True
    print(message)
    if enable_chat:
        agent.chain.verbose = True
        return agent.run([message])
    # 否则输出套路话术
    else:
        return "目前不支持"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=gpt_chat,
        title="晓东哥哥",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == '__main__':
    launch_gradio()
