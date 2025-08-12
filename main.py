import gradio as gr
from aiofiles import os
from faiss import IndexFlatL2
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

import utils.utils
from config import NVIDIA_API_KEY
from pipelines import rag

import json
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

from utils.utils import pprint


def chat_gen(message, history, return_buffer=True):
    buffer = ""
    for token in rag.rag_chain.stream(message):
        buffer += token
        yield buffer if return_buffer else token
    rag.update_chat_history({'input': message, 'output': buffer})

if __name__ == '__main__':
    demo = gr.ChatInterface(
        fn=chat_gen,
        type="messages",
        examples=["hello", "hola"],
        title="Rag Chat",
    )
    demo.launch()