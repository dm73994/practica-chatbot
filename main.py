import gradio as gr

from config import NVIDIA_API_KEY
from pipelines import rag

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