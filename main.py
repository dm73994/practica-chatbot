import gradio as gr

from config import NVIDIA_API_KEY
from pipelines import rag

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

if __name__ == '__main__':
    #print(NVIDIA_API_KEY)
    #rag.call_rag_chain("este rag funciona?")

    demo = gr.Interface(
        fn=greet,
        inputs=["text", "slider"],
        outputs="text"
    )
    demo.launch()