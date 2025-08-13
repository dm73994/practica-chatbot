import gradio as gr
from config import NVIDIA_API_KEY
from pipelines.chat_history import chat_history

from pipelines.route_clasification import full_chain

def chat_gen(message, history):
    """Función de chat"""
    try:
        print(f'Evaluando mensaje "{message}"')
        # streaming token por token
        for response in full_chain.stream({"input": message}):
            # response debería ser el buffer completo actualizado
            if isinstance(response, str):
                yield response
            elif hasattr(response, '__iter__'):
                # Si es un generador, tomar el último valor
                final_response = ""
                for token in response:
                    final_response = token
                if final_response:
                    yield final_response

    except Exception as e:
        yield f"Error: {str(e)}"

if __name__ == '__main__':
    demo = gr.ChatInterface(
        fn=chat_gen,
        type="messages",
        examples=["hello", "hola"],
        title="Rag Chat",
    )
    demo.launch(debug=False)