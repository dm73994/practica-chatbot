from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.llm import llama_instruct

### CHAT_HISTORY MANAGEMENT

chat_history = []

resume_chat_prompt = ChatPromptTemplate.from_template(
    "A continuación tienes una interacción previa en el formato:\n"
    "{input}\n\n"
    "Resume la conversación de forma muy breve, clara y directa, "
    "dejando solo lo esencial. Reformula la pregunta del usuario en una sola frase corta si es posible. "
    "Evita detalles irrelevantes y texto innecesario."
)

resume_chain = (
    resume_chat_prompt
    | llama_instruct
    | StrOutputParser()
)


def parse_to_history(data):
    """Accepts 'input'/'output' dictionary and return format string"""
    return f"User previously said: {data.get('input')}\nSystem previously responded: {data.get('output')}"


def update_chat_history(data):
    """Actualiza el historial con resumen"""
    global chat_history

    try:
        h = parse_to_history(data)
        response = resume_chain.invoke({'input': h})
        print(f'Historial Resumido: {response}')

        # Controlar el tamaño del historial (mantener últimas 10 interacciones)
        if len(chat_history) >= 10:
            chat_history = chat_history[-9:]  # Mantener las 9 más recientes

        chat_history.append(response)

    except Exception as e:
        print(f"Error actualizando historial: {e}")
        simple_entry = f"User: {data.get('input', '')[:100]}... | Bot: {data.get('output', '')[:100]}..."
        chat_history.append(simple_entry)