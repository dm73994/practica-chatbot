from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from models.llm import llama_instruct
from pipelines.chat_history import chat_history

chat_prompt_simple = PromptTemplate.from_template(
    """Eres un asistente de salud amigable del Hospital Universitario San José.

    Proporciona información general de salud, definiciones médicas básicas y consejos preventivos.

    NO diagnostiques ni prescribas. Para servicios específicos del HUSJ o emergencias, redirige apropiadamente.

    Mantén un tono profesional, empático y educativo.

    Historial: {history}

    Pregunta: {input}

    Respuesta:"""
)

chat_chain = (
    {
        'history': (lambda x: '\n'.join(chat_history)),
        'input': (lambda x: x.get('input', ''))
    }
    | chat_prompt_simple
    | llama_instruct
    | StrOutputParser()
)