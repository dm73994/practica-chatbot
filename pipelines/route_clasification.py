from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from models.llm import llama_instruct
from pipelines.chat import chat_chain
from pipelines.rag import rag_chain
from utils.utils import stream_chain, RPrint

classification_prompt = PromptTemplate.from_template(
    """Clasifica la siguiente pregunta del usuario en UNA de estas tres categorías:

    RAG: Preguntas específicamente sobre telemedicina, telesalud, medicina a distancia, consultas virtuales, telemonitoreo, teleconsultas, servicios médicos remotos, tecnología médica para atención virtual, o cualquier tema directamente relacionado con la prestación de servicios de salud a través de medios digitales/remotos. También incluye información específica del Hospital Universitario San José sobre estos servicios.

    CHAT: Conversación general, saludos, preguntas sobre salud general (enfermedades, síntomas, tratamientos tradicionales), definiciones médicas básicas, medicina presencial, especialidades médicas, anatomía, farmacología, nutrición, ejercicio, bienestar general, o cualquier tema de salud que NO sea específicamente sobre telemedicina.

    AGENT: Solicitudes para realizar acciones específicas como agendar citas, cancelar servicios, enviar documentos, realizar trámites, contactar departamentos, modificar horarios, o cualquier tarea que requiera interactuar con sistemas del hospital.

    EJEMPLOS:
    - "¿Cómo funciona la telemedicina en el HUSJ?" → RAG
    - "Dime algo curioso sobre telemedicina" → RAG
    - "¿Qué es una teleconsulta?" → RAG
    - "Ventajas de la telesalud" → RAG
    - "¿Qué es la hipertensión?" → CHAT
    - "Síntomas de diabetes" → CHAT
    - "¿Qué especialidades tiene el hospital?" → CHAT
    - "Tratamientos para el cáncer" → CHAT
    - "Hola, ¿cómo estás?" → CHAT
    - "Quiero agendar una cita de telemedicina" → AGENT
    - "Necesito cancelar mi consulta" → AGENT

    REGLA CLAVE: Solo clasifica como RAG si la pregunta menciona explícitamente telemedicina, telesalud, consultas virtuales, medicina remota o términos directamente relacionados con servicios médicos digitales/a distancia.

    IMPORTANTE: Responde ÚNICAMENTE con una palabra: RAG, CHAT, o AGENT.

    <question>
    {input}
    </question>

    Clasificación:"""
)

route_chain = (
    classification_prompt
    | llama_instruct
    | StrOutputParser()
)


def route(info):
    """Función de enrutamiento"""
    try:
        topic = info.get("topic", "").upper()
        user_input = info.get("input", "")

        print(f'User_route_info: {user_input}')

        if not user_input:
            return stream_chain(chat_chain, "Hola", return_buffer=True)

        if "RAG" in topic:
            return stream_chain(rag_chain, user_input)
        elif "CHAT" in topic:
            return stream_chain(chat_chain, user_input)
        else:
            return stream_chain(chat_chain, user_input)
    except Exception as e:
        print(f"Error en route: {e}")
        return stream_chain(chat_chain, info.get("input", "Error"))


full_chain = (
    RPrint() |
    {
        "topic": route_chain,
        "input": (lambda x: x.get("input", ""))
    }
    | RPrint()
    | RunnableLambda(route)
)
