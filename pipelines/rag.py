from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableAssign
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from models import llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnableAssign

from pipelines.chat_history import chat_history, update_chat_history
from utils.utils import long_reorder, docs2str, RPrint

### RAG CHAIN

docstore = FAISS.load_local("docstore_index", llm.vecstore_retriever, allow_dangerous_deserialization=True)
retriever = docstore.as_retriever()


chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente especializado en telemedicina del Hospital Universitario San José (HUSJ). "
     "Tu función es proporcionar información precisa y actualizada sobre los servicios de telemedicina "
     "basándote exclusivamente en los documentos oficiales de la institución.\n\n"

     "INSTRUCCIONES CLAVE:\n"
     "• Responde ÚNICAMENTE con información presente en los documentos recuperados\n"
     "• Si no encuentras información específica en los documentos, indícalo claramente\n"
     "• Cita las fuentes exactas que utilices en tu respuesta\n"
     "• Mantén un tono profesional pero accesible\n"
     "• Prioriza la precisión sobre la completitud\n"
     "• Si la pregunta no está relacionada con telemedicina del HUSJ, redirige amablemente al tema\n\n"

     "FORMATO DE RESPUESTA:\n"
     "• Proporciona la información solicitada de manera clara y estructurada\n"
     "• Incluye referencias específicas a los documentos utilizados\n"
     "• Si hay múltiples opciones o procedimientos, preséntelos de forma organizada\n"
     "• Finaliza con una invitación a hacer preguntas adicionales si es necesario"),

    ("system",
     "CONTEXTO DISPONIBLE:\n\n"
     "Historial de conversación relevante:\n{history}\n\n"
     "Documentos oficiales del HUSJ sobre telemedicina:\n{context}\n\n"
     "Analiza cuidadosamente este contexto antes de formular tu respuesta. "
     "Asegúrate de que toda la información proporcionada esté respaldada por los documentos recuperados."),

    ("human",
     "Pregunta del usuario sobre telemedicina en el Hospital Universitario San José: {input}")
])

rag_chain = (
    RPrint() |
    {
        'input': (lambda x: x if isinstance(x, str) else x.get('input', '')),
        'context': (lambda x: x if isinstance(x, str) else x.get('input', '')) | retriever | long_reorder | docs2str,
        'history': (lambda x: "\n".join(chat_history)),
    }
    | chat_prompt
    | RPrint()
    | llm.llama_instruct
    | RPrint()
    | StrOutputParser()
)


