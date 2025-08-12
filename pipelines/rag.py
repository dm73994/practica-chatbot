from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableAssign
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from models import llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnableAssign

from utils.utils import long_reorder, docs2str, parse_to_history, RPrint

### RAG CHAIN

docstore = FAISS.load_local("docstore_index", llm.vecstore_retriever, allow_dangerous_deserialization=True)
retriever = docstore.as_retriever()

chat_history = []
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un chatbot especializado en documentos. Ayuda al usuario a responder sus preguntas sobre documentos. "
     "Responde únicamente con la información recuperada y cita solo las fuentes que utilices. Mantén un tono conversacional."),

    ("system",
     "La siguiente información puede ser útil para tu respuesta:\n"
     "Historial de conversación recuperado:\n{history}\n\n"
     "Documentos recuperados:\n{context}"),

    ("human",
     "El usuario te acaba de hacer la siguiente pregunta: {input}")
])

rag_chain = (
    {
        'input': (lambda x: x),
        'context': retriever | long_reorder | docs2str,
        'history': (lambda x: chat_history),
    }
    | chat_prompt
    | RPrint()
    | llm.llama_instruct
    | RPrint()
    | StrOutputParser()
)

### CHAT_HISTORY MANAGEMENT

resume_chat_prompt = ChatPromptTemplate.from_template(
    "A continuación tienes una interacción previa en el formato:\n"
    "{input}\n\n"
    "Resume la conversación de forma muy breve, clara y directa, "
    "dejando solo lo esencial. Reformula la pregunta del usuario en una sola frase corta si es posible. "
    "Evita detalles irrelevantes y texto innecesario."
)

resume_chain = (
    resume_chat_prompt
    | llm.llama_instruct
    | StrOutputParser()
)

def update_chat_history(data):
    h = parse_to_history(data)
    response = resume_chain.invoke({'input': h})
    print(response)
    chat_history.append(h)
