from langchain_core.runnables import RunnableAssign

from models import llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnableAssign

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document chatbot. Help the user as they ask questions about documents. "
     "Answer only from retrieval and only cite sources that are used. Make your response conversational."),

    ("system",
     "The following information may be useful for your response:\n"
     "Document Retrieval:\n{context}"),

    ("human",
     "User just asked you a question: {input}")
])

rag_chain = (
    {
        'input': (lambda x: x),
        'context': (lambda x: 'No hay ningun contexto de momento, esto solo es una prueba')
    }
    | chat_prompt
    | llm.llm_model
    | StrOutputParser()
)

def call_rag_chain(text):
    for token in rag_chain.stream(text):
        print(token, end="")