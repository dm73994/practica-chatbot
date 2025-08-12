from functools import partial

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from rich.console import Console
from rich.style import Style
from pathlib import Path
import os


console = Console()
base_style =Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

long_reorder = RunnableLambda(LongContextReorder().transform_documents)

def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def parse_to_history(data):
    """Accepts 'input'/'output' dictionary and return format string"""
    return f"User previously said: {data.get('input')}\nSystem previously responded: {data.get('output')}"

def docs2str(docs, title="Tile"):
    """Useful utility for making chunks into context string. Optional but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, "metadata", {}).get("title", title)
        if doc_name:
            out_str += f"[Quote from {doc_name}"
        out_str += getattr(doc, 'page_content', str(doc)) + '\n'
    return out_str

def load_pdfs():
    """Función corregida para cargar PDFs"""
    pdf_folder = Path("C:/Users/dm312/Estudios/Universidad/Trabajo de grado/pdfs")
    all_docs = []

    if not pdf_folder.exists():
        print(f"Error: La carpeta {pdf_folder} no existe")
        return []

    # Obtener todos los archivos PDF
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron archivos PDF en {pdf_folder}")
        return []

    print(f"Encontrados {len(pdf_files)} archivos PDF")

    for pdf_file in pdf_files:
        try:
            print(f"Cargando: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                if "title" not in doc.metadata or not doc.metadata["title"]:
                    doc.metadata["title"] = pdf_file.name

            # Filtrar solo documentos con contenido
            docs_con_contenido = [doc for doc in docs if doc.page_content.strip()]

            if docs_con_contenido:
                all_docs.extend(docs_con_contenido)
                print(f"  ✓ Cargadas {len(docs_con_contenido)} páginas de {pdf_file.name}")
            else:
                print(f"  ⚠️ El PDF {pdf_file.name} no tiene contenido válido")

        except Exception as e:
            print(f"  ✗ Error cargando {pdf_file.name}: {str(e)}")

    print(f"\n🎉 Total de documentos cargados: {len(all_docs)}")
    return all_docs



def load_docs():
        """Función para cargar archivos .docx"""
        # Ruta de tu carpeta con archivos DOCX
        docx_folder = "C:/Users/dm312/Estudios/Universidad/Trabajo de grado/docxs"

        # Lista para almacenar todos los documentos
        all_docs = []

        # Verificar que la carpeta existe
        if not os.path.exists(docx_folder):
            print(f"Error: La carpeta {docx_folder} no existe")
            return []

        try:
            # Obtener lista de archivos
            files = os.listdir(docx_folder)
            docx_files = [file for file in files if file.lower().endswith((".docx", ".doc"))]

            if not docx_files:
                print(f"No se encontraron archivos DOCX/DOC en {docx_folder}")
                return []

            print(f"Encontrados {len(docx_files)} archivos DOCX/DOC")

            # Procesar cada archivo DOCX
            for file in docx_files:
                file_path = os.path.join(docx_folder, file)
                try:
                    print(f"Cargando: {file}")
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        if "title" not in doc.metadata or not doc.metadata["title"]:
                            doc.metadata["title"] = file.title()
                    all_docs.extend(docs)
                    print(f"  ✓ Documento cargado exitosamente")

                except Exception as e:
                    print(f"  ✗ Error cargando {file}: {str(e)}")
                    continue

            return all_docs
        except Exception as e:
            print(f"Error general: {str(e)}")
            return []