"""
Script de prueba rápida para desarrollo - procesa solo unos pocos documentos
"""
import os
from pathlib import Path
from typing import List
import shutil

import pdfplumber
import pytesseract
from PIL import Image

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuración para pruebas
PDF_DIR = "documents"
CHROMA_DB_DIR = "chroma_db_test"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_DOCS = 10  # Solo procesar los primeros 2 PDFs para pruebas

def process_pdf_simple(pdf_path: Path) -> List[Document]:
    """Procesar PDF de forma simple, solo las primeras 3 páginas"""
    print(f"📖 Procesando (test): {pdf_path.name}")
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        # Solo procesar las primeras 3 páginas para tests
        pages_to_process = min(3, len(pdf.pages))
        for i in range(pages_to_process):
            page = pdf.pages[i]
            text = page.extract_text()
            if text and len(text.strip()) > 10:
                docs.append(Document(
                    page_content=text, 
                    metadata={"source": str(pdf_path), "page": i + 1}
                ))
    return docs

def quick_test():
    """Ejecutar prueba rápida con pocos documentos"""
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    
    # Obtener solo los primeros PDFs
    pdf_files = list(Path(PDF_DIR).rglob("*.pdf"))[:MAX_DOCS]
    if not pdf_files:
        print("❌ No se encontraron archivos PDF en el directorio 'documents'")
        return
    
    print(f"🧪 MODO PRUEBA: Procesando {len(pdf_files)} archivos PDF")
    
    # Procesar documentos
    all_docs = []
    for pdf_file in pdf_files:
        docs = process_pdf_simple(pdf_file)
        all_docs.extend(docs)
    
    if not all_docs:
        print("❌ No se extrajeron documentos")
        return
    
    print(f"📄 Documentos extraídos: {len(all_docs)}")
    
    # Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_docs)
    print(f"📊 Fragmentos generados: {len(chunks)}")
    
    # Crear embeddings con configuración optimizada
    print("🔗 Generando embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )
    
    # Crear base vectorial
    vectorstore = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    
    print(f"✅ Base de datos de prueba creada en: {CHROMA_DB_DIR}")
    print("🧪 Prueba completada exitosamente")
    
    # Probar una consulta rápida
    test_query = "¿Qué contienen estos documentos?"
    results = vectorstore.similarity_search(test_query, k=MAX_DOCS)
    print(f"\n🔍 Prueba de consulta: '{test_query}'")
    print(f"📝 Resultados encontrados: {len(results)}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.page_content[:100]}...")

if __name__ == "__main__":
    quick_test()