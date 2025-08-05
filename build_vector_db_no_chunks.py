import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List
import shutil
import tarfile

import pdfplumber
import pytesseract
from PIL import Image

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Cambio aquí: usar langchain_huggingface en lugar de langchain_community
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuración
PDF_DIR = "documents"
CHROMA_DB_DIR = "chroma_db"
CHROMA_DB_TAR = "chroma_db_compressed.tar.gz"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 50  # Procesar chunks de a 50


# Procesar un único PDF con OCR
def process_pdf_with_ocr(pdf_path: Path) -> List[Document]:
    print(f"📖 Procesando: {pdf_path.name}")
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text or len(text.strip()) < 10:
                # Si no hay texto legible, aplicar OCR
                image = page.to_image(resolution=300)
                pil_img: Image.Image = image.original
                text = pytesseract.image_to_string(pil_img, lang="spa+eng")
            if text and len(text.strip()) > 0:
                docs.append(Document(page_content=text, metadata={"source": str(pdf_path), "page": i + 1}))
    return docs

# Procesar todos los PDFs en paralelo
def process_all_pdfs(pdf_dir: str) -> List[Document]:
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    print(f"📁 Encontrados {len(pdf_files)} archivos PDF")
    all_docs = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_pdf_with_ocr, pdf_files)
        for doc_list in results:
            all_docs.extend(doc_list)
    return all_docs

# Dividir en chunks y construir base vectorial
# Dividir en chunks y construir base vectorial
def build_chroma_db(docs: List[Document], output_dir: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"📊 Generando embeddings para {len(chunks)} fragmentos...")
    
    # Usar configuración optimizada para embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )
    
    # Procesar en lotes para evitar problemas de memoria
    vectorstore = None
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📦 Procesando en {total_batches} lotes de máximo {BATCH_SIZE} chunks")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch = chunks[i:i + BATCH_SIZE]
        print(f"🔄 Procesando lote {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(batch, embeddings, persist_directory=output_dir)
            print(f"✅ Base de datos creada con {len(batch)} chunks")
        else:
            vectorstore.add_documents(batch)
            print(f"✅ Añadidos {len(batch)} chunks a la base existente")
        
        # Persistir después de cada lote para seguridad
        vectorstore.persist()
        print(f"💾 Lote {batch_num} persistido")

    return vectorstore

# Comprimir base
def compress_chroma_db(db_dir: str, output_file: str):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(db_dir, arcname=os.path.basename(db_dir))

# Pipeline completo
def run_pipeline():
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    print("🔍 Procesando PDFs con OCR...")
    docs = process_all_pdfs(PDF_DIR)
    print(f"📄 Documentos procesados: {len(docs)}")
    print("🔗 Generando embeddings y base vectorial...")
    build_chroma_db(docs, CHROMA_DB_DIR)
    print("📦 Comprimiendo base...")
    compress_chroma_db(CHROMA_DB_DIR, CHROMA_DB_TAR)
    print(f"✅ Listo. Base comprimida en: {CHROMA_DB_TAR}")

if __name__ == "__main__":
    run_pipeline()
