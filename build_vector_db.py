import os
import json
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
BATCH_SIZE = 50  # Procesar archivos PDF de a 50
PROGRESS_FILE = "processing_progress.json"  # Archivo para guardar progreso

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

# Cargar progreso previo
def load_progress():
    """Cargar el progreso de procesamiento anterior"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed_files": [], "last_batch": 0}

# Guardar progreso
def save_progress(processed_files: List[str], batch_num: int):
    """Guardar el progreso actual"""
    progress = {
        "processed_files": processed_files,
        "last_batch": batch_num,
        "total_files": len(processed_files)
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

# Procesar PDFs en lotes con persistencia incremental
def process_and_store_in_batches(pdf_dir: str, output_dir: str):
    """Procesar PDFs en lotes y almacenar incrementalmente en la base vectorial"""
    
    # Cargar progreso anterior
    progress = load_progress()
    processed_files = set(progress["processed_files"])
    
    # Obtener todos los PDFs
    all_pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    pdf_files = [f for f in all_pdf_files if str(f) not in processed_files]
    
    print(f"📁 Total archivos PDF encontrados: {len(all_pdf_files)}")
    print(f"📄 Archivos ya procesados: {len(processed_files)}")
    print(f"🆕 Archivos pendientes: {len(pdf_files)}")
    
    if not pdf_files:
        print("✅ Todos los archivos ya han sido procesados")
        return
    
    # Crear embeddings una sola vez
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )
    
    # Cargar base vectorial existente si existe
    vectorstore = None
    if os.path.exists(output_dir):
        try:
            vectorstore = Chroma(persist_directory=output_dir, embedding_function=embeddings)
            print("📚 Base vectorial existente cargada")
        except Exception as e:
            print(f"⚠️ Error cargando base existente: {e}")
            print("🆕 Se creará una nueva base vectorial")
    
    # Procesar archivos en lotes
    total_batches = (len(pdf_files) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📦 Procesando {len(pdf_files)} archivos en {total_batches} lotes de máximo {BATCH_SIZE}")
    
    for i in range(0, len(pdf_files), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1 + progress["last_batch"]
        batch_files = pdf_files[i:i + BATCH_SIZE]
        
        print(f"\n🔄 === LOTE {batch_num} ({len(batch_files)} archivos) ===")
        
        try:
            # Procesar PDFs del lote en paralelo
            batch_docs = []
            with ProcessPoolExecutor() as executor:
                results = executor.map(process_pdf_with_ocr, batch_files)
                for doc_list in results:
                    batch_docs.extend(doc_list)
            
            if not batch_docs:
                print("⚠️ No se extrajeron documentos en este lote")
                continue
            
            # Dividir en chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(batch_docs)
            print(f"📊 Generados {len(chunks)} chunks para este lote")
            
            # Almacenar en base vectorial
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    chunks, 
                    embeddings, 
                    persist_directory=output_dir
                )
                print(f"✅ Base vectorial creada con {len(chunks)} chunks")
            else:
                vectorstore.add_documents(chunks)
                print(f"✅ Añadidos {len(chunks)} chunks a la base existente")
            
            # Persistir inmediatamente
            vectorstore.persist()
            print(f"💾 Lote {batch_num} persistido en base vectorial")
            
            # Actualizar progreso
            new_processed = [str(f) for f in batch_files]
            all_processed = list(processed_files) + new_processed
            save_progress(all_processed, batch_num)
            processed_files.update(new_processed)
            
            print(f"📝 Progreso guardado: {len(all_processed)} archivos procesados")
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Interrupción detectada en lote {batch_num}")
            print(f"📊 Progreso guardado hasta el lote anterior")
            break
            
        except Exception as e:
            print(f"❌ Error en lote {batch_num}: {str(e)}")
            print("⏭️ Continuando con el siguiente lote...")
            continue
    
    print(f"\n✅ Procesamiento completado")
    print(f"📁 Base vectorial disponible en: {output_dir}")
    
    return vectorstore

# Función para limpiar progreso y empezar de cero
def reset_progress():
    """Limpiar todo el progreso y empezar de cero"""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
    print("🧹 Progreso limpiado. Empezando de cero...")

# Comprimir base
def compress_chroma_db(db_dir: str, output_file: str):
    if not os.path.exists(db_dir):
        print(f"❌ No existe el directorio: {db_dir}")
        return
    
    print(f"📦 Comprimiendo {db_dir}...")
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(db_dir, arcname=os.path.basename(db_dir))
    print(f"✅ Base comprimida en: {output_file}")

# Mostrar estado actual
def show_status():
    """Mostrar el estado actual del procesamiento"""
    if os.path.exists(PROGRESS_FILE):
        progress = load_progress()
        print(f"📊 ESTADO ACTUAL:")
        print(f"   Archivos procesados: {progress['total_files']}")
        print(f"   Último lote: {progress['last_batch']}")
    else:
        print("📊 No hay progreso previo")
    
    if os.path.exists(CHROMA_DB_DIR):
        print(f"📚 Base vectorial existe en: {CHROMA_DB_DIR}")
    else:
        print("📚 No existe base vectorial")

# Pipeline completo con persistencia incremental
def run_pipeline(reset: bool = False):
    if reset:
        reset_progress()
    
    print("🚀 Iniciando procesamiento incremental")
    show_status()
    
    try:
        vectorstore = process_and_store_in_batches(PDF_DIR, CHROMA_DB_DIR)
        
        if vectorstore and os.path.exists(CHROMA_DB_DIR):
            compress_chroma_db(CHROMA_DB_DIR, CHROMA_DB_TAR)
            print("🎉 Pipeline completado exitosamente")
        
    except KeyboardInterrupt:
        print("\n⏹️ Procesamiento interrumpido por el usuario")
        print("💾 El progreso ha sido guardado")
        show_status()

if __name__ == "__main__":
    import sys
    
    # Permitir reset desde línea de comandos
    reset = "--reset" in sys.argv
    
    if "--status" in sys.argv:
        show_status()
    else:
        run_pipeline(reset=reset)