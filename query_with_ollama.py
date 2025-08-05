import subprocess
import time
import requests
import os
import warnings
from pathlib import Path

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)

# Importaciones actualizadas para evitar deprecaciones
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuración
CHROMA_DB_DIR = "chroma_db"
CHROMA_DB_TEST_DIR = "chroma_db_test"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
OLLAMA_HOST = "http://localhost:11434"

# Template de prompt mejorado en español para mayor flexibilidad
SPANISH_PROMPT_TEMPLATE = """
Eres un asistente experto que analiza documentos y responde preguntas en español de manera inteligente y flexible.

Contexto disponible:
{context}

Pregunta: {question}

Instrucciones:
- Responde SIEMPRE en español de manera clara y útil
- Analiza todo el contexto disponible para dar una respuesta completa
- Si la pregunta es general (como "temas importantes", "resumen", "principales puntos"), identifica y presenta los temas más relevantes del contexto
- Si la pregunta es específica, busca información relacionada aunque no sea exactamente literal
- Organiza tu respuesta de manera estructurada cuando sea apropiado (listas, puntos, etc.)
- Si realmente no hay información relacionada, entonces di "No tengo información suficiente"
- Siempre intenta ser útil e interpretativo

Respuesta completa en español:
"""

def verificar_dependencias():
    """Verificar que las dependencias estén correctamente instaladas"""
    try:
        import langchain_huggingface
        import langchain_chroma
        import langchain_ollama
        return True
    except ImportError as e:
        print(f"❌ Error de dependencias: {e}")
        print("💡 Ejecuta: pip install langchain-huggingface langchain-chroma langchain-ollama")
        return False

def is_ollama_running():
    """Verificar si Ollama ya está corriendo"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def verificar_modelo_ollama():
    """Verificar que el modelo de Ollama esté disponible"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "").split(":")[0] for model in models]
            return OLLAMA_MODEL in model_names
        return False
    except Exception:
        return False

def ensure_ollama_running():
    """Asegurar que Ollama esté corriendo y el modelo disponible"""
    if not is_ollama_running():
        print("🚀 Iniciando servidor Ollama...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)
        except FileNotFoundError:
            print("❌ Ollama no está instalado. Instálalo desde: https://ollama.ai")
            return False
    
    if not verificar_modelo_ollama():
        print(f"📥 Descargando modelo {OLLAMA_MODEL}...")
        try:
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error descargando el modelo {OLLAMA_MODEL}")
            return False
    
    print("✅ Ollama está corriendo y el modelo está disponible")
    return True

def cargar_base_vectorial():
    """Cargar la base vectorial con manejo de errores"""
    # Buscar base de datos disponible
    db_dir = None
    if os.path.exists(CHROMA_DB_DIR):
        db_dir = CHROMA_DB_DIR
        print(f"📚 Usando base vectorial principal: {CHROMA_DB_DIR}")
    elif os.path.exists(CHROMA_DB_TEST_DIR):
        db_dir = CHROMA_DB_TEST_DIR
        print(f"📚 Usando base vectorial de prueba: {CHROMA_DB_TEST_DIR}")
    else:
        print("❌ No se encontró ninguna base vectorial")
        print("💡 Ejecuta primero: python build_vector_db.py o python test_quick.py")
        return None
    
    try:
        print("🔗 Cargando embeddings...")
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Eliminar mensaje de carga
        # print("📦 Cargando base vectorial...")
        vectorstore = Chroma(
            persist_directory=db_dir, 
            embedding_function=embedding
        )
        
        # Verificar que la base no esté vacía
        test_results = vectorstore.similarity_search("test", k=1)
        if not test_results:
            print("⚠️ La base vectorial parece estar vacía")
            return None
        
        # Mostrar estadísticas de la base sin mensaje detallado
        sample_search = vectorstore.similarity_search("documento", k=5)
        print(f"✅ Base vectorial lista. Chequeo analizando ({len(sample_search)} documentos)")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error cargando base vectorial: {e}")
        return None

def detectar_tipo_pregunta(pregunta):
    """Detectar si es una pregunta general o específica para ajustar la búsqueda"""
    palabras_generales = [
        "principales", "importantes", "temas", "resumen", "contenido", 
        "qué contiene", "de qué trata", "principales puntos", "aspectos relevantes",
        "más importante", "destacado", "relevante", "significativo", "clave"
    ]
    
    pregunta_lower = pregunta.lower()
    es_general = any(palabra in pregunta_lower for palabra in palabras_generales)
    
    if es_general:
        return "general", 8  # Buscar más documentos para preguntas generales
    else:
        return "especifica", 4  # Buscar menos para preguntas específicas

def crear_cadena_qa(vectorstore):
    """Crear cadena de QA con prompt mejorado y búsqueda adaptativa"""
    try:
        # Configurar LLM con la nueva clase OllamaLLM
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.2,  # Ligeramente más creativo para interpretación
            num_predict=800,  # Más tokens para respuestas más completas
        )
        
        # Crear prompt template mejorado
        prompt = PromptTemplate(
            template=SPANISH_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        return llm, prompt, vectorstore
        
    except Exception as e:
        print(f"❌ Error creando componentes QA: {e}")
        return None, None, None

def procesar_pregunta(pregunta, llm, prompt, vectorstore):
    """Procesar pregunta con búsqueda adaptativa"""
    try:
        # Detectar tipo de pregunta
        tipo, k_docs = detectar_tipo_pregunta(pregunta)
        
        # Buscar documentos relevantes
        docs = vectorstore.similarity_search(pregunta, k=k_docs)
        
        if not docs:
            return "No se encontraron documentos relevantes para tu pregunta.", []
        
        # Preparar contexto
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Crear prompt completo
        prompt_text = prompt.format(context=context, question=pregunta)
        
        # Obtener respuesta del LLM usando invoke
        respuesta = llm.invoke(prompt_text)
        
        return respuesta, docs
        
    except Exception as e:
        print(f"❌ Error procesando pregunta: {e}")
        return "Error procesando la consulta.", []

def mostrar_fuentes(source_documents):
    """Mostrar fuentes de manera clara"""
    if not source_documents:
        return
    
    print("\n📚 Fuentes consultadas:")
    sources_set = set()
    for doc in source_documents:
        source = doc.metadata.get('source', 'Desconocido')
        page = doc.metadata.get('page', 'N/A')
        source_info = f"   • {Path(source).name} (Página {page})"
        sources_set.add(source_info)
    
    for source in sorted(sources_set):
        print(source)

def main():
    """Función principal mejorada"""
    print("🧠 Pregúntale a la Invencible")
    print("=" * 50)
    
    # Verificar dependencias
    if not verificar_dependencias():
        return
    
    # Asegurar que Ollama esté funcionando
    if not ensure_ollama_running():
        return
    
    # Cargar base vectorial
    vectorstore = cargar_base_vectorial()
    if not vectorstore:
        return
    
    # Crear componentes QA
    llm, prompt, vectorstore = crear_cadena_qa(vectorstore)
    if not llm:
        return
    
    print("\n✅ Sistema listo para consultas")
    print("💡 Tip: Puedes hacer preguntas generales ('temas importantes') o específicas")
    print("🚪 Escribe 'salir', 'exit' o 'quit' para terminar\n")
    
    try:
        while True:
            pregunta = input("🗨️  Pregunta: ").strip()
            
            if not pregunta:
                continue
                
            if pregunta.lower() in ["salir", "exit", "quit", "q"]:
                print("👋 ¡Hasta luego!")
                break
            
            print("🤔 Procesando...")
            respuesta, source_docs = procesar_pregunta(pregunta, llm, prompt, vectorstore)
            
            print(f"\n💡 Respuesta:")
            print(f"{respuesta}")
            
            mostrar_fuentes(source_docs)
            print("\n" + "-" * 50 + "\n")
                
    except KeyboardInterrupt:
        print("\n\n👋 Sesión interrumpida. ¡Hasta luego!")

if __name__ == "__main__":
    main()