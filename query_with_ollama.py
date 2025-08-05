import subprocess
import time
import requests
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Configuración
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
OLLAMA_HOST = "http://localhost:11434"

# Verificar si Ollama ya está corriendo
def is_ollama_running():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return response.status_code == 200
    except Exception:
        return False

# Intentar iniciar Ollama si no está corriendo
def ensure_ollama_running():
    if not is_ollama_running():
        print("🚀 Iniciando servidor Ollama...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(5)  # Esperar a que levante

# Main
def main():
    ensure_ollama_running()

    print("📦 Cargando base vectorial...")
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)

    llm = Ollama(model=OLLAMA_MODEL)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    print("🧠 Preguntale a la Invencible (escribí 'salir' para terminar)")
    while True:
        pregunta = input("🗨️  Pregunta: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        respuesta = qa_chain.run(pregunta)
        print(f"💡 Respuesta:\n{respuesta}\n")

if __name__ == "__main__":
    main()
