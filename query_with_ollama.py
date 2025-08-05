from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Configuración
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"  # Cambiar por el nombre del modelo que tengas cargado

# Cargar base de datos Chroma
print("📦 Cargando base vectorial...")
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)

# Inicializar modelo local vía Ollama
llm = Ollama(model=OLLAMA_MODEL)

# Construir cadena RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Interfaz de pregunta-respuesta simple
print("🧠 Preguntale a la Invencible (escribí 'salir' para terminar)")
while True:
    pregunta = input("🗨️  Pregunta: ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        break
    respuesta = qa_chain.run(pregunta)
    print(f"💡 Respuesta:\n{respuesta}\n")
