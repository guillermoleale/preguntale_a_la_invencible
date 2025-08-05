#!/bin/bash

echo "🧠 Preguntale a la Invencible"
echo "============================="

# 1. Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "📦 Activando entorno virtual..."
    source venv/bin/activate
else
    echo "⚠️ No se encontró entorno virtual. Se usará Python global."
fi

# 2. Verificar si Ollama está corriendo
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama ya está corriendo"
else
    echo "🚀 Iniciando servidor Ollama..."
    ollama serve &> /dev/null &
    sleep 5
fi

# 3. Ejecutar la app de preguntas
echo "🧠 Iniciando consola interactiva..."
python query_with_ollama.py
