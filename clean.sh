#!/bin/bash

echo "🧹 Limpiando archivos temporales..."

# Limpiar bases de datos
if [ -d "chroma_db" ]; then
    echo "🗑️ Eliminando base de datos principal..."
    rm -rf chroma_db
fi

if [ -d "chroma_db_test" ]; then
    echo "🗑️ Eliminando base de datos de prueba..."
    rm -rf chroma_db_test
fi

# Limpiar archivos comprimidos
if [ -f "chroma_db_compressed.tar.gz" ]; then
    echo "🗑️ Eliminando archivo comprimido..."
    rm chroma_db_compressed.tar.gz
fi

# Limpiar cache de Python
echo "🧹 Limpiando cache de Python..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Limpieza completada"