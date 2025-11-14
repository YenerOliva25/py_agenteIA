import base64
import io
import subprocess
import tempfile
import re
import os
import json
from flask import Flask, request, jsonify
import requests
from PIL import Image
from difflib import get_close_matches

# --------------------------
# LangChain y vectores
# --------------------------
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import pipeline
from difflib import get_close_matches

# --------------------------
# Archivo con información de medicamentos
# --------------------------
MEDS_FILE = 'medicamentos.json'
with open(MEDS_FILE, encoding="utf-8") as f:
    meds = json.load(f)

# --------------------------
# Embeddings y vector store
# --------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
##vectorstore = FAISS.from_texts(texts, embedding_model)
# Solo nombres de los medicamentos en embeddings
texts = list(meds.keys())
vectorstore = FAISS.from_texts(texts, embedding_model)


# --------------------------
# LLM para generar respuestas
# --------------------------
hf_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=128)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"  # concatena los textos recuperados
)

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)


# --------------------------
# Endpoint texto
# --------------------------
@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.get_json()
    if not data:
        return jsonify({"error": "El cuerpo no es JSON"}), 400

    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "No se envió pregunta"}), 400

    try:
        resultados = []
        nombres_agregados = set()

        # Buscar top 10 más similares y luego tomar 3
        docs = vectorstore.similarity_search(pregunta, k=10)

        for d in docs:
            nombre = d.page_content.strip()
            if nombre.lower() not in nombres_agregados and nombre in meds:
                info = meds[nombre]
                resultados.append({
                    "medication": nombre,
                    "informacion": info
                })
                nombres_agregados.add(nombre.lower())
            if len(resultados) >= 3:
                break

        if not resultados:
            resultados.append({
                "medication": "No identificado",
                "informacion": "No se encontró información relevante"
            })

        return jsonify({"resultados": resultados})

    except Exception as e:
        return jsonify({'error': str(e)}), 500




# ===============================================================
#  FUNCIÓN PROCESAR TEXTO
# ===============================================================
def procesar_texto(pregunta: str):
    """
    Recibe texto como: "para qué sirve la acetaminofén"
    Devuelve:
        {
           "medication": "Acetaminofen",
           "informacion": "Texto..."
        }
    """

    # 1. Detectar nombre del medicamento usando embeddings
    docs = vectorstore.similarity_search(pregunta, k=1)
    if not docs:
        return {
            "medication": "No identificado",
            "informacion": "No se encontró información relevante."
        }

    resultados = []
    for d in docs:
        nombre = d.page_content.strip()
        info = meds.get(nombre, "Información no encontrada")
        resultados.append({
            "medication": nombre,
            "informacion": info
        })

    return {"resultados": resultados}

# --------------------------
# Endpoint archivo
# --------------------------
@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        try:
            image = Image.open(file.stream)
        except Exception as e:
            return jsonify({'error': f'Error al abrir la imagen: {str(e)}'}), 400

        # Predicción multimodal
        med_name_raw = send_to_model(image)
        med_name = clean_med_name(med_name_raw)

        med_info = meds.get(med_name, "Información no encontrada")
        resultados = []
        nombres_agregados = set()

        # Agregar predicción exacta primero
        resultados.append({
            "medication": med_name,
            "informacion": med_info
        })
        nombres_agregados.add(med_name.lower())

        # Buscar top 3 similares usando FAISS
        docs = vectorstore.similarity_search(med_name, k=10)  # buscar varios para filtrar luego

        for d in docs:
            nombre = d.page_content.strip()
            if nombre.lower() not in nombres_agregados and nombre in meds:
                info = meds[nombre]
                resultados.append({
                    "medication": nombre,
                    "informacion": info
                })
                nombres_agregados.add(nombre.lower())
            if len(resultados) >= 3:
                break  # ya tenemos 3 resultados

        return jsonify({"resultados": resultados})

    except Exception as e:
        return jsonify({'error': str(e)}), 500





# --------------------------
# Funciones auxiliares
# --------------------------
def send_to_model(image: Image.Image):
    fmt = image.format or "PNG"
    ext = fmt.lower()
    if ext == "jpeg":
        ext = "jpg"

    tmp_path = os.path.join(tempfile.gettempdir(), f"tmp_ollama.{ext}")
    image.save(tmp_path, format=fmt)

    try:
        prompt = f"""
        Analiza la imagen ubicada en: {tmp_path}.
        Idioma: español.
        ❗ IMPORTANTE:
        Devuelve SOLO el nombre del medicamento, SIN dosis, SIN texto adicional.
        """
        print(prompt)
        result = subprocess.run(
            ["ollama", "run", "moondream", prompt],
            capture_output=True,
            encoding="utf-8",
            errors="ignore"
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error al ejecutar Ollama: {result.stderr}")

        text = result.stdout.strip()
        print("Texto detectado por OCR:", text)
        return text

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_med_info(predicted_name: str):
    if not predicted_name or predicted_name.strip() == "":
        return {"informacion": "Nombre no detectado"}

    name = predicted_name.lower().strip()

    # FAISS (similaridad por embeddings)
    docs = vectorstore.similarity_search(name, k=1)
    if docs:
        best_match = docs[0].page_content.strip()
        return meds.get(best_match, "Información no encontrada")

    return "Información no encontrada"



def clean_med_name(raw_name: str):
    if not raw_name:
        return "Nombre no detectado"

    name = raw_name.lower()

    # Quitar lo que no sirve (mg, tabletas, jarabe, etc)
    name = re.sub(r"\d+(mg|ml)?", "", name)
    name = re.sub(r"(gel|tabletas|capsulas|jarabe|suspensión|bebés|bebes)", "", name)
    name = re.sub(r"[^a-zA-Záéíóúñü]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    print("Nombre después de limpieza:", name)

    # 1. Buscar con FAISS (vector search)
    similar = vectorstore.similarity_search(name, k=1)
    if similar:
        best_match = similar[0].page_content.strip()
        print("Match por FAISS:", best_match)
        return best_match

    # 2. Fuzzy matching por caracteres (resuelve errores como IDSiNoFen → Acetaminofen)
    candidates = list(meds.keys())
    fuzzy = get_close_matches(name.capitalize(), candidates, n=1, cutoff=0.3)
    if fuzzy:
        print("Match por fuzzy:", fuzzy[0])
        return fuzzy[0]

    return "Nombre no encontrado"
         

# --------------------------
# Ejecutar app
# --------------------------
if __name__ == '__main__':
    print("Servidor corriendo en http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000)
