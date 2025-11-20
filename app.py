import traceback
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


# LangChain y vectores
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline


from transformers import pipeline
from difflib import get_close_matches
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
NOMBRE_MODELO = "moondream"


TMP_DIR = os.path.join(os.environ["USERPROFILE"], "tmp_ollama")
os.makedirs(TMP_DIR, exist_ok=True)

# Archivo con información de medicamentos
MEDS_FILE = 'medicamentos.json'
with open(MEDS_FILE, encoding="utf-8") as f:
    meds = json.load(f)

# Embeddings y vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Solo nombres de los medicamentos en embeddings
texts = list(meds.keys())
vectorstore = FAISS.from_texts(texts, embedding_model)


# Flask app
app = Flask(__name__)


# Endpoint texto
@app.route('/predict_text', methods=['POST'])
def predict_text():
    import traceback
    import json

    data = request.get_json()
    if not data:
        return jsonify({"error": "El cuerpo no es JSON"}), 400

    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "No se envió pregunta"}), 400

    try:
        # --- 1. BUSCAR MEDICAMENTOS SIMILARES ---
        resultados = []
        datoscontexto = []
        nombres_agregados = set()

        med_name = clean_med_name(pregunta)
        docs = vectorstore.similarity_search(med_name, k=1)

        for d in docs:
            nombre = d.page_content.strip()
            if nombre.lower() not in nombres_agregados and nombre in meds:
                info = meds[nombre]
                datoscontexto.append({
                    "medication": nombre,
                    "informacion": info
                })
                nombres_agregados.add(nombre.lower())

        # --- 2. Construir contexto ---
        if not datoscontexto:
            contexto = "No se encontró información en la base de datos de medicamentos."
        else:
            contexto = "\n".join(
                [f"- {r['medication']}: {r['informacion']}" for r in datoscontexto]
            )

        # --- 3. Preparar prompt para Ollama ---
        prompt = f"""
Eres un asistente experto en medicamentos.
Responde en español de forma clara, precisa y útil.
Solo usa la información proporcionada en el CONTEXTO. 
No agregues información de fuera de la base de datos.

PREGUNTA DEL USUARIO:
{pregunta}

CONTEXTO (información real de mi base de datos):
{contexto}

Instrucciones:
- Responde en 2 o 3 frases como máximo.
- Sé conciso y directo.
- Coloca al final de tu respuesta la información encontrada en la base de datos anteponiendo la frase "BASE PROPIA:".
- Si no encuentras algo para decir en el contexto, responde: "No hay información suficiente en la base de datos."
"""

        payload = {"model": MODEL_NAME, "prompt": prompt}

        # --- 4. Llamada a Ollama ---
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)

        # --- 5. Manejar streaming JSON ---
        frase_completa = ""
        lines = response.text.strip().splitlines()

        for line in lines:
            try:
                data_line = json.loads(line)
                if "response" in data_line:
                    frase_completa += data_line["response"]
            except json.JSONDecodeError:
                continue  # ignorar líneas inválidas

        frase_completa = frase_completa.strip()

        # --- 6. Responder con JSON  ---
        resultados.append({
            "medication": nombre,
            "informacion": frase_completa
        })  

        return jsonify({"resultados": resultados})      

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500



def es_palabra_valida(texto):
    # Quitar espacios al inicio y fin
    texto = texto.strip()
    # No aceptar si tiene más de 3 letras iguales consecutivas
    if re.search(r'(.)\1{3,}', texto):
        return False
    # Opcional: descartar si no tiene ninguna vocal
    if not re.search(r'[aeiouáéíóú]', texto.lower()):
        return False
    return True

# Endpoint archivo
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
        docs = vectorstore.similarity_search(med_name, k=3)  

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

        return jsonify({"resultados": resultados})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Funciones auxiliares
def send_to_model(image: Image.Image):
    

    tmp_path = os.path.join(TMP_DIR, "tmp_ollama.png")
    image.save(tmp_path, format="PNG")
    print("Archivo guardado:", os.path.exists(tmp_path))

    prompt_text = (
        "Analiza la imagen ubicada en: "
        f"{tmp_path}\n"
        "Idioma: español.\n"
        "❗ IMPORTANTE:\n"
        "Devuelve SOLO el nombre del medicamento, SIN dosis, SIN texto adicional."
    )

    print("Prompt enviado a Ollama:\n", prompt_text)

    result = subprocess.run(
        ["ollama", "run", "moondream", prompt_text],
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error al ejecutar Ollama: {result.stderr}")

    text = result.stdout.strip()
    print("Texto detectado por OCR:", text)
    return text


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
         

# Ejecutar app
if __name__ == '__main__':
    print("Servidor corriendo en http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000)
