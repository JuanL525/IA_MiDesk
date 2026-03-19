import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
from google import genai
import requests
import math 
from typing import List
import base64

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Microservicio IA - MiDesk (Google Gemini)")

SYSTEM_PROMPT = """
Eres el núcleo de Inteligencia Artificial de MiDesk, un sistema operativo virtual.
Tu objetivo es analizar lo que pide el usuario y ejecutar comandos en el sistema.

REGLA ESTRICTA DE CONTEXTO: Si el usuario te pregunta cosas que no tienen NADA que ver con MiDesk, organización, productividad o el uso de la PC, debes rechazar la solicitud amablemente recordando que eres el asistente de MiDesk.

DEBES responder ÚNICAMENTE con un objeto JSON válido con esta estructura exacta:
{
    "mensaje": "Tu respuesta amigable",
    "comando": "abrir" o "crear_nota" o "ninguno",
    "apps": ["lista", "de", "apps", "solo", "si", "el", "comando", "es", "abrir"],
    "contenido_nota": "El texto a guardar, solo si el comando es crear_nota. Vacío en otros casos."
}

Las aplicaciones disponibles en el sistema son: "notas", "calculadora", "navegador", "tareas", "terminal", "archivos".

Ejemplo 1 (Abrir):
Usuario: "Voy a programar, ábreme la terminal"
Respuesta: {"mensaje": "¡Entendido! Abriendo la terminal.", "comando": "abrir", "apps": ["terminal"], "contenido_nota": ""}

Ejemplo 2 (Crear Nota):
Usuario: "Anota que mañana tengo que entregar el informe de física a las 10am"
Respuesta: {"mensaje": "¡Listo! He creado una nota con tu recordatorio para mañana.", "comando": "crear_nota", "apps": [], "contenido_nota": "Entregar informe de física mañana a las 10am"}

Ejemplo 3 (Fuera de contexto):
Usuario: "¿Cómo se hace una pizza?"
Respuesta: {"mensaje": "Lo siento, soy el asistente exclusivo de MiDesk. Solo puedo ayudarte a organizar tu entorno de trabajo, crear notas o abrir aplicaciones.", "comando": "ninguno", "apps": [], "contenido_nota": ""}
"""

class ChatRequest(BaseModel):
    mensaje: str

@app.post("/chat")
def chat(req: ChatRequest):
    inicio = time.time()
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "GOOGLE_API_KEY no encontrada en .env"}

        client = genai.Client(api_key=api_key)
        prompt = f"{SYSTEM_PROMPT}\nUsuario: {req.mensaje}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=dict(
                response_mime_type="application/json",
                temperature=0.2 
            )
        )

        respuesta_ia_json = json.loads(response.text)

        fin = time.time()

        return {
            "respuesta": respuesta_ia_json,
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    except Exception as e:
        return {
            "error": "Fallo al generar respuesta de IA",
            "detalle": str(e)
        }

class FondoRequest(BaseModel):
    descripcion: str

@app.post("/generar-fondo")
def generar_fondo(req: FondoRequest):
    inicio = time.time()
    try:
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            return {"error": "HUGGINGFACE_API_KEY no encontrada en .env"}

        API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_api_key}"}

        prompt_final = f"desktop wallpaper, highly detailed, masterpiece, {req.descripcion}"

        respuesta_hf = requests.post(
            API_URL, 
            headers=headers, 
            json={
                "inputs": prompt_final,
                "parameters": {
                    "width": 1024,
                    "height": 576
                }
            }
        )

        if respuesta_hf.status_code != 200:
            return {
                "error": "Error en los servidores de Hugging Face", 
                "detalle": respuesta_hf.text
            }

        imagen_bytes = respuesta_hf.content

        with open("fondo_prueba.jpg", "wb") as archivo_imagen:
            archivo_imagen.write(imagen_bytes)
        
        imagen_base64 = base64.b64encode(imagen_bytes).decode('utf-8')
        formato_datos = f"data:image/jpeg;base64,{imagen_base64}"

        fin = time.time()

        return {
            "mensaje": "Fondo generado con éxito",
            "imagen": formato_datos,
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    except Exception as e:
        return {
            "error": "Fallo al generar el fondo de pantalla",
            "detalle": str(e)
        }


class ArchivoVirtual(BaseModel):
    id: str
    nombre: str
    contenido: str

class BusquedaRequest(BaseModel):
    consulta: str
    archivos: List[ArchivoVirtual]

# --- FUNCIÓN MATEMÁTICA PARA COMPARAR SIGNIFICADOS ---
def similitud_coseno(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(x * x for x in vec1))
    magnitude_v2 = math.sqrt(sum(x * x for x in vec2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return dot_product / (magnitude_v1 * magnitude_v2)

@app.post("/buscar")
def buscar_archivos(req: BusquedaRequest):
    inicio = time.time()
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "GOOGLE_API_KEY no encontrada en .env"}

        client = genai.Client(api_key=api_key)
        
        # 1. Convertimos la búsqueda del usuario a un vector numérico (Embedding)
        respuesta_consulta = client.models.embed_content(
            model="gemini-embedding-001", 
            contents=req.consulta
        )
        vector_consulta = respuesta_consulta.embeddings[0].values

        resultados = []

        # 2. Analizamos cada archivo del usuario
        for archivo in req.archivos:
            # Combinamos el nombre y el contenido para entender mejor de qué trata
            texto_archivo = f"Título: {archivo.nombre}. Contenido: {archivo.contenido}"
            
            # Convertimos el archivo a vector
            respuesta_archivo = client.models.embed_content(
                model="gemini-embedding-001", 
                contents=texto_archivo
            )
            vector_archivo = respuesta_archivo.embeddings[0].values
            
            # 3. Comparamos matemáticamente qué tan parecidos son sus significados
            similitud = similitud_coseno(vector_consulta, vector_archivo)
            
            # Guardamos el resultado con un porcentaje (0 a 100)
            porcentaje = round(similitud * 100, 2)
            resultados.append({
                "id": archivo.id,
                "nombre": archivo.nombre,
                "relevancia": porcentaje
            })

        # 4. Ordenamos de mayor a menor relevancia
        resultados_ordenados = sorted(resultados, key=lambda x: x["relevancia"], reverse=True)

        fin = time.time()

        return {
            "mensaje": "Búsqueda semántica completada",
            "resultados": resultados_ordenados,
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    except Exception as e:
        return {
            "error": "Fallo al realizar la búsqueda semántica",
            "detalle": str(e)
        }