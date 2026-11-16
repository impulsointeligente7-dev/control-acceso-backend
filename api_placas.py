import os
import re
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Control Acceso")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagenRequest(BaseModel):
    image: str

class PlacaResponse(BaseModel):
    placa: Optional[str] = None
    confianza: float = 0.0
    raw_text: Optional[str] = None
    error: Optional[str] = None

API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLAMA_API_KEY") or ""
BASE_URL = "https://api.groq.com/openai/v1"

# MODELOS QUE SI ESTAN DISPONIBLES EN GROQ (en orden de preferencia)
MODELOS_DISPONIBLES = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout - mas disponible
    "llama-3.2-11b-vision-preview",  # Llama 3.2 Vision 11B - fallback seguro
    "llama-3.2-90b-vision-preview",  # Llama 3.2 Vision 90B
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # Maverick - si tienes acceso
]

def limpiar_placa(texto: str) -> str:
    if not texto:
        return ""
    t = texto.upper()
    m = re.search(r'[A-Z]{1,2}[-]?[0-9]{2,4}[-]?[A-Z0-9]{2,4}|[A-Z]{3}[-\s]?[0-9]{3,4}|F[-]?[0-9]{1,3}[-]?[A-Z]{2,3}', t)
    if m:
        return re.sub(r'[^A-Z0-9]', '', m.group(0))[:10]
    # Fallback: buscar A-000-AAA
    m2 = re.search(r'[A-Z]\s*-\s*0{3}\s*-\s*[A-Z]{3}', t)
    if m2:
        return re.sub(r'[^A-Z0-9]', '', m2.group(0))
    return re.sub(r'[^A-Z0-9]', '', t)[:10]

@app.get("/")
def root():
    return {
        "status": "Control Acceso Backend OK",
        "model": MODELOS_DISPONIBLES[0],
        "modelos_probados": MODELOS_DISPONIBLES,
        "base_url": BASE_URL,
        "key_configurada": bool(API_KEY),
        "instrucciones": "Ve a console.groq.com para sacar tu key gratis"
    }

@app.post("/api/leer-placa", response_model=PlacaResponse)
def leer_placa(req: ImagenRequest):
    if not API_KEY:
        return PlacaResponse(placa=None, error="Falta GROQ_API_KEY en .env - Configura en Render Environment Variables")

    last_error = None
    for MODEL in MODELOS_DISPONIBLES:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
            image_url = req.image if req.image.startswith("data:image") else f"data:image/jpeg;base64,{req.image}"
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres lector de placas mexicanas. Extrae SOLO la placa visible. Responde JSON exacto: {\"placa\": \"A000AAA\"} o {\"placa\": \"ABC123\"}. Sin texto extra. Si no ves placa, responde {\"placa\": null}"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Lee la placa de este auto. Solo la placa, formato A-000-AAA o similar. Responde JSON."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=50
            )
            raw = response.choices[0].message.content
            import json
            try:
                cleaned = re.sub(r'```json|```', '', raw).strip()
                data = json.loads(cleaned)
                placa = data.get("placa")
            except:
                placa = limpiar_placa(raw)
            
            placa_limpia = limpiar_placa(placa) if placa else None
            if placa_limpia and len(placa_limpia) >= 4:
                return PlacaResponse(placa=placa_limpia, confianza=0.95, raw_text=f"Modelo {MODEL}: {raw}")
            else:
                last_error = f"Modelo {MODEL} no detecto placa: {raw}"
                continue
                
        except Exception as e:
            err_str = str(e)
            last_error = f"Modelo {MODEL} error: {err_str}"
            # Si es error 404 de modelo no existe, probar siguiente
            if "does not exist" in err_str or "404" in err_str or "model_not_found" in err_str:
                print(f"Modelo {MODEL} no disponible, probando siguiente...")
                continue
            else:
                # Si es otro error, también probar siguiente modelo
                print(f"Error con {MODEL}: {err_str}")
                continue
    
    # Si ninguno funciono
    raise HTTPException(status_code=500, detail=last_error or "No se pudo leer placa con ningun modelo")
