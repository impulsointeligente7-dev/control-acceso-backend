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
    modelo_usado: Optional[str] = None

API_KEY = os.getenv("GROQ_API_KEY") or ""
BASE_URL = "https://api.groq.com/openai/v1"

# GROQ CAMBIO TODO HOY - Llama 3.2 decommissioned, solo quedan Llama 4
# Probamos TODAS las variantes de nombre posibles
MODELOS_DISPONIBLES = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-4-scout-17b-16e-instruct",
    "llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-3.2-11b-vision-preview",
    "meta-llama/llama-3.2-90b-vision-preview",
]

def limpiar_placa(texto: str) -> str:
    if not texto:
        return ""
    t = texto.upper().replace(" ","")
    # A-000-AAA
    m = re.search(r'F?[A-Z]-?0{3}-?[A-Z]{3}|[A-Z]{3}-?[0-9]{3,4}', t)
    if m:
        return re.sub(r'[^A-Z0-9]', '', m.group(0))
    return re.sub(r'[^A-Z0-9]', '', t)[:10]

@app.get("/")
def root():
    return {
        "status": "Control Acceso Backend OK",
        "model": MODELOS_DISPONIBLES[0],
        "modelos_probados": MODELOS_DISPONIBLES,
        "base_url": BASE_URL,
        "key_configurada": bool(API_KEY),
        "key_preview": API_KEY[:10]+"..." if API_KEY else "NO KEY",
        "nota": "Groq decommissioned Llama 3.2 vision, usar Llama 4"
    }

@app.post("/api/leer-placa", response_model=PlacaResponse)
def leer_placa(req: ImagenRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Falta GROQ_API_KEY")

    errores = []
    for MODEL in MODELOS_DISPONIBLES:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
            image_url = req.image if req.image.startswith("data:image") else f"data:image/jpeg;base64,{req.image}"
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Lee la placa del auto. Es Coahuila A-000-AAA. Solo la placa sin guiones. Ejemplo: A000AAA. Solo responde la placa."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=30
            )
            raw = response.choices[0].message.content.strip()
            placa_limpia = limpiar_placa(raw)
            print(f"✅ {MODEL} -> {raw} -> {placa_limpia}")
            
            if placa_limpia and len(placa_limpia) >= 5:
                return PlacaResponse(placa=placa_limpia, confianza=0.95, raw_text=f"{MODEL}: {raw}", modelo_usado=MODEL)
            else:
                errores.append(f"{MODEL}: sin placa, raw='{raw}'")
                
        except Exception as e:
            err = str(e)[:300]
            errores.append(f"{MODEL}: {err}")
            print(f"❌ {MODEL}: {err}")
            continue
    
    raise HTTPException(status_code=500, detail=" | ".join(errores))
