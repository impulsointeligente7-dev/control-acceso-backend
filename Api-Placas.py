
"""
Backend Control Acceso - VERSION FACIL con GROQ (si no encuentras la key de Meta)
Groq es 100% gratis, te da la key en 10 segundos y usa el mismo Llama 4 Maverick
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, re, base64
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

app = FastAPI(title="Control Acceso - Placas con Groq/Llama", version="V16-GROQ")

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

# CONFIG - Puedes usar GROQ (gratis y fácil) o META oficial
# Para GROQ: api_key = gsk_..., base_url = https://api.groq.com/openai/v1, model = meta-llama/llama-4-maverick-17b-128e-instruct
# Para META: api_key = LLM_..., base_url = https://api.llama.com/compat/v1, model = Llama-4-Maverick-17B-128E-Instruct-FP8

API_KEY = os.getenv("LLAMA_API_KEY", os.getenv("GROQ_API_KEY", os.getenv("META_LLAMA_API_KEY", "")))
BASE_URL = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")  # por defecto GROQ porque es más fácil
MODEL = os.getenv("MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

def limpiar_placa(texto: str) -> str:
    if not texto:
        return ""
    t = texto.upper()
    patterns = [r'[A-Z]{3}[-\s]?[0-9]{3,4}', r'[0-9]{3,4}[-\s]?[A-Z]{3}']
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            placa = re.sub(r'[^A-Z0-9]', '', m.group(0))
            if 4 <= len(placa) <= 10:
                return placa[:10]
    limpio = re.sub(r'[^A-Z0-9]', '', t)
    return limpio[:10] if 4 <= len(limpio) <= 10 else ""

@app.get("/")
def root():
    return {"status": "Control Acceso Backend OK", "model": MODEL, "base_url": BASE_URL, "key_configurada": bool(API_KEY), "instrucciones": "Ve a console.groq.com para sacar tu key gratis"}

@app.post("/api/leer-placa", response_model=PlacaResponse)
def leer_placa(req: ImagenRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="No hay API Key. Configura GROQ_API_KEY en variables de entorno. Sácala gratis en https://console.groq.com/keys")
    if not req.image:
        raise HTTPException(status_code=400, detail="Imagen vacía")

    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        image_url = req.image if req.image.startswith("data:image") else f"data:image/jpeg;base64,{req.image}"

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Eres ANPR para México. Extrae SOLO el número de placa. Responde JSON {\"placa\": \"ABC1234\", \"confianza\": 0.95}. Si no ves placa, {\"placa\": null}. No inventes."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extrae la placa. Solo JSON."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ],
            temperature=0.1,
            max_tokens=100
        )
        raw = response.choices[0].message.content
        import json
        raw_clean = re.sub(r'```json|```', '', raw).strip()
        try:
            data = json.loads(raw_clean)
            placa = data.get("placa")
            conf = float(data.get("confianza", 0.9))
        except:
            placa = limpiar_placa(raw)
            conf = 0.7

        placa_limpia = limpiar_placa(placa) if placa else None
        return PlacaResponse(placa=placa_limpia, confianza=conf, raw_text=raw)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
