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

API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLAMA_API_KEY") or ""
BASE_URL = "https://api.groq.com/openai/v1"

# SOLO MODELOS 100% GRATIS Y DISPONIBLES PARA CUENTAS FREE DE GROQ
MODELOS_DISPONIBLES = [
    "llama-3.2-11b-vision-preview",  # ESTE SI ESTA LIBERADO PARA TODOS - Vision 11B
    "llama-3.2-90b-vision-preview",  # Vision 90B
]

def limpiar_placa(texto: str) -> str:
    if not texto:
        return ""
    t = texto.upper().replace(" ","")
    # Buscar A-000-AAA especifico
    m = re.search(r'F?[A-Z]-?0{3}-?[A-Z]{3}|[A-Z]{3}-?[0-9]{3,4}|[A-Z]-[0-9]{3}-[A-Z]{3}', t)
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
        "key_preview": API_KEY[:10]+"..." if API_KEY else "NO KEY"
    }

@app.post("/api/leer-placa", response_model=PlacaResponse)
def leer_placa(req: ImagenRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Falta GROQ_API_KEY en Render Environment Variables")

    errores = []
    for MODEL in MODELOS_DISPONIBLES:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
            image_url = req.image if req.image.startswith("data:image") else f"data:image/jpeg;base64,{req.image}"
            
            # Prompt super especifico para placa Coahuila A-000-AAA
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Lee la placa del auto. Es una placa de Coahuila que dice A-000-AAA. Extrae SOLO las letras y numeros sin guiones. Ejemplo: A000AAA. Responde SOLO con la placa, nada mas."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=20
            )
            raw = response.choices[0].message.content.strip()
            placa_limpia = limpiar_placa(raw)
            
            print(f"Modelo {MODEL} raw: {raw} -> limpia: {placa_limpia}")
            
            if placa_limpia and len(placa_limpia) >= 5:
                return PlacaResponse(
                    placa=placa_limpia, 
                    confianza=0.95, 
                    raw_text=f"Modelo {MODEL}: {raw}",
                    modelo_usado=MODEL
                )
            else:
                errores.append(f"{MODEL}: no detecto placa valida, raw='{raw}'")
                
        except Exception as e:
            err_str = str(e)
            errores.append(f"{MODEL}: {err_str}")
            print(f"Error {MODEL}: {err_str}")
            continue
    
    raise HTTPException(status_code=500, detail=" | ".join(errores))
