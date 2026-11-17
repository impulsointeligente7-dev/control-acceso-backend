import os
import re
import base64
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Control Acceso - Gemini Debug")

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
    modelo_usado: Optional[str] = None
    debug: Optional[str] = None

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

def limpiar_placa(texto: str) -> str:
    if not texto:
        return ""
    t = texto.upper().replace(" ","").replace("-","")
    m = re.search(r'F?[A-Z]?[0-9]{1,4}[A-Z]{2,3}|[A-Z]{1,2}[0-9]{3,4}[A-Z]{0,3}', t)
    if m:
        return re.sub(r'[^A-Z0-9]', '', m.group(0))
    return re.sub(r'[^A-Z0-9]', '', t)[:10]

@app.get("/")
def root():
    return {
        "status": "Control Acceso Backend OK - GEMINI DEBUG",
        "model": "gemini-1.5-flash",
        "gemini_configurada": bool(GEMINI_KEY),
        "gemini_preview": GEMINI_KEY[:15]+"..." if GEMINI_KEY else "NO KEY",
        "key_length": len(GEMINI_KEY) if GEMINI_KEY else 0,
        "key_start": GEMINI_KEY[:5] if GEMINI_KEY else ""
    }

@app.post("/api/leer-placa", response_model=PlacaResponse)
def leer_placa(req: ImagenRequest):
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="Falta GEMINI_API_KEY en Render Environment")

    image_data = req.image
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen base64 invalida: {e}")

    # Intentar Gemini con debug detallado
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        
        modelos = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
        
        for modelo in modelos:
            try:
                print(f"Probando {modelo} con key {GEMINI_KEY[:10]}...")
                model = genai.GenerativeModel(modelo)
                response = model.generate_content([
                    "Lee la placa del auto. Placa mexicana Coahuila A-000-AAA. Solo la placa sin guiones. Ejemplo A000AAA. Solo responde la placa.",
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ])
                raw = response.text.strip()
                placa = limpiar_placa(raw)
                print(f"✅ {modelo}: {raw} -> {placa}")
                if placa and len(placa) >= 4:
                    return PlacaResponse(placa=placa, confianza=0.98, raw_text=raw, modelo_usado=modelo, debug=f"OK con {modelo}")
            except Exception as e:
                err = str(e)
                print(f"❌ {modelo}: {err}")
                # Si es error de API no habilitada, decirlo claro
                if "API_KEY_INVALID" in err or "API key" in err or "not found" in err.lower():
                    raise HTTPException(status_code=500, detail=f"GEMINI API KEY invalida o API no habilitada: {err} | Ve a console.cloud.google.com, busca Generative Language API y habilitala para proyecto 102711703969")
                if "quota" in err.lower() or "limit" in err.lower():
                    raise HTTPException(status_code=500, detail=f"Cuota Gemini excedida: {err}")
                continue
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importando google-generativeai o configurando: {e}")

    raise HTTPException(status_code=500, detail=f"Todos los modelos Gemini fallaron. Revisa logs de Render en dashboard.render.com para ver error exacto. Key preview: {GEMINI_KEY[:15]}...")
