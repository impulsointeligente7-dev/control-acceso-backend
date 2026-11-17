import os, re, base64
from typing import Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Control Acceso - Gemini 2.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ImagenRequest(BaseModel):
    image: str
class PlacaResponse(BaseModel):
    placa: Optional[str]=None
    confianza: float=0.0
    raw_text: Optional[str]=None
    modelo_usado: Optional[str]=None

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or ""

def limpiar_placa(texto: str) -> str:
    if not texto: return ""
    t = texto.upper().replace(" ","").replace("-","")
    m = re.search(r'F?[A-Z]?[0-9]{1,4}[A-Z]{2,3}|[A-Z]{1,2}[0-9]{3,4}[A-Z]{0,3}|A000AAA', t)
    if m: return re.sub(r'[^A-Z0-9]', '', m.group(0))
    return re.sub(r'[^A-Z0-9]', '', t)[:10]

@app.get("/")
def root():
    return {"status":"Control Acceso Backend OK - GEMINI 2.5","model":"models/gemini-2.5-flash","gemini_configurada":bool(GEMINI_KEY),"key_preview":GEMINI_KEY[:15]+"..." if GEMINI_KEY else "NO KEY"}

@app.post("/api/leer-placa", response_model=PlacaResponse)
def leer_placa(req: ImagenRequest):
    if not GEMINI_KEY:
        raise HTTPException(500, detail="Falta GEMINI_API_KEY")
    image_data = req.image
    if "," in image_data: image_data = image_data.split(",")[1]
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(400, detail=f"Imagen invalida: {e}")

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        
        # Modelos que TU key si tiene segun tu captura de /api/modelos
        modelos = [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-flash-latest",
            "models/gemini-2.5-pro",
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash-preview-tts",
        ]
        
        for modelo in modelos:
            try:
                print(f"Probando {modelo}...")
                model = genai.GenerativeModel(modelo)
                response = model.generate_content([
                    "Eres lector de placas mexicanas. Lee la placa del auto en la imagen. Es Coahuila A-000-AAA o F-000-AAA. Responde SOLO con la placa sin guiones, ejemplo: A000AAA o F123ABC. Solo la placa.",
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ])
                raw = response.text.strip()
                placa = limpiar_placa(raw)
                print(f"✅ {modelo}: {raw} -> {placa}")
                if placa and len(placa) >= 4:
                    return PlacaResponse(placa=placa, confianza=0.98, raw_text=raw, modelo_usado=modelo)
            except Exception as e:
                print(f"❌ {modelo}: {e}")
                continue

    except Exception as e:
        raise HTTPException(500, detail=f"Error Gemini: {e}")

    raise HTTPException(500, detail="Ningun modelo Gemini pudo leer placa. Intenta con otra foto mas clara.")
