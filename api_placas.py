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

app = FastAPI(title="Control Acceso - Gemini ListModels")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ImagenRequest(BaseModel):
    image: str
class PlacaResponse(BaseModel):
    placa: Optional[str]=None
    confianza: float=0.0
    raw_text: Optional[str]=None
    modelo_usado: Optional[str]=None
    debug: Optional[str]=None

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or ""

def limpiar_placa(texto: str) -> str:
    if not texto: return ""
    t = texto.upper().replace(" ","").replace("-","")
    m = re.search(r'F?[A-Z]?[0-9]{1,4}[A-Z]{2,3}|[A-Z]{1,2}[0-9]{3,4}[A-Z]{0,3}', t)
    if m: return re.sub(r'[^A-Z0-9]', '', m.group(0))
    return re.sub(r'[^A-Z0-9]', '', t)[:10]

@app.get("/")
def root():
    return {"status":"Control Acceso Backend OK - GEMINI LIST","gemini_configurada":bool(GEMINI_KEY),"key_preview":GEMINI_KEY[:15]+"..." if GEMINI_KEY else "NO KEY"}

@app.get("/api/modelos")
def listar_modelos():
    if not GEMINI_KEY:
        raise HTTPException(500, detail="Falta GEMINI_API_KEY")
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        modelos = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                modelos.append(m.name)
        return {"modelos_disponibles": modelos, "key_preview": GEMINI_KEY[:15]+"..."}
    except Exception as e:
        raise HTTPException(500, detail=f"Error listando modelos: {e}. Habilita Generative Language API en https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=102711703969")

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
        
        # Listar modelos disponibles primero
        try:
            disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            print(f"Modelos disponibles: {disponibles}")
        except Exception as e:
            disponibles = []
            print(f"Error listando: {e}")

        # Probar modelos en orden, con nombres completos models/...
        modelos_a_probar = []
        # Si tenemos lista de disponibles, probarlos primero
        if disponibles:
            modelos_a_probar.extend(disponibles[:5])
        # Fallback con nombres comunes
        modelos_a_probar.extend([
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-flash-8b",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-exp",
            "models/gemini-pro",
            "models/gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ])

        for modelo in modelos_a_probar:
            try:
                print(f"Probando {modelo}...")
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
                # Si es API no habilitada, dar link directo
                if "has not been used" in err or "is not enabled" in err or "API_KEY_INVALID" in err:
                    raise HTTPException(status_code=500, detail=f"API no habilitada para proyecto 102711703969: {err}. Habilita en https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=102711703969")
                continue

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error general: {e}")

    raise HTTPException(status_code=500, detail=f"Ningun modelo Gemini funciono. Modelos probados: {modelos_a_probar}. Ve a /api/modelos para ver lista. Habilita API en https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=102711703969")
