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

app = FastAPI(title="Control Acceso - Gemini Vision")

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

# Keys: GROQ sigue pero usamos GEMINI ahora
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GROQ_KEY = os.getenv("GROQ_API_KEY") or ""

def limpiar_placa(texto: str) -> str:
    if not texto:
        return ""
    t = texto.upper().replace(" ","").replace("-","")
    # Buscar F + numeros + letras o A-000-AAA
    m = re.search(r'F?[A-Z]?[0-9]{1,4}[A-Z]{2,3}|[A-Z]{1,2}[0-9]{3,4}[A-Z]{0,3}|A000AAA', t)
    if m:
        return re.sub(r'[^A-Z0-9]', '', m.group(0))
    # Extraer todo alfanumerico que parezca placa
    alfanum = re.sub(r'[^A-Z0-9]', '', t)
    # Si contiene F y numeros, es Coahuila
    if 'F' in alfanum:
        # Buscar patron F + 3 digitos + letras
        mf = re.search(r'F[0-9]{1,4}[A-Z]{2,4}', alfanum)
        if mf:
            return mf.group(0)
    return alfanum[:10]

@app.get("/")
def root():
    return {
        "status": "Control Acceso Backend OK - GEMINI",
        "model": "gemini-1.5-flash / gemini-2.0-flash",
        "gemini_configurada": bool(GEMINI_KEY),
        "groq_configurada": bool(GROQ_KEY),
        "gemini_preview": GEMINI_KEY[:10]+"..." if GEMINI_KEY else "NO KEY - Ve a aistudio.google.com",
        "nota": "Groq Llama Vision decommissioned, usando Gemini Flash que si lee placas gratis"
    }

@app.post("/api/leer-placa", response_model=PlacaResponse)
def leer_placa(req: ImagenRequest):
    if not GEMINI_KEY and not GROQ_KEY:
        raise HTTPException(status_code=500, detail="Falta GEMINI_API_KEY. Ve a aistudio.google.com/app/apikey y saca una gratis, ponla en Render Environment Variables como GEMINI_API_KEY")

    # Limpiar base64
    image_data = req.image
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen base64 invalida: {e}")

    # INTENTAR GEMINI PRIMERO (gratis y funciona)
    if GEMINI_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            
            # Probar modelos Gemini en orden
            modelos_gemini = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
            
            for modelo in modelos_gemini:
                try:
                    model = genai.GenerativeModel(modelo)
                    response = model.generate_content([
                        "Lee la placa del auto en la imagen. Es una placa mexicana de Coahuila que dice A-000-AAA o F-000-AAA o similar. Responde SOLO con la placa sin guiones, ejemplo: A000AAA o F123ABC. Solo la placa, nada mas.",
                        {"mime_type": "image/jpeg", "data": image_bytes}
                    ])
                    raw = response.text.strip()
                    placa_limpia = limpiar_placa(raw)
                    print(f"✅ Gemini {modelo}: {raw} -> {placa_limpia}")
                    if placa_limpia and len(placa_limpia) >= 4:
                        return PlacaResponse(placa=placa_limpia, confianza=0.98, raw_text=f"{modelo}: {raw}", modelo_usado=modelo)
                except Exception as e:
                    print(f"❌ Gemini {modelo}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error importando google-generativeai: {e}")
            # Continuar a intentar Groq si Gemini falla

    # FALLBACK GROQ si Gemini no esta o falla (por si habilitan Llama 4 luego)
    if GROQ_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
            # Intentar con modelos que Groq tenga en ese momento
            modelos_groq = ["llama-3.2-11b-vision-preview", "meta-llama/llama-4-scout-17b-16e-instruct"]
            for MODEL in modelos_groq:
                try:
                    import base64 as b64
                    data_url = f"data:image/jpeg;base64,{b64.b64encode(image_bytes).decode()}"
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": "Lee la placa, solo la placa"},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]}],
                        max_tokens=20,
                        temperature=0
                    )
                    raw = resp.choices[0].message.content.strip()
                    placa = limpiar_placa(raw)
                    if placa:
                        return PlacaResponse(placa=placa, confianza=0.9, raw_text=raw, modelo_usado=MODEL)
                except:
                    continue
        except:
            pass

    raise HTTPException(status_code=500, detail="No se pudo leer placa. Configura GEMINI_API_KEY en Render: ve a aistudio.google.com/app/apikey, saca key gratis y ponla como GEMINI_API_KEY en Render Environment Variables")
