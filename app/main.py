from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from src.inference.transcribe import Transcriber
from src.inference.summarize import Summarizer
import time

app = FastAPI(title="Whisper Summarizer API")

# Initialisation des deux modèles
print("Chargement des modèles (Whisper + Summarizer)...")
transcriber = Transcriber(model_size="tiny") 
summarizer = Summarizer(model_name="sshleifer/distilbart-cnn-12-6")
print("Modèles prêts !")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}

@app.post("/transcribe")
async def process_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Format non supporté")

    temp_path = f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        start_time = time.time()
        
        # 1. Transcription
        print(f"Transcription de {file.filename}...")
        transcript = transcriber.transcribe(temp_path)
        
        # 2. Résumé
        print("Génération du résumé...")
        summary = summarizer.summarize(transcript)
        
        processing_time = time.time() - start_time
        
        return {
            "filename": file.filename,
            "transcription": transcript,
            "summary": summary,
            "processing_time_sec": round(processing_time, 2)
        }
    
    except Exception as e:
        print(f"Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
