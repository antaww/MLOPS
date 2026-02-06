from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import shutil
import os
from src.inference.transcribe import Transcriber
from src.inference.answerer import Answerer
import time

app = FastAPI(title="Whisper QA API")

# Initialisation des modèles
print("Chargement des modèles (Whisper + Answerer)...")
transcriber = Transcriber(model_size="tiny") 
answerer = Answerer(model_name="google/flan-t5-small")
print("Modèles prêts !")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API QA is running"}

@app.post("/ask-audio")
async def process_audio(
    file: UploadFile = File(...), 
    question: str = Form("De quoi parle cet audio ?")
):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Format non supporté")

    temp_path = f"temp_{file.filename}"
    
    try:
        # Sauvegarde temporaire
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        start_time = time.time()
        
        # 1. Transcription
        print(f"Transcription de {file.filename}...")
        transcript = transcriber.transcribe(temp_path)
        
        # 2. Answer
        print(f"Réponse à la question : {question}")
        answer = answerer.ask(transcript, question)
        
        processing_time = time.time() - start_time
        
        return {
            "filename": file.filename,
            "question": question,
            "transcription": transcript,
            "answer": answer,
            "processing_time_sec": round(processing_time, 2)
        }
    
    except Exception as e:
        print(f"Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
