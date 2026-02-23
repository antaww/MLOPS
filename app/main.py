from fastapi.responses import HTMLResponse
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
import shutil
import os
from src.inference.transcribe import Transcriber
from src.inference.answerer import Answerer
import time

app = FastAPI(title="Whisper QA API")

# Cache simple pour éviter de re-transcrire le même audio
# Format: { "url_ou_filename": {"transcript": "...", "duration": 123.4} }
transcription_cache = {}

# Initialisation des modèles
print("Chargement des modèles (Whisper + Answerer)...")
transcriber = Transcriber(model_size="base") 
answerer = Answerer(model_name="google/flan-t5-base")
print("Modèles prêts !")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio QA - Assistant Intelligent</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
            .glass { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); }
            .custom-scrollbar::-webkit-scrollbar { width: 6px; }
            .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
            .custom-scrollbar::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
        </style>
    </head>
    <body class="min-h-screen flex items-center justify-center p-4">
        <div class="max-w-2xl w-full space-y-8 glass p-8 rounded-3xl shadow-2xl border border-slate-200">
            <div class="text-center">
                <h1 class="text-4xl font-extrabold text-slate-900 tracking-tight mb-2">Audio QA</h1>
                <p class="text-slate-500">Posez des questions sur vos fichiers audio via URL</p>
            </div>

            <div class="space-y-4">
                <div>
                    <div class="flex justify-between items-center mb-1">
                        <label class="block text-sm font-semibold text-slate-700">URL du fichier MP3</label>
                        <span id="durationBadge" class="hidden px-2 py-0.5 bg-slate-100 text-slate-600 text-xs font-bold rounded-full"></span>
                    </div>
                    <input type="text" id="url" placeholder="https://exemple.com/audio.mp3" 
                        class="w-full p-3 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:outline-none transition-all">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-1">Votre question</label>
                    <input type="text" id="question" placeholder="De quoi parle cet audio ?" 
                        class="w-full p-3 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:outline-none transition-all">
                </div>
                <button onclick="askQuestion()" id="submitBtn"
                    class="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg transform transition active:scale-95 flex items-center justify-center">
                    <span>Analyser et Répondre</span>
                </button>
            </div>

            <!-- Loader -->
            <div id="loader" class="hidden flex flex-col items-center space-y-2">
                <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
                <p class="text-sm text-slate-500 italic">Traitement de l'audio (cela peut prendre quelques secondes)...</p>
            </div>

            <!-- Résultats -->
            <div id="resultArea" class="hidden space-y-6 pt-6 border-t border-slate-100">
                <div class="bg-blue-50 p-6 rounded-2xl border border-blue-100">
                    <h2 class="text-blue-800 text-xs font-bold uppercase tracking-wider mb-2">Réponse</h2>
                    <p id="answer" class="text-slate-900 text-lg leading-relaxed font-medium"></p>
                </div>
                <div>
                    <h2 class="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">Transcription</h2>
                    <p id="transcription" class="text-slate-600 text-sm leading-relaxed max-h-48 overflow-y-auto pr-2 custom-scrollbar"></p>
                </div>
            </div>
        </div>

        <script>
            function formatDuration(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.round(seconds % 60);
                return `${mins}:${secs.toString().padStart(2, '0')}`;
            }

            async function askQuestion() {
                const url = document.getElementById('url').value;
                const question = document.getElementById('question').value;
                const btn = document.getElementById('submitBtn');
                const loader = document.getElementById('loader');
                const resultArea = document.getElementById('resultArea');
                const durationBadge = document.getElementById('durationBadge');

                if (!url) { alert('Veuillez entrer une URL'); return; }

                // UI State
                btn.disabled = true;
                btn.classList.add('opacity-50');
                loader.classList.remove('hidden');
                resultArea.classList.add('hidden');

                const formData = new FormData();
                formData.append('url', url);
                formData.append('question', question || "De quoi parle cet audio ?");

                try {
                    const response = await fetch('/ask-audio', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    if (response.ok) {
                        document.getElementById('answer').innerText = data.answer;
                        document.getElementById('transcription').innerText = data.transcription;
                        
                        // Affichage de la durée
                        if (data.audio_duration) {
                            durationBadge.innerText = "Durée: " + formatDuration(data.audio_duration);
                            durationBadge.classList.remove('hidden');
                        }
                        
                        resultArea.classList.remove('hidden');
                    } else {
                        alert('Erreur: ' + (data.detail || 'Une erreur est survenue'));
                    }
                } catch (err) {
                    alert('Erreur réseau ou serveur');
                } finally {
                    btn.disabled = false;
                    btn.classList.remove('opacity-50');
                    loader.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/ask-audio")
async def process_audio(
    file: UploadFile = File(None), 
    url: str = Form(None),
    question: str = Form("De quoi parle cet audio ?")
):
    if not file and not url:
        raise HTTPException(status_code=400, detail="Vous devez fournir soit un fichier, soit une URL.")

    # Identifiant unique pour le cache (URL ou nom de fichier)
    cache_key = url if url else file.filename
    transcript = None
    duration = None

    # Vérification du cache
    if cache_key in transcription_cache:
        print(f"Utilisation de la transcription en cache pour : {cache_key}")
        cached_data = transcription_cache[cache_key]
        transcript = cached_data["transcript"]
        duration = cached_data["duration"]

    temp_path = None
    
    try:
        # Si pas en cache, on télécharge/lit et on transcrit
        if not transcript:
            if file:
                if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
                    raise HTTPException(status_code=400, detail="Format de fichier non supporté")
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            else:
                temp_path = f"temp_downloaded_{int(time.time())}.mp3"
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, follow_redirects=True)
                    if response.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"Impossible de télécharger l'audio (Status: {response.status_code})")
                    with open(temp_path, "wb") as f:
                        f.write(response.content)
            
            # 1. Transcription
            print(f"Transcription en cours...")
            transcript, duration = transcriber.transcribe(temp_path)
            
            # Mise en cache
            transcription_cache[cache_key] = {
                "transcript": transcript,
                "duration": duration
            }
        
        start_time = time.time()
        
        # 2. Answer
        print(f"Réponse à la question : {question}")
        answer = answerer.ask(transcript, question)
        
        processing_time = time.time() - start_time
        
        return {
            "filename": cache_key.split("/")[-1],
            "question": question,
            "transcription": transcript,
            "answer": answer,
            "audio_duration": duration,
            "processing_time_sec": round(processing_time, 2)
        }
    
    except Exception as e:
        print(f"Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
