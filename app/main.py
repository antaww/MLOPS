from fastapi.responses import HTMLResponse
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Header, Depends
import shutil
import os
from src.inference.transcribe import Transcriber
from src.inference.answerer import Answerer
import time
import json
import redis
from typing import Optional

app = FastAPI(title="Whisper QA API")

# Configuration de la sécurité
API_TOKENS = os.getenv("API_TOKENS", "dev-token-123,admin-token-456").split(",")

async def verify_token(x_token: Optional[str] = Header(None)):
    if not x_token or x_token not in API_TOKENS:
        raise HTTPException(status_code=401, detail="Token manquant ou invalide")
    return x_token

def track_usage(token: str):
    if redis_client:
        try:
            redis_client.incr(f"user_usage:{token}")
        except Exception as e:
            print(f"Erreur lors du tracking Redis: {e}")

# Connexion à Redis pour le cache (facultatif, repli sur dict si absent)
try:
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("Connecté à Redis pour le cache.")
except Exception:
    redis_client = None
    print("Redis non disponible, utilisation du cache en mémoire (dict).")

transcription_cache = {}

print("Chargement des modèles (Whisper TURBO + Answerer LARGE)...")
transcriber = Transcriber(model_size="large-v3-turbo") 
answerer = Answerer(model_name="Qwen/Qwen2.5-1.5B-Instruct")
print("Modèles prêts !")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio QA - Benchmarking</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #f1f5f9; }
            .glass { background: rgba(255, 255, 255, 0.98); backdrop-filter: blur(10px); }
            .custom-scrollbar::-webkit-scrollbar { width: 6px; }
            .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
            .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
        </style>
    </head>
    <body class="min-h-screen p-8 flex flex-col items-center">
        <div class="max-w-4xl w-full space-y-8 glass p-10 rounded-3xl shadow-2xl border border-slate-200">
            <div class="flex justify-between items-start">
                <div>
                    <h1 class="text-4xl font-extrabold text-slate-900 tracking-tight">Audio QA Benchmark</h1>
                    <p class="text-slate-500 mt-2">Analysez vos audios et testez la précision du modèle.</p>
                </div>
                <div class="text-right">
                    <span class="px-3 py-1 bg-green-100 text-green-700 text-xs font-bold rounded-full uppercase">Whisper Turbo + T5-Large</span>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Token API (Header X-Token)</label>
                        <input type="password" id="apiToken" placeholder="Votre token secret..." 
                            class="w-full p-3 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:outline-none transition-all">
                    </div>
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">URL du fichier MP3</label>
                        <input type="text" id="url" placeholder="https://..." 
                            class="w-full p-3 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:outline-none transition-all">
                    </div>
                    <div class="relative flex py-2 items-center">
                        <div class="flex-grow border-t border-slate-200"></div>
                        <span class="flex-shrink mx-4 text-slate-400 text-xs font-bold uppercase">OU</span>
                        <div class="flex-grow border-t border-slate-200"></div>
                    </div>
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Fichier local</label>
                        <input type="file" id="audioFile" accept=".mp3,.wav,.m4a" 
                            class="w-full text-sm p-2 border border-dashed border-slate-300 rounded-xl">
                    </div>
                </div>

                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-semibold text-slate-700 mb-1">Votre question</label>
                        <textarea id="question" rows="4" placeholder="Posez une question..." 
                            class="w-full p-3 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:outline-none transition-all"></textarea>
                    </div>
                </div>
            </div>

            <div class="flex gap-4">
                <button onclick="askQuestion()" id="submitBtn"
                    class="flex-1 py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-2xl shadow-lg transform transition active:scale-95 flex items-center justify-center">
                    Analyser & Répondre
                </button>
                <button onclick="runBenchmark()" id="testBtn"
                    class="px-6 py-4 bg-slate-800 hover:bg-black text-white font-bold rounded-2xl shadow-lg transform transition active:scale-95 flex items-center justify-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path></svg>
                    Lancer le Test (10 questions)
                </button>
            </div>

            <div id="loader" class="hidden flex flex-col items-center space-y-2 pt-4">
                <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
                <p id="loaderText" class="text-sm text-slate-500 italic">Analyse en cours...</p>
            </div>

            <!-- Zone de résultats Benchmark -->
            <div id="benchmarkResults" class="hidden space-y-4 pt-8 border-t border-slate-200">
                <h3 class="text-xl font-bold text-slate-900">Résultats du Test Automatique</h3>
                <div id="testGrid" class="grid grid-cols-1 gap-4">
                    <!-- Les tests s'afficheront ici -->
                </div>
            </div>

            <!-- Résultat unique -->
            <div id="resultArea" class="hidden space-y-6 pt-8 border-t border-slate-200">
                <div class="bg-blue-50 p-6 rounded-2xl border border-blue-100 shadow-sm">
                    <div class="flex justify-between items-center mb-2">
                        <h2 class="text-blue-800 text-xs font-bold uppercase tracking-wider">Réponse du modèle</h2>
                        <span id="langBadge" class="px-2 py-0.5 bg-blue-100 text-blue-600 text-[10px] font-black rounded-full uppercase"></span>
                    </div>
                    <p id="answer" class="text-slate-900 text-xl leading-relaxed font-semibold"></p>
                </div>
                <div class="bg-slate-50 p-6 rounded-2xl border border-slate-100">
                    <h2 class="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">Transcription complète</h2>
                    <p id="transcription" class="text-slate-600 text-sm leading-relaxed max-h-60 overflow-y-auto pr-2 custom-scrollbar italic"></p>
                </div>
            </div>
        </div>

        <script>
            const TEST_QUESTIONS = [
                "Qui sont les deux personnages principaux ?",
                "Dans quel magasin se trouvent-ils (parodie) ?",
                "Quel est le prix total de la caution pour louer la camionnette ?",
                "Cites les plats typiques du terroir suédois servis au restaurant.",
                "Quel est le nom du canapé mentionné au début par Ludo ?",
                "Pourquoi Ludo n'a-t-il pas pris le modèle Villasso ?",
                "Que cherche précisément la cliente qui crie des 'Ch'taumeuls' ?",
                "Quel est le prix de la location toutes les demi-heures ?",
                "Qu'est-ce que les petits gitans ont volé au magasin ?",
                "Quelle est la conclusion finale de Nico sur la mondialisation ?"
            ];

            // Charger le token depuis le localStorage au démarrage
            document.addEventListener('DOMContentLoaded', () => {
                const savedToken = localStorage.getItem('api_token');
                if (savedToken) document.getElementById('apiToken').value = savedToken;
            });

            async function runBenchmark() {
                const token = document.getElementById('apiToken').value;
                if (!token) { alert('Veuillez entrer un token API'); return; }
                localStorage.setItem('api_token', token);

                const url = document.getElementById('url').value;
                const fileInput = document.getElementById('audioFile');
                if (!url && fileInput.files.length === 0) { alert('Veuillez charger un audio d abord'); return; }

                const testGrid = document.getElementById('testGrid');
                const benchmarkResults = document.getElementById('benchmarkResults');
                const resultArea = document.getElementById('resultArea');
                const loader = document.getElementById('loader');
                const loaderText = document.getElementById('loaderText');

                testGrid.innerHTML = '';
                benchmarkResults.classList.remove('hidden');
                resultArea.classList.add('hidden');
                loader.classList.remove('hidden');

                for (let i = 0; i < TEST_QUESTIONS.length; i++) {
                    const q = TEST_QUESTIONS[i];
                    loaderText.innerText = `Question ${i+1}/${TEST_QUESTIONS.length} : ${q}`;
                    
                    const formData = new FormData();
                    if (url) formData.append('url', url);
                    else formData.append('file', fileInput.files[0]);
                    formData.append('question', q);

                    try {
                        const res = await fetch('/ask-audio', { 
                            method: 'POST', 
                            body: formData,
                            headers: { 'X-Token': token }
                        });
                        const data = await res.json();
                        
                        if (!res.ok) throw new Error(data.detail || 'Erreur API');

                        const card = document.createElement('div');
                        card.className = "p-4 bg-white border border-slate-200 rounded-xl shadow-sm";
                        card.innerHTML = `
                            <p class="text-xs font-bold text-slate-400 uppercase mb-1">Q${i+1}: ${q}</p>
                            <p class="text-slate-800 font-medium">${data.answer}</p>
                        `;
                        testGrid.appendChild(card);
                    } catch (e) {
                        alert(`Erreur : ${e.message}`);
                        break;
                    }
                }
                loader.classList.add('hidden');
            }

            async function askQuestion() {
                const token = document.getElementById('apiToken').value;
                if (!token) { alert('Veuillez entrer un token API'); return; }
                localStorage.setItem('api_token', token);

                const url = document.getElementById('url').value;
                const fileInput = document.getElementById('audioFile');
                const question = document.getElementById('question').value;
                const btn = document.getElementById('submitBtn');
                const loader = document.getElementById('loader');
                const resultArea = document.getElementById('resultArea');
                const benchmarkResults = document.getElementById('benchmarkResults');

                if (!url && fileInput.files.length === 0) { alert('Audio manquant'); return; }

                btn.disabled = true;
                loader.classList.remove('hidden');
                resultArea.classList.add('hidden');
                benchmarkResults.classList.add('hidden');

                const formData = new FormData();
                if (url) formData.append('url', url);
                else formData.append('file', fileInput.files[0]);
                formData.append('question', question || "De quoi parle cet audio ?");

                try {
                    const response = await fetch('/ask-audio', { 
                        method: 'POST', 
                        body: formData,
                        headers: { 'X-Token': token }
                    });
                    const data = await response.json();
                    if (response.ok) {
                        document.getElementById('answer').innerText = data.answer;
                        document.getElementById('transcription').innerText = data.transcription;
                        document.getElementById('langBadge').innerText = data.language + " | " + Math.round(data.audio_duration) + "s";
                        resultArea.classList.remove('hidden');
                    } else {
                        alert(`Erreur : ${data.detail || 'Inconnue'}`);
                    }
                } catch (err) { alert('Erreur réseau ou serveur'); } 
                finally { btn.disabled = false; loader.classList.add('hidden'); }
            }
        </script>
    </body>
    </html>
    """

@app.post("/ask-audio")
async def process_audio(
    file: UploadFile = File(None), 
    url: str = Form(None),
    question: str = Form("De quoi parle cet audio ?"),
    token: str = Depends(verify_token)
):
    track_usage(token)
    if not file and not url:
        raise HTTPException(status_code=400, detail="Audio manquant")

    cache_key = url if url else f"local_{file.filename}_{file.size}"
    transcript = None
    duration = None
    language = None

    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            cached_data = json.loads(cached)
            transcript = cached_data["transcript"]
            duration = cached_data["duration"]
            language = cached_data["language"]
    elif cache_key in transcription_cache:
        cached_data = transcription_cache[cache_key]
        transcript = cached_data["transcript"]
        duration = cached_data["duration"]
        language = cached_data["language"]

    temp_path = None
    try:
        if not transcript:
            if file:
                temp_path = f"temp_{int(time.time())}_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            else:
                temp_path = f"temp_downloaded_{int(time.time())}.mp3"
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, follow_redirects=True)
                    with open(temp_path, "wb") as f:
                        f.write(response.content)
            
            transcript, duration, language = transcriber.transcribe(temp_path)
            data_to_cache = {"transcript": transcript, "duration": duration, "language": language}
            if redis_client:
                redis_client.set(cache_key, json.dumps(data_to_cache))
            else:
                transcription_cache[cache_key] = data_to_cache
        
        answer = answerer.ask(transcript, question)
        return {
            "question": question,
            "transcription": transcript,
            "answer": answer,
            "audio_duration": duration,
            "language": language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
