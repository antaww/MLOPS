# MLOps Project: Whisper Transcription & Summarization

Ce projet vise à mettre en production un système de transcription audio (via OpenAI Whisper) couplé à un service de résumé de texte.

## Niveau 0 : Lancement local

Le système tourne entièrement dans des conteneurs Docker (API FastAPI + Redis pour le cache).

### Prérequis
- Docker Desktop installé sur Windows.
- (Optionnel) Make pour Windows (si disponible).

### Lancer le projet
Si vous avez `make` :
```bash
make run
```

Sinon (standard Windows) :
```powershell
docker-compose up --build
```

### Utilisation
L'API est accessible sur `http://localhost:8000`.
La documentation interactive Swagger est disponible sur `http://localhost:8000/docs`.

## Structure du projet
- `app/` : API FastAPI.
- `src/inference/` : Modèles Whisper et Summarizer.
- `deploy/` : Fichiers de configuration pour le déploiement.

$filePath = "c:\Users\antap\PycharmProjects\MLOPS\data\raw\test_2.wav"
>> $question = "what the link between capuccino and hitchcock and herman?"
>>
>> curl.exe -X POST "http://localhost:8000/ask-audio" `
>>   -H "accept: application/json" `
>>   -H "Content-Type: multipart/form-data" `
>>   -F "file=@$filePath" `
>>   -F "question=$question"

## activer gpu : 
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

python -c "import torch; print('GPU disponible :', torch.cuda.is_available()); print('Nom du GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun')"
