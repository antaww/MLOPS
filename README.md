# MLOps Project: Whisper Transcription & Summarization

Ce projet vise à mettre en production un système de transcription audio (via OpenAI Whisper) couplé à un service de résumé de texte.

## Niveau 0 : Lancement local

Le système tourne entièrement dans des conteneurs Docker.

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
