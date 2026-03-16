import time
from locust import HttpUser, task, between, events
import os

class WhisperQAUser(HttpUser):
    # Temps d'attente entre les tâches par utilisateur (1 à 5 secondes)
    wait_time = between(1, 5)
    
    # Header requis par l'API
    headers = {
        "X-Token": "dev-token-123"  # Token par défaut configuré dans app/main.py
    }

    @task
    def test_audio_qa_url(self):
        """Test de charge sur l'endpoint /ask-audio via une URL"""
        payload = {
            "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "question": "De quoi parle cet audio ?"
        }
        
        with self.client.post("/ask-audio", data=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Erreur {response.status_code}: {response.text}")

    @task(2)
    def test_root_page(self):
        """L'interface web (plus légère) est visitée plus souvent"""
        self.client.get("/")

# Événement au démarrage des tests
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"🚀 Démarrage du test de charge sur : {environment.host}")

# Événement à l'arrêt des tests
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("🛑 Fin du test de charge.")
