from faster_whisper import WhisperModel
import os
import torch

class Transcriber:
    def __init__(self, model_size="large-v3-turbo", device=None, compute_type=None):
        # Auto-détection du GPU (CUDA)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimisation du type de calcul
        if compute_type is None:
            # float16 est optimal pour le GPU, int8 pour le CPU
            compute_type = "float16" if device == "cuda" else "int8"
            
        print(f"--- Whisper configuré sur : {device.upper()} ({compute_type}) ---")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        # Retourne le texte, la durée et la langue détectée
        return text.strip(), info.duration, info.language
