from faster_whisper import WhisperModel
import os

class Transcriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        # On utilise int8 pour la quantification (Niveau 3 anticipé) et la vitesse
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
