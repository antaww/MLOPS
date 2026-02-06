from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Answerer:
    def __init__(self, model_name="google/flan-t5-small", device="cpu"):
        # Flan-T5 est excellent pour répondre à des questions sur un texte (RAG-like)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def ask(self, context, question):
        if not context or not question:
            return "Pas assez de données pour répondre."
        
        # On construit un prompt clair pour le modèle
        input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenisation
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Génération de la réponse
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"], 
                max_length=150,
                do_sample=False
            )
        
        # Décodage
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
