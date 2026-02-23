from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Answerer:
    def __init__(self, model_name="google/flan-t5-large", device=None):
        # Auto-détection du GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"--- Answerer configuré sur : {device.upper()} ---")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def ask(self, context, question):
        if not context or not question:
            return "Pas assez de données pour répondre."
        
        # STRATÉGIE : Question + Instruction en PREMIER pour éviter la troncation
        input_text = (
            f"Answer the following question using the context below.\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Answer:"
        )
        
        # Tokenisation (on garde le début du prompt qui contient la question)
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Génération
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"], 
                max_length=150,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sécurité : si le modèle ne donne rien de probant
        if not answer or len(answer.strip()) < 2:
            return "Désolé, je n'ai pas trouvé la réponse dans cet audio."
            
        return answer
