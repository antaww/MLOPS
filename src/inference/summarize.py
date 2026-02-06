from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def summarize(self, text):
        if not text or len(text.strip()) < 10:
            return ""
        
        # Tokenisation
        inputs = self.tokenizer(text[:4000], return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Génération du résumé
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                max_length=130, 
                min_length=30, 
                do_sample=False,
                num_beams=4
            )
        
        # Décodage
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
