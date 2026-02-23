from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Answerer:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"--- Answerer LLM (Qwen 1.5B) configuré sur : {device.upper()} ---")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # On charge le modèle en auto (float16 sur GPU) pour la vitesse
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map=device
        )

    def ask(self, context, question):
        if not context or not question:
            return "Pas assez de données pour répondre."
        
        # On s'assure que le contexte n'est pas trop long pour le modèle (max 8k ici par sécurité)
        context_window = context[-8000:] if len(context) > 8000 else context

        # Utilisation du template de chat pour forcer le comportement d'assistant
        messages = [
            {"role": "system", "content": "Tu es un assistant qui analyse des transcriptions audio. Réponds de manière concise et précise en français. Si la réponse n'est pas dans le texte, dis-le."},
            {"role": "user", "content": f"Voici une transcription :\n{context_window}\n\nQuestion : {question}"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=150,
                do_sample=False # Déterministe pour le benchmark
            )
        
        # On décode uniquement la partie générée
        response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        answer = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
        
        return answer.strip()
