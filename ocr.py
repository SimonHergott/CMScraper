import cv2
import base64
import numpy as np
import ollama

class VisionOCR:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionOCR, cls).__new__(cls)
            cls._instance.model_name = "qwen3.5:9b-q4_K_M"
            cls._instance.client = ollama.Client() # Defaults to localhost:11434
        return cls._instance

    def encode_image(self, frame):
        """Convertit une frame OpenCV (BGR) en base64 pour le modèle."""
        _, buffer = cv2.imencode(".png", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def process_ocr(self, frame, prompt="Extract the text visible in this image accurately."):
        """
        Prend une frame, renvoie la string prédite via Ollama API / VLM.
        """
        if frame is None or frame.size == 0:
            return ""

        base64_image = self.encode_image(frame)
        
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }
        ]

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": 0.1,
                "num_predict": 800
            }
        )

        return response["message"]["content"].strip()

# Instance globale pour l'import
ocr_engine = VisionOCR()

def image_to_string_vlm(frame, context_type="text"):
    """
    Fonction transparente pour remplacer Tesseract.
    context_type peut aider le modèle (ex: 'name', 'poll_option', 'percentage')
    """
    prompts = {
        "text": "Write only the text found in this image. No comments.",
        "name": "Extract the person's name or username from this image.",
        "poll": "Identify the poll question and options in this image.",
        "percentage": "What is the percentage value shown? Return only the number, without the percent symbol."
    }
    
    selected_prompt = prompts.get(context_type, prompts["text"])
    return ocr_engine.process_ocr(frame, prompt=selected_prompt)