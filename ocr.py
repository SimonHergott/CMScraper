import cv2
import base64
import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import LlamaVisionAdapter

class VisionOCR:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionOCR, cls).__new__(cls)
            cls._instance.llm = Llama(
                model_path="models/qwen2-9b-instruct-q4_k_m.gguf",
                n_gpu_layers=-1,
                n_ctx=4000,
                logits_all_cells=True,
                verbose=False
            )
            cls._instance.chat_handler = LlamaVisionAdapter() 
        return cls._instance

    def encode_image(self, frame):
        """Convertit une frame OpenCV (BGR) en base64 pour le modèle."""
        _, buffer = cv2.imencode(".png", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def process_ocr(self, frame, prompt="Extract the text visible in this image accurately."):
        """
        Remplace pytesseract.image_to_string.
        Prend une frame, renvoie la string prédite.
        """
        if frame is None or frame.size == 0:
            return ""

        base64_image = self.encode_image(frame)
        
        # Structure du message pour un VLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.1 # On reste très factuel pour l'OCR
        )

        return response["choices"][0]["message"]["content"].strip()

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
        "percentage": "What is the percentage value shown? Return only the number and % sign."
    }
    
    selected_prompt = prompts.get(context_type, prompts["text"])
    return ocr_engine.process_ocr(frame, prompt=selected_prompt)