import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import pipeline

load_dotenv()

class ImageVerificationPipeline:
    def __init__(self):
        print("[*] Initializing Dual-Strategy Image Pipeline...")
        
        # 1. Primary Strategy: Local Edge Vision Model
        print("[*] Loading Vision Model to CPU/GPU. Please wait...")
        # This downloads the model to your machine once and runs locally. No API limits!
        self.local_classifier = pipeline("image-classification", model="prithivMLmods/Deepfake-Detect-Siglip2")
        
        # 2. Backup Strategy: Sightengine API
        self.se_user = os.getenv("SIGHTENGINE_USER")
        self.se_secret = os.getenv("SIGHTENGINE_SECRET")
        self.se_endpoint = "https://api.sightengine.com/1.0/check.json"

    def _primary_hf_analysis(self, image_path):
        """Strategy 1: Local Offline Vision Inference."""

        results = self.local_classifier(image_path)

        print(f"[*] Local Model Raw Response: {results}")

        fake_score = None
        real_score = None

        for item in results:
            label = str(item.get("label", "")).lower().strip()
            score = float(item.get("score", 0.0))

            if "fake" in label:
                fake_score = score

            elif "real" in label:
                real_score = score

        # If explicit Fake/Real labels are available
        if fake_score is not None:
            return fake_score * 100

        # Fallback if the model returns only class labels
        for item in results:
            label = str(item.get("label", "")).lower().strip()
            score = float(item.get("score", 0.0))

            if "class 0" in label:
                fake_score = score
            elif "class 1" in label:
                real_score = score

        if fake_score is not None:
            return fake_score * 100

        if real_score is not None:
            return (1.0 - real_score) * 100

        raise ValueError(f"Unable to determine Fake/Real score from model output: {results}")

    def _fallback_sightengine_analysis(self, image_path):
        """Strategy 2: Sightengine Commercial API"""
        print("   -> HF Unavailable. Triggering Strategy 2: Sightengine Fallback...")
        
        with open(image_path, 'rb') as f:
            files = {'media': f}
            data = {
                'models': 'genai',
                'api_user': self.se_user,
                'api_secret': self.se_secret
            }
            response = requests.post(self.se_endpoint, files=files, data=data)
            
        result_json = response.json()
        if result_json.get("status") == "success":
            # Extracts the global AI probability score
            ai_score = result_json["type"]["ai_generated"]
            return ai_score * 100
        else:
            raise Exception(f"Sightengine API failed: {result_json}")

    def verify_image(self, image_path):
        print(f"\n[+] Analyzing Image: '{image_path}'")
        
        try:
            fake_probability = self._primary_hf_analysis(image_path)
            provider = "Hugging Face (Primary)"
        except Exception as e:
            print(f"   [!] HF Failed: {str(e)}")
            try:
                fake_probability = self._fallback_sightengine_analysis(image_path)
                provider = "Sightengine (Fallback)"
            except Exception as e2:
                return {"label": "Error", "reason": f"Both pipelines failed. HF: {e} | SE: {e2}"}
        
        final_score = round(fake_probability, 1)
        label = "Fake" if final_score > 50 else "Real"
        
        return {
            "modality": "image",
            "label": label,
            "confidence_score": f"{final_score}%",
            "provider_used": provider,
            "breakdown": {
                "ai_pixel_analysis": final_score
            }
        }

if __name__ == "__main__":
    engine = ImageVerificationPipeline()
    
    # Create a dummy image file if one doesn't exist for testing
    test_file = "images.jpg"
    if not os.path.exists(test_file):
        from PIL import Image
        Image.new('RGB', (100, 100), color='red').save(test_file)
        
    import json
    result = engine.verify_image(test_file)
    print("\nRESULT:")
    print(json.dumps(result, indent=2))