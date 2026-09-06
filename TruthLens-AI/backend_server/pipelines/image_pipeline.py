import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class ImageVerificationPipeline:
    def __init__(self):
        print("[*] Initializing Dual-Strategy Image Pipeline...")
        
        # 1. Primary Strategy: Hugging Face API
        self.hf_token = os.getenv("HF_API_TOKEN")
        self.hf_client = InferenceClient(api_key=self.hf_token)
        # Using a robust open-source model for artifact detection
        self.hf_model_id = "dima806/ai_vs_real_image_detection"
        
        # 2. Backup Strategy: Sightengine API
        self.se_user = os.getenv("SIGHTENGINE_USER")
        self.se_secret = os.getenv("SIGHTENGINE_SECRET")
        self.se_endpoint = "https://api.sightengine.com/1.0/check.json"

    def _primary_hf_analysis(self, image_path):
        """Strategy 1: Hugging Face Serverless Inference"""
        print("   -> Attempting Strategy 1: Hugging Face Cloud...")
        # Sends the image to the HF cloud classification endpoint
        results = self.hf_client.image_classification(image_path, model=self.hf_model_id)
        
        # The API returns a list of dictionaries: [{'label': 'artificial', 'score': 0.9}]
        for item in results:
            if item['label'].lower() in ['artificial', 'fake', 'ai-generated']:
                return item['score'] * 100
                
        # Fallback if 'fake' label isn't explicitly named
        real_score = next((item['score'] for item in results if item['label'].lower() in ['human', 'real']), 1.0)
        return (1.0 - real_score) * 100

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