import requests
import time
from config import MODEL_CONFIGS, DEFAULT_MODEL, FALLBACK_MODELS


class LLMManager:
    def __init__(self, model_name=None):
        self.current_model = model_name or DEFAULT_MODEL
        self.config = MODEL_CONFIGS[self.current_model]
        self.setup_model()
    
    def setup_model(self):
        try:
            if self._test_ollama_connection():
                print(f"Using {self.config['description']}")
                return
            else:
                print("Primary model not available, trying fallback...")
                self._try_fallback()
        except Exception as e:
            print(f"Model setup failed: {e}")
            self._try_fallback()
    
    def _test_ollama_connection(self):
        try:
            response = requests.get(f"{self.config['url']}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return self.config["model"] in model_names
        except:
            return False
    
    def _try_fallback(self):
        for fallback_model in FALLBACK_MODELS:
            try:
                self.current_model = fallback_model
                self.config = MODEL_CONFIGS[fallback_model]
                if self._test_ollama_connection():
                    print(f"Using fallback: {self.config['description']}")
                    return
            except Exception:
                continue
        raise Exception("No working Ollama models available")
    
    def generate(self, prompt, max_tokens=500):
        try:
            return self._ollama_generate(prompt, max_tokens)
        except Exception as e:
            return f"Generation failed: {str(e)}"
    
    def _ollama_generate(self, prompt, max_tokens):
        payload = {
            "model": self.config["model"],
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3}
        }
        
        response = requests.post(
            f"{self.config['url']}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def get_model_info(self):
        return {
            "name": self.current_model,
            "type": self.config["type"], 
            "description": self.config["description"]
        }


def test_available_models():
    available = []
    for model_name in MODEL_CONFIGS.keys():
        try:
            manager = LLMManager(model_name)
            test_response = manager.generate("Hello", max_tokens=5)
            if test_response and "failed" not in test_response.lower():
                available.append(model_name)
                print(f"{model_name}: Working")
            else:
                print(f"{model_name}: Failed test")
        except Exception as e:
            print(f"{model_name}: Setup failed - {str(e)}")
    return available


if __name__ == "__main__":
    available_models = test_available_models()
    print(f"Available models: {available_models}")
