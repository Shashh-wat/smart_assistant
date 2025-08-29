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
            if self.config["type"] == "ollama":
                if self._test_ollama_connection():
                    print(f"Using {self.config['description']}")
                    return
                else:
                    print("Ollama not available, trying API fallback...")
                    self._try_fallback()
            elif self.config["type"] == "huggingface_api":
                print(f"Using {self.config['description']}")
                return
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
                if self.config["type"] == "ollama":
                    if self._test_ollama_connection():
                        print(f"Using fallback: {self.config['description']}")
                        return
                elif self.config["type"] == "huggingface_api":
                    print(f"Using fallback: {self.config['description']}")
                    return
            except Exception:
                continue
        raise Exception("No working models available")
    
    def generate(self, prompt, max_tokens=500):
        try:
            if self.config["type"] == "ollama":
                return self._ollama_generate(prompt, max_tokens)
            elif self.config["type"] == "huggingface_api":
                return self._huggingface_api_generate(prompt, max_tokens)
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
    
    def _huggingface_api_generate(self, prompt, max_tokens):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.config["api_url"],
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 503:
                    print(f"Model loading, waiting 15 seconds... (attempt {attempt + 1})")
                    time.sleep(15)
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                        return generated_text
                    elif isinstance(result, dict):
                        return result.get("generated_text", str(result))
                    else:
                        return str(result)
                else:
                    print(f"HF API returned {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:
                        raise Exception(f"HF API error: {response.status_code}")
            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    raise Exception("HF API timeout")
                time.sleep(5)
        
        raise Exception("HF API failed after retries")
    
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