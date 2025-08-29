"""
Simplified Config - Ollama Only
Compatible with sophisticated preprocessing pipeline
"""

# Model Configuration
MODEL_CONFIGS = {
    "ollama_llama": {
        "type": "ollama",
        "model": "llama3.2:latest",
        "url": "http://localhost:11434",
        "description": "Local Ollama Llama 3.2"
    }
}

DEFAULT_MODEL = "ollama_llama"

# Fallback models (required by llm_manager.py)
FALLBACK_MODELS = ["ollama_llama"]  # Same as primary since we only have one

# Data Paths
TRANSCRIPTS_DIR = "data/transcripts"
PROCESSED_DIR = "data/processed"

# Token limits for processing
MAX_CONTEXT_TOKENS = 1500
MAX_RESPONSE_TOKENS = 500

# Processing settings
CHUNK_SIZE = 3000  # Characters per chunk for preprocessing
ENABLE_MATHEMATICAL_ANALYSIS = True
ENABLE_SEMANTIC_CLUSTERING = True