Sales Insight Bot: A purely local LLM-powered system for analyzing sales call transcripts and answering questions through an intelligent chat interface. Uses Ollama for local inference with structured data extraction and mathematical analysis to provide accurate, contextual responses without any external APIs. 


# Complete Local Setup Guide 
 # Clone repository 
 
 
 git clone https://github.com/Shashh-wat/smart_assistant

# Install Python dependencies

pip install streamlit requests scikit-learn numpy pandas


# Create data directories

mkdir -p data/transcripts data/processed

# Install Ollama (Mac/Linux)

curl -fsSL https://ollama.ai/install.sh | sh.



# Pull different models for comparison

ollama pull llama3.2:latest.

ollama pull llama3.1:8b.

ollama pull mistral:7b.


ollama pull qwen2.5:7b.


# Start Ollama server (keep running in terminal)


ollama serve.


(to keep it running is not mandatory , however curl testing it once is advised)

(curl http://localhost:11434/api/tags)



# Step 3: Add Your Transcript Data Place your .txt transcript files in data/transcripts/

Example file structure:

data/transcripts/

├── interview_shashwat.txt.

├── sales_call_client_a.txt.

└── feedback_session_q4.txt.

# Step 4: Run Preprocessing Pipeline 

Basic preprocessing with default model 

python preprocess.py


Preprocessing with specific model

python preprocess.py --model ollama_mistral

# Custom input/output directories

python preprocess.py --input data/transcripts/ --output data/processed/

What this does:
- Reads all .txt files from transcripts folder
- Extracts semantic data using LLM (speakers, topics, feedback, assignments)
- Calculates mathematical metadata (speaker ratios, content density, engagement peaks)
- Saves structured JSON to data/processed/all_meetings.json

# Step 5: Query Your Data Web interface (recommended) streamlit run app.py
Standalone interactive mode

python query.py --interactive

Single query mode

python query.py --query "What feedback did Shashwat get?"


Custom data file

python query.py --data data/processed/all_meetings.json --interactive 

## Model Switching & Result Variations Configuring Different Models Edit config.py to experiment with models:

MODEL_CONFIGS = { "ollama_llama": { "type": "ollama", "model": "llama3.2:latest", "url": "http://localhost:11434", "description": "Local Ollama Llama 3.2" }, "ollama_mistral": { "type": "ollama", "model": "mistral:7b", "url": "http://localhost:11434", "description": "Local Ollama Mistral 7B" } }

# Running Preprocessing with Different Models Process with Llama (most accurate semantic extraction) python preprocess.py --model ollama_llama

Process with Mistral (faster, more concise) python preprocess.py --model ollama_mistral

Compare results by switching models in Streamlit interface

# Expected Model Differences 

Llama 3.2: More detailed semantic extraction, better context understanding, nuanced feedback analysis 

Mistral 7B: Faster processing, more concise responses, different interpretation style



100% Local Processing: Complete privacy, no data transmission.

Mathematical Intelligence: Quantitative analysis guides optimal context selection Model.

Flexibility: Easy switching between Ollama models for different needs.

Sales-Optimized: Designed specifically for sales meeting analysis patterns.

Token Efficiency: Fits relevant context within LLM limits without information loss

# Troubleshooting
Preprocessing fails: Ensure Ollama is running with ollama serve 

No processed data: Run python preprocess.py before launching app 

Empty responses: Verify .txt files exist in data/transcripts/ 

Model errors: Check Ollama model is pulled with ollama list 

Import errors: Install all dependencies with pip install command above

## Sample Use Cases

Below are some sample use cases demonstrating the system in action:

[Sample Query 1](images/img1.png)

[Sample Query 2](images/img2.png)

[Sample Query 3](images/img3.png)

[Sample Query 4](images/img4.png)

[Sample Query 5](images/img5.png)

[Sample Query 6](images/img6.png)
