Sales Insight Bot A purely local LLM-powered system for analyzing sales call transcripts and answering questions through an intelligent chat interface. Uses Ollama for local inference with structured data extraction and mathematical analysis to provide accurate, contextual responses without any external APIs. Complete Local Setup Guide Step 1: Environment Setup bash# Clone repository git clone https://github.com/Shashh-wat/smart_assistant

# Install Python dependencies

pip install streamlit requests scikit-learn numpy pandas


# Create data directories

mkdir -p data/transcripts data/processed Step 2: Local Ollama Installation bash# Install Ollama (Mac/Linux) curl -fsSL https://ollama.ai/install.sh | sh


# Install Ollama (Mac/Linux)

curl -fsSL https://ollama.ai/install.sh | sh



# Pull different models for comparison

ollama pull llama3.2:latest
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b


# Start Ollama server (keep running in terminal)

ollama serve
Start Ollama server (keep running in terminal)

ollama serve 

# Step 3: Add Your Transcript Data Place your .txt transcript files in data/transcripts/

Example file structure:

data/transcripts/

├── interview_shashwat.txt 
├── sales_call_client_a.txt 
└── feedback_session_q4.txt

# Step 4: Run Preprocessing Pipeline Basic preprocessing with default model python preprocess.py
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

python query.py --data data/processed/all_meetings.json --interactive Model Switching & Result Variations Configuring Different Models Edit config.py to experiment with models: pythonMODEL_CONFIGS = { "ollama_llama": { "type": "ollama", "model": "llama3.2:latest", "url": "http://localhost:11434", "description": "Local Ollama Llama 3.2" }, "ollama_mistral": { "type": "ollama", "model": "mistral:7b", "url": "http://localhost:11434", "description": "Local Ollama Mistral 7B" } }

# Running Preprocessing with Different Models Process with Llama (most accurate semantic extraction) python preprocess.py --model ollama_llama

Process with Mistral (faster, more concise) python preprocess.py --model ollama_mistral
Compare results by switching models in Streamlit interface

# Expected Model Differences 

Llama 3.2: More detailed semantic extraction, better context understanding, nuanced feedback analysis 

Mistral 7B: Faster processing, more concise responses, different interpretation style

#Design Approach: 
Solving the Context Window Challenge The Central Problem Sales transcripts contain 10,000+ characters but LLMs have context limits. How do you provide accurate answers without losing critical information? 

# Mathematical Solution 
Thought Process Behind Mathematical Analysis The mathematical utilities solve a fundamental information retrieval problem. Rather than randomly selecting transcript segments or truncating content, mathematical analysis provides objective signals about content importance and relevance.

Mathematical Metadata Components

    Speaker Pattern Analysis (_analyze_speakers) python# Problem: "Who was the main participant?" requires scanning entire transcript

Solution: Mathematical speaking ratio calculation

def _analyze_speakers(self, text):
return { "speaking_ratios": {"Shashwat": 0.58, "Rohit": 0.42}, "dominant_speaker": "Shashwat" }
# Uses regex patterns to detect speaker transitions
# Calculates word counts per speaker across conversation 
# Provides precise participation metrics return { "speaking_ratios": {"Shashwat": 0.58, "Rohit": 0.42}, "dominant_speaker": "Shashwat" }

    Content Density Analysis (_analyze_content_density) python# Problem: Which segments contain business-critical information?

Solution: Term frequency analysis for different content types

def _analyze_content_density(self, text):  # Scans for technical terms, financial mentions, questions # Calculates density ratios to identify content-rich segments return 
{ "technical_density": 0.12,   # 12% technical terms 
"financial_density": 0.05,     # 5% financial content 
"question_density":0.08}      # 8% questions (engagement indicator) 

    Engagement Peak Detection (_analyze_conversation_flow) python# Problem: Where are the most important discussion moments?

Solution: Mathematical engagement scoring algorithm

def _analyze_conversation_flow(self, text): # Calculates engagement scores based on: 
# - Response length (longer = more engaged)
# - Question frequency (questions = interaction) 
# - Technical term density (complexity = importance) 
# - Financial mentions (budget = business critical) 

return { "peak_moments": [45, 122, 189],     # Line numbers of high engagement 
"conversation_energy": 2.4}      # Overall interaction intensity 

    Topic Importance Weighting (_calculate_topic_importance) python# Problem: All extracted topics aren't equally important for queries

#Solution: Mathematical importance scoring

def _calculate_topic_importance(self, semantic_data): # Weights based on frequency, diversity, content length 
# Guides context selection during query processing 
return { "feedback_given": 0.85,  # High weight - always include
"technical_topics": 0.67,         # Medium weight - include if space 
"casual_chat": 0.12}              # Low weight - exclude when tight 

#Why Mathematical Metadata Enhances Sales Use Cases Context Selection Intelligence:

When user asks "What feedback was given?", the system uses importance weights to prioritize feedback-related content over casual conversation. Speaker-Aware Responses: Mathematical speaker analysis enables queries like "Who dominated the conversation?" without manually scanning transcripts. Content Quality Assessment: Density metrics help identify transcript segments with the highest business value (technical discussions, financial negotiations, objection handling). 

Engagement-Driven Context: Peak detection ensures LLM receives context from the most meaningful conversation moments, not random segments. Query Processing Strategy Meeting-Specific Queries 

Example: "What happened in meeting 2?" Strategy: Load complete semantic + mathematical context for target meeting Context Fed to LLM: Full meeting data with mathematical patterns for enhanced understanding Meeting-Agnostic Queries Example: "What feedback was given across all calls?" Strategy: Aggregate feedback data using importance weights across all meetings 


#Context Fed to LLM: Consolidated feedback summaries with mathematical validation Query Classification System pythondef _classify_query(self, query: str): 
# feedback_focused: "feedback", "performance", 
"evaluation" # meeting_specific: "meeting 1", "call 2", specific identifiers 
# entity_focused: "budget", "technical", "objections" 
# aggregate: "all", "common", "overall", "summary"
Each classification triggers different context selection strategies optimized for that query type.

Project Structure 
sales-insight-bot/ 
├── app.py # Streamlit web interface 
├── query.py # Query engine (standalone + Streamlit) 
├── preprocess.py # Preprocessing pipeline 
├── config.py # Configuration settings 
├── utils/ │ 
    ├── init.py # Package initialization │ 
    ├── llm_manager.py # LLM abstraction layer │
    ├── math_utils.py # Mathematical NLP analysis │ 
    └── data_utils.py # Data management utilities 
└── data/ 
├── transcripts/ # Raw .txt transcript files 
└── processed/ # Generated JSON files

Demo Queries & Sample Outputs

    Speaker Analysis Query: "Who were the speakers and how much did each person talk?" Output: Speaker participation analysis:

    Shashwat: 58% speaking time (dominant speaker)
    Rohit: 42% speaking time

Mathematical analysis shows 127 speaking turns for Shashwat and 89 for Rohit, indicating interactive dialogue format typical of technical interviews. 2. Feedback Extraction Query: "What specific feedback was given to Shashwat?" Output: Feedback summary for Shashwat:

    Performance: "pretty good" overall assessment
    Strengths: Strong mathematical and analytical skills
    Areas for improvement: "improve on basics" in foundational concepts
    Assignment: GitHub repository creation, 24-hour deadline, emphasis on innovation

The mathematical importance weighting identified this feedback as high-priority content (0.85 importance score). 3. Technical Content Analysis Query: "What technical topics were discussed?" Output: Technical discussion analysis:

    Core topics: Transformer architectures, neural networks, API integration
    Technical density: 12% of conversation content (above average)
    Peak technical engagement: Lines 45-67, 122-145
    Research focus: Recent papers in NLP and computer vision

Mathematical clustering identified 3 distinct technical topic groups with transformer architectures having highest importance score. 4. Assignment Details Query: "What assignment was given?" Output: Assignment extracted from high-engagement conversation segment:

    Deliverable: GitHub repository demonstrating technical capabilities
    Timeline: 24 hours from meeting conclusion
    Requirements: Innovation and practical application of discussed concepts
    Context: Performance evaluation follow-up

Speaker analysis confirms assignment was given during peak engagement period (conversation energy: 2.4/3.0). 5. Cross-Meeting Summary Query: "Summarize all meetings comprehensively" Output: Comprehensive analysis across all processed meetings:

    Total meetings: 1 technical interview
    Participants: Shashwat (58% speaking), Rohit (42% speaking)
    Primary focus: AI/ML technical evaluation with feedback delivery
    Key outcomes: Performance assessment, skill evaluation, practical assignment
    Technical depth: High (12% technical density, multiple AI/ML concepts)
    Business impact: Candidate assessment with actionable development path Technical Benefits

100% Local Processing: Complete privacy, no data transmission Mathematical Intelligence: Quantitative analysis guides optimal context selection Model Flexibility: Easy switching between Ollama models for different needs Sales-Optimized: Designed specifically for sales meeting analysis patterns Token Efficiency: Fits relevant context within LLM limits without information loss

Troubleshooting Preprocessing fails: Ensure Ollama is running with ollama serve No processed data: Run python preprocess.py before launching app Empty responses: Verify .txt files exist in data/transcripts/ Model errors: Check Ollama model is pulled with ollama list Import errors: Install all dependencies with pip install command above
## Sample Use Cases

Below are some sample use cases demonstrating the system in action:

[Sample Query 1](images/img1.png)

[Sample Query 2](images/img2.png)

[Sample Query 3](images/img3.png)

[Sample Query 4](images/img4.png)

[Sample Query 5](images/img5.png)

[Sample Query 6](images/img6.png)
