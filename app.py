#!/usr/bin/env python3
"""
Sales Insight Bot - Web UI
Streamlit interface for querying sales call transcripts

Usage: streamlit run app.py
"""

import streamlit as st
import os
import json
from datetime import datetime

from utils.llm_manager import LLMManager
from utils.data_utils import load_processed_data, generate_data_report
from query import QueryEngine
from config import PROCESSED_DIR, MODEL_CONFIGS


# Page configuration
st.set_page_config(
    page_title="Sales Insight Bot",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def load_data_and_model():
    """Load processed data and initialize model"""
    st.sidebar.title("🔧 Configuration")
    
    model_options = {name: config["description"] for name, config in MODEL_CONFIGS.items()}
    selected_model = st.sidebar.selectbox(
        "Select LLM Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    data_files = []
    if os.path.exists(PROCESSED_DIR):
        data_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.json')]
    
    if data_files:
        selected_file = st.sidebar.selectbox(
            "Select processed data:",
            options=data_files,
            index=0 if data_files else None
        )
        data_path = os.path.join(PROCESSED_DIR, selected_file)
    else:
        st.sidebar.error("❌ No processed data found!")
        st.sidebar.info("Run preprocessing first: `python preprocess.py`")
        return None, None
    
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Loading data and initializing model..."):
            try:
                processed_data = load_processed_data(data_path)
                if not processed_data:
                    st.error("Failed to load processed data")
                    return None, None
                
                # FIXED: Proper QueryEngine initialization
                query_engine = QueryEngine(processed_data=processed_data, model_name=selected_model)
                
                st.session_state.processed_data = processed_data
                st.session_state.query_engine = query_engine
                st.session_state.chat_history = []
                
                st.success(f"✅ Loaded {len(processed_data)} meetings with {query_engine.llm.get_model_info()['description']}")
                
            except Exception as e:
                st.error(f"❌ Setup failed: {str(e)}")
                return None, None
    
    return st.session_state.processed_data, st.session_state.query_engine


def display_data_overview():
    """Display overview of processed data"""
    if st.session_state.processed_data:
        st.subheader("📊 Data Overview")
        
        summary = generate_data_report(st.session_state.processed_data)
        st.info(summary)
        
        for meeting_id, meeting_data in st.session_state.processed_data.items():
            with st.expander(f"🎯 {meeting_id.upper()} Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Semantic Data:**")
                    semantic = meeting_data.get("semantic_data", {})
                    
                    if "executive_summary" in semantic:
                        st.write(f"*Summary:* {semantic['executive_summary']}")
                    
                    if semantic.get("key_topics"):
                        st.write(f"*Topics:* {', '.join(semantic['key_topics'][:5])}")
                    
                    if semantic.get("feedback_given"):
                        st.write(f"*Feedback:* {', '.join(semantic['feedback_given'][:3])}")
                
                with col2:
                    st.write("**Mathematical Analysis:**")
                    math_features = meeting_data.get("mathematical_features", {})
                    
                    if "speaker_analysis" in math_features:
                        speakers = math_features["speaker_analysis"].get("speaking_ratios", {})
                        for speaker, ratio in speakers.items():
                            st.write(f"*{speaker}:* {ratio:.1%} speaking time")
                    
                    if "content_density" in math_features:
                        density = math_features["content_density"]
                        st.write(f"*Technical Density:* {density.get('technical_density', 0):.3f}")


def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    st.title(" Sales Insight Bot")
    st.markdown("Ask questions about your sales call transcripts!")
    
    # Sidebar for configuration
    processed_data, query_engine = load_data_and_model()
    
    if processed_data:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["💬 Chat", "📊 Data Overview"])
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "timestamp" in message:
                    st.caption(f"*{message['timestamp']}*")
        
        # Chat input - MOVED OUTSIDE OF TABS
        if prompt := st.chat_input("Ask about your sales calls..."):
            if not st.session_state.query_engine:
                st.error("⚠️ Please load data first using the sidebar")
            else:
                # Add user message to chat
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": timestamp
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                    st.caption(f"*{timestamp}*")
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Analyzing..."):
                        try:
                            response = st.session_state.query_engine.process_query(prompt)
                            st.write(response)
                            
                            response_time = datetime.now().strftime("%H:%M:%S")
                            st.caption(f"*{response_time}*")
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response,
                                "timestamp": response_time
                            })
                            
                        except Exception as e:
                            error_msg = f"❌ Error processing query: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
        
        # Tab content
        with tab1:
            st.subheader("💡 Sample Queries")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("What feedback did Shashwat get?"):
                    st.session_state['sample_query'] = "What feedback did Shashwat get?"
                    st.rerun()
                
                if st.button("What technical topics were discussed?"):
                    st.session_state['sample_query'] = "What technical topics were discussed?"
                    st.rerun()
            
            with col2:
                if st.button("What assignment was given?"):
                    st.session_state['sample_query'] = "What assignment was given?"
                    st.rerun()
                
                if st.button("Who dominated the conversation?"):
                    st.session_state['sample_query'] = "Who dominated the conversation?"
                    st.rerun()
        
        with tab2:
            display_data_overview()
    
    else:
        st.markdown("""
        ### Welcome! Get started in 3 steps:
        
        1. **Add transcripts** to `data/transcripts/` folder (.txt files)
        2. **Run preprocessing**: `python preprocess.py`
        3. **Load data** using the sidebar and start chatting!
        
        ### Sample Questions You Can Ask:
        - "What feedback was given to Shashwat?" 
        - "What technical topics were discussed?"
        - "What assignment was given?"
        - "Who were the key speakers?"
        """)


if __name__ == "__main__":
    main()