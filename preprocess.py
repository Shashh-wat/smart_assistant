#!/usr/bin/env python3
"""
Sales Transcript Preprocessing Pipeline
Converts raw transcripts into structured, queryable data

Usage: python preprocess.py --input data/transcripts/ --model ollama_llama
"""

import os
import json
import glob
import argparse
import re
from datetime import datetime

from utils.llm_manager import LLMManager
from utils.math_utils import MathematicalAnalyzer
from utils.data_utils import save_processed_data, create_directories
from config import TRANSCRIPTS_DIR, PROCESSED_DIR


class TranscriptPreprocessor:
    def __init__(self, model_name=None):
        self.llm = LLMManager(model_name)
        self.math_analyzer = MathematicalAnalyzer()
        print(f"🤖 Initialized with {self.llm.get_model_info()['description']}")
    
    def process_all_transcripts(self, input_dir):
        """Process all transcript files in directory"""
        transcript_files = glob.glob(os.path.join(input_dir, "*.txt"))
        
        if not transcript_files:
            print(f" No .txt files found in {input_dir}")
            return
        
        print(f"📁 Found {len(transcript_files)} transcript files")
        
        processed_data = {}
        for i, file_path in enumerate(transcript_files):
            meeting_id = f"meeting_{i+1:03d}"
            print(f"\n Processing {os.path.basename(file_path)} as {meeting_id}...")
            
            processed_data[meeting_id] = self.process_single_transcript(file_path, meeting_id)
        
        # Save all processed data
        output_file = os.path.join(PROCESSED_DIR, "all_meetings.json")
        save_processed_data(processed_data, output_file)
        print(f"\n All transcripts processed! Saved to {output_file}")
        
        return processed_data
    
    def process_single_transcript(self, file_path, meeting_id):
        """Process a single transcript file"""
        # Read raw transcript
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        print(f"   📄 Raw transcript: {len(raw_text)} characters")
        
        # Stage 1: LLM Semantic Extraction
        semantic_data = self.extract_semantic_data(raw_text)
        print(f"    Semantic extraction: Complete")
        
        # Stage 2: Mathematical Analysis
        math_metadata = self.math_analyzer.analyze_transcript(raw_text, semantic_data)
        print(f"   Mathematical analysis: Complete")
        
        # Stage 3: Create structured output
        processed_result = {
            "metadata": {
                "meeting_id": meeting_id,
                "source_file": os.path.basename(file_path),
                "processed_at": datetime.now().isoformat(),
                "character_count": len(raw_text),
                "model_used": self.llm.get_model_info()["name"]
            },
            "semantic_data": semantic_data,
            "mathematical_features": math_metadata,
            "raw_text_preview": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        }
        
        return processed_result
    
    def extract_semantic_data(self, raw_text):
        """Use LLM to extract structured semantic information"""
        chunks = self._chunk_transcript(raw_text)
        
        all_extractions = []
        for chunk in chunks:
            extraction = self._extract_chunk_data(chunk)
            all_extractions.append(extraction)
        
        return self._consolidate_extractions(all_extractions)
    
    def _chunk_transcript(self, text, max_chars=3000):
        """Split transcript into manageable chunks"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk + line) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = line
            else:
                current_chunk += "\n" + line
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _extract_chunk_data(self, chunk):
        """Extract structured data from a single chunk - UPDATED"""
        prompt = f"""
        Analyze this sales call transcript segment and extract key information.
        Pay special attention to feedback, evaluations, and performance comments.
        Return ONLY valid JSON in this exact format:
        {{
            "speakers": ["Speaker1", "Speaker2"],
            "key_topics": ["topic1", "topic2"],
            "financial_mentions": ["$50k budget", "Q1 timeline"],
            "technical_topics": ["API integration", "ML model"],
            "objections_concerns": ["security worry", "timeline concern"],
            "products_discussed": ["CRM system", "analytics platform"],
            "next_steps": ["send proposal", "schedule demo"],
            "pain_points": ["manual process", "data silos"],
            "decision_makers": ["John (CTO)", "Sarah (Budget owner)"],
            "feedback_given": ["improve on basics", "math is strong", "try to be innovative"],
            "performance_comments": ["pretty good", "papers are good", "needs work on X"],
            "assignments_given": ["GitHub repo", "24 hours deadline", "be innovative"]
        }}
        
        Look for:
        - Any evaluative statements about candidates or performance
        - Feedback, criticism, praise, or advice given
        - Performance assessments or suggestions for improvement
        - Instructions or assignments given
        - Comments about strengths or weaknesses
        
        Transcript segment:
        {chunk}
        """
        
        response = self.llm.generate(prompt, max_tokens=500)
        
        # Parse JSON response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback: extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # Return empty structure if parsing fails
            return {
                "speakers": [], "key_topics": [], "financial_mentions": [],
                "technical_topics": [], "objections_concerns": [], "products_discussed": [],
                "next_steps": [], "pain_points": [], "decision_makers": [],
                "feedback_given": [], "performance_comments": [], "assignments_given": []
            }
    
    def _consolidate_extractions(self, extractions):
        """Merge multiple chunk extractions - UPDATED"""
        consolidated = {
            "speakers": set(), "key_topics": [], "financial_mentions": [],
            "technical_topics": [], "objections_concerns": [], "products_discussed": [],
            "next_steps": [], "pain_points": [], "decision_makers": set(),
            "feedback_given": [], "performance_comments": [], "assignments_given": []
        }
        
        for extraction in extractions:
            for key, values in extraction.items():
                if key in ["speakers", "decision_makers"]:
                    consolidated[key].update(values)
                else:
                    consolidated[key].extend(values)
        
        # Convert sets to lists and remove duplicates
        consolidated["speakers"] = list(consolidated["speakers"])
        consolidated["decision_makers"] = list(consolidated["decision_makers"])
        
        for key in ["key_topics", "financial_mentions", "technical_topics", 
                   "objections_concerns", "products_discussed", "next_steps", "pain_points",
                   "feedback_given", "performance_comments", "assignments_given"]:
            consolidated[key] = list(set(consolidated[key]))
        
        # Generate executive summary
        consolidated["executive_summary"] = self._generate_executive_summary(consolidated)
        
        return consolidated
    
    def _generate_executive_summary(self, structured_data):
        """Generate executive summary from structured data"""
        prompt = f"""
        Create a 2-3 sentence executive summary of this sales call:
        
        Topics: {structured_data['key_topics']}
        Technical: {structured_data['technical_topics']}
        Feedback: {structured_data['feedback_given']}
        Next Steps: {structured_data['next_steps']}
        """
        
        return self.llm.generate(prompt, max_tokens=150)


def main():
    parser = argparse.ArgumentParser(description="Preprocess sales call transcripts")
    parser.add_argument("--input", default=TRANSCRIPTS_DIR)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=PROCESSED_DIR)
    
    args = parser.parse_args()
    
    create_directories([args.input, args.output])
    
    try:
        preprocessor = TranscriptPreprocessor(args.model)
        processed_data = preprocessor.process_all_transcripts(args.input)
        print(f"\n🎉 Preprocessing complete!")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")


if __name__ == "__main__":
    main()
