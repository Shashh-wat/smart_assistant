#!/usr/bin/env python3
"""
Sales Insight Query Engine - Bulletproof Version
Works both standalone and with Streamlit integration

Usage: 
python query.py --data data/processed/all_meetings.json --interactive
python query.py --query "What budget was discussed in meeting 1?"
"""

import os
import json
import argparse
import re
import numpy as np
from typing import Dict, List, Any, Optional

from utils.llm_manager import LLMManager
from utils.data_utils import load_processed_data
from config import PROCESSED_DIR, MAX_CONTEXT_TOKENS


class QueryEngine:
    def __init__(self, processed_data_path: Optional[str] = None, model_name: Optional[str] = None, processed_data: Optional[Dict] = None):
        """
        Initialize QueryEngine with flexible input options
        
        Args:
            processed_data_path: Path to JSON file (for standalone usage)
            model_name: LLM model to use
            processed_data: Pre-loaded data dict (for Streamlit integration)
        """
        self.llm = LLMManager(model_name)
        
        # Handle data loading - supports both file path and direct data
        if processed_data is not None:
            # Direct data passed (Streamlit mode)
            self.processed_data = processed_data
            print(f"Query engine ready with {len(self.processed_data)} meetings")
        elif processed_data_path:
            # Load from file (standalone mode)
            self.processed_data = self._load_data(processed_data_path)
        else:
            # No data provided
            print("Warning: No data provided to QueryEngine")
            self.processed_data = {}
        
        if self.processed_data:
            print(f"Model: {self.llm.get_model_info()['description']}")
    
    def _load_data(self, data_path: str) -> Dict[str, Any]:
        """Load preprocessed meeting data from file"""
        if not os.path.exists(data_path):
            print(f"Processed data not found at {data_path}")
            print("Run preprocessing first: python preprocess.py")
            return {}
        
        processed_data = load_processed_data(data_path)
        if not processed_data:
            print("No valid processed data found")
            return {}
        
        print(f"Loaded {len(processed_data)} meetings from {data_path}")
        return processed_data
    
    def process_query(self, user_query: str) -> str:
        """Main query processing function with mathematical enhancement"""
        if not self.processed_data:
            return "No processed data available. Please run preprocessing first."
        
        print(f"Processing query: '{user_query}'")
        
        # Step 1: Classify query type
        query_type = self._classify_query(user_query)
        print(f"Query classification: {query_type}")
        
        # Step 2: Select relevant context
        relevant_context = self._select_relevant_context(user_query, query_type)
        context_size = len(json.dumps(relevant_context))
        print(f"Context assembled: ~{context_size} characters")
        
        # Step 3: Build enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(user_query, relevant_context, query_type)
        
        # Step 4: Generate response
        print("Generating response...")
        response = self.llm.generate(enhanced_prompt, max_tokens=400)
        
        return response.strip()
    
    def _classify_query(self, query: str) -> str:
        """Classify query type to optimize context selection"""
        query_lower = query.lower()
        
        # Meeting-specific patterns
        if re.search(r'\b(?:meeting|call|interview)\s+\d+\b', query_lower):
            return "meeting_specific"
        
        # Feedback/Performance patterns - HIGH PRIORITY
        feedback_keywords = [
            'feedback', 'performance', 'told', 'said about', 'advice', 'improve',
            'criticism', 'evaluation', 'assessment', 'comment', 'suggestion',
            'strengths', 'weaknesses', 'recommendation'
        ]
        if any(keyword in query_lower for keyword in feedback_keywords):
            return "feedback_focused"
        
        # Entity-focused patterns
        entity_keywords = ['budget', 'financial', 'objection', 'technical', 'concern', 'next step']
        if any(word in query_lower for word in entity_keywords):
            return "entity_focused"
        
        # Aggregate patterns
        aggregate_keywords = ['all', 'across', 'common', 'overall', 'summary', 'total']
        if any(word in query_lower for word in aggregate_keywords):
            return "aggregate"
        
        # Default to meeting_specific for single meeting data
        return "meeting_specific"
    
    def _select_relevant_context(self, query: str, query_type: str) -> Dict[str, Any]:
        """Select relevant context based on query type"""
        
        if query_type == "feedback_focused":
            return self._get_feedback_context(query)
        elif query_type == "meeting_specific":
            return self._get_meeting_specific_context(query)
        elif query_type == "aggregate":
            return self._get_aggregate_context()
        elif query_type == "entity_focused":
            return self._get_entity_focused_context(query)
        else:
            return self._get_meeting_specific_context(query)
    
    def _get_feedback_context(self, query: str) -> Dict[str, Any]:
        """Get context focused on feedback and performance"""
        context = {
            "query_type": "feedback_focused",
            "feedback_data": {},
            "performance_data": {}
        }
        
        # Extract feedback from all meetings
        for meeting_id, meeting_data in self.processed_data.items():
            semantic = meeting_data.get("semantic_data", {})
            
            feedback_info = {}
            
            # Get feedback-related data
            if semantic.get("feedback_given"):
                feedback_info["feedback_given"] = semantic["feedback_given"]
            if semantic.get("performance_comments"):
                feedback_info["performance_comments"] = semantic["performance_comments"]
            if semantic.get("assignments_given"):
                feedback_info["assignments_given"] = semantic["assignments_given"]
            
            # Also include general context
            feedback_info["executive_summary"] = semantic.get("executive_summary", "")
            feedback_info["key_topics"] = semantic.get("key_topics", [])
            
            if feedback_info:
                context["feedback_data"][meeting_id] = feedback_info
        
        return context
    
    def _extract_meeting_id(self, query: str) -> Optional[str]:
        """Extract meeting ID from query"""
        match = re.search(r'\b(?:meeting|call|interview)\s+(\d+)\b', query.lower())
        if match:
            meeting_num = int(match.group(1))
            return f"meeting_{meeting_num:03d}"
        return None
    
    def _get_meeting_specific_context(self, query: str) -> Dict[str, Any]:
        """Load full context from specific meeting"""
        meeting_id = self._extract_meeting_id(query)
        
        # If no specific meeting mentioned, use first meeting
        if not meeting_id and self.processed_data:
            meeting_id = list(self.processed_data.keys())[0]
        
        if meeting_id and meeting_id in self.processed_data:
            meeting_data = self.processed_data[meeting_id]
            return {
                "query_type": "meeting_specific",
                "target_meeting": meeting_id,
                "semantic_data": meeting_data.get("semantic_data", {}),
                "mathematical_features": meeting_data.get("mathematical_features", {}),
                "metadata": meeting_data.get("metadata", {})
            }
        else:
            return self._get_aggregate_context()
    
    def _get_aggregate_context(self) -> Dict[str, Any]:
        """Get lightweight context from all meetings"""
        context = {
            "query_type": "aggregate",
            "all_meetings": {}
        }
        
        for meeting_id, meeting_data in self.processed_data.items():
            semantic = meeting_data.get("semantic_data", {})
            
            context["all_meetings"][meeting_id] = {
                "executive_summary": semantic.get("executive_summary", ""),
                "key_topics": semantic.get("key_topics", [])[:3],
                "feedback_given": semantic.get("feedback_given", []),
                "performance_comments": semantic.get("performance_comments", [])
            }
        
        return context
    
    def _get_entity_focused_context(self, query: str) -> Dict[str, Any]:
        """Get context focused on specific entities"""
        query_lower = query.lower()
        
        entity_mapping = {
            "financial": ["financial_mentions", "budget", "cost", "price"],
            "objections": ["objections_concerns", "concern", "worry", "issue"],
            "technical": ["technical_topics", "technical", "technology", "api"],
            "next_steps": ["next_steps", "action", "follow", "step"]
        }
        
        focused_entities = []
        for entity_type, keywords in entity_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                focused_entities.append(entity_type)
        
        if not focused_entities:
            focused_entities = ["key_topics"]
        
        context = {
            "query_type": "entity_focused",
            "focused_entities": focused_entities,
            "entity_data": {}
        }
        
        for meeting_id, meeting_data in self.processed_data.items():
            semantic = meeting_data.get("semantic_data", {})
            
            meeting_entities = {}
            for entity_type in focused_entities:
                if entity_type == "financial":
                    entities = semantic.get("financial_mentions", [])
                elif entity_type == "objections":
                    entities = semantic.get("objections_concerns", [])
                elif entity_type == "technical":
                    entities = semantic.get("technical_topics", [])
                else:
                    entities = semantic.get(entity_type, [])
                
                if entities:
                    meeting_entities[entity_type] = entities
            
            if meeting_entities:
                context["entity_data"][meeting_id] = meeting_entities
        
        return context
    
    def _build_enhanced_prompt(self, query: str, context: Dict[str, Any], query_type: str) -> str:
        """Build enhanced prompt prioritizing semantic content for feedback"""
        
        prompt = f"""SALES CALL ANALYSIS SYSTEM

USER QUESTION: {query}

"""
        
        # For feedback queries, prioritize semantic content
        if query_type == "feedback_focused":
            feedback_data = context.get("feedback_data", {})
            prompt += f"""FEEDBACK AND PERFORMANCE DATA (Primary Source):
{json.dumps(feedback_data, indent=2)}

Instructions: Focus on the actual feedback, performance comments, and assignments given. Use direct quotes and specific statements from the conversation.

"""
        
        elif query_type == "meeting_specific":
            semantic_data = context.get("semantic_data", {})
            prompt += f"""CONVERSATION CONTENT (Primary Source):
{json.dumps(semantic_data, indent=2)}

MATHEMATICAL PATTERNS (Supporting Context):
{json.dumps(context.get("mathematical_features", {}), indent=2)}

Instructions: Use the conversation content as your primary source. Reference specific statements and dialogue. Use mathematical patterns to support your analysis.

"""
        
        else:
            # For other query types, use existing logic
            prompt += f"""SEMANTIC CONTENT:
{json.dumps(context.get('all_meetings', context.get('entity_data', {})), indent=2)}

Instructions: Provide specific, factual answers based on the conversation content.

"""
        
        prompt += "RESPONSE:"
        
        return prompt
    
    def interactive_mode(self):
        """Run interactive command-line chat"""
        print("\n" + "="*60)
        print("SALES INSIGHT BOT - Interactive Mode")
        print("="*60)
        print("Ask questions about your sales calls")
        print("Available meetings:", list(self.processed_data.keys()))
        print("Examples:")
        print("   - 'What feedback was given to Shashwat?'")
        print("   - 'What technical topics were discussed in meeting 1?'")
        print("   - 'What assignment was given?'")
        print("   - 'What were Shashwat's strengths and weaknesses?'")
        print("\nType 'help' for more examples, 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if not user_input:
                    print("Please enter a question")
                    continue
                
                # Process query
                print("\nProcessing...")
                response = self.process_query(user_input)
                print(f"\nANSWER:\n{response}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Display help with sample queries"""
        help_text = """
SAMPLE QUERIES BY TYPE:

Feedback & Performance:
- "What feedback did Shashwat get?"
- "What were the interviewer's comments?"
- "What assignment was given to the candidate?"
- "What did Rohit tell Shashwat about his performance?"

Technical Discussion:
- "What technical topics were discussed?"
- "What AI/ML concepts were covered?"
- "What research was mentioned?"

Other Queries:
- "What are the next steps?"
- "What concerns were raised?"
        """
        print(help_text)


def main():
    parser = argparse.ArgumentParser(description="Query sales call insights")
    parser.add_argument("--data", default=os.path.join(PROCESSED_DIR, "all_meetings.json"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--query", type=str)
    
    args = parser.parse_args()
    
    try:
        engine = QueryEngine(processed_data_path=args.data, model_name=args.model)
        
        if not engine.processed_data:
            print("Cannot proceed without processed data")
            return
            
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    if args.interactive:
        engine.interactive_mode()
        
    elif args.query:
        print("Single Query Mode")
        response = engine.process_query(args.query)
        print(f"\nANSWER:\n{response}")
        
    else:
        print("Usage Options:")
        print("  Interactive: python query.py --interactive")
        print("  Single query: python query.py --query 'What feedback was given?'")


if __name__ == "__main__":
    main()