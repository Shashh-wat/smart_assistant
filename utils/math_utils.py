"""
Mathematical NLP Analysis for Sales Transcripts
Calculates quantitative metadata to enhance LLM understanding
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json


class MathematicalAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def analyze_transcript(self, raw_text, semantic_data):
        """
        Generate mathematical metadata for transcript
        This enhances LLM prompting with quantitative insights
        """
        return {
            "speaker_analysis": self._analyze_speakers(raw_text),
            "content_density": self._analyze_content_density(raw_text),
            "conversation_flow": self._analyze_conversation_flow(raw_text),
            "semantic_clusters": self._analyze_semantic_clusters(raw_text),
            "engagement_metrics": self._calculate_engagement_metrics(raw_text),
            "topic_importance": self._calculate_topic_importance(semantic_data)
        }
    
    def _analyze_speakers(self, text):
        """Mathematical analysis of speaker patterns"""
        # Extract speaker turns with regex patterns
        speaker_patterns = [
            r'^([A-Z][a-z]+)\s*\(\d+%\)',  # "Shashwat (58%)"
            r'([A-Z][a-z]+)(?=:|\s+(?:says?|mentions?))',  # Speaker names before speech
            r'\b([A-Z][a-z]+)\s+(?:says?|mentions?|explains?)',  # "John says"
            r'^([A-Z][a-z]+)\s*:',  # "Speaker:"
            r'^\[[\d:]+\]\s*([A-Z][a-z]+)'  # "[00:01:23] Speaker"
        ]
        
        speakers = Counter()
        total_words = 0
        
        # Count words per speaker (approximate)
        lines = text.split('\n')
        current_speaker = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with speaker name
            speaker_found = False
            for pattern in speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    current_speaker = match.group(1)
                    speaker_found = True
                    break
            
            # Count words for current speaker
            word_count = len(line.split())
            total_words += word_count
            
            if current_speaker:
                speakers[current_speaker] += word_count
        
        # Calculate speaking ratios
        if speakers and total_words > 0:
            total_speaker_words = sum(speakers.values())
            speaking_ratios = {
                speaker: round(count / total_speaker_words, 3) 
                for speaker, count in speakers.items()
            }
        else:
            # Fallback: try to extract from Otter-style metadata
            speaker_ratios = self._extract_otter_speaker_ratios(text)
            if speaker_ratios:
                speaking_ratios = speaker_ratios
                speakers = Counter({k: int(v * 1000) for k, v in speaker_ratios.items()})
            else:
                speaking_ratios = {"Unknown": 1.0}
                speakers = Counter({"Unknown": total_words})
        
        return {
            "speaking_ratios": speaking_ratios,
            "turn_counts": dict(speakers),
            "total_speakers": len(speakers),
            "dominant_speaker": max(speaking_ratios.items(), key=lambda x: x[1])[0] if speakers else "Unknown"
        }
    
    def _extract_otter_speaker_ratios(self, text):
        """Extract speaker ratios from Otter.ai style metadata"""
        # Look for patterns like "Shashwat (58%), Rohit (42%)"
        pattern = r'([A-Z][a-z]+)\s*\((\d+)%\)'
        matches = re.findall(pattern, text)
        
        if matches:
            total_percentage = sum(int(match[1]) for match in matches)
            if total_percentage > 50:  # Valid percentage distribution
                return {match[0]: int(match[1]) / 100 for match in matches}
        
        return None
    
    def _analyze_content_density(self, text):
        """Calculate mathematical content density metrics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Technical term detection (expanded patterns)
        technical_patterns = [
            r'\b(?:API|ML|AI|neural|transformer|RAG|vector|algorithm)\b',
            r'\b(?:backend|frontend|database|deployment|architecture)\b',
            r'\b(?:Python|JavaScript|React|Flask|FastAPI)\b',
            r'\b(?:model|training|inference|embedding|tokenizer)\b',
            r'\b(?:clustering|classification|regression|NLP)\b'
        ]
        
        technical_matches = 0
        for pattern in technical_patterns:
            technical_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Financial term detection
        financial_patterns = [
            r'\$[\d,]+',
            r'\b(?:budget|cost|price|revenue|profit|ROI|funding)\b',
            r'\b(?:quarter|Q[1-4]|fiscal|investment)\b'
        ]
        
        financial_matches = 0
        for pattern in financial_patterns:
            financial_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Question density
        questions = len(re.findall(r'\?', text))
        
        # Exclamation density (excitement/emphasis)
        exclamations = len(re.findall(r'!', text))
        
        return {
            "technical_density": round(technical_matches / len(words), 4) if words else 0,
            "financial_density": round(financial_matches / len(words), 4) if words else 0,
            "question_density": round(questions / len(sentences), 4) if sentences else 0,
            "exclamation_density": round(exclamations / len(sentences), 4) if sentences else 0,
            "avg_sentence_length": round(len(words) / len(sentences), 2) if sentences else 0,
            "total_word_count": len(words),
            "total_sentences": len(sentences),
            "vocabulary_diversity": round(len(set(words)) / len(words), 4) if words else 0
        }
    
    def _analyze_conversation_flow(self, text):
        """Analyze mathematical patterns in conversation flow"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Calculate engagement scores for each line
        engagement_scores = []
        for i, line in enumerate(lines):
            # Longer responses = higher engagement
            length_score = min(len(line.split()) / 20, 1.0)  # Normalize to 0-1
            
            # Questions indicate engagement
            question_score = 0.3 if '?' in line else 0
            
            # Technical terms indicate deep discussion
            tech_terms = len(re.findall(r'\b(?:API|ML|AI|neural|transformer|architecture)\b', line, re.IGNORECASE))
            tech_score = min(tech_terms * 0.1, 0.5)
            
            # Exclamations indicate excitement
            exclamation_score = 0.2 if '!' in line else 0
            
            # Financial terms indicate important business discussion
            financial_terms = len(re.findall(r'\$[\d,]+|\b(?:budget|cost|price)\b', line, re.IGNORECASE))
            financial_score = min(financial_terms * 0.15, 0.3)
            
            total_engagement = length_score + question_score + tech_score + exclamation_score + financial_score
            engagement_scores.append(min(total_engagement, 2.0))  # Cap at 2.0
        
        # Find engagement peaks (local maxima)
        peaks = []
        if len(engagement_scores) >= 3:
            for i in range(1, len(engagement_scores) - 1):
                if (engagement_scores[i] > engagement_scores[i-1] and 
                    engagement_scores[i] > engagement_scores[i+1] and 
                    engagement_scores[i] > 0.5):
                    peaks.append(i)
        
        # Calculate conversation momentum (rate of change)
        momentum_changes = []
        if len(engagement_scores) >= 2:
            for i in range(1, len(engagement_scores)):
                momentum_changes.append(engagement_scores[i] - engagement_scores[i-1])
        
        return {
            "engagement_scores": engagement_scores,
            "peak_moments": peaks,
            "average_engagement": round(np.mean(engagement_scores), 3) if engagement_scores else 0,
            "engagement_variance": round(np.var(engagement_scores), 3) if engagement_scores else 0,
            "momentum_changes": momentum_changes,
            "conversation_energy": round(np.sum(np.abs(momentum_changes)), 3) if momentum_changes else 0
        }
    
    def _analyze_semantic_clusters(self, text):
        """Use mathematical clustering to find topic groups"""
        # Split text into segments for clustering
        segments = re.split(r'\n\s*\n', text)  # Split by paragraph breaks
        segments = [seg.strip() for seg in segments if len(seg.strip()) > 50]
        
        if len(segments) < 2:
            return {
                "cluster_assignments": [],
                "cluster_topics": {},
                "num_clusters": 0
            }
        
        try:
            # Create TF-IDF vectors
            vectors = self.vectorizer.fit_transform(segments)
            
            # Determine optimal number of clusters (2-5 based on data size)
            n_clusters = min(max(len(segments) // 3, 2), 5)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Analyze cluster characteristics
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_topics = {}
            
            for i in range(n_clusters):
                # Get centroid for this cluster
                centroid = kmeans.cluster_centers_[i]
                
                # Find top terms for this cluster
                top_indices = centroid.argsort()[-5:][::-1]  # Top 5 terms
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Calculate cluster size and importance
                cluster_size = np.sum(cluster_labels == i)
                importance_score = float(np.max(centroid))
                
                cluster_topics[f"cluster_{i}"] = {
                    "top_terms": top_terms,
                    "segments_count": int(cluster_size),
                    "importance_score": round(importance_score, 3),
                    "cluster_percentage": round(cluster_size / len(segments), 3)
                }
            
            return {
                "cluster_assignments": cluster_labels.tolist(),
                "cluster_topics": cluster_topics,
                "num_clusters": n_clusters,
                "silhouette_score": self._calculate_silhouette_score(vectors, cluster_labels)
            }
            
        except Exception as e:
            print(f"    Clustering failed: {e}")
            return {
                "cluster_assignments": [],
                "cluster_topics": {},
                "num_clusters": 0
            }
    
    def _calculate_silhouette_score(self, vectors, labels):
        """Calculate clustering quality score"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1:  # Need at least 2 clusters
                score = silhouette_score(vectors, labels)
                return round(float(score), 3)
        except:
            pass
        return 0.0
    
    def _calculate_engagement_metrics(self, text):
        """Calculate mathematical engagement and attention metrics"""
        # Split into paragraphs and analyze length distribution
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        word_counts = [len(p.split()) for p in paragraphs]
        
        # Calculate statistical metrics for paragraph lengths
        if word_counts:
            engagement_stats = {
                "avg_paragraph_length": round(np.mean(word_counts), 2),
                "engagement_std": round(np.std(word_counts), 2),
                "max_engagement_burst": max(word_counts),
                "min_engagement": min(word_counts),
                "engagement_coefficient_variation": round(np.std(word_counts) / np.mean(word_counts), 3) if np.mean(word_counts) > 0 else 0,
                "engagement_range": max(word_counts) - min(word_counts),
                "total_paragraphs": len(paragraphs)
            }
            
            # Calculate engagement quartiles
            quartiles = np.percentile(word_counts, [25, 50, 75])
            engagement_stats.update({
                "engagement_q1": round(quartiles[0], 2),
                "engagement_median": round(quartiles[1], 2),
                "engagement_q3": round(quartiles[2], 2)
            })
            
        else:
            engagement_stats = {
                "avg_paragraph_length": 0,
                "engagement_std": 0,
                "max_engagement_burst": 0,
                "min_engagement": 0,
                "engagement_coefficient_variation": 0,
                "engagement_range": 0,
                "total_paragraphs": 0,
                "engagement_q1": 0,
                "engagement_median": 0,
                "engagement_q3": 0
            }
        
        # Calculate time-series engagement (if timestamps available)
        engagement_stats.update(self._analyze_temporal_engagement(text))
        
        return engagement_stats
    
    def _analyze_temporal_engagement(self, text):
        """Analyze engagement changes over time"""
        # Look for timestamp patterns
        timestamp_pattern = r'\[(\d{2}):(\d{2}):(\d{2})\]'
        timestamps = re.findall(timestamp_pattern, text)
        
        if len(timestamps) < 2:
            return {
                "temporal_analysis": False,
                "engagement_trend": 0.0,
                "peak_time_periods": []
            }
        
        # Convert timestamps to seconds
        time_points = []
        for h, m, s in timestamps:
            total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            time_points.append(total_seconds)
        
        # Calculate engagement trend (linear regression slope)
        if len(time_points) >= 2:
            # Simple linear regression
            x = np.array(time_points)
            y = np.array(range(len(time_points)))  # Proxy for engagement
            
            # Calculate slope
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            
            return {
                "temporal_analysis": True,
                "engagement_trend": round(slope, 6),
                "conversation_duration": max(time_points) - min(time_points),
                "timestamp_count": len(timestamps)
            }
        
        return {
            "temporal_analysis": False,
            "engagement_trend": 0.0
        }
    
    def _calculate_topic_importance(self, semantic_data):
        """Calculate mathematical importance weights for extracted topics"""
        importance_weights = {}
        
        # Weight based on frequency and diversity
        for category, items in semantic_data.items():
            if isinstance(items, list) and items:
                # Frequency weight: more items = higher importance
                frequency_weight = len(items) / 10  # Normalize to 0-1 range
                
                # Diversity weight: unique items vs total items
                unique_items = len(set(str(item).lower() for item in items))
                diversity_weight = unique_items / len(items)
                
                # Content length weight: longer descriptions = higher importance
                avg_length = np.mean([len(str(item).split()) for item in items])
                length_weight = min(avg_length / 5, 1.0)  # Normalize
                
                # Combined importance score
                combined_weight = (frequency_weight * 0.4 + 
                                 diversity_weight * 0.3 + 
                                 length_weight * 0.3)
                
                importance_weights[category] = round(min(combined_weight, 1.0), 3)
            else:
                importance_weights[category] = 0.0
        
        return importance_weights
    
    def calculate_cross_transcript_similarity(self, transcript1, transcript2):
        """Calculate mathematical similarity between two transcripts"""
        try:
            # Create TF-IDF vectors for both transcripts
            documents = [transcript1, transcript2]
            vectors = self.vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(vectors)
            similarity_score = similarity_matrix[0][1]  # Similarity between doc 0 and doc 1
            
            return round(float(similarity_score), 4)
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0
    
    def calculate_information_density(self, text):
        """Calculate information density using entropy-like metrics"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Calculate word frequency distribution
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calculate entropy (information density)
        entropy = 0.0
        for word, freq in word_freq.items():
            probability = freq / total_words
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize entropy by theoretical maximum
        max_entropy = np.log2(len(word_freq)) if len(word_freq) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return round(normalized_entropy, 4)
    
    def analyze_question_patterns(self, text):
        """Analyze mathematical patterns in questions asked"""
        questions = re.findall(r'[^.!?]*\?', text)
        
        if not questions:
            return {
                "total_questions": 0,
                "avg_question_length": 0,
                "question_types": {}
            }
        
        # Analyze question types
        question_types = {
            "what_questions": len([q for q in questions if q.lower().strip().startswith('what')]),
            "how_questions": len([q for q in questions if q.lower().strip().startswith('how')]),
            "why_questions": len([q for q in questions if q.lower().strip().startswith('why')]),
            "can_questions": len([q for q in questions if q.lower().strip().startswith('can')]),
            "other_questions": 0
        }
        
        question_types["other_questions"] = len(questions) - sum(question_types.values())
        
        # Question length analysis
        question_lengths = [len(q.split()) for q in questions]
        
        return {
            "total_questions": len(questions),
            "avg_question_length": round(np.mean(question_lengths), 2),
            "question_types": question_types,
            "question_length_variance": round(np.var(question_lengths), 2) if len(question_lengths) > 1 else 0
        }
    
    def calculate_conversation_complexity(self, text, semantic_data):
        """Calculate overall conversation complexity score"""
        # Factor 1: Vocabulary complexity
        vocab_diversity = self.calculate_information_density(text)
        
        # Factor 2: Topic diversity
        all_topics = []
        for category in ["key_topics", "technical_topics", "objections_concerns"]:
            all_topics.extend(semantic_data.get(category, []))
        topic_diversity = len(set(all_topics)) / max(len(all_topics), 1) if all_topics else 0
        
        # Factor 3: Technical density
        content_metrics = self._analyze_content_density(text)
        technical_complexity = content_metrics.get("technical_density", 0) * 10  # Scale up
        
        # Factor 4: Question complexity
        question_metrics = self.analyze_question_patterns(text)
        question_complexity = question_metrics.get("avg_question_length", 0) / 10  # Normalize
        
        # Weighted combination
        complexity_score = (vocab_diversity * 0.3 + 
                          topic_diversity * 0.25 + 
                          technical_complexity * 0.25 + 
                          question_complexity * 0.2)
        
        return {
            "complexity_score": round(complexity_score, 4),
            "vocab_diversity": round(vocab_diversity, 4),
            "topic_diversity": round(topic_diversity, 4),
            "technical_complexity": round(technical_complexity, 4),
            "question_complexity": round(question_complexity, 4)
        }


def main():
    """Standalone test of mathematical analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test mathematical analysis")
    parser.add_argument("--test-file", help="Single transcript file to analyze")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    
    args = parser.parse_args()
    
    analyzer = MathematicalAnalyzer()
    
    if args.test_file:
        print(f" Analyzing {args.test_file}...")
        
        with open(args.test_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create dummy semantic data for testing
        semantic_data = {
            "key_topics": ["AI", "ML", "transformers"],
            "technical_topics": ["API", "backend", "neural networks"],
            "financial_mentions": ["$50k"],
            "objections_concerns": ["security concerns"],
            "next_steps": ["send assignment"]
        }
        
        results = analyzer.analyze_transcript(text, semantic_data)
        print(" Mathematical Analysis Results:")
        print(json.dumps(results, indent=2))
        
        # Additional complexity analysis
        complexity = analyzer.calculate_conversation_complexity(text, semantic_data)
        print("\n Conversation Complexity:")
        print(json.dumps(complexity, indent=2))
        
    elif args.test:
        print(" Running basic mathematical analysis tests...")
        
        # Test with sample text
        sample_text = """
        Shashwat (58%), Rohit (42%)
        
        Good afternoon, sir. I work with AI and ML technologies.
        Can you tell me about transformers? What is self-attention?
        We have a $50k budget for this quarter. 
        I'm concerned about the security implications.
        """
        
        sample_semantic = {
            "key_topics": ["AI", "ML", "transformers", "self-attention"],
            "technical_topics": ["transformers", "self-attention"],
            "financial_mentions": ["$50k budget"],
            "objections_concerns": ["security implications"]
        }
        
        results = analyzer.analyze_transcript(sample_text, sample_semantic)
        
        print(" Speaker Analysis:", json.dumps(results["speaker_analysis"], indent=2))
        print(" Content Density:", json.dumps(results["content_density"], indent=2))
        print(" Topic Importance:", json.dumps(results["topic_importance"], indent=2))
        
    else:
        print(" Usage:")
        print("  python utils/math_utils.py --test")
        print("  python utils/math_utils.py --test-file data/transcripts/sample.txt")


if __name__ == "__main__":
    main()
