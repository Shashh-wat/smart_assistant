"""
Data Management Utilities - Sophisticated Version
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter


def create_directories(dir_paths: List[str]):
    """Create directories if they don't exist"""
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"📁 Created directory: {dir_path}")
        else:
            # Check if it's a file instead of directory
            if os.path.isfile(dir_path):
                print(f"⚠️ {dir_path} exists as file, removing...")
                os.remove(dir_path)
                os.makedirs(dir_path)
                print(f"📁 Created directory: {dir_path}")


def save_processed_data(data: Dict[str, Any], output_path: str):
    """Save processed data to JSON file with comprehensive metadata"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Calculate comprehensive statistics
        stats = calculate_dataset_statistics(data)
        
        save_metadata = {
            "processed_at": datetime.now().isoformat(),
            "total_meetings": len(data),
            "dataset_statistics": stats,
            "processing_metadata": {
                "extraction_completeness": calculate_extraction_completeness(data),
                "mathematical_features_available": check_mathematical_features(data),
                "semantic_categories_covered": get_semantic_categories(data)
            },
            "meetings": data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved processed data to {output_path}")
        print(f"📊 Dataset contains {len(data)} meetings with {stats['total_extracted_entities']} total entities")
        
    except Exception as e:
        print(f"❌ Failed to save data: {e}")


def calculate_dataset_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive statistics across all meetings"""
    stats = {
        "total_extracted_entities": 0,
        "avg_entities_per_meeting": 0,
        "most_common_topics": [],
        "most_common_technical": [],
        "speaker_distribution": {},
        "content_metrics": {
            "total_characters": 0,
            "total_words": 0,
            "avg_technical_density": 0,
            "avg_question_density": 0
        }
    }
    
    all_topics = []
    all_technical = []
    all_speakers = []
    total_entities = 0
    technical_densities = []
    question_densities = []
    
    for meeting_data in data.values():
        semantic = meeting_data.get("semantic_data", {})
        mathematical = meeting_data.get("mathematical_features", {})
        
        # Count entities
        for category, items in semantic.items():
            if isinstance(items, list):
                total_entities += len(items)
                if category == "key_topics":
                    all_topics.extend(items)
                elif category == "technical_topics":
                    all_technical.extend(items)
        
        # Speaker analysis
        speaker_analysis = mathematical.get("speaker_analysis", {})
        if "speaking_ratios" in speaker_analysis:
            for speaker, ratio in speaker_analysis["speaking_ratios"].items():
                all_speakers.append(speaker)
        
        # Content metrics
        content_density = mathematical.get("content_density", {})
        if content_density:
            technical_densities.append(content_density.get("technical_density", 0))
            question_densities.append(content_density.get("question_density", 0))
            stats["content_metrics"]["total_words"] += content_density.get("total_word_count", 0)
        
        # Character count from metadata
        metadata = meeting_data.get("metadata", {})
        stats["content_metrics"]["total_characters"] += metadata.get("character_count", 0)
    
    # Calculate final statistics
    stats["total_extracted_entities"] = total_entities
    stats["avg_entities_per_meeting"] = round(total_entities / len(data), 1) if data else 0
    
    # Most common items
    if all_topics:
        topic_counts = Counter(all_topics)
        stats["most_common_topics"] = topic_counts.most_common(5)
    
    if all_technical:
        tech_counts = Counter(all_technical)
        stats["most_common_technical"] = tech_counts.most_common(5)
    
    if all_speakers:
        speaker_counts = Counter(all_speakers)
        stats["speaker_distribution"] = dict(speaker_counts.most_common())
    
    # Average densities
    if technical_densities:
        stats["content_metrics"]["avg_technical_density"] = round(np.mean(technical_densities), 4)
    if question_densities:
        stats["content_metrics"]["avg_question_density"] = round(np.mean(question_densities), 4)
    
    return stats


def calculate_extraction_completeness(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate how complete the semantic extraction is"""
    required_categories = [
        "speakers", "key_topics", "technical_topics", "feedback_given",
        "assignments_given", "next_steps", "objections_concerns"
    ]
    
    completeness_scores = {}
    
    for category in required_categories:
        meetings_with_category = 0
        total_items = 0
        
        for meeting_data in data.values():
            semantic = meeting_data.get("semantic_data", {})
            if category in semantic and semantic[category]:
                meetings_with_category += 1
                if isinstance(semantic[category], list):
                    total_items += len(semantic[category])
                elif isinstance(semantic[category], str) and semantic[category].strip():
                    total_items += 1
        
        coverage_ratio = meetings_with_category / len(data) if data else 0
        avg_items = total_items / len(data) if data else 0
        
        completeness_scores[category] = {
            "coverage_ratio": round(coverage_ratio, 3),
            "avg_items_per_meeting": round(avg_items, 2),
            "total_items": total_items
        }
    
    return completeness_scores


def check_mathematical_features(data: Dict[str, Any]) -> Dict[str, bool]:
    """Check which mathematical features are available"""
    mathematical_checks = {
        "speaker_analysis": False,
        "content_density": False,
        "conversation_flow": False,
        "semantic_clusters": False,
        "engagement_metrics": False,
        "topic_importance": False
    }
    
    for meeting_data in data.values():
        mathematical = meeting_data.get("mathematical_features", {})
        
        for feature in mathematical_checks.keys():
            if feature in mathematical and mathematical[feature]:
                mathematical_checks[feature] = True
    
    return mathematical_checks


def get_semantic_categories(data: Dict[str, Any]) -> List[str]:
    """Get all semantic categories found in the data"""
    all_categories = set()
    
    for meeting_data in data.values():
        semantic = meeting_data.get("semantic_data", {})
        all_categories.update(semantic.keys())
    
    return sorted(list(all_categories))


def load_processed_data(data_path: str) -> Dict[str, Any]:
    """Load processed data from JSON file"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "meetings" in data:
            meetings_data = data["meetings"]
            print(f"📊 Loaded {len(meetings_data)} meetings")
            
            # Print dataset summary if available
            if "dataset_statistics" in data:
                stats = data["dataset_statistics"]
                print(f"   Total entities: {stats.get('total_extracted_entities', 'Unknown')}")
                print(f"   Total words: {stats.get('content_metrics', {}).get('total_words', 'Unknown')}")
            
            return meetings_data
        else:
            # Legacy format
            return data
            
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return {}


def validate_processed_data(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive validation of processed data structure"""
    validation_report = {
        "total_meetings": len(processed_data),
        "valid_meetings": 0,
        "issues": [],
        "warnings": [],
        "completeness_score": 0.0,
        "quality_metrics": {}
    }
    
    required_keys = ["metadata", "semantic_data", "mathematical_features"]
    semantic_required = ["speakers", "key_topics", "executive_summary"]
    mathematical_required = ["speaker_analysis", "content_density"]
    
    quality_scores = []
    
    for meeting_id, meeting_data in processed_data.items():
        meeting_valid = True
        meeting_quality = 0.0
        
        # Check required top-level keys
        for key in required_keys:
            if key not in meeting_data:
                validation_report["issues"].append(f"{meeting_id}: Missing '{key}'")
                meeting_valid = False
            else:
                meeting_quality += 0.33  # 33% for each required key
        
        # Check semantic data completeness
        if "semantic_data" in meeting_data:
            semantic = meeting_data["semantic_data"]
            semantic_score = 0
            
            for key in semantic_required:
                if key in semantic and semantic[key]:
                    semantic_score += 1
            
            semantic_completeness = semantic_score / len(semantic_required)
            meeting_quality += semantic_completeness * 0.4  # 40% for semantic completeness
            
            # Count non-empty categories
            non_empty_categories = sum(1 for v in semantic.values() 
                                     if (isinstance(v, list) and v) or 
                                        (isinstance(v, str) and v.strip()))
            
            if non_empty_categories < 5:
                validation_report["warnings"].append(
                    f"{meeting_id}: Only {non_empty_categories} semantic categories populated"
                )
        
        # Check mathematical features
        if "mathematical_features" in meeting_data:
            mathematical = meeting_data["mathematical_features"]
            math_score = 0
            
            for key in mathematical_required:
                if key in mathematical and mathematical[key]:
                    math_score += 1
            
            math_completeness = math_score / len(mathematical_required)
            meeting_quality += math_completeness * 0.3  # 30% for mathematical completeness
        
        # Check metadata
        if "metadata" in meeting_data:
            metadata = meeting_data["metadata"]
            required_metadata = ["meeting_id", "processed_at", "character_count"]
            
            missing_metadata = [key for key in required_metadata if key not in metadata]
            if missing_metadata:
                validation_report["warnings"].append(
                    f"{meeting_id}: Missing metadata fields: {missing_metadata}"
                )
        
        if meeting_valid:
            validation_report["valid_meetings"] += 1
        
        quality_scores.append(meeting_quality)
    
    # Calculate overall completeness score
    validation_report["completeness_score"] = round(np.mean(quality_scores), 3) if quality_scores else 0.0
    
    # Quality metrics
    validation_report["quality_metrics"] = {
        "avg_quality_score": round(np.mean(quality_scores), 3) if quality_scores else 0.0,
        "min_quality_score": round(min(quality_scores), 3) if quality_scores else 0.0,
        "max_quality_score": round(max(quality_scores), 3) if quality_scores else 0.0,
        "quality_std": round(np.std(quality_scores), 3) if len(quality_scores) > 1 else 0.0
    }
    
    return validation_report


def generate_data_report(processed_data: Dict[str, Any]) -> str:
    """Generate comprehensive data report"""
    if not processed_data:
        return "📭 No processed data available"
    
    report_sections = []
    
    # Basic statistics
    report_sections.append(f"📊 **Dataset Overview**")
    report_sections.append(f"   Total Meetings: {len(processed_data)}")
    
    # Validation report
    validation = validate_processed_data(processed_data)
    report_sections.append(f"   Valid Meetings: {validation['valid_meetings']}")
    report_sections.append(f"   Quality Score: {validation['completeness_score']:.1%}")
    
    if validation['issues']:
        report_sections.append(f"   Issues: {len(validation['issues'])}")
    
    # Content statistics
    stats = calculate_dataset_statistics(processed_data)
    report_sections.append(f"\n🎯 **Content Analysis**")
    report_sections.append(f"   Total Entities Extracted: {stats['total_extracted_entities']}")
    report_sections.append(f"   Avg Entities per Meeting: {stats['avg_entities_per_meeting']}")
    report_sections.append(f"   Total Words: {stats['content_metrics']['total_words']:,}")
    
    # Top topics
    if stats['most_common_topics']:
        top_topics = [f"{topic} ({count})" for topic, count in stats['most_common_topics'][:3]]
        report_sections.append(f"   Top Topics: {', '.join(top_topics)}")
    
    # Technical analysis
    if stats['content_metrics']['avg_technical_density'] > 0:
        tech_density = stats['content_metrics']['avg_technical_density']
        report_sections.append(f"   Avg Technical Density: {tech_density:.1%}")
    
    return "\n".join(report_sections)


def export_summary_csv(processed_data: Dict[str, Any], output_path: str):
    """Export meeting summaries to CSV for external analysis"""
    try:
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'meeting_id', 'character_count', 'word_count', 
                'num_speakers', 'key_topics_count', 'technical_topics_count',
                'feedback_count', 'assignments_count', 'technical_density',
                'question_density', 'executive_summary'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for meeting_id, meeting_data in processed_data.items():
                metadata = meeting_data.get('metadata', {})
                semantic = meeting_data.get('semantic_data', {})
                mathematical = meeting_data.get('mathematical_features', {})
                
                content_density = mathematical.get('content_density', {})
                speaker_analysis = mathematical.get('speaker_analysis', {})
                
                row = {
                    'meeting_id': meeting_id,
                    'character_count': metadata.get('character_count', 0),
                    'word_count': metadata.get('word_count', 0),
                    'num_speakers': speaker_analysis.get('total_speakers', 0),
                    'key_topics_count': len(semantic.get('key_topics', [])),
                    'technical_topics_count': len(semantic.get('technical_topics', [])),
                    'feedback_count': len(semantic.get('feedback_given', [])),
                    'assignments_count': len(semantic.get('assignments_given', [])),
                    'technical_density': content_density.get('technical_density', 0),
                    'question_density': content_density.get('question_density', 0),
                    'executive_summary': semantic.get('executive_summary', '')[:200] + '...'  # Truncate
                }
                
                writer.writerow(row)
        
        print(f"📊 Exported summary CSV to {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to export CSV: {e}")


def main():
    """Test data utilities"""
    print("🧪 Testing data utilities...")
    
    # Test directory creation
    test_dirs = ['test_output', 'test_output/subdir']
    create_directories(test_dirs)
    
    # Clean up
    import shutil
    if os.path.exists('test_output'):
        shutil.rmtree('test_output')
    
    print("✅ Data utilities tests passed")


if __name__ == "__main__":
    main()