"""
Example: Using the Teacher Model Agent

This script demonstrates how to use the intelligent Teacher Agent
to automatically select and load the best teacher model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agents import TeacherModelAgent, quick_teacher_setup
import json


def load_sample_data(data_path: str, max_samples: int = 100):
    """Load sample data from JSONL file"""
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def example_1_quick_setup():
    """Example 1: Quick one-liner teacher setup"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Teacher Setup (One-Liner)")
    print("="*60)
    
    # Load sample data
    data_samples = load_sample_data("data/imdb_train.jsonl", max_samples=50)
    
    # Quick setup - one line!
    result = quick_teacher_setup(
        data_samples=data_samples,
        resource_constraint="medium"  # low, medium, or high
    )
    
    print("\n✅ Teacher Model Ready!")
    print(f"   Model: {result['model_name']}")
    print(f"   Task: {result['task']}")
    print(f"   Device: {result['device']}")
    
    if result['validation']:
        print(f"   Validation Accuracy: {result['validation']['accuracy']:.2%}")
        print(f"   Status: {result['validation']['recommendation']}")
    
    return result


def example_2_detailed_recommendations():
    """Example 2: See all teacher recommendations before loading"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Detailed Teacher Recommendations")
    print("="*60)
    
    # Initialize agent
    agent = TeacherModelAgent(device="cpu")  # or "cuda", "mps"
    
    # Load sample data
    data_samples = load_sample_data("data/imdb_train.jsonl", max_samples=50)
    
    # Detect task
    task = agent.detect_task_from_data(data_samples)
    print(f"\n🔍 Detected Task: {task}")
    
    # Get recommendations
    recommendations = agent.recommend_teachers(
        task=task,
        dataset_size=len(data_samples),
        resource_constraint="medium"
    )
    
    print(f"\n📋 Teacher Recommendations ({len(recommendations)} candidates):\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.model_name}")
        print(f"   Confidence: {rec.confidence:.0%}")
        print(f"   Size: {rec.estimated_size} ({rec.download_size} download)")
        print(f"   Task Fit: {rec.task_fit:.0%}")
        print(f"   Reasoning: {rec.reasoning}")
        print(f"   Fine-tuning needed: {'Yes' if rec.requires_finetuning else 'No'}")
        print()
    
    # Load the best one
    best = recommendations[0]
    print(f"✨ Loading best teacher: {best.model_name}")
    
    model, tokenizer = agent.load_teacher(
        model_name=best.model_name,
        task_type=task,
        num_labels=2
    )
    
    print("✅ Teacher loaded successfully!")
    
    return model, tokenizer, recommendations


def example_3_custom_task():
    """Example 3: Specify custom task and resource constraints"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Task with Resource Constraints")
    print("="*60)
    
    agent = TeacherModelAgent(device="mps")  # Mac M2 MPS
    
    # Custom task
    task = "sentiment_analysis"
    
    # Get recommendations for LOW resources (smaller models preferred)
    print("\n🔧 Resource Constraint: LOW (prefer smaller models)")
    low_recs = agent.recommend_teachers(
        task=task,
        dataset_size=1000,
        resource_constraint="low"
    )
    
    print("\nTop 3 for LOW resources:")
    for i, rec in enumerate(low_recs[:3], 1):
        print(f"  {i}. {rec.model_name} - {rec.estimated_size} (confidence: {rec.confidence:.0%})")
    
    # Get recommendations for HIGH resources (larger models preferred)
    print("\n🚀 Resource Constraint: HIGH (prefer larger models)")
    high_recs = agent.recommend_teachers(
        task=task,
        dataset_size=1000,
        resource_constraint="high"
    )
    
    print("\nTop 3 for HIGH resources:")
    for i, rec in enumerate(high_recs[:3], 1):
        print(f"  {i}. {rec.model_name} - {rec.estimated_size} (confidence: {rec.confidence:.0%})")


def example_4_validate_teacher():
    """Example 4: Validate teacher quality before using"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Teacher Quality Validation")
    print("="*60)
    
    agent = TeacherModelAgent()
    
    # Load data
    data_samples = load_sample_data("data/imdb_train.jsonl", max_samples=100)
    
    # Auto-select with validation
    result = agent.auto_select_and_load(
        data_samples=data_samples,
        resource_constraint="medium",
        validate=True  # Enable validation
    )
    
    print("\n📊 Validation Results:")
    val = result['validation']
    print(f"   Accuracy: {val['accuracy']:.2%}")
    print(f"   Samples Tested: {val['samples_tested']}")
    print(f"   Passed: {'✅ Yes' if val['passed'] else '❌ No'}")
    print(f"   Recommendation: {val['recommendation']}")
    
    if not val['passed']:
        print("\n⚠️  Teacher needs fine-tuning before distillation!")
    else:
        print("\n✅ Teacher is ready for distillation!")


def example_5_save_report():
    """Example 5: Save recommendation report to file"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Save Recommendation Report")
    print("="*60)
    
    agent = TeacherModelAgent()
    
    # Get recommendations
    recommendations = agent.recommend_teachers(
        task="sentiment_analysis",
        dataset_size=5000,
        resource_constraint="medium"
    )
    
    # Save report
    output_path = Path("teacher_recommendations.json")
    agent.save_recommendation_report(recommendations, output_path)
    
    print(f"\n💾 Report saved to: {output_path}")
    print("\nReport contains:")
    print("   - Model names and sizes")
    print("   - Confidence scores")
    print("   - Reasoning for each recommendation")
    print("   - Download sizes")
    print("   - Fine-tuning requirements")


def main():
    """Run all examples"""
    print("\n" + "🤖"*30)
    print("TEACHER MODEL AGENT EXAMPLES")
    print("🤖"*30)
    
    try:
        # Run examples
        example_1_quick_setup()
        example_2_detailed_recommendations()
        example_3_custom_task()
        example_4_validate_teacher()
        example_5_save_report()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED!")
        print("="*60)
        
    except FileNotFoundError:
        print("\n⚠️  Note: Some examples need data/imdb_train.jsonl to run")
        print("You can still see the demonstration of how to use the agent!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
