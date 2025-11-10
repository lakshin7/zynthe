#!/usr/bin/env python3
"""
Quick test script for AutoStudentBuilder

Usage:
    python test_auto_student.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from core.auto_student import AutoStudentBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def test_basic_generation():
    """Test basic student generation"""
    print("\n" + "="*70)
    print("TEST 1: Basic Student Generation")
    print("="*70 + "\n")
    
    builder = AutoStudentBuilder(teacher_name="bert-base-uncased")
    
    student = builder.generate(
        compression_ratio=0.5,
        strategy='balanced',
        validate=True,
        save=True
    )
    
    print("\n✓ Test 1 passed!\n")
    return student

def test_multiple_strategies():
    """Test multiple strategies"""
    print("\n" + "="*70)
    print("TEST 2: Multiple Sizing Strategies")
    print("="*70 + "\n")
    
    builder = AutoStudentBuilder(teacher_name="roberta-base")
    
    strategies = ['conservative', 'balanced', 'aggressive']
    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")
        student = builder.generate(
            compression_ratio=0.5,
            strategy=strategy,
            validate=True,
            save=False
        )
        print(f"✓ {strategy}: {student['total_params']:,} params\n")
    
    print("\n✓ Test 2 passed!\n")

def test_multiple_candidates():
    """Test multi-candidate generation"""
    print("\n" + "="*70)
    print("TEST 3: Multiple Candidate Generation")
    print("="*70 + "\n")
    
    builder = AutoStudentBuilder(teacher_name="bert-base-uncased")
    
    candidates = builder.generate_multiple(
        compression_ratios=[0.3, 0.5, 0.7],
        strategies=['balanced'],
        save=False
    )
    
    print(f"\n✓ Generated {len(candidates)} candidates")
    for i, candidate in enumerate(candidates, 1):
        print(f"  {i}. {candidate['strategy']} @ {candidate['compression_ratio']:.1%}: "
              f"{candidate['total_params']:,} params")
    
    print("\n✓ Test 3 passed!\n")

def test_memory_estimation():
    """Test memory feasibility"""
    print("\n" + "="*70)
    print("TEST 4: Memory Feasibility Estimation")
    print("="*70 + "\n")
    
    builder = AutoStudentBuilder(teacher_name="bert-base-uncased")
    
    ratios = [0.3, 0.5, 0.7, 0.9]
    for ratio in ratios:
        student = builder.generate(
            compression_ratio=ratio,
            strategy='balanced',
            validate=True,
            save=False
        )
        
        estimates = builder.estimate_training_time(student, dataset_size=1000)
        print(f"Compression {ratio:.1%}: {estimates['estimated_memory_gb']:.2f} GB, "
              f"~{estimates['estimated_time_minutes']:.1f} min")
    
    print("\n✓ Test 4 passed!\n")

def test_custom_teacher():
    """Test with custom teacher config"""
    print("\n" + "="*70)
    print("TEST 5: Custom Teacher Configuration")
    print("="*70 + "\n")
    
    custom_teacher = {
        'num_layers': 10,
        'hidden_size': 640,
        'num_attention_heads': 10,
        'intermediate_size': 2560,
        'vocab_size': 30522,
        'total_params': 80_000_000
    }
    
    builder = AutoStudentBuilder(
        teacher_name="custom-teacher",
        teacher_config=custom_teacher
    )
    
    student = builder.generate(
        compression_ratio=0.5,
        strategy='balanced',
        validate=True,
        save=False
    )
    
    print("\n✓ Test 5 passed!\n")

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("AutoStudentBuilder Test Suite")
    print("="*70)
    
    try:
        # Test 1: Basic generation
        student = test_basic_generation()
        
        # Test 2: Multiple strategies
        test_multiple_strategies()
        
        # Test 3: Multiple candidates
        test_multiple_candidates()
        
        # Test 4: Memory estimation
        test_memory_estimation()
        
        # Test 5: Custom teacher
        test_custom_teacher()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
        # Print example usage
        print("\nExample Usage:")
        print("-" * 70)
        print("""
from core.auto_student import AutoStudentBuilder

# Basic usage
builder = AutoStudentBuilder(teacher_name="bert-base-uncased")
student = builder.generate(
    compression_ratio=0.5,
    strategy='balanced',
    validate=True,
    save=True
)

# Multiple candidates
candidates = builder.generate_multiple(
    compression_ratios=[0.3, 0.5, 0.7],
    strategies=['conservative', 'balanced', 'aggressive']
)

# Estimate training
estimates = builder.estimate_training_time(student, dataset_size=10000)
print(f"Estimated time: {estimates['estimated_time_minutes']:.1f} minutes")
        """)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
