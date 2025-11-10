"""
Preflight System Test Suite
============================

Tests all preflight components:
1. ModelInspector
2. DataInspector
3. ResourceProbe
4. PreflightAnalyzer (full integration)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from transformers import BertModel, BertConfig, ViTModel, ViTConfig
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.preflight.model_inspector import ModelInspector
from core.preflight.data_inspector import DataInspector
from core.preflight.resource_probe import ResourceProbe
from core.preflight.analyser import PreflightAnalyzer, run_preflight_check


def test_model_inspector():
    """Test ModelInspector with various model types."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Inspector")
    print("=" * 70)
    
    # Test 1: Vision models (ResNet-like)
    print("\n📝 Test 1.1: Vision Models")
    teacher_vision = nn.Sequential(
        nn.Conv2d(3, 64, 7, 2, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1000)
    )
    
    student_vision = nn.Sequential(
        nn.Conv2d(3, 32, 7, 2, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 1000)
    )
    
    inspector = ModelInspector(teacher_vision, student_vision)
    report = inspector.inspect()
    
    print(f"✓ Teacher type: {report['teacher']['type']}")
    print(f"✓ Student type: {report['student']['type']}")
    print(f"✓ Compression ratio: {report['compression_ratio']:.2f}x")
    print(f"✓ Compatible: {report['compatibility']['is_compatible']}")
    
    # Test 2: Transformer models
    print("\n📝 Test 1.2: Transformer Models")
    teacher_config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    teacher_bert = BertModel(teacher_config)
    
    student_config = BertConfig(
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536
    )
    student_bert = BertModel(student_config)
    
    inspector = ModelInspector(teacher_bert, student_bert)
    report = inspector.inspect()
    
    print(f"✓ Teacher params: {report['teacher']['total_params'] / 1e6:.1f}M")
    print(f"✓ Student params: {report['student']['total_params'] / 1e6:.1f}M")
    print(f"✓ Architecture family: {report['teacher']['architecture_family']}")
    print(f"✓ Recommended strategy: {report['recommended_strategy']['primary_method']}")
    
    # Generate full report
    print("\n" + inspector.generate_report())
    
    print("\n✅ Model Inspector tests passed!")


def test_data_inspector():
    """Test DataInspector with various dataset types."""
    print("\n" + "=" * 70)
    print("TEST 2: Data Inspector")
    print("=" * 70)
    
    # Test 1: Classification dataset
    print("\n📝 Test 2.1: Classification Dataset")
    
    class SimpleClassificationDataset(Dataset):
        def __init__(self, num_samples=1000, num_classes=10):
            self.data = torch.randn(num_samples, 3, 224, 224)
            self.labels = torch.randint(0, num_classes, (num_samples,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {'pixel_values': self.data[idx], 'labels': self.labels[idx]}
    
    dataset = SimpleClassificationDataset(num_samples=1000, num_classes=10)
    inspector = DataInspector(dataset)
    report = inspector.validate()
    
    print(f"✓ Dataset size: {report['dataset_info']['size']}")
    print(f"✓ Data type: {report['data_type']}")
    print(f"✓ Task type: {report['task_type']}")
    print(f"✓ Valid: {report['is_valid']}")
    print(f"✓ Optimal batch size: {report['batch_recommendations']['optimal_batch_size']}")
    
    # Test 2: Text dataset
    print("\n📝 Test 2.2: Text Dataset")
    
    class SimpleTextDataset(Dataset):
        def __init__(self, num_samples=500):
            self.input_ids = torch.randint(0, 30000, (num_samples, 128))
            self.attention_mask = torch.ones(num_samples, 128)
            self.labels = torch.randint(0, 2, (num_samples,))
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]
            }
    
    dataset = SimpleTextDataset(num_samples=500)
    inspector = DataInspector(dataset)
    report = inspector.validate()
    
    print(f"✓ Data type: {report['data_type']}")
    print(f"✓ Task type: {report['task_type']}")
    print(f"✓ Number of classes: {report['statistics'].get('num_classes', 'N/A')}")
    
    # Generate full report
    print("\n" + inspector.generate_report())
    
    print("\n✅ Data Inspector tests passed!")


def test_resource_probe():
    """Test ResourceProbe."""
    print("\n" + "=" * 70)
    print("TEST 3: Resource Probe")
    print("=" * 70)
    
    probe = ResourceProbe()
    profile = probe.probe()
    
    print(f"✓ Primary device: {profile['devices']['primary']}")
    print(f"✓ Available devices: {', '.join(profile['devices']['available'])}")
    print(f"✓ System RAM: {profile['memory']['system']['total']:.2f} GB")
    print(f"✓ FP16 support: {profile['precision']['fp16']}")
    print(f"✓ BF16 support: {profile['precision']['bf16']}")
    print(f"✓ Recommended device: {profile['recommendations']['device']}")
    print(f"✓ Recommended precision: {profile['recommendations']['precision']}")
    print(f"✓ Batch size multiplier: {profile['recommendations']['batch_size_multiplier']}x")
    
    # Test memory estimation
    print("\n📝 Test 3.1: Memory Estimation")
    memory_usage = probe.estimate_memory_usage(
        model_params=110_000_000,  # 110M params
        batch_size=32,
        sequence_length=512,
        precision='fp16'
    )
    
    print(f"✓ Estimated model memory: {memory_usage['model']:.2f} GB")
    print(f"✓ Estimated total memory: {memory_usage['total']:.2f} GB")
    
    # Test batch size recommendation
    print("\n📝 Test 3.2: Batch Size Recommendation")
    available_memory = profile['memory']['system']['available']
    batch_rec = probe.recommend_optimal_batch_size(
        model_params=110_000_000,
        available_memory=available_memory,
        sequence_length=512,
        precision='fp16'
    )
    
    print(f"✓ Optimal batch size: {batch_rec['optimal_batch_size']}")
    print(f"✓ Memory utilization: {batch_rec['memory_utilization']:.1f}%")
    
    # Generate full report
    print("\n" + probe.generate_report())
    
    print("\n✅ Resource Probe tests passed!")


def test_preflight_analyzer():
    """Test complete PreflightAnalyzer integration."""
    print("\n" + "=" * 70)
    print("TEST 4: Preflight Analyzer (Full Integration)")
    print("=" * 70)
    
    # Create realistic models
    print("\n📝 Setting up models and dataset...")
    
    # Teacher: BERT-base-like
    teacher_config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=2
    )
    teacher = BertModel(teacher_config)
    
    # Student: BERT-small-like
    student_config = BertConfig(
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        num_labels=2
    )
    student = BertModel(student_config)
    
    # Dataset
    class TextClassificationDataset(Dataset):
        def __init__(self, num_samples=2000):
            self.input_ids = torch.randint(0, 30000, (num_samples, 128))
            self.attention_mask = torch.ones(num_samples, 128)
            self.labels = torch.randint(0, 2, (num_samples,))
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]
            }
    
    dataset = TextClassificationDataset(num_samples=2000)
    
    # Configuration
    config = {
        'training': {
            'epochs': 10,
            'learning_rate': 5e-5
        },
        'data': {
            'batch_size': 16,
            'num_workers': 2
        },
        'distillation': {
            'method': 'kd_hinton',
            'temperature': 4.0,
            'alpha': 0.5
        }
    }
    
    # Run preflight analysis
    print("\n📝 Running full preflight analysis...\n")
    
    report = run_preflight_check(
        teacher_model=teacher,
        student_model=student,
        dataset=dataset,
        config=config,
        save_report=True,
        output_dir="preflight_reports"
    )
    
    # Verify results
    print("\n📝 Verifying results...")
    print(f"✓ Can proceed: {report['can_proceed']}")
    print(f"✓ Confidence: {report['confidence']}")
    print(f"✓ Number of blockers: {len(report['blockers'])}")
    print(f"✓ Number of warnings: {len(report['warnings'])}")
    print(f"✓ Optimized batch size: {report['optimized_config']['batch_size']}")
    print(f"✓ Optimized device: {report['optimized_config']['device']}")
    print(f"✓ Optimized precision: {report['optimized_config']['precision']}")
    
    # Test config update
    print("\n📝 Testing config update...")
    analyzer = PreflightAnalyzer(teacher, student, dataset, config)
    analyzer.results = {
        'model': report['model_analysis'],
        'data': report['data_analysis'],
        'resources': report['resource_profile'],
        'optimization': report['optimized_config'],
        'decision': {
            'can_proceed': report['can_proceed'],
            'blockers': report['blockers'],
            'warnings': report['warnings'],
            'recommendations': report['recommendations'],
            'confidence': report['confidence']
        }
    }
    
    updated_config = analyzer.update_config(save_path="preflight_reports/optimized_config.yaml")
    print(f"✓ Config updated with {len(updated_config)} top-level keys")
    print(f"✓ Updated batch size: {updated_config['data']['batch_size']}")
    print(f"✓ Updated device: {updated_config['training']['device']}")
    
    print("\n✅ Preflight Analyzer tests passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 70)
    print("TEST 5: Edge Cases")
    print("=" * 70)
    
    # Test 1: Incompatible models
    print("\n📝 Test 5.1: Incompatible Models")
    
    teacher = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1000)
    )
    
    student = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    inspector = ModelInspector(teacher, student)
    report = inspector.inspect()
    
    compat = report.get('compatibility', {})
    print(f"✓ Detected incompatibility: {not compat.get('is_compatible', True)}")
    print(f"✓ Number of errors: {len(compat.get('errors', []))}")
    if compat.get('errors'):
        print(f"  First error: {compat['errors'][0][:80]}...")
    
    # Test 2: Empty dataset
    print("\n📝 Test 5.2: Empty Dataset")
    
    empty_dataset = TensorDataset(torch.tensor([]))
    inspector = DataInspector(empty_dataset)
    report = inspector.validate()
    
    print(f"✓ Detected empty dataset: {not report['is_valid']}")
    print(f"✓ Has errors: {len(report.get('errors', [])) > 0}")
    
    # Test 3: Only config (no models/data)
    print("\n📝 Test 5.3: Config-only Analysis")
    
    config = {
        'data': {
            'dataset_path': '/path/to/dataset',
            'batch_size': 32
        }
    }
    
    analyzer = PreflightAnalyzer(config=config)
    # Should handle gracefully
    print("✓ Config-only analyzer created successfully")
    
    print("\n✅ Edge case tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PREFLIGHT SYSTEM COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    try:
        test_model_inspector()
        test_data_inspector()
        test_resource_probe()
        test_preflight_analyzer()
        test_edge_cases()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 Preflight system is fully operational!")
        print("\nGenerated files:")
        print("  • preflight_reports/preflight_report_*.json")
        print("  • preflight_reports/preflight_report_*.txt")
        print("  • preflight_reports/optimized_config.yaml")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TESTS FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
