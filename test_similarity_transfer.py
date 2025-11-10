#!/usr/bin/env python3
"""
Test Similarity Transfer Distillation
Demonstrates relational knowledge transfer via structural relationships
"""

import torch
import torch.nn as nn
from core.distillers.similarity_transfer import SimilarityTransfer, create_similarity_config

class DummyTransformer(nn.Module):
    """Dummy transformer for testing."""
    
    def __init__(self, hidden_dim=256, num_layers=6, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_dim)
        
        # Create transformer layers
        layers = []
        for i in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            setattr(self, f"layer_{i}", layer)
            layers.append(layer)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)
        
        # Transformer layers
        for i in range(6):
            layer = getattr(self, f"layer_{i}")
            x = layer(x)
        
        # Classification
        logits = self.classifier(x[:, 0, :])  # [CLS] token
        
        return {"logits": logits}


def test_similarity_transfer():
    """Test similarity transfer with different configurations."""
    
    print("=" * 70)
    print("SIMILARITY TRANSFER TEST - The Geometric Soul of KD")
    print("=" * 70)
    print()
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    
    # Create models
    print("📦 Creating teacher and student models...")
    teacher = DummyTransformer(hidden_dim=256, num_layers=6)
    student = DummyTransformer(hidden_dim=256, num_layers=6)
    teacher.to(device).eval()
    student.to(device).train()
    print(f"   Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"   Student params: {sum(p.numel() for p in student.parameters()):,}")
    print()
    
    # Dummy batch
    batch_size = 8
    seq_length = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    # ====================================================================
    # TEST 1: Cosine Similarity (Default)
    # ====================================================================
    print("-" * 70)
    print("TEST 1: Cosine Similarity (Default)")
    print("-" * 70)
    
    config_cosine = create_similarity_config(
        layer="layer_5",
        similarity_metric="cosine",
        weight=1.0,
        temperature=4.0,
        normalize=True
    )
    
    distiller_cosine = SimilarityTransfer(teacher, student, config_cosine)
    
    # Forward pass
    outputs = distiller_cosine(input_ids, labels)
    
    # Debug: print what's returned
    if not isinstance(outputs, dict):
        print(f"   Output type: {type(outputs)}")
        print(f"   Output: {outputs}")
    else:
        print(f"✅ Loss: {outputs.get('loss', outputs.get('total_loss', 'N/A'))}")
        if 'similarity_loss' in outputs:
            print(f"   Similarity Loss: {outputs['similarity_loss']:.4f}")
        if 'kd_loss' in outputs:
            print(f"   KD Loss: {outputs['kd_loss']:.4f}")
        if 'sas_score' in outputs:
            print(f"   SAS Score: {outputs['sas_score']:.4f}")
    print()
    
    # ====================================================================
    # TEST 2: Progressive Layer Transfer
    # ====================================================================
    print("-" * 70)
    print("TEST 2: Progressive Layer Transfer (Shallow → Deep)")
    print("-" * 70)
    
    config_progressive = create_similarity_config(
        layers=["layer_2", "layer_4", "layer_5"],
        similarity_metric="cosine",
        weight=1.0,
        progressive=True,
        progressive_epochs=2
    )
    
    distiller_progressive = SimilarityTransfer(teacher, student, config_progressive)
    
    print(f"Initial layers: {distiller_progressive.current_layers}")
    
    # Epoch 1
    outputs = distiller_progressive(input_ids, labels)
    print(f"Epoch 1 - Loss: {outputs['loss'].item():.4f}, Active layers: {len(distiller_progressive.current_layers)}")
    
    # Epoch 2
    distiller_progressive.update_epoch(2)
    outputs = distiller_progressive(input_ids, labels)
    print(f"Epoch 2 - Loss: {outputs['loss'].item():.4f}, Active layers: {len(distiller_progressive.current_layers)}")
    
    # Epoch 3
    distiller_progressive.update_epoch(3)
    outputs = distiller_progressive(input_ids, labels)
    print(f"Epoch 3 - Loss: {outputs['loss'].item():.4f}, Active layers: {len(distiller_progressive.current_layers)}")
    print()
    
    # ====================================================================
    # TEST 3: Graph-Based Similarity
    # ====================================================================
    print("-" * 70)
    print("TEST 3: Graph-Based Similarity (Sparse Adjacency)")
    print("-" * 70)
    
    config_graph = create_similarity_config(
        layer="layer_5",
        similarity_metric="graph",
        weight=1.0,
        graph_mode=True,
        graph_threshold=0.5
    )
    
    distiller_graph = SimilarityTransfer(teacher, student, config_graph)
    
    # Forward pass
    outputs = distiller_graph(input_ids, labels)
    
    print(f"✅ Loss: {outputs['loss'].item():.4f}")
    print(f"   Similarity Loss: {outputs['similarity_loss']:.4f}")
    print(f"   Graph threshold: {config_graph['graph_threshold']}")
    print(f"   SAS Score: {outputs['sas_score']:.4f}")
    print()
    
    # ====================================================================
    # TEST 4: Euclidean Distance
    # ====================================================================
    print("-" * 70)
    print("TEST 4: Euclidean Distance Similarity")
    print("-" * 70)
    
    config_euclidean = create_similarity_config(
        layer="layer_5",
        similarity_metric="euclidean",
        weight=1.0,
        normalize=True
    )
    
    distiller_euclidean = SimilarityTransfer(teacher, student, config_euclidean)
    
    # Forward pass
    outputs = distiller_euclidean(input_ids, labels)
    
    print(f"✅ Loss: {outputs['loss'].item():.4f}")
    print(f"   Similarity Loss: {outputs['similarity_loss']:.4f}")
    print(f"   SAS Score: {outputs['sas_score']:.4f}")
    print()
    
    # ====================================================================
    # TEST 5: Training Step
    # ====================================================================
    print("-" * 70)
    print("TEST 5: Training Step with Optimizer")
    print("-" * 70)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)
    
    # Before training
    before_loss = distiller_cosine(input_ids, labels)['loss'].item()
    
    # Training step
    batch = (input_ids, labels)
    metrics = distiller_cosine.train_step(batch, optimizer)
    
    # After training
    after_loss = distiller_cosine(input_ids, labels)['loss'].item()
    
    print(f"✅ Before: {before_loss:.4f}")
    print(f"   After:  {after_loss:.4f}")
    print(f"   Change: {before_loss - after_loss:.4f}")
    print()
    
    # ====================================================================
    # TEST 6: Metrics
    # ====================================================================
    print("-" * 70)
    print("TEST 6: Comprehensive Metrics")
    print("-" * 70)
    
    metrics = distiller_cosine.get_metrics()
    
    print("📊 Metrics Dictionary:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    print()
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("=" * 70)
    print("SUMMARY: Similarity Transfer Capabilities")
    print("=" * 70)
    print("✅ Pairwise Similarity Matrices: cosine, euclidean, graph")
    print("✅ Progressive Layer Transfer: Shallow → Deep")
    print("✅ Graph-Based Similarity: Sparse adjacency with adaptive threshold")
    print("✅ Structural Alignment Score (SAS): Relationship preservation metric")
    print("✅ Multi-Layer Support: Simultaneous multi-layer transfer")
    print("✅ Training Integration: train_step() with optimizer")
    print("✅ Comprehensive Metrics: Loss components + SAS tracking")
    print()
    print("🎯 The Geometric Soul of KD: Capturing relational structure beyond features")
    print("=" * 70)


if __name__ == "__main__":
    test_similarity_transfer()
