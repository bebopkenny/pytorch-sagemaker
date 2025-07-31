#!/usr/bin/env python3
"""
Utility script to create a sample dataset for COMET quality estimation training.

This script generates synthetic data with the required columns: source, mt, reference, score.
Use this for testing the training pipeline before using real data.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def create_sample_dataset(num_samples: int = 1000, output_path: str = "sample_data.tsv"):
    """
    Create a sample dataset for quality estimation training.
    
    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the dataset
    """
    np.random.seed(42)  # For reproducibility
    
    # Sample source sentences (English)
    source_templates = [
        "Hello, how are you today?",
        "The weather is very nice outside.",
        "I would like to order some food.",
        "Can you help me with this problem?",
        "The meeting is scheduled for tomorrow.",
        "This is a very important document.",
        "Please send me the report by Friday.",
        "The train arrives at 3:30 PM.",
        "I need to buy groceries after work.",
        "The movie was really entertaining.",
    ]
    
    # Sample reference translations (Spanish)
    reference_templates = [
        "Hola, ¿cómo estás hoy?",
        "El clima está muy agradable afuera.",
        "Me gustaría pedir algo de comida.",
        "¿Puedes ayudarme con este problema?",
        "La reunión está programada para mañana.",
        "Este es un documento muy importante.",
        "Por favor envíame el reporte antes del viernes.",
        "El tren llega a las 3:30 PM.",
        "Necesito comprar comestibles después del trabajo.",
        "La película fue realmente entretenida.",
    ]
    
    # Generate variations and MT translations with different quality levels
    data = []
    
    for i in range(num_samples):
        # Select base templates
        src_idx = i % len(source_templates)
        source = source_templates[src_idx]
        reference = reference_templates[src_idx]
        
        # Add some variation to source
        if np.random.random() > 0.7:
            source = source.replace("very", "extremely")
        if np.random.random() > 0.8:
            source = source.replace(".", "!")
        
        # Generate MT with different quality levels
        quality_level = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
        
        if quality_level == 'high':
            # High quality: mostly correct translation
            mt = reference
            if np.random.random() > 0.9:  # 10% chance of minor error
                mt = mt.replace("muy", "bastante")
            score = np.random.normal(0.8, 0.1)  # High score around 0.8
            
        elif quality_level == 'medium':
            # Medium quality: some errors
            mt = reference
            if np.random.random() > 0.5:  # 50% chance of error
                mt = mt.replace("está", "es")
            if np.random.random() > 0.7:  # 30% chance of word order issue
                words = mt.split()
                if len(words) > 3:
                    # Swap two words
                    idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                    words[idx1], words[idx2] = words[idx2], words[idx1]
                    mt = " ".join(words)
            score = np.random.normal(0.5, 0.15)  # Medium score around 0.5
            
        else:  # low quality
            # Low quality: significant errors
            mt = reference
            # Multiple types of errors
            if np.random.random() > 0.3:  # 70% chance of wrong word
                mt = mt.replace("hoy", "ayer")
            if np.random.random() > 0.4:  # 60% chance of missing word
                words = mt.split()
                if len(words) > 2:
                    words.pop(np.random.randint(len(words)))
                    mt = " ".join(words)
            if np.random.random() > 0.5:  # 50% chance of extra word
                mt = mt + " extra"
            score = np.random.normal(0.2, 0.1)  # Low score around 0.2
        
        # Ensure score is in valid range [0, 1]
        score = max(0.0, min(1.0, score))
        
        data.append({
            'source': source,
            'mt': mt,
            'reference': reference,
            'score': round(score, 4)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, sep='\t', index=False)
    
    print(f"✓ Created sample dataset with {len(df)} samples")
    print(f"✓ Saved to: {output_path}")
    print(f"\nDataset statistics:")
    print(f"- Columns: {list(df.columns)}")
    print(f"- Score range: {df['score'].min():.3f} - {df['score'].max():.3f}")
    print(f"- Score mean: {df['score'].mean():.3f}")
    print(f"- Score std: {df['score'].std():.3f}")
    print(f"\nFirst 3 samples:")
    print(df.head(3).to_string(index=False))
    
    return df

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Create sample dataset for COMET training")
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="sample_data.tsv",
        help="Output file path (default: sample_data.tsv)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['tsv', 'csv'],
        default='tsv',
        help="Output format (default: tsv)"
    )
    
    args = parser.parse_args()
    
    # Adjust output path based on format
    if args.format == 'csv' and not args.output_path.endswith('.csv'):
        args.output_path = args.output_path.replace('.tsv', '.csv')
    elif args.format == 'tsv' and not args.output_path.endswith('.tsv'):
        args.output_path = args.output_path.replace('.csv', '.tsv')
    
    # Create dataset
    create_sample_dataset(args.num_samples, args.output_path)
    
    print(f"\n📖 Usage example:")
    print(f"python train_comet_model.py --data_path {args.output_path} --epochs 3 --batch_size 8")

if __name__ == "__main__":
    main()