#!/usr/bin/env python3
"""
Training script for machine translation quality estimation using COMET v2.2.2

This script trains a custom COMET model on quality estimation datasets and saves
the trained model for SageMaker deployment.

Usage:
    python train_comet_model.py --data_path data.tsv --epochs 10 --batch_size 16 --learning_rate 1e-5
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import logging

# COMET imports
from comet.models import download_model, load_from_checkpoint
from comet.encoders import TransformerEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityEstimationDataset(Dataset):
    """
    Dataset class for quality estimation with source, MT, reference, and score.
    """
    
    def __init__(self, data: pd.DataFrame, tokenizer=None, max_length: int = 512):
        """
        Args:
            data: DataFrame with columns ['source', 'mt', 'reference', 'score']
            tokenizer: Tokenizer for text preprocessing (optional for COMET)
            max_length: Maximum sequence length
        """
        self.data = data.reset_index(drop=True)
        self.max_length = max_length
        
        # Validate required columns
        required_cols = ['source', 'mt', 'reference', 'score']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a sample formatted for COMET model input.
        """
        row = self.data.iloc[idx]
        
        # COMET expects specific format for input
        sample = {
            'src': str(row['source']),
            'mt': str(row['mt']),
            'ref': str(row['reference']),
            'score': float(row['score'])
        }
        
        return sample


class COMETTrainer:
    """
    Trainer class for fine-tuning COMET models on quality estimation tasks.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Unbabel/wmt22-comet-da",
        device: str = "auto",
        learning_rate: float = 1e-5,
        batch_size: int = 16,
        gradient_clipping: float = 1.0,
        model_dir: str = "./model_artifacts"
    ):
        """
        Initialize the COMET trainer.
        
        Args:
            model_name_or_path: COMET model name or path to checkpoint
            device: Device to use for training ('auto', 'cpu', 'cuda')
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            gradient_clipping: Maximum gradient norm for clipping
            model_dir: Directory to save model artifacts
        """
        self.model_name_or_path = model_name_or_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping
        self.model_dir = Path(model_dir)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and optimizer (will be set in load_model)
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """
        Load COMET model from checkpoint or download if needed.
        """
        logger.info(f"Loading COMET model: {self.model_name_or_path}")
        
        try:
            # First try to load as a local checkpoint
            if os.path.exists(self.model_name_or_path):
                self.model = load_from_checkpoint(self.model_name_or_path)
                logger.info("Loaded model from local checkpoint")
            else:
                # Download and load pre-trained model
                model_path = download_model(self.model_name_or_path)
                self.model = load_from_checkpoint(model_path)
                logger.info("Downloaded and loaded pre-trained model")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set model to training mode
            self.model.train()
            
            # Initialize optimizer
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-6
            )
            
            logger.info("Model loaded and optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_batch(self, batch: List[Dict]) -> Tuple[List[Dict], torch.Tensor]:
        """
        Prepare a batch for model input.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Tuple of (model_inputs, target_scores)
        """
        # Prepare inputs for COMET model
        model_inputs = []
        scores = []
        
        for sample in batch:
            model_inputs.append({
                'src': sample['src'],
                'mt': sample['mt'],
                'ref': sample['ref']
            })
            scores.append(sample['score'])
        
        # Convert scores to tensor
        target_scores = torch.tensor(scores, dtype=torch.float32, device=self.device)
        
        return model_inputs, target_scores
    
    def training_step(self, batch: List[Dict]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training samples
            
        Returns:
            Training loss for this batch
        """
        self.optimizer.zero_grad()
        
        # Prepare batch
        model_inputs, target_scores = self.prepare_batch(batch)
        
        # Forward pass
        try:
            predictions = self.model.predict(model_inputs, batch_size=len(model_inputs))
            pred_scores = torch.tensor(predictions, dtype=torch.float32, device=self.device)
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise
        
        # Calculate loss
        loss = self.criterion(pred_scores, target_scores)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clipping > 0:
            clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def validation_step(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Perform validation on the entire validation set.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Prepare batch
                model_inputs, target_scores = self.prepare_batch(batch)
                
                # Forward pass
                predictions = self.model.predict(model_inputs, batch_size=len(model_inputs))
                pred_scores = torch.tensor(predictions, dtype=torch.float32, device=self.device)
                
                # Calculate loss
                loss = self.criterion(pred_scores, target_scores)
                total_loss += loss.item()
                
                # Store predictions and targets for correlation
                all_predictions.extend(predictions)
                all_targets.extend(target_scores.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        pearson_corr, _ = pearsonr(all_predictions, all_targets)
        spearman_corr, _ = spearmanr(all_predictions, all_targets)
        
        self.model.train()  # Set back to training mode
        
        return {
            'val_loss': avg_loss,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int = 5
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'pearson_correlation': [],
            'spearman_correlation': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                loss = self.training_step(batch)
                total_train_loss += loss
                num_batches += 1
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_train_loss / num_batches
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                              f"Avg Loss: {avg_loss:.4f}")
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / len(train_dataloader)
            
            # Validation phase
            val_metrics = self.validation_step(val_dataloader)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['val_loss'])
            history['pearson_correlation'].append(val_metrics['pearson_correlation'])
            history['spearman_correlation'].append(val_metrics['spearman_correlation'])
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Pearson Correlation: {val_metrics['pearson_correlation']:.4f}")
            logger.info(f"Spearman Correlation: {val_metrics['spearman_correlation']:.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_model(f"best_model_epoch_{epoch+1}")
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            print("-" * 50)
        
        logger.info("Training completed!")
        return history
    
    def save_model(self, model_name: str = "final_model"):
        """
        Save the trained model for SageMaker deployment.
        
        Args:
            model_name: Name for the saved model
        """
        save_path = self.model_dir / f"{model_name}.pt"
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.hparams,
        }, save_path)
        
        # Also save the entire model using COMET's save method if available
        try:
            model_checkpoint_path = self.model_dir / f"{model_name}_checkpoint.ckpt"
            # For COMET models, we might need to use their specific save method
            if hasattr(self.model, 'save_checkpoint'):
                self.model.save_checkpoint(model_checkpoint_path)
            else:
                # Fallback to PyTorch Lightning checkpoint save
                torch.save(self.model, model_checkpoint_path)
        except Exception as e:
            logger.warning(f"Could not save COMET checkpoint format: {e}")
        
        logger.info(f"Model saved to {save_path}")


def load_dataset(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split dataset into train and validation sets.
    
    Args:
        data_path: Path to the dataset file (CSV or TSV)
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    logger.info(f"Loading dataset from {data_path}")
    
    # Determine file type and load accordingly
    if data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        # Try to load as CSV first, then TSV
        try:
            df = pd.read_csv(data_path)
        except:
            df = pd.read_csv(data_path, sep='\t')
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Dataset columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['source', 'mt', 'reference', 'score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    # Remove rows with missing values
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(f"Removed {initial_len - len(df)} rows with missing values")
    
    # Split dataset
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Can't stratify continuous scores
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    
    return train_df, val_df


def create_data_loaders(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = QualityEstimationDataset(train_df)
    val_dataset = QualityEstimationDataset(val_df)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        collate_fn=lambda x: x  # Return list of samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x
    )
    
    return train_loader, val_loader


def main():
    """
    Main training function with CLI interface.
    """
    parser = argparse.ArgumentParser(description="Train COMET quality estimation model")
    
    # Data arguments
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to training data (CSV/TSV with columns: source, mt, reference, score)"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Unbabel/wmt22-comet-da",
        help="COMET model name or path to checkpoint (default: Unbabel/wmt22-comet-da)"
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="./model_artifacts",
        help="Directory to save trained model (default: ./model_artifacts)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5,
        help="Learning rate for optimizer (default: 1e-5)"
    )
    parser.add_argument(
        "--gradient_clipping", 
        type=float, 
        default=1.0,
        help="Maximum gradient norm for clipping (default: 1.0)"
    )
    
    # Device arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto)"
    )
    
    # Misc arguments
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    logger.info("Starting COMET quality estimation training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load and split dataset
        train_df, val_df = load_dataset(
            data_path=args.data_path,
            test_size=args.test_size,
            random_state=args.random_seed
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_df=train_df,
            val_df=val_df,
            batch_size=args.batch_size
        )
        
        # Initialize trainer
        trainer = COMETTrainer(
            model_name_or_path=args.model_name,
            device=args.device,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_clipping=args.gradient_clipping,
            model_dir=args.model_dir
        )
        
        # Load model
        trainer.load_model()
        
        # Train model
        history = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs
        )
        
        # Save final model
        trainer.save_model("final_model")
        
        # Save training history
        history_path = Path(args.model_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model artifacts saved to: {args.model_dir}")
        logger.info(f"Training history saved to: {history_path}")
        
        # Print final metrics
        final_pearson = history['pearson_correlation'][-1]
        final_spearman = history['spearman_correlation'][-1]
        final_val_loss = history['val_loss'][-1]
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Final Pearson Correlation: {final_pearson:.4f}")
        print(f"Final Spearman Correlation: {final_spearman:.4f}")
        print(f"Model saved to: {args.model_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()