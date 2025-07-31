#!/usr/bin/env python3
"""
PyTorch Training Script for Machine Translation Quality Estimation using COMET v2.2.2

This script trains a quality estimation model that evaluates the quality of machine-translated
output given the source sentence, MT output, and reference translation.

Author: Generated for SageMaker deployment
"""

import argparse
import os
import sys
import json
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# COMET imports
from comet.models import download_model, load_from_checkpoint
from comet import COMETModel

# Optional imports for advanced logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with 'pip install wandb' for experiment tracking.")

try:
    import comet_ml
    COMET_ML_AVAILABLE = True
except ImportError:
    COMET_ML_AVAILABLE = False
    warnings.warn("comet_ml not available. Install with 'pip install comet_ml' for experiment tracking.")


class QualityEstimationDataset(Dataset):
    """
    Custom Dataset for Quality Estimation training.
    
    Expected input format:
    - CSV/TSV with columns: source, mt, reference, score
    - score should be a float between 0 and 1 (quality score)
    """
    
    def __init__(self, data_path: str, separator: str = '\t'):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the CSV/TSV file
            separator: Column separator ('\t' for TSV, ',' for CSV)
        """
        self.data_path = data_path
        self.separator = separator
        
        # Load and validate data
        self.data = self._load_data()
        self._validate_data()
        
        logging.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV/TSV file."""
        try:
            data = pd.read_csv(self.data_path, sep=self.separator)
            return data
        except Exception as e:
            raise ValueError(f"Error loading data from {self.data_path}: {e}")
    
    def _validate_data(self):
        """Validate that required columns exist and data is properly formatted."""
        required_columns = ['source', 'mt', 'reference', 'score']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_data = self.data[required_columns].isnull().sum()
        if missing_data.any():
            logging.warning(f"Found missing values: {missing_data.to_dict()}")
        
        # Validate score range
        score_min, score_max = self.data['score'].min(), self.data['score'].max()
        logging.info(f"Score range: {score_min:.3f} to {score_max:.3f}")
        
        if score_min < 0 or score_max > 1:
            logging.warning(f"Scores outside [0,1] range detected. Consider normalizing.")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys: 'source', 'mt', 'reference', 'score'
        """
        row = self.data.iloc[idx]
        return {
            'source': str(row['source']),
            'mt': str(row['mt']),
            'reference': str(row['reference']),
            'score': float(row['score'])
        }


class COMETQualityEstimator:
    """
    Wrapper class for COMET model with training capabilities.
    """
    
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da", device: str = "auto"):
        """
        Initialize the COMET Quality Estimator.
        
        Args:
            model_name: COMET model name or path to checkpoint
            device: Device to run model on ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load the COMET model."""
        try:
            if os.path.exists(self.model_name):
                # Load from local checkpoint
                logging.info(f"Loading model from checkpoint: {self.model_name}")
                self.model = load_from_checkpoint(self.model_name)
            else:
                # Download and load pre-trained model
                logging.info(f"Downloading model: {self.model_name}")
                model_path = download_model(self.model_name)
                self.model = load_from_checkpoint(model_path)
            
            # Move model to device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            logging.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def predict_batch(self, sources: List[str], mts: List[str], 
                     references: Optional[List[str]] = None) -> List[float]:
        """
        Predict quality scores for a batch of translations.
        
        Args:
            sources: List of source sentences
            mts: List of machine translations
            references: Optional list of reference translations
            
        Returns:
            List of quality scores
        """
        # Prepare data for COMET
        data = []
        for i in range(len(sources)):
            sample = {"src": sources[i], "mt": mts[i]}
            if references and i < len(references):
                sample["ref"] = references[i]
            data.append(sample)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(data, batch_size=32, gpus=1 if self.device == "cuda" else 0)
        
        return predictions
    
    def get_embeddings(self, sources: List[str], mts: List[str], 
                      references: Optional[List[str]] = None) -> torch.Tensor:
        """
        Get embeddings from the model for fine-tuning.
        
        Args:
            sources: List of source sentences
            mts: List of machine translations
            references: Optional list of reference translations
            
        Returns:
            Tensor of embeddings
        """
        # This method would be used for fine-tuning scenarios
        # Implementation depends on the specific COMET model architecture
        raise NotImplementedError("Embedding extraction not implemented for this COMET version")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data
    """
    sources = [item['source'] for item in batch]
    mts = [item['mt'] for item in batch]
    references = [item['reference'] for item in batch]
    scores = torch.tensor([item['score'] for item in batch], dtype=torch.float32)
    
    return {
        'sources': sources,
        'mts': mts,
        'references': references,
        'scores': scores
    }


def compute_correlations(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlations.
    
    Args:
        predictions: Predicted scores
        targets: True scores
        
    Returns:
        Dictionary with correlation metrics
    """
    pearson_corr, pearson_p = pearsonr(predictions, targets)
    spearman_corr, spearman_p = spearmanr(predictions, targets)
    
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    return {
        'pearson': pearson_corr,
        'pearson_p': pearson_p,
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'mse': mse,
        'mae': mae
    }


def train_epoch(model: COMETQualityEstimator, dataloader: DataLoader, 
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: str, clip_grad_norm: Optional[float] = None) -> float:
    """
    Train for one epoch.
    
    Args:
        model: COMET quality estimator
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        clip_grad_norm: Gradient clipping norm (optional)
        
    Returns:
        Average training loss
    """
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Get predictions from COMET model
        try:
            predictions = model.predict_batch(
                batch['sources'], 
                batch['mts'], 
                batch['references']
            )
            predictions = torch.tensor(predictions, dtype=torch.float32, device=device)
            targets = batch['scores'].to(device)
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_grad_norm is not None and hasattr(model.model, 'parameters'):
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), clip_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logging.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def evaluate(model: COMETQualityEstimator, dataloader: DataLoader, 
             criterion: nn.Module, device: str) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model.
    
    Args:
        model: COMET quality estimator
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, correlation_metrics)
    """
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                # Get predictions
                predictions = model.predict_batch(
                    batch['sources'],
                    batch['mts'],
                    batch['references']
                )
                predictions_tensor = torch.tensor(predictions, dtype=torch.float32, device=device)
                targets = batch['scores'].to(device)
                
                # Compute loss
                loss = criterion(predictions_tensor, targets)
                total_loss += loss.item()
                
                # Collect predictions and targets for correlation
                all_predictions.extend(predictions)
                all_targets.extend(batch['scores'].numpy())
                num_batches += 1
                
            except Exception as e:
                logging.error(f"Error in evaluation batch: {e}")
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute correlations
    if all_predictions and all_targets:
        correlations = compute_correlations(np.array(all_predictions), np.array(all_targets))
    else:
        correlations = {}
    
    return avg_loss, correlations


def save_model(model: COMETQualityEstimator, model_dir: str, epoch: int, 
               loss: float, metrics: Dict[str, float]):
    """
    Save model artifacts for SageMaker deployment.
    
    Args:
        model: Trained model
        model_dir: Directory to save model
        epoch: Current epoch
        loss: Current loss
        metrics: Performance metrics
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model checkpoint
    model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pt")
    
    # Create model state dict
    model_state = {
        'epoch': epoch,
        'model_name': model.model_name,
        'loss': loss,
        'metrics': metrics,
        'device': model.device
    }
    
    # If the model has state_dict (for fine-tuned models)
    if hasattr(model.model, 'state_dict'):
        model_state['model_state_dict'] = model.model.state_dict()
    
    torch.save(model_state, model_path)
    
    # Save metrics and config
    config_path = os.path.join(model_dir, "training_config.json")
    config = {
        'model_name': model.model_name,
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics,
        'device': model.device,
        'model_path': model_path
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save best model (for SageMaker deployment)
    best_model_path = os.path.join(model_dir, "model.pt")
    torch.save(model_state, best_model_path)
    
    logging.info(f"Model saved to {model_dir}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def setup_experiment_tracking(args: argparse.Namespace) -> Optional[Any]:
    """Setup experiment tracking with wandb or comet_ml."""
    experiment = None
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
        experiment = wandb
        logging.info("Initialized wandb experiment tracking")
    
    elif args.use_comet_ml and COMET_ML_AVAILABLE:
        experiment = comet_ml.Experiment(
            project_name=args.comet_project,
            experiment_name=args.experiment_name
        )
        experiment.log_parameters(vars(args))
        logging.info("Initialized comet_ml experiment tracking")
    
    return experiment


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train COMET Quality Estimation Model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data (CSV/TSV)")
    parser.add_argument("--val_data", type=str, required=True,
                       help="Path to validation data (CSV/TSV)")
    parser.add_argument("--data_separator", type=str, default="\t",
                       help="Data separator (\\t for TSV, , for CSV)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Unbabel/wmt22-comet-da",
                       help="COMET model name or path to checkpoint")
    parser.add_argument("--model_dir", type=str, default="./model_artifacts",
                       help="Directory to save model artifacts")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    parser.add_argument("--clip_grad_norm", type=float, default=None,
                       help="Gradient clipping norm")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for training")
    
    # Logging arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path (optional)")
    
    # Experiment tracking arguments
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use wandb for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="comet-qe",
                       help="Wandb project name")
    parser.add_argument("--use_comet_ml", action="store_true",
                       help="Use comet_ml for experiment tracking")
    parser.add_argument("--comet_project", type=str, default="comet-qe",
                       help="Comet ML project name")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name for tracking")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logging.info("Starting COMET Quality Estimation training")
    logging.info(f"Arguments: {vars(args)}")
    
    # Setup experiment tracking
    experiment = setup_experiment_tracking(args)
    
    try:
        # Load datasets
        logging.info("Loading datasets...")
        train_dataset = QualityEstimationDataset(args.train_data, args.data_separator)
        val_dataset = QualityEstimationDataset(args.val_data, args.data_separator)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2
        )
        
        # Initialize model
        logging.info(f"Loading COMET model: {args.model_name}")
        model = COMETQualityEstimator(args.model_name, args.device)
        
        # Setup training components
        criterion = nn.MSELoss()
        
        # Note: COMET models are typically not fine-tuned directly
        # This is a simplified training loop for demonstration
        # In practice, you might want to fine-tune specific layers or use a different approach
        if hasattr(model.model, 'parameters'):
            optimizer = optim.Adam(model.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            logging.warning("Model parameters not accessible for optimization. Running evaluation only.")
            optimizer = None
        
        # Training loop
        best_val_loss = float('inf')
        best_metrics = {}
        
        for epoch in range(args.epochs):
            logging.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            start_time = time.time()
            
            # Training phase
            if optimizer is not None:
                train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                       model.device, args.clip_grad_norm)
                logging.info(f"Training Loss: {train_loss:.4f}")
            else:
                train_loss = 0.0
                logging.info("Skipping training (model not trainable)")
            
            # Validation phase
            val_loss, val_metrics = evaluate(model, val_loader, criterion, model.device)
            
            # Log results
            epoch_time = time.time() - start_time
            logging.info(f"Validation Loss: {val_loss:.4f}")
            logging.info(f"Validation Metrics: {val_metrics}")
            logging.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # Log to experiment tracker
            if experiment:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch_time': epoch_time,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }
                
                if hasattr(experiment, 'log'):
                    experiment.log(log_dict)
                elif hasattr(experiment, 'log_metrics'):
                    experiment.log_metrics(log_dict, step=epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                save_model(model, args.model_dir, epoch + 1, val_loss, val_metrics)
                logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Final logging
        logging.info("\n--- Training Completed ---")
        logging.info(f"Best Validation Loss: {best_val_loss:.4f}")
        logging.info(f"Best Metrics: {best_metrics}")
        
        # Save final model
        save_model(model, args.model_dir, args.epochs, best_val_loss, best_metrics)
        
        # Create model.tar.gz for SageMaker
        logging.info("Creating model.tar.gz for SageMaker deployment...")
        import tarfile
        with tarfile.open("model.tar.gz", "w:gz") as tar:
            tar.add(args.model_dir, arcname=".")
        logging.info("model.tar.gz created successfully")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Clean up experiment tracking
        if experiment and hasattr(experiment, 'finish'):
            experiment.finish()


if __name__ == "__main__":
    main()