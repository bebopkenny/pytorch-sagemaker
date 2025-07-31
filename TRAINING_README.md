# COMET Quality Estimation Training Script

This repository contains a comprehensive training script for fine-tuning COMET v2.2.2 models on custom quality estimation datasets. The trained models can be deployed on AWS SageMaker for production use.

## 🎯 Overview

The training script (`train_comet_model.py`) provides:
- **Model Loading**: Support for both pre-trained and custom COMET checkpoints
- **Dataset Handling**: Flexible input formats (CSV/TSV) with automatic validation
- **Training Loop**: Full PyTorch training with validation, metrics, and best model saving
- **CLI Interface**: Command-line arguments for easy hyperparameter tuning
- **SageMaker Ready**: Output models in format ready for SageMaker deployment

## 📋 Requirements

### Core Dependencies
```bash
pip install -r train_requirements.txt
```

Key packages:
- `unbabel-comet==2.2.2` - COMET model library
- `torch>=1.11.0` - PyTorch for training
- `pandas>=1.3.0` - Data handling
- `scikit-learn>=1.0.0` - Train/validation split and metrics
- `scipy>=1.7.0` - Statistical metrics (Pearson/Spearman correlation)

## 📊 Data Format

Your training data must be a CSV or TSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `source` | Source sentence | "Hello, how are you?" |
| `mt` | Machine translation output | "Hola, ¿cómo estás?" |
| `reference` | Reference translation | "Hola, ¿cómo te va?" |
| `score` | Quality score (0.0 to 1.0) | 0.85 |

### Example Dataset
```tsv
source	mt	reference	score
Hello, how are you?	Hola, ¿cómo estás?	Hola, ¿cómo te va?	0.85
The weather is nice.	El clima es bueno.	El tiempo está agradable.	0.72
I need help.	Necesito ayuda.	Necesito ayuda.	1.0
```

## 🚀 Quick Start

### 1. Create Sample Data (Optional)
Generate synthetic data for testing:
```bash
python create_sample_dataset.py --num_samples 1000 --output_path sample_data.tsv
```

### 2. Basic Training
Train with default settings:
```bash
python train_comet_model.py --data_path sample_data.tsv
```

### 3. Custom Training
Train with custom hyperparameters:
```bash
python train_comet_model.py \
    --data_path your_data.tsv \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --model_dir ./my_model_artifacts
```

## 🔧 Command Line Arguments

### Data Arguments
- `--data_path` (required): Path to training data (CSV/TSV)
- `--test_size`: Fraction for validation split (default: 0.2)

### Model Arguments
- `--model_name`: COMET model name or checkpoint path (default: "Unbabel/wmt22-comet-da")
- `--model_dir`: Directory to save trained model (default: "./model_artifacts")

### Training Arguments
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate for optimizer (default: 1e-5)
- `--gradient_clipping`: Maximum gradient norm for clipping (default: 1.0)

### System Arguments
- `--device`: Device for training - "auto", "cpu", or "cuda" (default: "auto")
- `--random_seed`: Random seed for reproducibility (default: 42)

## 📈 Training Process

### 1. Data Loading & Validation
- Loads CSV/TSV data and validates required columns
- Splits data into train/validation sets
- Creates PyTorch DataLoaders with proper batching

### 2. Model Initialization
- Downloads and loads pre-trained COMET model
- Sets up AdamW optimizer with weight decay
- Configures MSE loss for regression task

### 3. Training Loop
- Trains for specified epochs with mini-batching
- Applies gradient clipping for stability
- Validates after each epoch with correlation metrics
- Saves best model based on validation loss

### 4. Model Saving
- Saves final model as `.pt` file
- Attempts to save COMET-compatible checkpoint
- Exports training history as JSON

## 📊 Evaluation Metrics

The script tracks several metrics during training:

### Loss Metrics
- **Training Loss**: MSE loss on training data
- **Validation Loss**: MSE loss on validation data

### Correlation Metrics
- **Pearson Correlation**: Linear correlation between predicted and true scores
- **Spearman Correlation**: Rank correlation between predicted and true scores

### Example Output
```
Epoch 5/5 completed in 45.23s
Train Loss: 0.0234
Val Loss: 0.0189
Pearson Correlation: 0.8756
Spearman Correlation: 0.8901
```

## 📁 Output Structure

After training, you'll find these files in your model directory:

```
model_artifacts/
├── final_model.pt                    # Final trained model
├── best_model_epoch_N.pt            # Best model from training
├── final_model_checkpoint.ckpt      # COMET-compatible checkpoint
├── training_history.json            # Training metrics over time
└── best_model_epoch_N_checkpoint.ckpt
```

## 🔄 Model Loading for Inference

Load your trained model for inference:

```python
import torch
from comet import load_from_checkpoint

# Load the checkpoint
model = load_from_checkpoint("model_artifacts/final_model_checkpoint.ckpt")

# Or load the PyTorch state dict
checkpoint = torch.load("model_artifacts/final_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
data = [{"src": "Hello", "mt": "Hola", "ref": "Hola"}]
scores = model.predict(data)
```

## 🚀 SageMaker Deployment

The saved models are ready for SageMaker deployment. Use the existing infrastructure:

1. **Package Model**: Use the saved `.ckpt` files
2. **Deploy**: Use existing `deploy_sagemaker.py` script
3. **Inference**: The model follows the same interface as pre-trained COMET models

## 🛠️ Advanced Usage

### Custom Model Starting Point
Train from a specific checkpoint:
```bash
python train_comet_model.py \
    --data_path data.tsv \
    --model_name "./path/to/custom/checkpoint.ckpt"
```

### Multi-GPU Training
The script automatically detects CUDA availability:
```bash
# Force CPU usage
python train_comet_model.py --data_path data.tsv --device cpu

# Force CUDA usage
python train_comet_model.py --data_path data.tsv --device cuda
```

### Hyperparameter Tuning
Example training configurations:

```bash
# Conservative fine-tuning
python train_comet_model.py \
    --data_path data.tsv \
    --epochs 3 \
    --learning_rate 5e-6 \
    --batch_size 8

# Aggressive fine-tuning
python train_comet_model.py \
    --data_path data.tsv \
    --epochs 15 \
    --learning_rate 2e-5 \
    --batch_size 32
```

## 📋 Best Practices

### Data Preparation
1. **Quality Scores**: Ensure scores are normalized to [0, 1] range
2. **Data Balance**: Include examples across different quality levels
3. **Size**: Minimum 1000 samples recommended, 10k+ for best results
4. **Validation**: Always inspect your data for inconsistencies

### Training Tips
1. **Start Small**: Begin with fewer epochs to validate your pipeline
2. **Monitor Metrics**: Watch for overfitting (train loss << validation loss)
3. **Learning Rate**: Start with 1e-5, adjust based on convergence
4. **Batch Size**: Larger batches for stable gradients, smaller for memory constraints

### Model Selection
1. **Validation Loss**: Primary metric for model selection
2. **Correlation**: Higher Pearson/Spearman indicates better quality prediction
3. **Convergence**: Stop training when validation metrics plateau

## 🐛 Troubleshooting

### Common Issues

**Memory Errors**:
```bash
# Reduce batch size
python train_comet_model.py --data_path data.tsv --batch_size 4
```

**CUDA Errors**:
```bash
# Force CPU training
python train_comet_model.py --data_path data.tsv --device cpu
```

**Model Loading Errors**:
- Ensure COMET model name is correct
- Check internet connection for downloading
- Verify checkpoint file exists for local models

**Data Format Errors**:
- Verify CSV/TSV has required columns: source, mt, reference, score
- Check for missing values in critical columns
- Ensure score values are numeric and in [0, 1] range

## 📚 References

- [COMET Paper](https://arxiv.org/abs/2009.09025)
- [Unbabel COMET GitHub](https://github.com/Unbabel/COMET)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [AWS SageMaker PyTorch](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html)

## 📞 Support

For issues with:
- **Training Script**: Check logs and error messages
- **COMET Library**: Refer to official COMET documentation
- **SageMaker Deployment**: Use existing deployment scripts in this repository