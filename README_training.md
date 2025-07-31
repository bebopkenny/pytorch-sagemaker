# COMET Quality Estimation Training Script

This repository contains a comprehensive PyTorch training script for machine translation quality estimation using COMET v2.2.2. The model evaluates the quality of machine-translated output given the source sentence, MT output, and reference translation.

## Features

- **Quality Estimation**: Trains models to score translation quality (0-1 scale)
- **COMET Integration**: Uses COMET v2.2.2 for state-of-the-art quality estimation
- **Flexible Data Loading**: Supports CSV/TSV input with customizable separators
- **Comprehensive Metrics**: Pearson/Spearman correlation, MSE, MAE
- **SageMaker Ready**: Automatically packages models for AWS SageMaker deployment
- **Experiment Tracking**: Optional integration with Weights & Biases and Comet ML
- **Robust Training**: Gradient clipping, early stopping, and comprehensive logging

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements_training.txt
```

2. **For experiment tracking (optional):**
```bash
# For Weights & Biases
pip install wandb

# For Comet ML
pip install comet_ml
```

## Data Format

The training script expects CSV or TSV files with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `source` | Original source sentence | "The cat is sitting on the mat." |
| `mt` | Machine translation output | "El gato está sentado en la alfombra." |
| `reference` | Human reference translation | "El gato está sentado sobre la alfombra." |
| `score` | Quality score (0.0 to 1.0) | 0.85 |

### Sample Data Files

- `sample_train_data.tsv` - Example training data
- `sample_val_data.tsv` - Example validation data

## Usage

### Basic Training

```bash
python train_comet_qe.py \
    --train_data sample_train_data.tsv \
    --val_data sample_val_data.tsv \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-5
```

### Advanced Training with All Options

```bash
python train_comet_qe.py \
    --train_data data/train.tsv \
    --val_data data/val.tsv \
    --data_separator "\t" \
    --model_name "Unbabel/wmt22-comet-da" \
    --model_dir "./model_artifacts" \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --clip_grad_norm 1.0 \
    --device cuda \
    --log_level INFO \
    --log_file training.log \
    --use_wandb \
    --wandb_project "my-comet-qe" \
    --experiment_name "experiment-v1"
```

### CSV Data Format

For CSV files, use comma separator:

```bash
python train_comet_qe.py \
    --train_data data/train.csv \
    --val_data data/val.csv \
    --data_separator "," \
    --epochs 10
```

## Command Line Arguments

### Data Arguments
- `--train_data`: Path to training data (CSV/TSV) [Required]
- `--val_data`: Path to validation data (CSV/TSV) [Required]
- `--data_separator`: Column separator ("\t" for TSV, "," for CSV) [Default: "\t"]

### Model Arguments
- `--model_name`: COMET model name or checkpoint path [Default: "Unbabel/wmt22-comet-da"]
- `--model_dir`: Directory to save model artifacts [Default: "./model_artifacts"]

### Training Arguments
- `--epochs`: Number of training epochs [Default: 10]
- `--batch_size`: Training batch size [Default: 16]
- `--learning_rate`: Learning rate [Default: 1e-5]
- `--weight_decay`: Weight decay for optimizer [Default: 0.01]
- `--clip_grad_norm`: Gradient clipping norm [Default: None]

### Device Arguments
- `--device`: Device to use ("auto", "cuda", "cpu") [Default: "auto"]

### Logging Arguments
- `--log_level`: Logging level ("DEBUG", "INFO", "WARNING", "ERROR") [Default: "INFO"]
- `--log_file`: Log file path (optional)

### Experiment Tracking Arguments
- `--use_wandb`: Enable Weights & Biases tracking
- `--wandb_project`: W&B project name [Default: "comet-qe"]
- `--use_comet_ml`: Enable Comet ML tracking
- `--comet_project`: Comet ML project name [Default: "comet-qe"]
- `--experiment_name`: Experiment name for tracking

## Output

The script generates:

1. **Model Artifacts** (in `--model_dir`):
   - `model.pt`: Best model checkpoint
   - `model_epoch_X.pt`: Epoch-specific checkpoints
   - `training_config.json`: Training configuration and metrics

2. **SageMaker Package**:
   - `model.tar.gz`: Ready-to-deploy SageMaker model package

3. **Logs**:
   - Console output with training progress
   - Optional log file
   - Experiment tracking (if enabled)

## Metrics

The script tracks the following metrics:

- **Training Loss**: MSE loss during training
- **Validation Loss**: MSE loss on validation set
- **Pearson Correlation**: Linear correlation between predicted and true scores
- **Spearman Correlation**: Rank correlation between predicted and true scores
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Mean Squared Error (MSE)**: Average squared difference

## Model Architecture

The script uses pre-trained COMET models for quality estimation:

- **Default Model**: `Unbabel/wmt22-comet-da`
- **Input**: Source sentence, MT output, reference translation
- **Output**: Quality score (0.0 to 1.0)
- **Loss Function**: MSE Loss
- **Optimizer**: Adam with weight decay

## SageMaker Deployment

After training, the script automatically creates `model.tar.gz` containing:

```
model_artifacts/
├── model.pt              # Best model checkpoint
├── training_config.json  # Configuration and metrics
└── model_epoch_X.pt      # Epoch checkpoints
```

This package is ready for deployment to AWS SageMaker.

## Example Training Output

```
2024-01-15 10:30:15 - INFO - Starting COMET Quality Estimation training
2024-01-15 10:30:15 - INFO - Loading datasets...
2024-01-15 10:30:16 - INFO - Loaded 15 samples from sample_train_data.tsv
2024-01-15 10:30:16 - INFO - Loaded 5 samples from sample_val_data.tsv
2024-01-15 10:30:17 - INFO - Loading COMET model: Unbabel/wmt22-comet-da
2024-01-15 10:30:25 - INFO - Model loaded successfully on cuda

--- Epoch 1/10 ---
2024-01-15 10:30:26 - INFO - Batch 0/1, Loss: 0.0234
2024-01-15 10:30:27 - INFO - Training Loss: 0.0234
2024-01-15 10:30:28 - INFO - Validation Loss: 0.0198
2024-01-15 10:30:28 - INFO - Validation Metrics: {'pearson': 0.89, 'spearman': 0.87, 'mse': 0.0198, 'mae': 0.112}
2024-01-15 10:30:28 - INFO - Epoch Time: 2.15s
2024-01-15 10:30:28 - INFO - New best model saved with validation loss: 0.0198
```

## Notes

- **Pre-trained Models**: The script uses pre-trained COMET models, which are typically not fine-tuned directly
- **Evaluation Focus**: Primary use case is evaluation of existing quality estimation capabilities
- **Memory Requirements**: COMET models require significant GPU memory (8GB+ recommended)
- **Batch Size**: Adjust batch size based on available GPU memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Fails**: Check internet connection and model name
3. **Data Format Errors**: Verify CSV/TSV column names and format
4. **Import Errors**: Install missing dependencies from requirements.txt

### Performance Tips

1. **Use GPU**: Set `--device cuda` for faster training
2. **Optimize Batch Size**: Find the largest batch size that fits in memory
3. **Monitor Metrics**: Use experiment tracking to monitor training progress
4. **Data Quality**: Ensure quality scores are properly normalized (0-1 range)

## License

This script is designed for use with COMET v2.2.2 and follows the same licensing terms.