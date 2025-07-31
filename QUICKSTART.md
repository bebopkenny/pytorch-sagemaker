# Quick Start Guide - COMET Quality Estimation Training

## 🚀 Get Started in 5 Minutes

This guide will get you training a COMET quality estimation model quickly.

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU training)

### Step 1: Install Dependencies

```bash
# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements_training.txt
```

### Step 2: Prepare Your Data

Create TSV files with columns: `source`, `mt`, `reference`, `score`

**Option A: Use Sample Data (for testing)**
```bash
# Sample files are already provided:
# - sample_train_data.tsv
# - sample_val_data.tsv
```

**Option B: Create Your Own Data**
```bash
# Create train.tsv and val.tsv with this format:
# source	mt	reference	score
# Hello world	Hola mundo	Hola mundo	1.0
# Good morning	Buenos días	Buenos días	0.9
```

### Step 3: Run Training

**Basic Training (CPU)**
```bash
python train_comet_qe.py \
    --train_data sample_train_data.tsv \
    --val_data sample_val_data.tsv \
    --epochs 5 \
    --batch_size 8 \
    --device cpu
```

**GPU Training (Recommended)**
```bash
python train_comet_qe.py \
    --train_data sample_train_data.tsv \
    --val_data sample_val_data.tsv \
    --epochs 10 \
    --batch_size 16 \
    --device cuda
```

### Step 4: Monitor Progress

The script will output:
```
--- Epoch 1/10 ---
Training Loss: 0.0234
Validation Loss: 0.0198
Validation Metrics: {'pearson': 0.89, 'spearman': 0.87}
Epoch Time: 2.15s
```

### Step 5: Use Your Trained Model

After training, you'll have:
```
model_artifacts/
├── model.pt              # Best model
├── training_config.json  # Training info
└── model.tar.gz          # SageMaker package
```

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_comet_qe.py --batch_size 4 --device cuda
```

**2. Model Download Slow**
```bash
# Use a different model
python train_comet_qe.py --model_name "Unbabel/wmt20-comet-da"
```

**3. Import Errors**
```bash
# Install missing packages
pip install unbabel-comet==2.2.2 protobuf==3.20.*
```

### Performance Tips

1. **Start Small**: Use sample data first to verify everything works
2. **Check GPU**: Run `nvidia-smi` to check GPU availability
3. **Monitor Memory**: Use `--batch_size 4` if you see memory errors
4. **Use Experiment Tracking**: Add `--use_wandb` for better monitoring

## 📊 Expected Results

With the sample data, you should see:
- **Training Loss**: Decreasing from ~0.1 to ~0.01
- **Pearson Correlation**: > 0.8 (good quality estimation)
- **Training Time**: 1-5 minutes depending on hardware

## 🔧 Advanced Usage

### Custom Data Format
```bash
# For CSV files
python train_comet_qe.py \
    --train_data data.csv \
    --val_data val.csv \
    --data_separator ","
```

### Experiment Tracking
```bash
# With Weights & Biases
python train_comet_qe.py \
    --use_wandb \
    --wandb_project "my-qe-project" \
    --experiment_name "experiment-1"
```

### Production Training
```bash
python train_comet_qe.py \
    --train_data large_train.tsv \
    --val_data large_val.tsv \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --clip_grad_norm 1.0 \
    --log_file training.log
```

## 🧪 Test Your Setup

Run the test script to verify everything works:
```bash
python test_training.py
```

This will check:
- ✓ All dependencies are installed
- ✓ Sample data loads correctly
- ✓ Model saving works
- ✓ All functions are working

## 📝 Next Steps

1. **Prepare Real Data**: Replace sample data with your quality estimation dataset
2. **Tune Hyperparameters**: Experiment with learning rates and batch sizes
3. **Deploy to SageMaker**: Use the generated `model.tar.gz` file
4. **Monitor Performance**: Use the correlation metrics to assess quality

## 🆘 Need Help?

- Check the full documentation in `README_training.md`
- Review error messages carefully
- Ensure your data format matches the expected columns
- Start with CPU training if GPU issues occur

**Happy Training! 🎯**