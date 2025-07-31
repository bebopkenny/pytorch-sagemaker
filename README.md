# PyTorch COMET Model for AWS SageMaker

This project demonstrates how to deploy a PyTorch-based COMET (Crosslingual Optimized Metric for Evaluation of Translation) model to AWS SageMaker. The COMET model evaluates machine translation quality by providing scores between 0 and 1.

## 📁 Project Structure

```
├── inference/
│   ├── code/
│   │   ├── inference.py        # SageMaker inference functions
│   │   └── requirements.txt    # Python dependencies
│   └── wmt20-comet-qe-da/     # Model artifacts (created by setup)
│       ├── checkpoints/
│       │   └── model.ckpt
│       └── hparams.yaml
├── setup_model.py             # Download and setup model artifacts
├── test_inference.py          # Local testing script
├── deploy_sagemaker.py        # SageMaker deployment script
├── check_endpoint_status.py   # Endpoint status monitoring
├── test_endpoint.py           # Endpoint testing script
├── model.tar.gz              # Packaged model for SageMaker (created by setup)
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Setup Environment

Make sure you have the following requirements:
- Python 3.10+
- AWS CLI configured with appropriate permissions
- SageMaker execution role (if running in SageMaker)

### 2. Install Dependencies and Setup Model

```bash
# Install basic dependencies
pip install boto3 sagemaker torch wget

# Run the setup script to download and prepare the model
python setup_model.py
```

### 3. Test Locally (Recommended)

Before deploying to SageMaker, test the model locally:

```bash
python test_inference.py
```

Expected output:
```
✓ Deserialized Data: {'src': 'This is a sample text to be translated.', 'mt': 'Dies ist ein zu übersetzender Beispieltext.'}
✓ Model loaded successfully
✓ Prediction: {'predictions': 0.5172236561775208}
✓ Serialized Data: {'predictions': 0.5172236561775208}
🎉 Local testing completed successfully!
```

### 4. Deploy to SageMaker

```bash
python deploy_sagemaker.py
```

### 5. Monitor Deployment

```bash
python check_endpoint_status.py
```

### 6. Test Deployed Endpoint

```bash
python test_endpoint.py
```

## 📋 Detailed Usage

### Setup Script (`setup_model.py`)

This script automates the entire setup process:

1. **Install Dependencies**: Installs required Python packages
2. **Download Model**: Downloads the COMET model from Unbabel's S3 bucket
3. **Extract Files**: Extracts the model artifacts
4. **Organize Structure**: Creates the proper directory structure for SageMaker
5. **Create Archive**: Packages everything into `model.tar.gz`

```bash
python setup_model.py
```

### Local Testing (`test_inference.py`)

Tests all SageMaker inference functions locally before deployment:

- `model_fn()`: Model loading
- `input_fn()`: Input deserialization
- `predict_fn()`: Prediction generation
- `output_fn()`: Output serialization

```bash
python test_inference.py
```

### SageMaker Deployment (`deploy_sagemaker.py`)

Handles the complete deployment process:

1. Sets up SageMaker session and roles
2. Uploads model artifacts to S3
3. Creates PyTorch model object
4. Deploys to specified instance type (default: `ml.p3.2xlarge`)

```bash
python deploy_sagemaker.py
```

**Note**: Deployment typically takes 5-10 minutes.

### Endpoint Monitoring (`check_endpoint_status.py`)

Monitor endpoint deployment status:

```bash
python check_endpoint_status.py
```

Features:
- Lists all available endpoints
- Shows real-time status updates
- Can wait for endpoint to be ready
- Monitors every 30 seconds

### Endpoint Testing (`test_endpoint.py`)

Test deployed endpoints with various options:

```bash
python test_endpoint.py
```

Test modes:
1. **Single Test**: Original example from the article
2. **Multiple Tests**: Three different translation pairs
3. **Custom Input**: Your own source and translation texts

## 🧠 Understanding COMET Scores

COMET provides translation quality scores between 0 and 1:
- **Higher scores** (closer to 1): Better translation quality
- **Lower scores** (closer to 0): Poorer translation quality

Example interpretations:
- `0.8-1.0`: Excellent translation
- `0.6-0.8`: Good translation
- `0.4-0.6`: Average translation
- `0.2-0.4`: Poor translation
- `0.0-0.2`: Very poor translation

## 🔧 Configuration Options

### Instance Types

The deployment script uses `ml.p3.2xlarge` by default. You can modify this in `deploy_sagemaker.py`:

```python
# Smaller instances (may be slower)
instance_type='ml.m5.large'
instance_type='ml.m5.xlarge'

# GPU instances (recommended)
instance_type='ml.p3.2xlarge'  # Default
instance_type='ml.p3.8xlarge'  # More powerful
```

### Model Versions

The project uses COMET version 2.2.2. To use different versions, modify:
- `setup_model.py`: Change the download URL
- `inference/code/requirements.txt`: Update the package version

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors During Local Testing**
   ```bash
   # Make sure all dependencies are installed
   pip install unbabel-comet==2.2.2 protobuf==3.20.*
   ```

2. **SageMaker Permission Errors**
   - Ensure your AWS credentials have SageMaker permissions
   - Check that execution role has access to S3 buckets

3. **Endpoint Deployment Fails**
   - Check CloudWatch logs for detailed error messages
   - Verify instance type is available in your region
   - Ensure model.tar.gz was created successfully

4. **Model Loading Errors**
   - Verify the model artifacts were downloaded correctly
   - Check that the directory structure matches requirements

### Getting Help

- Check AWS CloudWatch logs for detailed error messages
- Review SageMaker documentation for deployment issues
- Verify all file paths and permissions

## 📊 Expected Performance

- **Local Testing**: ~5-10 seconds for model loading, <1 second per prediction
- **Endpoint Deployment**: 5-10 minutes
- **Endpoint Inference**: 1-3 seconds per request
- **Model Size**: ~2.5GB

## 🔒 Security Notes

- Model artifacts are downloaded from Unbabel's official S3 bucket
- Ensure your AWS credentials are properly secured
- Consider using IAM roles with minimal required permissions
- Delete endpoints when not in use to avoid charges

## 💰 Cost Considerations

- **Instance Costs**: `ml.p3.2xlarge` costs ~$3.06/hour
- **Storage Costs**: S3 storage for model artifacts
- **Data Transfer**: Minimal for inference requests

Remember to delete endpoints when not needed:
```python
# In your AWS console or via boto3
predictor.delete_endpoint()
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 License

This project follows the same license terms as the COMET model from Unbabel.

---

**Happy coding! 🚀**