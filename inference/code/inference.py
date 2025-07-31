# import libraries
import os
import logging
import time
import json

import comet
from comet import load_from_checkpoint
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
def model_fn(model_dir):
    logger.info("Loading model from directory: %s", model_dir)
    model_path = os.path.join(model_dir, 'wmt20-comet-qe-da', 'checkpoints', 'model.ckpt')
    model = load_from_checkpoint(model_path)
    logger.info("Model loaded successfully")
    return model

# Deserialize the request body
def input_fn(request_body, request_content_type):
    logger.info("Deserializing the input data")
    if request_content_type == "application/json":
        data_dict = json.loads(request_body)
        logger.info("Input data deserialized successfully")
        return data_dict
    else:
        message = f"Unsupported content type: {request_content_type}"
        logger.error(message)
        raise ValueError(message)

# Generate prediction
def predict_fn(input_data: Dict[str, str], model: Any):
    logger.info("Generating prediction")
    data = [{"src": input_data['src'], "mt": input_data['mt']}]
    start_time = time.time()
    model_output = model.predict(data)
    elapsed_time = time.time() - start_time

    # Print the model output for debugging
    logger.info(f"Model output[0]: {model_output[0]}")
    logger.info(f"Type of model output[0]: {type(model_output[0])}")

    return {"predictions": model_output[0][0]}

# Serialize the prediction result
def output_fn(prediction, content_type):
    logger.info("Serializing the prediction result")
    if content_type == "application/json":
        logger.info("Returning JSON string")
        return prediction
    else:
        message = f"Unsupported content type: {content_type}"
        logger.error(message)
        raise ValueError(message)