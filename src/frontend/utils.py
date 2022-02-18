import time
import os 

import pandas as pd
import logging
import psutil

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

import torch

from src.utility import Path
from src.utility import get_config, set_seed, get_device
from src.trainer import T5Trainer
from src.generator import T5Generator

def predict(sent, tokenizer, model, config_path):
    configs = get_config(config_path)
    generator = T5Generator(tokenizer, model, configs)
    res = generator.generate([sent], fix=True)
    data = res[0]
    return data


# TODO: add utility function for inference multiple item
def predict_bulk(_):
    time.sleep(5)

def loader(prefix, config_path):
    configs = get_config(config_path)
    set_seed(configs["main"]["seed"])
    tokenizer = None
    model_name = configs.get("main").get("pretrained")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    saved_path = os.path.join(Path.MODEL, prefix)
    device = get_device()
    model_path = os.path.join(saved_path, "model-best.pt")
    checkpoint = torch.load(model_path, map_location=device)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    return tokenizer, model

def print_memory_usage():
    logging.info(f"RAM memory % used: {psutil.virtual_memory()[2]}")

def visualize_triplet_opinion(predictions):
    aspects, sentiments, polarity = [], [], []
    for pred in predictions:
        aspects.append(pred[0])
        sentiments.append(pred[1])
        polarity.append(pred[2])

    return pd.DataFrame({
        "Aspect (Term)" : aspects,
        "Sentiment (Term)" : sentiments, 
        "Polarity" : polarity,
        "Original (output)" : predictions
    })