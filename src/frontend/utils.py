import time

import pandas as pd
import logging
import psutil

# TODO: add utility function for inference single item
def predict(_):
    # Dummy Data
    time.sleep(5)
    data = [("kamarnya", "bersih", "positif"), ("pelayanan", "kurang ramah", "negatif")]
    return data


# TODO: add utility function for inference multiple item
def predict_bulk(_):
    time.sleep(5)


# TODO: add model loader...
def load_model(_):
    return []


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