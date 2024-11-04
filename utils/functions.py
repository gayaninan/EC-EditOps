import json
from datasets import load_metric
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import *

def read_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def prepare_data(data):
    text = []
    for item in data:
        # input_sentence = f"Correct the following sentence: {item['incorrect']} Output: {item['corrected']}" #cloze prompt
        text.append(item)
    return text

def split_dataset(data):
    train_data = prepare_data(data["train"])
    val_data = prepare_data(data["validation"])
    test_data = prepare_data(data["test"])
    return train_data, val_data, test_data

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def read_local(csv_file):
    data = pd.read_csv(csv_file)
    train_data, temp_data = train_test_split(data, test_size=(1 - 0.8)) # 0.8, 0.1, 0.1
    test_data, validation_data = train_test_split(temp_data, test_size=0.1 / (0.1 + 0.1))
    return train_data.values.tolist(), validation_data.values.tolist(), test_data.values.tolist()

def load_config(config_file):
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)
    return Config(data)
