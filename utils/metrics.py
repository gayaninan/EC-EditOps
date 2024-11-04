from sklearn.metrics import precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import jiwer
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    # decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    wer_scores = []
    bleu_scores = []

    all_preds = []
    all_targets = []

    for prediction, target in zip(predictions, labels):
        wer_score = jiwer.wer(target, prediction)
        wer_scores.append(wer_score)

        reference = [word_tokenize(target)]
        hypothesis = word_tokenize(prediction)
        bleu_score = sentence_bleu(reference, hypothesis)
        bleu_scores.append(bleu_score)

        target_words = target.split()
        prediction_words = prediction.split()
        all_preds.extend(prediction_words)
        all_targets.extend(target_words)

    overall_wer = sum(wer_scores) / len(wer_scores)

    overall_bleu = sum(bleu_scores) / len(bleu_scores)

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='micro')

    return {
        'wer': overall_wer,
        'bleu': overall_bleu,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }



