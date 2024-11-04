import pandas as pd
import json
from datasets import load_dataset

# Load medical dictionary
def load_medical_dict(file_path):
    with open(file_path, 'r') as file:
        medical_dict = json.load(file)
    medical_terms = set(term for terms in medical_dict.values() for term in terms)
    return medical_terms

def medical_wer(refs, hyp, medical_terms):
    # Tokenize and lowercase the text
    ref_words = refs.lower().split()
    hyp_words = hyp.lower().split()
    
    # Keep only medical terms
    ref_words_medical = [word for word in ref_words if word in medical_terms]
    hyp_words_medical = [word for word in hyp_words if word in medical_terms]

    # Initialize edit distance matrix
    dp = [[0] * (len(hyp_words_medical) + 1) for _ in range(len(ref_words_medical) + 1)]
    for i in range(len(ref_words_medical) + 1):
        for j in range(len(hyp_words_medical) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif ref_words_medical[i - 1] == hyp_words_medical[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # The last cell of dp contains the edit distance
    edit_distance = dp[-1][-1]
    
    # Calculate WER
    if len(ref_words_medical) == 0:
        return 0 if len(hyp_words_medical) == 0 else 1
    return edit_distance / len(ref_words_medical)

# Load the medical dictionary
medical_terms = load_medical_dict('/Users/gayanin/github/RefinedLM/analysis/scripts/medical_terms.json')


def get_medical_wer(dataset):
    # data = load_dataset(dataset)
    df = pd.read_csv(dataset)

    df['baseline_medical_WER'] = df.apply(lambda row: medical_wer(row['refs'], row['trans'], medical_terms), axis=1)
    df['model_medical_WER'] = df.apply(lambda row: medical_wer(row['refs'], row['model_corrected'], medical_terms), axis=1)

    # Calculate and print the average improvement
    baseline = df['baseline_medical_WER'].mean()
    models = df['model_medical_WER'].mean()

    print(f"Baseline Medical WER: {baseline:.2f}")
    print(f"Models Medical WER: {models:.2f}")

get_medical_wer('model_outputs/pubmed-abs-ins-03-gcd.csv')
