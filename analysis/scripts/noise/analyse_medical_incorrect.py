import pandas as pd
from collections import Counter

def validate_corrections(df):
    # Function to calculate the differences between two sentences
    def calculate_diff(ref, trans):
        ref_words = ref.split()
        trans_words = trans.split()
        # Count the words in both sentences
        ref_count = Counter(ref_words)
        trans_count = Counter(trans_words)
        # Find words that are in trans but not in ref or the count is different
        diff = trans_count - ref_count
        return set(diff.elements())
    
    # Function to evaluate how many incorrect words were corrected
    def evaluate_correction(ref, trans, corrected):
        incorrect_words = calculate_diff(ref, trans)
        corrected_words = calculate_diff(ref, corrected)
        # Words that were incorrect but are not present in the corrected version
        corrected_errors = incorrect_words - corrected_words
        return len(corrected_errors), len(incorrect_words)
    
    # Apply the evaluation for each row
    df['correction_evaluation'] = df.apply(lambda x: evaluate_correction(x['refs'], x['trans'], x['model_corrected']), axis=1)
    return df

# Example usage
data = {
    'refs': ["This is the correct sentence.", "The quick brown fox jumps over the lazy dog."],
    'trans': ["This is the corect sentence.", "The quick brown fox jump over the lazy dog."],
    'model_corrected': ["This is the correct sentence.", "The quick brown fox jumps over the lazy dog."]
}
df = pd.DataFrame(data)

# Validate corrections
df = validate_corrections(df)
print(df)
