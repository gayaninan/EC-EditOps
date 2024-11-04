import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from wordcloud import WordCloud

def detailed_validate_corrections_ordered(df):
    # Identifying incorrect words by comparing 'trans' and 'refs', preserving order and repetition
    def find_incorrect_words(ref, trans):
        ref_words = ref.split()
        trans_words = trans.split()
        ref_word_count = Counter(ref_words)
        incorrect_words = []
        
        for word in trans_words:
            if word not in ref_word_count or ref_word_count[word] == 0:
                incorrect_words.append(word)
            else:
                ref_word_count[word] -= 1
        
        return incorrect_words
    
    # Checking if incorrect words have been corrected, considering order and repetition
    def check_corrections(incorrect_words, corrected_sentence):
        corrected_sentence_words = corrected_sentence.split()
        
        # Track corrected words by checking if they're not in the corrected words set or changed position
        corrected = []
        corrected_word_count = Counter(corrected_sentence_words)
        
        for word in incorrect_words:
            if word not in corrected_word_count or corrected_word_count[word] == 0:
                corrected.append(word)
            else:
                corrected_word_count[word] -= 1
        
        # The list 'corrected' contains words that were incorrect but are not present in the corrected version
        # in their original incorrect sequence or are less frequent, indicating correction
        return corrected, incorrect_words
    
    # Apply the detailed evaluation for each row and store results in new columns
    corrections = [
        check_corrections(find_incorrect_words(row['refs'], row['trans']), row['model_corrected'])
        for _, row in df.iterrows()
    ]
    
    df['corrected_words'], df['incorrect_words'] = zip(*corrections)
    
    return df

def plot_detailed_correction_evaluation(df):
    # Calculate the number of corrected words and the total number of incorrect words for each sentence
    num_corrected_words = df['corrected_words'].apply(len)
    num_incorrect_words = df['incorrect_words'].apply(len)
    
    index = range(len(df))  # Sentence indices
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(index, num_incorrect_words, width=0.4, label='Total Incorrect Words', align='center', alpha=0.6)
    plt.bar(index, num_corrected_words, width=0.4, label='Incorrect Words Corrected', align='edge', alpha=0.6)
    
    # Adding titles and labels
    plt.xlabel('Sentence Index')
    plt.ylabel('Number of Words')
    plt.title('Correction Effectiveness: Incorrect Words vs. Corrected Words')
    plt.xticks(index)
    plt.legend()
    
    # Showing the plot
    plt.tight_layout()
    plt.savefig("new_fib.png")

def evaluate_correction_success(df, success_threshold=0.5):
    """
    Evaluates correction success based on the percentage of incorrect words that were corrected.
    Adds a 'correction_success' column indicating whether the correction meets the success threshold.
    
    Parameters:
    - df: DataFrame with 'corrected_words' and 'incorrect_words' columns.
    - success_threshold: The minimum fraction of incorrect words that need to be corrected for the correction to be considered successful.
    """
    
    def success_rate(row):
        # Calculate the success rate as the fraction of incorrect words that were corrected
        if len(row['incorrect_words']) > 0:
            return len(row['corrected_words']) / len(row['incorrect_words'])
        else:
            return 1.0  # If there were no incorrect words, consider it as fully successful
    
    # Apply the success rate calculation to each row
    df['correction_success_rate'] = df.apply(success_rate, axis=1)
    
    # Determine if the correction was successful based on the threshold
    df['correction_successful'] = df['correction_success_rate'] >= success_threshold
    
    return df

def plot_corrected_words_wordcloud(df):
    # Combine all corrected words into a single list
    all_corrected_words = sum(df['corrected_words'], [])
    
    # Join the words into a single string, as WordCloud expects input as a string
    corrected_words_string = ' '.join(all_corrected_words)
    
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(corrected_words_string)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axis labels and ticks
    plt.title('Word Cloud of Corrected Words')
    plt.savefig('cloud.png')

def evaluate_medical_correction_success(df, medical_terms):
    """
    Evaluates correction success specifically for medical words.
    Adds columns indicating the count of corrected medical words and their success rate.
    """
    def find_incorrect_medical_words(incorrect_words):
        return [word for word in incorrect_words if word in medical_terms]
    
    def medical_success_rate(row):
        incorrect_medical_words = find_incorrect_medical_words(row['incorrect_words'])
        corrected_medical_words = find_incorrect_medical_words(row['corrected_words'])
        if len(incorrect_medical_words) > 0:
            return len(corrected_medical_words) / len(incorrect_medical_words)
        else:
            return 1.0  # If there were no incorrect medical words, consider it as fully successful
    
    # Apply the success rate calculation specifically for medical words
    df['medical_correction_success_rate'] = df.apply(medical_success_rate, axis=1)
    
    return df


# Example usage
data = {
    'refs': ["This is the correct sentence sentence.", "The quick brown fox jumps the lazy dog."],
    'trans': ["This is the corect sentce sentenc.", "The quick brown fox jump over the lay dog."],
    'model_corrected': ["This is the correct sentence sentence.", "The quick brown fox jump the lazy dog."]
}
# df = pd.DataFrame(data)

df = pd.read_csv('/Users/gayanin/github/RefinedLM/model_outputs/pubmed-abs-ins-04-kaggle.csv')

# Validate corrections
df = detailed_validate_corrections_ordered(df[:1000])
plot_detailed_correction_evaluation(df)
df = evaluate_correction_success(df)
plot_corrected_words_wordcloud(df)
average_success_rate = df['correction_success_rate'].mean()
print(f"Average Success Rate: {average_success_rate:.2f}")

# medical_terms = set(["diabetes", "hypertension", "aspirin", "metformin"])
# df = evaluate_medical_correction_success(df, medical_terms)
# plot_corrected_words_wordcloud(df)
# average_medical_success_rate = df['medical_correction_success_rate'].mean()
# print(f"Average Medical Success Rate: {average_medical_success_rate:.2f}")