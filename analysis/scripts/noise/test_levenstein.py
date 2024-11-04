import random
import numpy as np

def tokenize(sentence):
    return sentence.split()

def detokenize(tokens):
    return ' '.join(tokens)

def word_levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return word_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def noise_sentence_word_level(sentence, target_distance, vocabulary):
    words = tokenize(sentence)
    current_distance = 0
    while current_distance < target_distance:
        operation = random.choice(['insert', 'delete', 'substitute'])
        if operation == 'insert' and len(words) > 0:
            pos = random.randint(0, len(words))
            word = random.choice(vocabulary)
            words.insert(pos, word)
        elif operation == 'delete' and len(words) > 1:
            pos = random.randint(0, len(words) - 1)
            del words[pos]
        elif operation == 'substitute' and len(words) > 0:
            pos = random.randint(0, len(words) - 1)
            word = random.choice(vocabulary)
            words[pos] = word
        current_distance = word_levenshtein_distance(tokenize(sentence), words)
    return detokenize(words)

# Example usage
original_sentence = "hello world from the algorithm"
target_distance = 3
# Define a simple vocabulary for demonstration purposes
vocabulary = ["hello", "world", "algorithm", "from", "the", "example", "simple", "test"]

noised_sentence = noise_sentence_word_level(original_sentence, target_distance, vocabulary)

print(f"Original: {original_sentence}")
print(f"Noised: {noised_sentence}")
