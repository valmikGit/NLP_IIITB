import pandas as pd
from collections import defaultdict
import ast
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

train_file = "train_data.csv"
test_file = "validation_data.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Dictionary for tag vs count
tag_dict = defaultdict(int)

# Dictionary for {word, tag} vs count
word_tag_dict = defaultdict(int)

# Dictionary for {tag i, tag i + 1} vs count
two_tag_dict = defaultdict(int)

# Dictionary of first word vs count
first_word_dict = defaultdict(int)

# Total sentence count
sentence_count = 0

for index, row in train_data.iterrows():
    sentence = row.iloc[0]

    # Ensure the sentence is a list, otherwise try converting
    if isinstance(sentence, str):
        try:
            sentence = ast.literal_eval(sentence)
        except (ValueError, SyntaxError):
            print(f"Skipping row {index}: Invalid format -> {sentence}")
            continue  # Skip this row

    if not isinstance(sentence, list):
        print(f"Skipping row {index}: Not a list -> {sentence}")
        continue
    
    sentence_count += 1
    
    # Check if the length of the sentence is more than zero
    if len(sentence) > 0:
        first_word_dict[(lemmatizer.lemmatize(sentence[0][0]), sentence[0][1])] += 1

    prev_tag = None
    for word, tag in sentence:
        lemmatized_word = lemmatizer.lemmatize(word)  # Apply lemmatization
        tag_dict[tag] += 1
        word_tag_dict[(lemmatized_word, tag)] += 1

        if prev_tag is not None:
            two_tag_dict[(prev_tag, tag)] += 1
        prev_tag = tag

print(sentence_count)

# Initial probability calculation
initial_probability = defaultdict(int)
for key, value in first_word_dict.items():
    initial_probability[key[1]] += value

for key, value in initial_probability.items():
    initial_probability[key] = value/sentence_count

# Emission probability calculation
emission_probability = defaultdict(lambda: defaultdict(int))
for key, value in word_tag_dict.items():
    emission_probability[key[1]][key[0]] = value/tag_dict[key[1]]

# Transition probability calculation
transition_probability = defaultdict(lambda: defaultdict(int))
for key, value in two_tag_dict.items():
    transition_probability[key[0]][key[1]] = value/tag_dict[key[0]]

# Viterbi algorithm
def viterbi_algorithm(sentence:list, unique_tags:list, initial_prob:defaultdict, transition_prob:defaultdict, emission_prob:defaultdict):
    n = len(sentence)  # Number of words
    tags_list = list(unique_tags)  # Convert set to list for indexing

    # Initialize Viterbi and backpointer as a list of dictionaries
    viterbi = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Step 1: Initialization
    for tag in tags_list:
        viterbi[0][tag] = initial_prob.get(tag, 0) * emission_prob.get(tag, {}).get(sentence[0], 0)
        backpointer[0][tag] = None

    # Step 2: Recursion
    for t in range(1, n):  # Iterate over words
        for curr_tag in tags_list:
            max_prob, best_prev_tag = max(
                (viterbi[t - 1].get(prev_tag, 0) * transition_prob.get(prev_tag, {}).get(curr_tag, 0) * emission_prob.get(curr_tag, {}).get(sentence[t], 0), prev_tag)
                for prev_tag in tags_list
            )
            viterbi[t][curr_tag] = max_prob
            backpointer[t][curr_tag] = best_prev_tag

    # Step 3: Termination
    best_tags = []
    best_last_tag = max(tags_list, key=lambda tag: viterbi[-1].get(tag, 0))
    best_tags.append(best_last_tag)

    # Step 4: Backtracking
    for t in range(n - 1, 0, -1):
        best_last_tag = backpointer[t][best_last_tag]
        best_tags.insert(0, best_last_tag)

    return best_tags

# Prepare test sentences
predicted_tags = []
actual_tags = []

# Convert string representation of lists into actual lists
validation_sentences = []
for _, row in test_data.iterrows():
    try:
        sentence = ast.literal_eval(row.iloc[0])  # Convert string to list of (word, tag) tuples
        if isinstance(sentence, list) and len(sentence) > 0:
            validation_sentences.append([(lemmatizer.lemmatize(word), tag) for word, tag in sentence])
    except (SyntaxError, ValueError):
        continue  # Skip invalid rows

# Check if valid sentences exist
if not validation_sentences:
    print("No valid test sentences found. Please check your dataset.")
    exit()

# Extract list of tags
tags_list = list(tag_dict.keys())

# Run Viterbi on test sentences
for sentence in validation_sentences:
    words = [word for word, _ in sentence]
    
    if not words:  # Skip empty sentences
        continue
    
    actual_tags.extend([tag for _, tag in sentence])
    predicted_tags.extend(
        viterbi_algorithm(
            words,
            tags_list,
            initial_probability,
            transition_probability,
            emission_probability
        )
    )

# Evaluate accuracy
accuracy = accuracy_score(actual_tags, predicted_tags)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plot confusion matrix
unique_tags = list(tag_dict.keys())
cm = confusion_matrix(actual_tags, predicted_tags, labels=unique_tags)
plt.figure(figsize=(20, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=unique_tags, yticklabels=unique_tags, cmap="Blues")
plt.xlabel("Predicted Tag")
plt.ylabel("Actual Tag")
plt.title("Confusion Matrix")
plt.show()