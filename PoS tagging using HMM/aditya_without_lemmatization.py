 
import pandas as pd
from collections import defaultdict
import ast

# train_file = "/kaggle/input/nlp-project/TRAIN - TRAIN.csv.csv"
# test_file = "/kaggle/input/nlp-project/TEST - TEST.csv.csv"

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("validation_data.csv")

 
# import ast

# # Function to process each row
# def process_text(text):
#     try:
#         tokens = ast.literal_eval(text)  # Convert string representation of list to actual list
#         processed_tokens = [
#             (word if tag in ['PRON', 'PROPN', 'NOUN'] else word.lower(), tag)
#             for word, tag in tokens
#         ]
#         return str(processed_tokens)  # Convert back to string format
#     except Exception as e:
#         return text  # Return original if any error occurs

# # Identify the column name dynamically
# train_col = train_data.columns[0]
# test_col = test_data.columns[0]

# # Apply transformation
# train_data[train_col] = train_data[train_col].apply(process_text)
# test_data[test_col] = test_data[test_col].apply(process_text)

# # Display processed data
# train_data.head(), test_data.head()


 
import ast
import string

# Function to process each row
def process_text(text):
    try:
        tokens = ast.literal_eval(text)  # Convert string representation of list to actual list
        processed_tokens = [
            (word if tag in ['PRON', 'PROPN', 'NOUN'] else word.lower(), 'PUNCT' if word in string.punctuation else tag)
            for word, tag in tokens
        ]
        return str(processed_tokens)  # Convert back to string format
    except Exception as e:
        return text  # Return original if any error occurs

# Identify the column name dynamically
train_col = train_data.columns[0]
test_col = test_data.columns[0]

# Apply transformation
train_data[train_col] = train_data[train_col].apply(process_text)
test_data[test_col] = test_data[test_col].apply(process_text)

# Display processed data
train_data.head(), test_data.head()


 
# !pip install nltk


 
# import ast
# import string
# import nltk
# from nltk.stem import WordNetLemmatizer

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Function to convert words to lowercase (except PRON, PROPN, NOUN)
# def convert_to_lowercase(word, tag):
#     return word if tag in ['PRON', 'PROPN', 'NOUN'] else word.lower()

# # Function to replace punctuation tags
# def handle_punctuation(word, tag):
#     if all(char in string.punctuation for char in word):  
#         return word, 'PUNCT'  # Keep word unchanged, update tag
#     return word, tag  # Return as is if not punctuation

# # Function to apply lemmatization
# def apply_lemmatization(word):
#     return lemmatizer.lemmatize(word)

# # Main function to process each row
# def process_text(text):
#     try:
#         tokens = ast.literal_eval(text)  # Convert string representation of list to actual list
#         processed_tokens = []

#         for word, tag in tokens:
#             word, tag = handle_punctuation(word, tag)  # Handle punctuation
#             word = convert_to_lowercase(word, tag)  # Convert to lowercase if needed
#             word = apply_lemmatization(word)  # Apply lemmatization

#             processed_tokens.append((word, tag))

#         return str(processed_tokens)  # Convert back to string format
#     except Exception as e:
#         return text  # Return original if any error occurs

# # Identify the column name dynamically
# train_col = train_data.columns[0]
# test_col = test_data.columns[0]

# # Apply transformation
# train_data[train_col] = train_data[train_col].apply(process_text)
# test_data[test_col] = test_data[test_col].apply(process_text)

# # Display processed data
# train_data.head(), test_data.head()


 
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
        first_word_dict[(sentence[0][0], sentence[0][1])] += 1

    prev_tag = None
    for word, tag in sentence:
        tag_dict[tag] += 1
        word_tag_dict[(word, tag)] += 1

        if prev_tag is not None:
            two_tag_dict[(prev_tag, tag)] += 1
        prev_tag = tag

 
# print(sentence_count)

 
# Initial probability calculation
initial_probability = defaultdict(int)
for key, value in first_word_dict.items():
    initial_probability[key[1]] += value

for key, value in initial_probability.items():
    initial_probability[key] = value/sentence_count

 
# # Printing initial probability vector
# for key, value in initial_probability.items():
#     print(f"Key = {key}, Value = {value}")

 
# Emission probability calculation
emission_probability = defaultdict(lambda: defaultdict(int))

for key, value in word_tag_dict.items():
    emission_probability[key[1]][key[0]] = value/tag_dict[key[1]]

 
# Printing emission probability matrix
# for tag in emission_probability.keys():
#     for word, prob in emission_probability[tag].items():
#         print(f"Tag = {tag}, Word = {word}, Prob = {prob}")

 
# Transition probability calculation
transition_probability = defaultdict(lambda: defaultdict(int))

for key, value in two_tag_dict.items():
    transition_probability[key[0]][key[1]] = value/tag_dict[key[0]]

 
# Viterbi algorithm
import numpy as np

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
            validation_sentences.append(sentence)
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

 
from sklearn.metrics import accuracy_score

# Evaluate accuracy
accuracy = accuracy_score(actual_tags, predicted_tags)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get unique tags from tag_dict
unique_tags = list(tag_dict.keys())

cm = confusion_matrix(actual_tags, predicted_tags, labels=list(unique_tags))
plt.figure(figsize=(20, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=unique_tags,
    yticklabels=unique_tags,
    cmap="Blues",
)
plt.xlabel("Predicted Tag")
plt.ylabel("Actual Tag")
plt.title("Confusion Matrix")
plt.show()

 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get unique tags from tag_dict
unique_tags = list(tag_dict.keys())

# Compute confusion matrix
cm = confusion_matrix(actual_tags, predicted_tags, labels=unique_tags)

# Normalize by row (actual tag count)
cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(20, 8))
sns.heatmap(
    cm_percentage,
    annot=True,
    fmt=".2f",
    xticklabels=unique_tags,
    yticklabels=unique_tags,
    cmap="Blues",
    linewidths=0.5
)
plt.xlabel("Predicted Tag")
plt.ylabel("Actual Tag")
plt.title("Confusion Matrix (Percentage)")
plt.show()


