import numpy as np
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

from nltk.corpus import stopwords
from scipy.odr import exponential

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import warnings

warnings.filterwarnings(action='ignore')

file_name = "Gutenburg.txt"
with open(file_name, "r", encoding="utf-8", errors="ignore") as file:
    content = file.read()
    cleaned_text = content.replace("\n", " ")

data = []

# we get rid of stop words because they provide no unique context
# and the model can focus  on more meaningful word relationships
for i in sent_tokenize(cleaned_text):
    temp_words = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        if j.lower().isalpha() and j.lower() not in stop_words:
            temp_words.append(j.lower())
    if len(temp_words) > 1:
        data.append(temp_words)

# removing "rare" words
word_counts = Counter(word for sentence in data for word in sentence)
vocab = [word for word, count in word_counts.items() if count >= 5]

# give ids to words and words to ids
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for i, word in enumerate(vocab)}

# clean the data, so that it removes words that were not given an id
new_filtered_data = [[word for word in sentence if word in word_to_id] for sentence in data]
new_filtered_data = [s for s in new_filtered_data if len(s) > 1]


window_size = 5
X = []
Y = []

# create pairs with 5 words from the left and 5 words from the right of a specific word
for sentence in new_filtered_data:
    word_ids = [word_to_id[word] for word in sentence]

    for i, target_id in enumerate(word_ids):

        start = max(0, i - window_size)
        end = min(len(word_ids), i + window_size + 1)

        for j in range(start, end):
            if i == j:
                continue

            context_id = word_ids[j]
            X.append(target_id)
            Y.append(context_id)



input_words = np.array(X, dtype=np.int32)
target_words = np.array(Y, dtype=np.int32)

input_w = []
output_w = []
# define the number of dimensions we want to use, also the number of
# columns in the input weight
embedding_size = 50

for i in range(len(vocab)):
    j = 0
    word_values_input = []
    word_values_output = []
    while j < embedding_size:
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)

        word_values_input.append(x)
        word_values_output.append(y)
        j += 1

    input_w.append(word_values_input)
    output_w.append(word_values_output)

# transform the lists into numpy arrays for efficiency
input_weights = np.array(input_w)
output_weights = np.array(output_w)

# we select the number of times we want to go through the loop
epochs = 20
# how much we shift the weights from the two matrices after predictions error
step_size = 0.01

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(input_words)):
        target_id = input_words[i]
        context_id = target_words[i]
        target_word_vector = input_weights[target_id]
        # doing the dot product of two rows from the two layers to get the similarity
        score = np.dot(output_weights, target_word_vector)

        # apply softmax function to get the probability of each word
        # appearing near the given input word
        exponential_score = np.exp(score - np.max(score))
        softmax = exponential_score / np.sum(exponential_score)

        #calculate the error that the model made
        error = softmax.copy()
        error[context_id] -= 1

        # backpropagation
        # calculate the gradients using the outer product function in numpy
        output_gradient = np.outer(error, target_word_vector)
        input_gradient = np.dot(output_weights.T, error)

        # update the weights
        output_weights -= step_size * output_gradient
        input_weights[target_id] -= step_size * input_gradient
        total_loss -= np.log(softmax[context_id])
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")


def get_neighbors(target_word, n=10):
    if target_word not in word_to_id:
        return f"'{target_word}' not in vocabulary."

    target_idx = word_to_id[target_word]
    v_a = input_weights[target_idx]

    similarities = []
    for i in range(len(vocab)):
        v_b = input_weights[i]
        norm_a = np.linalg.norm(v_a)
        norm_b = np.linalg.norm(v_b)
        # apply the cosine similarity formula
        score1 = np.dot(v_a, v_b) / (norm_a * norm_b)
        similarities.append((id_to_word[i], score1))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[1:n + 1]




