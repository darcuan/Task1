import numpy as np
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

from nltk.corpus import stopwords
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

input_weights = []
output_weights = []

embedding_size = 100

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

    input_weights.append(word_values_input)
    output_weights.append(word_values_output)


