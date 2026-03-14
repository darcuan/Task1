import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import warnings

warnings.filterwarnings(action='ignore')

# process the file
file_name = "Gutenburg.txt"
with open(file_name, "r", encoding="utf-8", errors="ignore") as file:
    content = file.read()
    cleaned_text = content.replace("\n", " ")

data = []
# loop through each sentence
for i in sent_tokenize(cleaned_text):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        if j.lower().isalpha() and j.lower() not in stop_words:
            temp.append(j.lower())
    if len(temp) > 1:
        data.append(temp)

# removing "rare" words
word_counts = Counter(word for sentence in data for word in sentence)
vocab = [word for word, count in word_counts.items() if count >= 5]

# dive ids to words and words to ids
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for i, word in enumerate(vocab)}

# clean the data
new_filtered_data = [[word for word in sentence if word in word_to_id] for sentence in data]
new_filtered_data = [s for s in new_filtered_data if len(s) > 1]
