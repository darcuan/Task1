import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
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
