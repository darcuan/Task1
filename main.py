import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt_tab')

import warnings

warnings.filterwarnings(action='ignore')

file_name = "Gutenburg.txt"

with open(file_name, "r", encoding="utf-8", errors="ignore") as file:
    content = file.read()
    cleaned_text = content.replace("\n", " ")
    print("File loaded")

print(sent_tokenize("Hello world. This is NLP."))