import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import os
import string
import re

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # Remove navigation, course, and other non-content elements
    text = re.sub(r'Click the right arrow to continue|Audio Available|3DS COM© .*?\d{4}', '', text)
    text = re.sub(r'[\[\]]', '', text)  # Remove brackets
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^A-Za-z0-9\s\.\,\-\(\)]', '', text)  # Keep alphanumerics and basic punctuation
    text = text.strip()

    # Specific cleanup for known repeated artifacts
    text = re.sub(r'DASSAULT.*?STEMES', '', text)
    text = re.sub(r'DEIMiA|ELMIA', 'DELMIA', text)  # Correct known text artifacts
    text = re.sub(r'\-+', '-', text)  # Reduce multiple hyphens to one
    text = re.sub(r'\.{2,}', '.', text)  # Reduce multiple periods to one

    return text

def preprocess_text(text):
    text = clean_text(text)  # Clean the text first

    # Break up the text into sentences
    sentences = sent_tokenize(text.lower())  # Convert to lowercase for uniformity

    # Clean each sentence
    processed_sentences = []
    seen_sentences = set()  # Track sentences to avoid duplicates
    for sentence in sentences:
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)

        # Tokenize the sentence
        tokens = word_tokenize(sentence)

        # Retain tokens that are alphabetic or punctuation
        tokens = [word for word in tokens if word.isalpha() or word in string.punctuation]

        # Filter out stopwords, but retain important words and punctuation
        tokens = [word for word in tokens if word.lower() not in stopwords.words('english') or word in string.punctuation]

        # Rejoin the tokens into a single string
        processed_sentence = ' '.join(tokens)
        processed_sentences.append(processed_sentence)

    # Rejoin sentences into the cleaned text
    return ' '.join(processed_sentences)

def load_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    return preprocess_text(text)

def save_preprocessed(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    raw_data_dir = '../data/raw/'
    processed_data_dir = '../data/processed/'

    os.makedirs(processed_data_dir, exist_ok=True)

    for file_name in os.listdir(raw_data_dir):
        raw_file_path = os.path.join(raw_data_dir, file_name)
        processed_text = load_and_preprocess(raw_file_path)
        processed_file_path = os.path.join(processed_data_dir, file_name)
        save_preprocessed(processed_text, processed_file_path)
