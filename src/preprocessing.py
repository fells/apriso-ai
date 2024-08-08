import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    return ' '.join(tokens)

def load_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return preprocess_text(text)

def save_preprocessed(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    raw_data_dir = 'data/raw/'
    processed_data_dir = 'data/processed/'

    os.makedirs(processed_data_dir, exist_ok=True)

    for file_name in os.listdir(raw_data_dir):
        raw_file_path = os.path.join(raw_data_dir, file_name)
        processed_text = load_and_preprocess(raw_file_path)
        processed_file_path = os.path.join(processed_data_dir, file_name)
        save_preprocessed(processed_text, processed_file_path)
