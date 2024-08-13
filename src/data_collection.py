import fitz  # PyMuPDF
import os


def extract_text_from_pdf(pdf_path):
    text = ""
    # Open the PDF file using PyMuPDF
    document = fitz.open(pdf_path)

    # Extract text from each page
    for page_num in range(len(document)):
        page = document[page_num]
        text += page.get_text()

    return text


def save_data(text, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)


if __name__ == "__main__":
    pdf_path = '../data/brand.pdf'  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    save_data(extracted_text, '../data/raw/extracted_text.txt')
    print("Text extracted and saved successfully.")
