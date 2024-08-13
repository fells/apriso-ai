import fitz  # PyMuPDF
import re


def extract_main_content_from_pdf(pdf_path):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    main_content = []

    # Enhanced exclusion keywords and patterns
    exclusion_keywords = [
        "Click", "Hover", "button", "arrow", "continue", "Zoom", "Picture", "Download", "Cancel",
        "User Instructions", "Play", "Replay", "Navigate", "Glossary", "Menu", "Opens in new window",
        "U icons", "3DS", "Dassault Systèmes", "Confidential Information", "S  Dassault Systemes I",
        "Versailles Commercial Register", "Equity Promise", "corner to jump", "access key definitions",
        "Listen to John Eskuri", "Thank you for joining the", "Magic User Experience", "All rights reserved",
        "registered trademarks", "subsidiaries", "Use of", "S  Dassa ult Syst emes", "Confident ial Inf ormat ion",
        "Confidentia l Information J", "S  Dassault Syst emes Confident ial Inf ormat ion"
    ]
    exclusion_patterns = [
        r'http[s]?://\S+',  # URLs
        r'\b[A-Z]{2,}\b',  # Acronyms
        r'[^\w\s]',  # Special characters
        r'\d{2,}',  # Numbers (years, times, etc.)
        r'Confidential\s+Information',  # Confidential Information
        r'S\s+Dassault\s+Systemes\s+Confidential\s+Information',  # Company Info
        r'\bProducers\s+of\s+Goods\s+and\s+SeMces\b',  # Producers of Goods and Services
        r'Our\s+focus\s+From\s+documentation\s+to\s+Experience',  # Boilerplate phrases
        r'\bConfident\s+ial\s+Information\b',  # Truncated Phrases
        r'\bRephcable\s+and\s+Industry\s+Relevant\s+Apps\b',  # Truncated/Incorrect phrases
    ]

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text("text")

        # Split text into lines for more granular filtering
        lines = text.splitlines()

        # Filter lines that contain any of the exclusion keywords
        filtered_lines = [
            line for line in lines
            if not any(keyword.lower() in line.lower() for keyword in exclusion_keywords)
        ]

        # Further clean and filter the text using regex patterns
        cleaned_lines = []
        for line in filtered_lines:
            for pattern in exclusion_patterns:
                line = re.sub(pattern, '', line)
            # Discard lines that are too short, fragmented, or contain significant irregular spacing
            if len(line.strip()) > 25 and not re.search(r'\s{2,}', line):
                cleaned_lines.append(line.strip())

        # Remove redundant lines (i.e., lines that appear multiple times)
        unique_lines = []
        seen_lines = set()
        for line in cleaned_lines:
            # Check if line is a near-duplicate or paraphrase
            if not any(existing_line in line or line in existing_line for existing_line in seen_lines):
                unique_lines.append(line)
                seen_lines.add(line)

        main_content.extend(unique_lines)

    pdf_document.close()

    return main_content


# Save the extracted content to a text file or print it out for verification
def save_extracted_content(content, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in content:
            file.write(line + "\n")

# Example usage
pdf_path = '../data/brand.pdf'  # Replace with your PDF file path
output_file = '../data/raw/extracted_text.txt'
extracted_content = extract_main_content_from_pdf(pdf_path)
save_extracted_content(extracted_content, output_file)
print("Text extracted and saved successfully.")