import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

def fetch_sitemap(sitemap_url):
    response = requests.get(sitemap_url)
    sitemap_content = response.text
    return sitemap_content

def parse_sitemap(sitemap_content):
    sitemap = ET.fromstring(sitemap_content)
    urls = [url.text for url in sitemap.findall('.//loc')]
    return urls

def scrape_url(url):
    page_response = requests.get(url)
    soup = BeautifulSoup(page_response.content, 'html.parser')
    return soup.get_text()

def save_data(text, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    sitemap_url = 'https://customgpt-streamlit.s3.amazonaws.com/customgpt-streamlit/6f851836-a1dc-420b-b47c-db0b808be882.xml'
    sitemap_content = fetch_sitemap(sitemap_url)
    urls = parse_sitemap(sitemap_content)

    for i, url in enumerate(urls):
        page_text = scrape_url(url)
        save_data(page_text, f"data/raw/page_{i}.txt")
