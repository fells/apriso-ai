import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import os

def fetch_sitemap(sitemap_url):
    response = requests.get(sitemap_url)
    if response.status_code == 200:
        sitemap_content = response.text
        return sitemap_content
    else:
        raise Exception(f"Failed to fetch sitemap. Status code: {response.status_code}")

def parse_sitemap(sitemap_content):
    sitemap = ET.fromstring(sitemap_content)
    namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    urls = [url.text for url in sitemap.findall('.//ns:loc', namespaces)]
    return urls

def scrape_url(url):
    try:
        page_response = requests.get(url)
        page_response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(page_response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"An error occurred while fetching {url}: {e}")
        return ""

def save_data(text, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    sitemap_url = 'https://customgpt-streamlit.s3.amazonaws.com/customgpt-streamlit/6f851836-a1dc-420b-b47c-db0b808be882.xml'
    sitemap_content = fetch_sitemap(sitemap_url)
    print("Sitemap fetched successfully.")
    urls = parse_sitemap(sitemap_content)
    print(f"Parsed {len(urls)} URLs.")

    for i, url in enumerate(urls):
        print(f"Scraping URL {i + 1}/{len(urls)}: {url}")
        page_text = scrape_url(url)
        save_data(page_text, f"../data/raw/page_{i}.txt")
        print(f"Saved data from {url} to ../data/raw/page_{i}.txt")

