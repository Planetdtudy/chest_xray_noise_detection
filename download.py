import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL of the directory containing the images
base_url = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/"

# Send a GET request to fetch the raw HTML content
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links in the directory
links = soup.find_all('a')

# Filter out links that are not PNG files
png_links = [urljoin(base_url, link.get('href')) for link in links if link.get('href').endswith('.png')]

# Create a directory to save the images
save_folder = "Montgomery_CXR"
os.makedirs(save_folder, exist_ok=True)

# Download each PNG file
for url in png_links:
    filename = os.path.join(save_folder, os.path.basename(url))
    with requests.get(url, stream=True) as r:
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {url}")
