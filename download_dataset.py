import os
import requests
import zipfile

DATASET_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
ZIP_PATH = "plantvillage.zip"
EXTRACT_PATH = "dataset"

def download_data():
    if not os.path.exists(ZIP_PATH):
        print("Downloading dataset... This may take a while.")
        response = requests.get(DATASET_URL, stream=True)
        with open(ZIP_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

def extract_data():
    if not os.path.exists(EXTRACT_PATH):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("Extraction complete.")

if __name__ == "__main__":
    download_data()
    extract_data()
