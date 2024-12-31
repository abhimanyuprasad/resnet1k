import os
import requests
import zipfile
from tqdm import tqdm

def download_tiny_imagenet(root_dir):
    """Download Tiny-ImageNet dataset"""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, "tiny-imagenet-200.zip")
    
    # Download
    print("Downloading Tiny-ImageNet...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    # Extract
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root_dir)
    
    # Clean up
    os.remove(zip_path)
    print("Download complete!")

if __name__ == "__main__":
    download_tiny_imagenet("./tiny-imagenet") 