import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_url(url, filepath):
    """
    Downloads a file from a given URL and saves it to the specified filepath.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Stream the download with a progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filepath, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(filepath)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)

    print(f"Downloaded {os.path.basename(filepath)} successfully!")


# Set the root directory and model URL
root_dir = "./models"  # Directory to save the model
resource = "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_vista3d.pt"

# Download the model if it does not already exist
model_path = os.path.join(root_dir, "model.pt")
if not os.path.exists(model_path):
    download_url(url=resource, filepath=model_path)

# Verify that the model exists
if os.path.exists(model_path):
    print("Model downloaded and saved successfully!")
else:
    print("Model download failed.")
