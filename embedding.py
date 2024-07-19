import os
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import roboflow

roboflow.login(force=True)

roboflow.download_dataset(dataset_url="https://universe.roboflow.com/team-roboflow/coco-128/dataset/2", model_format="coco", location="data/coco-128")

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return image, image_path  # Return image and image path

def get_data_paths(dir: str | list[str], data_formats: list, prefix: str = '') -> list[str]:
    try:
        f = []  # data files
        for d in dir if isinstance(dir, list) else [dir]:
            p = Path(d)
            if p.is_dir():
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')
        data_files = sorted(x for x in f if x.split('.')[-1].lower() in data_formats)
        return data_files
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {dir}: {e}') from e

def get_image_embeddings(data_dir, model_name="openai/clip-vit-base-patch32", batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the CLIP model from Hugging Face
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Create a dataset and dataloader
    image_paths = get_data_paths(data_dir, data_formats=["jpg", "jpeg", "png"])
    print(f"Found {len(image_paths)} images.")
    dataset = ImageDataset(image_paths, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # Lists to store image embeddings and paths
    image_embeddings = []
    paths = []

    # Process images in batches
    with torch.no_grad():
        for images, image_paths in dataloader:
            images = images.to(device)
            embeddings = model.get_image_features(pixel_values=images)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            image_embeddings.append(embeddings.cpu().numpy())
            paths.extend(image_paths)  # Collect image paths

    # Convert lists to numpy arrays
    image_embeddings = np.vstack(image_embeddings)
    
    return image_embeddings, paths  # Return both embeddings and paths

def save_embeddings_and_paths(image_embeddings, image_paths, embeddings_file='image_embeddings.npy', paths_file='image_paths.txt'):
    # Save image embeddings to a .npy file
    np.save(embeddings_file, image_embeddings)
    print(f"Image embeddings saved to '{embeddings_file}'.")

    # Save image paths to a text file
    with open(paths_file, 'w') as f:
        for path in image_paths:
            f.write("%s\n" % path)
    print(f"Image paths saved to '{paths_file}'.")

# Use example
data_dir = "data/coco-128/train"
image_embeddings, image_paths = get_image_embeddings(data_dir)

# Save embeddings and paths using the function
save_embeddings_and_paths(image_embeddings, image_paths)
