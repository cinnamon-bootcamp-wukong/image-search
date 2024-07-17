import os
import glob
from pathlib import Path
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        return image

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

def get_image_embeddings(data_dir, model_name="ViT-B/32", batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the CLIP model
    model, preprocess = clip.load(model_name, device=device)
    
    # Create a dataset and dataloader
    image_paths = get_data_paths(data_dir, data_formats=["jpg", "jpeg", "png"])
    print(f"Found {len(image_paths)} images.")
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # List to store image embeddings
    image_embeddings = []

    # Process images in batches
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            image_embeddings.append(embeddings.cpu().numpy())

    # Convert list to numpy array
    image_embeddings = np.vstack(image_embeddings)
    
    return image_embeddings, image_paths

data_dir = "data/coco-128/train"
image_embeddings, image_paths = get_image_embeddings(data_dir)