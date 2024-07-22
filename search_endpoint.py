from search_algo import ImageSearchEngine
from image_encoding import ImageEncoder
from encode import embedding_function
import torch
import glob
from PIL import Image
from pathlib import Path
import numpy as np
import os
import io
import requests
import json

from fastapi import FastAPI, UploadFile, Request
app = FastAPI()

def get_data_paths(dir: str | list[str], data_formats: list, prefix: str = '') -> list[str]:
    """
    Get list of files in a folder that have a file extension in the data_formats.

    Args:
      dir (str | list[str]): Dir or list of dirs containing data.
      data_formats (list): List of file extensions. Ex: ['jpg', 'png']
      prefix (str): Prefix for logging messages.

    Returns:
      A list of strings.
    """
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device is : {device}')

image_paths = get_data_paths('data/coco-128/train', data_formats=["jpg", "jpeg", "png"])
embeddings = np.load('image_embeddings.npy')
search_engigne = ImageSearchEngine(embeddings, image_paths, device = device)

@app.post("/search")
async def similarity_search(file: UploadFile, request: Request):

    response = await embedding_function(file)
    encoded_image = np.array([response])
    print(encoded_image.shape)
    similar_images = search_engigne.get_similar_images(encoded_image, top_k = 1)
    return {"similar_images": similar_images.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8500)
