import faiss
import torch
import clip
import json
from PIL import Image

class ImageSearchEngine:
    """
    A class to perform image search using CLIP model embeddings and FAISS for efficient similarity search.

    Attributes:
        images (list): A list of PIL.Image objects to be processed.
        embeddings (numpy.ndarray): Precomputed image embeddings for the images.
        image_paths_path (str): Path to a JSON file containing image paths corresponding to the embeddings.
        top_k (int): The number of top similar images to retrieve for each query image.
    """
    
    def __init__(self, images, embeddings, image_paths_path, top_k):
        """
        Initializes the ImageSearchEngine with images, embeddings, path to image paths JSON, and top_k.

        Args:
            images (list): A list of PIL.Image objects to be processed.
            embeddings (numpy.ndarray): Precomputed image embeddings for the images.
            image_paths_path (str): Path to a JSON file containing image paths corresponding to the embeddings.
            top_k (int): The number of top similar images to retrieve for each query image.
        """

        self.images = images
        self.embeddings = embeddings
        self.image_paths_path = image_paths_path
        self.top_k = top_k

    def __create_faiss_index(self):
        """
        Creates and returns a FAISS index for the provided embeddings and loads image paths from a JSON file.

        Returns:
            tuple: A tuple containing the FAISS index and a list of image paths.
        """

        d = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(self.embeddings)

        with open(self.image_paths_path, "r") as f:
            image_paths = json.load(f)
        return index, image_paths

    def get_similar_images(self, model_name="ViT-B/32", device="cpu"):
        """
        Retrieves the top_k similar images for each image in the provided list of images.

        Args:
            model_name (str, optional): The name of the CLIP model to use. Defaults to "ViT-B/32".
            device (str, optional): The device to run the CLIP model on. Defaults to "cpu".

        Returns:
            list: A list of lists, where each inner list contains PIL.Image objects of similar images.
        """
        
        clip_model, clip_preprocess = clip.load(model_name, device=device)
        processed_images = torch.stack([clip_preprocess(img) for img in self.images]).to(device)
        index, image_paths = self.__create_faiss_index()

        with torch.no_grad():
            embeddings = clip_model.encode_image(processed_images)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu().numpy()

        D, I = index.search(embeddings, self.top_k)
        similar_images = [[image_paths[i] for i in indices] for indices in I]
        similar_images = [[Image.open(img_path) for img_path in paths] for paths in similar_images]

        return similar_images
