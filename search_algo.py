import faiss
import torch
import clip
import json
from PIL import Image
import numpy as np


class ImageSearchEngine:
    """
    A class to perform image search using CLIP model embeddings and FAISS for efficient similarity search.

    Attributes:
        images (list): A list of PIL.Image objects to be processed.
        top_k (int): The number of top similar images to retrieve for each query image.
    """

    def __init__(self, embeddings, image_paths_path, device="cpu"):
        """
        Initializes the ImageSearchEngine with images, embeddings, path to image paths JSON, and top_k.

        Args:

            embeddings (numpy.ndarray): Precomputed image embeddings for the images.
            image_paths_path (str): Path to a JSON file containing image paths corresponding to the embeddings.
            top_k (int): The number of top similar images to retrieve for each query image.
        """
        self.device = device
        self.embeddings = embeddings
        self.image_paths_path = image_paths_path

    def create_faiss_index(self):
        """
        Creates and returns a FAISS index for the provided embeddings and loads image paths from a JSON file.

        Returns:
            tuple: A tuple containing the FAISS index and a list of image paths.
        """

        d = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(self.embeddings)

        return index

    def get_embeddings(self):
      return self.embeddings

    def get_similar_images(self, encoded_images, top_k = 1):
        """
        Retrieves the top_k similar images for each image in the provided list of images.

        Args:
            encoded_images np.array: a np.array (output of encode.py).
            top_k (int): The number of top similar images to retrieve for each query image.
        Returns:
            list: A list of lists, where each inner list contains PIL.Image objects of similar images.
        """


        index = self.create_faiss_index()

        D, I = index.search(encoded_images.astype(np.float32), top_k)
        similar_images = [[self.image_paths_path[i] for i in indices] for indices in I]
        similar_images = np.array([np.array([np.array(Image.open(img_path)) for img_path in paths]) for paths in similar_images])

        return similar_images
