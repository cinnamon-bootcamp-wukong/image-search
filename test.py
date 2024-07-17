
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List

class ImageEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load the CLIP model and processor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a list of PIL Images into feature vectors using CLIP.
        
        Parameters:
        - images: List of PIL Image objects.

        Returns:
        - A numpy array containing the encoded feature vectors.
        """
        # List to store image embeddings
        image_embeddings = []

        # Process images in batches
        with torch.no_grad():
            for image in images:
                # Preprocess the image
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                # Get the image features
                embeddings = self.model.get_image_features(pixel_values=pixel_values)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)  # Normalize embeddings
                image_embeddings.append(embeddings.cpu().numpy())

        # Convert list to numpy array
        return np.vstack(image_embeddings)

# Example usage
if __name__ == "__main__":
    # Load images from a directory or any other source
    image_paths = ["data/coco-128/test/000000000009_jpg.rf.6acc173402df5523069e146edb03ff4b.jpg", "data/coco-128/test/000000000025_jpg.rf.ed74f70d3b9ede1832934740b7ac60c7.jpg"]  # Update with your image paths
    images = [Image.open(path).convert("RGB") for path in image_paths]

    # Initialize the encoder and encode the images
    encoder = ImageEncoder()
    embeddings = encoder.encode_images(images)

    # Output the embeddings
    for idx, emb in enumerate(embeddings):
        print(f"Image {idx+1} Embedding: {emb}")
