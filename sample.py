import torch
from image_encoding import ImageEncoder
from PIL import Image

encoder = ImageEncoder()

im = [Image.open('Loss.png')]

print(encoder.encode_images(im))