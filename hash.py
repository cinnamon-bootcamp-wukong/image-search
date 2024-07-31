import hashlib
from PIL import Image
import random
import numpy as np

class Hashing:
    def __init__(self):
        pass

    def get_hash(self, img_path):
        img = np.array(Image.open(img_path))
        img_bytes = img.tobytes()
        hash_code = hashlib.sha256(img_bytes).hexdigest()
        return hash_code
    