import hashlib
from PIL import Image
import random
import numpy as np

class Hashing:
    def __init__(self):
        pass

    def get_hash(self, img):
        img = np.array(img)
        img_bytes = img.tobytes()
        hash_code = hashlib.sha256(img_bytes).hexdigest()
        return hash_code
    