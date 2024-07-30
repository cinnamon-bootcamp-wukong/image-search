from locust import HttpUser, task, between
from PIL import Image
import random

global image_path
image_path = ['sample_data/dataset.png', 'sample_data/result_page_1.jpg', 'sample_data/Loss.png']

class AppUswer(HttpUser):
    wait_time = between(2, 5)

    @task 
    def search_page(self):
        random_image = random.choice(image_path)
        f =  open(random_image, "rb")
        files = {'file': f}  
        
        self.client.post('/fast_encode', files=files)