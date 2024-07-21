# Image search using CLIP & FAISS

This is a part of the assignment from Cinnamon AI bootcamp.

This program has 2 endpoints, one for encoding images using CLIP and one for searching for closest images (in the COCO 128 dataset) with the input image using CLIP and FAISS.

## Run the program
Clone this repository and switch to `search_endpoint` branch
```bash
git clone https://github.com/cinnamon-bootcamp-wukong/image-search
cd image-search
git checkout search_endpoint
```
The program has 2 parts:
- Encode endpoint: Lies on port 8000, employs batch processing with at most 5s wait time to process a batch with maximum of 16 images.
To run the endpoint:
```bash
fastapi run --host 0.0.0.0 encode.py
```

To use the encode endpoint:
```bash
curl -L -X POST -F "file=@/path/to/your/image.jpg" http://$your_ip:8000/encode/
```
This will return a NumPy array converted to a list containing the CLIP encoding of the given image. Note: ***DO*** remember the `-L` flag.

- Search endpoint: Lies on port 8500
To run the endpoint:
```bash
fastapi run --host 0.0.0.0 --port 8500 encode.py
```


```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://$your_ip:8500/search/
```
This will return the NumPy array-converted-to-list representation of the closest image to the input.
