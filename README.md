# Image search using CLIP & FAISS

This is a part of the assignment from Cinnamon AI bootcamp.

This program has 2 endpoints, one for encoding images using CLIP and one for searching for closest images (in the COCO 128 dataset) with the input image using CLIP and FAISS.

***You can use our web application by visiting the link below:***
```bash
[http://35.163.120.104:8501/](http://35.163.120.104:8501/)
```
## Run the program
Install `conda` package manager via Anaconda or Miniconda if it is not installed.

Install `faiss`:
```bash
conda install -c conda-forge faiss-gpu
```
`faiss-gpu` is replaced with `faiss-cpu` if you don't have a NVIDIA GPU.

Clone this repository
```bash
git clone https://github.com/cinnamon-bootcamp-wukong/image-search
cd image-search
pip install -r requirements.txt
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
```bash
fastapi run --host 0.0.0.0 --port 8500 search_endpoint.py
```


```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://$your_ip:8500/search/
```
This will return the NumPy array-converted-to-list representation of the closest image to the input.

Or you can:
```bash
streamlit run app/ui.pi
```
This will goto the UI of webpage.
