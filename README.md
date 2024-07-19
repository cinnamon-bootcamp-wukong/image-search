# Image search using CLIP & FAISS
## Usage
```
git clone -b search_endpoint https://github.com/cinnamon-bootcamp-wukong/image-search
cd image-search
```
```
python embedding.py # Download embedded database
```
```
python search_endpoint.py # Run API
curl -X POST -F "file=@{your_image.jpg}" http://127.0.0.1:8000/search # Test Search
```
