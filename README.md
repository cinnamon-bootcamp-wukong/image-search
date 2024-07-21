# Image search using CLIP & FAISS
## Usage
```
git clone -b search_endpoint https://github.com/cinnamon-bootcamp-wukong/image-search
cd image-search
```
```
python embedding.py # Download embedded database
```
Make sure that your folder look like this

```shell
.
├── app
│   ├── img
│   │   └── logo.jpeg
│   └── ui.py
├── data
│   └── coco-128
│       ├── test
│       ├── train
│       └── valid
├── embedding.py
├── encode.py
├── image_embeddings.npy
├── image_encoding.py
├── image_paths.txt
├── README.md
├── search_algo.py
└── search_endpoint.py
```

```
python search_endpoint.py # Run API
streamlit run app/ui.py # Run UI
curl -X POST -F "file=@{your_image.jpg}" http://127.0.0.1:8000/search # Test Search
```
