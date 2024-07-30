# Image search using CLIP & FAISS

This is a part of the assignment from Cinnamon AI bootcamp.

## Usage

### 1. Set up environment
```
git clone https://github.com/cinnamon-bootcamp-wukong/image-search
cd image-search
```

### 2. **Install Required Packages:**

- Install the required packages from `requirements.txt`:

```sh
pip install -r requirements.txt
```

- If you have already install `requirments.txt` (for wukong's members):
```
pip install locust
```
3. Script
- Run `fast_encode` endpoint:
```
python encode.py
```
- Start `locust`:
```
locust -f locus_file.py
```
- Take the given url in command line, such as `http://localhost:8089/`:
- Fill in the blank :
    * User : 100
    * Spawm : 1
    * Host(crucial) : your fastapi endpoint (`http://127.0.0.1:8000`)
