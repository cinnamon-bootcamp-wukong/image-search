from fastapi import FastAPI, UploadFile
from typing import List
from pydantic import BaseModel
from threading import Timer, Lock, Event
from cache import CacheDatabase
import time
import io
import asyncio
from PIL import Image
import json
import numpy as np


from image_encoding import ImageEncoder

app = FastAPI()
encoder = ImageEncoder()


class Item(BaseModel):
    data: str

class BatchResponse(BaseModel):
    index: int
    result: str

encode_batch = []
encode_batch_lock = Lock()
encode_batch_event = Event()
batch_limit = 16
batch_timeout = 5  # seconds

encode_responses = []


def search(*args, **kwargs):
    pass


def encode_process_batch():
    print('encode_process_batch')
    global encode_batch, encode_responses
    with encode_batch_lock:
        if encode_batch:
            print(f"Processing batch of {len(encode_batch)} items")
            # Simulate batch processing and create a response
            encoding_results = encoder.encode_images(encode_batch).tolist()
            results = [BatchResponse(index=i, result=json.dumps(encoding_result)) for i, encoding_result in enumerate(encoding_results)]
            for i, result in enumerate(results):
                encode_responses[i].set_result(result)
            encode_batch.clear()
            encode_responses.clear()
            encode_batch_event.set()  # Notify all waiting requests
        else:
            print("No items to process")
        encode_batch_event.clear()

def encode_batch_timer():
    print('encode_batch_timer')
    encode_process_batch()
    Timer(batch_timeout, encode_batch_timer).start()

Timer(batch_timeout, encode_batch_timer).start()

@app.post("/encode", response_model=BatchResponse)
async def add_to_batch(file: UploadFile):
    print('add_to_batch')
    global encode_batch, encode_responses
    response_event = asyncio.Future()
    with encode_batch_lock:
        content = await file.read()
        im = Image.open(io.BytesIO(content))
        encode_batch.append(im)
        encode_responses.append(response_event)
        if len(encode_batch) >= batch_limit:
            encode_process_batch()
    await response_event
    return response_event.result()


@app.post("/fast_encode")
async def add_to_batch_fast(file: UploadFile):
    print('add_to_batch_fast')
    content = await file.read()
    im = Image.open(io.BytesIO(content))
    db = CacheDatabase()
    if db.find_by_img(im) is not None:
        encoding_results = db.find_by_img(im).tolist()
        return {"Find in cache" : encoding_results}
    else:
        encoding_results = encoder.encode_images([im]).tolist()
        db.execute(im, np.array(encoding_results))
    return {"result" : encoding_results}

@app.post("/embedding")
async def embedding_function(file : UploadFile):
    print('Embedding')
    content = await file.read()
    image_list = []
    im = Image.open(io.BytesIO(content))
    image_list.append(im)
    results = encoder.encode_images(image_list).tolist()
    print('Done Encoding')
    return results[0]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
