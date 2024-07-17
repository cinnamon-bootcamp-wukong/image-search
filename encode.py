from fastapi import FastAPI, UploadFile
from typing import List
from pydantic import BaseModel
from threading import Timer, Lock, Event
import time
import asyncio
from PIL import Image
import pickle, base64

app = FastAPI()

class Item(BaseModel):
    data: str

class BatchResponse(BaseModel):
    index: int
    b64: str

batch = []
batch_lock = Lock()
batch_event = Event()
batch_limit = 16
batch_timeout = 5  # seconds

responses = []

def process_batch():
    global batch, responses
    with batch_lock:
        if batch:
            print(f"Processing batch of {len(batch)} items")
            # Simulate batch processing and create a response
            results = [BatchResponse(index=i, result=f"Processed: {item.data}") for i, item in enumerate(batch)]
            for i, result in enumerate(results):
                responses[i].set_result(result)
            batch.clear()
            responses.clear()
            batch_event.set()  # Notify all waiting requests
        else:
            print("No items to process")
        batch_event.clear()

def batch_timer():
    process_batch()
    Timer(batch_timeout, batch_timer).start()

Timer(batch_timeout, batch_timer).start()

@app.post("/encode", response_model=BatchResponse)
async def add_to_batch(file: UploadFile):
    global batch, responses
    response_event = asyncio.Future()
    with batch_lock:
        im = Image.open(file)
        batch.append(im)
        responses.append(response_event)
        if len(batch) >= batch_limit:
            process_batch()
    await response_event
    return response_event.result()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)