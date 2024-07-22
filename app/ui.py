import io
import json
import numpy as np

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

# interact with FastAPI endpoint
backend = "http://0.0.0.0:8500/search"

icon = Image.open("app/img/logo.jpeg")
st.set_page_config(
    page_title="Image-search",
    page_icon=icon,
)

with st.sidebar:
    st.image(icon)
    st.subheader("Image-search")
    st.caption("=== Wukong ===")

    st.subheader(":arrow_up: Upload image")
    input_image = st.file_uploader("Choose image")

def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=80
    )

    return r

st.header("Image search with CLIP + FAISS")


col1, col2 = st.columns(2)

if input_image is not None:
    with col1:
        st.subheader(":camera: Input")
        segments = process(input_image, backend)
        original_image = Image.open(input_image).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)

    if st.button(":arrows_counterclockwise: SEARCH"):
        data = json.loads(segments.content)
        np_arr = np.array(data["similar_images"][0][0])
        similar_image = Image.fromarray(np.uint8(np_arr)).convert('RGB')
        with col2:
            st.subheader(":mag: Output")
            col2.header("Similar image")
            col2.image(similar_image, use_column_width=True)
