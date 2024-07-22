import io
import json
import numpy as np

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

"""
    This Streamlit application provides a user interface for performing image searches using a FastAPI backend.

    The application allows users to upload an image and receive a similar image as a result. The user interface includes:
        - An image uploader for users to select an image from their local system.
        - A button to trigger the search.
        - Display areas for the uploaded image and the search result.

    The code includes the following key components:
        - `process(image, server_url: str)`: A function that sends the uploaded image to a FastAPI endpoint for processing. It uses `MultipartEncoder` to handle image file uploads and `requests` to send the POST request to the server.
        - Streamlit configuration and layout settings:
        - The page title and icon are set.
        - A sidebar is used to display the application icon and provide upload functionality.
        - The main page is divided into two columns for displaying the original and similar images.

    Usage:
        1. Upload an image using the file uploader in the sidebar.
        2. Click the "SEARCH" button to send the image to the backend server and receive a similar image.
        3. View both the original uploaded image and the search result displayed on the page.
"""

# interact with FastAPI endpoint
backend = "http://127.0.0.1:8500/search"

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