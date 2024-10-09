from inference_sdk import InferenceHTTPClient
from gpt_analyzer import *
from vision_analyzer import *
import os
import json
import cv2
from datetime import datetime

import streamlit as st
from PIL import Image


def main():
    st.title("Image Input Streamlit App")

    user_text = st.text_input("Enter some text:")

    if user_text:

        res = get_gpt_prompt_response(user_text, "You are a robot designed to respond to user input in a funny way.")
        st.write(res)

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)

        imgres = imageAnalyzer.analyze_image(image, 0, 0)

        # Display the image
        st.image(imgres, caption="Uploaded Image", use_column_width=True)

        st.write("Image Uploaded Successfully!")


if __name__ == "__main__":
    main()


