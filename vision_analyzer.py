from inference_sdk import InferenceHTTPClient
import streamlit as st
import cv2

from gpt_analyzer import *
import numpy as np

key = st.secrets["ROBOFLOW_API_KEY"]

class imageAnalyzer:
    CONF_THRESHOLD = 0.5

    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=key
    )

    @classmethod
    def drawBbox(cls, image, x1, y1, x2, y2, label, color):
        text_color = (255, 255, 255)  # White color for text
        cv2.rectangle(image, (x1, y1 + 10), (x2, y2), color, 2)
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] + 5), (x2, y1 + 10), color, -1)
        cv2.putText(image, label, (x1, y1 + 10), font, font_scale, text_color, thickness)
        return image

    @classmethod
    def interpret_data(cls, results):
        prompt = results
        system_message = "TEMPLATE PARA JSON ANALISI"
        return get_gpt_prompt_response(prompt, system_message)

    @classmethod
    def analyze_image(cls, image, cx, cy):
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        result_image = np.array(image)

        result = cls.CLIENT.infer(image, model_id="yolov8n-640")
        predictions = result.get('predictions', [])

        for prediction in predictions:
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']
            confidence = prediction['confidence']
            label = prediction['class']

            if confidence >= cls.CONF_THRESHOLD:
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)
                color = (0, 255, 0)  # Green color for bounding box

                result_image = cls.drawBbox(result_image, x1, y1, x2, y2, label, color)
        return result_image


