
import requests
import streamlit as st

open_ai_key = st.secrets['open_ai_key']


def get_gpt_prompt_response(prompt, system_message):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {open_ai_key}"
    }

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # Prepare the payload
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 350
    }

    # Make the API request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()


    output_message = response_data['choices'][0]['message']['content']

    return output_message