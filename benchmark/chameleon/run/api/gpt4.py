from openai import OpenAI
import base64
import requests
import os 
import glob 
import numpy as np 
import pdb
import pandas as pd
import json 
from tqdm import tqdm
from time import sleep
client = OpenAI()
# OpenAI API Key
api_key = os.environ['OPENAI_API_KEY']

GPT_4_TURBO_CKPT = 'gpt-4-0125-preview'
GPT_3_5_TURBO_CKPT = 'gpt-3.5-turbo-0125'
GPT_4o = 'gpt-4o'
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4_vision(text_query, image_paths, temperature=0.0, max_tokens=1024, model = "gpt-4-turbo"):
    # Encode all images in the list
    images_base64 = [encode_image(path) for path in image_paths]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Create the content with text and all images
    content = [{"type": "text", "text": text_query}]
    for img_base64 in images_base64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    success = False
    while not success:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            success = True
        except Exception as e:
            print(e)
            time.sleep(10)  # Replaced `sleep` with `time.sleep`

    print("response from api", response)
    # Extract the response content or return an empty string if there's an error
    try:
        response_content = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        response_content = ""

    return response_content

def call_gpt4(text_query, system_content="You are a helpful mathematician in solving graph problems.", temperature=0.0, max_tokens=1024, tools=None, tool_choice="auto"):
    success = False
    while not success:
        try:
            response = client.chat.completions.create(
                model=GPT_4_TURBO_CKPT, #gpt-4 turbo
                # response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text_query },
                ],
                temperature = temperature,
                max_tokens = max_tokens,
                tools = tools,
                tool_choice = tool_choice
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls

    except Exception as e:
        print(e)
        content = ""
        tool_calls = ""

    return content, tool_calls 

def call_gpt4o(text_query, system_content="You are a helpful mathematician in solving graph problems.", temperature=0.0, max_tokens=1024, tools=None, tool_choice="auto"):
    success = False
    while not success:
        try:
            response = client.chat.completions.create(
                model=GPT_4o,
                # response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text_query },
                ],
                temperature = temperature,
                max_tokens = max_tokens,
                tools = tools,
                tool_choice = tool_choice
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
    except Exception as e:
        print(e)
        content = ""
        tool_calls = ""
    
    return content, tool_calls 

def call_gpt3_5(text_query, system_content="You are a helpful mathematician in solving graph problems.", temperature=0.0, max_tokens=1024, tools=None, tool_choice="auto")
    success = False
    while not success:
        try:
            response = client.chat.completions.create(
                model=GPT_3_5_TURBO_CKPT,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text_query + "<json>"},
                ],
                temperature = temperature,
                max_tokens = max_tokens,
                tools = tools,
                tool_choice = tool_choice
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
                
    except Exception as e:
        print(e)
        content = ""
        tool_calls = ""
    return content, tool_calls 
