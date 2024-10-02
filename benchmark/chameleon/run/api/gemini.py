import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os 
from PIL import Image
import pdb
import requests
from time import sleep
import json

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
print("Gemini key", GOOGLE_API_KEY)
gemini_client = genai.configure(api_key=GOOGLE_API_KEY)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def call_gemini_pro(model, prompt, temperature, max_tokens, tools, tool_choice):
    response = gemini_client.generate_content(
        model=model, 
        prompt=prompt,
        temperature=temperature,
        max_tokens=args.policy_max_tokens, 
        tools=max_tokens,
        tool_choice=tool_choice 
    )
    print("response", response)
    choice = response.choices[0]  
    return choice

if __name__ == '__main__':
    call_gemini_pro(model, prompt, temperature, max_tokens, tools, tool_choice)
    