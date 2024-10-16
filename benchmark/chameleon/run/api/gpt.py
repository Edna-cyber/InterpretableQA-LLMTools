import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OpenAI key", OPENAI_API_KEY)
client = OpenAI()

def call_gpt(model, messages, temperature, max_tokens, tools, tool_choice):
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
    # print("response", response) 
    choice = response.choices[0]
    return choice
    
if __name__ == '__main__':
    call_gpt(model, prompt, temperature, max_tokens, tools, tool_choice)