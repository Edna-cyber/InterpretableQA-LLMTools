import anthropic
import os 
import base64
from PIL import Image
import pdb
from time import sleep
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

CLAUDE_CKPT = "claude-3-opus-20240229"

def call_claude3(messages, temperature, max_tokens, tools, tool_choice):
    response = anthropic_client.completions.create(
        model=CLAUDE_CKPT, 
        prompt=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,  
        tool_choice=tool_choice
    )
    choice = response['completion']
    return choice

if __name__ == '__main__':
    call_claude3(model, messages, temperature, max_tokens, tools, tool_choice)