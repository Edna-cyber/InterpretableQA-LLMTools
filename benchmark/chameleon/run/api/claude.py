import anthropic
import os 
import base64
from PIL import Image
import pdb
from time import sleep
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

CLAUDE_CKPT = "claude-3-opus-20240229"

def call_claude3(prompt, temperature, max_tokens, tools, tool_choice, system):
    if tool_choice == "none":
        response = anthropic_client.messages.create(
            model=CLAUDE_CKPT,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system
        )
    else:
        response = anthropic_client.messages.create(
            model=CLAUDE_CKPT,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            system=system
        )
    return response

if __name__ == '__main__':
    call_claude3(model, prompt, temperature, max_tokens, tools, tool_choice)