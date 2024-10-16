import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OpenAI key", OPENAI_API_KEY)
client = OpenAI()

def call_gpt(model, messages, temperature, max_tokens, tools, tool_choice):
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
    # print("response", response) 
    choice = response.choices[0]
    response_message = choice.message
    content = response_message.content
    response_without_tools = {
        "role": choice.message.role,
        "content": content
    }
    tool_calls = response_message.tool_calls
    if tool_choice=="none" or (tool_choice=="auto" and not tool_calls):
        return response_without_tools, None
    else:
        tool_call = tool_calls[0]
        response_with_tools = {
            "role": choice.message.role,
            "content": content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type, 
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
            ]
        }
        return response_with_tools, tool_call
    
if __name__ == '__main__':
    call_gpt(model, prompt, temperature, max_tokens, tools, tool_choice)