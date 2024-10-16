import os
import google.generativeai as genai
import vertexai
import requests
from ..tools.tools_set import tools_gemini
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    ToolConfig
)
import time

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

tools_gemini_converted = []
for x in tools_gemini:
    if 'parameters' in x['function_declarations'][0].keys():
        tool = FunctionDeclaration(
            name=x['function_declarations'][0]['name'],
            description=x['function_declarations'][0]['description'],
            parameters=x['function_declarations'][0]['parameters']
        )
    else:
        tool = FunctionDeclaration(
            name=x['function_declarations'][0]['name'],
            description=x['function_declarations'][0]['description'],
           parameters = {'type': 'object',
            'properties': {'input_query': {'type': 'string',
              'description': 'The exact original user prompt input'}
              }
             }
      )
    tools_gemini_converted.append(tool)
tools_gemini_converted = [
    Tool(
        function_declarations=tools_gemini_converted
    ) 
]

def call_gemini_pro(model, messages, temperature, max_tokens, tool_choice):
    if tool_choice=="none":
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.NONE
            )
        )
    elif tool_choice=="required":
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY
            )
        )
    elif tool_choice=="auto":
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
            )
        )
        
    gemini_model = GenerativeModel(
    model,
    generation_config=GenerationConfig(temperature=temperature, maxOutputTokens=max_tokens),
    tools=tools_gemini_converted,
    tool_config=tool_config
)
    chat = gemini_model.start_chat(response_validation=False)
    response = chat.send_message(str(messages))
    content = response.candidates[0].content
    print("content", content)
    response_without_tools = {
        "role": content.role,
        "parts": content.parts
    }
    function_call = content.parts[0].function_call
    if tool_choice=="none" or (tool_choice=="auto" and not function_call):
        return response_without_tools, None
    else:
        response_with_tools = {
            "role": content.role,
            "parts": [{
                "functionCall": function_call
            }]
        }
        return response_with_tools, function_call

if __name__ == '__main__':
    call_gemini_pro(model, messages, temperature, max_tokens, tool_choice)
    