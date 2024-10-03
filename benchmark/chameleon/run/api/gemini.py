import requests
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)




tools_gemini_converted = [
    FunctionDeclaration(
        name=x['function_declarations'][0]['name'],
        description=x['function_declarations'][0]['description'],
        parameters=x['function_declarations'][0]['parameters'] if 'parameters' in x['function_declarations'][0].keys() else None,
    )

    for x in tools_gemini
    
]