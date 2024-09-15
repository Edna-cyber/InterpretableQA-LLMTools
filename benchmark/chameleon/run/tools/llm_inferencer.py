def llm_inferencer():
    try:
        return "Provide a direct answer to the question without using any extra tools. Format your response as a dictionary with the key 'ans' and place your answer inside the dictionary, like this: {'ans': your_answer}. Nest, call the Finish tool with this dictionary as variable_values."
    except Exception as e:
        return "Error: "+str(e)