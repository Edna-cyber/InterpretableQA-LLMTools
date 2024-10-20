def llm_inferencer():
    try:
        return "Provide a direct answer to the question without using any extra tools. Format your response as a dictionary with the key 'ans' and place your answer inside the dictionary, like this: {'ans': your_answer}. Next, make sure to call the Finish tool, passing this dictionary as variable_values. This step is crucial."
    except Exception as e:
        return "Error: "+str(e)