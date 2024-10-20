import ast

def finish(variable_values, answer_variable, answer_type, choices="[]"):
    try:
        variable_values = variable_values.replace('\\"', "'").replace("\\'", "'")
        print("variable_values", variable_values)
        variable_values = ast.literal_eval(variable_values)
        choices = ast.literal_eval(choices)
        if not isinstance(variable_values, dict):
            return "Error: variable_values must be a string that evaluates to a dictionary."
        if answer_variable not in variable_values:
            return "Error: answer_variable must be a key inside variable_values."
        type_map = {"list": list, "float": float, "integer": int, "string": str}
        if not isinstance(variable_values[answer_variable], type_map[answer_type]):
            return "Error: the final answer should be of type {} not {}. Modify variable values accordingly and call finish tool again.".format(answer_type, type(variable_values[answer_variable]))
        if choices and variable_values[answer_variable] not in choices:
            return "Error: the final answer must be one of the elements in {}. Modify variable values accordingly and call finish tool again.".format(choices)
        return variable_values[answer_variable]
    except Exception as e:
        if "malformed node or string" in str(e):
            return "Error: "+str(e)+" variable_values needs to be a string representation of a dictionary"
        return "Error: "+str(e)
        

if __name__ == "__main__":
    arguments = {
    "variable_values": "{\"percentage_four_authors\": 764 / 3585}",
    "answer_variable": "top_categories",
    "answer_type": "list"
}
    # "{'approval_rates': 'cpc_category  decision\\n0          H01       0.0', 'top_categories': ['H01']}"
    print(finish(**arguments))
    
