import ast

def finish(variable_values, answer_variable, answer_type):
    try:
        variable_values = ast.literal_eval(variable_values)
        if not isinstance(variable_values, dict):
            return "Error: variable_values must be a string that evaluates to a dictionary."
        if answer_variable not in variable_values:
            return "Error: answer_variable must be a key inside variable_values."
        type_map = {"list": list, "float": float, "integer": int, "string": str}
        if not isinstance(variable_values[answer_variable], type_map[answer_type]):
            return "Error: The final answer should be of type {} not {}".format(answer_type, type(variable_values[answer_variable]))
        return variable_values[answer_variable]
    except Exception as e:
        return "Error: "+str(e)
        

if __name__ == "__main__":
    arguments = {"variable_values":"{\'approval_rates\':   cpc_category  decision\\n0          H01       0.0, \'top_categories\': [\'H01\']}","answer_variable":"top_categories","answer_type":"list"}
    print(finish(**arguments))