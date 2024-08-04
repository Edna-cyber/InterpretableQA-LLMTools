import ast

def finish(variable_values, answer_variable):
    variable_values = ast.literal_eval(variable_values)
    if not isinstance(variable_values, dict):
        return "Error: variable_values must be a string that evaluates to a dictionary."
    if answer_variable not in variable_values:
        return "Error: answer_variable must be a key inside variable_values."
    return variable_values[answer_variable]

if __name__ == "__main__":
    arguments = {'variable_values': "{'typical_time_span': 101.14309776124641}", 'answer_variable': 'typical_time_span'}
    print(finish(**arguments))