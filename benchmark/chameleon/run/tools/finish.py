def finish(variable_values, answer_variable):
    if not isinstance(variable_values, dict):
        return "Error: variable_values must be of type dict."
    if answer_variable not in variable_values:
        return "Error: answer_variable must be a key inside variable_values."
    return variable_values[answer_variable]