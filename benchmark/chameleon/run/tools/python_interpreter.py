import ast
import types
import pandas as pd
import numpy as np

def execute(python_code, input_var_values="{}"):
    """
    Executes the provided Python code.
    """    
    try: 
        input_var_values = ast.literal_eval(input_var_values)
        local_scope = {}
        exec(python_code, {**globals(), **input_var_values}, local_scope)
        variable_values = {}
        combined = {**globals(), **input_var_values, **local_scope}
        for var_name, var_value in combined.items(): 
            excluded_types = (types.ModuleType, types.FunctionType, pd.DataFrame)
            if not var_name.startswith('__') and var_name not in ["python_code","input_var_values", "variable_values"] and not isinstance(var_value, excluded_types) and var_name not in input_var_values:
                if isinstance(var_value, pd.Series):
                    variable_values[var_name] = var_value.head().to_dict()
                elif isinstance(var_value, (list, np.ndarray)):
                    variable_values[var_name] = var_value[:10]
                elif isinstance(var_value, dict):
                    variable_values[var_name] = dict(list(var_value.items())[:10])
                else:
                    variable_values[var_name] = var_value
        return variable_values
    except Exception as e:
        if "'df'" in str(e):
           return "Error: "+str(e)+"\nUse pandas_interpreter instead." 
        return "Error: "+str(e)

if __name__ == "__main__":
#     python_code = """
# import geopy
# import geopy.distance
# latitude = 40.05555
# longitude = -75.090723
# distance = geopy.distance.distance(kilometers=5)
# _, lo_max, _ = distance.destination((latitude, longitude), bearing=90)
# _, lo_min, _ = distance.destination((latitude, longitude), bearing=270)
# la_max, _, _ = distance.destination((latitude, longitude), bearing=0)
# la_min, _, _ = distance.destination((latitude, longitude), bearing=180)
# result = (la_max, la_min, lo_max, lo_min)
# """
#     answer = execute(python_code)
#     print(answer)
    python_code = "res = num_a/num_b"
    answer = execute(python_code, "{'num_a': 20, 'num_b': 25}")
    print(answer)