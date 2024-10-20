import types
import pandas as pd
import numpy as np

def execute(python_code):
    """
    Executes the provided Python code.
    """    
    try: 
        python_code = python_code.replace("\\n", "\n")
        exec(python_code, globals())
        variable_values = {}
        combined = globals() | locals()
        for var_name, var_value in combined.items(): 
            excluded_types = (types.ModuleType, types.FunctionType, pd.DataFrame)
            if not var_name.startswith('__') and var_name not in ["python_code","variable_values"] and not isinstance(var_value, excluded_types):
                if isinstance(var_value, pd.Series):
                    variable_values[var_name] = var_value.head().to_dict()
                elif isinstance(var_value, (list, dict, np.ndarray)):
                    variable_values[var_name] = var_value[:10]
                else:
                    variable_values[var_name] = var_value
        if variable_values=={}:
            return "Error: the return value is empty. Ensure that the solution is assigned to a variable in the code."
        return variable_values
    except Exception as e:
        if "'df'" in str(e):
            return "Error: "+str(e)+"\nUse pandas_interpreter instead." 
        elif "ast.BinOp" in str(e):
            return "Error: "+str(e)+"\nEnsure any arithmetic operation is evaluated by a different tool before storing the result in 'variable_values'." 
        return "Error: "+str(e)

if __name__ == "__main__":
    python_code = """
import geopy
import geopy.distance
latitude = 40.05555
longitude = -75.090723
distance = geopy.distance.distance(kilometers=5)
_, lo_max, _ = distance.destination((latitude, longitude), bearing=90)
_, lo_min, _ = distance.destination((latitude, longitude), bearing=270)
la_max, _, _ = distance.destination((latitude, longitude), bearing=0)
la_min, _, _ = distance.destination((latitude, longitude), bearing=180)
result = (la_max, la_min, lo_max, lo_min)
"""
    answer = execute(python_code)
    print(answer)