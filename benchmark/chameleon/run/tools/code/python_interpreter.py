import types

def execute(python_code):
    """
    Executes the provided Python code.
    """
    try: 
        exec(python_code)
        variable_values = {}
        for var_name, var_value in locals().items(): 
            if not var_name.startswith('__') and var_name not in ["python_code","variable_values"] and not isinstance(var_value, types.ModuleType):
                variable_values[var_name] = var_value
        return variable_values
    except Exception as e:
        if "'df'" in str(e):
           return "Error: "+str(e)+"\nUse pandas_interpreter instead." 
        return "Error: "+str(e)

if __name__ == "__main__":
    python_code = "import geopy\nimport geopy.distance\nlatitude = 40.05555\nlongitude = -75.090723\n_, lo_max, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=90)\n_, lo_min, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=270)\nla_max, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=0)\nla_min, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=180)\nresult = (la_max, la_min, lo_max, lo_min)" ###
    answer = execute(python_code)
    print(answer)
