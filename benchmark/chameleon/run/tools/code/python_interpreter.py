def execute(python_code):
    """
    Executes the provided Python code and updates the 'ans' in global_var.
    """
    global_var = {"ans": 0}
    try: 
        exec(python_code, global_var)
        return str(global_var['ans'])
    except Exception as e:
        return "Error: "+str(e)

if __name__ == "__main__":
    python_code = "import geopy\nimport geopy.distance\nlatitude = 40.05555\nlongitude = -75.090723\n_, lo_max, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=90)\n_, lo_min, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=270)\nla_max, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=0)\nla_min, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=180)\nans = (la_max, la_min, lo_max, lo_min)"
    answer = execute(python_code)
    print(answer)
