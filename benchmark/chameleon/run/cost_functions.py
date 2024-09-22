import math

def calc_cost1(function_type, function_arguments):
    if function_type=="Calculate":
        return 2
    if function_type=="LoadDB":
        return 3   
    if function_type=="TFIDF":
        return 5
    if function_type=="PandasInterpreter":
        lines = function_arguments["pandas_code"].splitlines()
        num_lines = len(lines) 
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        return math.sqrt(num_lines)*max(num_packages,1)
    if function_type=="PythonInterpreter":
        lines = function_arguments["python_code"].splitlines()
        num_lines = len(lines) 
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        return math.sqrt(num_lines)*max(num_packages,1)
    if function_type=="Forecaster":
        if function_arguments["model_name"]=="linear_regression":
            return 6
        elif function_arguments["model_name"]=="ARIMA":
            return 8
    if function_type=="TextualClassifier":
        if function_arguments["model_name"]=="logistic_regression":
            return 7
        elif function_arguments["model_name"]=="cnn":
            return 15
        elif function_arguments["model_name"]=="bert-base-uncased":
            return 20
    if function_type=="LLMInferencer":
        return 30
    if function_type=="Finish":
        return 0

def calc_cost2(function_type, function_arguments):
    if function_type=="Calculate":
        return 48
    if function_type=="LoadDB":
        return 47
    if function_type=="TFIDF":
        return 45
    if function_type=="PandasInterpreter":
        lines = function_arguments["pandas_code"].splitlines()
        num_lines = len(lines) 
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        return 50-math.sqrt(num_lines)*max(num_packages,1)
    if function_type=="PythonInterpreter":
        lines = function_arguments["python_code"].splitlines()
        num_lines = len(lines) 
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        return 50-math.sqrt(num_lines)*max(num_packages,1)
    if function_type=="Forecaster":
        if function_arguments["model_name"]=="linear_regression":
            return 44
        elif function_arguments["model_name"]=="ARIMA":
            return 42
    if function_type=="TextualClassifier":
        if function_arguments["model_name"]=="logistic_regression":
            return 43
        elif function_arguments["model_name"]=="cnn":
            return 35
        elif function_arguments["model_name"]=="bert-base-uncased":
            return 30
    if function_type=="LLMInferencer":
        return 20
    if function_type=="Finish":
        return 0
    
def calc_cost3(function_type, function_arguments):
    if function_type=="Calculate":
        return 2
    if function_type=="LoadDB":
        return 3   
    if function_type=="TFIDF":
        return 5
    if function_type=="PandasInterpreter":
        lines = function_arguments["pandas_code"].splitlines()
        num_lines = len(lines)  
        if num_lines<10:
            lines_cost = 4
        elif num_lines<=20:
            lines_cost = 10
        elif num_lines<=100:
            lines_cost = 15
        else:
            lines_cost = 20
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        if num_packages<2:
            packages_cost = 1
        elif num_packages<=5:
            packages_cost = 1.5
        else:
            packages_cost = 2
        return lines_cost*packages_cost
    if function_type=="PythonInterpreter":
        lines = function_arguments["python_code"].splitlines()
        num_lines = len(lines)  
        if num_lines<10:
            lines_cost = 4
        elif num_lines<=20:
            lines_cost = 10
        elif num_lines<=100:
            lines_cost = 15
        else:
            lines_cost = 20
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        if num_packages<2:
            packages_cost = 1
        elif num_packages<=5:
            packages_cost = 1.5
        else:
            packages_cost = 2
        return lines_cost*packages_cost
    if function_type=="Forecaster":
        if function_arguments["model_name"]=="linear_regression":
            return 6
        elif function_arguments["model_name"]=="ARIMA":
            return 8
    if function_type=="TextualClassifier":
        if function_arguments["model_name"]=="logistic_regression":
            return 7
        elif function_arguments["model_name"]=="cnn":
            return 15
        elif function_arguments["model_name"]=="bert-base-uncased":
            return 20
    if function_type=="LLMInferencer":
        return 30
    if function_type=="Finish":
        return 0

def calc_cost4(function_type, function_arguments):
    if function_type=="Calculate":
        return 48
    if function_type=="LoadDB":
        return 47
    if function_type=="TFIDF":
        return 45
    if function_type=="PandasInterpreter":
        lines = function_arguments["pandas_code"].splitlines()
        num_lines = len(lines)  
        if num_lines<10:
            lines_cost = 4
        elif num_lines<=20:
            lines_cost = 10
        elif num_lines<=100:
            lines_cost = 15
        else:
            lines_cost = 20
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        if num_packages<2:
            packages_cost = 1
        elif num_packages<=5:
            packages_cost = 1.5
        else:
            packages_cost = 2
        return 50-lines_cost*packages_cost
    if function_type=="PythonInterpreter":
        lines = function_arguments["python_code"].splitlines()
        num_lines = len(lines)  
        if num_lines<10:
            lines_cost = 4
        elif num_lines<=20:
            lines_cost = 10
        elif num_lines<=100:
            lines_cost = 15
        else:
            lines_cost = 20
        num_packages = 0
        for line in lines:
            if "from" and "import" in line:
                num_packages += 1
            elif "import" in line:
                num_packages += len(line.split(","))
        if num_packages<2:
            packages_cost = 1
        elif num_packages<=5:
            packages_cost = 1.5
        else:
            packages_cost = 2
        return 50-lines_cost*packages_cost
    if function_type=="Forecaster":
        if function_arguments["model_name"]=="linear_regression":
            return 44
        elif function_arguments["model_name"]=="ARIMA":
            return 42
    if function_type=="TextualClassifier":
        if function_arguments["model_name"]=="logistic_regression":
            return 43
        elif function_arguments["model_name"]=="cnn":
            return 35
        elif function_arguments["model_name"]=="bert-base-uncased":
            return 30
    if function_type=="LLMInferencer":
        return 20
    if function_type=="Finish":
        return 0