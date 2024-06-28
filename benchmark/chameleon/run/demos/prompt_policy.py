
prompt_header_clean = """
You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows:

- Calculate[mathOp]: This module conducts an arithmetic operation and returns the result. It takes in an arithmetic operation and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB[DBName; subset; split]: This module loads a database specified by the DBName, subset, and a boolean value split, and returns the loaded dataframe or dataset dictionary. The DBName can be "hupd". The subset is in the format of startYear-endYear. When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. Normally, we only use "LoadDB" when the question requires data from a specific structured database.

- PandasInterpreter[pythonCode]: This module interprets Pandas code written in Python that involves operations on a DataFrame df, and returns the result. Normally, we only use "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataframe. We can only use "PandasInterpreter" after loading the dataframe with "LoadDB".

- PythonInterpreter[pythonCode]: This module interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only use "PythonInterpreter" when the question requires complex computations. We don't use "PythonInterpreter" when the question requires data manipulation performed on a specific structured dataframe.

- Classifier[modelName; predictorSection; target]: This module runs a specified classifier model on the given predictorSection to predict the target. The modelName can be "logistic_regression" or "distilbert-base-uncased". The predictorSection is a predictor variable of the classifier model, which is natural language requiring tokenization. Normally, we use the "Classifier" module for binary or multi-class classification tasks.

- Finish[answer]: This module returns the final answer and finishes the task. This module is the final module in the sequence that encapsulates the result of all previous modules.

Below are some examples that map the problem to the modules.
"""

prompt_header_rank = """
You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows, ordered by their interpretability from highest to lowest:

- Finish[answer]: This module returns the final answer and finishes the task. This module is the final module in the sequence that encapsulates the result of all previous modules.

- Calculate[mathOp]: This module conducts an arithmetic operation and returns the result. It takes in an arithmetic operation and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB[DBName; subset; split]: This module loads a database specified by the DBName, subset, and a boolean value split, and returns the loaded dataframe or dataset dictionary. The DBName can be "hupd". The subset is in the format of startYear-endYear. When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. Normally, we only use "LoadDB" when the question requires data from a specific structured database.

- PandasInterpreter[pythonCode]: This module interprets Pandas code written in Python that involves operations on a DataFrame df, and returns the result. Normally, we only use "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataframe. We can only use "PandasInterpreter" after loading the dataframe with "LoadDB".

- PythonInterpreter[pythonCode]: This module interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only use "PythonInterpreter" when the question requires complex computations. We don't use "PythonInterpreter" when the question requires data manipulation performed on a specific structured dataframe.

- Classifier[modelName; predictorSection; target]: This module runs a specified classifier model on the given predictorSection to predict the target. The modelName can be "logistic_regression" or "distilbert-base-uncased". The predictorSection is a predictor variable of the classifier model, which is natural language requiring tokenization. Normally, we use the "Classifier" module for binary or multi-class classification tasks.

Below are some examples that map the problem to the modules. When addressing a question, modules positioned higher on this list are preferred over those that are lower.
"""

prompt_header_formula = """
You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows, with the formulas used to calculate their interpretability costs defined in {}. Within these {}, each () contains a variable whose value is determined by the specific if condition it meets.

- Calculate[mathOp] {2}: This module conducts an arithmetic operation and returns the result. It takes in an arithmetic operation and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB[DBName; subset; split] {3}: This module loads a database specified by the DBName, subset, and a boolean value split, and returns the loaded dataframe or dataset dictionary. The DBName can be "hupd". The subset is in the format of startYear-endYear. When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. Normally, we only use "LoadDB" when the question requires data from a specific structured database.

- PandasInterpreter[pythonCode] {(if the number of lines of pythonCode < 10, 4; if the number of lines of pythonCode is between 10 and 20, 7; if the number of lines of pythonCode is between 21 and 100, 9; if the number of lines of pythonCode > 100, 10.) * (if the number of imported packages in pythonCode < 2, 1; if the number of imported packages in pythonCode is between 2 and 5, 1.5; if the number of imported packages in pythonCode > 5, 2)}: This module interprets Pandas code written in Python that involves operations on a DataFrame df, and returns the result. Normally, we only use "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataframe. We can only use "PandasInterpreter" after loading the dataframe with "LoadDB".

- PythonInterpreter[pythonCode] {(if the number of lines of pythonCode < 10, 4; if the number of lines of pythonCode is between 10 and 20, 7; if the number of lines of pythonCode is between 21 and 100, 9; if the number of lines of pythonCode > 100, 10.) * (if the number of imported packages in pythonCode < 2, 1; if the number of imported packages in pythonCode is between 2 and 5, 1.5; if the number of imported packages in pythonCode > 5, 2)}: This module interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only use "PythonInterpreter" when the question requires complex computations. We don't use "PythonInterpreter" when the question requires data manipulation performed on a specific structured dataframe.

- Classifier[modelName; predictorSection; target] {(if modelName is "logistic_regression", 7; if modelName is "distilbert-base-uncased", 10)}: This module runs a specified classifier model on the given predictorSection to predict the target. The modelName can be "logistic_regression" or "distilbert-base-uncased". The predictorSection is a predictor variable of the classifier model, which is natural language requiring tokenization. Normally, we use the "Classifier" module for binary or multi-class classification tasks.

- Finish[answer] {1}: This module returns the final answer and finishes the task. This module is the final module in the sequence that encapsulates the result of all previous modules.

Below are some examples that map problems to modules. When addressing a question, first calculate the interpretability cost for each module. The interpretability cost of a set of modules is the sum of the costs of each module in that set. Choose the set of modules with the lowest total interpretability cost, as modules with lower costs are preferred over those with higher costs.
"""

# If got more than 1 argument, need to separate with ;
# Needs an example with dbloader where the question doesn't involve specific years

prompt_example_clean = """
Question: What is the 20th Fibonacci number?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Question: Which month had the highest number of patent applications in 2016?

Modules: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"]

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"]

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. Please provide only the sequence of Modules like the examples above and nothing else.
"""

prompt_example_compare = """
Question: What is the 20th Fibonacci number?

Modules1: ["Calculate[0+0]", "Calculate[0+1]", "Calculate[0+1]", "Calculate[1+1]", "Calculate[1+2]", "Calculate[2+3]", "Calculate[3+5]", "Calculate[5+8]", "Calculate[8+13]", "Calculate[13+21]", "Calculate[21+34]", "Calculate[34+55]", "Calculate[55+89]", "Calculate[89+144]", "Calculate[144+233]", "Calculate[233+377]", "Calculate[377+610]", "Calculate[610+987]", "Calculate[987+1597]", "Calculate[1597+2584]", "Finish[4181]"]
Modules2: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Thought: Modules2 is selected because it's more interpretable. 

Best Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Question: Which month had the highest number of patent applications in 2016?

Modules1: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"]
Modules2: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter["import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]", "Finish[12]"]

Thought: Modules1 is selected because it's more interpretable.

Best Modules: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"]

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"]
Modules2: ["LoadDB[hupd; 2015-2017; True]", "Classifier[distilbert-base-uncased; abstract; decision]"]

Thought: Modules1 is selected because it's more interpretable. 

Best Modules: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"]

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. Please provide only the sequence of Best Modules like the examples above and nothing else.
"""

prompt_example_compare_full = """
Question: What is the 20th Fibonacci number?

Modules1: ["Calculate[0+0]", "Calculate[0+1]", "Calculate[0+1]", "Calculate[1+1]", "Calculate[1+2]", "Calculate[2+3]", "Calculate[3+5]", "Calculate[5+8]", "Calculate[8+13]", "Calculate[13+21]", "Calculate[21+34]", "Calculate[34+55]", "Calculate[55+89]", "Calculate[89+144]", "Calculate[144+233]", "Calculate[233+377]", "Calculate[377+610]", "Calculate[610+987]", "Calculate[987+1597]", "Calculate[1597+2584]", "Finish[4181]"]
Modules2: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Thought: Modules2 is selected because it's more interpretable. 

Best Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Question: Which month had the highest number of patent applications in 2016?

Modules1: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"]
Modules2: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter["import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]", "Finish[12]"]

Thought: Modules1 is selected because it's more interpretable.

Best Modules: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"]

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"]
Modules2: ["LoadDB[hupd; 2015-2017; True]", "Classifier[distilbert-base-uncased; abstract; decision]"]

Thought: Modules1 is selected because it's more interpretable. 

Best Modules: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"]

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. Please provide the sequence of Modules1, Modules2, Best Modules, and Thought like the examples above and nothing else.
"""

prompt_example_formula = """
Question: What is the 20th Fibonacci number?

Modules1: ["Calculate[0+0]", "Calculate[0+1]", "Calculate[0+1]", "Calculate[1+1]", "Calculate[1+2]", "Calculate[2+3]", "Calculate[3+5]", "Calculate[5+8]", "Calculate[8+13]", "Calculate[13+21]", "Calculate[21+34]", "Calculate[34+55]", "Calculate[55+89]", "Calculate[89+144]", "Calculate[144+233]", "Calculate[233+377]", "Calculate[377+610]", "Calculate[610+987]", "Calculate[987+1597]", "Calculate[1597+2584]", "Finish[4181]"]
Modules2: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Thought: Total interpretability cost of Modules1 is calculated as follows: "Calculate[0+0]" {2}, "Calculate[0+1]" {2}, "Calculate[0+1]" {2}, "Calculate[1+1]" {2}, "Calculate[1+2]" {2}, "Calculate[2+3]" {2}, "Calculate[3+5]" {2}, "Calculate[5+8]" {2}, "Calculate[8+13]" {2}, "Calculate[13+21]" {2}, "Calculate[21+34]" {2}, "Calculate[34+55]" {2}, "Calculate[55+89]" {2}, "Calculate[89+144]" {2}, "Calculate[144+233]" {2}, "Calculate[233+377]" {2}, "Calculate[377+610]" {2}, "Calculate[610+987]" {2}, "Calculate[987+1597]" {2}, "Calculate[1597+2584]" {2}, "Finish[4181]" {1}. Summing these costs: 2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+1=41.
Total interpretability cost of Modules2 is calculated as follows: "PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]" {4 (the number of lines of pythonCode < 10) * 1 (the number of imported packages in pythonCode < 2) = 4}, "Finish[4181]" {1}. Summing these costs: 4+1=5.
Therefore, Modules2 is selected because it has a lower total interpretability cost of 5 compared to 41 for Modules1.

Best Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"] {5}

Question: Which month had the highest number of patent applications in 2016?

Modules1: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"]
Modules2: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter["import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]", "Finish[12]"]

Thought: Total interpretability cost of Modules1 is calculated as follows: "LoadDB[hupd; 2016-2016; False]" {3}, "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]" {4 (the number of lines of pythonCode < 10) * 1 (the number of imported packages in pythonCode < 2) = 4}, "Finish[12]" {1}. Summing these costs: 3+4+1=8.
Total interpretability cost of Modules2 is calculated as follows: "LoadDB[hupd; 2016-2016; False]" {3}, "PandasInterpreter["import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]" {4 (the number of lines of pythonCode < 10) * 1.5 (the number of imported packages in pythonCode is between 2 and 5) = 6}, "Finish[12]" {1}. Summing these costs: 3+6+1=10.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 8 compared to 10 for Modules2.

Best Modules: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"] {8}

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"]
Modules2: ["LoadDB[hupd; 2015-2017; True]", "Classifier[distilbert-base-uncased; abstract; decision]"]

Thought: Total interpretability cost of Modules1 is calculated as follows: "LoadDB[hupd; 2015-2017; True]" {3}, "Classifier[logistic_regression; abstract; decision]" {7 (modelName is "logistic_regression")}. Summing these costs: 3+7=10.
Total interpretability cost of Modules2 is calculated as follows: "LoadDB[hupd; 2015-2017; True]" {3}, "Classifier[distilbert-base-uncased; abstract; decision]" {10 (modelName is "logistic_regression")}. Summing these costs: 3+10=13.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 10 compared to 13 for Modules2.

Best Modules: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"] {10}

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. Please provide only the sequence of Best Modules like those from the examples above and nothing else.
""" 

prompt_example_formula_full = """
Question: What is the 20th Fibonacci number?

Modules1: ["Calculate[0+0]", "Calculate[0+1]", "Calculate[0+1]", "Calculate[1+1]", "Calculate[1+2]", "Calculate[2+3]", "Calculate[3+5]", "Calculate[5+8]", "Calculate[8+13]", "Calculate[13+21]", "Calculate[21+34]", "Calculate[34+55]", "Calculate[55+89]", "Calculate[89+144]", "Calculate[144+233]", "Calculate[233+377]", "Calculate[377+610]", "Calculate[610+987]", "Calculate[987+1597]", "Calculate[1597+2584]", "Finish[4181]"]
Modules2: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Thought: Total interpretability cost of Modules1 is calculated as follows: "Calculate[0+0]" {2}, "Calculate[0+1]" {2}, "Calculate[0+1]" {2}, "Calculate[1+1]" {2}, "Calculate[1+2]" {2}, "Calculate[2+3]" {2}, "Calculate[3+5]" {2}, "Calculate[5+8]" {2}, "Calculate[8+13]" {2}, "Calculate[13+21]" {2}, "Calculate[21+34]" {2}, "Calculate[34+55]" {2}, "Calculate[55+89]" {2}, "Calculate[89+144]" {2}, "Calculate[144+233]" {2}, "Calculate[233+377]" {2}, "Calculate[377+610]" {2}, "Calculate[610+987]" {2}, "Calculate[987+1597]" {2}, "Calculate[1597+2584]" {2}, "Finish[4181]" {1}. Summing these costs: 2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+1=41.
Total interpretability cost of Modules2 is calculated as follows: "PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]" {4 (the number of lines of pythonCode < 10) * 1 (the number of imported packages in pythonCode < 2) = 4}, "Finish[4181]" {1}. Summing these costs: 4+1=5.
Therefore, Modules2 is selected because it has a lower total interpretability cost of 5 compared to 41 for Modules1.

Best Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"] {5}

Question: Which month had the highest number of patent applications in 2016?

Modules1: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"]
Modules2: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter["import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]", "Finish[12]"]

Thought: Total interpretability cost of Modules1 is calculated as follows: "LoadDB[hupd; 2016-2016; False]" {3}, "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]" {4 (the number of lines of pythonCode < 10) * 1 (the number of imported packages in pythonCode < 2) = 4}, "Finish[12]" {1}. Summing these costs: 3+4+1=8.
Total interpretability cost of Modules2 is calculated as follows: "LoadDB[hupd; 2016-2016; False]" {3}, "PandasInterpreter["import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]" {4 (the number of lines of pythonCode < 10) * 1.5 (the number of imported packages in pythonCode is between 2 and 5) = 6}, "Finish[12]" {1}. Summing these costs: 3+6+1=10.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 8 compared to 10 for Modules2.

Best Modules: ["LoadDB[hupd; 2016-2016; False]", "PandasInterpreter[import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]]", "Finish[12]"] {8}

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"]
Modules2: ["LoadDB[hupd; 2015-2017; True]", "Classifier[distilbert-base-uncased; abstract; decision]"]

Thought: Total interpretability cost of Modules1 is calculated as follows: "LoadDB[hupd; 2015-2017; True]" {3}, "Classifier[logistic_regression; abstract; decision]" {7 (modelName is "logistic_regression")}. Summing these costs: 3+7=10.
Total interpretability cost of Modules2 is calculated as follows: "LoadDB[hupd; 2015-2017; True]" {3}, "Classifier[distilbert-base-uncased; abstract; decision]" {10 (modelName is "logistic_regression")}. Summing these costs: 3+10=13.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 10 compared to 13 for Modules2.

Best Modules: ["LoadDB[hupd; 2015-2017; True]", "Classifier[logistic_regression; abstract; decision]"] {10}

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. Please provide only the sequence of Modules1, Modules2, Thought, and Best Modules like the examples above and nothing else.
""" 

prompt = prompt_header_formula+prompt_example_formula_full