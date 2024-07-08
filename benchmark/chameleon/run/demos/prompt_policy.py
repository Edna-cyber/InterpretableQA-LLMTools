import json

prompt_header_clean = """
You need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question.

The tools are defined as follows:

- Calculate(query): This tool conducts an arithmetic operation and returns the result. It takes in an arithmetic operation and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB(target_db, duration, split): This tool loads a database specified by the target_db, duration, and a boolean value split, and returns the loaded dataframe or dataset dictionary. The target_db can be "hupd". The duration is in the format of startYear-endYear. When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. Normally, we only use "LoadDB" when the question requires data from a specific structured database.

- PandasInterpreter(pandas_code): This tool interprets Pandas code written in Python that involves operations on a DataFrame df, and returns the result. Normally, we only use "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataframe. We must first use LoadDB before we can use PandasInterpreter.

- PythonInterpreter(python_code): This tool interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only use "PythonInterpreter" when the question requires complex computations. We don't use "PythonInterpreter" when the question requires data manipulation performed on a specific structured dataframe.

- Classifier(model_name, section, target, num_classes): This tool runs a specified classifier model on the given section to predict the target, which has num_classes number of classes. The model_name can be "logistic_regression" or "distilbert-base-uncased". The section is a predictor variable of the classifier model, which is natural language requiring tokenization. The default value of num_classes is 2 for binary classification. Normally, we use the "Classifier" tool for binary or multi-class classification tasks.

Below are some examples that map the problem to the tools.
"""

prompt_header_rank = """
You need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question.

The tools are defined as follows, ordered by their interpretability from highest to lowest:

- Calculate(query): This tool conducts an arithmetic operation and returns the result. It takes in an arithmetic operation and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB(target_db, duration, split): This tool loads a database specified by the target_db, duration, and a boolean value split, and returns the loaded dataframe or dataset dictionary. The target_db can be "hupd". The duration is in the format of startYear-endYear. When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. Normally, we only use "LoadDB" when the question requires data from a specific structured database.

- PandasInterpreter(pandas_code): This tool interprets Pandas code written in Python that involves operations on a DataFrame df, and returns the result. Normally, we only use "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataframe. We can only use "PandasInterpreter" after loading the dataframe with "LoadDB".

- PythonInterpreter(python_code): This tool interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only use "PythonInterpreter" when the question requires complex computations. We don't use "PythonInterpreter" when the question requires data manipulation performed on a specific structured dataframe.

- Classifier(model_name, section, target, num_classes): This tool runs a specified classifier model on the given section to predict the target, which has num_classes number of classes. The model_name can be "logistic_regression" or "distilbert-base-uncased". The section is a predictor variable of the classifier model, which is natural language requiring tokenization. The default value of num_classes is 2 for binary classification. Normally, we use the "Classifier" tool for binary or multi-class classification tasks.

Below are some examples that map the problem to the tools. When addressing a question, tools positioned higher on this list are preferred over those that are lower.
"""

prompt_header_formula = """
You need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question.

The tools are defined as follows, with the formulas used to calculate their interpretability costs. Each () contains a variable whose value is determined by the specific if condition it meets.

- Calculate(query) Cost is 2: This tool conducts an arithmetic operation and returns the result. It takes in an arithmetic operation and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB(target_db, duration, split) Cost is 3: This tool loads a database specified by the target_db, duration, and a boolean value split, and returns the loaded dataframe or dataset dictionary. The target_db can be "hupd". The duration is in the format of startYear-endYear. When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. Normally, we only use "LoadDB" when the question requires data from a specific structured database.

- PandasInterpreter(pandas_code) Cost is (if the number of lines of pandas_code < 10, 4; if the number of lines of pandas_code is between 10 and 20, 7; if the number of lines of pandas_code is between 21 and 100, 9; if the number of lines of pandas_code > 100, 10.) * (if the number of imported packages in pandas_code < 2, 1; if the number of imported packages in pandas_code is between 2 and 5, 1.5; if the number of imported packages in pandas_code > 5, 2): This tool interprets Pandas code written in Python that involves operations on a DataFrame df, and returns the result. Normally, we only use "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataframe. We can only use "PandasInterpreter" after loading the dataframe with "LoadDB".

- PythonInterpreter(python_code) Cost is (if the number of lines of python_code < 10, 4; if the number of lines of python_code is between 10 and 20, 7; if the number of lines of python_code is between 21 and 100, 9; if the number of lines of python_code > 100, 10.) * (if the number of imported packages in python_code < 2, 1; if the number of imported packages in python_code is between 2 and 5, 1.5; if the number of imported packages in python_code > 5, 2): This tool interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only use "PythonInterpreter" when the question requires complex computations. We don't use "PythonInterpreter" when the question requires data manipulation performed on a specific structured dataframe.

- Classifier(model_name, section, target, num_classes) Cost is (if model_name is "logistic_regression", 7; if model_name is "distilbert-base-uncased", 10): This tool runs a specified classifier model on the given section to predict the target, which has num_classes number of classes. The model_name can be "logistic_regression" or "distilbert-base-uncased". The section is a predictor variable of the classifier model, which is natural language requiring tokenization. The default value of num_classes is 2 for binary classification. Normally, we use the "Classifier" tool for binary or multi-class classification tasks.

Below are some examples that map problems to tools. When addressing a question, first calculate the interpretability cost for each tool. The interpretability cost of a set of tools is the sum of the costs of each tool in that set. Choose the set of tools with the lowest total interpretability cost, as tools with lower costs are preferred over those with higher costs.
"""

# If got more than 1 argument, need to separate with ;
# Needs an example with dbloader where the question doesn't involve specific years

prompt_example_clean = """
Question: What is the 20th Fibonacci number?

Modules: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

Question: Which month had the highest number of patent applications in 2016?

Modules: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)

Now, you need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question. Please provide only the sequence of Modules like the examples above and nothing else.
"""

prompt_example_compare = """
Question: What is the 20th Fibonacci number?

Modules1: Calculate(0+0), Calculate(0+1), Calculate(0+1), Calculate(1+1), Calculate(1+2), Calculate(2+3), Calculate(3+5), Calculate(5+8), Calculate(8+13), Calculate(13+21), Calculate(21+34), Calculate(34+55), Calculate(55+89), Calculate(89+144), Calculate(144+233), Calculate(233+377), Calculate(377+610), Calculate(610+987), Calculate(987+1597), Calculate(1597+2584)
Modules2: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

Thought: Modules2 is selected because it's more interpretable. 

Best Modules: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

Question: Which month had the highest number of patent applications in 2016?

Modules1: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])
Modules2: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0])

Thought: Modules1 is selected because it's more interpretable.

Best Modules: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)
Modules2: LoadDB(hupd, 2015-2017, True), Classifier(distilbert-base-uncased, abstract, decision)

Thought: Modules1 is selected because it's more interpretable. 

Best Modules: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)

Now, you need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question. Please provide only the sequence of Best Modules like the examples above and nothing else.
"""

prompt_example_compare_full = """
Question: What is the 20th Fibonacci number?

Modules1: Calculate(0+0), Calculate(0+1), Calculate(0+1), Calculate(1+1), Calculate(1+2), Calculate(2+3), Calculate(3+5), Calculate(5+8), Calculate(8+13), Calculate(13+21), Calculate(21+34), Calculate(34+55), Calculate(55+89), Calculate(89+144), Calculate(144+233), Calculate(233+377), Calculate(377+610), Calculate(610+987), Calculate(987+1597), Calculate(1597+2584)
Modules2: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

Thought: Modules2 is selected because it's more interpretable. 

Best Modules: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

Question: Which month had the highest number of patent applications in 2016?

Modules1: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])
Modules2: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0])

Thought: Modules1 is selected because it's more interpretable.

Best Modules: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)
Modules2: LoadDB(hupd, 2015-2017, True), Classifier(distilbert-base-uncased, abstract, decision)

Thought: Modules1 is selected because it's more interpretable. 

Best Modules: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)

Now, you need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question. Please provide the sequence of Modules1, Modules2, Best Modules, and Thought like the examples above and nothing else.
"""

prompt_example_formula = """
Question: What is the 20th Fibonacci number?

Modules1: Calculate(0+0), Calculate(0+1), Calculate(0+1), Calculate(1+1), Calculate(1+2), Calculate(2+3), Calculate(3+5), Calculate(5+8), Calculate(8+13), Calculate(13+21), Calculate(21+34), Calculate(34+55), Calculate(55+89), Calculate(89+144), Calculate(144+233), Calculate(233+377), Calculate(377+610), Calculate(610+987), Calculate(987+1597), Calculate(1597+2584)
Modules2: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

Thought: Total interpretability cost of Modules1 is calculated as follows: Calculate(0+0): 2, Calculate(0+1): 2, Calculate(0+1): 2, Calculate(1+1): 2, Calculate(1+2): 2, Calculate(2+3): 2, Calculate(3+5): 2, Calculate(5+8): 2, Calculate(8+13): 2, Calculate(13+21): 2, Calculate(21+34): 2, Calculate(34+55): 2, Calculate(55+89): 2, Calculate(89+144): 2, Calculate(144+233): 2, Calculate(233+377): 2, Calculate(377+610): 2, Calculate(610+987): 2, Calculate(987+1597): 2, Calculate(1597+2584): 2. Summing these costs: 2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+1=40.
Total interpretability cost of Modules2 is calculated as follows: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n): 4 (the number of lines of python_code < 10) * 1 (the number of imported packages in python_code < 2) = 4. Summing these costs: 4.
Therefore, Modules2 is selected because it has a lower total interpretability cost of 4 compared to 40 for Modules1.

Best Modules: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n) Cost is 4

Question: Which month had the highest number of patent applications in 2016?

Modules1: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])
Modules2: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0])

Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2016-2016, False): 3, PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]): 4 (the number of lines of pandas_code < 10) * 1 (the number of imported packages in pandas_code < 2) = 4. Summing these costs: 3+4=7.
Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2016-2016, False): 3, PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]): 4 (the number of lines of pandas_code < 10) * 1.5 (the number of imported packages in pandas_code is between 2 and 5) = 6. Summing these costs: 3+6=9.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 7 compared to 9 for Modules2.

Best Modules: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]) Cost is 7
    
Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)
Modules2: LoadDB(hupd, 2015-2017, True), Classifier(distilbert-base-uncased, abstract, decision)

Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2015-2017, True): 3, Classifier(logistic_regression, abstract, decision): 7 (model_name is "logistic_regression"). Summing these costs: 3+7=10.
Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2015-2017, True): 3, Classifier(distilbert-base-uncased, abstract, decision): 10 (model_name is "logistic_regression"). Summing these costs: 3+10=13.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 10 compared to 13 for Modules2.

Best Modules: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision) Cost is 10

Now, you need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question. Please provide only the sequence of Best Modules like those from the examples above and nothing else.
""" 

prompt_example_formula_full = """
Question: What is the 20th Fibonacci number?

Modules1: Calculate(0+0), Calculate(0+1), Calculate(0+1), Calculate(1+1), Calculate(1+2), Calculate(2+3), Calculate(3+5), Calculate(5+8), Calculate(8+13), Calculate(13+21), Calculate(21+34), Calculate(34+55), Calculate(55+89), Calculate(89+144), Calculate(144+233), Calculate(233+377), Calculate(377+610), Calculate(610+987), Calculate(987+1597), Calculate(1597+2584)
Modules2: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

Thought: Total interpretability cost of Modules1 is calculated as follows: Calculate(0+0): 2, Calculate(0+1): 2, Calculate(0+1): 2, Calculate(1+1): 2, Calculate(1+2): 2, Calculate(2+3): 2, Calculate(3+5): 2, Calculate(5+8): 2, Calculate(8+13): 2, Calculate(13+21): 2, Calculate(21+34): 2, Calculate(34+55): 2, Calculate(55+89): 2, Calculate(89+144): 2, Calculate(144+233): 2, Calculate(233+377): 2, Calculate(377+610): 2, Calculate(610+987): 2, Calculate(987+1597): 2, Calculate(1597+2584): 2. Summing these costs: 2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+1=40.
Total interpretability cost of Modules2 is calculated as follows: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n): 4 (the number of lines of python_code < 10) * 1 (the number of imported packages in python_code < 2) = 4. Summing these costs: 4.
Therefore, Modules2 is selected because it has a lower total interpretability cost of 4 compared to 40 for Modules1.

Best Modules: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n) Cost is 4

Question: Which month had the highest number of patent applications in 2016?

Modules1: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])
Modules2: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0])

Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2016-2016, False): 3, PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]): 4 (the number of lines of pandas_code < 10) * 1 (the number of imported packages in pandas_code < 2) = 4. Summing these costs: 3+4=7.
Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2016-2016, False): 3, PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]): 4 (the number of lines of pandas_code < 10) * 1.5 (the number of imported packages in pandas_code is between 2 and 5) = 6. Summing these costs: 3+6=9.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 7 compared to 9 for Modules2.

Best Modules: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]) Cost is 7

Question: Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?

Modules1: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)
Modules2: LoadDB(hupd, 2015-2017, True), Classifier(distilbert-base-uncased, abstract, decision)

Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2015-2017, True): 3, Classifier(logistic_regression, abstract, decision): 7 (model_name is "logistic_regression"). Summing these costs: 3+7=10.
Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2015-2017, True): 3, Classifier(distilbert-base-uncased, abstract, decision): 10 (model_name is "logistic_regression"). Summing these costs: 3+10=13.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 10 compared to 13 for Modules2.

Best Modules: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision) Cost is 10

Now, you need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question. Please provide only the sequence of Modules (including as many as necessary, e.g., Modules1, Modules2, etc.), Thought, and Best Modules like the examples above and nothing else. Then, execute the tools and arguments one by one, following the sequence specified after 'Best Modules: ' precisely.
""" 

prompt = prompt_header_clean+prompt_example_clean
prompt_formula = prompt_header_formula+prompt_example_formula_full

messages = [
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'PythonInterpreter',
                    'arguments': json.dumps({
                        "python_code": "# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'PythonInterpreter',
        'content': '4181'
    },
    {
        'role': 'assistant',
        'content': 'The 20th Fibonacci number is 4181.'
    },
    {
        'role': 'user',
        'content': 'Which month had the highest number of patent applications in 2016?'
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2016-2016", "split": False
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'LoadDB',
        'content': "We have successfully loaded the hupd dataframe, including the following columns: 'patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category'. It has the following structure: patent_number decision  ... icpr_category cpc_category 0 <NA>  PENDING  ... F16  F16 1 <NA>  PENDING  ... C12  C12 2 <NA>  PENDING  ... H04  H04 3 <NA>  PENDING  ... G06  G06 4 <NA>  PENDING  ... H02  H02"
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": "import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'PandasInterpreter',
        'content': '12'
    },
    {
        'role': 'assistant',
        'content': 'The month with the highest number of patent applications in 2016 was December.'
    },
    {
        'role': 'user',
        'content': "Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?"
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2015-2017", "split": True
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'LoadDB',
        'content': "We have successfully loaded the hupd dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 18011}) validation: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 12008})})"
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'Classifier',
                    'arguments': json.dumps({
                        "model_name": "logistic_regression", "section": "abstract", "target": "decision"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Classifier',
        'content': 'ACCEPTED'
    },
    {
        'role': 'assistant',
        'content': "The patent application described in the abstract is predicted to be accepted."
    }
]


messages_formula = [
    {
        'role': 'system',
        'content': """
    You need to act as a policy model that determines the sequence of tools with the lowest total interpretability cost that can be executed sequentially to solve the question. Follow these steps:

    1. Generate Solutions: First, list as many solutions as possible. Each solution should be a sequence of tools that can be used to solve the question.

    2. Calculate Interpretability Costs: Calculate the total interpretability cost for each solution. The interpretability cost of each tool in the solution is defined by the formulas below. Tools with lower interpretability costs are preferred over those with higher costs.

    Interpretability Cost Formulas:

    1. Calculate: Cost is 2

    2. LoadDB: Cost is 3

    3. PandasInterpreter: Cost is based on the number of lines of Python code and the number of imported packages:
        - Number of Lines of Python Code:
            - If less than 10 lines: 4
            - If between 10 and 20 lines: 7
            - If between 21 and 100 lines: 9
            - If more than 100 lines: 10
        - Number of Imported Packages:
            - If fewer than 2 packages: 1
            - If between 2 and 5 packages: 1.5
            - If more than 5 packages: 2
        - Formula: (Cost based on number of lines) * (Cost based on number of packages)

    4. PythonInterpreter: Cost is similar to PandasInterpreter, based on the number of lines of Python code and the number of imported packages:
        - Number of Lines of Python Code:
            - If less than 10 lines: 4
            - If between 10 and 20 lines: 7
            - If between 21 and 100 lines: 9
            - If more than 100 lines: 10
        - Number of Imported Packages:
            - If fewer than 2 packages: 1
            - If between 2 and 5 packages: 1.5
            - If more than 5 packages: 2
        - Formula: (Cost based on number of lines) * (Cost based on number of packages)

    5. Classifier: Cost is based on the model name:
        - If model name is "logistic_regression": 7
        - If model name is "distilbert-base-uncased": 10
    """
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """Modules1: Calculate(0+0), Calculate(0+1), Calculate(0+1), Calculate(1+1), Calculate(1+2), Calculate(2+3), Calculate(3+5), Calculate(5+8), Calculate(8+13), Calculate(13+21), Calculate(21+34), Calculate(34+55), Calculate(55+89), Calculate(89+144), Calculate(144+233), Calculate(233+377), Calculate(377+610), Calculate(610+987), Calculate(987+1597), Calculate(1597+2584)
        Modules2: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n)

        Thought: Total interpretability cost of Modules1 is calculated as follows: Calculate(0+0): 2, Calculate(0+1): 2, Calculate(0+1): 2, Calculate(1+1): 2, Calculate(1+2): 2, Calculate(2+3): 2, Calculate(3+5): 2, Calculate(5+8): 2, Calculate(8+13): 2, Calculate(13+21): 2, Calculate(21+34): 2, Calculate(34+55): 2, Calculate(55+89): 2, Calculate(89+144): 2, Calculate(144+233): 2, Calculate(233+377): 2, Calculate(377+610): 2, Calculate(610+987): 2, Calculate(987+1597): 2, Calculate(1597+2584): 2. Summing these costs: 2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+1=40.
        Total interpretability cost of Modules2 is calculated as follows: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n): 4 (the number of lines of python_code < 10) * 1 (the number of imported packages in python_code < 2) = 4. Summing these costs: 4.
        Therefore, Modules2 is selected because it has a lower total interpretability cost of 4 compared to 40 for Modules1.

        Best Modules: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n) Cost is 4""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'PythonInterpreter',
                    'arguments': json.dumps({
                        "python_code": "# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'PythonInterpreter',
        'content': '4181'
    },
    {
        'role': 'assistant',
        'content': 'The 20th Fibonacci number is 4181.'
    },
    {
        'role': 'user',
        'content': 'Which month had the highest number of patent applications in 2016?'
    },
    {
        'role': 'assistant',
        'content': """Modules1: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])
        Modules2: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0])

        Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2016-2016, False): 3, PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]): 4 (the number of lines of pandas_code < 10) * 1 (the number of imported packages in pandas_code < 2) = 4. Summing these costs: 3+4=7.
        Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2016-2016, False): 3, PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nans = counter.most_common()[0][0]): 4 (the number of lines of pandas_code < 10) * 1.5 (the number of imported packages in pandas_code is between 2 and 5) = 6. Summing these costs: 3+6=9.
        Therefore, Modules1 is selected because it has a lower total interpretability cost of 7 compared to 9 for Modules2.

        Best Modules: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]) Cost is 7""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2016-2016", "split": False
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'LoadDB',
        'content': "We have successfully loaded the hupd dataframe, including the following columns: 'patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category'. It has the following structure: patent_number decision  ... icpr_category cpc_category 0 <NA>  PENDING  ... F16  F16 1 <NA>  PENDING  ... C12  C12 2 <NA>  PENDING  ... H04  H04 3 <NA>  PENDING  ... G06  G06 4 <NA>  PENDING  ... H02  H02"
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": "import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'PandasInterpreter',
        'content': '12'
    },
    {
        'role': 'assistant',
        'content': 'The month with the highest number of patent applications in 2016 was December.'
    },
    {
        'role': 'user',
        'content': "Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?"
    },
    {
        'role': 'assistant',
        'content': """Modules1: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)
        Modules2: LoadDB(hupd, 2015-2017, True), Classifier(distilbert-base-uncased, abstract, decision)

        Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2015-2017, True): 3, Classifier(logistic_regression, abstract, decision): 7 (model_name is "logistic_regression"). Summing these costs: 3+7=10.
        Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2015-2017, True): 3, Classifier(distilbert-base-uncased, abstract, decision): 10 (model_name is "logistic_regression"). Summing these costs: 3+10=13.
        Therefore, Modules1 is selected because it has a lower total interpretability cost of 10 compared to 13 for Modules2.

        Best Modules: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision) Cost is 10""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2015-2017", "split": True
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'LoadDB',
        'content': "We have successfully loaded the hupd dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 18011}) validation: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 12008})})"
    },
    {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'Classifier',
                    'arguments': json.dumps({
                        "model_name": "logistic_regression", "section": "abstract", "target": "decision"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Classifier',
        'content': 'ACCEPTED'
    },
    {
        'role': 'assistant',
        'content': "The patent application described in the abstract is predicted to be accepted."
    }
]




