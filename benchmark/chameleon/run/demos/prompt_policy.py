import json

messages = [
    {
        'role': 'system',
        'content': """
You need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question.

Then calculate the cumulative cost at runtime. During each step of execution, add the cost to the cumulative cost. If the result of a tool call contains 'Error: ', do not add this cost to the cumulative cost. The cumulative cost stays the same.
    
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

Initial Cumulative Cost per question is 0. Below are some examples that map the problem to the tools.
"""
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """To execute: PythonInterpreter(def solution(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(19)\n) Cost is 4 (the number of lines of python_code < 10) * 1 (the number of imported packages in python_code < 2) = 4. Cumulative Cost is 4.""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'PythonInterpreter',
                    'arguments': json.dumps({
                        "python_code": """
def solution(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

ans = solution(19)
"""
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
        'content': 'The 20th Fibonacci number is 4181. Cumulative Cost is 4.'
    },
    {
        'role': 'user',
        'content': 'Which month had the highest number of patent applications in 2016?'
    },
    {
        'role': 'assistant',
        'content': """To execute: LoadDB(hupd, 2016-2016, False) Cost is 3. Cumulative Cost is 3.""",
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
        'content': """To execute: PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]) Cost is 4 (the number of lines of pandas_code < 10) * 1 (the number of imported packages in pandas_code < 2) = 4. Cumulative Cost is 7.""",
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
        'content': 'The month with the highest number of patent applications in 2016 was December. Cumulative cost is 7.'
    },
    {
        'role': 'user',
        'content': "Predict whether the patent application described in the following abstract will be accepted: 'A hydraulic control and/or safety device, particularly for utility apparatuses or systems or appliances, which is preferably able to carry out a flow shut-off and/or limitation, particularly in the event of fault of the utility apparatus or system or appliance, and/or one or more features that improve the device and/or the apparatus performance. In particular, the device can carry out the function of the fluid treatment, so as to be particularly reliable, as it prevents at least the formation of deposits on its mechanical components designed to limit the water flow.'?"
    },
    {
        'role': 'assistant',
        'content': """To execute: LoadDB(hupd, 2015-2017, True) Cost is 3. Cumulative Cost is 3.""",
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
        'content': """To execute: Classifier(logistic_regression, abstract, decision) Cost is 7 (model_name is "logistic_regression"). Cumulative Cost is 10.""",
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
        'content': "The patent application described in the abstract is predicted to be accepted. Cumulative Cost is 10."
    }
]


messages_formula = [
    {
        'role': 'system',
        'content': """
You need to act as a policy model that determines the sequence of tools with the lowest total interpretability cost that can be executed sequentially to solve the question. Follow these steps:

1. Generate Solutions: First, list as many solutions as possible. Each solution should be a sequence of tools that can be used to solve the question.

2. Calculate Interpretability Costs: Calculate the total interpretability cost for each solution. The interpretability cost of each tool in the solution is defined by the formulas below. Tools with lower interpretability costs are preferred over those with higher costs.

3. Execute the solution with a lower total interpretability cost and calculate the cumulative cost at runtime. During each step of execution, add the cost to the cumulative cost. If the result of a tool call contains 'Error: ', do not add this cost to the cumulative cost. The cumulative cost stays the same.

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

Initial Cumulative Cost per question is 0. Below are some examples that map the problem to the tools.
    """
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """Modules1: Calculate(0+0), Calculate(0+1), Calculate(0+1), Calculate(1+1), Calculate(1+2), Calculate(2+3), Calculate(3+5), Calculate(5+8), Calculate(8+13), Calculate(13+21), Calculate(21+34), Calculate(34+55), Calculate(55+89), Calculate(89+144), Calculate(144+233), Calculate(233+377), Calculate(377+610), Calculate(610+987), Calculate(987+1597), Calculate(1597+2584)
Modules2: PythonInterpreter(def solution(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(19)\n)

Thought: Total interpretability cost of Modules1 is calculated as follows: Calculate(0+0): 2, Calculate(0+1): 2, Calculate(0+1): 2, Calculate(1+1): 2, Calculate(1+2): 2, Calculate(2+3): 2, Calculate(3+5): 2, Calculate(5+8): 2, Calculate(8+13): 2, Calculate(13+21): 2, Calculate(21+34): 2, Calculate(34+55): 2, Calculate(55+89): 2, Calculate(89+144): 2, Calculate(144+233): 2, Calculate(233+377): 2, Calculate(377+610): 2, Calculate(610+987): 2, Calculate(987+1597): 2, Calculate(1597+2584): 2. Summing these costs: 2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+1=40.
Total interpretability cost of Modules2 is calculated as follows: PythonInterpreter(# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(19)\n): 4 (the number of lines of python_code < 10) * 1 (the number of imported packages in python_code < 2) = 4. Summing these costs: 4.
Therefore, Modules2 is selected because it has a lower total interpretability cost of 4 compared to 40 for Modules1.

Best Modules: PythonInterpreter(def solution(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(19)\n)

To execute: PythonInterpreter(def solution(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(19)\n) Cost is 4 (the number of lines of python_code < 10) * 1 (the number of imported packages in python_code < 2) = 4. Cumulative Cost is 4.""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'PythonInterpreter',
                    'arguments': json.dumps({
                        "python_code": """
def solution(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

ans = solution(19)
"""
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
        'content': 'The 20th Fibonacci number is 4181. Cumulative Cost is 4.'
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

Best Modules: LoadDB(hupd, 2016-2016, False), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0])

To execute: LoadDB(hupd, 2016-2016, False) Cost is 3. Cumulative Cost is 3.""",
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
        'content': """To execute: PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nans = df['filing_month'].mode()[0]) Cost is 4 (the number of lines of pandas_code < 10) * 1 (the number of imported packages in pandas_code < 2) = 4. Cumulative Cost is 7.""",
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
        'content': 'The month with the highest number of patent applications in 2016 was December. Cumulative cost is 7.'
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

Best Modules: LoadDB(hupd, 2015-2017, True), Classifier(logistic_regression, abstract, decision)

To execute: LoadDB(hupd, 2015-2017, True) Cost is 3. Cumulative Cost is 3.""",
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
        'content': """To execute: Classifier(logistic_regression, abstract, decision) Cost is 7 (model_name is "logistic_regression"). Cumulative Cost is 10.""",
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
        'content': "The patent application described in the abstract is predicted to be accepted. Cumulative Cost is 10."
    }
]




