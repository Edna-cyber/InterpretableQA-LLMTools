import json

messages = [
    {
        'role': 'system',
        'content': """You need to act as a policy model, that given a question and a set of tools, determines the sequence of tools that can be executed sequentially can solve the question. Below are some examples that map the problem to the tools."""
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """To execute: PythonInterpreter(def solution(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(19)\n)""", 
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
        'content': "{'ans': 4181}"
    },
    {
        'role': 'assistant',
        'content': """To execute: Finish({'ans': 4181}, ans, integer)""",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'ans': 4181}", "answer_variable": "ans", "answer_type": "integer"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Finish',
        'content': "4181"
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
        'content': """To execute: LoadDB(hupd, 2016-2016, None, None)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "train_duration": "2016-2016", "test_duration": "None", "outcome_col": "None"
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
        'content': """To execute: PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0])""",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": "import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0]"
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
        'content': "{'month':12}"
    },
    {
        'role': 'assistant',
        'content': """To execute: Finish({'month':12}, month, integer)""",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'month':12}", "answer_variable": "month", "answer_type": "integer"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'Finish',
        'content': "12"
    },
    {
        'role': 'assistant',
        'content': 'The month with the highest number of patent applications in 2016 was December.'
    },
    {
        'role': 'user',
        'content': 'Train a model using patent applications from 2004 to 2006, and then use it to predict the decisions for patent applications filed in 2007.'
    },
    {
        'role': 'assistant',
        'content': """To execute: LoadDB(hupd, 2004-2006, 2007-2007, decision)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "train_duration": "2004-2006", "test_duration": "2007-2007", "outcome_col": "decision"
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
        'content': "We have successfully loaded the hupd dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 18011}) test: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 12008})})"
    },
    {
        'role': 'assistant',
        'content': """To execute: TextualClassifier(logistic_regression, abstract, decision)""",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'TextualClassifier',
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
        'content': "{'predictions': ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED',...]}"
    },
    {
        'role': 'assistant',
        'content': """To execute: Finish({'predictions': ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED',...]}, predictions, list)""",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'predictions': ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED']}", "answer_variable": "predictions", "answer_type": "list"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'Finish',
        'content': "['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED']"
    },
    {
        'role': 'assistant',
        'content': "The patent applications from 2007 are predicted to receive the following decisions: ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED']."
    }
]


messages_formula = [
    {
        'role': 'system',
        'content': """
You need to act as a policy model that determines the sequence of tools with the lowest total interpretability cost that can be executed sequentially to solve the question. Follow these steps:

1. Generate Solutions: First, list AS MANY solutions AS POSSIBLE (at most 4). Each solution should be a sequence of tools that can be used to solve the question.

2. Calculate Interpretability Costs: Calculate the total interpretability cost for each solution. The interpretability cost of each tool in the solution is defined by the formulas below. Tools with lower interpretability costs are preferred over those with higher costs.

3. Execute the solution with the lowest total interpretability cost.

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

5. TextualClassifier: Cost is based on the model name:
    - If model name is "logistic_regression": 7
    - If model name is "distilbert-base-uncased": 10

Below are some examples that map the problem to the tools.
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

To execute: PythonInterpreter(def solution(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(19)\n)""",
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
        'content': "{'ans':4181}"
    },
    {
        'role': 'assistant',
        'content': """To execute: Finish({'ans': 4181}, ans, integer)""",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'ans': 4181}", "answer_variable": "ans", "answer_type": "integer"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Finish',
        'content': "4181"
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
        'content': """Modules1: LoadDB(hupd, 2016-2016, None, None), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0])
Modules2: LoadDB(hupd, 2016-2016, None, None), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0])

Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2016-2016, None, None): 3, PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0]): 4 (the number of lines of pandas_code < 10) * 1 (the number of imported packages in pandas_code < 2) = 4. Summing these costs: 3+4=7.
Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2016-2016, None, None): 3, PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]): 4 (the number of lines of pandas_code < 10) * 1.5 (the number of imported packages in pandas_code is between 2 and 5) = 6. Summing these costs: 3+6=9.
Therefore, Modules1 is selected because it has a lower total interpretability cost of 7 compared to 9 for Modules2.

Best Modules: LoadDB(hupd, 2016-2016, None, None), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0])

To execute: LoadDB(hupd, 2016-2016, None, None)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "train_duration": "2016-2016", "test_duration": "None", "outcome_col": "None"
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
        'content': """To execute: PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0])""",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": "import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0]"
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
        'content': "{'month':12}"
    },
    {
        'role': 'assistant',
        'content': """To execute: Finish({'month':12}, month, integer)""",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'month':12}", "answer_variable": "month", "answer_type":"integer"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'Finish',
        'content': "12"
    },
    {
        'role': 'assistant',
        'content': 'The month with the highest number of patent applications in 2016 was December.'
    },
     {
        'role': 'user',
        'content': 'Train a model using patent applications from 2004 to 2006, and then use it to predict the decisions for patent applications filed in 2007.'
    },
    {
        'role': 'assistant',
        'content': """Modules1: LoadDB(hupd, 2004-2006, 2007-2007, decision), Classifier(logistic_regression, abstract, decision)
Modules2: LoadDB(hupd, 2004-2006, 2007-2007, decision), Classifier(distilbert-base-uncased, title, decision)
Modules3: LoadDB(hupd, 2004-2006, 2007-2007, decision), Classifier(naive_bayes, summary, decision)
Modules4: LoadDB(hupd, 2004-2006, 2007-2007, decision), Classifier(cnn, full_description, decision)

Thought: Total interpretability cost of Modules1 is calculated as follows: LoadDB(hupd, 2004-2006, 2007-2007, decision): 3, Classifier(logistic_regression, abstract, decision): 7 (model_name is "logistic_regression"). Summing these costs: 3+7=10.
Total interpretability cost of Modules2 is calculated as follows: LoadDB(hupd, 2004-2006, 2007-2007, decision): 3, Classifier(distilbert-base-uncased, title, decision): 10 (model_name is "logistic_regression"). Summing these costs: 3+10=13.
Total interpretability cost of Modules3 is calculated as follows: LoadDB(hupd, 2004-2006, 2007-2007, decision): 3, Classifier(naive_bayes, summary, decision): 
Total interpretability cost of Modules4 is calculated as follows: LoadDB(hupd, 2004-2006, 2007-2007, decision): 3, Classifier(cnn, full_description, decision):
Therefore, Modules1 is selected because it has a lower total interpretability cost of 10 compared to 13 for Modules2, for Modules 3, and for Modules 4. ###

Best Modules: LoadDB(hupd, 2004-2006, 2007-2007, decision), Classifier(logistic_regression, abstract, decision)

To execute: LoadDB(hupd, 2004-2006, 2007-2007, decision)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "train_duration": "2004-2006", "test_duration": "2007-2007", "outcome_col": "decision"
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
        'content': "We have successfully loaded the hupd dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 18011}) test: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'main_cpc_label', 'main_ipcr_label', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category', '__index_level_0__'], num_rows: 12008})})"
    },
    {
        'role': 'assistant',
        'content': """To execute: TextualClassifier(logistic_regression, abstract, decision)""",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'TextualClassifier',
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
        'content': "{'predictions': ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED',...]}"
    },
    {
        'role': 'assistant',
        'content': """To execute: Finish({'predictions': ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED',...]}, predictions, list)""",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'predictions': ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED']}", "answer_variable": "predictions", "answer_type": "list"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'Finish',
        'content': "['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED']"
    },
    {
        'role': 'assistant',
        'content': "The patent applications from 2007 are predicted to receive the following decisions: ['ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'ACCEPTED', 'REJECTED', 'REJECTED', 'ACCEPTED']."
    }
]




