import json

messages = [
    {
        'role': 'system',
        'content': """You need to act as a policy model, that given a question and a set of tools, determines the optimal sequence of tools that can be executed sequentially to answer the question as accurately as possible. Below are some examples that map the problem to the tools."""
    },
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
""", 
                        "input_var_values": "{}"
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
        'content': "",
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
        'content': "",
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
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": """
import pandas as pd
df['filing_month'] = df['filing_date'].apply(lambda x: x.month)
month = df['filing_month'].mode()[0]
"""

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
        'content': "",
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
        'content': 'Train a model using the first 2000 rows of NeurIPS papers dataset and then use it to predict the poster sessions for the papers with unique indices ID-2001,ID-2500,ID-2486,ID-2759,ID-3300.'
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "neurips", "train_duration": "0-2000", "test_duration": "2001-3585", "outcome_col": "Poster Session"
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
        'content': "We have successfully loaded the neurips dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['Title', 'Authors', 'Location', 'Abstract', 'Topic', 'Oral', 'Poster Session', '__index_level_0__'],num_rows: 1981}) test: Dataset({features: ['Title', 'Authors', 'Location', 'Abstract', 'Topic', 'Oral', '__index_level_0__'],num_rows: 1571})})"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'TestSampler',
                    'arguments': json.dumps({
                        "indices": "ID-2001,ID-2500,ID-2486,ID-2759,ID-3300"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'TestSampler',
        'content': "Done sampling the test set according to the specified unique indices."
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'TextualClassifier',
                    'arguments': json.dumps({
                        "model_name": "logistic_regression", "section": "Abstract", "target": "Poster Session", "one_v_all": "None"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'TextualClassifier',
        'content': "{'predictions': [4.0, 2.0, 6.0, 2.0, 4.0]}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_3',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'predictions': [4.0, 2.0, 6.0, 2.0, 4.0]}", "answer_variable": "predictions", "answer_type": "list"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_3',
        'role': 'tool',
        'name': 'Finish',
        'content': "[4.0, 2.0, 6.0, 2.0, 4.0]"
    },
    {
        'role': 'assistant',
        'content': "The predicted poster sessions for the remaining papers in the dataset are: 4.0, 2.0, 6.0, 2.0, and 4.0."
    },
    {
        'role': 'user',
        'content': 'Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.'
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "train_duration": "2007-2009", "test_duration": "2010-2011", "outcome_col": "claims"
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
        'content': "We have successfully loaded the hupd dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category'],num_rows: 5713}) test: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'filing_date', 'patent_issue_date', 'date_published', 'icpr_category', 'cpc_category'],num_rows: 1110})})"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": """
import pandas as pd
df['year'] = df['filing_date'].dt.year
df['len_claims'] = df['claims'].apply(len)
average_claims_per_year = df.groupby('year')['len_claims'].mean()
"""
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
        'content': "{'average_claims_per_year': {2007: 6194.2525, 2008: 5842.907314349525, 2009: 6151.568627450981}}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Forecaster',
                    'arguments': json.dumps({
                        "model_name": "linear_regression", "previous_data": [6194.2525, 5842.907314349525, 6151.568627450981], "forecast_len": 2
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'Forecaster',
        'content': "{'forecast_predictions': [6020.225608051151, 5998.883671776641])}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values":{'forecast_predictions': [6020.225608051151, 5998.883671776641]}, "answer_variable": 'forecast_predictions', "answer_type": "list"
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
        'content': "[6020.225608051151, 5998.883671776641]"
    },
    {
        'role': 'assistant',
        'content': "The predicted average length of claims for patent applications in 2010 and 2011 are respectively 6020 characters and 5999 characters."
    },
    {
        'role': 'user',
        'content': "I have 2 documents. Document 1: 'Machine learning is a specialized branch of artificial intelligence.' Document 2: 'Artificial intelligence is a broad field that encompasses various technologies.' Which of these two is more relevant to the search query 'machine learning'?"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'TFIDF',
                    'arguments': json.dumps({
                        "query": "machine learning", "document": "Machine learning is a specialized branch of artificial intelligence." 
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'TFIDF',
        'content': "{'match':1}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'TFIDF',
                    'arguments': json.dumps({
                        "query": "machine learning", "document": "Artificial intelligence is a broad field that encompasses various technologies." 
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'TFIDF',
        'content': "{'match':0}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'PythonInterpreter',
                    'arguments': json.dumps({
                        "python_code": """
def get_most_relevant_document(match_doc1, match_doc2):
    if match_doc1 > match_doc2:
        return 'Document 1'
    elif match_doc2 > match_doc1:
        return 'Document 2'
    else:
        return 'Both documents are equally relevant'
most_relevant = get_most_relevant_document(match1, match2)
""", 
                        "input_var_values": "{'match1': 1, 'match2': 0}"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'PythonInterpreter',
        'content': "{'most_relevant': 'Document 1'}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_3',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'most_relevant': 'Document 1'}", "answer_variable": "most_relevant", "answer_type": "string"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_3',
        'role': 'tool',
        'name': 'Finish',
        'content': "Document 1"
    },
    {
        'role': 'assistant',
        'content': "Document 1 is more relevant to the search query 'machine learning'."
    },
    {
        'role': 'user',
        'content': "Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LLMInferencer',
                    'arguments': ""
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'LLMInferencer',
        'content': "Provide a direct answer to the question without using any extra tools. Format your response as a dictionary with the key 'ans' and place your answer inside the dictionary, like this: {'ans': your_answer}. Nest, call the Finish tool with this dictionary as variable_values."
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_3',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}", "answer_variable": "ans", "answer_type": "string"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_3',
        'role': 'tool',
        'name': 'Finish',
        'content': "Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models"
    },
    {
        'role': 'assistant',
        'content': "A common theme that links these papers is 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'."
    }
]


messages_formula = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculate: 2
TestSampler: 2
LoadDB: 3
TFIDF: 5
PandasInterpreter: (Lines of Code) × (Packages)
Lines of Code: <10 (4), 10-20 (10), 21-100 (15), >100 (20)
Packages: <2 (1), 2-5 (1.5), >5 (2)
PythonInterpreter: Same as PandasInterpreter
TextualClassifier:
"logistic_regression": 7
"naive_bayes": 8
"cnn": 15
"distilbert-base-uncased": 20
Forecaster:
"linear_regression": 6
"ARIMA": 8
LLMInferencer: 30
Accuracy cannot be sacrificed for interpretability. Examples are provided for tool mapping.
    """
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """Modules1: Calculate(0+0), Calculate(0+1), Calculate(0+1), Calculate(1+1), Calculate(1+2), Calculate(2+3), Calculate(3+5), Calculate(5+8), Calculate(8+13), Calculate(13+21), Calculate(21+34), Calculate(34+55), Calculate(55+89), Calculate(89+144), Calculate(144+233), Calculate(233+377), Calculate(377+610), Calculate(610+987), Calculate(987+1597), Calculate(1597+2584), Finish(..., ..., integer)
Modules2: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish(..., ..., integer)

Cost Analysis:
Modules1 Cost:
Calculate operations (20 * 2) + Finish: 40 + 0 = 40
Modules2 Cost:
PythonInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 4 + 0 = 4

Best Modules: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish(..., ..., integer)""",
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
""", 
                        "input_var_values": "{}"
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
        'content': "",
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
        'content': """Modules1: LoadDB(hupd, 2016-2016, None, None), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish(..., ..., integer)
Modules2: LoadDB(hupd, 2016-2016, None, None), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]), Finish(..., ..., integer)

Cost Analysis:
Modules1 Cost:
LoadDB: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 3 + 4 + 0 = 7
Modules2 Cost:
LoadDB: 3
PandasInterpreter: 4 (lines) * 1.5 (packages) = 6
Finish: 0
Total: 3 + 6 + 0 = 9

Best Modules: LoadDB(hupd, 2016-2016, None, None), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish(..., ..., integer)""",
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
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": """
import pandas as pd
df['filing_month'] = df['filing_date'].apply(lambda x: x.month)
month = df['filing_month'].mode()[0]
"""
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
        'content': "",
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
        'content': 'Train a model using the first 2000 rows of NeurIPS papers dataset and then use it to predict the poster sessions for the papers with unique indices ID-2001,ID-2500,ID-2486,ID-2759,ID-3300.'
    },
    {
        'role': 'assistant',
        'content': """Modules1: LoadDB(neurips, 0-2000, 2001-3585, Poster Session), TestSampler(ID-2001,ID-2500,ID-2486,ID-2759,ID-3300), TextualClassifier(logistic_regression, Abstract, Poster Session), Finish(..., ..., list)
Modules2: LoadDB(neurips, 0-2000, 2001-3585, Poster Session), TestSampler(ID-2001,ID-2500,ID-2486,ID-2759,ID-3300), TextualClassifier(distilbert-base-uncased, Abstract, Poster Session), Finish(..., ..., list)
Modules3: LoadDB(neurips, 0-2000, 2001-3585, Poster Session), TestSampler(ID-2001,ID-2500,ID-2486,ID-2759,ID-3300), TextualClassifier(naive_bayes, Title, Poster Session), Finish(..., ..., list)
Modules4: LoadDB(neurips, 0-2000, 2001-3585, Poster Session), TestSampler(ID-2001,ID-2500,ID-2486,ID-2759,ID-3300), TextualClassifier(cnn, Topic, Poster Session), Finish(..., ..., list)

Cost Analysis:
Modules1 Cost:
LoadDB: 3
TestSampler: 2
TextualClassifier (logistic_regression): 7
Finish: 0
Total: 3 + 2 + 7 + 0 = 12
Modules2 Cost:
LoadDB: 3
TestSampler: 2
TextualClassifier (distilbert-base-uncased): 20
Finish: 0
Total: 3 + 2 + 20 + 0 = 25
Modules3 Cost:
LoadDB: 3
TestSampler: 2
TextualClassifier (naive_bayes): 8
Finish: 0
Total: 3 + 2 + 8 + 0 = 13
Modules4 Cost:
LoadDB: 3
TestSampler: 2
TextualClassifier (cnn): 15
Finish: 0
Total: 3 + 2 + 15 + 0 = 20

Best Modules: LoadDB(neurips, 0-2000, 2001-3585, Poster Session), TestSampler(ID-2001,ID-2500,ID-2486,ID-2759,ID-3300), TextualClassifier(logistic_regression, Abstract, Poster Session), Finish(..., ..., list)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "neurips", "train_duration": "0-2000", "test_duration": "2001-3585", "outcome_col": "Poster Session"
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
        'content': "We have successfully loaded the neurips dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['Title', 'Authors', 'Location', 'Abstract', 'Topic', 'Oral', 'Poster Session', '__index_level_0__'],num_rows: 1981}) test: Dataset({features: ['Title', 'Authors', 'Location', 'Abstract', 'Topic', 'Oral', '__index_level_0__'],num_rows: 1571})})"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'TestSampler',
                    'arguments': json.dumps({
                        "indices": "ID-2001,ID-2500,ID-2486,ID-2759,ID-3300"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'TestSampler',
        'content': "Done sampling the test set according to the specified unique indices."
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'TextualClassifier',
                    'arguments': json.dumps({
                        "model_name": "logistic_regression", "section": "Abstract", "target": "Poster Session", "one_v_all": "None"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'TextualClassifier',
        'content': "{'predictions': [4.0, 2.0, 6.0, 2.0, 4.0]}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_3',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'predictions': [4.0, 2.0, 6.0, 2.0, 4.0]}", "answer_variable": "predictions", "answer_type": "list"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_3',
        'role': 'tool',
        'name': 'Finish',
        'content': "[4.0, 2.0, 6.0, 2.0, 4.0]"
    },
    {
        'role': 'assistant',
        'content': "The predicted poster sessions for the remaining papers in the dataset are: 4.0, 2.0, 6.0, 2.0, and 4.0."
    },
    {
        'role': 'user',
        'content': 'Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.'
    },
    {
        'role': 'assistant',
        'content': """Modules1: LoadDB(hupd, 2007-2009, 2010-2011, claims), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\nmean_claims_per_year_list = df.groupby('year')['len_claims'].mean().tolist()\npred=sum(mean_claims_per_year_list)/len(mean_claims_per_year_list)\npreds=[pred]*(2011-2010+1)), Finish(..., ..., list)
Modules2: LoadDB(hupd, 2007-2009, 2010-2011, claims), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(ARIMA, previous_data, 2), Finish(..., ..., list)
Modules3: LoadDB(hupd, 2007-2009, 2010-2011, claims), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish(..., ..., list)

Cost Analysis:
Modules1 Cost:
LoadDB: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 3 + 4 + 0 = 7
Modules2 Cost:
LoadDB: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Forecaster (ARIMA): 8
Finish: 0
Total: 3 + 4 + 8 + 0 = 15
Modules3 Cost:
LoadDB: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Forecaster (linear_regression): 6
Finish: 0
Total: 3 + 4 + 6 + 0 = 13

Reasoning: Modules1's approach assumes that the average length of claims remains constant from one year to the next, overlooking trends or changes over time. This static approach may result in less accurate predictions compared to methods that account for temporal variations.

Best Modules: LoadDB(hupd, 2007-2009, 2010-2011, claims), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish(..., ..., list)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LoadDB',
                    'arguments': json.dumps({
                        "target_db": "hupd", "train_duration": "2007-2009", "test_duration": "2010-2011", "outcome_col": "claims"
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
        'content': "We have successfully loaded the hupd dataset dict that has the following structure: DatasetDict({train: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'filing_date', 'patent_issue_date', 'date_published', 'examiner_id', 'icpr_category', 'cpc_category'],num_rows: 5713}) test: Dataset({features: ['patent_number', 'decision', 'title', 'abstract', 'claims', 'background', 'summary', 'full_description', 'filing_date', 'patent_issue_date', 'date_published', 'icpr_category', 'cpc_category'],num_rows: 1110})})"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'PandasInterpreter',
                    'arguments': json.dumps({
                        "pandas_code": """
import pandas as pd
df['year'] = df['filing_date'].dt.year
df['len_claims'] = df['claims'].apply(len)
average_claims_per_year = df.groupby('year')['len_claims'].mean()
"""
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
        'content': "{'average_claims_per_year': {2007: 6194.2525, 2008: 5842.907314349525, 2009: 6151.568627450981}}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Forecaster',
                    'arguments': json.dumps({
                        "model_name": "linear_regression", "previous_data": [6194.2525, 5842.907314349525, 6151.568627450981], "forecast_len": 2
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'Forecaster',
        'content': "{'forecast_predictions': [6020.225608051151, 5998.883671776641])}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values":{'forecast_predictions': [6020.225608051151, 5998.883671776641]}, "answer_variable": 'forecast_predictions', "answer_type": "list"
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
        'content': "[6020.225608051151, 5998.883671776641]"
    },
    {
        'role': 'assistant',
        'content': "The predicted average length of claims for patent applications in 2010 and 2011 are respectively 6020 characters and 5999 characters."
    },
    {
        'role': 'user',
        'content': "I have 2 documents. Document 1: 'Machine learning is a specialized branch of artificial intelligence.' Document 2: 'Artificial intelligence is a broad field that encompasses various technologies.' Which of these two is more relevant to the search query 'machine learning'?"
    },
    {
        'role': 'assistant',
        'content': """Modules1: PythonInterpreter(from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.metrics.pairwise import cosine_similarity\n\ndef get_most_relevant_document(query, doc1, doc2):\n vectorizer = TfidfVectorizer()\n tfidf_matrix = vectorizer.fit_transform([query, doc1, doc2])\n similarity_doc1 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]\n similarity_doc2 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0][0]\n if similarity_doc1 > similarity_doc2:\n return 'Document 1'\n elif similarity_doc2 > similarity_doc1:\n return 'Document 2'\n else:\n return 'Both documents are equally relevant'\n\nquery = 'machine learning'\ndoc1 = 'Machine learning is a specialized branch of artificial intelligence.'\ndoc2 = 'Artificial intelligence is a broad field that encompasses various technologies.'\n\nmost_relevant = get_most_relevant_document(query, doc1, doc2)), Finish(..., ..., string)
Modules2: TFIDF(machine learning, Machine learning is a specialized branch of artificial intelligence.), TFIDF(machine learning, Artificial intelligence is a broad field that encompasses various technologies.), PythonInterpreter(def get_most_relevant_document(match_doc1, match_doc2):\n if match_doc1 > match_doc2:\n return 'Document 1'\n elif match_doc2 > match_doc1:\n return 'Document 2'\n else:\n return 'Both documents are equally relevant'\nmost_relevant = get_most_relevant_document(1, 0)), Finish(..., ..., string)
Modules3: LLMInferencer(), Finish(..., ..., string)

Cost Analysis:
Modules1 Cost:
PythonInterpreter: 10 (lines) * 1.5 (packages) = 15
Finish: 0
Total: 15 + 0 = 15
Modules2 Cost:
TFIDF (2x): 5 + 5 = 10
PythonInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 10 + 4 = 14
Modules3 Cost:
LLMInferencer: 30
Finish: 0
Total: 30 + 0 = 30

Best Modules: TFIDF(machine learning, Machine learning is a specialized branch of artificial intelligence.), TFIDF(machine learning, Artificial intelligence is a broad field that encompasses various technologies.), PythonInterpreter(def get_most_relevant_document(match_doc1, match_doc2):\n if match_doc1 > match_doc2:\n return 'Document 1'\n elif match_doc2 > match_doc1:\n return 'Document 2'\n else:\n return 'Both documents are equally relevant'\nmost_relevant = get_most_relevant_document(1, 0))""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'TFIDF',
                    'arguments': json.dumps({
                        "query": "machine learning", "document": "Machine learning is a specialized branch of artificial intelligence." 
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'TFIDF',
        'content': "{'match':1}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_1',
                'function': {
                    'name': 'TFIDF',
                    'arguments': json.dumps({
                        "query": "machine learning", "document": "Artificial intelligence is a broad field that encompasses various technologies." 
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'TFIDF',
        'content': "{'match':0}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_2',
                'function': {
                    'name': 'PythonInterpreter',
                    'arguments': json.dumps({
                        "python_code": """
def get_most_relevant_document(match_doc1, match_doc2):
    if match_doc1 > match_doc2:
        return 'Document 1'
    elif match_doc2 > match_doc1:
        return 'Document 2'
    else:
        return 'Both documents are equally relevant'
most_relevant = get_most_relevant_document(match1, match2)
""", 
                        "input_var_values": "{'match1':1, 'match2':0}"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_2',
        'role': 'tool',
        'name': 'PythonInterpreter',
        'content': "{'most_relevant': 'Document 1'}"
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_3',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'most_relevant': 'Document 1'}", "answer_variable": "most_relevant", "answer_type": "string"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_3',
        'role': 'tool',
        'name': 'Finish',
        'content': "Document 1"
    },
    {
        'role': 'assistant',
        'content': "Document 1 is more relevant to the search query 'machine learning'."
    },
    {
        'role': 'user',
        'content': "Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'"
    },
    {
        'role': 'assistant',
        'content': """Modules1: LLMInferencer(), Finish(..., ..., string)
Modules2: PandasInterpreter(from collections import Counter\nmost_frequent_topic = Counter(df[df["Title"] == "4D Panoptic Scene Graph Generation"]["Topic"].str.split("/").values[0] + df[df["Title"] == "VoxDet: Voxel Learning for Novel Instance Detection"]["Topic"].str.split("/").values[0] + df[df["Title"] == "L2T-DLN: Learning to Teach with Dynamic Loss Network"]["Topic"].str.split("/").values[0]).most_common(1)[0][0]), Finish(..., ..., string)

Cost Analysis:
Modules1 Cost:
LLMInferencer: 30
Finish: 0
Total: 30 + 0 = 30
Modules2 Cost:
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 4 + 0 = 4
Reasoning:
Modules2’s approach of using the most frequent topic as a common theme oversimplifies the problem and might not be as accurate or insightful. Modules1, despite its higher cost, uses an advanced inference model that could offer more precise results.

Best Modules: LLMInferencer(), Finish(..., ..., string)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'LLMInferencer',
                    'arguments': ""
                },

                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'LLMInferencer',
        'content': "Provide a direct answer to the question without using any extra tools. Format your response as a dictionary with the key 'ans' and place your answer inside the dictionary, like this: {'ans': your_answer}. Nest, call the Finish tool with this dictionary as variable_values." 
    },
    {
        'role': 'assistant',
        'content': "",
        'tool_calls': [
            {
                'id': 'call_3',
                'function': {
                    'name': 'Finish',
                    'arguments': json.dumps({
                        "variable_values": "{'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}", "answer_variable": "ans", "answer_type": "string"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_3',
        'role': 'tool',
        'name': 'Finish',
        'content': "Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models"
    },
    {
        'role': 'assistant',
        'content': "A common theme that links these papers is 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'."
    }
]




