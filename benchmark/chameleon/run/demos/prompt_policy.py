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
        'content': """Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)""", 
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
        'content': """Solution: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "[2016]"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
        'content': """We have successfully loaded the hupd dataframe, including the following columns: 'index'(e.g.23369, <class 'numpy.int64'>), 'patent_number'(e.g.9993790, <class 'numpy.int64'>), 'decision'(e.g.'ACCEPTED', <class 'str'>), 'title'(e.g.ANTIMICROB..., <class 'str'>), 'abstract'(e.g.Antimicrob..., <class 'str'>), 'claims'(e.g.1. An isol..., <class 'str'>), 'background'(e.g.<SOH> BACK..., <class 'str'>), 'summary'(e.g.<SOH> SUMM..., <class 'str'>), 'full_description'(e.g.CROSS-REFE..., <class 'str'>), 'filing_date'(e.g.2016-12-08 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'patent_issue_date'(e.g.2018-06-12 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'date_published'(e.g.2018-08-09 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'examiner_id'(e.g.9993790, <class 'numpy.int64'>), 'icpr_category'(e.g.A61, <class 'str'>), 'cpc_category'(e.g.A61, <class 'str'>).
It has 13491 rows."""
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
        'content': "Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.' Return either '2' or 'not 2'."
    },
    {
        'role': 'assistant',
        'content': """Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'TextualClassifier',
                    'arguments': json.dumps({
                        "database": "neurips", "model_name": "logistic_regression", "section": "Abstract", "text": "We propose a Bayesian encoder ... and has state-of-the-art predictive performance.", "target": "Poster Session", "one_v_all": "2"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'TextualClassifier',
        'content': "{'prediction': '2'}"
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
                        "variable_values": "{'prediction': '2'}", "answer_variable": "prediction", "answer_type": "string", "choices": "['2','not 2']"
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
        'content': "2"
    },
    {
        'role': 'assistant',
        'content': "This paper is predicted to be assigned to Poster Session 2."
    },
    {
        'role': 'user',
        'content': 'Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.'
    },
    {
        'role': 'assistant',
        'content': """Solution: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "list(range(2007,2010))"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
        'content': """We have successfully loaded the hupd dataframe, including the following columns: 'index'(e.g.0, <class 'numpy.int64'>), 'patent_number'(e.g.7869297, <class 'numpy.int64'>), 'decision'(e.g.'ACCEPTED', <class 'str'>), 'title'(e.g.Method for..., <class 'str'>), 'abstract'(e.g.A method f..., <class 'str'>), 'claims'(e.g.1. A metho..., <class 'str'>), 'background'(e.g.<SOH> BACK..., <class 'str'>), 'summary'(e.g.<SOH> SUMM..., <class 'str'>), 'full_description'(e.g.BACKGROUND..., <class 'str'>), 'filing_date'(e.g.2007-07-18 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'patent_issue_date'(e.g.2011-01-11 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'date_published'(e.g.2008-02-28 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'examiner_id'(e.g.7869297, <class 'numpy.int64'>), 'icpr_category'(e.g.B21, <class 'str'>), 'cpc_category'(e.g.H01, <class 'str'>).
It has 5714 rows."""
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
        'content': "Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'"
    },
    {
        'role': 'assistant',
        'content': """Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)""",
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
                'id': 'call_1',
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
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Finish',
        'content': "Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models"
    },
    {
        'role': 'assistant',
        'content': "A common theme that links these papers is 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'."
    }
]





messages_formula_1 = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 2
DBLoader: 3
PandasInterpreter: sqrt(Lines of Code) × max(Packages,1)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 6
"ARIMA": 8
TextualClassifier:
"logistic_regression": 7
"cnn": 15
"bert-base-uncased": 20
LLMInferencer: 30
Finish: 0
Accuracy cannot be sacrificed for interpretability. Examples are provided for tool mapping.
    """
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """Solution1: Calculator(0+0), Calculator(0+1), Calculator(0+1), Calculator(1+1), Calculator(1+2), Calculator(2+3), Calculator(3+5), Calculator(5+8), Calculator(8+13), Calculator(13+21), Calculator(21+34), Calculator(34+55), Calculator(55+89), Calculator(89+144), Calculator(144+233), Calculator(233+377), Calculator(377+610), Calculator(610+987), Calculator(987+1597), Calculator(1597+2584), Finish({'ans': 4181}, ans, integer)
Solution2: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)

Cost Analysis:
Solution1 Cost:
Calculator operations (20 * 2) + Finish: 40 + 0 = 40
Solution2 Cost:
PythonInterpreter: 3.32 (11 lines) * 1 (0 package) = 3.32
Finish: 0
Total: 3.32 + 0 = 3.32

Best Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)""",
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
        'content': """Solution1: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)
Solution2: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]), Finish({'month':12}, month, integer)

Cost Analysis:
Solution1 Cost:
DBLoader: 3
PandasInterpreter: 1.73 (3 lines) * 1 (1 package) = 1.73
Finish: 0
Total: 3 + 1.73 + 0 = 4.73
Solution2 Cost:
DBLoader: 3
PandasInterpreter: 2.24 (5 lines) * 1 (1 package) = 2.24
Finish: 0
Total: 3 + 2.24 + 0 = 5.24

Best Solution: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "[2016]"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
        'content': """We have successfully loaded the hupd dataframe, including the following columns: 'index'(e.g.23369, <class 'numpy.int64'>), 'patent_number'(e.g.9993790, <class 'numpy.int64'>), 'decision'(e.g.'ACCEPTED', <class 'str'>), 'title'(e.g.ANTIMICROB..., <class 'str'>), 'abstract'(e.g.Antimicrob..., <class 'str'>), 'claims'(e.g.1. An isol..., <class 'str'>), 'background'(e.g.<SOH> BACK..., <class 'str'>), 'summary'(e.g.<SOH> SUMM..., <class 'str'>), 'full_description'(e.g.CROSS-REFE..., <class 'str'>), 'filing_date'(e.g.2016-12-08 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'patent_issue_date'(e.g.2018-06-12 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'date_published'(e.g.2018-08-09 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'examiner_id'(e.g.9993790, <class 'numpy.int64'>), 'icpr_category'(e.g.A61, <class 'str'>), 'cpc_category'(e.g.A61, <class 'str'>).
It has 13491 rows."""
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
        'content': "Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.' Return either '2' or 'not 2'."
    },
    {
        'role': 'assistant',
        'content': """Solution1: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])
Solution2: TextualClassifier(neurips, bert-base-uncased, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])
Solution3: TextualClassifier(neurips, cnn, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])

Cost Analysis:
Solution1 Cost:
TextualClassifier (logistic_regression): 7
Finish: 0
Total: 7 + 0 = 7
Solution2 Cost:
TextualClassifier (bert-base-uncased): 20
Finish: 0
Total: 20 + 0 = 20
Solution3 Cost:
TextualClassifier (cnn): 15
Finish: 0
Total: 15 + 0 = 15

Best Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'TextualClassifier',
                    'arguments': json.dumps({
                        "database": "neurips", "model_name": "logistic_regression", "section": "Abstract", "text": "We propose a Bayesian encoder ... and has state-of-the-art predictive performance.", "target": "Poster Session", "one_v_all": "2"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'TextualClassifier',
        'content': "{'prediction': '2'}"
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
                        "variable_values": "{'prediction': '2'}", "answer_variable": "prediction", "answer_type": "string", "choices": "['2','not 2']"
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
        'content': "2"
    },
    {
        'role': 'assistant',
        'content': "This paper is predicted to be assigned to Poster Session 2."
    },
    {
        'role': 'user',
        'content': 'Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.'
    },
    {
        'role': 'assistant',
        'content': """Solution1: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\nmean_claims_per_year_list = df.groupby('year')['len_claims'].mean().tolist()\npred=sum(mean_claims_per_year_list)/len(mean_claims_per_year_list)\npreds=[pred]*(2011-2010+1)), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution2: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(ARIMA, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution3: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)

Cost Analysis:
Solution1 Cost:
DBLoader: 3
PandasInterpreter: 2.83 (8 lines) * 1 (1 package) = 2.83
Finish: 0
Total: 3 + 2.83 + 0 = 5.83
Solution2 Cost:
DBLoader: 3
PandasInterpreter: 2.24 (5 lines) * 1 (1 package) = 2.24
Forecaster (ARIMA): 8
Finish: 0
Total: 3 + 2.24 + 8 + 0 = 13.24
Solution3 Cost:
DBLoader: 3
PandasInterpreter: 2.24 (6 lines) * 1 (1 package) = 2.24
Forecaster (linear_regression): 6
Finish: 0
Total: 3 + 2.24 + 6 + 0 = 11.24

Accuracy Consideration: Solution1's approach assumes that the average length of claims remains constant from one year to the next, overlooking trends or changes over time. This static approach may result in less accurate predictions compared to methods that account for temporal variations.

Best Solution: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "list(range(2007,2010))"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
        'content': """We have successfully loaded the hupd dataframe, including the following columns: 'index'(e.g.0, <class 'numpy.int64'>), 'patent_number'(e.g.7869297, <class 'numpy.int64'>), 'decision'(e.g.'ACCEPTED', <class 'str'>), 'title'(e.g.Method for..., <class 'str'>), 'abstract'(e.g.A method f..., <class 'str'>), 'claims'(e.g.1. A metho..., <class 'str'>), 'background'(e.g.<SOH> BACK..., <class 'str'>), 'summary'(e.g.<SOH> SUMM..., <class 'str'>), 'full_description'(e.g.BACKGROUND..., <class 'str'>), 'filing_date'(e.g.2007-07-18 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'patent_issue_date'(e.g.2011-01-11 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'date_published'(e.g.2008-02-28 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'examiner_id'(e.g.7869297, <class 'numpy.int64'>), 'icpr_category'(e.g.B21, <class 'str'>), 'cpc_category'(e.g.H01, <class 'str'>).
It has 5714 rows."""
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
        'content': "Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'"
    },
    {
        'role': 'assistant',
        'content': """Solution1: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)
Solution2: PandasInterpreter(from collections import Counter\nmost_frequent_topic = Counter(df[df["Title"] == "4D Panoptic Scene Graph Generation"]["Topic"].str.split("/").values[0] + df[df["Title"] == "VoxDet: Voxel Learning for Novel Instance Detection"]["Topic"].str.split("/").values[0] + df[df["Title"] == "L2T-DLN: Learning to Teach with Dynamic Loss Network"]["Topic"].str.split("/").values[0]).most_common(1)[0][0]), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)

Cost Analysis:
Solution1 Cost:
LLMInferencer: 30
Finish: 0
Total: 30 + 0 = 30
Solution2 Cost:
PandasInterpreter: 2.65 (7 lines) * 1 (1 package) = 2.65
Finish: 0
Total: 2.65 + 0 = 2.65
Accuracy Consideration:
Modules2’s approach of using the most frequent topic as a common theme oversimplifies the problem and might not be as accurate or insightful. Solution1, despite its higher cost, uses an advanced inference model that could offer more precise results.

Best Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)""",
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
                'id': 'call_1',
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
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Finish',
        'content': "Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models"
    },
    {
        'role': 'assistant',
        'content': "A common theme that links these papers is 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'."
    }
]




messages_formula_2 = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 48
DBLoader: 47
PandasInterpreter: 50-sqrt(Lines of Code) × max(Packages,1)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 44
"ARIMA": 42
TextualClassifier:
"logistic_regression": 43
"cnn": 35
"bert-base-uncased": 30
LLMInferencer: 20
Finish: 0
Accuracy cannot be sacrificed for interpretability. Examples are provided for tool mapping.
    """
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """Solution1: Calculator(0+0), Calculator(0+1), Calculator(0+1), Calculator(1+1), Calculator(1+2), Calculator(2+3), Calculator(3+5), Calculator(5+8), Calculator(8+13), Calculator(13+21), Calculator(21+34), Calculator(34+55), Calculator(55+89), Calculator(89+144), Calculator(144+233), Calculator(233+377), Calculator(377+610), Calculator(610+987), Calculator(987+1597), Calculator(1597+2584), Finish({'ans': 4181}, ans, integer)
Solution2: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)

Cost Analysis:
Solution1 Cost:
Calculator operations (20 * 48) + Finish: 960 + 0 = 960
Solution2 Cost:
PythonInterpreter: 50 - 3.32 (11 lines) * 1 (0 package) = 46.68
Finish: 0
Total: 46.68 + 0 = 46.68

Best Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)""",
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
        'content': """Solution1: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)
Solution2: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]), Finish({'month':12}, month, integer)

Cost Analysis:
Solution1 Cost:
DBLoader: 47
PandasInterpreter: 50 - 1.73 (3 lines) * 1 (1 package) = 48.27
Finish: 0
Total: 47 + 48.27 + 0 = 95.27
Solution2 Cost:
DBLoader: 47
PandasInterpreter: 50 - 2.24 (5 lines) * 1 (1 package) = 47.76
Finish: 0
Total: 47 + 47.76 + 0 = 94.76

Best Solution: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]), Finish({'month':12}, month, integer)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "[2016]"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
        'content': """We have successfully loaded the hupd dataframe, including the following columns: 'index'(e.g.23369, <class 'numpy.int64'>), 'patent_number'(e.g.9993790, <class 'numpy.int64'>), 'decision'(e.g.'ACCEPTED', <class 'str'>), 'title'(e.g.ANTIMICROB..., <class 'str'>), 'abstract'(e.g.Antimicrob..., <class 'str'>), 'claims'(e.g.1. An isol..., <class 'str'>), 'background'(e.g.<SOH> BACK..., <class 'str'>), 'summary'(e.g.<SOH> SUMM..., <class 'str'>), 'full_description'(e.g.CROSS-REFE..., <class 'str'>), 'filing_date'(e.g.2016-12-08 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'patent_issue_date'(e.g.2018-06-12 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'date_published'(e.g.2018-08-09 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'examiner_id'(e.g.9993790, <class 'numpy.int64'>), 'icpr_category'(e.g.A61, <class 'str'>), 'cpc_category'(e.g.A61, <class 'str'>).
It has 13491 rows."""
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
from collections import Counter
df['filing_date'] = pd.to_datetime(df['filing_date'])
df['filing_month'] = df['filing_date'].apply(lambda x: x.month)
counter = Counter(df['filing_month'])
month = counter.most_common()[0][0]
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
        'content': "Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.' Return either '2' or 'not 2'."
    },
    {
        'role': 'assistant',
        'content': """Solution1: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2', 'not 2'])
Solution2: TextualClassifier(neurips, bert-base-uncased, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2', 'not 2'])
Solution3: TextualClassifier(neurips, cnn, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2', 'not 2'])

Cost Analysis:
Solution1 Cost:
TextualClassifier (logistic_regression): 43
Finish: 0
Total: 43 + 0 = 43
Solution2 Cost:
TextualClassifier (bert-base-uncased): 30
Finish: 0
Total: 30 + 0 = 30
Solution3 Cost:
TextualClassifier (cnn): 35
Finish: 0
Total: 35 + 0 = 35

Best Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2', 'not 2'])""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'TextualClassifier',
                    'arguments': json.dumps({
                        "database": "neurips", "model_name": "bert-base-uncased", "section": "Abstract", "text": "We propose a Bayesian encoder ... and has state-of-the-art predictive performance.", "target": "Poster Session", "one_v_all": "2"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'TextualClassifier',
        'content': "{'prediction': '2'}"
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
                        "variable_values": "{'prediction': '2'}", "answer_variable": "prediction", "answer_type": "string", "choices": "['2','not 2']"
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
        'content': "2"
    },
    {
        'role': 'assistant',
        'content': "This paper is predicted to be assigned to Poster Session 2."
    },
    {
        'role': 'user',
        'content': 'Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.'
    },
    {
        'role': 'assistant',
        'content': """Solution1: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\nmean_claims_per_year_list = df.groupby('year')['len_claims'].mean().tolist()\npred=sum(mean_claims_per_year_list)/len(mean_claims_per_year_list)\npreds=[pred]*(2011-2010+1)), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution2: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(ARIMA, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution3: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)

Cost Analysis:
Solution1 Cost:
DBLoader: 47
PandasInterpreter: 50 - 2.83 (8 lines) * 1 (1 package) = 47.17
Finish: 0
Total: 47 + 47.17 + 0 = 94.17
Solution2 Cost:
DBLoader: 47
PandasInterpreter: 50 - 2.24 (5 lines) * 1 (1 package) = 47.76
Forecaster (ARIMA): 42
Finish: 0
Total: 47 + 47.76 + 42 + 0 = 136.76
Solution3 Cost:
DBLoader: 47
PandasInterpreter: 50 - 2.45 (5 lines) * 1 (1 package) = 47.76
Forecaster (linear_regression): 44
Finish: 0
Total: 47 + 47.76 + 44 + 0 = 138.76

Accuracy Consideration: Solution1's approach assumes that the average length of claims remains constant from one year to the next, overlooking trends or changes over time. This static approach may result in less accurate predictions compared to methods that account for temporal variations.

Best Solution: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "list(range(2007,2010))"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
        'content': """We have successfully loaded the hupd dataframe, including the following columns: 'index'(e.g.0, <class 'numpy.int64'>), 'patent_number'(e.g.7869297, <class 'numpy.int64'>), 'decision'(e.g.'ACCEPTED', <class 'str'>), 'title'(e.g.Method for..., <class 'str'>), 'abstract'(e.g.A method f..., <class 'str'>), 'claims'(e.g.1. A metho..., <class 'str'>), 'background'(e.g.<SOH> BACK..., <class 'str'>), 'summary'(e.g.<SOH> SUMM..., <class 'str'>), 'full_description'(e.g.BACKGROUND..., <class 'str'>), 'filing_date'(e.g.2007-07-18 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'patent_issue_date'(e.g.2011-01-11 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'date_published'(e.g.2008-02-28 00:00:00, <class 'pandas._libs.tslibs.timestamps.Timestamp'>), 'examiner_id'(e.g.7869297, <class 'numpy.int64'>), 'icpr_category'(e.g.B21, <class 'str'>), 'cpc_category'(e.g.H01, <class 'str'>).
It has 5714 rows."""
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
                        "model_name": "ARIMA", "previous_data": [6194.2525, 5842.907314349525, 6151.568627450981], "forecast_len": 2
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
        'content': "{'forecast_predictions': [5870.034667489979, 6141.236787097841])}"
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
                        "variable_values":{'forecast_predictions': [5870.034667489979, 6141.236787097841]}, "answer_variable": 'forecast_predictions', "answer_type": "list"
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
        'content': "[5870.034667489979, 6141.236787097841]"
    },
    {
        'role': 'assistant',
        'content': "The predicted average length of claims for patent applications in 2010 and 2011 are respectively 5870 characters and 6141 characters."
    },
    {
        'role': 'user',
        'content': "Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'"
    },
    {
        'role': 'assistant',
        'content': """Solution1: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)
Solution2: PandasInterpreter(from collections import Counter\nmost_frequent_topic = Counter(df[df["Title"] == "4D Panoptic Scene Graph Generation"]["Topic"].str.split("/").values[0] + df[df["Title"] == "VoxDet: Voxel Learning for Novel Instance Detection"]["Topic"].str.split("/").values[0] + df[df["Title"] == "L2T-DLN: Learning to Teach with Dynamic Loss Network"]["Topic"].str.split("/").values[0]).most_common(1)[0][0]), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)

Cost Analysis:
Solution1 Cost:
LLMInferencer: 20
Finish: 0
Total: 20 + 0 = 20
Solution2 Cost:
PandasInterpreter: 50 - 2.65 (7 lines) * 1 (1 package) = 47.35
Finish: 0
Total: 47.35 + 0 = 47.35

Best Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)""",
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
                'id': 'call_1',
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
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Finish',
        'content': "Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models"
    },
    {
        'role': 'assistant',
        'content': "A common theme that links these papers is 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'."
    }
]




messages_formula_3 = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 2
DBLoader: 3
PandasInterpreter: (Lines of Code) × (Packages)
Lines of Code: <10 (4), 10-20 (10), 21-100 (15), >100 (20)
Packages: <2 (1), 2-5 (1.5), >5 (2)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 6
"ARIMA": 8
TextualClassifier:
"logistic_regression": 7
"cnn": 15
"bert-base-uncased": 20
LLMInferencer: 30
Finish: 0
Accuracy cannot be sacrificed for interpretability. Examples are provided for tool mapping.
    """
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """Solution1: Calculator(0+0), Calculator(0+1), Calculator(0+1), Calculator(1+1), Calculator(1+2), Calculator(2+3), Calculator(3+5), Calculator(5+8), Calculator(8+13), Calculator(13+21), Calculator(21+34), Calculator(34+55), Calculator(55+89), Calculator(89+144), Calculator(144+233), Calculator(233+377), Calculator(377+610), Calculator(610+987), Calculator(987+1597), Calculator(1597+2584), Finish({'ans': 4181}, ans, integer)
Solution2: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)

Cost Analysis:
Solution1 Cost:
Calculator operations (20 * 2) + Finish: 40 + 0 = 40
Solution2 Cost:
PythonInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 4 + 0 = 4

Best Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)""",
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
        'content': """Solution1: DBLoader(hupd, 2016-2016), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)
Solution2: DBLoader(hupd, 2016-2016), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]), Finish({'month':12}, month, integer)

Cost Analysis:
Solution1 Cost:
DBLoader: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 3 + 4 + 0 = 7
Solution2 Cost:
DBLoader: 3
PandasInterpreter: 4 (lines) * 1.5 (packages) = 6
Finish: 0
Total: 3 + 6 + 0 = 9

Best Solution: DBLoader(hupd, 2016-2016), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2016-2016"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
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
        'content': "Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.'"
    },
    {
        'role': 'assistant',
        'content': """Solution1: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)
Solution2: TextualClassifier(neurips, bert-base-uncased, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)
Solution3: TextualClassifier(neurips, cnn, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)

Cost Analysis:
Solution1 Cost:
TextualClassifier (logistic_regression): 7
Finish: 0
Total: 7 + 0 = 7
Solution2 Cost:
TextualClassifier (bert-base-uncased): 20
Finish: 0
Total: 20 + 0 = 20
Solution3 Cost:
TextualClassifier (cnn): 15
Finish: 0
Total: 15 + 0 = 15

Best Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'TextualClassifier',
                    'arguments': json.dumps({
                        "database": "neurips", "model_name": "logistic_regression", "section": "Abstract", "text": "We propose a Bayesian encoder ... and has state-of-the-art predictive performance.", "target": "Poster Session", "one_v_all": "2"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'TextualClassifier',
        'content': "{'prediction': '2'}"
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
                        "variable_values": "{'prediction': '2'}", "answer_variable": "prediction", "answer_type": "string"
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
        'content': "2"
    },
    {
        'role': 'assistant',
        'content': "This paper is predicted to be assigned to Poster Session 2."
    },
    {
        'role': 'user',
        'content': 'Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.'
    },
    {
        'role': 'assistant',
        'content': """Solution1: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\nmean_claims_per_year_list = df.groupby('year')['len_claims'].mean().tolist()\npred=sum(mean_claims_per_year_list)/len(mean_claims_per_year_list)\npreds=[pred]*(2011-2010+1)), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution2: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(ARIMA, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution3: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)

Cost Analysis:
Solution1 Cost:
DBLoader: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 3 + 4 + 0 = 7
Solution2 Cost:
DBLoader: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Forecaster (ARIMA): 8
Finish: 0
Total: 3 + 4 + 8 + 0 = 15
Solution3 Cost:
DBLoader: 3
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Forecaster (linear_regression): 6
Finish: 0
Total: 3 + 4 + 6 + 0 = 13

Accuracy Consideration: Solution1's approach assumes that the average length of claims remains constant from one year to the next, overlooking trends or changes over time. This static approach may result in less accurate predictions compared to methods that account for temporal variations.

Best Solution: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2007-2009"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
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
        'content': "Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'"
    },
    {
        'role': 'assistant',
        'content': """Solution1: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)
Solution2: PandasInterpreter(from collections import Counter\nmost_frequent_topic = Counter(df[df["Title"] == "4D Panoptic Scene Graph Generation"]["Topic"].str.split("/").values[0] + df[df["Title"] == "VoxDet: Voxel Learning for Novel Instance Detection"]["Topic"].str.split("/").values[0] + df[df["Title"] == "L2T-DLN: Learning to Teach with Dynamic Loss Network"]["Topic"].str.split("/").values[0]).most_common(1)[0][0]), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)

Cost Analysis:
Solution1 Cost:
LLMInferencer: 30
Finish: 0
Total: 30 + 0 = 30
Solution2 Cost:
PandasInterpreter: 4 (lines) * 1 (packages) = 4
Finish: 0
Total: 4 + 0 = 4
Accuracy Consideration:
Modules2’s approach of using the most frequent topic as a common theme oversimplifies the problem and might not be as accurate or insightful. Solution1, despite its higher cost, uses an advanced inference model that could offer more precise results.

Best Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)""",
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
                'id': 'call_1',
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
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Finish',
        'content': "Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models"
    },
    {
        'role': 'assistant',
        'content': "A common theme that links these papers is 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'."
    }
]




messages_formula_4 = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 48
DBLoader: 47
PandasInterpreter: 50 - (Lines of Code) × (Packages)
Lines of Code: <10 (4), 10-20 (10), 21-100 (15), >100 (20)
Packages: <2 (1), 2-5 (1.5), >5 (2)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 44
"ARIMA": 42
TextualClassifier:
"logistic_regression": 43
"cnn": 35
"bert-base-uncased": 30
LLMInferencer: 20
Finish: 0
Accuracy cannot be sacrificed for interpretability. Examples are provided for tool mapping.
    """
    },
    {
        'role': 'user',
        'content': 'What is the 20th Fibonacci number?'
    },
    {
        'role': 'assistant',
        'content': """Solution1: Calculator(0+0), Calculator(0+1), Calculator(0+1), Calculator(1+1), Calculator(1+2), Calculator(2+3), Calculator(3+5), Calculator(5+8), Calculator(8+13), Calculator(13+21), Calculator(21+34), Calculator(34+55), Calculator(55+89), Calculator(89+144), Calculator(144+233), Calculator(233+377), Calculator(377+610), Calculator(610+987), Calculator(987+1597), Calculator(1597+2584), Finish({'ans': 4181}, ans, integer)
Solution2: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)

Cost Analysis:
Solution1 Cost:
Calculator operations (20 * 48) + Finish: 960 + 0 = 960
Solution2 Cost:
PythonInterpreter: 50 - 4 (lines) * 1 (packages) = 46
Finish: 0
Total: 46 + 0 = 46

Best Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)""",
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
        'content': """Solution1: DBLoader(hupd, 2016-2016), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)
Solution2: DBLoader(hupd, 2016-2016), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]), Finish({'month':12}, month, integer)

Cost Analysis:
Solution1 Cost:
DBLoader: 47
PandasInterpreter: 50 - 4 (lines) * 1 (packages) = 46
Finish: 0
Total: 47 + 46 + 0 = 93
Solution2 Cost:
DBLoader: 47
PandasInterpreter: 50 - 4 (lines) * 1.5 (packages) = 44
Finish: 0
Total: 47 + 44 + 0 = 91

Best Solution: DBLoader(hupd, 2016-2016), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x
.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2016-2016"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
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
        'content': "Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.'"
    },
    {
        'role': 'assistant',
        'content': """Solution1: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)
Solution2: TextualClassifier(neurips, bert-base-uncased, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)
Solution3: TextualClassifier(neurips, cnn, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)

Cost Analysis:
Solution1 Cost:
TextualClassifier (logistic_regression): 43
Finish: 0
Total: 43 + 0 = 43
Solution2 Cost:
TextualClassifier (bert-base-uncased): 30
Finish: 0
Total: 30 + 0 = 30
Solution3 Cost:
TextualClassifier (cnn): 35
Finish: 0
Total: 35 + 0 = 35

Best Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'TextualClassifier',
                    'arguments': json.dumps({
                        "database": "neurips", "model_name": "logistic_regression", "section": "Abstract", "text": "We propose a Bayesian encoder ... and has state-of-the-art predictive performance.", "target": "Poster Session", "one_v_all": "2"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'TextualClassifier',
        'content': "{'prediction': '2'}"
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
                        "variable_values": "{'prediction': '2'}", "answer_variable": "prediction", "answer_type": "string"
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
        'content': "2"
    },
    {
        'role': 'assistant',
        'content': "This paper is predicted to be assigned to Poster Session 2."
    },
    {
        'role': 'user',
        'content': 'Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.'
    },
    {
        'role': 'assistant',
        'content': """Solution1: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\nmean_claims_per_year_list = df.groupby('year')['len_claims'].mean().tolist()\npred=sum(mean_claims_per_year_list)/len(mean_claims_per_year_list)\npreds=[pred]*(2011-2010+1)), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution2: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(ARIMA, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution3: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)

Cost Analysis:
Solution1 Cost:
DBLoader: 47
PandasInterpreter: 50 - 4 (lines) * 1 (packages) = 46
Finish: 0
Total: 47 + 46 + 0 = 93
Solution2 Cost:
DBLoader: 47
PandasInterpreter: 50 - 4 (lines) * 1 (packages) = 46
Forecaster (ARIMA): 42
Finish: 0
Total: 47 + 46 + 42 + 0 = 135
Solution3 Cost:
DBLoader: 47
PandasInterpreter: 50 - 4 (lines) * 1 (packages) = 46
Forecaster (linear_regression): 44
Finish: 0
Total: 47 + 46 + 44 + 0 = 137

Accuracy Consideration: Solution1's approach assumes that the average length of claims remains constant from one year to the next, overlooking trends or changes over time. This static approach may result in less accurate predictions compared to methods that account for temporal variations.

Best Solution: DBLoader(hupd, 2007-2009), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)""",
        'tool_calls': [
            {
                'id': 'call_0',
                'function': {
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2007-2009"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
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
        'content': "Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'"
    },
    {
        'role': 'assistant',
        'content': """Solution1: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)
Solution2: PandasInterpreter(from collections import Counter\nmost_frequent_topic = Counter(df[df["Title"] == "4D Panoptic Scene Graph Generation"]["Topic"].str.split("/").values[0] + df[df["Title"] == "VoxDet: Voxel Learning for Novel Instance Detection"]["Topic"].str.split("/").values[0] + df[df["Title"] == "L2T-DLN: Learning to Teach with Dynamic Loss Network"]["Topic"].str.split("/").values[0]).most_common(1)[0][0]), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)

Cost Analysis:
Solution1 Cost:
LLMInferencer: 20
Finish: 0
Total: 20 + 0 = 20
Solution2 Cost:
PandasInterpreter: 50 - 4 (lines) * 1 (packages) = 46
Finish: 0
Total: 46 + 0 = 46
Accuracy Consideration:
Modules2’s approach of using the most frequent topic as a common theme oversimplifies the problem and might not be as accurate or insightful. Solution1, despite its higher cost, uses an advanced inference model that could offer more precise results.

Best Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)""",
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
                'id': 'call_1',
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
        'tool_call_id': 'call_1',
        'role': 'tool',
        'name': 'Finish',
        'content': "Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models"
    },
    {
        'role': 'assistant',
        'content': "A common theme that links these papers is 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'."
    }
]




messages_subset = [
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
                    'name': 'DBLoader',
                    'arguments': json.dumps({
                        "target_db": "hupd", "duration": "2016-2016"
                    })
                },
                'type': 'function'
            }
        ]
    },
    {
        'tool_call_id': 'call_0',
        'role': 'tool',
        'name': 'DBLoader',
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
    }
]



# TEXT CHUNK
messages_no_example_text = [
    {
        'role': 'system',
        'content': """You need to act as a policy model, that given a question and a set of tools, determines the optimal sequence of tools that can be executed sequentially to answer the question as accurately as possible."""}
]

messages_one_example_text = [
    {
        'role': 'system',
        'content': """You need to act as a policy model, that given a question and a set of tools, determines the optimal sequence of tools that can be executed sequentially to answer the question as accurately as possible. Below is one example that maps the problem to the tools:

Question: What is the 20th Fibonacci number?
Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)"""}
]

messages_text = [
    {
        'role': 'system',
        'content': """You need to act as a policy model, that given a question and a set of tools, determines the optimal sequence of tools that can be executed sequentially to answer the question as accurately as possible. Below are some examples that map the problem to the tools:

Question: What is the 20th Fibonacci number?
Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)

Question: Which month had the highest number of patent applications in 2016?
Solution: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)

Question: Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.' Return either '2' or 'not 2'.
Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])

Question: Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.
Solution: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
      
Question: Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'
Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)"""}
]

messages_formula_1_no_example_text = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 2
DBLoader: 3
PandasInterpreter: sqrt(Lines of Code) × max(Packages,1)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 6
"ARIMA": 8
TextualClassifier:
"logistic_regression": 7
"cnn": 15
"bert-base-uncased": 20
LLMInferencer: 30
Finish: 0
Accuracy cannot be sacrificed for interpretability."""}
]

messages_formula_1_one_example_text = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 2
DBLoader: 3
PandasInterpreter: sqrt(Lines of Code) × max(Packages,1)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 6
"ARIMA": 8
TextualClassifier:
"logistic_regression": 7
"cnn": 15
"bert-base-uncased": 20
LLMInferencer: 30
Finish: 0
Accuracy cannot be sacrificed for interpretability. Below is one example that maps the problem to the tools:

Question: What is the 20th Fibonacci number?
Solution1: Calculator(0+0), Calculator(0+1), Calculator(0+1), Calculator(1+1), Calculator(1+2), Calculator(2+3), Calculator(3+5), Calculator(5+8), Calculator(8+13), Calculator(13+21), Calculator(21+34), Calculator(34+55), Calculator(55+89), Calculator(89+144), Calculator(144+233), Calculator(233+377), Calculator(377+610), Calculator(610+987), Calculator(987+1597), Calculator(1597+2584), Finish({'ans': 4181}, ans, integer)
Solution1 Cost: Calculator operations (20 * 2) + Finish = 40
Solution2: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)
Solution2 Cost: PythonInterpreter: 3.32 (11 lines) * 1 (0 package) + Finish = 3.32
Best Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)"""}
]

messages_formula_1_text = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 2
DBLoader: 3
PandasInterpreter: sqrt(Lines of Code) × max(Packages,1)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 6
"ARIMA": 8
TextualClassifier:
"logistic_regression": 7
"cnn": 15
"bert-base-uncased": 20
LLMInferencer: 30
Finish: 0
Accuracy cannot be sacrificed for interpretability. Below are some examples that map the problem to the tools:

Question: What is the 20th Fibonacci number?
Solution1: Calculator(0+0), Calculator(0+1), Calculator(0+1), Calculator(1+1), Calculator(1+2), Calculator(2+3), Calculator(3+5), Calculator(5+8), Calculator(8+13), Calculator(13+21), Calculator(21+34), Calculator(34+55), Calculator(55+89), Calculator(89+144), Calculator(144+233), Calculator(233+377), Calculator(377+610), Calculator(610+987), Calculator(987+1597), Calculator(1597+2584), Finish({'ans': 4181}, ans, integer)
Solution1 Cost: Calculator operations (20 * 2) + Finish = 40
Solution2: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)
Solution2 Cost: PythonInterpreter: 3.32 (11 lines) * 1 (0 package) + Finish = 3.32
Best Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)

Question: Which month had the highest number of patent applications in 2016?
Solution1: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)
Solution1 Cost: DBLoader: 3 + PandasInterpreter: 1.73 (3 lines) * 1 (1 package) + Finish = 4.73
Solution2: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\nfrom collections import Counter\ndf['filing_month'] = df['filing_date'].apply(lambda x.month)\ncounter = Counter(df['filing_month'])\nmonth = counter.most_common()[0][0]), Finish({'month':12}, month, integer)
Solution2 Cost: DBLoader: 3 + PandasInterpreter: 2.24 (5 lines) * 1 (1 package) + Finish = 5.24
Best Solution: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)

Question: Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.' Return either '2' or 'not 2'.
Solution1: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])
Solution1 Cost: TextualClassifier (logistic_regression): 7 + Finish = 7
Solution2: TextualClassifier(neurips, bert-base-uncased, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])
Solution2 Cost: TextualClassifier (bert-base-uncased): 20 + Finish = 20
Solution3: TextualClassifier(neurips, cnn, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])
Solution3 Cost: TextualClassifier (cnn): 15 + Finish = 15
Best Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])

Question: Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.
Solution1: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\nmean_claims_per_year_list = df.groupby('year')['len_claims'].mean().tolist()\npred=sum(mean_claims_per_year_list)/len(mean_claims_per_year_list)\npreds=[pred]*(2011-2010+1)), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution1 Cost: DBLoader: 3 + PandasInterpreter: 2.83 (8 lines) * 1 (1 package) + Finish = 5.83
Solution2: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(ARIMA, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution2 Cost: DBLoader: 3 + PandasInterpreter: 2.24 (5 lines) * 1 (1 package) + Forecaster (ARIMA): 8 + Finish: 0 = 13.24
Solution3: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
Solution3 Cost: DBLoader: 3 + PandasInterpreter: 2.24 (6 lines) * 1 (1 package) + Forecaster (linear_regression): 6 + Finish = 11.24
Accuracy Consideration: Solution1's approach assumes that the average length of claims remains constant from one year to the next, overlooking trends or changes over time. This static approach may result in less accurate predictions compared to methods that account for temporal variations.
Best Solution: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
      
Question: Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'
Solution1: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)
Solution1 Cost: LLMInferencer: 30 + Finish = 30
Solution2: PandasInterpreter(from collections import Counter\nmost_frequent_topic = Counter(df[df["Title"] == "4D Panoptic Scene Graph Generation"]["Topic"].str.split("/").values[0] + df[df["Title"] == "VoxDet: Voxel Learning for Novel Instance Detection"]["Topic"].str.split("/").values[0] + df[df["Title"] == "L2T-DLN: Learning to Teach with Dynamic Loss Network"]["Topic"].str.split("/").values[0]).most_common(1)[0][0]), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)
Solution2 Cost: PandasInterpreter: 2.65 (7 lines) * 1 (1 package) + Finish = 2.65
Accuracy Consideration: Modules2’s approach of using the most frequent topic as a common theme oversimplifies the problem and might not be as accurate or insightful. Solution1, despite its higher cost, uses an advanced inference model that could offer more precise results.
Best Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)"""}
]

messages_examples_formula_1_text = [
    {
        'role': 'system',
        'content': """Act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:
1.Generate Solutions: List 2-4 sequences of tools that can solve the question.
2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.
3.Execute the Lowest Cost Solution.
Interpretability Costs:
Calculator: 2
DBLoader: 3
PandasInterpreter: sqrt(Lines of Code) × max(Packages,1)
PythonInterpreter: Same as PandasInterpreter
Forecaster:
"linear_regression": 6
"ARIMA": 8
TextualClassifier:
"logistic_regression": 7
"cnn": 15
"bert-base-uncased": 20
LLMInferencer: 30
Finish: 0
Accuracy cannot be sacrificed for interpretability. Below are some examples that map the problem to the tools:

Question: What is the 20th Fibonacci number?
Solution: PythonInterpreter(def solution(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\nans = solution(19)\n), Finish({'ans': 4181}, ans, integer)

Question: Which month had the highest number of patent applications in 2016?
Solution: DBLoader(hupd, [2016]), PandasInterpreter(import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x.month)\nmonth = df['filing_month'].mode()[0]), Finish({'month':12}, month, integer)

Question: Determine if a NeurIPS paper, based on the following abstract, is assigned to Poster Session 2: 'We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.' Return either '2' or 'not 2'.
Solution: TextualClassifier(neurips, logistic_regression, Abstract, We propose a Bayesian encoder ... and has state-of-the-art predictive performance, Poster Session, 2), Finish({'predictions': '2'}, predictions, string, ['2','not 2'])

Question: Using the patent applications from 2007 to 2009, predict the average length of claims for patent applications in 2010 and 2011.
Solution: DBLoader(hupd, list(range(2007,2010))), PandasInterpreter(import pandas as pd\ndf['year'] = df['filing_date'].dt.year\ndf['len_claims'] = df['claims'].apply(len)\naverage_claims_per_year = df.groupby('year')['len_claims'].mean())), Forecaster(linear_regression, previous_data, 2), Finish({'forecast_predictions': [6020.225608051151, 5998.883671776641]}, forecast_predictions, list)
      
Question: Identify a common theme that links the NeurIPS papers titled '4D Panoptic Scene Graph Generation,' 'VoxDet: Voxel Learning for Novel Instance Detection,' and 'L2T-DLN: Learning to Teach with Dynamic Loss Network.'
Solution: LLMInferencer(), Finish({'ans': 'Advanced Techniques for 3D Scene Understanding and Adaptive Learning Models'}, ans, string)"""}
]