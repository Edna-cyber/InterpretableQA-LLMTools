
prompt_header_clean = """
You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows:

- Calculate[formula]: This module calculates a given formula and returns the result. It takes in a mathematical formula and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB[DBName; subsetNames]: This module loads a database specified by the database name and subset names, and returns the loaded database. It accepts a database name and subset names, returning the corresponding database. The DBName can be one of the following: hupd. Normally, we consider using "LoadDB" only when the question requires data from a specific structured dataset.

- AutoLoadDB[DBName; trainStartDate; trainEndDate; validationStartDate; validationEndDate]: This module loads a database specified by the database name and date ranges for training and validation. It directly loads the database from Hugging Face and separates it into training and validation subsets based on the provided date ranges. Normally, we only consider using "AutoLoadDB" when the question specifies the training and validation sets or needs to be solved through a machine learning algorithm. 

- TargetFilter[targetColumn; filterCondition]: This module modifies a database in place by removing the rows that don't satisfy the filter condition. It accepts a target column and a filter condition, and the default filter condition is "not NA." Example conditions include "not NA," "keep ACCEPT,REJECT," and "remove 0,1." We always use "TargetFilter" after loading the database with either "LoadDB" or "AutoLoadDB".

- PandasInterpreter[Python, split]: This module interprets Pandas code written in Python, executes it on a dataframe specified by split, and returns the result. It takes in Python code and a dataframe specified by split, and returns the result of the code execution. Choices for split are "all", "train", or "validation". Normally, we only consider using "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataset.

- PythonInterpreter[Python]: This module interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only consider using "PythonInterpreter" when the question requires complex computations or custom data manipulation.

- LogisticRegression[Python]: This module applies a logistic regression model using Python code and returns the result. It takes in Python code to define and fit a logistic regression model on a specified dataset, returning the model's predictions. Normally, we only consider using "LogisticRegression" when the question involves binary classification tasks.

- DistilbertBaseUncased[Python]:  This module applies the DistilBERT base uncased model using Python code and returns the result. It takes in Python code to load, fine-tune, and evaluate the DistilBERT base uncased model on a specified dataset, returning the model's predictions. Normally, we consider using "DistilbertBaseUncased" when the question involves natural language processing tasks such as text classification, sentiment analysis, or named entity recognition.

- Finish[answer]: This module returns the final answer and finishes the task. This module is the final module in the sequence that encapsulates the result of all previous modules.

Below are some examples that map the problem to the modules.
"""

prompt_header_rank = """
You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows, ordered by their interpretability from highest to lowest:

- Finish[answer]: This module returns the final answer and finishes the task. This module is the final module in the sequence that encapsulates the result of all previous modules.

- Calculate[formula]: This module calculates a given formula and returns the result. It takes in a mathematical formula and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB[DBName; subsetNames]: This module loads a database specified by the database name and subset names, and returns the loaded database. It accepts a database name and subset names, returning the corresponding database. The DBName can be one of the following: hupd. Normally, we consider using "LoadDB" only when the question requires data from a specific structured dataset.

- AutoLoadDB[DBName; trainStartDate; trainEndDate; validationStartDate; validationEndDate]: This module loads a database specified by the database name and date ranges for training and validation. It directly loads the database from Hugging Face and separates it into training and validation subsets based on the provided date ranges. Normally, we only consider using "AutoLoadDB" when the question specifies the training and validation sets or needs to be solved through a machine learning algorithm. 

- TargetFilter[targetColumn; filterCondition]: This module modifies a database in place by removing the rows that don't satisfy the filter condition. It accepts a target column and a filter condition, and the default filter condition is "not NA." Example conditions include "not NA," "keep ACCEPT,REJECT," and "remove 0,1." We always use "TargetFilter" after loading the database with either "LoadDB" or "AutoLoadDB".

- PandasInterpreter[Python, split]: This module interprets Pandas code written in Python, executes it on a dataframe specified by split, and returns the result. It takes in Python code and a dataframe specified by split, and returns the result of the code execution. Choices for split are "all", "train", or "validation". Normally, we only consider using "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataset.

- PythonInterpreter[Python]: This module interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only consider using "PythonInterpreter" when the question requires complex computations or custom data manipulation.

- LogisticRegression[Python]: This module applies a logistic regression model using Python code and returns the result. It takes in Python code to define and fit a logistic regression model on a specified dataset, returning the model's predictions. Normally, we only consider using "LogisticRegression" when the question involves binary classification tasks.

- DistilbertBaseUncased[Python]:  This module applies the DistilBERT base uncased model using Python code and returns the result. It takes in Python code to load, fine-tune, and evaluate the DistilBERT base uncased model on a specified dataset, returning the model's predictions. Normally, we consider using "DistilbertBaseUncased" when the question involves natural language processing tasks such as text classification, sentiment analysis, or named entity recognition.

Below are some examples that map the problem to the modules. When addressing a question, modules positioned higher on this list are preferred over those that are lower.
"""

prompt_header_formula = """
You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows, with the formulas used to calculate their interpretability scores defined in {}:

- Calculate[formula] {9}: This module calculates a given formula and returns the result. It takes in a mathematical formula and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB[DBName; subsetNames] {8.5}: This module loads a database specified by the database name and subset names, and returns the loaded database. It accepts a database name and subset names, returning the corresponding database. The DBName can be one of the following: hupd. Normally, we consider using "LoadDB" only when the question requires data from a specific structured dataset.

- AutoLoadDB[DBName; trainStartDate; trainEndDate; validationStartDate; validationEndDate] {8}: This module loads a database specified by the database name and date ranges for training and validation. It directly loads the database from Hugging Face and separates it into training and validation subsets based on the provided date ranges. Normally, we only consider using "AutoLoadDB" when the question specifies the training and validation sets or needs to be solved through a machine learning algorithm. 

- TargetFilter[targetColumn; filterCondition] {if filterCondition is "not NA", then 7.5; otherwise, 7.}: This module modifies a database in place by removing the rows that don't satisfy the filter condition. It accepts a target column and a filter condition, and the default filter condition is "not NA." Example conditions include "not NA," "keep ACCEPT,REJECT," and "remove 0,1." We always use "TargetFilter" after loading the database with either "LoadDB" or "AutoLoadDB".

- PandasInterpreter[Python, split] {if the number of lines of Python code is less than 5, 7; If the number of lines of Python code is between 5 and 10, 6.5; if the number of lines of Python code is greater than 10, 6.}: This module interprets Pandas code written in Python, executes it on a dataframe specified by split, and returns the result. It takes in Python code and a dataframe specified by split, and returns the result of the code execution. Choices for split are "all", "train", or "validation". Normally, we only consider using "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataset.

- PythonInterpreter[Python] {[if the number of lines of Python code is less than 20, 7; if the number of lines of Python code is between 20 and 50, 6.5; if the number of lines of Python code is greater than 50, 6.] divide by [if the number of imported packages is less than 5, 1; if the number of imported packages is between 5 and 10, 1.5; if the number of imported packages is greater than 10, 2]}: This module interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only consider using "PythonInterpreter" when the question requires complex computations or custom data manipulation.

- LogisticRegression[Python] {6}: This module applies a logistic regression model using Python code and returns the result. It takes in Python code to define and fit a logistic regression model on a specified dataset, returning the model's predictions. Normally, we only consider using "LogisticRegression" when the question involves binary classification tasks.

- DistilbertBaseUncased[Python] {3}:  This module applies the DistilBERT base uncased model using Python code and returns the result. It takes in Python code to load, fine-tune, and evaluate the DistilBERT base uncased model on a specified dataset, returning the model's predictions. Normally, we consider using "DistilbertBaseUncased" when the question involves natural language processing tasks such as text classification, sentiment analysis, or named entity recognition.

- Finish[answer] {10}: This module returns the final answer and finishes the task. This module is the final module in the sequence that encapsulates the result of all previous modules.

Below are some examples that map the problem to the modules. When addressing a question, modules that achieve higher interpretability scores are preferred over those that achieve lower scores.
"""

# If got more than 1 argument, need to separate with ;
# Needs an example with dbloader where the question doesn't involve specific years

prompt_example_clean = """
Question: What was the percentage of patents accepted in 2017?

Modules: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; all]", "Finish[9.388567293777134]"]

Question: What is the 20th Fibonacci number?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.
"""

prompt_example_compare = """
Question: What was the percentage of patents accepted in 2017?

Modules1: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; all]", "Finish[9.388567293777134]"]
Modules2: ["AutoLoadDB[hupd; 2017-01-01; 2017-12-31; 2017-01-01; 2017-12-31]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; train]", "Finish[9.388567293777134]"]

Best Module: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; all]", "Finish[9.388567293777134]"]

Thought: Modules1 is selected because it's more interpretable. 

Question: What is the 20th Fibonacci number?

Modules1: ["Calculate[0+0]", "Calculate[0+1]", "Calculate[0+1]", "Calculate[1+1]", "Calculate[1+2]", "Calculate[2+3]", "Calculate[3+5]", "Calculate[5+8]", "Calculate[8+13]", "Calculate[13+21]", "Calculate[21+34]", "Calculate[34+55]", "Calculate[55+89]", "Calculate[89+144]", "Calculate[144+233]", "Calculate[233+377]", "Calculate[377+610]", "Calculate[610+987]", "Calculate[987+1597]", "Calculate[1597+2584]", "Finish[4181]"]
Modules2: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Best Module: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Thought: Modules2 is selected because it's more interpretable. 

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Best Module: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Thought: Modules is selected because it's the only solution. 

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. You only need to output Best Module. 
"""

prompt_example_compare_rank = """
Question: What was the percentage of patents accepted in 2017?

Modules1: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; all]", "Finish[9.388567293777134]"]
Modules2: ["AutoLoadDB[hupd; 2017-01-01; 2017-12-31; 2017-01-01; 2017-12-31]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; train]", "Finish[9.388567293777134]"]

Best Module: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; all]", "Finish[9.388567293777134]"]

Thought: Modules1 uses LoadDB, which is ranked higher in interpretability than AutoLoadDB. All the other modules in both Modules1 and Modules2 are the same. Consequently, Modules1 is chosen because it offers greater interpretability.

Question: What is the 20th Fibonacci number?

Modules1: ["Calculate[0+0]", "Calculate[0+1]", "Calculate[0+1]", "Calculate[1+1]", "Calculate[1+2]", "Calculate[2+3]", "Calculate[3+5]", "Calculate[5+8]", "Calculate[8+13]", "Calculate[13+21]", "Calculate[21+34]", "Calculate[34+55]", "Calculate[55+89]", "Calculate[89+144]", "Calculate[144+233]", "Calculate[233+377]", "Calculate[377+610]", "Calculate[610+987]", "Calculate[987+1597]", "Calculate[1597+2584]", "Finish[4181]"]
Modules2: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Best Module: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Thought: Modules1 uses Calculate, which is ranked higher in interpretability than PythonInterpreter. All the other modules in both Modules1 and Modules2 are the same. However, Modules1 has 21 modules in total, while Modules2 only has 2 modules. Consequently, Modules2 is chosen for its greater interpretability.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Best Module: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Thought: Modules is selected because it's the only solution. 

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. You only need to output Best Module. 
""" 

prompt_example_compare_formula = """
Question: What was the percentage of patents accepted in 2017?

Modules1: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; all]", "Finish[9.388567293777134]"]
Modules2: ["AutoLoadDB[hupd; 2017-01-01; 2017-12-31; 2017-01-01; 2017-12-31]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; train]", "Finish[9.388567293777134]"]

Best Module: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; all]", "Finish[9.388567293777134]"]

Thought: Modules1 uses LoadDB, while Modules2 uses AutoLoadDB. All the other modules in both Modules1 and Modules2 are the same. Using LoadDB has an interpretability score of {8.5}, therefore using LoadDB once has an interpretability score of {8.5} / (1 ** (1/10)) = 8.5. Using AutoLoadDB has an interpretability score of {8}, therefore using AutoLoadDB once has an interpretability score of {8} / (1 ** (1/10)) = 8. Therefore, Modules1 has a higher interpretability score than Modules2. As a result, Modules1 is selected.

Question: What is the 20th Fibonacci number?

Modules1: ["Calculate[0+0]", "Calculate[0+1]", "Calculate[0+1]", "Calculate[1+1]", "Calculate[1+2]", "Calculate[2+3]", "Calculate[3+5]", "Calculate[5+8]", "Calculate[8+13]", "Calculate[13+21]", "Calculate[21+34]", "Calculate[34+55]", "Calculate[55+89]", "Calculate[89+144]", "Calculate[144+233]", "Calculate[233+377]", "Calculate[377+610]", "Calculate[610+987]", "Calculate[987+1597]", "Calculate[1597+2584]", "Finish[4181]"]
Modules2: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Best Module: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(20)\n]", "Finish[4181]"]

Thought: Modules1 uses Calculate for 20 times, while Modules2 uses PythonInterpreter once. All the other modules in both Modules1 and Modules2 are the same. Using Calculate has an interpretability score of {9}, therefore using Calculate for 20 times has an interpretability score of {9} / (20 ** (1/10)) = 6.67. Using PythonInterpreter that has 12 lines of code with no package import has an interpretability score of {[7]/[1]=[7]}, therefore using PythonInterpreter once has an interpretability score of {7} / (1 ** (1/10)) = 7. Therefore, Modules2 has a higher interpretability score than Modules2. As a result, Modules2 is selected.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Best Module: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Thought: Modules is selected because it's the only solution. 

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question. 
""" 

prompt = prompt_header_formula+prompt_example_compare_formula

# verify the thought chain, if correct, add You only need to output Best Module. into prompt. 