
prompt = """
You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows:

- Calculate[formula]: This module calculates a given formula and returns the result. It takes in a mathematical formula and returns the calculated result. Normally, we only consider using "Calculate" when the question involves mathematical computations.

- LoadDB[DBName; subsetNames]: This module loads a database specified by the database name and subset names, and returns the loaded database. It accepts a database name and subset names, returning the corresponding database. The DBName can be one of the following: hupd. Normally, we consider using "LoadDB" only when the question requires data from a specific structured dataset.

- AutoLoadDB[DBName; trainStartDate; trainEndDate; validationStartDate; validationEndDate]: This module loads a database specified by the database name and date ranges for training and validation. It directly loads the database from Hugging Face and separates it into training and validation subsets based on the provided date ranges. Normally, we only consider using "AutoLoadDB" when the question specifies the training and validation sets or needs to be solved through a machine learning algorithm. 

- TargetFilter[targetColumn; filterCondition]: This module modifies a database in place by removing the rows that don't satisfy the filter condition. It accepts a target column and a filter condition, and the default filter condition is "not NA." Example conditions include "not NA," "keep ACCEPT,REJECT," and "remove 0,1." We always use "TargetFilter" after loading the database with either "LoadDB" or "AutoLoadDB".

- PandasInterpreter[Python]: This module interprets Pandas code written in Python and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only consider using "PandasInterpreter" when the question requires data manipulation performed on a specific structured dataset.

- PythonInterpreter[Python]: This module interprets Python code and returns the result. It takes in Python code and returns the result of the code execution. Normally, we only consider using "PythonInterpreter" when the question requires complex computations or custom data manipulation.

- Finish[answer]: This module returns the final answer and finishes the task. This module is the final module in the sequence that encapsulates the result of all previous modules.

Below are some examples that map the problem to the modules.

Question: What was the percentage of patents accepted in 2017?

Modules: ["LoadDB[hupd; 2017-2017]", "TargetFilter[decision; not NA]", "PandasInterpreter[import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted; None]", "Finish[9.388567293777134]"]

Question: What is the 100th Fibonacci number?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution(n):\n    # Calculate the nth Fibonacci number\n    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nans = solution(100)\n]", "Finish[354224848179261915075]"]

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Modules: ["PythonInterpreter[# solution in Python:\n\ndef solution():\n # Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n golf_balls_initial = 58\n golf_balls_lost_tuesday = 23\n golf_balls_lost_wednesday = 2\n golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n result = golf_balls_left\n return result]", "Finish[33]"]

Now, you need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.
"""

# If got more than 1 argument, need to separate with ;
# Needs an example with dbloader where the question doesn't involve specific years