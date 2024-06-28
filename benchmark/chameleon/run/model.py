import os
import re
import sys
import json
from openai import OpenAI
from ast import literal_eval
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import *
from demos import prompt_policy

# import solvers
from tools.code.python_interpreter import execute as python_interpreter
from tools.math.calculator import calculator, WolframAlphaCalculator
from tools.table.tabtools import table_toolkits
from tools import finish
import jsonlines

db = table_toolkits() 
ACTION_LIST = {
    'Calculate': WolframAlphaCalculator,
    'LoadDB': db.db_loader, 
    'PandasInterpreter': db.pandas_interpreter, 
    'PythonInterpreter': python_interpreter,
    'Classifier': db.classifier,
    'Finish': finish
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "Calculate",
            "description": "Conduct an arithmetic operation",
            "parameters": {
                "type": "object",
                "properties": {
                    "mathOp": {
                        "type": "string",
                        "description": "An arithmetic operation, e.g. 2*3.",
                    }
                },
                "required": ["mathOp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "LoadDB",
            "description": "Load a database specified by the DBName, subset, and a boolean value split. Normally, we only use LoadDB when the question requires data from a specific structured database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "DBName": {
                        "type": "string",
                        "description": "The name of the database to be loaded, e.g. hupd",
                    },
                    "subset": {
                        "type": "string",
                        "description": "The subset of the database is specified by the range of years in the format startYear-endYear, inclusive on both ends, e.g. 2016-2018.",
                    },
                    "split": {
                        "type": "boolean",
                        "description": "When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. The default value of split is False.",
                    }
                },
                "required": ["DBName", "subset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PandasInterpreter",
            "description": "Interpret Pandas code written in Python. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We can only use PandasInterpreter after loading the dataframe with LoadDB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pythonCode": {
                        "type": "string",
                        "description": "Pandas code written in Python that involves operations on a DataFrame df",
                    }
                },
                "required": ["pythonCode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PythonInterpreter",
            "description": "Interprets Python code. Normally, we only use PythonInterpreter when the question requires complex computations. We don't use PythonInterpreter when the question requires data manipulation performed on a specific structured dataframe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pythonCode": {
                        "type": "string",
                        "description": "Python code",
                    }
                },
                "required": ["pythonCode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Classifier",
            "description": "Run a specified classifier model on the given predictorSection to predict the target. Normally, we use the Classifier module for binary or multi-class classification tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modelName": {
                        "type": "string",
                        "description": "The modelName can be logistic_regression or distilbert-base-uncased.",
                    },
                    "predictorSection": {
                        "type": "string",
                        "description": "The predictor variable of the classifier model, which is natural language requiring tokenization.",
                    },
                    "target": {
                        "type": "string",
                        "description": "The target variable of the classifier model.",
                    }
                },
                "required": ["modelName", "predictorSection", "target"], 
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Finish",
            "description": "Return the final answer and finish the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to be returned",
                    }
                },
                "required": ["answer"], 
            },
        },
    }
]

openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)

client = OpenAI()

class solver:

    def __init__(self, args):
        # arguments
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.args = args
        # external arguments
        self.api_key = openai.api_key
        self.examples, self.pids = self.load_data()
        
    def load_data(self):
        examples = ''
        pids = []
        file_path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/{}/{}-{}.jsonl".format(self.args.hardness, self.args.dataset, self.args.hardness) #<YOUR_OWN_PATH>
        with open(file_path, 'r') as f:
            contents = []
            for item in jsonlines.Reader(f):
                contents.append(item)
                pids.append(item['qid'])
        examples = {item['qid']: item for item in contents}
        return examples, pids

    def predict_modules(self):
        # get the module input
        test_prompt, full_prompt = self.build_prompt_for_policy()
        messages=[
            {"role": "user", "content": full_prompt},
        ]
        print(f'PROMPT: \n{full_prompt}\n' + '-' * 20 + '\n')
        # execute the module
        response = client.chat.completions.create(model=self.policy_engine, messages=messages, temperature=self.policy_temperature, max_tokens=self.policy_max_tokens, tools=tools, tool_choice="auto")
        response_message = response.choices[0].message
        print(f'GPT RESPONSE: \n{response_message}\n')
        # if "Best Modules:" in modules:
        #     cost_start_ind = modules.rfind("{") ###
        #     cost_end_ind = modules.rfind("}") ###
        #     single_cost = int(modules[cost_start_ind+1:cost_end_ind]) ###
        #     start_ind = modules.find("Best Modules: ")+len("Best Modules: ")
        #     end_ind = modules.rfind("]") # find("Thought:")
        #     modules = modules[start_ind:end_ind+1]
        # # modules = self.update_modules(modules)
        # # update the cache
        # self.cache["modules:input"] = test_prompt
        # self.cache["modules:output"] = modules
        # return modules, single_cost ###
    
    