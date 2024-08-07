import os
import re
import sys
import json
import jsonlines
import argparse
import random
from tqdm import tqdm
from demos import prompt_policy
from openai import OpenAI
from collections import defaultdict
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utilities import *
from model import solver

from tools.finish import finish
from tools.code.python_interpreter import execute as python_interpreter
from tools.code.forecaster import forecast as forecaster
from tools.llm.llm_inferencer import llm_inferencer 
from tools.math.calculator import calculator
from tools.table.tabtools import table_toolkits
import datetime

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

db = table_toolkits()
ACTION_LIST = {
    'Calculate': calculator,
    'LoadDB': db.db_loader, 
    'PandasInterpreter': db.pandas_interpreter, 
    'PythonInterpreter': python_interpreter,
    'Forecaster': forecaster,
    'TextualClassifier': db.textual_classifier,
    'LLMInterpreter': llm_inferencer,
    'Finish': finish
}

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "Calculate",
            "description": "Conduct an arithmetic operation",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_query": {
                        "type": "string",
                        "description": "An arithmetic operation containing only numbers and operators, e.g. 2*3.",
                    }
                },
                "required": ["input_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "LoadDB",
            "description": "Load a database specified by the DBName, train and test subsets, and a column to be predicted. Normally, we only use LoadDB when the question requires data from a specific structured database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_db": {
                        "type": "string",
                        "description": "The name of the database to be loaded. The only choices for target_db are hupd (a patent dataset) and neurips (a papers dataset).",
                    },
                    "train_duration": {
                        "type": "string",
                        "description": "The training subset of the database is specified by a range that's inclusive on both ends. When target_db is hupd, specify the range of years in the format startYear-endYear, e.g. 2004-2006. When target_db is neurips, specify the range of rows in the format 0-endRow, e.g. 0-2000. When the task does not involve prediction and the target_db is neurips, use the default range 0-3585.",
                    },
                    "test_duration": {
                        "type": "string",
                        "description": "The testing subset of the database is specified by a range that's inclusive on both ends. When target_db is hupd, specify the range of years in the format startYear-endYear, e.g. 2016-2018. When target_db is neurips, specify the range of rows in the format startRow-3585, e.g. 2001-3585, where startRow must be one more than the endRow of train_duration. When the task does not involve prediction, set this value to None.",
                    },
                    "outcome_col": {
                        "type": "string",
                        "description": "The column to predict if the task involves making a prediction. If no prediction is required, set this value to None.",
                    }
                },
                "required": ["target_db", "train_duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PandasInterpreter",
            "description": "Interpret Pandas code written in Python. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We must first use LoadDB before we can use PandasInterpreter. We do not use this tool for general Python computations or tasks unrelated to dataframes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pandas_code": {
                        "type": "string",
                        "description": "Pandas code written in Python that involves operations on a DataFrame df",
                    }
                },
                "required": ["pandas_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PythonInterpreter",
            "description": "Interpret Python code. Normally, we only use PythonInterpreter when the question requires complex computations. We do not use this tool for tasks that can be performed with Pandas on dataframes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "Python code",
                    }
                },
                "required": ["python_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Forecaster",
            "description": "Run a specified forecast model on the previous data to predict the next forecast_len data points",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model_name can be linear_regression or ARIMA",
                    },
                    "previous_data": {
                        "type": "string",
                        "description": "A list of past data points used to train the forecast model",
                    },
                    "forecast_len": {
                        "type": "integer",
                        "description": "The number of data points to be predicted by the forecast model",
                    } 
                },
                "required": ["model_name", "previous_data", "forecast_len"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TextualClassifier",
            "description": "Run a specified classifier model on the given textual predictorSection to predict the target. Normally, we use the TextualClassifier module for classification tasks that work with textual data as its input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model_name can be logistic_regression, distilbert-base-uncased, cnn, or naive_bayes.",
                    },
                    "section": {
                        "type": "string",
                        "description": "The predictor variable of the classifier model, which is a column that consists of natural language requiring tokenization.",
                    },
                    "target": {
                        "type": "string",
                        "description": "The target variable of the classifier model.",
                    },
                    "one_v_all": {
                        "type": "string",
                        "description": "The class label for a one-vs-all classification task. When it's set to default value None, the model will predict all possible classes.",
                    }
                },
                "required": ["model_name", "section", "target"], 
            },
        },
    }, 
    {
        "type": "function",
        "function": {
            "name": "LLMInterpreter",
            "description": "Use the current LLM to generate an answer."
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Finish",
            "description": "Terminate the task and return the final answer. You must use Finish as the final module for solving each question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable_values": {
                        "type": "string",
                        "description": "A string that evaluates to a dictionary of variables and their corresponding values.",
                    },
                    "answer_variable": {
                        "type": "string",
                        "description": "A key among the variable_values dictionary that corresponds to the variable which best addresses the question.",
                    },
                    "answer_type": {
                        "type": "string",
                        "description": "A string specifying the required type for the final answer. The only choices are list, float, integer, and string."
                    }
                },
                "required": ["variable_values", "answer_variable", "answer_type"], 
            },
        },
    }
]

def calc_cost1(function_type, function_arguments):
    if function_type=="Calculate":
        return 2
    if function_type=="LoadDB":
        return 3
    if function_type=="PandasInterpreter":
        num_lines = len(function_arguments["pandas_code"].splitlines()) 
        num_packages = function_arguments["pandas_code"].count('import')
        if num_lines<10:
            lines_cost = 4
        elif num_lines<=20:
            lines_cost = 7
        elif num_lines<=100:
            lines_cost = 9
        else:
            lines_cost = 10
        if num_packages<2:
            packages_cost = 1
        elif num_packages<=5:
            packages_cost = 1.5
        else:
            packages_cost = 2
        return lines_cost*packages_cost
    if function_type=="PythonInterpreter":
        num_lines = len(function_arguments["python_code"].splitlines()) 
        num_packages = function_arguments["python_code"].count('import')
        if num_lines<10:
            lines_cost = 4
        elif num_lines<=20:
            lines_cost = 7
        elif num_lines<=100:
            lines_cost = 9
        else:
            lines_cost = 10
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
        elif function_arguments["model_name"]=="naive_bayes":
            return 8
        elif function_arguments["model_name"]=="cnn":
            return 15
        elif function_arguments["model_name"]=="distilbert-base-uncased":
            return 20
    if function_type=="LLMInterpreter":
        return 30
    if function_type=="Finish":
        return 0

def calc_cost2(function_type, function_arguments): ###
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--model', type=str, default='chameleon')
    parser.add_argument('--label', type=str, default='chameleon_chatgpt')
    parser.add_argument('--test_split', type=str, default='test1k', 
                        choices=['dev', 'dev1k', 'test', 'test1k'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--hardness", type=str, default="easy")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--gpt", type=str, default="gpt3")
    # module prediction
    parser.add_argument('--modules', nargs='+', default=None, help='default modules') 
    parser.add_argument('--policy_engine', type=str, default="gpt-3.5-turbo", help='engine for module prediction')
    parser.add_argument('--policy_temperature', type=float, default=0., help='temperature for module prediction')
    parser.add_argument('--policy_max_tokens', type=int, default=128, help='max tokens for module prediction')

    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    # Build the solver
    solver = solver(args)
    print(f"# Number of test examples: {len(solver.examples)}\n") 

    # Get the result file
    result_root = f"{args.output_root}" 
    os.makedirs(result_root, exist_ok=True)
    result_file = f"{result_root}/{args.label}_{args.test_split}.json"
    print("result_file", result_file)
    cache_file = f"{result_root}/{args.label}_{args.test_split}_cache.jsonl"
    cache = []
    cost_function = calc_cost1 # Change with experiment

    total_count, total_cost = 0, 0
    count, performance, errors, cost, cost_original = defaultdict(int), {}, defaultdict(int), defaultdict(int), defaultdict(list)
    pids = solver.pids
    
    for pid in tqdm(pids): # pids
        if total_count < 10:
            print("\n\n===================================\n")
            print(f"# [Pid]: {pid}\n") # problem id

        total_count += 1  
        example = solver.examples[pid] # get one example 
        user_prompt = example["question"] 
        question_type = example["question_type"]
        per_question_cost = 0
        count[question_type] += 1

        messages = prompt_policy.messages.copy() # Change with experiment
        # messages = prompt_policy.messages_formula.copy()
        
        messages.append({"role": "user", "content": user_prompt})
        logs = [{"role": "user", "content": user_prompt}]
        function_type = None
        llm_answer = None
        iterations = 0

        while iterations<10:
            try:
                response = client.chat.completions.create(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools, tool_choice="auto")
                # print("response", response) 
                choice = response.choices[0]
                response_message = choice.message
                tool_calls = response_message.tool_calls
                content = response_message.content
         
                if tool_calls:
                    tool_call = tool_calls[0]
                    
                    response_with_tools = {
                        "role": choice.message.role,
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type, 
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    }
                    # print(response_with_tools) 
                    messages.append(response_with_tools) 
                    logs.append(response_with_tools)
                                
                    function_type = tool_call.function.name
                    function = ACTION_LIST[function_type]
                    function_arguments = json.loads(tool_call.function.arguments)
                    function_response = function(**function_arguments)
                    # print("function_type", function_type) ###
                    # print("function_arguments", function_arguments) ###
                    if not (isinstance(function_response, str) and function_response.startswith("Error:")):
                        cost[question_type] += cost_function(function_type, function_arguments)
                        total_cost += cost_function(function_type, function_arguments)
                        per_question_cost += cost_function(function_type, function_arguments)
                    
                    # print("1") ###
                    
                    tool_call_response = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_type,
                        "content": str(function_response) if function_response is not None else "",
                    } ###
                    # print(tool_call_response) 
                    llm_answer = function_response
                    messages.append(tool_call_response)  
                    logs.append(tool_call_response)
                    iterations += 1
                    # print("3") ###
                else:
                    response_without_tools = {
                        "role": choice.message.role,
                        "content": content
                    }
                    messages.append(response_without_tools) 
                    logs.append(response_without_tools)
                    break
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                logs.append(json.dumps(str(e)))
                break
        
        gt_answer = example["answer"]
        
        def is_json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except (TypeError, OverflowError):
                return False
        if llm_answer is None:
            errors[question_type] += 1
        elif not is_json_serializable(llm_answer):
            llm_answer = None
            errors[question_type] += 1
        else:
            # Calculate performance metric
            if question_type in ["1", "3", "6"]: # R2 -> threshold correct / incorrect
                # if question_type not in performance:
                #     performance[question_type] = [0,[]]
                # try:
                #     performance[question_type][0] += (llm_answer-gt_answer)**2
                #     performance[question_type][1].append(gt_answer)
                if question_type not in performance:
                    performance[question_type] = 0
                try:
                    performance[question_type] += int(abs(llm_answer-gt_answer)<=0.005*gt_answer)
                except:
                    errors[question_type] += 1
            elif question_type in ["2","5"]: # set intersection
                if question_type not in performance:
                    performance[question_type] = 0
                try:
                    performance[question_type] += len(set(gt_answer)&set(llm_answer)) / len(set(gt_answer))
                except:
                    errors[question_type] += 1
            elif question_type in ["4"]: # exact match
                if question_type not in performance:
                    performance[question_type] = 0
                try:
                    performance[question_type] += int(llm_answer==gt_answer)
                except:
                    errors[question_type] += 1
            elif question_type in []: # F1
                pass
        
        cost_original[question_type].append(per_question_cost) 
        logs.append({"LLM Answer": llm_answer})
        logs.append({"Ground-Truth Answer": gt_answer})
        cache.append({"qid": pid, "question_type": example["question_type"], "question": example["question"], "LLM Answer": llm_answer, "Ground-Truth Answer": gt_answer})
        if not os.path.exists('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.gpt, datetime_string, args.hardness, args.version)): #<YOUR_OWN_PATH>
            os.makedirs('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.gpt, datetime_string, args.hardness, args.version)) #<YOUR_OWN_PATH>
            logs_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.gpt, datetime_string, args.hardness, args.version) #<YOUR_OWN_PATH>
        with open(os.path.join(logs_dir, f"{pid}.txt"), 'w') as f:
            for item in logs:
                f.write(f"{item}\n")
    
    with jsonlines.open(cache_file, mode='w') as writer:
        for row in cache:
            writer.write(row)

    for key in performance.keys():
        if key in []: ### "1","3"
            actual_mean = sum(performance[key][1]) / len(performance[key][1])
            sstot = sum((x-actual_mean)**2 for x in performance[key][1])
            performance[key] = 1 - performance[key][0]/sstot
        elif key in ["1","2","3","4","5","6"]: 
            performance[key] = performance[key] / (count[key]-errors[key])
        elif key in []: 
            pass
    for key in errors.keys():
        errors[key] = errors[key] / count[key]
    for key in cost.keys():
        cost[key] = cost[key] / count[key]
    performance = dict(sorted(performance.items())) # between 0 and 1
    errors = dict(sorted(errors.items()))
    cost = dict(sorted(cost.items()))
    cost_original = dict(sorted(cost_original.items()))
    count = dict(sorted(count.items()))
    agg_cost = round(total_cost / total_count, 2)
    
    # save the result
    result = {'performance': performance, 'error': errors, 'cost': cost, 'agg_cost': agg_cost, 'cost_original': cost_original, 'count': count, 'total_count': total_count, 'args': vars(args)}
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4, separators=(',', ': '))
