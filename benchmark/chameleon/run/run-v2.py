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

from tools.code.python_interpreter import execute as python_interpreter
from tools.math.calculator import calculator, WolframAlphaCalculator
from tools.table.tabtools import table_toolkits
from tools.finish import finish
import datetime

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

db = table_toolkits()
ACTION_LIST = {
    'Calculate': WolframAlphaCalculator,
    'LoadDB': db.db_loader, 
    'PandasInterpreter': db.pandas_interpreter, 
    'PythonInterpreter': python_interpreter,
    'Classifier': db.classifier,
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
                        "description": "An arithmetic operation, e.g. 2*3.",
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
            "description": "Load a database specified by the DBName, subset, and a boolean value split. Normally, we only use LoadDB when the question requires data from a specific structured database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_db": {
                        "type": "string",
                        "description": "The name of the database to be loaded. The only choices for target_db are hupd (a patent dataset).",
                    },
                    "duration": {
                        "type": "string",
                        "description": "The subset of the database is specified by the range of years in the format startYear-endYear, inclusive on both ends, e.g. 2016-2018.",
                    },
                    "split": {
                        "type": "boolean",
                        "description": "When split is False, it loads an entire dataframe; when split is True, it loads a dataset dictionary comprising training and validation datasets. The default value is False.",
                    }
                },
                "required": ["target_db", "duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PandasInterpreter",
            "description": "Interpret Pandas code written in Python. The final result must be assigned to variable ans. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We must first use LoadDB before we can use PandasInterpreter. We do not use this tool for general Python computations or tasks unrelated to dataframes.",
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
            "description": "Interprets Python code. Normally, we only use PythonInterpreter when the question requires complex computations. We don't use PythonInterpreter when the question requires data manipulation performed on a specific structured dataframe. We do not use this tool for tasks that can be performed with Pandas on dataframes.",
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
            "name": "Classifier",
            "description": "Run a specified classifier model on the given predictorSection to predict the target. Normally, we use the Classifier module for binary or multi-class classification tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The modelName can be logistic_regression or distilbert-base-uncased.",
                    },
                    "section": {
                        "type": "string",
                        "description": "The predictor variable of the classifier model, which is natural language requiring tokenization.",
                    },
                    "target": {
                        "type": "string",
                        "description": "The target variable of the classifier model.",
                    }, 
                    "num_classes": {
                        "type": "integer",
                        "description": "The number of classes in the classification task. The default value is 2.",
                    }
                },
                "required": ["model_name", "section", "target"], 
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Finish",
            "description": "Return the final answer and finish the task. All solutions need to be terminated with Finish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "argument": {
                        "type": "string",
                        "description": "The final answer to be returned",
                    }
                },
                "required": ["argument"], 
            },
        },
    }
]

def calc_cost(function_type, function_arguments):
    if function_type=="Calculate":
        return 2
    if function_type=="LoadDB":
        return 3
    if function_type=="PandasInterpreter":
        num_lines = len(function_arguments["pandas_code"].split('\n'))
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
        num_lines = len(function_arguments["python_code"].split('\n'))
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
    if function_type=="Classifier":
        if function_arguments["model_name"]=="logistic_regression":
            return 7
        if function_arguments["model_name"]=="distilbert-base-uncased":
            return 10
    if function_type=="Finish":
        return 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--model', type=str, default='chameleon')
    parser.add_argument('--label', type=str, default='chameleon_chatgpt')
    parser.add_argument('--task_name', type=str, default='hupd') 
    parser.add_argument('--test_split', type=str, default='test1k', 
                        choices=['dev', 'dev1k', 'test', 'test1k'])
    parser.add_argument('--test_number', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--dataset", type=str, default="hupd")
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
    result_root = f"{args.output_root}/{args.task_name}" 
    os.makedirs(result_root, exist_ok=True)
    result_file = f"{result_root}/{args.label}_{args.test_split}.json"
    print("result_file", result_file)
    cache_file = f"{result_root}/{args.label}_{args.test_split}_cache.jsonl"
    cache = []

    total_count, total_correct, total_cost, total_reliability = 0, 0, 0, 0
    count, correct, cost = defaultdict(int), defaultdict(int), defaultdict(int)
    pids = solver.pids
    
    for pid in tqdm(pids):
        if total_count < 10:
            print("\n\n===================================\n")
            print(f"# [Pid]: {pid}\n") # problem id

        total_count += 1  
        example = solver.examples[pid] # get one example 
        user_prompt = example["question"] 
        question_type = example["question_type"]
        gt_cost = 0
        count[question_type] += 1

        messages = [{"role": "system", "content": prompt_policy.prompt.strip()}]+prompt_policy.messages
        # messages = [{"role": "system", "content": prompt_policy.prompt_formula.strip()}]+prompt_policy.messages_formula
        # messages = prompt_policy.messages_formula
        formula = False
        
        messages.append({"role": "user", "content": user_prompt})
        logs = [{"role": "user", "content": user_prompt}]
        function_type = None
        llm_answer = None
        llm_cost = None
        iterations = 0
        while iterations<15:
            try:
                response = client.chat.completions.create(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools, tool_choice="auto")
                choice = response.choices[0]
                response_message = choice.message
                tool_calls = response_message.tool_calls
                content = response_message.content
                if formula and iterations == 0: # only used for the formula prompt
                    start_ind = content.rfind("Cost is ")+len("Cost is ")
                    llm_cost = int(content[start_ind:])
                    content += "Execute the tools and arguments one by one, following the sequence specified after 'Best Modules: ' precisely."
                                
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
                    cost[question_type] += calc_cost(function_type, function_arguments)
                    total_cost += calc_cost(function_type, function_arguments)
                    gt_cost += calc_cost(function_type, function_arguments)
                    function_response = function(**function_arguments)
                    
                    tool_call_response = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_type,
                        "content": function_response,
                    }
                    # print(tool_call_response) 
                    if function_type!="Finish":
                        llm_answer = function_response
                    messages.append(tool_call_response)  
                    logs.append(tool_call_response)
                    iterations += 1
                else:
                    break
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                logs.append(json.dumps(str(e)))
                break
        
        gt_answer = str(example["answer"])
        if llm_answer==gt_answer:
            correct[question_type] += 1
            total_correct += 1
        if llm_cost==gt_cost:
            total_reliability += 1
        
        logs.append({"LLM Answer": llm_answer})
        logs.append({"Ground-Truth Answer": gt_answer})
        cache.append({"qid": pid, "question_type": example["question_type"], "question": example["question"], "LLM Answer": llm_answer, "Ground-Truth Answer": gt_answer})
        if not os.path.exists('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version)): #<YOUR_OWN_PATH>
            os.makedirs('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version)) #<YOUR_OWN_PATH>
            logs_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version) #<YOUR_OWN_PATH>
        with open(os.path.join(logs_dir, f"{pid}.txt"), 'w') as f:
            for item in logs:
                f.write(f"{item}\n")
    
    with jsonlines.open(cache_file, mode='w') as writer:
        for row in cache:
            writer.write(row)

    acc = {}
    for key in count:
        if key not in correct:
            acc[key] = 0
        else:
            acc[key] = correct[key] / count[key] * 100
    agg_acc = total_correct / total_count * 100
    agg_cost = total_cost / total_count
    agg_reliability = total_reliability / total_count * 100
    if not llm_cost:
        agg_reliability = "NA"
        
    # save the result
    result = {'acc': acc, 'agg_acc': agg_acc, 'cost': cost, 'agg_cost': agg_cost, 'agg_reliability': agg_reliability, 'count': total_count, 'args': vars(args)}
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4, separators=(',', ': '))
