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
from tools.simple_model.forecaster import forecast as forecaster
from tools.llm.llm_inferencer import llm_inferencer 
from tools.math.calculator import calculator
from tools.table.tabtools import table_toolkits
import datetime

from api.gpt4 import call_gpt4, call_gpt3_5
from api.gemini import call_gemini_pro
from api.claude import call_claude3

from tools.tools_set import tools_gpt, tools_gemini


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
                if args.policy_engine == "gpt-3.5-turbo":
                    content, tool_calls = call_gpt3_5(user_prompt, system_content=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gpt, tool_choice="auto")
                elif args.policy_engine == "gpt-4-turbo":
                    content, tool_calls = call_gpt4(user_prompt, system_content=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gpt, tool_choice="auto")
                elif args.policy_engine == "gemini":
                    content = call_gemini_pro(user_prompt, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gemini, tool_choice="auto")
                elif args.policy_engine == "claude":
                    content = call_claude3(user_prompt, image_path=None, temperature=args.policy_temperature)
                else:
                    raise ValueError("Invalid engine")
         
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
                    if not (isinstance(function_response, str) and function_response.startswith("Error:")):
                        cost[question_type] += cost_function(function_type, function_arguments)
                        total_cost += cost_function(function_type, function_arguments)
                        per_question_cost += cost_function(function_type, function_arguments)
                    
                    tool_call_response = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_type,
                        "content": str(function_response) if function_response is not None else "",
                    } 
                    # print(tool_call_response) 
                    llm_answer = function_response
                    messages.append(tool_call_response)  
                    logs.append(tool_call_response)
                    iterations += 1
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
            if question_type in ["1", "3", "6"]: # threshold correct / incorrect
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
