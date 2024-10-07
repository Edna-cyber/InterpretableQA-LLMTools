import os
import re
import ast
import sys
import math
import statistics
import json
import jsonlines
import argparse
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, r2_score
from demos import prompt_policy
from openai import OpenAI
from collections import defaultdict
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utilities import *
from model import solver

from tools.finish import finish
from tools.python_interpreter import execute as python_interpreter
from tools.forecaster import forecast as forecaster
from tools.tfidf import tfidf as tfidf
from tools.llm_inferencer import llm_inferencer 
from tools.calculator import calculator
from tools.tabtools import table_toolkits, LogisticRegression, BasicCNNModel
import datetime

from api.gpt import call_gpt
from api.gemini import call_gemini_pro
from api.claude import call_claude3

from tools.tools_set import tools_gpt, tools_gemini
from cost_functions import calc_cost1, calc_cost2, calc_cost3, calc_cost4

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

db = table_toolkits()
ACTION_LIST = {
    'Calculator': calculator,
    'DBLoader': db.db_loader,
    'TFIDFMatcher': tfidf,
    'PandasInterpreter': db.pandas_interpreter, 
    'PythonInterpreter': python_interpreter,
    'Forecaster': forecaster,
    'TextualClassifier': db.textual_classifier,
    'LLMInferencer': llm_inferencer,
    'Finish': finish
}

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--label', type=str, default='chameleon_chatgpt')
    parser.add_argument("--hardness", type=str, default="easy")
    parser.add_argument("--prompt", type=str, default="clean")
    parser.add_argument("--formula", type=str, default="")
    parser.add_argument('--policy_engine', type=str, default="gpt-3.5-turbo", help='engine for module prediction')
    parser.add_argument('--policy_temperature', type=float, default=0., help='temperature for module prediction')
    parser.add_argument('--policy_max_tokens', type=int, default=128, help='max tokens for module prediction')

    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args

if __name__ == "__main__":
    args = parse_args()
    random.seed(1111)

    # Build the solver
    solver = solver(args)
    print(f"# Number of test examples: {len(solver.examples)}\n") 

    # Get the result file
    result_root = f"{args.output_root}" 
    os.makedirs(result_root, exist_ok=True)
    cache_file = f"{result_root}/{args.policy_engine}-{args.hardness}-{args.prompt}-{args.formula}-cache-{datetime_string}.jsonl"
    cache = []
    if args.formula=="formula1":
        cost_function = calc_cost1
    elif args.formula=="formula2":
        cost_function = calc_cost2
    elif args.formula=="formula3":
        cost_function = calc_cost3
    elif args.formula=="formula4":
        cost_function = calc_cost4

    pids = solver.pids
    
    for pid in tqdm(pids): ###
        # print("pid", pid)
        db.data = None # force reset
        example = solver.examples[pid] # get one example 
        if args.prompt=="interp":
            user_prompt = "Now, you need to act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps:1.Generate Solutions: List 2-4 sequences of tools that can solve the question.2.Calculate and Compare Costs: Determine the total interpretability cost for each sequence. Prefer tools with lower costs.3.Execute the Lowest Cost Solution. Question: "+example["question"]
        else: 
            user_prompt = "Now, you need to act as a policy model and determine the sequence of modules that can be executed sequentially can solve the question: "+example["question"]
        question_type = int(example["question_type"])
        per_question_cost = 0
        tool_count, tool_cost = defaultdict(int), defaultdict(int) 

        if args.prompt=="clean":
            messages = prompt_policy.messages.copy() 
        elif args.prompt=="subset":
            messages = prompt_policy.messages_subset.copy()
        elif args.prompt=="interp":
            if args.formula=="formula1":
                messages = prompt_policy.messages_formula_1.copy()
            elif args.formula=="formula2":
                messages = prompt_policy.messages_formula_2.copy()
            elif args.formula=="formula3":
                messages = prompt_policy.messages_formula_3.copy()
            elif args.formula=="formula4":
                messages = prompt_policy.messages_formula_4.copy()
        
        messages.append({"role": "user", "content": user_prompt})
        logs = [{"role": "user", "content": user_prompt}]
        function_type = None
        llm_answer = None
        iterations = 0

        while iterations<10:
            if args.hardness!="easy":
                db.prediction = True
            try:
                response = client.chat.completions.create(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gpt, tool_choice="auto")
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
                    function_cost = cost_function(function_type, function_arguments)
                    if not (isinstance(function_response, str) and function_response.startswith("Error:")):
                        tool_count[function_type] += 1
                        tool_cost[function_type] += function_cost
                        per_question_cost += function_cost
                        # print("per_question_cost", per_question_cost) ###
                    
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
                    if args.prompt=="interp" and iterations==1:
                        if "Modules2:" not in content:
                            more_solutions_messgae = {
                                "role": "user",
                                "content": "Regenerate. Create at least 2 solutions (Modules1, Modules2)."
                            }
                            messages.append(more_solutions_messgae)
                            logs.append(more_solutions_messgae)
                            # reset costs
                            if not (isinstance(function_response, str) and function_response.startswith("Error:")):
                                tool_count[function_type] -= 1
                                tool_cost[function_type] -= function_cost
                                per_question_cost -= function_cost
                                # print("per_question_cost", per_question_cost) ###
                            continue
                        cost_analysis_ind = content.find("Cost Analysis")
                        modules = content[:cost_analysis_ind].split("Modules")
                        cleaned_modules = []
                        for x in modules:
                            if x!="":
                                clean_x = re.sub(r"^\d:", "", x).strip()
                                cleaned_modules.append(clean_x)
                        if len(cleaned_modules)>len(set(cleaned_modules)):
                            no_duplicate_message = {
                                "role": "user",
                                "content": "Regenerate. Do not create duplicate solutions."
                            }
                            messages.append(no_duplicate_message)
                            logs.append(no_duplicate_message)
                            # reset costs
                            if not (isinstance(function_response, str) and function_response.startswith("Error:")):
                                tool_count[function_type] -= 1
                                tool_cost[function_type] -= function_cost
                                per_question_cost -= function_cost
                                # print("per_question_cost", per_question_cost) ###
                else:
                    response_without_tools = {
                        "role": choice.message.role,
                        "content": content
                    }
                    messages.append(response_without_tools) 
                    logs.append(response_without_tools)
                    iterations += 1
                    if iterations==1:
                        follow_up_message = {
                            "role": "user",
                            "content": "Please finish your thought."
                        }
                        messages.append(follow_up_message)
                        logs.append(follow_up_message)
                    else:
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
        if not is_json_serializable(llm_answer):
            llm_answer = None
        
        logs.append({"Question Type": question_type})
        logs.append({"Cost": per_question_cost})
        logs.append({"LLM Answer": llm_answer})
        logs.append({"Ground-Truth Answer": gt_answer})
        tool_count = dict(sorted(tool_count.items()))
        logs.append({"Tool Count": tool_count})
        tool_cost = dict(sorted(tool_cost.items()))
        logs.append({"Tool Cost": tool_cost})
        cache.append({"qid": pid, "question_type": example["question_type"], "question": example["question"], "Cost": per_question_cost, "Tool Count": tool_count, "Tool Cost": tool_cost, "LLM Answer": llm_answer, "Ground-Truth Answer": gt_answer})
        logs_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.policy_engine, args.hardness, args.prompt, args.formula) # <YOUR_OWN_PATH>
        if not os.path.exists('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.policy_engine, args.hardness, args.prompt, args.formula)): # <YOUR_OWN_PATH>
            os.makedirs('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.policy_engine, args.hardness, args.prompt, args.formula)) # <YOUR_OWN_PATH>
        with open(os.path.join(logs_dir, f"{pid}.txt"), 'w') as f:
            for item in logs:
                f.write(f"{item}\n")
    
    with jsonlines.open(cache_file, mode='w') as writer:
        for row in cache:
            writer.write(row)