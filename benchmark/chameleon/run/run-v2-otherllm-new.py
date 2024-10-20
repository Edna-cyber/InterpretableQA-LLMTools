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

from api.gpt_new import call_gpt
from api.gemini_new import call_gemini_pro
from api.claude import call_claude3

from tools.tools_set import tools_gpt, tools_gemini
from cost_functions import calc_cost1, calc_cost2, calc_cost3, calc_cost4

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

db = table_toolkits()
ACTION_LIST = {
    'Calculator': calculator,
    'DBLoader': db.db_loader,
    'PandasInterpreter': db.pandas_interpreter, 
    'PythonInterpreter': python_interpreter,
    'Forecaster': forecaster,
    'TextualClassifier': db.textual_classifier,
    'LLMInferencer': llm_inferencer,
    'Finish': finish
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument("--hardness", type=str, default="easy")
    parser.add_argument("--prompt", type=str, default="clean")
    parser.add_argument("--formula", type=str, default="")
    parser.add_argument('--policy_engine', type=str, default="gpt-3.5-turbo", help='engine for module prediction')
    parser.add_argument('--policy_temperature', type=float, default=0., help='temperature for module prediction')
    parser.add_argument('--policy_max_tokens', type=int, default=128, help='max tokens for module prediction')
    parser.add_argument('--split', type=str, default="")

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
    
    logs_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.policy_engine, args.hardness, args.prompt, args.formula) # <YOUR_OWN_PATH> 
    if not os.path.exists('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.policy_engine, args.hardness, args.prompt, args.formula)): # <YOUR_OWN_PATH>
        os.makedirs('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.policy_engine, args.hardness, args.prompt, args.formula)) # <YOUR_OWN_PATH>
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
    length = len(pids)
    if args.split=="firsthalf":
        pids = pids[:length//2]
    elif args.split=="secondhalf":
        pids = pids[length//2:]
    
    for pid in tqdm(pids): 
        # with open(os.path.join(logs_dir, f"{pid}.txt"), 'r', encoding='utf-8', errors='ignore') as file:
        #     content = file.read()
        #     if "429" not in content: # request quota error
        #         continue
        time.sleep(1)
        # print("pid", pid)
        db.data = None # force reset
        example = solver.examples[pid] # get one example 
        if args.prompt=="clean":
            user_prompt = "Now, you need to act as a policy model and determine the sequence of tools that can be executed sequentially to solve the question. The solution must follow the structure as in the text CONTENT of the examples and end with the Finish tool. You must use more than just the finish tool. Respond using natural, conversational text. Do not respond with multi_tool_use.parallel JSON or technical formats. Question: "+example["question"]
        elif args.prompt=="cleantext" or args.prompt=="cleanlimited" or args.prompt=="oneexample":
            user_prompt = "Now write your response in the format of the text CONTENT of the examples provided (Solution). Do not respond with multi_tool_use.parallel JSON. The solution must follow this structure as in the examples:  SolutionX: Tool1(parameters), Tool2(parameters), Tool3(parameters) and it must end with the Finish tool. You must use more than just the finish tool. Question: "+example["question"] 
        elif args.prompt=="interp":
            user_prompt = "Now, you need to act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps: 1. Generate multiple solutions with different total costs, aiming to minimize the total cost. The solutions must follow the structure as in the text CONTENT of the examples and end with the Finish tool. You must use more than just the finish tool. Provide at least Solution1 and Solution2, and optionally Solution3 and Solution4. 2. Calculate the interpretability cost for each solution. Then, select the best solution that has the lowest total cost WITHOUT COMPROMISING ACCURACY OF ADDRESSING THE QUESTION. Question: "+example["question"]
        elif args.prompt=="interptext" or args.prompt=="interplimited" or args.prompt=="interponeexample":
            user_prompt = "Now, write your response in the format of the text CONTENT of the examples provided (Solution1, Solution1 Cost, Solution2, Solution2 Cost, and the Best Solution): 1. Generate multiple solutions with different total costs, aiming to minimize the total cost. Each solution must follow this structure as in the examples:   SolutionX: Tool1(parameters), Tool2(parameters), Tool3(parameters) and it must end with the Finish tool. You must use more than just the finish tool. Provide at least Solution1 and Solution2, and optionally Solution3 and Solution4. 2. Calculate the interpretability cost for each solution. Then, select the best solution that has the lowest total cost WITHOUT COMPROMISING ACCURACY OF ADDRESSING THE QUESTION. Question: " + example["question"]
        elif args.prompt=="noexample":
            user_prompt = "Now, you need to act as a policy model and determine the sequence of tools that can be executed sequentially to solve the question. The solution must end with the Finish tool. Respond using natural, conversational text. Do not respond with multi_tool_use.parallel JSON or technical formats. You must use more than just the finish tool. Question: "+example["question"]
        elif args.prompt=="interpnoexample" or args.prompt=="interpexamples":
            user_prompt = "Now, you need to act as a policy model to find the lowest total interpretability cost for solving a question with a given set of tools. Follow these steps: 1. Generate multiple solutions with different total costs, aiming to minimize the total cost. The solutions must end with the Finish tool. You must use more than just the finish tool. Provide at least Solution1 and Solution2, and optionally Solution3 and Solution4. 2. Calculate the interpretability cost for each solution. Then, select the best solution that has the lowest total cost WITHOUT COMPROMISING ACCURACY OF ADDRESSING THE QUESTION. Question: "+example["question"]
        question_type = int(example["question_type"])
        per_question_cost = 0
        tool_count, tool_cost = defaultdict(int), defaultdict(int) 

        if args.prompt=="clean":
            messages = prompt_policy.messages.copy()
        elif args.prompt=="cleantext" or args.prompt=="cleanlimited":
            messages = prompt_policy.messages_text.copy()
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
        elif args.prompt=="interptext" or args.prompt=="interplimited":
            if args.formula=="formula1":
                messages = prompt_policy.messages_formula_1_text.copy()
        elif args.prompt=="noexample":
            messages = prompt_policy.messages_no_example_text.copy()
        elif args.prompt=="oneexample":
            messages = prompt_policy.messages_one_example_text.copy()
        elif args.prompt=="interpnoexample":
            if args.formula=="formula1":
                messages = prompt_policy.messages_formula_1_no_example_text.copy()
        elif args.prompt=="interponeexample":
            if args.formula=="formula1":
                messages = prompt_policy.messages_formula_1_one_example_text.copy()
        elif args.prompt=="interpexamples":
            if args.formula=="formula1":
                messages = prompt_policy.messages_examples_formula_1_text.copy()
                
        if args.policy_engine.startswith("gemini"):
            messages = [{"role": messages[0]["role"], "parts":[{"text":messages[0]["content"]}]}]
        messages.append({"role": "user", "content": user_prompt})
        logs = [{"role": "user", "content": user_prompt}]
        function_type = None
        llm_answer = None
        generation_iterations = 0
        execution_iterations = 0
        
        # solution generation
        while generation_iterations<1:
            if args.policy_engine.startswith("gemini"):
                time.sleep(1)
            if args.hardness!="easy":
                db.prediction = True
            try:
                if args.policy_engine == "gpt-3.5-turbo" or args.policy_engine == "gpt-4-turbo":
                    response_without_tools, _ = call_gpt(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gpt, tool_choice="none")
                elif args.policy_engine == "gemini-1.5-pro-001":
                    response_without_tools, _ = call_gemini_pro(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tool_choice="none")
                elif args.policy_engine == "claude":
                    choice = call_claude3(messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gpt, tool_choice="none")
                    content = choice['text']
                else:
                    raise ValueError("Invalid engine")
                
                messages.append(response_without_tools) 
                logs.append(response_without_tools)
                generation_iterations += 1
            except Exception as e:
                print(f"An error occurred during solution generation: {e}")
                logs.append(json.dumps(str(e)))
                break
            
        messages = messages[1:]
        
        if args.prompt=="clean" or args.prompt=="cleantext" or args.prompt=="cleanlimited" or args.prompt=="oneexample":
            user_prompt = "Execute the tool calls in the given order of 'Solution'. The 'content' of your response MUST BE None, while the 'tool_calls' of your response MUST reflect each tool and its arguments from the 'Solution', one at a time! Ensure that the execution concludes with the use of the Finish tool. If you encounter an error during execution, you can make slight adjustments to the tool's arguments according to the error message." 
        elif args.prompt=="interp" or args.prompt=="interptext" or args.prompt=="interplimited" or args.prompt=="interponeexample":      
            user_prompt = "Execute the tool calls in the given order of Best Solution using the 'tool_calls' parameter. The 'content' of your response MUST BE None. Ensure that the execution concludes with the use of the Finish tool. If you encounter an error during execution, you can make slight adjustments to the tool's arguments according to the error message."
        elif args.prompt=="noexample" or args.prompt=="interpnoexample" or args.prompt=="interpexamples":  
            user_prompt = "Execute the tool calls. The 'content' of your response MUST BE None, while the 'tool_calls' of your response MUST reflect your proposed solution, one tool call at a time! Ensure that the execution concludes with the use of the Finish tool. If you encounter an error during execution, you can make slight adjustments to the tool's arguments according to the error message." 

        execution_message = {
            "role": "user",
            "content": user_prompt
        }
        messages.append(execution_message)
        logs.append(execution_message)
                        
        # solution execution
        started_execution=False
        tool_choice="required"
        while execution_iterations<10:
            if args.policy_engine.startswith("gemini"):
                time.sleep(0.2) # 1
            if args.hardness!="easy":
                db.prediction = True
            try:
                if args.policy_engine == "gpt-3.5-turbo" or args.policy_engine == "gpt-4-turbo":
                    response_tools, tool_call = call_gpt(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gpt, tool_choice=tool_choice)
                elif args.policy_engine == "gemini-1.5-pro-001":
                    response_tools, tool_call = call_gemini_pro(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tool_choice=tool_choice)
                elif args.policy_engine == "claude":
                    choice = call_claude3(messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools_gpt, tool_choice=tool_choice)
                    tool_calls = choice.get('tool_calls', None)
                    content = choice['text']
                else:
                    raise ValueError("Invalid engine")
         
                if tool_call:
                    started_execution=True
                    if args.policy_engine.startswith("gpt"):
                        tool_choice = "auto"
                    
                    if args.policy_engine.startswith("gpt"):
                        function_type = tool_call.function.name
                        function_arguments = json.loads(tool_call.function.arguments)
                    elif args.policy_engine.startswith("gemini"):
                        function_type = tool_call.name
                        function_arguments = tool_call.args
                        
                    if function_type=="LLMInferencer" and "LLMInferencer" in tool_count: # preventing calling LLMInferencer multiple times unnecessarily
                        if args.policy_engine.startswith("gpt"):
                            response_finish = {
                                "role": "user",
                                "content": "Now, call the 'finish' tool using the previous output."
                            }
                        elif args.policy_engine.startswith("gemini"):
                            response_finish = {
                                "role": "user",
                                "parts": [{"text": "Now, call the 'finish' tool using the previous output."}]
                            }
                        messages.append(response_finish) 
                        logs.append(response_finish)
                        continue
                    
                    # print(response_with_tools) 
                    messages.append(response_tools) 
                    logs.append(response_tools)
                    
                    function = ACTION_LIST[function_type]
                    function_response = function(**function_arguments)
                    function_cost = cost_function(function_type, function_arguments)
                    if not (isinstance(function_response, str) and function_response.startswith("Error:")):
                        tool_count[function_type] += 1
                        tool_cost[function_type] += function_cost
                        per_question_cost += function_cost
                        # print("per_question_cost", per_question_cost)
                    
                    if args.policy_engine.startswith("gpt"):
                        tool_call_response = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_type,
                            "content": str(function_response) if function_response is not None else "",
                        } 
                    elif args.policy_engine.startswith("gemini"):
                        tool_call_response = {
                            "role": "user",
                            "parts": [{
                                "functionResponse": {
                                    "name": function_type, 
                                    "response": function_response,
                                }
                            }]
                        } 
                    # print(tool_call_response) 
                    llm_answer = function_response
                    messages.append(tool_call_response)  
                    logs.append(tool_call_response)
                    execution_iterations += 1
                    if function_type=="Finish":
                        break
                else:
                    messages.append(response_tools) 
                    logs.append(response_tools)
                    execution_iterations += 1
                    if started_execution==False:
                        if args.prompt=="clean" or args.prompt=="cleantext" or args.prompt=="cleanlimited" or args.prompt=="oneexample":
                            finish_thought = "USE 'tool_calls' IN YOUR RESPONSE FOR EXECUTION OF 'Solution'! Ensure that the execution concludes with the use of the Finish tool. If you encounter an error during execution, you can make slight adjustments to the tool's arguments according to the error message." 
                        elif args.prompt=="interp" or args.prompt=="interptext" or args.prompt=="interplimited" or args.prompt=="interponeexample":
                            finish_thought = "USE 'tool_calls' IN YOUR RESPONSE FOR EXECUTION OF 'Best Solution'! Ensure that the execution concludes with the use of the Finish tool. If you encounter an error during execution, you can make slight adjustments to the tool's arguments according to the error message."
                        elif args.prompt=="noexample" or args.prompt=="interpnoexample" or args.prompt=="interpexamples":
                            finish_thought = "USE 'tool_calls' IN YOUR RESPONSE FOR EXECUTION! Ensure that the execution concludes with the use of the Finish tool. If you encounter an error during execution, you can make slight adjustments to the tool's arguments according to the error message." 

                        finish_thought_message = {
                            "role": "user",
                            "content": finish_thought
                        }
                        messages.append(finish_thought_message)
                        logs.append(finish_thought_message)
                    else:
                        break
                    
            except Exception as e:
                print(f"An error occurred during solution execution for {pid}: {e}")
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
        
        with open(os.path.join(logs_dir, f"{pid}.txt"), 'w') as f:
            for item in logs: 
                f.write(f"{item}\n")
    
    with jsonlines.open(cache_file, mode='w') as writer:
        for row in cache:
            writer.write(row)