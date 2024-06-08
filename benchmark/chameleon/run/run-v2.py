import os
import re
import sys
import json
import argparse
import random
from tqdm import tqdm
from demos import prompt_policy
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utilities import *
from model import solver 

from tools.code.python_interpreter import execute as python_interpreter
from tools.math.calculator import calculator, WolframAlphaCalculator
from tools.table.tabtools import table_toolkits
from tools.finish import finish
import jsonlines
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
print(openai.api_key)

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
    cache_file = f"{result_root}/{args.label}_{args.test_split}_cache.json"
    cache_jsonl = f"{result_root}/{args.label}_{args.test_split}_cache.jsonl"
    result_file = f"{result_root}/{args.label}_{args.test_split}.json"
    print(result_file)

    count, correct, wrong = 0, 0, 0
    pids = solver.pids[count:] # only use the remaining problems

    for pid in tqdm(pids):
        solver.cache = {"pid": pid} # clear the cache

        if count < 10:
            print("\n\n===================================\n")
            print(f"# [Pid]: {pid}\n") # problem id

        count += 1  # number of current results
        solver.cache["example"] = solver.examples[pid] # get one example 

        # [1] Predict the modules
        modules = solver.predict_modules()
        modules = modules[1:-1]
        modules_lst = modules.split('", "')
        modules = []
        if len(modules_lst)>1: 
            for i in range(len(modules_lst)):
                if i==0:
                    modules.append(modules_lst[i][1:])
                elif i==len(modules_lst)-1:
                    modules.append(modules_lst[i][:-1])
                else:
                    modules.append(modules_lst[i])
        else:
            modules = [module[1:-1] for module in modules_lst]
        
        # [2] Execute the modules 
        if count < 10:
            print(f"# [Modules]\n{modules}\n")
        
        context = ""
        logs = ""
        i = 0
        while i<len(modules):
            try:
                attempts = 0
                demo_prompt = prompt_policy.prompt.strip() 
                question = solver.cache["example"]["question"]
                if context != "":
                    if output.startswith("Error:"):
                        if attempts==0:
                            i -= 1
                            attempts += 1
                    module = modules[i]
                    test_prompt =  f"Question: {question}\n\n{context}-->{module}\n\nLast action output: {output}\n\nFill ONLY the currect {module} action with arguments and nothing else:\n"
                else:
                    module = modules[i]
                    test_prompt =  f"Question: {question}\n\n{module}\n\nFill ONLY the currect {module} action with arguments and nothing else:\n"
                print("i", i)
                print("module", module)
                full_prompt = demo_prompt + "\n\n" + test_prompt
                messages=[
                    {"role": "user", "content": full_prompt},
                ]
                # print("test_prompt", test_prompt) 
                # execute the module
                action = get_chat_response(messages, openai.api_key, args.policy_engine, args.policy_temperature, args.policy_max_tokens) 
                print("action", action) 
                if action[0]=="[" and action[-1]=="]":
                    current = action[2:action.find(",")-1] # first element in list string, and then remove double quotes
                else:
                    current = action
                # print("current", current)
                left_bracket = current.find("[") 
                right_bracket = current.rfind("]") 
                action_type = current[:left_bracket] 
                argument = current[left_bracket+1:right_bracket] 
                # print("action_type", action_type) 
                # print("argument", argument) 
                if context == "":
                    context = module+"["+argument+"]"
                else:
                    context = context+"-->"+module+"["+argument+"]"
                # print("context", context) 
                argument = argument.replace("'", "").replace('"', '')
                argument_lst = argument.split(";")
                argument_lst = [x.strip() for x in argument_lst]
                # print("argument_lst", argument_lst) 
                output = ACTION_LIST[action_type](*argument_lst)
                # print("output", output) 
                i += 1
                # input()
                logs = logs + "\n"+"="*30+"\n"+context+"\n\n"+output
                if count < 10:
                    print(f"======== [Module]: {module} ========\n")
                    print(f"# [Input]\n{input}\n")
                    print(f"# [Output]\n{output}\n")
            except Exception as e:
                print(f"An error occurred: {e}")
                logs = logs + "\n"+"="*30+"\n"+context+"\n\n"+str(e)
                break
        # remove all the "\n" in the context
        context = re.sub("\n", "", context)
        # print("context", context)
        logs = logs + "\nGround-Truth Answer: "+str(solver.cache["example"]["answer"])
        if not os.path.exists('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version)): #<YOUR_OWN_PATH>
            os.makedirs('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version)) #<YOUR_OWN_PATH>
            logs_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version) #<YOUR_OWN_PATH>
        with open(os.path.join(logs_dir, pid+'.txt'), 'w') as f:
            f.write(logs)

        acc = correct / count * 100
        with open(cache_file, 'w') as f:
            try:
                f.write(json.dumps(solver.cache, indent=2, separators=(',', ': ')) + "\n")
            except Exception as e:
                print(e)
                print(solver.cache)
        
        with open(cache_jsonl, 'w') as f:
            try:
                json.dump(solver.cache, f)
                f.write('\n')
            except Exception as e:
                print(e)
                print(solver.cache)

        # save the result
        result = {'acc': acc, 'correct': correct, 'wrong':wrong, 'count': count, 'args': vars(args)}
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, separators=(',', ': '))
