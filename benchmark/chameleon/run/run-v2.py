import os
import re
import sys
import json
import argparse
import random
from tqdm import tqdm
from demos import prompt_policy
from openai import OpenAI
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
                        "description": "The name of the database to be loaded, e.g. hupd",
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
            "description": "Interpret Pandas code written in Python. Normally, we only use PandasInterpreter when the question requires data manipulation performed on a specific structured dataframe. We can only use PandasInterpreter after loading the dataframe with LoadDB.",
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
            "description": "Interprets Python code. Normally, we only use PythonInterpreter when the question requires complex computations. We don't use PythonInterpreter when the question requires data manipulation performed on a specific structured dataframe.",
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
            "description": "Return the final answer and finish the task.",
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

    count, correct, cost = 0, 0, 0
    pids = solver.pids[count:] # only use the remaining problems

    for pid in tqdm(pids):
        if count < 10:
            print("\n\n===================================\n")
            print(f"# [Pid]: {pid}\n") # problem id

        count += 1  # number of current results
        example = solver.examples[pid] # get one example 
        system_prompt = prompt_policy.prompt.strip() 
        user_prompt = example["question"]
        
        context = ""
        logs = ""
        i = 0
        
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        print("system_prompt", system_prompt) ###
        print("user_prompt", user_prompt) ###
        response = client.chat.completions.create(model=args.policy_engine, messages=messages, temperature=args.policy_temperature, max_tokens=args.policy_max_tokens, tools=tools, tool_choice="auto")
        print("response", response) ###
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        print("tool_calls", tool_calls) ###
        if tool_calls:
            messages.append(response_message)
            print("messages", messages) ###
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = ACTION_LIST[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                print("function_response", function_response)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            print("new messages", messages) ###
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )  # get a new response from the model where it can see the function response
            print("second_response", second_response) ###
            messages.append(second_response)
        print(messages) ###
    
        # while i<len(modules):
        #     try:
        #         attempts = 0
        #         demo_prompt = prompt_policy.prompt.strip() 
        #         question = example["question"]
        #         if context != "":
        #             if output.startswith("Error:"):
        #                 if attempts==0:
        #                     i -= 1
        #                     attempts += 1
        #             module = modules[i]
        #             test_prompt =  f"Question: {question}\n\n{context}-->{module}\n\nLast action output: {output}\n\nFill ONLY the currect {module} action with arguments and nothing else:\n"
        #         else:
        #             module = modules[i]
        #             test_prompt =  f"Question: {question}\n\n{module}\n\nFill ONLY the currect {module} action with arguments and nothing else:\n"
        #         # print("i", i)
        #         # print("module", module)
        #         # need to replace the prompt after predicting the modules
        #         demo_prompt = re.sub(r'Please provide only the sequence of Modules1, Modules2, Thought, and Best Modules like the examples above and nothing else.', 'Please provide only the sequence of Modules like the examples above and nothing else.', demo_prompt) ###
        #         demo_prompt = re.sub(r'Please provide only the sequence of Best Modules like those from the examples above and nothing else.', 'Please provide only the sequence of Modules like the examples above and nothing else.', demo_prompt)
        #         full_prompt = demo_prompt + "\n\n" + test_prompt
        #         messages=[
        #             {"role": "user", "content": full_prompt},
        #         ]
        #         # print("test_prompt", test_prompt) 
        #         # execute the module
        #         action = get_chat_response(messages, openai.api_key, args.policy_engine, args.policy_temperature, args.policy_max_tokens) 
        #         # print("action", action) 
        #         if action[0]=="[" and action[-1]=="]":
        #             current = action[2:action.find(",")-1] # first element in list string, and then remove double quotes
        #         else:
        #             current = action
        #         # print("current", current)
        #         left_bracket = current.find("[") 
        #         right_bracket = current.rfind("]") 
        #         action_type = current[:left_bracket] 
        #         argument = current[left_bracket+1:right_bracket] 
        #         # print("action_type", action_type) 
        #         # print("argument", argument) 
        #         if context == "":
        #             context = module+"["+argument+"]"
        #         else:
        #             context = context+"-->"+module+"["+argument+"]"
        #         # print("context", context) 
        #         argument_lst = argument.split(";")
        #         argument_lst = [x.strip() for x in argument_lst]
        #         # print("argument_lst", argument_lst) 
        #         output = ACTION_LIST[action_type](*argument_lst)
        #         # print("output", output) 
        #         i += 1
        #         # input()
        #         logs = logs + "\n"+"="*30+"\n"+context+"\n\n"+output
        #         if count < 10:
        #             print(f"======== [Module]: {module} ========\n")
        #             print(f"# [Input]\n{input}\n")
        #             print(f"# [Output]\n{output}\n")
        #     except Exception as e:
        #         print(f"An error occurred: {e}")
        #         logs = logs + "\n"+"="*30+"\n"+context+"\n\n"+str(e)
        #         break
        # remove all the "\n" in the context
        context = re.sub("\n", "", context)
        # print("context", context)
        llm_answer = logs.strip().split('\n')[-1]
        # print("llm_answer", llm_answer) 
        gt_answer = str(example["answer"])
        # print("gt_answer", gt_answer) 
        if llm_answer==gt_answer:
            correct += 1
        elif gt_answer[0]=="[" and gt_answer[-1]=="]": # gt_answer is type list
            correct += int(llm_answer==gt_answer[1:-1])
        logs = logs + "\nGround-Truth Answer: "+gt_answer
        if not os.path.exists('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version)): #<YOUR_OWN_PATH>
            os.makedirs('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version)) #<YOUR_OWN_PATH>
            logs_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}-{}'.format(args.gpt, datetime_string, args.dataset, args.hardness, args.version) #<YOUR_OWN_PATH>
        with open(os.path.join(logs_dir, pid+'.txt'), 'w') as f:
            f.write(logs)

    acc = correct / count * 100
    cost = cost / count

    # save the result
    result = {'acc': acc, 'correct': correct, 'count': count, 'cost': cost, 'args': vars(args)}
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, separators=(',', ': '))
