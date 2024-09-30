import os
import re
import ast
import sys
import math
import datetime
import statistics
import json
import jsonlines
import argparse
import random
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from collections import defaultdict
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utilities import *
from model import solver
from tools.tabtools import table_toolkits
import datetime

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

db = table_toolkits()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument("--hardness", type=str, default="easy")
    parser.add_argument("--prompt", type=str, default="clean")
    parser.add_argument("--formula", type=str, default="")
    parser.add_argument('--policy_engine', type=str, default="gpt-3.5-turbo", help='engine for module prediction')

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

    # Get the logs file
    logs_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/{}-{}-{}-{}'.format(args.policy_engine, args.hardness, args.prompt, args.formula) # <YOUR_OWN_PATH> ###
    result_root = f"{args.output_root}" 
    result_file = f"{result_root}/{args.policy_engine}-{args.hardness}-{args.prompt}-{args.formula}-test-{current_datetime}.json"
    
    total_count = 0
    count, valid_performance, errors, cost, cost_original = defaultdict(int), {}, defaultdict(int), defaultdict(int), {}
    # tool_count, avg_tool_cost = defaultdict(int), defaultdict(int) 
        
    for filename in os.listdir(logs_dir):
        total_count += 1
        file_path = os.path.join(logs_dir, filename)
        with open(file_path, 'r') as f:
            content = f.read()
            question_type = int(re.search(r"(?<='Question Type': )\d+", content).group(0))
            if question_type not in cost_original:
                cost_original[question_type] = {"valid":[], "invalid":[], "correct":[], "incorrect":[]}
            count[question_type] += 1
            per_question_cost = float(re.search(r"(?<='Cost': )\d+\.?\d*", content).group(0))
            llm_answer = re.search(r"(?<='LLM Answer': )(.*?)}", content).group(1)
            gt_answer = re.search(r"(?<='Ground-Truth Answer': )(.*?)}", content).group(1)
            # question_tool_count = re.search(r"\{'Tool Count':\s*(\{.*?\})\}", content).group(1)
            # question_tool_count = ast.literal_eval(question_tool_count)
            # if total_count<5:
            #     print("question_tool_count", question_tool_count)
            #     print(type(question_tool_count))
            # question_tool_cost = re.search(r"\{'Tool Cost':\s*(\{.*?\})\}", content).group(1)
            # question_tool_cost = ast.literal_eval(question_tool_cost)
            # if total_count<5:
            #     print("question_tool_cost", question_tool_cost)
            #     print(type(question_tool_cost))
            # for key in question_tool_count.keys():
            #     if key in tool_count:
            #         tool_count[key] += int(question_tool_count[key])
            #     else:
            #         tool_count[key] = int(question_tool_count[key])
            # for key in question_tool_cost.keys():
            #     if key in avg_tool_cost:
            #         avg_tool_cost[key] += int(question_tool_cost[key])
            #     else:
            #         avg_tool_cost[key] = int(question_tool_cost[key])
                    
            if "[" in llm_answer and "]" in llm_answer:
                try:
                    llm_answer = ast.literal_eval(llm_answer)
                except:
                    pass
            if "[" in gt_answer and "]" in gt_answer:
                try:
                    gt_answer = ast.literal_eval(gt_answer)
                except:
                    pass
            if question_type not in valid_performance:
                if question_type in [8,9,10,11,12]:
                    valid_performance[question_type] = [[],[]]
                else:
                    valid_performance[question_type] = 0
                
            if llm_answer=="None" or llm_answer=="null" or llm_answer=="":
                print("None filename", filename) ##
                errors[question_type] += 1
                cost_original[question_type]["invalid"].append(per_question_cost)
            elif isinstance(llm_answer, str) and llm_answer.startswith("Error:"):
                print("Error: filename", filename) ##
                errors[question_type] += 1
                cost_original[question_type]["invalid"].append(per_question_cost)
            elif per_question_cost==0:
                print("cost 0 filename", filename) ##
                errors[question_type] += 1
                cost_original[question_type]["invalid"].append(per_question_cost)
                continue
            else:
                # Calculate valid_performance metric
                if question_type in [1,3,6]: # threshold correct / incorrect
                    try:
                        llm_answer = float(llm_answer)
                        gt_answer = float(gt_answer)
                        metric = int(abs(llm_answer-gt_answer)<=0.005*gt_answer)
                        valid_performance[question_type] += metric
                        cost_original[question_type]["valid"].append(per_question_cost)
                        if metric==1:
                            cost_original[question_type]["correct"].append(per_question_cost)
                        else:
                            cost_original[question_type]["incorrect"].append(per_question_cost)
                    except:
                        print("data type filename", filename) ##
                        errors[question_type] += 1
                        cost_original[question_type]["invalid"].append(per_question_cost)
                elif question_type in [2,5]: # set intersection
                    try:
                        valid_performance[question_type] += len(set(gt_answer)&set(llm_answer)) / len(set(gt_answer))
                        cost_original[question_type]["valid"].append(per_question_cost)
                    except:
                        print("data type filename", filename) ##
                        errors[question_type] += 1
                        cost_original[question_type]["invalid"].append(per_question_cost)
                elif question_type in [4]: # within list
                    try:
                        llm_answer = llm_answer.replace("'", "").replace('"', "")
                        metric = int(llm_answer in gt_answer)
                        valid_performance[question_type] += metric
                        cost_original[question_type]["valid"].append(per_question_cost)
                        if metric==1:
                            cost_original[question_type]["correct"].append(per_question_cost)
                        else:
                            cost_original[question_type]["incorrect"].append(per_question_cost)
                    except:
                        print("data type filename", filename) ##
                        errors[question_type] += 1
                        cost_original[question_type]["invalid"].append(per_question_cost)
                elif question_type in [7]: # average R2
                    try: 
                        valid_performance[question_type] += max(0, r2_score(gt_answer,llm_answer))
                        cost_original[question_type]["valid"].append(per_question_cost)
                    except:
                        print("data type filename", filename) ##
                        errors[question_type] += 1
                        cost_original[question_type]["invalid"].append(per_question_cost)
                elif question_type in [8,9,10,11,12]: # F1
                    try:
                        gt_answer = gt_answer.replace("'", "").replace('"', "")
                        llm_answer = llm_answer.replace("'", "").replace('"', "")
                        valid_performance[question_type][0].append(gt_answer)
                        valid_performance[question_type][1].append(llm_answer)
                        cost_original[question_type]["valid"].append(per_question_cost)
                    except:
                        print("data type filename", filename) ##
                        errors[question_type] += 1
                        cost_original[question_type]["invalid"].append(per_question_cost)
                else: # 13 Exact match
                    try:
                        metric = int(llm_answer==gt_answer)
                        valid_performance[question_type] += metric
                        cost_original[question_type]["valid"].append(per_question_cost)
                        if metric==1:
                            cost_original[question_type]["correct"].append(per_question_cost)
                        else:
                            cost_original[question_type]["incorrect"].append(per_question_cost)
                    except:
                        print("data type filename", filename) ##
                        errors[question_type] += 1
                        cost_original[question_type]["invalid"].append(per_question_cost)         

    # aggregate
    for key in valid_performance.keys():
        if count[key]==errors[key]:
            valid_performance[key] = 0
            continue
        if key in [8,9,10,11,12]:
            original_groundtruth = set(valid_performance[key][0])  
            groundtruth_labels = original_groundtruth
            if len(groundtruth_labels)==1:
                if "not" in valid_performance[key][0][0]:
                    groundtruth_labels.add(valid_performance[key][0][0].replace("not ", ""))
                else:
                    groundtruth_labels.add("not "+valid_performance[key][0][0])
            groundtruths = []
            predictions = []
            for i in range(len(valid_performance[key][1])):
                if valid_performance[key][1][i] in groundtruth_labels:
                    predictions.append(valid_performance[key][1][i])
                else:
                    opposite_label = (groundtruth_labels-{valid_performance[key][0][i]}).pop()
                    predictions.append(opposite_label)
                groundtruths.append(valid_performance[key][0][i])
            if len(original_groundtruth)==1:
                valid_performance[key] = f1_score(groundtruths, predictions, pos_label=groundtruths[0])
            else:
                valid_performance[key] = f1_score(groundtruths, predictions, pos_label=groundtruths[0], average='macro')
        else:
            valid_performance[key] = valid_performance[key] / (count[key]-errors[key])
      
    for key in errors.keys():
        errors[key] = errors[key] / count[key]    
    valid_performance = dict(sorted(valid_performance.items())) 
    overall_performance = {}
    for key in valid_performance.keys():
        overall_performance[key] = valid_performance[key]*(1-errors[key])
    errors = dict(sorted(errors.items()))
    cost_original = dict(sorted(cost_original.items()))
    for key in cost_original.keys():
        cost[key] = defaultdict(int)
        if len(cost_original[key]["valid"])!=0:
            cost[key]["valid_mean"] = statistics.mean(cost_original[key]["valid"])
            cost[key]["valid_median"] = statistics.median(cost_original[key]["valid"])
            cost[key]["valid_trimmed_mean"] = (sum(cost_original[key]["valid"])-max(cost_original[key]["valid"])-min(cost_original[key]["valid"]))/(len(cost_original[key]["valid"])-2) if len(cost_original[key]["valid"])>2 else cost[key]["valid_mean"]
            if len(cost_original[key]["valid"])>=2:
                cost[key]["valid_variance"] = statistics.variance(cost_original[key]["valid"])
        if len(cost_original[key]["invalid"])!=0:
            cost[key]["invalid_mean"] = statistics.mean(cost_original[key]["invalid"])
            cost[key]["invalid_median"] = statistics.median(cost_original[key]["invalid"])
            cost[key]["invalid_trimmed_mean"] = (sum(cost_original[key]["invalid"])-max(cost_original[key]["invalid"])-min(cost_original[key]["invalid"]))/(len(cost_original[key]["invalid"])-2) if len(cost_original[key]["invalid"])>2 else cost[key]["invalid_mean"]
            if len(cost_original[key]["invalid"])>=2:
                cost[key]["invalid_variance"] = statistics.variance(cost_original[key]["invalid"])
    cost = dict(sorted(cost.items()))
    count = dict(sorted(count.items()))
    # tool_count = dict(sorted(tool_count.items()))
    # for key in avg_tool_cost.keys():
    #     avg_tool_cost[key] = avg_tool_cost[key] / tool_count[key]
    # avg_tool_cost = dict(sorted(avg_tool_cost.items()))
    
    # save the result
    result = {'overall_performance': overall_performance, 'valid_performance': valid_performance, 'errors': errors, 'cost': cost, 'cost_original': cost_original, 'count': count, 'total_count': total_count, 'args': vars(args)} # 'tool_count': tool_count, 'avg_tool_cost': avg_tool_cost, 
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4, separators=(',', ': '))


    