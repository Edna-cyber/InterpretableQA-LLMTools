import os
import math
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardness", type=str, default="easy")
    parser.add_argument("--prompt1", type=str, default="noexample")
    parser.add_argument("--prompt2", type=str, default="interpnoexample")
    parser.add_argument("--formula", type=str, default="")
    parser.add_argument('--policy_engine', type=str, default="gpt-3.5-turbo", help='engine for module prediction')
    parser.add_argument('--policy_engine2', type=str, default="")

    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args

if __name__ == "__main__":
    args = parse_args()
    results_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/results'
    json_dir = results_dir #os.path.join(results_dir, 'current')
    output_image_dir = os.path.join(results_dir, 'new_images') #'images' #new_images
    
    with open(os.path.join(json_dir, f"{args.policy_engine}-{args.hardness}-{args.prompt1}-{args.formula}-test.json"), 'r') as f:
        data1 = json.load(f)
    with open(os.path.join(json_dir, f"{args.policy_engine2}-{args.hardness}-{args.prompt1}-{args.formula}-test.json"), 'r') as f:
        data2 = json.load(f)
    # with open(os.path.join(json_dir, f"{args.policy_engine}-{args.hardness}-{args.prompt2}-{args.formula}-test.json"), 'r') as f:
    #     data2 = json.load(f)
        
    def get_bar_plot(data):
        cost_dict = data["cost"]
        performance_dict = data["overall_performance"]
        categories, costvalues, costerrors, performancevalues = [], [], [], []
        for key in list(cost_dict.keys()):
            if key=="9" and args.hardness=="medium":
                continue
            if key=="10" and args.hardness=="medium":
                categories.append("9")
            else:
                categories.append(key)
            costvalues.append(cost_dict[key]["valid_mean"] if "valid_mean" in cost_dict[key] else 0)
            costerrors.append(math.sqrt(cost_dict[key]["valid_variance"]) if "valid_variance" in cost_dict[key] else 0)
            performancevalues.append(performance_dict[key])
        return categories, costvalues, costerrors, performancevalues
    
    categories, costvalues1, costerrors1, performancevalues1 = get_bar_plot(data1)
    categories, costvalues2, costerrors2, performancevalues2 = get_bar_plot(data2)
    if args.hardness=="hard":
        with open(os.path.join(json_dir, f"{args.policy_engine}-medium-{args.prompt1}-{args.formula}-test.json"), 'r') as f:
            medium_data1 = json.load(f)
        # with open(os.path.join(json_dir, f"{args.policy_engine}-medium-{args.prompt2}-{args.formula}-test.json"), 'r') as f:
        #     medium_data2 = json.load(f)
        with open(os.path.join(json_dir, f"{args.policy_engine2}-medium-{args.prompt1}-{args.formula}-test.json"), 'r') as f:
            medium_data2 = json.load(f)
        added_categories, added_costvalues1, added_costerrors1, added_performancevalues1 = get_bar_plot(medium_data1) 
        added_categories, added_costvalues2, added_costerrors2, added_performancevalues2 = get_bar_plot(medium_data2)
        categories.insert(0,"10")
        costvalues1.insert(0,added_costvalues1[2])
        costvalues2.insert(0,added_costvalues2[2])
        costerrors1.insert(0,added_costerrors1[2])
        costerrors2.insert(0,added_costerrors2[2])
        performancevalues1.insert(0,added_performancevalues1[2])
        performancevalues2.insert(0,added_performancevalues2[2])
    
    n = len(categories)

    bar_width = 0.35  
    x = np.arange(n)
    
    # Cost
    fig, ax = plt.subplots()

    bar1 = ax.bar(x - bar_width/2, costvalues1, bar_width, label='gpt3.5', yerr=costerrors1, capsize=5) #cleanexamples, clean
    bar2 = ax.bar(x + bar_width/2, costvalues2, bar_width, label='gpt4', yerr=costerrors2, capsize=5) #interpformula, interp

    ax.set_xlabel('Question Id')
    ax.set_ylabel('Cost')
    # ax.set_title(f'{args.prompt1} prompt: Comparison of costs for {args.hardness} questions \n with {args.policy_engine} {args.formula}', fontsize=8)
    # ax.set_title(f'Different interpretability prompt: Comparison of costs for {args.hardness} questions \n with {args.policy_engine} {args.formula}', fontsize=8)
    ax.set_title(f'{args.prompt1} Different LLMs: Comparison of costs for {args.hardness} questions \n with {args.policy_engine} {args.formula}', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    output_image_path = os.path.join(output_image_dir, f'{args.prompt1}_Different_LLMs_Comparison_of_costs_for_{args.hardness}_questions_with_{args.policy_engine}_{args.formula}.png')
    # output_image_path = os.path.join(output_image_dir, f'Different_interpretability_prompt_Comparison_of_costs_for_{args.hardness}_questions_with_{args.policy_engine}_{args.formula}.png')
    # output_image_path = os.path.join(output_image_dir, f'{args.prompt1}_prompt_Comparison_of_costs_for_{args.hardness}_questions_with_{args.policy_engine}_{args.formula}.png')
    plt.savefig(output_image_path)
    
    # Performance
    fig, ax = plt.subplots()

    bar1 = ax.bar(x - bar_width/2, performancevalues1, bar_width, label='gpt3.5', capsize=5) #cleanexamples, clean
    bar2 = ax.bar(x + bar_width/2, performancevalues2, bar_width, label='gpt4', capsize=5) #interpformula, interp

    ax.set_xlabel('Question Id')
    ax.set_ylabel('Performance')
    # ax.set_title(f'{args.prompt1} prompt: Comparison of performance for {args.hardness} questions \n with {args.policy_engine} {args.formula}', fontsize=8)
    # ax.set_title(f'Different interpretability prompt: Comparison of performance for {args.hardness} questions \n with {args.policy_engine} {args.formula}', fontsize=8)
    ax.set_title(f'{args.prompt1} Different LLMs: Comparison of performance for {args.hardness} questions \n with {args.policy_engine} {args.formula}', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # output_image_path = os.path.join(output_image_dir, f'{args.prompt1}_prompt_Comparison_of_performance_for_{args.hardness}_questions_with_{args.policy_engine}_{args.formula}.png')
    # output_image_path = os.path.join(output_image_dir, f'Different_interpretability_prompt_Comparison_of_performance_for_{args.hardness}_questions_with_{args.policy_engine}_{args.formula}.png')
    output_image_path = os.path.join(output_image_dir, f'{args.prompt1}_Different_LLMs_Comparison_of_performance_for_{args.hardness}_questions_with_{args.policy_engine}_{args.formula}.png')
    plt.savefig(output_image_path)
        


