import os
import math
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardness", type=str, default="easy")
    parser.add_argument("--prompt1", type=str, default="clean")
    parser.add_argument("--prompt2", type=str, default="interp")
    parser.add_argument("--formula", type=str, default="")
    parser.add_argument('--policy_engine', type=str, default="gpt-3.5-turbo", help='engine for module prediction')

    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args

if __name__ == "__main__":
    args = parse_args()
    results_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/results'
    json_dir = os.path.join(results_dir, 'current')
    output_image_dir = os.path.join(results_dir, 'images')
    
    with open(os.path.join(json_dir, f"{args.policy_engine}-{args.hardness}-{args.prompt1}-{args.formula}-test.json"), 'r') as f:
        data1 = json.load(f)
    with open(os.path.join(json_dir, f"{args.policy_engine}-{args.hardness}-{args.prompt2}-{args.formula}-test.json"), 'r') as f:
        data2 = json.load(f)
        
    def get_bar_plot(data):
        cost_original_dict = data["cost_original"]
        cost_dict = data["cost"]
        categories, values, errors = [], [], []
        for key in list(cost_dict.keys()):
            categories.append(key)
            values.append(cost_dict[key]["valid_mean"] if "valid_mean" in cost_dict[key] else 0)
            errors.append(math.sqrt(cost_dict[key]["valid_variance"]) if "valid_variance" in cost_dict[key] else 0)
        return categories, values, errors
    
    categories, values1, errors1 = get_bar_plot(data1)
    categories, values2, errors2 = get_bar_plot(data2)
    
    n = len(categories)

    bar_width = 0.35  
    x = np.arange(n)

    fig, ax = plt.subplots()

    bar1 = ax.bar(x - bar_width/2, values1, bar_width, label='clean', yerr=errors1, capsize=5)
    bar2 = ax.bar(x + bar_width/2, values2, bar_width, label='interp', yerr=errors2, capsize=5)

    ax.set_xlabel('Question Id')
    ax.set_ylabel('Cost')
    ax.set_title(f'Comparison of costs for {args.hardness} questions with {args.policy_engine} {args.formula}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    output_image_path = os.path.join(output_image_dir, f'Comparison_of_costs_for_{args.hardness}_questions_with_{args.policy_engine}_{args.formula}.png')
    plt.savefig(output_image_path)
        
    # for key in list(cost_original_dict.keys()):
    #     for category in ["valid", "invalid", "correct", "incorrect"]:
    #         data_points = cost_original_dict[key][category]
    #         plt.hist(data_points, edgecolor='black')
    #         plt.title(f"Interpretability Cost of {args.policy_engine}_{args.hardness}_{args.prompt}_{args.formula}_Question{key}_{category}")
    #         plt.xlabel('Cost')
    #         plt.ylabel('Frequency')
    #         output_image_path = os.path.join(output_image_dir, f'{args.policy_engine}_{args.hardness}_{args.prompt}_{args.formula}_Question{key}_{category}.png')
    #         plt.savefig(output_image_path)
    #         plt.clf()
    #         print(f'{args.policy_engine}_{args.hardness}_{args.prompt}_{args.formula}_Question{key}_{category}.png saved!')
    

# acc_dict = data["acc"]
# agg_acc = data["agg_acc"]
# cost_dict = data["cost"]
# cost_original_dict = data["cost_original"]
# agg_cost = data["agg_cost"]
# total_count = data["total_count"]

# accuracy = [float(x[:-1]) for x in list(acc_dict.values())]
# cost = list(cost_dict.values())
# cost_std_errors = []
# for key in list(cost_original_dict.keys()):
#     if len(cost_original_dict[key])>1:
#         cost_std_errors.append(np.std(cost_original_dict[key], ddof=1)/np.sqrt(len(cost_original_dict[key])))
#     else:
#         cost_std_errors.append(np.nan)

# x = np.arange(len(acc_dict))
# width = 0.35

# fig = plt.figure()
# bar1 = plt.bar(x - width/2, accuracy, width, color='tab:blue')
# plt.xlabel('Question Type')
# plt.title('Accuracy of {} Questions'.format(str(total_count)))
# plt.xticks(x)

# for bar in bar1:
#     height = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,  # x coordinate
#         height,                             # y coordinate
#         f'{height:.2f}%',                   # text
#         ha='center',                        # horizontal alignment
#         va='bottom'                         # vertical alignment
#     )
# text = 'Aggregate Accuracy: {}'.format(agg_acc)
# plt.figtext(0.3, 0.8, text, ha="center", fontsize=9)
# plt.savefig(os.path.join(result_dir, "figures", plot1_name))

# fig = plt.figure()
# bar2 = plt.bar(x + width/2, cost, width, label='Cost', color='tab:red', yerr=cost_std_errors, capsize=5)
# plt.xlabel('Question Type')
# plt.title('Interpretability Cost of {} Questions'.format(str(total_count)))
# plt.xticks(x)
    
# for bar in bar2:
#     height = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,  
#         height,                             
#         f'{height:.2f}',                   
#         ha='center',                        
#         va='bottom'                         
#     )

# text = 'Aggregate Cost: {}'.format(agg_cost) 
# plt.figtext(0.3, 0.8, text, ha="center", fontsize=9)
# plt.savefig(os.path.join(result_dir, "figures", plot2_name))