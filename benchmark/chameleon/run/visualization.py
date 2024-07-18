import os
import json
import numpy as np
import matplotlib.pyplot as plt

result_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/results/hupd' # <YOUR_OWN_PATH>
plot1_name = 'easy_accuracy.png' # Change
plot2_name = "easy_cost.png" # Change
with open(os.path.join(result_dir, 'chameleon_chatgpt_test.json'), 'r') as f:
    data = json.load(f)
    
acc_dict = data["acc"]
agg_acc = data["agg_acc"]
cost_dict = data["cost"]
cost_original_dict = data["cost_original"]
agg_cost = data["agg_cost"]
total_count = data["total_count"]

accuracy = list(acc_dict.values())
cost = list(cost_dict.values())
cost_std_errors = []
for key in list(cost_original_dict.keys()):
    if len(cost_original_dict[key])>1:
        cost_std_errors.append(np.std(cost_original_dict[key], ddof=1)/np.sqrt(len(cost_original_dict[key])))
    else:
        cost_std_errors.append(np.nan)

x = np.arange(len(acc_dict))
width = 0.35

fig = plt.figure()
bar1 = plt.bar(x - width/2, accuracy, width, color='tab:blue')
plt.xlabel('Question Type')
plt.title('Accuracy of {} Questions'.format(str(total_count)))
plt.xticks(x)

for bar in bar1:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x coordinate
        height,                             # y coordinate
        f'{height:.2f}%',                   # text
        ha='center',                        # horizontal alignment
        va='bottom'                         # vertical alignment
    )
text = 'Aggregate Accuracy: {:.2f}%'.format(agg_acc)
plt.figtext(0.3, 0.8, text, ha="center", fontsize=9)
plt.savefig(os.path.join(result_dir, "figures", plot1_name))

fig = plt.figure()
bar2 = plt.bar(x + width/2, cost, width, label='Cost', color='tab:red', yerr=cost_std_errors, capsize=5)
plt.xlabel('Question Type')
plt.title('Interpretability Cost of {} Questions'.format(str(total_count)))
plt.xticks(x)
    
for bar in bar2:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  
        height,                             
        f'{height:.2f}',                   
        ha='center',                        
        va='bottom'                         
    )

text = 'Aggregate Cost: {:.2f}'.format(agg_cost)
plt.figtext(0.3, 0.8, text, ha="center", fontsize=9)
plt.savefig(os.path.join(result_dir, "figures", plot2_name))