import os
import json
import numpy as np
import matplotlib.pyplot as plt

result_dir = '/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/results/hupd' # <YOUR_OWN_PATH>
fig_name = 'easy_results.png' # Change
with open(os.path.join(result_dir, 'chameleon_chatgpt_test.json'), 'r') as f:
    data = json.load(f)
    
acc_dict = data["acc"]
agg_acc = data["agg_acc"]
cost_dict = data["cost"]
agg_cost = data["agg_cost"]
total_count = data["count"]

accuracy = list(acc_dict.values())
cost = list(cost_dict.values())

x = np.arange(len(acc_dict))
width = 0.35

fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='tab:blue')
bar2 = ax.bar(x + width/2, cost, width, label='Cost', color='tab:red')

ax.set_xlabel('Question Type')
ax.set_title('Accuracy and Interpretability Cost of {} Questions'.format(str(total_count)))
ax.set_xticks(x)
ax.legend()

for bar in bar1:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # x coordinate
        height,                             # y coordinate
        f'{height:.2f}%',                   # text
        ha='center',                        # horizontal alignment
        va='bottom'                         # vertical alignment
    )
    
for bar in bar2:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  
        height,                             
        f'{height}',                   
        ha='center',                        
        va='bottom'                         
    )

text = 'Aggregate Accuracy: {:.2f}% Aggregate Cost: {:.2f}'.format(agg_acc, agg_cost)
plt.figtext(0.5, 0.7, text, ha="center", fontsize=9)
plt.savefig(os.path.join(result_dir, "figures", fig_name))