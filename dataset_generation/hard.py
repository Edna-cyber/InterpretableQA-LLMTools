import os
import numpy as np
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

df_hupd_heldout = []
for year in range(2013,2019):
    df_year = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv").format(year))
    df_hupd_heldout.append(df_year)
df_hupd_heldout = pd.concat(df_hupd_heldout, axis=0, ignore_index=True)
df_neurips_heldout_heldout = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv")).iloc[3001:]

question_id = 1
question_type_count = {11:100, 12:100, 13:100}
question_types = [11,12,13]

with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/hard.jsonl', mode='w') as writer:
    while question_id<=10: # 300
        question_type = random.choice(question_types)
        if question_type==11:
            # Do the following two patents belong to the same cpc category: {title1}, {title2}? Return 'Yes' or 'No'.
            same = random.choice(["Yes", "No"])
            df_hupd_heldout["cpc_category"] = df_hupd_heldout["main_cpc_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
            not_na_indices = df_hupd_heldout.index[df_hupd_heldout["cpc_category"] is not None and df_hupd_heldout["cpc_category"]!=np.nan].tolist()
            random_index = random.choice(not_na_indices)
            first_category = df_hupd_heldout.at[random_index,"cpc_category"]
            title1 = df_hupd_heldout.at[random_index,"title"]
            if same=="Yes":
                new_indices = df_hupd_heldout.index[df_hupd_heldout["cpc_category"]==first_category].tolist()
            else:
                new_indices = df_hupd_heldout.index[df_hupd_heldout["cpc_category"] is not None and df_hupd_heldout["cpc_category"]!=np.nan and df_hupd_heldout["cpc_category"]!=first_category].tolist()
            title2 = df_hupd_heldout.at[random.choice(new_indices), "title"]
            question = "Do the following two patents belong to the same cpc category: {}, {}? Return 'Yes' or 'No'.".format(title1,title2)
            answer = same
            # use None to signify not adding to the questions / answers
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[11] -= 1
                if question_type_count[11]==0:
                    question_types.remove(11)
                question_id += 1
        elif question_type==12:
            # Determine if this abstract-title pair is from the same submission: Abstract: {abstract}. Title: {title}
            n = len(df_neurips_heldout)
            abstract_id = random.randint(0,n)
            abstract = df_neurips_heldout.at[abstract_id,"Abstract"]
            yes_or_no = random.choice(["Yes", "No"])
            if yes_or_no=="Yes":
                title = df_neurips_heldout.at[abstract_id,"Title"]
            else:
                title_id = random.randint(0,n)
                while title_id==abstract_id:
                    title_id = random.randint(0,n)
                title = df_neurips_heldout.at[title_id,"Title"]
            question = "Determine if this abstract-title pair is from the same submission: Abstract: {}. Title: {}. Return 'Yes' or 'No'.".format(abstract,title)
            answer = yes_or_no
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[12] -= 1
                if question_type_count[12]==0:
                    question_types.remove(12)
                question_id += 1
        else:
            # Which topic amongst {topic1}, {topic2}, {topic3} is the best fit for title {title}?
            n = len(df_neurips_heldout)
            title_id = random.randint(0,n)
            title = df_neurips_heldout.at[title_id, "Title"]
            true_topics = df_neurips_heldout.at[title_id, "Topic"].split("/")
            false_topics = set(["Deep Learning", "Reinforcement Learning", "Health", "Applications", "Theory", "Data-centric AI", "Probabilistic Methods", 
                                "Social Aspects", "Optimization", "Trustworthy Machine Learning", "Accountability, Transparency and Interpretability", "Robustness"])-set(true_topics)
            three_topics = random.sample(false_topics, 2)+[true_topics[0]]
            random.shuffle(three_topics)
            topic1, topic2, topic3 = three_topics
            question = "Which topic amongst {}, {}, {} is the best fit for title {}?".format(topic1, topic2, topic3, title)
            answer = true_topics[0]
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[13] -= 1
                if question_type_count[13]==0:
                    question_types.remove(13)
                question_id += 1

