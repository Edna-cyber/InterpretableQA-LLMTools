import os
import numpy as np
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"
df_hupd = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd.csv"))
df_hupd["patent_number"] = df_hupd["patent_number"].astype("Int64")
df_hupd["filing_date"] = pd.to_datetime(df_hupd["filing_date"])
df_hupd["patent_issue_date"] = pd.to_datetime(df_hupd["patent_issue_date"])
df_hupd["date_published"] = pd.to_datetime(df_hupd["date_published"])
df_hupd["examiner_id"] = df_hupd["patent_number"].astype("Int64")

df_neurips = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
df_neurips["Poster Session"] = df_neurips["Poster Session"].astype("float64")
df_neurips["Authors"] = df_neurips["Authors"].apply(eval)
df_neurips["Authors Num"] = df_neurips["Authors Num"].astype("Int64")

df_hupd_heldout = df_hupd[df_hupd["filing_date"].dt.year.isin(list(range(2013,2019)))].reset_index(drop=True)
df_neurips_heldout = df_neurips.iloc[3001:].reset_index(drop=True)

question_id = 1
question_type_count = {11:100, 12:100, 13:100}
question_types = [12] #[11,12,13]
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/hard.jsonl', mode='w') as writer:
    while question_id<=10: # 300
        question_type = random.choice(question_types)
        if question_type==11:
            # Predict if the following two patents, which are not present in the database, belong to the same cpc category: {title1}, {title2}? Return 'Yes' or 'No'.
            same = random.choice(["Yes", "No"])
            not_na_indices = df_hupd_heldout.index[df_hupd_heldout["cpc_category"] is not None and df_hupd_heldout["cpc_category"]!=np.nan].tolist()
            random_index = random.choice(not_na_indices)
            first_category = df_hupd_heldout.at[random_index,"cpc_category"]
            title1 = df_hupd_heldout.at[random_index,"title"]
            if same=="Yes":
                new_indices = df_hupd_heldout.index[df_hupd_heldout["cpc_category"]==first_category].tolist()
            else:
                new_indices = df_hupd_heldout.index[(df_hupd_heldout["cpc_category"].notna()) & (df_hupd_heldout["cpc_category"] != first_category)].tolist()
            title2 = df_hupd_heldout.at[random.choice(new_indices), "title"]
            question = "Predict if the following two patents, which are not present in the database, belong to the same cpc category: {}, {}? Return 'Yes' or 'No'.".format(title1,title2)
            answer = same
            # use None to signify not adding to the questions / answers
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[11] -= 1
                if question_type_count[11]==0:
                    question_types.remove(11)
                question_id += 1
        elif question_type==12:
            # Predict if this abstract-title pair, which are not present in the database, is from the same submission: Abstract: {abstract}. Title: {title}
            indices = df_neurips_heldout.index.tolist()
            abstract_id = random.choice(indices)
            abstract = df_neurips_heldout.at[abstract_id,"Abstract"]
            yes_or_no = random.choice(["Yes", "No"])
            if yes_or_no=="Yes":
                title = df_neurips_heldout.at[abstract_id,"Title"]
            else:
                title_id = random.choice(indices)
                while title_id==abstract_id:
                    title_id = random.choice(indices)
                title = df_neurips_heldout.at[title_id,"Title"]
            question = "Predict if this abstract-title pair, which are not present in the database, is from the same submission: Abstract: {}. Title: {}. Return 'Yes' or 'No'.".format(abstract,title)
            answer = yes_or_no
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[12] -= 1
                if question_type_count[12]==0:
                    question_types.remove(12)
                question_id += 1
        else:
            # Predict the best fit topic for the title, which are not present in the database: {title}. Options: {topic1}, {topic2}, {topic3}.
            indices = df_neurips_heldout[df_neurips_heldout["Topic"].notna()].index.tolist()
            title_id = random.choice(indices)
            title = df_neurips_heldout.at[title_id, "Title"]
            true_topic = df_neurips_heldout.at[title_id, "Topic"]
            false_topics = set(["Deep Learning", "Social Aspects", "Optimization", "Applications", "Theory", "Probabilistic Methods", "Reinforcement Learning", "Optimization"])-{true_topic}
            three_topics = random.sample(false_topics,2)+[true_topic]
            random.shuffle(three_topics)
            topic1, topic2, topic3 = three_topics
            question = "Predict the best fit topic for the title, which are not present in the database: {}. Options: {}, {}, {}.".format(title, topic1, topic2, topic3)
            answer = true_topic
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[13] -= 1
                if question_type_count[13]==0:
                    question_types.remove(13)
                question_id += 1

