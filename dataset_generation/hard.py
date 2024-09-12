import os
import numpy as np
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

# NeurIPS Template 5: Are the abstract '{abstract}' and the title '{title}' from the same submission? 
def match(yes_or_no):
    return yes_or_no

# NeurIPS Template 6: Which topic amongst {topic1}, {topic2}, {topic3} fit {title} the best?
def topic_match(topic):
    return topic

# NeurIPS Template 7: Using the first {num-1} papers for training, determine a threshold for the number of authors among the {num}-3585 NeurIPS papers. Papers with more authors than this threshold should be more often oral presentations compared to papers with fewer authors.
def threshold(num):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv")) 
    df = df.iloc[num:]
    df["num_authors"] = df["Authors"].str.split(" · ").apply(len)
    df["oral_bool"] = df["Oral"].apply(lambda x: 1 if x =="TRUE" else 0)
    number_of_authors = df["num_authors"].unique()
    thresholds = []
    for threshold in range(min(number_of_authors), max(number_of_authors)+1):
        more_than_threshold = df[df["num_authors"]>threshold]["oral_bool"].mean()
        less_than_threshold = df[df["num_authors"]<=threshold]["oral_bool"].mean()
        if more_than_threshold>less_than_threshold:
            thresholds.append(threshold)
    return thresholds

question_id = 1
question_type_count = {11:100, 12:100, 13:100}
question_types = [11,12,13]

df_hupd_2018 = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_2018.csv"))
df_neurips = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/hard.jsonl', mode='w') as writer:
    while question_id<=10: # 300
        question_type = random.choice(question_types)
        if question_type==11:
            # Do the following two patents belong to the same cpc category: {title1}, {title2}? Return 'Yes' or 'No'.
            same = random.choice(["Yes", "No"])
            df["cpc_category"] = df["main_cpc_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
            not_na_indices = df.index[df["cpc_category"] is not None and df["cpc_category"]!=np.nan].tolist()
            random_index = random.choice(not_na_indices)
            first_category = df.at[random_index,"cpc_category"]
            title1 = df.at[random_index,"title"]
            if same=="Yes":
                new_indices = df.index[df["cpc_category"]==first_category].tolist()
            else:
                new_indices = df.index[df["cpc_category"] is not None and df["cpc_category"]!=np.nan and df["cpc_category"]!=first_category].tolist()
            title2 = df.at[random.choice(new_indices), "title"]
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
            abstract_id = random.randint(0,3585)
            abstract = df_neurips.at[abstract_id,"Abstract"]
            yes_or_no = random.choice(["Yes", "No"])
            if yes_or_no=="Yes":
                title = df_neurips.at[abstract_id,"Title"]
            else:
                title_id = random.randint(0,3585)
                while title_id==abstract_id:
                    title_id = random.randint(0,3585)
                title = df_neurips.at[title_id,"Title"]
            question = "Are the abstract '{}' and the title '{}' from the same submission? Return 'Yes' or 'No'.".format(abstract,title)
            answer = match(yes_or_no)
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[12] -= 1
                if question_type_count[12]==0:
                    question_types.remove(12)
                question_id += 1
        else:
            # Which topic amongst {topic1}, {topic2}, {topic3} fit {title} the best?
            title_id = random.randint(0,3585)
            title = df_neurips.at[title_id, "Title"]
            true_topics = df_neurips.at[title_id, "Topic"].split(" · ")
            false_topics = set(["Deep Learning", "Reinforcement Learning", "Health", "Applications", "Theory", "Data-centric AI", "Probabilistic Methods", "Social Aspects", "Optimization"])-set(true_topics)
            three_topics = random.sample(false_topics, 2)+[true_topics[0]]
            random.shuffle(three_topics)
            topic1, topic2, topic3 = three_topics
            question = "Which topic amongst {topic1}, {topic2}, {topic3} fit {title} the best?".format(topic1, topic2, topic3, title)
            answer = topic_match(true_topics[0])
            if answer:
                writer.write({"qid": "hard-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[13] -= 1
                if question_type_count[13]==0:
                    question_types.remove(13)
                question_id += 1

