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

# HUPD Template 5: First, group the patent applications by month from {start_year} to 2012. For each month, calculate the percentage of applications that were accepted. Then, estimate the percentage of patents filed in the first {n} months of 2013 that will be accepted based on the monthly acceptance rates from the previous years. Return a list of acceptance percentages for each of the first {n} months of 2013, with each percentage expressed as a value between 0 and 100.
def forecast(n):
    df_filtered = df_hupd[df_hupd["filing_date"].dt.year==2013].reset_index(drop=True)
    df_filtered["month"] = df_filtered["filing_date"].dt.month
    df_filtered["acceptance"] = df_filtered["decision"].apply(lambda x: 1 if x =="ACCEPTED" else 0)
    first_few_months = df_filtered.groupby("month")['acceptance'].mean().sort_index()*100 ## maybe need a way to fill in when a month of data is missing
    del df_filtered
    first_n = first_few_months.head(n).tolist()
    return first_n


question_id = 1
question_type_count = {7:100, 8:100, 9:100, 10:100}
question_types = [7,8,9,10]
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/medium.jsonl', mode='w') as writer:
    while question_id<=400:
        question_type = random.choice(question_types) 
        if question_type==7:
            # First, group the patent applications by month from {} to 2012. For each month, calculate the percentage of applications that were accepted. Then, estimate the percentage of patents filed in the first {} months of 2013 that will be accepted based on the monthly acceptance rates from the previous years. Return a list of acceptance percentages for each of the first {} months of 2013, with each percentage expressed as a value between 0 and 100.
            start_year = random.randint(2007,2012)
            n = random.randint(2,4)
            question = "First, group the patent applications by month from {} to 2012. For each month, calculate the percentage of applications that were accepted. Then, estimate the percentage of patents filed in the first {} months of 2013 that will be accepted based on the monthly acceptance rates from the previous years. Return a list of acceptance percentages for each of the first {} months of 2013, with each percentage expressed as a value between 0 and 100.".format(start_year,n,n)
            answer = forecast(n)
            # use None to signify not adding to the questions / answers
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[7] -= 1
                if question_type_count[7]==0:
                    question_types.remove(7)
                question_id += 1
        elif question_type==8:
            # For a patent application, which is not present in the database, with an abstract {abstract_content}, predict whether it will get accepted. Return either 'ACCEPTED' or 'not ACCEPTED'.
            year_not_in_the_range = random.randint(2013,2018)
            df_filtered = df_hupd[df_hupd["filing_date"].dt.year==year_not_in_the_range].reset_index(drop=True)
            indices_to_choose = df_filtered.index[(df_filtered['decision']=="ACCEPTED") | (df_filtered['decision']=="REJECTED")].to_list()
            index = random.choice(indices_to_choose)
            abstract_content = df_filtered.at[index,"abstract"]
            decision = df_filtered.at[index,"decision"]
            question = "For a patent application, which is not present in the database, with an abstract {}, predict whether it will get accepted. Return either 'ACCEPTED' or 'not ACCEPTED'.".format(abstract_content)
            answer = decision
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[8] -= 1
                if question_type_count[8]==0:
                    question_types.remove(8)
                question_id += 1
        elif question_type==9:
            # For a NeurIPS paper, which is not present in the database, with title {title_content}, predict whether it belongs to {topic}? Return either '{topic}' or 'not {topic}'.
            topic = random.choice(["Deep Learning", "Reinforcement Learning", "Health", "Applications", "Theory", "Data-centric AI", "Probabilistic Methods", "Social Aspects", "Optimization"]) 
            df_filtered = df_neurips.iloc[3001:]
            indices_to_choose = df_filtered.index[df_filtered["Topic"].notna()].to_list()
            index = random.choice(indices_to_choose)
            title_content = df_filtered.at[index,"Title"]
            if topic in df_filtered.at[index,"Topic"]:
                belong = topic
            else:
                belong = "not "+topic
            question = "For a NeurIPS paper, which is not present in the database, with title {}, predict whether it belongs to {}? Return either '{}' or 'not {}'.".format(title_content, topic, topic, topic)
            answer = belong
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[9] -= 1
                if question_type_count[9]==0:
                    question_types.remove(9)
                question_id += 1
        else:
            # For a NeurIPS paper, which is not present in the database, with abstract {abstract_content}, predict whether it will be accepted as an oral presentation? Return either ‘oral’ or ‘not oral’.
            df_filtered = df_neurips.iloc[3001:]
            oral = random.choice(["oral", "not oral"])
            if oral=="oral":
                indices_to_choose = df_filtered.index[(df_filtered["Oral"].notna()) & (df_filtered["Oral"]=="oral")].to_list()
            else:
                indices_to_choose = df_filtered.index[(df_filtered["Oral"].notna()) & (df_filtered["Oral"]=="not oral")].to_list()
            index = random.choice(indices_to_choose)
            abstract_content = df_filtered.at[index,"Abstract"]
            question = "For a NeurIPS paper, which is not present in the database, with abstract {}, predict whether it will be accepted as an oral presentation? Return either ‘oral’ or ‘not oral’.".format(abstract_content)
            answer = oral
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[10] -= 1
                if question_type_count[10]==0:
                    question_types.remove(10)
                question_id += 1

