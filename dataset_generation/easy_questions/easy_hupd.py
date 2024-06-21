import os
import copy
import pandas as pd
import json
import random
import tqdm
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/hupd/"

# Template 0: What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
def average_pendency(start_year, end_year):
    total_len = 0
    pedencies_sum = 0
    
    for year in range(start_year, end_year+1):
        df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
        df = df[df["decision"] == "ACCEPTED"]
        total_len += len(df)
        df["filing_date"] = pd.to_datetime(df['filing_date'])
        df["patent_issue_date"] = pd.to_datetime(df['patent_issue_date'])
        # Calculate pendencies directly as days
        pendencies = (df["patent_issue_date"] - df["filing_date"]).dt.days
        pedencies_sum += pendencies.sum()
        del df
    
    return pedencies_sum / total_len

# Template 1: What were the top{#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}?
def top_accepted_category(num, category, year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
    if category=="IPCR categories":
        cat = "main_ipcr_label"
    else:
        cat = "main_cpc_label"
    col = cat.replace("label", "code")
    df[col] = df[cat].apply(lambda x:x[:3] if isinstance(x, str) else x)
    df["acceptance"] = df["decision"].apply(lambda x: 1 if x =="ACCEPTED" else 0)
    top_categories = df.groupby(col)['acceptance'].mean().nlargest(num)
    top_n = top_categories.index.tolist()
    return top_n

# Template 2: How does the number of patent applications filed in {year1} compare proportionally to {year2}?
def compare_applications(year_1, year_2):
    df1 = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year_1))))
    len_df1 = len(df1)
    del df1
    df2 = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year_2))))
    len_df2 = len(df2)
    del df2
    return len_df1 / len_df2

# Template 3: Which application took the longest time to be published after filing in {year}?
def longest_time(year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
    df["date_published"] = pd.to_datetime(df["date_published"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["duration"] = df["date_published"]-df["filing_date"]
    sorted_df = df.sort_values(by="duration", ascending=False).reset_index()    
    return sorted_df.at[0,"patent_number"]
    
# Template 4: How many examiners reviewed patent applications each year between {start_year} and {end_year}?
def common_examiners(start_year, end_year):
    examiners = set()
    for year in range(start_year, end_year+1):
        df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
        if not examiners:
            examiners = set(df["examiner_id"].unique())
        else:
            examiners = examiners & set(df["examiner_id"].unique())
        del df
    return len(examiners)

# # What was the average number of inventors per application from {start_year} to {end_year}?
# def number_inventors(start_year, end_year):
#     for year in range(start_year, end_year+1):
#         df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))

questions = []
question_id = 1
while question_id<=5: #100
    question_type = random.randint(3,3) #(0, 8)
    if question_type == 0:
        # What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
        start_year = random.randint(2015,2018)
        end_year = random.randint(start_year,2018)
        question = "What was the average time between the filing and issuance of patents from {} to {}?".format(start_year, end_year)
        answer = average_pendency(start_year, end_year)
    elif question_type == 1:
        # What were the top{#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}?
        num = random.randint(2,5)
        category = random.choice(["IPCR categories", "CPC categories"]) 
        year = random.randint(2015,2018) 
        question = "What were the top{} {} with the highest percentage of patent acceptance in {}?".format(num, category, year)
        answer = top_accepted_category(num, category, year)
    elif question_type == 2:
        # How does the number of patent applications filed in {year1} compare proportionally to {year2}?
        year_1 = random.randint(2015,2018)
        year_2 = random.randint(2015,2018)
        while year_2==year_1:
            year_2 = random.randint(2015,2018)
        question = "How does the number of patent applications filed in {} compare proportionally to {}?".format(year_1, year_2)
        answer = compare_applications(year_1, year_2)
    elif question_type == 3:
        # Which application took the longest time to be published after filing in {year}?
        year = random.randint(2015,2018)
        question = "Which application took the longest time to be published after filing in {}?".format(year)
        answer = longest_time(year)
    elif question_type == 4:
        # How many examiners reviewed patent applications each year between {start_year} and {end_year}?
        start_year = random.randint(2015,2018)
        end_year = random.randint(start_year,2018)
        question = "How many examiners reviewed patent applications each year between {} and {}?".format(start_year, end_year)
        answer = common_examiners(start_year, end_year)
    # elif question_type == 5:
    #     # What was the average number of inventors per application from {start_year} to {end_year}?
    #     start_year = random.randint(2015,2018)
    #     end_year = random.randint(start_year,2018)
    #     question = "What was the average number of inventors per application from {start_year} to {end_year}?".format(start_year, end_year)
    #     answer = number_inventors(start_year, end_year)
    # elif question_type == 6:
        # Who were the top 5 most prolific inventors based on the number of applications in {year}?
    
    # use None to signify not adding to the questions / answers
    if answer:
        questions.append({"qid": "easy-hupd-{:0>4d}".format(question_id), "question":question, "answer":answer})
        question_id += 1

with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/easy/hupd-easy.jsonl', mode='w') as writer:
    for row in questions:
        writer.write(row)