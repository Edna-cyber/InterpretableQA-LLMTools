import os
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
    df[col] = df[cat].apply(lambda x:x[:3])
    grouped = df.groupby(col).size().reset_index(name="counts")
    df_accepted = df[df["decision"]=="ACCEPTED"]
    del df
    grouped_accepted = df_accepted.groupby(col).size().reset_index(name="accepted_counts")
    del df_accepted
    merged = pd.merge(grouped, grouped_accepted, on=col, how='inner')
    merged["accepted_p"] = merged["accepted_counts"] / merged["counts"]
    sorted_merged = merged.sort_values(by="accepted_p", ascending=False)
    top_n = list(sorted_merged.head(num)[col])
    return top_n

# Template 2: How does the number of patent applications in {year1} compare to {year2}?
def compare_applications(year_1, year_2):
    df1 = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year_1))))
    len_df1 = len(df1)
    del df1
    df2 = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year_2))))
    len_df2 = len(df2)
    return len_df1 / len_df2

# Template 3: Which applications took the longest time to be published after filing in {year}?
def longest_time(year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
    df = df[df["decision"]=="ACCEPTED"]
    df["date_published"] = pd.to_datetime(df["date_published"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["duration"] = df["date_published"]-df["filing_date"]
    sorted_df = df.sort_values(by="duration", ascending=False).reset_index()
    longest_time = sorted_df.at[0,"duration"]
    applications = []
    i = 0
    while sorted_df.at[i,"duration"]==longest_time:
        applications.append(int(float(sorted_df.at[i,"patent_number"])))
        i += 1
    return applications
    
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
while question_id<=30: #100
    question_type = random.randint(0,4) #(0, 8)
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
        # How does the number of patent applications in {year1} compare to {year2}?
        year_1 = random.randint(2015,2018)
        year_2 = random.randint(2015,2018)
        while year_2==year_1:
            year_2 = random.randint(2015,2018)
        question = "How does the number of patent applications in {} compare to {}?".format(year_1, year_2)
        answer = compare_applications(year_1, year_2)
    elif question_type == 3:
        # Which applications took the longest time to be published after filing in {year}?
        year = random.randint(2015,2018)
        question = "Which applications took the longest time to be published after filing in {}?".format(year)
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